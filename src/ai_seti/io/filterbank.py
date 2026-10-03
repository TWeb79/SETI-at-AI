"""Lightweight filterbank I/O.

v0.1 loaded the whole selection through blimpy. Here we parse the SIGPROC header
ourselves and memory-map the data, so a worker only touches the channel window it
was assigned. HDF5 (Breakthrough Listen .h5, bitshuffle-compressed) is read with
h5py + hdf5plugin and chunk-aligned slicing. blimpy is no longer required.
"""
from __future__ import annotations

import contextlib
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

INT_KEYS = {"telescope_id", "machine_id", "data_type", "barycentric", "pulsarcentric",
            "nbits", "nsamples", "nchans", "nifs", "nbeams", "ibeam"}
DOUBLE_KEYS = {"tstart", "tsamp", "fch1", "foff", "refdm", "az_start", "za_start",
               "src_raj", "src_dej"}
STRING_KEYS = {"source_name", "rawdatafile"}
NBITS_DTYPE = {8: np.uint8, 16: np.uint16, 32: np.float32, 64: np.float64}
GBT_COARSE_WIDTH_HZ = 187.5e6 / 64   # 2.9296875 MHz


@dataclass
class FilterbankHeader:
    fch1: float                 # MHz, centre of channel 0
    foff: float                 # MHz per channel (negative for BL data)
    nchans: int
    tsamp: float                # seconds
    nsamples: int
    nifs: int = 1
    nbits: int = 32
    tstart: float = 0.0         # MJD
    source_name: str = "unknown"
    data_offset: int = 0        # bytes (SIGPROC only)
    extra: dict = field(default_factory=dict)

    @property
    def bytes_per_value(self) -> int:
        return self.nbits // 8

    @property
    def dtype(self):
        return NBITS_DTYPE[self.nbits]

    def frequencies(self, start: int = 0, stop: int | None = None) -> np.ndarray:
        stop = self.nchans if stop is None else stop
        return self.fch1 + self.foff * np.arange(start, stop, dtype=np.float64)

    @property
    def f_min(self) -> float:
        return float(min(self.fch1, self.fch1 + self.foff * (self.nchans - 1)))

    @property
    def f_max(self) -> float:
        return float(max(self.fch1, self.fch1 + self.foff * (self.nchans - 1)))

    def fine_per_coarse(self) -> int | None:
        """Guess fine channels per GBT coarse channel (2^20 hi-res, 2^10 mid-res)."""
        if not self.foff:
            return None
        n = GBT_COARSE_WIDTH_HZ / abs(self.foff * 1e6)
        p = round(np.log2(n)) if n > 1 else 0
        if p > 0 and abs(2 ** p - n) / n < 0.01 and self.nchans % 2 ** p == 0:
            return 2 ** p
        return None

    def channel_range(self, f_start: float | None, f_stop: float | None) -> tuple[int, int]:
        """Channel index range [c0, c1) covering a frequency interval in MHz.

        Both bounds are clamped into the band and c1 is never allowed below c0: a request
        that lies entirely outside the band yields an empty range rather than an inverted
        one, which would otherwise reach read_window as a negative-width slice.
        """
        if f_start is None and f_stop is None:
            return 0, self.nchans
        lo = self.f_min if f_start is None else f_start
        hi = self.f_max if f_stop is None else f_stop
        if not self.foff:
            raise ValueError("Cannot map a frequency range: header foff is zero")
        a = (lo - self.fch1) / self.foff
        b = (hi - self.fch1) / self.foff
        c0 = max(0, min(self.nchans, int(np.floor(min(a, b)))))
        c1 = max(0, min(self.nchans, int(np.ceil(max(a, b))) + 1))
        return (c0, c1) if c1 > c0 else (c0, c0)

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in ("fch1", "foff", "nchans", "tsamp", "nsamples",
                                            "nifs", "nbits", "tstart", "source_name")}
        d.update({k: v for k, v in self.extra.items() if isinstance(v, (int, float, str))})
        return d


# --------------------------------------------------------------------------- SIGPROC

def _read_string(buf: bytes, pos: int, allow_empty: bool = False) -> tuple[str, int]:
    (n,) = struct.unpack_from("<i", buf, pos)
    # Keys are never empty, but values can be: BL's .8.0001 products carry rawdatafile="".
    if not (0 if allow_empty else 1) <= n < 256:
        raise ValueError("Corrupt SIGPROC header string")
    pos += 4
    return buf[pos:pos + n].decode("ascii", errors="replace"), pos + n


def parse_sigproc_header(buf: bytes, file_size: int | None = None) -> FilterbankHeader:
    """Parse a SIGPROC header from the first bytes of a .fil file.

    Raises EOFError if the buffer ends before HEADER_END (caller should read more).
    """
    try:
        key, pos = _read_string(buf, 0)
    except struct.error as exc:
        raise EOFError from exc
    if key != "HEADER_START":
        raise ValueError("Not a SIGPROC filterbank file (missing HEADER_START)")
    values: dict = {}
    while True:
        try:
            key, pos = _read_string(buf, pos)
        except struct.error as exc:
            raise EOFError from exc
        if key == "HEADER_END":
            break
        if pos + 8 > len(buf):
            raise EOFError
        if key in INT_KEYS:
            (values[key],) = struct.unpack_from("<i", buf, pos)
            pos += 4
        elif key in DOUBLE_KEYS:
            (values[key],) = struct.unpack_from("<d", buf, pos)
            pos += 8
        elif key in STRING_KEYS:
            values[key], pos = _read_string(buf, pos, allow_empty=True)
        else:
            raise ValueError(f"Unknown SIGPROC header key: {key!r}")
    nchans, nifs, nbits = values["nchans"], values.get("nifs", 1), values.get("nbits", 32)
    nsamples = values.get("nsamples", 0)
    if not nsamples and file_size:
        nsamples = (file_size - pos) // (nchans * nifs * nbits // 8)
    known = {"fch1", "foff", "nchans", "tsamp", "nsamples", "nifs", "nbits", "tstart",
             "source_name"}
    return FilterbankHeader(
        fch1=values["fch1"], foff=values["foff"], nchans=nchans, tsamp=values["tsamp"],
        nsamples=int(nsamples), nifs=nifs, nbits=nbits, tstart=values.get("tstart", 0.0),
        source_name=values.get("source_name", "unknown"), data_offset=pos,
        extra={k: v for k, v in values.items() if k not in known},
    )


def read_header(path: Path) -> FilterbankHeader:
    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return _h5_header(path)
    size = path.stat().st_size
    with path.open("rb") as fh:
        buf = fh.read(4096)
        while True:
            try:
                return parse_sigproc_header(buf, size)
            except EOFError as exc:
                more = fh.read(4096)
                if not more:
                    raise ValueError("Truncated SIGPROC header") from exc
                buf += more


def read_window(path: Path, chan_start: int = 0, chan_stop: int | None = None,
                header: FilterbankHeader | None = None, if_index: int = 0) -> np.ndarray:
    """Return float32 array [time, channel] for one channel window, touching only that window."""
    path = Path(path)
    header = header or read_header(path)
    chan_stop = header.nchans if chan_stop is None else chan_stop
    if path.suffix.lower() in (".h5", ".hdf5"):
        import h5py
        with contextlib.suppress(ImportError):
            import hdf5plugin  # noqa: F401  (registers the bitshuffle filter)
        with h5py.File(path, "r") as h5:
            ds = h5["data"]
            if ds.ndim == 3:
                return np.asarray(ds[:, if_index, chan_start:chan_stop], dtype=np.float32)
            return np.asarray(ds[:, chan_start:chan_stop], dtype=np.float32)
    mm = np.memmap(path, dtype=header.dtype, mode="r", offset=header.data_offset,
                   shape=(header.nsamples, header.nifs, header.nchans))
    return np.ascontiguousarray(mm[:, if_index, chan_start:chan_stop], dtype=np.float32)


def _h5_header(path: Path) -> FilterbankHeader:
    import h5py
    with contextlib.suppress(ImportError):
        import hdf5plugin  # noqa: F401
    with h5py.File(path, "r") as h5:
        ds = h5["data"]
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in ds.attrs.items()}
        shape = ds.shape
    nifs = shape[1] if len(shape) == 3 else 1
    to_py = lambda v: v.item() if hasattr(v, "item") else v  # noqa: E731
    attrs = {k: to_py(v) for k, v in attrs.items()}
    return FilterbankHeader(
        fch1=float(attrs["fch1"]), foff=float(attrs["foff"]), nchans=int(shape[-1]),
        tsamp=float(attrs.get("tsamp", 1.0)), nsamples=int(shape[0]), nifs=int(nifs),
        nbits=int(attrs.get("nbits", 32)), tstart=float(attrs.get("tstart", 0.0)),
        source_name=str(attrs.get("source_name", "unknown")),
        extra={k: v for k, v in attrs.items()
               if k not in ("fch1", "foff", "nchans", "tsamp", "nbits", "tstart",
                            "source_name", "DIMENSION_LABELS", "CLASS", "VERSION")
               and isinstance(v, (int, float, str))},
    )


def write_sigproc(path: Path, data: np.ndarray, header: FilterbankHeader) -> None:
    """Write a float32 [time, channel] array as a SIGPROC .fil (used by tests and demos)."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<i", len(b)) + b

    out = s("HEADER_START")
    out += s("source_name") + s(header.source_name)
    for key in ("telescope_id", "machine_id", "data_type"):
        out += s(key) + struct.pack("<i", int(header.extra.get(key, 6 if key == "telescope_id"
                                                                 else 1 if key == "data_type"
                                                                 else 10)))
    for key, val in (("fch1", header.fch1), ("foff", header.foff), ("tstart", header.tstart),
                     ("tsamp", header.tsamp), ("src_raj", header.extra.get("src_raj", 0.0)),
                     ("src_dej", header.extra.get("src_dej", 0.0))):
        out += s(key) + struct.pack("<d", float(val))
    for key, val in (("nchans", data.shape[1]), ("nifs", 1), ("nbits", 32)):
        out += s(key) + struct.pack("<i", int(val))
    out += s("HEADER_END")
    with Path(path).open("wb") as fh:
        fh.write(out)
        fh.write(np.ascontiguousarray(data, dtype="<f4").tobytes())
