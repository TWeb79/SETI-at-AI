"""Where the data comes from, and how it is cut into SETI@home-style work units.

SETI@home itself stopped sending work on 31 March 2020 and is hibernating; its
BOINC scheduler hands out no tasks. `SetiAtHomeSource.probe()` checks the live
server status so the client can resume automatically if the project ever wakes up.
In the meantime the auto-loader pulls from the Breakthrough Listen Open Data
archive, run by the same Berkeley SETI Research Center group.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import SearchConfig
from .io import remote
from .io.filterbank import FilterbankHeader, read_header, write_sigproc

BL_API = "http://seti.berkeley.edu/opendata/api"
# The archive ignores offset/page/skip but returns a stable order, so "paging" means asking
# for a longer list until it contains enough rows we have not seen (backlog B11).
MAX_QUERY_ROWS = 2000
SAH_STATUS = "https://setiathome.berkeley.edu/server_status.php?xml=1"


@dataclass
class WorkUnit:
    unit_id: str
    kind: str                    # "local" | "remote"
    location: str                # path or URL
    header: FilterbankHeader
    chan_start: int              # padded window actually loaded
    chan_stop: int
    core_start: int              # hits are kept only if they start in the core
    core_stop: int
    meta: dict = field(default_factory=dict)

    @property
    def n_channels(self) -> int:
        return self.chan_stop - self.chan_start


def drift_padding(header: FilterbankHeader, cfg: SearchConfig) -> int:
    """Channels of guard band a work unit needs on each side of its core.

    Must match the drift ceiling `drift_search` will actually search, otherwise a signal
    starting in the last channels of a core drifts into unpadded zeros and is lost.
    """
    return drift_ceil_ch_per_step(header, cfg) * max(header.nsamples, 1) + 8


def drift_ceil_ch_per_step(header: FilterbankHeader, cfg: SearchConfig) -> int:
    """Max |drift| in channels per time sample, from the header and the config caps."""
    foff_hz = abs(header.foff) * 1e6
    k = int(np.ceil(cfg.max_drift_rate_hz_s * header.tsamp / foff_hz)) if foff_hz else 1
    # The two caps are independent limits; the binding one is the smaller. (Taking
    # max(cfg_cap, k) here would make the configured cap unreachable.)
    return max(1, min(k, max(1, cfg.max_drift_ch_per_step)))


def drift_resolvable(header: FilterbankHeader, cfg: SearchConfig) -> bool:
    """True if the configured drift rate moves a tone at least one channel over the scan.

    On BL mid-res products (2.86 kHz x 1.07 s x 272) the 4 Hz/s limit moves a tone ~1,170 Hz
    over the whole observation, under one channel: drift there is unmeasurable, not zero (B19).
    """
    foff_hz = abs(header.foff) * 1e6
    return not foff_hz or cfg.max_drift_rate_hz_s * header.nsamples * header.tsamp >= foff_hz


def unit_too_large(header: FilterbankHeader, cfg: SearchConfig) -> str | None:
    """Why one work unit of this file would not fit in memory, or None if it does.

    BL's high-time-resolution .8.0001 products are 8,192 channels x ~840,000 samples: a single
    unit would be ~27 GB of float32 fetched with one Range request per sample row. They are
    pulsar/transient data, not drift-search data, so refuse them up front.
    """
    width = min(int(cfg.channels_per_unit), header.nchans) + 2 * drift_padding(header, cfg)
    mb = min(width, header.nchans) * header.nsamples * 4 / 2**20
    if mb > cfg.max_download_mb:
        return (f"one work unit would be {mb:,.0f} MB ({header.nsamples:,} time samples); "
                f"limit is max_download_mb = {cfg.max_download_mb:,}. High-time-resolution "
                "products are not searched for drifting tones.")
    return None


def split(location: str, header: FilterbankHeader, cfg: SearchConfig, kind: str = "local",
          chan_range: tuple[int, int] | None = None, meta: dict | None = None) -> list[WorkUnit]:
    """Cut an observation into overlapping channel windows (the SETI@home 'splitter')."""
    c0, c1 = chan_range or (0, header.nchans)
    width = max(1024, int(cfg.channels_per_unit))
    pad = drift_padding(header, cfg)
    stem = Path(urllib.parse.urlparse(location).path).stem if kind == "remote" else Path(location).stem
    units = []
    for core0 in range(c0, c1, width):
        core1 = min(core0 + width, c1)
        units.append(WorkUnit(
            unit_id=f"{stem}:{core0}-{core1}", kind=kind, location=str(location), header=header,
            # Pad into neighbouring channels even outside chan_range: a --max-units chunk
            # edge is not a data edge, and a tone drifting across it needs its guard band.
            chan_start=max(0, core0 - pad), chan_stop=min(header.nchans, core1 + pad),
            core_start=core0, core_stop=core1, meta=dict(meta or {}),
        ))
    return units


def load_unit(wu: WorkUnit) -> np.ndarray:
    from .io.filterbank import read_window
    if wu.kind == "remote":
        return remote.remote_window(wu.location, wu.header, wu.chan_start, wu.chan_stop)
    return read_window(Path(wu.location), wu.chan_start, wu.chan_stop, wu.header)


# --------------------------------------------------------------------------- sources

class LocalSource:
    """Every .fil/.h5 in a directory (optionally watched for new arrivals)."""
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    def observations(self, done: set[str], partial: frozenset[str] | set[str] = frozenset()):
        for p in sorted(self.directory.glob("**/*")):
            if p.suffix.lower() in (".fil", ".h5", ".hdf5") and str(p) not in done:
                hdr = read_header(p)
                yield str(p), "local", hdr, {"target": hdr.source_name, "source": "local"}


class SyntheticSource:
    """Generates a GBT-like synthetic coarse channel, writes it as a real .fil file."""
    def __init__(self, out_dir: Path, n: int = 1, n_chan: int = 1 << 20, seed: int = 42):
        self.out_dir, self.n, self.n_chan, self.seed = Path(out_dir), n, n_chan, seed
        self.injections: dict[str, list] = {}

    def observations(self, done: set[str], partial: frozenset[str] | set[str] = frozenset()):
        from .ai.simulate import synthetic_observation
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(self.n):
            path = self.out_dir / f"synthetic_{self.seed + i}.fil"
            if str(path) in done:
                continue
            data, hdr, injections = synthetic_observation(n_chan=self.n_chan, seed=self.seed + i)
            header = FilterbankHeader(**hdr)
            write_sigproc(path, data, header)
            self.injections[str(path)] = [j.to_dict() for j in injections]
            (path.with_suffix(".injections.json")).write_text(
                json.dumps(self.injections[str(path)], indent=1))
            yield str(path), "local", read_header(path), {
                "target": header.source_name, "source": "synthetic", "telescope": "simulated GBT"}


class BreakthroughListenSource:
    """Auto-loader for the Breakthrough Listen Open Data archive.

    SIGPROC (.fil) files are streamed per work unit with HTTP Range requests; HDF5
    files are downloaded (with a size guard) into a local cache first.
    """
    def __init__(self, target: str = "", telescope: str = "GBT", file_types: str = "filterbank",
                 limit: int = 20, cache_dir: Path = Path("data/raw"), max_download_mb: int = 2048,
                 extra: dict | None = None):
        self.params = {"target": target, "telescopes": telescope, "file-types": file_types,
                       "limit": str(limit), **(extra or {})}
        self.cache_dir = Path(cache_dir)
        self.max_download_mb = max_download_mb

    def query(self, limit: int | None = None, timeout: float = 60.0,
              retries: int = 3) -> list[dict]:
        params = {**self.params, **({"limit": str(limit)} if limit else {})}
        url = f"{BL_API}/query-files?" + urllib.parse.urlencode(
            {k: v for k, v in params.items() if v not in (None, "")})
        body, _, _ = remote._get(url, timeout=timeout, retries=retries)
        payload = json.loads(body)
        rows = payload.get("data", payload) if isinstance(payload, dict) else payload
        return [r for r in rows if isinstance(r, dict) and r.get("url")]

    def new_rows(self, done: set[str], partial: frozenset[str] | set[str] = frozenset()) -> list[dict]:
        """Up to `limit` rows not in `done`, widening the query past rows already seen.

        Half the slots go to files not started yet: one high-res file needs hundreds of
        `--max-units` runs, and resumed files would otherwise take every slot (backlog B29).
        """
        want = int(self.params.get("limit") or 20)
        half = (want + 1) // 2
        n = want
        while True:
            rows = self.query(n)
            new = [r for r in rows if r["url"] not in done]
            fresh = [r for r in new if r["url"] not in partial]
            if (len(new) >= want and len(fresh) >= half) or len(rows) < n or n >= MAX_QUERY_ROWS:
                resumed = [r for r in new if r["url"] in partial]
                picked = fresh[:half]
                picked += resumed[:want - len(picked)]
                return picked + fresh[half:half + want - len(picked)]
            n = min(n * 4, MAX_QUERY_ROWS)

    def observations(self, done: set[str], partial: frozenset[str] | set[str] = frozenset()):
        for row in self.new_rows(done, partial):
            url = row["url"]
            meta = {"target": row.get("target"), "telescope": row.get("telescope"),
                    "ra": row.get("ra"), "decl": row.get("decl"), "mjd": row.get("mjd"),
                    "center_freq": row.get("center_freq"), "md5sum": row.get("md5sum"),
                    "cadence_url": row.get("cadence_url"), "source": "breakthrough_listen",
                    "url": url}
            path = urllib.parse.urlparse(url).path.lower()
            try:
                if path.endswith(".fil"):
                    try:
                        hdr = remote.remote_header(url)
                        yield url, "remote", hdr, meta
                        continue
                    except remote.RangeNotSupported:
                        pass
                local = self.cache_dir / Path(path).name
                if not local.exists():
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    remote.download(url, local, self.max_download_mb)
                yield str(local), "local", read_header(local), meta
            except Exception as exc:  # skip unreadable / oversized files, keep crunching
                yield None, "error", None, {**meta, "error": str(exc)}


class SetiAtHomeSource:
    """Checks whether the SETI@home BOINC project is distributing work again."""
    @staticmethod
    def probe(timeout: float = 15.0) -> dict:
        try:
            req = urllib.request.Request(SAH_STATUS, headers={"User-Agent": remote.USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", "replace")
        except Exception as exc:
            # URLError wraps the useful part ("The handshake operation timed out") in .reason.
            reason = getattr(exc, "reason", exc)
            return {"reachable": False, "distributing": False, "detail": f"unreachable ({reason})"}
        ready = [int(x) for x in re.findall(r"<results_ready_to_send>(\d+)<", text)]
        distributing = any(r > 0 for r in ready)
        return {"reachable": True, "distributing": distributing,
                "results_ready_to_send": sum(ready),
                "detail": "work available" if distributing else
                          "no tasks queued (project hibernating since 2020-03-31)"}
