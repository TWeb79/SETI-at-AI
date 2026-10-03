"""Stream channel windows of remote SIGPROC files via HTTP Range requests.

A Breakthrough Listen high-resolution product is several GB, but a SETI@home-style
work unit only needs ~1 MHz of band. Because SIGPROC stores one spectrum per time
step contiguously, each time step of a channel window is one contiguous byte range:
we fetch the 16 ranges concurrently instead of downloading the whole file — the
same idea as the SETI@home "splitter", done on the client.
"""
from __future__ import annotations

import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .filterbank import FilterbankHeader, parse_sigproc_header

USER_AGENT = "AI-SETI/0.2 (+https://github.com/; research client)"


class RangeNotSupported(RuntimeError):
    pass


def _get(url: str, start: int | None = None, stop: int | None = None,
         timeout: float = 60.0, retries: int = 3) -> tuple[bytes, dict, int]:
    headers = {"User-Agent": USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if stop is None else stop - 1}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                if start is not None and status != 206:
                    raise RangeNotSupported(f"Server ignored Range header ({status}) for {url}")
                want = None if stop is None or start is None else stop - start
                body = resp.read(want) if want else resp.read()
                return body, dict(resp.headers), status
        except RangeNotSupported:
            raise
        except Exception as exc:  # network hiccup: exponential back-off
            last = exc
            time.sleep(1.5 ** attempt)
    raise ConnectionError(f"GET {url} failed after {retries} attempts: {last}")


def remote_header(url: str) -> FilterbankHeader:
    total = None
    buf = b""
    chunk = 16384
    while True:
        body, hdrs, _ = _get(url, len(buf), len(buf) + chunk)
        buf += body
        m = re.search(r"/(\d+)$", hdrs.get("Content-Range", ""))
        if m:
            total = int(m.group(1))
        try:
            return parse_sigproc_header(buf, total)
        except EOFError as exc:
            if len(buf) > 1 << 20:
                raise ValueError("SIGPROC header larger than 1 MB?") from exc


def remote_window(url: str, header: FilterbankHeader, chan_start: int, chan_stop: int,
                  max_threads: int = 8, if_index: int = 0) -> np.ndarray:
    """Fetch [time, chan_start:chan_stop] as float32 using one Range request per time step."""
    bpv = header.bytes_per_value
    row = header.nifs * header.nchans * bpv

    def fetch(t: int) -> np.ndarray:
        a = header.data_offset + t * row + (if_index * header.nchans + chan_start) * bpv
        body, _, _ = _get(url, a, a + (chan_stop - chan_start) * bpv)
        return np.frombuffer(body, dtype=np.dtype(header.dtype).newbyteorder("<"))

    with ThreadPoolExecutor(max_workers=max_threads) as pool:
        rows = list(pool.map(fetch, range(header.nsamples)))
    return np.vstack(rows).astype(np.float32, copy=False)


def download(url: str, dest, max_mb: int = 2048, progress=None) -> None:
    """Plain streamed download with a size guard (used for .h5 products)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        size = int(resp.headers.get("Content-Length") or 0)
        if size and size > max_mb * 1024 * 1024:
            raise ValueError(f"{url} is {size / 2**20:.0f} MB > --max-download-mb {max_mb}")
        done = 0
        tmp = f"{dest}.part"
        with open(tmp, "wb") as fh:
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if done > max_mb * 1024 * 1024:
                    raise ValueError("Download exceeded size guard")
                if progress:
                    progress(done, size)
    import os
    os.replace(tmp, dest)
