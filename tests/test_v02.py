import http.server
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ai_seti.ai.simulate import Injection, inject, noise_waterfall
from ai_seti.config import SearchConfig
from ai_seti.dsp.dedoppler import drift_search, taylor_tree
from ai_seti.dsp.preprocess import normalize
from ai_seti.io.filterbank import FilterbankHeader, read_header, read_window, write_sigproc
from ai_seti.pipeline import run_units, score_candidates
from ai_seti.rfi import cadence_filter
from ai_seti.sources import split


def test_taylor_tree_matches_brute_force_endpoints():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 256)).astype(np.float32)
    tree = taylor_tree(x)
    assert np.allclose(tree[0], x.sum(0), atol=1e-4)
    assert np.isclose(tree[15][10], sum(x[t, 10 + t] for t in range(16)), atol=1e-4)


def test_drift_search_recovers_positive_and_negative_drifts():
    rng = np.random.default_rng(1)
    z = rng.normal(size=(16, 16384)).astype(np.float32)
    for t in range(16):
        z[t, 4000 + round(3.4 * t)] += 3
        z[t, 9000 - round(7.2 * t)] += 3
    hits, _ = drift_search(z, tsamp=1.0, foff_mhz=1e-6, max_drift_hz_s=9, snr_threshold=8)
    found = {(h.start_channel, round(h.drift_ch_per_step)) for h in hits}
    assert (4000, 3) in found
    assert (9000, -7) in found


def test_noise_only_has_no_hits_at_default_threshold():
    rng = np.random.default_rng(2)
    z = rng.normal(size=(16, 32768)).astype(np.float32)
    hits, _ = drift_search(z, tsamp=1.0, foff_mhz=1e-6, max_drift_hz_s=4, snr_threshold=10)
    assert hits == []


def test_normalize_removes_bandpass_and_dc_spike():
    rng = np.random.default_rng(3)
    d = noise_waterfall(16, 8192, rng, fine_per_coarse=8192)
    z = normalize(d, block=256, fine_per_coarse=8192)
    assert abs(float(np.median(z))) < 0.05
    assert 0.8 < float(np.std(z)) < 1.25
    assert abs(float(z[:, 4096].mean())) < 2


def _write_fil(path: Path, n_chan=65536):
    rng = np.random.default_rng(4)
    d = noise_waterfall(16, n_chan, rng)
    inject(d, Injection("technosignature_like", n_chan * 0.305, 2.5, 3.0, 1.0), rng)
    hdr = FilterbankHeader(fch1=1420.0, foff=-2.7939677238464355e-06, nchans=n_chan,
                           tsamp=18.253611008, nsamples=16, source_name="TEST")
    write_sigproc(path, d, hdr)
    return d


def test_sigproc_roundtrip_and_window(tmp_path):
    p = tmp_path / "t.fil"
    d = _write_fil(p)
    hdr = read_header(p)
    assert hdr.nchans == d.shape[1] and hdr.nsamples == 16
    assert np.allclose(read_window(p, 100, 200, hdr), d[:, 100:200])


def test_pipeline_finds_injected_signal_once(tmp_path):
    p = tmp_path / "t.fil"
    _write_fil(p)
    hdr = read_header(p)
    cfg = SearchConfig(channels_per_unit=16384, workers=1)
    units = split(str(p), hdr, cfg)
    results = list(run_units(units, cfg, workers=1))
    cands = score_candidates(results, cfg)
    near = cands[(cands["channel"] - 65536 * 0.305).abs() <= 3]
    assert len(near) == 1                       # found, and not duplicated by unit overlap
    assert near.iloc[0]["drift_ch_per_step"] > 2


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = Path(self.translate_path(self.path))
        data = path.read_bytes()
        rng = self.headers.get("Range")
        a, b = rng.split("=")[1].split("-")
        a = int(a)
        b = int(b) if b else len(data) - 1
        b = min(b, len(data) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self.wfile.write(data[a:b + 1])

    def log_message(self, *args):
        pass


def test_remote_range_streaming(tmp_path):
    from functools import partial

    from ai_seti.io.remote import remote_header, remote_window
    p = tmp_path / "r.fil"
    d = _write_fil(p, 8192)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0),
                                          partial(_RangeHandler, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/r.fil"
        hdr = remote_header(url)
        assert hdr.nchans == 8192 and hdr.nsamples == 16
        assert np.allclose(remote_window(url, hdr, 1000, 1300), d[:, 1000:1300])
    finally:
        srv.shutdown()


class _EmptyRangeHandler(http.server.BaseHTTPRequestHandler):
    """Answers every Range request with a well-formed but empty 206.

    This is what a truncating proxy or a half-open link looks like to the client: the status
    line and `Content-Range` are correct, and the body is simply absent.
    """

    def do_GET(self):
        type(self).requests += 1
        self.send_response(206)
        self.send_header("Content-Range", "bytes 0-16383/1073741824")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


def test_remote_header_gives_up_when_the_server_stops_sending():
    """An empty 206 must terminate the header read (backlog B4).

    Before the fix the buffer never grew, so the `len(buf) > 1 << 20` escape was unreachable
    and the identical Range was re-requested forever — measured at 10,798 requests in 5 s
    against a local server. Against the real Breakthrough Listen archive that is an
    unthrottled request flood from every volunteer whose connection hiccups.
    """
    from ai_seti.io.remote import remote_header

    handler = type("Counting", (_EmptyRangeHandler,), {"requests": 0})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/obs.fil"
        started = time.monotonic()
        with pytest.raises(ValueError, match="No SIGPROC header"):
            remote_header(url, max_requests=8)
        elapsed = time.monotonic() - started
    finally:
        srv.shutdown()
    assert elapsed < 1.0, "must fail fast instead of retrying indefinitely"
    assert handler.requests <= 2, \
        f"at most one wasted request was needed, saw {handler.requests}"


def test_cadence_filter_keeps_on_only_signal():
    sig = pd.DataFrame({"frequency_mhz": [1420.0, 1421.0], "drift_rate_hz_s": [0.5, 0.0],
                        "snr": [20, 30]})
    rfi_only = pd.DataFrame({"frequency_mhz": [1421.0], "drift_rate_hz_s": [0.0], "snr": [30]})
    dt = 300 / 86400
    scans = [
        {"name": "A", "mjd": 0.0, "is_on": True, "hits": sig},
        {"name": "B_OFF", "mjd": dt, "is_on": False, "hits": rfi_only},
        {"name": "A2", "mjd": 2 * dt, "is_on": True,
         "hits": sig.assign(frequency_mhz=sig.frequency_mhz + sig.drift_rate_hz_s * 600 / 1e6)},
    ]
    ev = cadence_filter(scans)
    assert list(ev["frequency_mhz"]) == [1420.0]
