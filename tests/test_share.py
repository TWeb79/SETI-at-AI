import http.server
import json
import threading
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from ai_seti.share import build_findings, finding_id, passes_gate, share
from ai_seti.state import Ledger


def _report(tmp: Path, source="breakthrough_listen") -> Path:
    d = tmp / "run"
    d.mkdir()
    pd.DataFrame([
        {"frequency_mhz": 1420.123456, "drift_rate_hz_s": 0.42, "snr": 18.0, "bandwidth_ch": 1,
         "on_fraction": 1.0, "channel": 5, "zero_drift": 0, "interest": 80.0,
         "ai_class": "technosignature_like", "p_technosignature_like": 0.9, "anomaly": 0.6,
         "known_rfi_band": False},
        {"frequency_mhz": 1575.42, "drift_rate_hz_s": 0.0, "snr": 40.0, "bandwidth_ch": 2,
         "on_fraction": 1.0, "channel": 9, "zero_drift": 1, "interest": 75.0,
         "ai_class": "rfi_zero_drift", "p_technosignature_like": 0.01, "anomaly": 0.2,
         "known_rfi_band": True},
    ]).to_csv(d / "candidates.csv", index=False)
    np.save(d / "snippets.npy", np.zeros((2, 16, 16), dtype=np.float32))
    (d / "metadata.json").write_text(json.dumps({
        "location": "/home/someone/secret/path.fil",
        "observation": {"target": "HIP 12345", "telescope": "GBT", "source": source,
                        "url": "http://example.org/obs.fil", "md5sum": "abc", "mjd": 59000.5},
        "header": {"tstart": 59000.5, "tsamp": 18.25, "foff": -2.79e-6, "source_name": "HIP"},
        "config": {"snr_threshold": 10}, "versions": {"ai_seti": "0.2.0"}}))
    return d


class _Capture(http.server.BaseHTTPRequestHandler):
    posts: ClassVar[list] = []
    existing: ClassVar[int] = 0

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        _Capture.posts.append((self.path, json.loads(self.rfile.read(n)),
                               self.headers.get("Authorization")))
        self._send(201, {"html_url": "https://github.example/issues/1"})

    def do_GET(self):   # GitHub search
        self._send(200, {"total_count": _Capture.existing,
                         "items": [{"html_url": "https://github.example/issues/0"}]})

    def log_message(self, *a):
        pass


@pytest.fixture()
def server():
    _Capture.posts, _Capture.existing = [], 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Capture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_record_is_stable_private_and_gated(tmp_path):
    recs = build_findings(_report(tmp_path), top=5)
    assert len(recs) == 2
    r = recs[0]
    assert "max_drift_ch_per_step" in r["processing"]["config"], "B17: the binding drift cap"
    assert r["status"] == "unverified_candidate"
    assert r["finding_id"] == finding_id(r["observation"], 1420.123456, 0.42)
    assert "secret" not in json.dumps(r)                    # local paths never leave
    assert r["observation"]["data_url"] == "http://example.org/obs.fil"
    assert passes_gate(recs[0])[0] and not passes_gate(recs[1])[0]
    assert not passes_gate(recs[0], require_cadence=True)[0]


def test_webhook_and_ledger_dedup(tmp_path, server):
    recs = build_findings(_report(tmp_path), top=5)
    ledger = Ledger(path=tmp_path / "state.json")
    res, skipped = share(recs, ["bundle", "webhook"], bundle_dir=tmp_path / "share", require_cadence=False,
                         ledger=ledger, webhook=server + "/hook", webhook_format="discord")
    assert [r.ok for r in res] == [True, True]
    assert (tmp_path / "share" / f"{recs[0]['finding_id']}.zip").exists()
    assert "content" in _Capture.posts[0][1]
    assert any("zero drift" in why for _, why in skipped)   # RFI never sent
    res2, skipped2 = share(recs, ["webhook"], bundle_dir=tmp_path / "share", ledger=ledger, require_cadence=False,
                           webhook=server + "/hook")
    assert res2 == [] and any("already shared" in w for _, w in skipped2)


def test_github_issue_created_once(tmp_path, server, monkeypatch):
    monkeypatch.setenv("AI_SETI_GITHUB_TOKEN", "t0ken")
    recs = build_findings(_report(tmp_path), top=1)
    res, _ = share(recs, ["github"], bundle_dir=tmp_path, repo="org/finds", github_api=server, require_cadence=False)
    assert res[0].ok and _Capture.posts[0][0] == "/repos/org/finds/issues"
    assert _Capture.posts[0][2] == "Bearer t0ken"
    assert recs[0]["finding_id"] in _Capture.posts[0][1]["title"]
    _Capture.existing = 1                                    # someone already reported it
    res, _ = share(recs, ["github"], bundle_dir=tmp_path, repo="org/finds", github_api=server, require_cadence=False)
    assert "already reported" in res[0].detail and len(_Capture.posts) == 1


def test_synthetic_data_stays_local(tmp_path, server):
    recs = build_findings(_report(tmp_path, source="synthetic"), top=1)
    res, skipped = share(recs, ["webhook"], bundle_dir=tmp_path, webhook=server, require_cadence=False)
    assert res == [] and "synthetic" in skipped[0][1] and _Capture.posts == []


def test_uncadenced_finding_goes_to_bundle_but_not_remote_by_default(tmp_path, server):
    """B8: without an explicit opt-out, only cadence-passing finds are published."""
    recs = build_findings(_report(tmp_path), top=5, min_interest=0)[:1]
    res, skipped = share(recs, ["bundle", "webhook"], bundle_dir=tmp_path / "share", webhook=server)
    assert [r.destination for r in res] == ["bundle"] and _Capture.posts == []
    assert any("no cadence pass" in why for _, why in skipped)
