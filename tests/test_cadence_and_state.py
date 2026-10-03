"""Regression tests for the cadence, sharing and ledger defects found in the 0.3.0 review."""
import http.server
import json
import threading

import pandas as pd
import pytest

from ai_seti.rfi import cadence_filter, cadence_is_testable
from ai_seti.share import _cadence_status, share
from ai_seti.state import Ledger


@pytest.fixture()
def server(tmp_path):
    """Local endpoint that accepts any POST, so share() sees a real 2xx."""
    received: list = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/hook", tmp_path, received
    finally:
        httpd.shutdown()


def tmp_path_for(server):
    return server[1]


def _hits(freqs=(1420.5,), drifts=(0.5,), snrs=(20.0,)) -> pd.DataFrame:
    return pd.DataFrame({"frequency_mhz": list(freqs), "drift_rate_hz_s": list(drifts),
                         "snr": list(snrs)})


def _shift(dh: pd.DataFrame, secs: float, rate: float = 0.5) -> pd.DataFrame:
    return dh.assign(frequency_mhz=dh.frequency_mhz + rate * secs / 1e6)


# ------------------------------------------------- cadence cannot be passed trivially

def test_cadence_rejects_single_on_scan_without_off():
    """A lone ON scan with no OFF scan must not be reported as a cadence pass.

    Before the fix the reference hit matched itself, giving on_found/len(ons) == 1.0 and
    off_found == 0, so an unconditional "passed_cadence".
    """
    hit = _hits()
    scans = [{"name": "A", "mjd": 0.0, "is_on": True, "hits": hit}]
    ok, reason = cadence_is_testable(scans)
    assert not ok and "ON scan" in reason
    assert cadence_filter(scans).empty, "a single ON scan must not produce an event"


def test_cadence_requires_an_off_scan():
    hit = _hits()
    scans = [
        {"name": "A", "mjd": 0.0, "is_on": True, "hits": hit},
        {"name": "A2", "mjd": 0.001, "is_on": True, "hits": _shift(hit, 86.4)},
    ]
    ok, reason = cadence_is_testable(scans)
    assert not ok and "OFF" in reason
    assert cadence_filter(scans).empty


def test_cadence_passes_with_on_and_off_and_rejects_off_persistence():
    """The real case: the tone follows the drift across ON scans and is absent when OFF."""
    on1, on2 = _hits(), _shift(_hits(), 86.4)
    scans = [
        {"name": "A", "mjd": 0.0, "is_on": True, "hits": on1},
        {"name": "B_OFF", "mjd": 0.0005, "is_on": False, "hits": _hits(freqs=(1435.0,), drifts=(0.0,))},
        {"name": "A2", "mjd": 0.001, "is_on": True, "hits": on2},
    ]
    ok, _ = cadence_is_testable(scans)
    assert ok
    events = cadence_filter(scans)
    assert len(events) == 1
    assert events.iloc[0]["n_on_scans"] == 2
    assert events.iloc[0]["off_scans_checked"] == 1


def test_cadence_rejects_signal_also_present_in_off_scan():
    """The distinguishing power of cadence: a tone in every scan is terrestrial."""
    hit = _hits()
    scans = [
        {"name": "A", "mjd": 0.0, "is_on": True, "hits": hit},
        {"name": "B_OFF", "mjd": 0.0005, "is_on": False, "hits": hit},
        {"name": "A2", "mjd": 0.001, "is_on": True, "hits": _shift(hit, 86.4)},
    ]
    assert cadence_filter(scans).empty


def test_cadence_requires_the_drift_to_be_reproducible():
    """Same frequency in both ON scans but the drift not reproduced is not a pass."""
    scans = [
        {"name": "A", "mjd": 0.0, "is_on": True, "hits": _hits(drifts=(0.5,))},
        {"name": "B_OFF", "mjd": 0.0005, "is_on": False, "hits": _hits(freqs=(1435.0,), drifts=(0.0,))},
        {"name": "A2", "mjd": 0.001, "is_on": True,
         "hits": _shift(_hits(drifts=(0.5,)), 86.4).assign(drift_rate_hz_s=3.0)},
    ]
    assert cadence_filter(scans).empty


# ------------------------------------------- share must not honour a hollow pass

def test_cadence_status_rejects_thin_events():
    """A hand-written or pre-fix events.csv claiming one ON scan must grade as failed."""
    row = pd.Series({"frequency_mhz": 1420.5})
    hollow = pd.DataFrame([{"frequency_mhz": 1420.5, "n_on_scans": 1, "off_scans_checked": 0}])
    assert _cadence_status(row, hollow) == "failed"
    no_off = pd.DataFrame([{"frequency_mhz": 1420.5, "n_on_scans": 3, "off_scans_checked": 0}])
    assert _cadence_status(row, no_off) == "failed"
    proper = pd.DataFrame([{"frequency_mhz": 1420.5, "n_on_scans": 3, "off_scans_checked": 3}])
    assert _cadence_status(row, proper) == "passed"


def test_cadence_status_without_events():
    row = pd.Series({"frequency_mhz": 1420.5})
    assert _cadence_status(row, None) == "not_run"
    assert _cadence_status(row, pd.DataFrame()) == "failed"


# --------------------------------------------------------- dedupe keys on finding_id

def _rec(fid: str, freq: float = 1420.5) -> dict:
    """Minimal shareable record: passes the gate and has no snippet."""
    return {
        "finding_id": fid, "status": "unverified_candidate",
        "signal": {"frequency_mhz": freq, "drift_rate_hz_s": 0.4, "snr": 20.0,
                   "bandwidth_ch": 1, "on_fraction": 1.0, "channel": 5},
        "assessment": {"interest": 90.0, "ai_class": "technosignature_like",
                       "p_technosignature_like": 0.9, "anomaly": 0.5,
                       "known_rfi_band": False, "cadence": "not_run", "rank_in_run": 1},
        "observation": {"target": "T", "telescope": None, "mjd": 59000.0, "ra": None,
                        "decl": None, "data_url": None, "md5sum": None,
                        "source": "breakthrough_listen", "synthetic": False,
                        "tsamp_s": 18.25, "channel_width_hz": 2.79},
        "checklist": [
            {"check": "Drifts (non-zero Doppler drift, as an off-Earth source would)", "result": True},
            {"check": "Outside known satellite / RFI bands", "result": True},
        ],
        "disclaimer": "Statistical candidate from automated analysis. Not a claim.",
    }


def test_share_dedupes_by_id_not_record_value(tmp_path):
    """Two distinct findings with identical payloads must both be delivered.

    The previous implementation tested `rec not in batch`, i.e. deep dict equality. Records
    differ only by finding_id, so this passed by luck; it would have dropped a legitimate
    finding the moment two records compared equal, and cost O(n^2) to get there.
    """
    a, b = _rec("AIS-AAA"), _rec("AIS-BBB")
    sent, skipped = share([a, b], ["bundle"], bundle_dir=tmp_path / "out")
    assert [r.finding_id for r in sent] == ["AIS-AAA", "AIS-BBB"]
    assert skipped == []


def test_share_ledger_blocks_resend_of_the_same_id(tmp_path):
    rec = _rec("AIS-CCC")
    ledger = Ledger(path=tmp_path / "state.json")
    first, _ = share([rec], ["bundle"], bundle_dir=tmp_path / "out", ledger=ledger)
    assert [r.finding_id for r in first] == ["AIS-CCC"]
    second, skipped = share([rec], ["bundle"], bundle_dir=tmp_path / "out", ledger=ledger)
    assert second == []
    assert any("already shared" in why for _, why in skipped)


def test_share_ledger_records_one_entry_per_destination(server):
    """webhook:discord must normalise to 'webhook' so a later json send is still blocked.

    Also pins that the ledger only records *successful* sends.
    """
    url, tmp_path, received = server
    rec = _rec("AIS-DDD")
    ledger = Ledger(path=tmp_path / "state.json")
    share([rec], ["webhook"], bundle_dir=tmp_path / "out", ledger=ledger,
          webhook=url, webhook_format="discord")
    assert received, "the finding should have reached the webhook"
    assert ledger.shared["AIS-DDD"] == ["webhook"]
    _, skipped = share([rec], ["webhook"], bundle_dir=tmp_path / "out", ledger=ledger,
                       webhook=url, webhook_format="json")
    assert any("already shared" in why for _, why in skipped)
    assert len(received) == 1, "the second send must not have reached the webhook"


def test_failed_send_is_not_recorded_in_the_ledger(tmp_path):
    """A webhook that errors must stay eligible for a later retry."""
    rec = _rec("AIS-EEE")
    ledger = Ledger(path=tmp_path / "state.json")
    res, _ = share([rec], ["webhook"], bundle_dir=tmp_path / "out", ledger=ledger,
                   webhook="http://127.0.0.1:1/hook")   # nothing listening
    assert res and not res[0].ok
    assert "AIS-EEE" not in ledger.shared


# ------------------------------------------------------------------------- ledger

def test_ledger_roundtrip_and_best(tmp_path):
    path = tmp_path / "state.json"
    led = Ledger(path=path)
    led.record_unit({"n_channels": 1000, "timings": {"load": 0.5}, "hits": [{}, {}]})
    assert led.record_best({"interest": 80.0, "frequency_mhz": 1420.0, "_snippet": [1, 2]})
    assert not led.record_best({"interest": 10.0})
    assert not led.record_best({})
    led.done("obs-a")
    led.done("obs-a")
    led.save()
    back = Ledger.load(path)
    assert back.work_units == 1 and back.channels == 1000
    assert back.hits == 2 and back.cpu_seconds == 0.5
    assert back.observations_done == ["obs-a"]
    assert back.best["interest"] == 80.0
    assert "_snippet" not in back.best, "private columns must not enter the ledger"


def test_ledger_ignores_unknown_keys(tmp_path):
    """A ledger written by another version must load, not die on an unexpected keyword."""
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"work_units": 7, "channels": 42, "from_the_future": True}))
    led = Ledger.load(path)
    assert led.work_units == 7 and led.channels == 42
    led.save()
    assert "from_the_future" not in json.loads(path.read_text())


def test_ledger_save_is_atomic(tmp_path):
    """No leftover temp files, and the published file is complete JSON."""
    path = tmp_path / "state.json"
    led = Ledger(path=path)
    led.record_unit({"n_channels": 5, "timings": {}, "hits": []})
    led.save()
    led.work_units += 10
    led.save()
    assert json.loads(path.read_text())["work_units"] == 11
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"], "temp file was left behind"
