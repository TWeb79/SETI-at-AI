"""Regression tests for the 0.3.0 hardening pass.

Each test names the defect it pins down, so a revert fails loudly rather than silently
reintroducing the behaviour. See implementationplan.md for the mapping.
"""

import numpy as np
import pandas as pd

from ai_seti.ai.features import SCALAR_FEATURES
from ai_seti.ai.model import LABELS, HitScorer, _sklearn_version
from ai_seti.ai.simulate import Injection, inject, noise_waterfall
from ai_seti.config import SearchConfig
from ai_seti.dsp.dedoppler import drift_search
from ai_seti.dsp.preprocess import normalize
from ai_seti.io.filterbank import FilterbankHeader
from ai_seti.pipeline import score_candidates
from ai_seti.report import write_outputs
from ai_seti.sources import drift_ceil_ch_per_step, drift_padding, split

FEATURE_WIDTH = len(SCALAR_FEATURES) + 16 * 16   # scalars + de-drifted snippet


# ---------------------------------------------------------------- classifier artifact

def test_bundled_classifier_loads():
    """The shipped hit_classifier.joblib must unpickle under the pinned scikit-learn.

    Reverting the `<1.9` upper bound in pyproject.toml makes this fail: the artifact was
    pickled with 1.8.x and scikit-learn refuses to unpickle across minor versions.
    """
    scorer = HitScorer()
    assert scorer.load_error is None, f"bundled model failed to load: {scorer.load_error}"
    assert scorer.available, "bundled classifier is unavailable; every hit would be 'unscored'"
    assert list(scorer.meta.get("labels", [])) == list(LABELS)


def test_bundled_classifier_predicts_finite_probabilities():
    """A loaded model must produce real probabilities, not the all-NaN fallback."""
    scorer = HitScorer()
    rng = np.random.default_rng(0)
    probs = scorer.predict(rng.normal(size=(5, FEATURE_WIDTH)).astype(np.float32))
    assert probs.shape == (5, len(LABELS))
    assert np.isfinite(probs).all(), "predictions are NaN: the classifier is not actually running"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_model_load_failure_is_reported(tmp_path):
    """A corrupt model file must set load_error rather than degrading silently."""
    bad = tmp_path / "corrupt.joblib"
    bad.write_bytes(b"not a pickle at all")
    scorer = HitScorer(bad)
    assert not scorer.available
    assert scorer.load_error is not None
    assert "not a pickle" not in scorer.load_error  # the error names the cause, not the payload
    assert np.isnan(scorer.predict(np.zeros((2, FEATURE_WIDTH), dtype=np.float32))).all()


def test_missing_model_file_is_reported_by_name(tmp_path):
    """A configured-but-absent model must name the path the user actually asked for."""
    missing = tmp_path / "absent.joblib"
    scorer = HitScorer(missing)
    assert not scorer.available
    assert str(missing) in scorer.load_error


def test_sklearn_version_is_recorded_in_trained_metadata():
    """`train()` stamps the versions that produced the artifact so a mismatch is explainable."""
    meta = HitScorer().meta
    assert meta.get("sklearn_version"), "trained model does not record its scikit-learn version"
    assert meta.get("sklearn_version") == _sklearn_version(), (
        "the bundled artifact was trained under a different scikit-learn than is installed; "
        "retrain with `ai-seti train` or restore the pin in pyproject.toml")


# ------------------------------------------------------------ unscored is not silent

def _hit_frame(ai_class: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "unit_id": "u:0-10", "channel": 5, "start_channel": 5, "drift_ch_per_step": 0.3,
        "snr": 20.0, "drift_rate_hz_s": 0.4, "frequency_mhz": 1420.5, "snr_map": None,
        "zero_drift": 0.0, "bandwidth_ch": 2.0, "on_fraction": 1.0, "modulation": 0.1,
        "kurtosis": 3.0, "wobble_rms": 1.0, "sidelobe_ratio": 0.1, "off_track_level": 0.0,
        "peak_to_mean": 4.0, "abs_drift_ch": 0.3, "ai_class": ai_class,
    }])


def test_score_candidates_flags_unavailable_model(tmp_path, caplog):
    """A missing model must be named in the output, not reported as a plain 'unscored'."""
    cfg = SearchConfig(model_path=str(tmp_path / "absent.joblib"))
    with caplog.at_level("WARNING"):
        cands = score_candidates([{"hits": _hit_frame("x").to_dict("records")}], cfg)
    assert (cands["ai_class"] == "model_unavailable").all()
    assert cands.attrs["ai_note"], "the reason must travel with the frame, not be dropped"
    assert "absent.joblib" in cands.attrs["ai_note"]

def test_score_candidates_reports_disabled_ai():
    cfg = SearchConfig(use_ai=False)
    cands = score_candidates([{"hits": _hit_frame("x").to_dict("records")}], cfg)
    assert (cands["ai_class"] == "ai_disabled").all()


def test_write_outputs_records_ai_note(tmp_path):
    """The inactive-AI warning must survive into the run metadata and the HTML report."""
    cfg = SearchConfig(model_path=str(tmp_path / "absent.joblib"))
    cands = score_candidates([{"hits": _hit_frame("x").to_dict("records")}], cfg)
    stats = write_outputs([], cands, tmp_path / "out", cfg, "t")
    assert "ai_note" in stats and "classifier" in stats["ai_note"]
    meta = (tmp_path / "out" / "metadata.json").read_text()
    assert "ai_note" in meta
    html = (tmp_path / "out" / "report.html").read_text()
    assert "ai_note" in html and 'id="warn"' in html


# -------------------------------------------------------------------- drift cap works

def _header(nsamples: int = 16) -> FilterbankHeader:
    return FilterbankHeader(fch1=1420.0, foff=-2.7939677238464355e-06, nchans=65536,
                            tsamp=18.253611008, nsamples=nsamples, source_name="T")


def test_drift_cap_limits_searched_paths():
    """max_drift_ch_per_step must actually bound the search.

    Before the fix the cap was never passed to drift_search and its only other reader
    computed min(max(k,1), max(cap, k)), which is always k: the knob did nothing.
    """
    rng = np.random.default_rng(2)
    z = normalize(noise_waterfall(16, 16384, rng), block=256)
    inject(z, Injection("technosignature_like", 8000.0, 8.0, 4.0, 1.0), rng)
    # 8 ch/step at 2.79 Hz channels is 22 Hz/s, far beyond the 4 Hz/s drift ceiling, so the
    # search only reaches it because the cap is absent.
    loose, _ = drift_search(z, tsamp=18.253611008, foff_mhz=2.7939677238464355e-06,
                            max_drift_hz_s=4.0, snr_threshold=8.0)
    capped, _ = drift_search(z, tsamp=18.253611008, foff_mhz=2.7939677238464355e-06,
                             max_drift_hz_s=4.0, max_ch_per_step=4, snr_threshold=8.0)
    assert any(abs(h.drift_ch_per_step - 8.0) < 1.0 for h in loose), "search should find the 8 ch/step tone"
    assert not any(abs(h.drift_ch_per_step - 8.0) < 1.0 for h in capped), \
        "a 4 ch/step cap must exclude an 8 ch/step tone"


def test_drift_search_recovers_a_tone_at_gbt_coarse_resolution():
    """The search must work at the real channel width, not just the narrow test grids.

    Every other drift test uses a synthetic 16384-channel block at an invented resolution.
    Production is a GBT coarse channel: 2.794 Hz channels sampled every 18.25 s, where the
    4 Hz/s ceiling is already ~26 channels per step. This pins that regime end to end, and
    pins the `foff_mhz` unit convention -- `drift_search` multiplies by 1e6 internally, so a
    channel width quoted in Hz must be passed as 2.7939677238464355e-06, not 2.794. Passing
    the Hz value silently collapses the searchable drift to zero and the search reports
    nothing at all (backlog B1/B2).
    """
    foff_mhz = 2.7939677238464355e-06  # 2.794 Hz, expressed in MHz as drift_search expects
    rng = np.random.default_rng(11)
    n_chan = 32768
    data = noise_waterfall(16, n_chan, rng, fine_per_coarse=n_chan)
    inject(data, Injection("technosignature_like", 8000.0, 13.0, 6.0, 1.0), rng)
    z = normalize(data, block=512, fine_per_coarse=n_chan)

    hits, snr = drift_search(z, tsamp=18.253611008, foff_mhz=foff_mhz, max_drift_hz_s=4.0,
                             max_ch_per_step=32, snr_threshold=10.0, max_hits=50)

    best = max(hits, key=lambda h: h.snr)
    assert abs(best.start_channel - 8000) <= 2, "start channel must be located, not approximated"
    assert abs(best.drift_ch_per_step - 13.0) <= 0.5, \
        "a 13 ch/step tone must be labelled 13 ch/step, not smeared to a neighbour"
    assert best.snr > 10.0, "the recovered SNR must clear the threshold it was found at"
    assert float(snr[8000]) > 10.0, "the SNR map must carry the detection for the dashboard"


def test_drift_ceil_respects_configured_cap():
    cfg = SearchConfig(max_drift_rate_hz_s=4.0, max_drift_ch_per_step=4)
    hdr = _header()
    k_needed = int(np.ceil(4.0 * hdr.tsamp / abs(hdr.foff * 1e6)))
    assert k_needed > 4, "test header must need more drift than the cap allows"
    assert drift_ceil_ch_per_step(hdr, cfg) == 4
    assert drift_padding(hdr, cfg) >= 4 * hdr.nsamples


def test_drift_padding_covers_searched_drift():
    """The guard band and the search must agree, or edge signals are searched but unseen."""
    hdr = _header(nsamples=64)
    for cap in (1, 2, 4, 8):
        cfg = SearchConfig(max_drift_ch_per_step=cap)
        searchable = drift_ceil_ch_per_step(hdr, cfg) * (hdr.nsamples - 1)
        assert drift_padding(hdr, cfg) >= searchable


def test_split_units_are_padded_by_the_shared_rule(tmp_path):
    hdr = _header()
    cfg = SearchConfig(channels_per_unit=8192, max_drift_ch_per_step=4)
    units = split("obs.fil", hdr, cfg, "local")
    assert len(units) == hdr.nchans // 8192
    pad = drift_padding(hdr, cfg)
    first = units[0]
    assert first.chan_start == 0 and first.chan_stop >= first.core_stop + pad - 8
    mid = units[1]
    assert mid.chan_start == mid.core_start - pad, "interior units must be padded on both sides"
