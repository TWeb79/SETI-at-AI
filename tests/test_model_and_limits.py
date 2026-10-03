"""Regression tests for the 0.3.0 hardening pass.

Each test names the defect it pins down, so a revert fails loudly rather than silently
reintroducing the behaviour. See implementationplan.md for the mapping.
"""

import fnmatch
from pathlib import Path

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _would_be_ignored(rel_path: str) -> list[str]:
    """Approximate gitignore matching; returns the patterns that exclude `rel_path`, or [].

    Later patterns win, as in git: a matching negation (`!`) re-includes the path, which is
    what keeps the `.gitkeep` placeholders trackable under `reports/*` and `data/raw/*`. A
    pattern matches when it covers the path or any ancestor directory.
    """
    parts = rel_path.split("/")
    candidates = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
    matched: list[str] = []
    ignored = False
    for raw in (PROJECT_ROOT / ".gitignore").read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        pattern = line.lstrip("!").rstrip("/")
        if any(fnmatch.fnmatch(c, pattern) for c in candidates):
            ignored = not negate
            matched = [] if negate else [*matched, line]
    return matched if ignored else []


def test_bundled_model_artifact_is_not_gitignored():
    """The shipped classifier must stay in version control.

    If this artifact is ever excluded, a fresh clone silently ships with no AI layer and
    every candidate comes back `unscored` — precisely the failure the 0.3.0 pass was built to
    stop hiding. A broad `*.pkl`/`*.joblib`/`src/` rule would reintroduce it invisibly.
    """
    artifact = Path("src/ai_seti/ai/hit_classifier.joblib")
    assert (PROJECT_ROOT / artifact).is_file(), "the bundled model must exist in the tree"
    assert _would_be_ignored(str(artifact)) == [], \
        f"{artifact} must be committed; matched {artifact} would be ignored"


def test_runtime_data_is_gitignored():
    """Regenerable observations, caches and reports must not be committable by accident."""
    for rel in ("data/state.json", "data/raw/bl/obs.fil", "reports/crunch/x/report.html",
                "reports/share/finding.json", ".coverage", ".venv/bin/python"):
        assert _would_be_ignored(rel), f"{rel} should be ignored"
    for rel in ("README.md", "pyproject.toml", "requirements.txt", "configs/default.json",
                "tests/test_core.py", "reports/.gitkeep", "data/raw/.gitkeep"):
        assert _would_be_ignored(rel) == [], f"{rel} must stay trackable"


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


def test_far_out_of_range_snr_is_not_trusted_as_technosignature():
    """B8: the classifier saturates at p=1 for any strong drifting carrier."""
    rows = pd.concat([_hit_frame("x")] * 2, ignore_index=True)
    rows["snr"] = [20.0, 10_000.0]
    rows["channel"] = [5, 900]
    for label in LABELS:
        rows[f"p_{label}"] = 1.0 if label == "technosignature_like" else 0.0
    cands = score_candidates([{"hits": rows.to_dict("records")}], SearchConfig())
    strong = cands[cands["snr"] == 10_000.0].iloc[0]
    normal = cands[cands["snr"] == 20.0].iloc[0]
    assert strong["ai_class"] == "out_of_distribution" and strong["interest"] <= 50
    assert normal["ai_class"] == "technosignature_like" and normal["interest"] > 50


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


def test_drift_search_never_reports_beyond_the_configured_rate():
    """Reported drift rates must respect `max_drift_rate_hz_s` (backlog B9).

    The Taylor tree always sweeps drift 0…1 ch/step because of its `for d in range(tp)`
    term, even when `k_max == 0`. On a coarse product one channel per step is far past the
    configured ceiling: the BL mid-resolution product (`.0002`, 2.86 kHz × 1.07 s) makes one
    channel per step ±2,665 Hz/s. A real run reported candidates at −276 to −1,225 Hz/s
    with a 4 Hz/s limit.
    """
    foff_mhz = 2860.0e-06          # 2.86 kHz channels: 1 ch/step is 2,672 Hz/s at tsamp 1.07
    tsamp = 1.07
    rng = np.random.default_rng(21)
    data = noise_waterfall(256, 8192, rng)
    inject(data, Injection("technosignature_like", 4000.0, 0.5, 8.0, 1.0), rng)
    z = normalize(data, block=256)

    # 4 Hz/s is 0.0015 ch/step on this product, so the 0.5 ch/step tone sits far outside the
    # ceiling and must not be reported at all. Before the fix it came back at -1,329.7 Hz/s.
    tight, _ = drift_search(z, tsamp=tsamp, foff_mhz=foff_mhz, max_drift_hz_s=4.0,
                            snr_threshold=8.0)
    assert all(abs(h.drift_rate_hz_s) <= 4.0 for h in tight), \
        "no reported hit may exceed the configured rate"
    assert not any(abs(h.drift_ch_per_step - 0.5) < 0.05 for h in tight), \
        "a 0.5 ch/step tone is ~1,336 Hz/s here and must not survive a 4 Hz/s ceiling"

    # The same search with a ceiling that admits the tone must still recover it, so the cap
    # filters the reported range rather than silently disabling the drift search.
    loose, _ = drift_search(z, tsamp=tsamp, foff_mhz=foff_mhz, max_drift_hz_s=1400.0,
                            snr_threshold=8.0)
    assert any(abs(h.drift_ch_per_step - 0.5) < 0.05 for h in loose), \
        "a ceiling above the tone's own rate must still recover it"
    assert all(abs(h.drift_rate_hz_s) <= 1400.0 for h in loose)


def test_drift_search_skips_the_tree_when_only_zero_drift_is_in_range():
    """Below 1/span channels per step, the column sum is the whole answer.

    A coarse product can make `max_drift_rate_hz_s` correspond to far less than one channel
    per step. Skipping the tree is then an exact shortcut, not an approximation, because
    `tree[0]` is precisely the column sum.
    """
    rng = np.random.default_rng(22)
    data = noise_waterfall(64, 4096, rng)
    inject(data, Injection("technosignature_like", 2000.0, 0.0, 8.0, 1.0), rng)
    z = normalize(data, block=256)
    # 0.01 Hz/s at 2.86 kHz channels is 3.7e-6 ch/step, far below the 1/63 tree resolution.
    hits, _ = drift_search(z, tsamp=1.07, foff_mhz=2860.0e-06, max_drift_hz_s=0.01,
                           snr_threshold=8.0)
    assert hits, "a zero-drift carrier must survive the ceiling"
    assert all(h.drift_ch_per_step == 0.0 for h in hits)


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


def test_partial_range_units_are_padded_beyond_the_range_edges():
    """--max-units chunk edges are not data edges: a tone drifting across one needs its pad."""
    hdr, cfg = _header(), SearchConfig(channels_per_unit=8192)
    pad = drift_padding(hdr, cfg)
    units = split("x.fil", hdr, cfg, chan_range=(16384, 32768))
    assert units[0].chan_start == 16384 - pad and units[-1].chan_stop == 32768 + pad
    assert (units[0].core_start, units[-1].core_stop) == (16384, 32768)


def test_far_out_of_range_snr_is_capped_without_a_classifier():
    """B8 review: the heuristic fallback trusted a SNR-10,000 carrier just as blindly."""
    rows = _hit_frame("x")
    rows["snr"] = 10_000.0
    cands = score_candidates([{"hits": rows.to_dict("records")}], SearchConfig(use_ai=False))
    assert cands.iloc[0]["interest"] <= 50


def test_midres_hits_are_drift_unresolved_not_stationary():
    """B19: on BL mid-res products drift cannot move a tone one channel; that isn't 'zero'."""
    from ai_seti.share import checklist, passes_gate
    from ai_seti.sources import drift_resolvable

    midres = FilterbankHeader(fch1=8001.46, foff=-2.86102294921875e-03, nchans=1 << 20,
                              tsamp=1.0737418239999998, nsamples=272)
    assert not drift_resolvable(midres, SearchConfig())
    assert drift_resolvable(_header(), SearchConfig())

    rows = pd.concat([_hit_frame("x")] * 2, ignore_index=True)
    rows["zero_drift"] = 1.0
    rows["drift_unresolved"] = [True, False]
    rows["channel"] = [5, 900]
    cands = score_candidates([{"hits": rows.to_dict("records")}], SearchConfig(use_ai=False))
    unresolved = cands[cands["drift_unresolved"]].iloc[0]
    stationary = cands[~cands["drift_unresolved"]].iloc[0]
    assert unresolved["interest"] > stationary["interest"], "no zero-drift penalty when unmeasurable"
    rec = {"checklist": checklist(unresolved, "not_run", False), "assessment": {"cadence": "not_run"}}
    ok, why = passes_gate(rec)
    assert not ok and "not measurable" in why


def test_training_set_fills_every_class():
    """B20: the noise class used to stop at ~22% of the requested examples, silently."""
    from ai_seti.ai.model import build_training_set

    _, y = build_training_set(n_per_class=15, seed=3)
    assert np.bincount(y, minlength=len(LABELS)).tolist() == [15] * len(LABELS)


def test_training_shortfall_is_reported(monkeypatch, caplog):
    import ai_seti.ai.model as model

    real = model._training_example
    monkeypatch.setattr(model, "_training_example",
                        lambda kind, rng, **k: None if kind == "noise" else real(kind, rng, **k))
    with caplog.at_level("WARNING"):
        _, y = model.build_training_set(n_per_class=2, seed=0)
    assert (y == LABELS.index("noise")).sum() == 0
    assert "noise has 0 of 2" in caplog.text
