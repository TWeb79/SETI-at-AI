"""Tests for the HDF5 reader, config loading and the benchmark harness.

The `.h5` path matters: Breakthrough Listen serves much of its archive as
bitshuffle-compressed HDF5, and `BreakthroughListenSource` downloads those rather than
streaming them, so the reader is the only thing standing between that data and the
pipeline.
"""
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from ai_seti.config import SearchConfig
from ai_seti.core import robust_candidates
from ai_seti.io.filterbank import read_header, read_window

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_h5(path, data, *, three_d=False, attrs=None):
    """Write `data` as an HDF5 `data` dataset. For `three_d`, `data` is already [time, if, freq]."""
    base = {"fch1": 1420.0, "foff": -2.7939677238464355e-06, "tsamp": 18.253611008,
            "tstart": 59000.5, "nbits": 32, "source_name": "HIP12345"}
    base.update(attrs or {})
    assert data.ndim == (3 if three_d else 2), "caller must pass data already shaped"
    with h5py.File(path, "w") as h5:
        ds = h5.create_dataset("data", data=data)
        for k, v in base.items():
            ds.attrs[k] = v


def test_h5_header_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(size=(8, 4096)).astype(np.float32)
    p = tmp_path / "obs.h5"
    _write_h5(p, data)

    hdr = read_header(p)
    assert hdr.nchans == 4096 and hdr.nsamples == 8
    assert hdr.fch1 == 1420.0 and hdr.source_name == "HIP12345"
    assert hdr.nbits == 32 and hdr.tstart == 59000.5
    assert hdr.nifs == 1
    assert hdr.frequencies(0, 2)[0] == pytest.approx(1420.0)
    json.dumps(hdr.to_dict())  # must not raise: header metadata is serialised verbatim


def test_h5_window_read_matches_source(tmp_path):
    rng = np.random.default_rng(2)
    data = rng.normal(size=(8, 4096)).astype(np.float32)
    p = tmp_path / "obs.h5"
    _write_h5(p, data)
    hdr = read_header(p)
    window = read_window(p, 1000, 1300, hdr)
    assert window.shape == (8, 300)
    assert np.allclose(window, data[:, 1000:1300])
    assert np.allclose(read_window(p, None, None, hdr), data), "default window is the whole band"


def test_h5_three_dimensional_with_if_axis(tmp_path):
    """BL products are often [time, if, freq]; the reader must pick the requested IF."""
    rng = np.random.default_rng(3)
    n_if = 2
    data = rng.normal(size=(6, n_if, 2048)).astype(np.float32)
    p = tmp_path / "pol.h5"
    _write_h5(p, data, three_d=True)

    hdr = read_header(p)
    assert hdr.nifs == n_if and hdr.nchans == 2048 and hdr.nsamples == 6
    for i in range(n_if):
        assert np.allclose(read_window(p, 10, 20, hdr, if_index=i), data[:, i, 10:20])


def test_h5_data_reaches_the_pipeline(tmp_path):
    """End to end: an HDF5 file splits, loads and yields the injected narrowband tone."""
    from ai_seti.ai.simulate import Injection, inject, noise_waterfall
    from ai_seti.pipeline import run_units, score_candidates
    from ai_seti.sources import split

    rng = np.random.default_rng(4)
    data = noise_waterfall(16, 16384, rng)
    inject(data, Injection("technosignature_like", 8000.0, 2.0, 6.0, 1.0), rng)
    assert data.shape == (16, 16384)
    p = tmp_path / "obs.h5"
    _write_h5(p, data)

    hdr = read_header(p)
    cfg = SearchConfig(channels_per_unit=8192, workers=1, snr_threshold=8.0)
    results = list(run_units(split(str(p), hdr, cfg), cfg, workers=1))
    cands = score_candidates(results, cfg)
    assert len(cands), "the injected tone should survive the HDF5 path"
    assert (cands["channel"] - 8000).abs().min() <= 4


# --------------------------------------------------------------------------- config

def test_config_load_defaults_and_overrides(tmp_path):
    cfg = SearchConfig.load(None)
    assert cfg.snr_threshold == SearchConfig().snr_threshold
    assert cfg.known_rfi_bands_mhz, "default RFI bands must survive with no config file"
    assert SearchConfig.load(tmp_path / "nope.json").snr_threshold == cfg.snr_threshold

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"snr_threshold": 14.5, "workers": 3,
                                "known_rfi_bands_mhz": [[100.0, 200.0]],
                                "not_a_real_key": "ignored"}))
    cfg = SearchConfig.load(path)
    assert cfg.snr_threshold == 14.5 and cfg.workers == 3
    assert cfg.known_rfi_bands_mhz == [(100.0, 200.0)], "band lists load as tuples of floats"
    assert not hasattr(cfg, "not_a_real_key")


def test_config_roundtrips_through_dict():
    cfg = SearchConfig(snr_threshold=11.0, reporter_handle="someone")
    data = cfg.to_dict()
    assert data["snr_threshold"] == 11.0 and data["reporter_handle"] == "someone"
    assert json.dumps(data), "config must stay JSON-serialisable for metadata.json"
    assert SearchConfig(**data) == cfg


def test_config_shipped_default_is_loadable():
    """configs/default.json is what the CLI loads by default; it must stay in sync."""
    shipped = PROJECT_ROOT / "configs" / "default.json"
    if not shipped.is_file():
        pytest.skip("configs/default.json not present in this checkout")
    cfg = SearchConfig.load(shipped)
    defaults = SearchConfig()
    for key in ("channels_per_unit", "snr_threshold", "max_drift_rate_hz_s", "use_ai",
                "share_require_cadence"):   # B8: a pinned False would publish uncadenced finds
        assert getattr(cfg, key) == getattr(defaults, key), \
            f"configs/default.json sets {key} differently from the dataclass default"


# ------------------------------------------------------------------------ benchmark

def test_benchmark_calibrates_and_reports_a_table():
    """A tiny run must produce thresholds and one row per (snr, drift) cell."""
    from ai_seti.benchmark import run
    res = run(trials=2, snrs=(3.0,), drifts=(0.0, 2.0), n_noise=4, seed=1)
    table = res["table"]
    assert isinstance(table, pd.DataFrame) and len(table) == 2
    assert set(table.columns) == {"snr_per_sample", "drift_ch_per_step",
                                  "v0.1_recovery", "v0.2_recovery"}
    assert res["thresholds"]["v0.1_robust_z"] > 0
    assert res["thresholds"]["v0.2_snr"] > 0
    assert res["seconds_per_chunk"]["v0.2"] >= 0
    # A strong, slow tone is the easy case for both detectors; neither should be blind.
    easy = table[(table.snr_per_sample == 3.0) & (table.drift_ch_per_step == 0.0)]
    assert float(easy.iloc[0]["v0.2_recovery"]) >= 0.5


def test_benchmark_detects_strong_drift_better_than_legacy():
    """The headline claim of the project: at equal false-alarm rate, v0.2 beats v0.1 on drift."""
    from ai_seti.benchmark import run
    res = run(trials=3, snrs=(3.0,), drifts=(8.0,), n_noise=6, seed=2)
    row = res["table"].iloc[0]
    assert float(row["v0.2_recovery"]) > float(row["v0.1_recovery"]), \
        "the de-Doppler detector should beat the time-median detector on a drifting tone"


# --------------------------------------------------------------------------- legacy

def test_robust_candidates_handles_three_dimensional_input():
    rng = np.random.default_rng(5)
    cube = rng.normal(size=(64, 512))
    cube[:, 100] += 8.0
    out = robust_candidates(cube, threshold=4.0)
    assert len(out) == 1 and out.iloc[0]["channel_index"] == 100
    assert out.iloc[0]["candidate_type"] == "persistent_narrowband_outlier"
    assert out.iloc[0]["status"] == "unverified"


def test_web_ui_port_and_bind_address():
    """B23/B24: project 61 -> 8061 (RULES_ports.md), and localhost only."""
    import tomllib

    cfg = tomllib.loads((PROJECT_ROOT / ".streamlit" / "config.toml").read_text())
    assert cfg["server"]["port"] == 8061
    assert cfg["server"]["address"] == "127.0.0.1"
    assert "DASHBOARD_PORT=8061" in (PROJECT_ROOT / "ports.env").read_text()


def test_ci_cannot_be_cancelled_by_rerunning_an_old_run_and_linters_are_pinned():
    """B45: a re-run of an old run cancelled CI for the latest commit on main."""
    import tomllib

    ci = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "group: ci-${{ github.event.pull_request.number || github.sha }}" in ci
    dev = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())[
        "project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("ruff==") for d in dev) and any(d.startswith("mypy==") for d in dev)
