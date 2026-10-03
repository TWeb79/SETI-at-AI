"""Tests for the cross-run summary report.

The summary is an additive tool: it reads the folders `report.write_outputs` produced. The
central guarantee is that running it changes nothing about how any individual run reports
itself, so that guarantee is tested directly rather than assumed.

Author: Inventions4All - github:TWeb79
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import pytest

from ai_seti.summary import best_overall, collect, discover_runs, warnings_for, write_summary


def _make_run(root: Path, name: str, *, hits=100, best_interest=40.0,
              ai_note="", drift_searched=4.0, drift_wanted=4.0, errors=0,
              best_class="technosignature_like", mirror=False, multi=False,
              with_candidates=True) -> Path:
    """Write one run folder shaped exactly like a real `write_outputs` result."""
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    stats = {"work_units": 4, "errors": errors, "channels": 1_000_000, "hits": hits,
             "spikes": 3, "pulses": 1, "cpu_seconds": 120.0, "wall_seconds": 30.0,
             "channels_per_second": 33333.3, "workers": 2,
             "drift_searched_hz_s": drift_searched}
    if ai_note:
        stats["ai_note"] = ai_note
    (folder / "metadata.json").write_text(json.dumps({
        "title": f"AI-SETI search of {name}",
        "processed_utc": "2026-10-03T12:00:00+00:00",
        "versions": {"ai_seti": "0.3.0", "numpy": "2.0"},
        "config": {"max_drift_rate_hz_s": drift_wanted, "snr_threshold": 10.0},
        "stats": stats,
        "observation": {"target": name.upper(), "telescope": "GBT"},
        "cadence": {"status": "untestable"},
    }), encoding="utf-8")
    if with_candidates:
        pd.DataFrame([
            # A filler hit that must never outrank the flagged top candidate below.
            {"frequency_mhz": 1420.5, "drift_rate_hz_s": 0.02, "snr": 30.0,
             "interest": 1.0, "ai_class": "rfi_zero_drift",
             "mirror_image": False, "multi_target": False},
            {"frequency_mhz": 1419.0, "drift_rate_hz_s": -0.5, "snr": 60.0,
             "interest": best_interest, "ai_class": best_class,
             "mirror_image": mirror, "multi_target": multi},
        ]).to_csv(folder / "candidates.csv", index=False)
    (folder / "report.html").write_text(f"<html><body>run {name}</body></html>", encoding="utf-8")
    return folder


def _payload(html: str) -> dict:
    return json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))


# --------------------------------------------------------------------- discovery

def test_discover_runs_finds_nested_folders_and_ignores_loose_files(tmp_path):
    _make_run(tmp_path / "crunch", "obsA")
    _make_run(tmp_path / "analysis", "obsB")
    (tmp_path / "notarun").mkdir()
    (tmp_path / "notarun" / "candidates.csv").write_text("x", encoding="utf-8")
    found = {p.name for p in discover_runs(tmp_path)}
    assert found == {"obsA", "obsB"}


def test_discover_runs_tolerates_a_missing_tree(tmp_path):
    assert discover_runs(tmp_path / "nope") == []


# --------------------------------------------------------------------- collection

def test_collect_sums_totals_and_keeps_one_row_per_run(tmp_path):
    _make_run(tmp_path, "a", hits=10, best_interest=50.0)
    _make_run(tmp_path, "b", hits=30, best_interest=80.0)
    df = collect(tmp_path)
    runs = df.iloc[:-1]
    assert len(runs) == 2
    totals = df.iloc[-1]
    assert totals["hits"] == 40                      # 10 + 30
    assert totals["cpu_seconds"] == 240.0            # 120 + 120
    assert totals["cpu_hours"] == pytest.approx(240 / 3600, abs=1e-3)
    assert totals["best_interest"] == 80.0


def test_collect_exposes_real_numbers_not_strings(tmp_path):
    """A column of objects breaks sorting and the totals arithmetic downstream."""
    _make_run(tmp_path, "a")
    _make_run(tmp_path, "b", with_candidates=False)   # leaves holes in the numbers
    df = collect(tmp_path)
    assert pd.api.types.is_numeric_dtype(df["best_interest"]), \
        "best_interest must stay numeric even when one run has no candidates"
    assert pd.api.types.is_numeric_dtype(df["hits"])
    # A run with no candidates.csv must still be listed, just without a best hit.
    empty = df[df["run"] == "b"].iloc[0]
    assert pd.isna(empty["best_interest"])


def test_collect_pins_column_order_and_never_drops_a_column(tmp_path):
    """A first row missing a key must not remove that column from every other row."""
    _make_run(tmp_path, "a")
    _make_run(tmp_path, "b", with_candidates=False)
    df = collect(tmp_path)
    assert "best_ai_class" in df.columns
    assert df[df["run"] == "a"].iloc[0]["best_ai_class"] == "technosignature_like"


# --------------------------------------------------------------------- warnings

def test_warning_flags_a_top_candidate_that_is_an_instrument_artefact(tmp_path):
    _make_run(tmp_path, "clean", best_interest=90.0)
    _make_run(tmp_path, "mirrored", best_interest=10.0, mirror=True)
    _make_run(tmp_path, "shared", best_interest=11.0, multi=True)
    warns = " ".join(warnings_for(collect(tmp_path)))
    assert "2 run(s)" in warns and "mirror image" in warns


def test_warning_flags_a_run_that_searched_less_drift_than_it_asked_for(tmp_path):
    _make_run(tmp_path, "degraded", drift_searched=0.05, drift_wanted=4.0)
    df = collect(tmp_path)
    assert bool(df.iloc[0]["drift_degraded"]) is True
    assert any("narrower drift range" in w for w in warnings_for(df))


def test_warning_flags_an_inactive_ai_layer(tmp_path):
    _make_run(tmp_path, "noai", ai_note="hit classifier unavailable")
    warns = " ".join(warnings_for(collect(tmp_path)))
    assert "no working AI layer" in warns


def test_warning_flags_failed_work_units(tmp_path):
    _make_run(tmp_path, "bad", errors=3)
    assert any("failed work units" in w for w in warnings_for(collect(tmp_path)))


def test_a_healthy_tree_raises_no_warnings(tmp_path):
    _make_run(tmp_path, "a")
    _make_run(tmp_path, "b")
    assert warnings_for(collect(tmp_path)) == []


# --------------------------------------------------------------------- selection

def test_best_overall_picks_the_highest_interest_run_not_the_totals_row(tmp_path):
    _make_run(tmp_path, "weak", best_interest=5.0)
    _make_run(tmp_path, "strong", best_interest=91.0)
    best = best_overall(collect(tmp_path))
    assert best["best_interest"] == 91.0
    assert best["run"] == "strong"


def test_best_overall_is_empty_when_no_run_has_candidates(tmp_path):
    _make_run(tmp_path, "a", with_candidates=False)
    assert best_overall(collect(tmp_path)) == {}


# --------------------------------------------------------------------- output

def test_write_summary_emits_csv_and_html_with_working_relative_links(tmp_path):
    run = _make_run(tmp_path / "crunch", "obsA")
    html_path, _df = write_summary(tmp_path / "crunch", tmp_path / "crunch")
    assert html_path.name == "summary.html"
    assert (tmp_path / "crunch" / "summary.csv").is_file()

    payload = _payload(html_path.read_text(encoding="utf-8"))
    linked = [r for r in payload["rows"] if r["report"]]
    assert len(linked) == 1
    resolved = (tmp_path / "crunch" / linked[0]["report"]).resolve()
    assert resolved == (run / "report.html").resolve(), "the link must reach the run's report"
    assert payload["run_count"] == 1


def test_write_summary_works_when_the_output_is_outside_the_reports_tree(tmp_path):
    """A summary written elsewhere must still link back to each run."""
    _make_run(tmp_path / "reports", "obsA")
    html_path, _ = write_summary(tmp_path / "reports", tmp_path / "elsewhere")
    payload = _payload(html_path.read_text(encoding="utf-8"))
    linked = [r["report"] for r in payload["rows"] if r["report"]]
    assert len(linked) == 1
    assert (tmp_path / "elsewhere" / linked[0]).resolve().is_file()


def test_totals_are_numeric_in_the_html_not_nan_strings(tmp_path):
    """The totals strip formats these; a float coerced to text would render as 0.00."""
    _make_run(tmp_path, "a")
    _make_run(tmp_path, "b")
    payload = _payload(write_summary(tmp_path, tmp_path)[0].read_text(encoding="utf-8"))
    totals = dict(payload["totals"])
    assert totals["CPU hours"] == "0.07"        # 240 CPU seconds, not 240/3600 again
    assert totals["wall hours"] == "0.02"
    assert totals["hits"] == "200"
    assert totals["channels"] == "2,000,000"


def test_summary_does_not_modify_any_existing_run_report(tmp_path):
    """The whole point: a run's own report must survive a summary pass byte for byte."""
    run = _make_run(tmp_path / "crunch", "obsA")
    names = sorted(p.name for p in run.iterdir())
    before = {n: hashlib.sha256((run / n).read_bytes()).hexdigest() for n in names}

    write_summary(tmp_path / "crunch", tmp_path / "crunch")

    after_names = sorted(p.name for p in run.iterdir())
    assert after_names == names, "a summary pass must not add or remove files in a run folder"
    assert {n: hashlib.sha256((run / n).read_bytes()).hexdigest() for n in names} == before


def test_summary_handles_a_corrupt_metadata_file_without_crashing(tmp_path):
    bad = tmp_path / "broken"
    bad.mkdir()
    (bad / "metadata.json").write_text("{not json", encoding="utf-8")
    _make_run(tmp_path, "good")
    df = collect(tmp_path)
    assert len(df) == 3                       # the broken run is still listed
    good = df[df["run"] == "good"].iloc[0]
    assert good["best_interest"] == 40.0
