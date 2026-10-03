"""Combine every finished run under a reports/ tree into one overview.

This is additive: it reads the folders `report.write_outputs` already produced and never
changes how a single run is reported. Each run's own `report.html` stays the authority for
its observation; the summary is the index across all of them, plus the cross-run arithmetic
no single report can show (totals, best-ever candidate, and which runs are degraded).

A folder counts as a run when it contains `metadata.json`; `candidates.csv` is optional, so a
run that failed part way through is still listed rather than silently missing.

Author: Inventions4All - github:TWeb79
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

RUN_COLUMNS = [
    "run", "target", "telescope", "processed_utc", "version",
    "work_units", "errors", "channels", "hits", "spikes", "pulses",
    "cpu_seconds", "wall_seconds", "channels_per_second", "workers",
    "drift_searched_hz_s", "max_drift_rate_hz_s", "drift_degraded", "cadence_status",
    "best_frequency_mhz", "best_drift_hz_s", "best_snr", "best_interest", "best_ai_class",
    "best_mirror_image", "best_multi_target", "et_count", "ai_note",
]

# Runs that failed part way through leave these absent, so the frame must be coerced to real
# numeric/boolean dtypes or sorting and aggregation break on object columns downstream.
NUMERIC_COLUMNS = [
    "work_units", "errors", "channels", "hits", "spikes", "pulses", "cpu_seconds",
    "wall_seconds", "channels_per_second", "workers", "drift_searched_hz_s",
    "max_drift_rate_hz_s", "best_frequency_mhz", "best_drift_hz_s", "best_snr",
    "best_interest", "et_count",
]
BOOL_COLUMNS = ["drift_degraded", "best_mirror_image", "best_multi_target"]

TOTAL_KEYS = ["work_units", "errors", "channels", "hits", "spikes", "pulses",
              "cpu_seconds", "wall_seconds"]


def discover_runs(root: Path) -> list[Path]:
    """Folders under `root` holding a completed run's `metadata.json`, in a stable order."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted({p.parent for p in root.rglob("metadata.json")})


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _top_candidate(folder: Path) -> dict:
    """The highest-interest hit in a run, plus how many hits the AI called ET-like.

    Returns an empty dict when the run has no `candidates.csv`, which keeps a failed run in
    the overview as a row with no candidate rather than dropping it.
    """
    path = folder / "candidates.csv"
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return {}
    if df.empty or "interest" not in df.columns:
        return {}
    top = df.loc[df["interest"].idxmax()]
    classes = df.get("ai_class", pd.Series(dtype=str)).astype(str)
    return {
        "best_frequency_mhz": top.get("frequency_mhz"),
        "best_drift_hz_s": top.get("drift_rate_hz_s"),
        "best_snr": top.get("snr"),
        "best_interest": top.get("interest"),
        "best_ai_class": top.get("ai_class"),
        "best_mirror_image": bool(top.get("mirror_image", False)),
        "best_multi_target": bool(top.get("multi_target", False)),
        "et_count": int((classes == "technosignature_like").sum()),
    }


def summarise_run(folder: Path, root: Path) -> dict:
    """One overview row for one run folder. Tolerates older metadata.json layouts."""
    folder = Path(folder)
    meta = _read_json(folder / "metadata.json")
    stats = meta.get("stats") or {}
    cfg = meta.get("config") or {}
    obs = meta.get("observation") or {}
    cadence = meta.get("cadence") or {}

    searched = stats.get("drift_searched_hz_s")
    wanted = cfg.get("max_drift_rate_hz_s")
    # A run that searched less drift than its own config asked for saw a narrower sky than
    # its results suggest; that must not be invisible in an overview.
    degraded = bool(searched is not None and wanted is not None and searched < wanted * 0.999)

    row = {
        "run": str(folder.relative_to(root)),
        "target": obs.get("target") or "",
        "telescope": obs.get("telescope") or "",
        "processed_utc": meta.get("processed_utc", ""),
        "version": (meta.get("versions") or {}).get("ai_seti", ""),
        "drift_degraded": degraded,
        "cadence_status": cadence.get("status", ""),
        "ai_note": stats.get("ai_note", ""),
        "drift_searched_hz_s": searched,
        "max_drift_rate_hz_s": wanted,
    }
    for key in ("work_units", "errors", "channels", "hits", "spikes", "pulses",
                "cpu_seconds", "wall_seconds", "channels_per_second", "workers"):
        row[key] = stats.get(key)
    row.update(_top_candidate(folder))
    return row


def collect(root: Path) -> pd.DataFrame:
    """Every run under `root` as a DataFrame, with a totals row appended."""
    root = Path(root)
    rows = [summarise_run(f, root) for f in discover_runs(root)]
    # Build from the rows, then pin the column order. Passing `columns=` to the constructor
    # would drop any key the first row happens to be missing.
    df = pd.DataFrame(rows)
    for col in RUN_COLUMNS:
        if col not in df.columns:
            df[col] = None
    df = df[list(RUN_COLUMNS)]
    totals: dict[str, Any] = {"run": f"TOTAL ({len(rows)} runs)", "target": "", "telescope": "",
                          "processed_utc": "", "version": "", "ai_note": "",
                          "cadence_status": "",
                          "drift_degraded": bool(df["drift_degraded"].any())}
    for key in TOTAL_KEYS:
        vals = pd.to_numeric(df[key], errors="coerce")
        totals[key] = round(float(vals.sum(skipna=True)), 2)
    # Hourly figures read better than accumulated seconds once a tree of runs is aggregated.
    totals["cpu_hours"] = round(float(totals["cpu_seconds"]) / 3600, 3)
    totals["wall_hours"] = round(float(totals["wall_seconds"]) / 3600, 3)
    totals["best_interest"] = pd.to_numeric(df["best_interest"], errors="coerce").max()
    out = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    # Coerce after the concat: the totals row has no per-run values, so coercing first
    # would let it turn numeric and boolean columns back into object dtype.
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in BOOL_COLUMNS:
        out[col] = out[col].fillna(False).astype(bool)
    return out


def best_overall(df: pd.DataFrame) -> dict:
    """The single best candidate across every run, ignoring the totals row."""
    runs = df[df["run"] != df["run"].iloc[-1]] if not df.empty else df
    runs = runs[runs["best_interest"].notna()] if not runs.empty else runs
    if runs.empty:
        return {}
    return runs.loc[pd.to_numeric(runs["best_interest"], errors="coerce").idxmax()].to_dict()


def warnings_for(df: pd.DataFrame) -> list[str]:
    """Cross-run problems worth stating before any table is read."""
    runs = df.iloc[:-1] if len(df) else df          # drop the totals row
    out: list[str] = []
    errors = runs[runs["errors"].fillna(0) > 0]
    if not errors.empty:
        out.append(f"{len(errors)} run(s) finished with failed work units.")
    inactive = runs[runs["ai_note"].fillna("").astype(str) != ""]
    if not inactive.empty:
        out.append(f"{len(inactive)} run(s) had no working AI layer; interest fell back to a "
                   "heuristic in those reports.")
    degraded = runs[runs["drift_degraded"].fillna(False).astype(bool)]
    if not degraded.empty:
        out.append(f"{len(degraded)} run(s) searched a narrower drift range than their config "
                   "asked for, so they cannot rule out faster signals.")
    flagged = runs[runs["best_mirror_image"].fillna(False).astype(bool)
                   | runs["best_multi_target"].fillna(False).astype(bool)]
    if not flagged.empty:
        out.append(f"{len(flagged)} run(s) top out with a candidate flagged as a mirror image "
                   "or seen in another target: an instrument artefact, not a discovery.")
    return out


def write_summary(root: Path, out_dir: Path) -> tuple[Path, pd.DataFrame]:
    """Write `summary.csv` and `summary.html` for every run under `root`."""
    from .summary_html import render_summary_html

    root, out_dir = Path(root), Path(out_dir)
    df = collect(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    html_path = out_dir / "summary.html"
    html_path.write_text(
        render_summary_html(df, best_overall(df), warnings_for(df), root, out_dir),
        encoding="utf-8")
    return html_path, df


def summary_metadata(df: pd.DataFrame, root: Path) -> dict:
    """Provenance for the summary itself, written alongside it."""
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "root": str(root),
        "runs": max(len(df) - 1, 0),
        "warnings": warnings_for(df),
    }
