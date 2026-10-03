"""Share signal findings back — the "report results" half of SETI@home.

SETI@home clients uploaded every result to Berkeley. That server no longer accepts
results, so AI-SETI packages each finding as a self-describing, reproducible record
(`ai-seti-finding/1`) and sends it to one or more destinations:

* bundle  – local folder + zip (finding.json, finding.png, README) to attach anywhere
* github  – an issue in a community repository (deduplicated by finding ID)
* webhook – JSON POST to any endpoint; Slack / Discord message formats built in

Guard rails, because a false "we found aliens" spreads faster than a correction:
* every record says `unverified_candidate`, with a checklist of what was and was not checked
* only candidates passing a minimum gate leave the machine (configurable)
* synthetic/demo data is never sent to remote destinations unless explicitly allowed
* local file paths and host details are stripped; a reporter handle is opt-in
* the same finding (same data, ~same frequency and drift) gets the same ID everywhere,
  so communities can merge independent reports instead of counting them twice
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

SCHEMA = "ai-seti-finding/1"
CONFIG_KEYS = ("snr_threshold", "max_drift_rate_hz_s", "channels_per_unit", "bandpass_block")


# --------------------------------------------------------------------------- records

def finding_id(obs: dict, freq_mhz: float, drift_hz_s: float) -> str:
    """Stable ID: same data + same signal (10 Hz / 0.05 Hz/s bins) -> same ID for everyone."""
    data_key = (obs.get("md5sum") or obs.get("data_url")
                or f"{obs.get('target')}|{obs.get('mjd')}")
    key = f"{data_key}|{round(freq_mhz * 1e5)}|{round(drift_hz_s / 0.05)}"
    return "AIS-" + hashlib.sha256(key.encode()).hexdigest()[:12].upper()


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _cadence_status(row: pd.Series, events: pd.DataFrame | None) -> str:
    if events is None:
        return "not_run"
    if events.empty:
        return "failed"
    near = events[(events["frequency_mhz"] - row["frequency_mhz"]).abs() < 50e-6]
    if near.empty:
        return "failed"
    # Defence in depth: an events.csv produced before the ON/OFF minimum was enforced, or
    # hand-written, must not be able to grant a cadence pass on one ON scan.
    if "n_on_scans" in near and int(near["n_on_scans"].max()) < 2:
        return "failed"
    if "off_scans_checked" in near and int(near["off_scans_checked"].max()) < 1:
        return "failed"
    return "passed"


def checklist(row: pd.Series, cadence: str, bands_flag: bool) -> list[dict]:
    return [
        {"check": "Drifts (non-zero Doppler drift, as an off-Earth source would)",
         "result": None if bool(row.get("drift_unresolved", False))
         else not bool(row.get("zero_drift", 0))},
        {"check": "Narrowband (<= 4 channels)", "result": bool(row.get("bandwidth_ch", 99) <= 4)},
        {"check": "Present in >= 75% of time samples",
         "result": bool(row.get("on_fraction", 0) >= 0.75)},
        {"check": "Outside known satellite / RFI bands", "result": not bands_flag},
        {"check": "Not a mirror image about the coarse-channel centre",
         "result": not bool(row.get("mirror_image", False))},
        {"check": "Not also seen in a different target",
         "result": not bool(row.get("multi_target", False))},
        {"check": "Passed ON/OFF cadence filter (absent when pointing away)",
         "result": None if cadence == "not_run" else cadence == "passed"},
        {"check": "Cross-checked with an independent pipeline (e.g. turboSETI)", "result": None},
        {"check": "Re-observed at a later date", "result": None},
    ]


def build_findings(report_dir: Path, top: int = 3, min_interest: float = 0.0,
                   handle: str | None = None, cadence_events: Path | None = None) -> list[dict]:
    report_dir = Path(report_dir)
    cands = pd.read_csv(report_dir / "candidates.csv")
    meta = json.loads((report_dir / "metadata.json").read_text(encoding="utf-8"))
    snips = np.load(report_dir / "snippets.npy") if (report_dir / "snippets.npy").exists() else None
    events = pd.read_csv(cadence_events) if cadence_events and Path(cadence_events).exists() else None
    obs_in = meta.get("observation", {}) or {}
    header = meta.get("header", {}) or {}
    synthetic = obs_in.get("source") == "synthetic"
    public_url = obs_in.get("url") if str(obs_in.get("url", "")).startswith("http") else None
    if public_url is None and str(meta.get("location", "")).startswith("http"):
        public_url = meta["location"]
    obs = {
        "target": obs_in.get("target") or header.get("source_name"),
        "telescope": obs_in.get("telescope"),
        "mjd": _clean(obs_in.get("mjd") or header.get("tstart")),
        "ra": obs_in.get("ra"), "decl": obs_in.get("decl"),
        "data_url": public_url, "md5sum": obs_in.get("md5sum"),
        "source": obs_in.get("source"), "synthetic": synthetic,
        "tsamp_s": _clean(header.get("tsamp")),
        "channel_width_hz": _clean(abs(header.get("foff", 0.0)) * 1e6) if header else None,
    }
    out = []
    for i, row in cands.head(top).iterrows():
        if row.get("interest", 0) < min_interest:
            continue
        cad = _cadence_status(row, events)
        bands = bool(row.get("known_rfi_band", False))
        rec = {
            "schema": SCHEMA,
            "finding_id": finding_id(obs, row["frequency_mhz"], row["drift_rate_hz_s"]),
            "status": "unverified_candidate",
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "reporter": {"handle": handle, "software": f"ai-seti {__version__}"},
            "signal": {k: _clean(row.get(k)) for k in
                       ("frequency_mhz", "drift_rate_hz_s", "snr", "bandwidth_ch",
                        "on_fraction", "channel")},
            "assessment": {"interest": _clean(row.get("interest")),
                           "ai_class": row.get("ai_class"),
                           "p_technosignature_like": _clean(row.get("p_technosignature_like")),
                           "anomaly": _clean(row.get("anomaly")),
                           "known_rfi_band": bands, "cadence": cad,
                           "rank_in_run": int(i) + 1},
            "observation": obs,
            "processing": {"config": {k: meta.get("config", {}).get(k) for k in CONFIG_KEYS},
                           "versions": meta.get("versions", {})},
            "checklist": checklist(row, cad, bands),
            "snippet_16x16": (np.round(snips[i], 3).tolist()
                              if snips is not None and i < len(snips) else None),
            "disclaimer": ("Statistical candidate from automated analysis. Most such candidates "
                           "are radio interference. Not a claim of extraterrestrial origin."),
        }
        body = json.dumps(rec, sort_keys=True, separators=(",", ":"), default=str)
        rec["payload_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        out.append(rec)
    return out


def passes_gate(rec: dict, require_cadence: bool = False) -> tuple[bool, str]:
    checks = {c["check"].split(" (")[0]: c["result"] for c in rec["checklist"]}
    if checks.get("Drifts") is None:
        return False, "drift not measurable at this resolution (cannot rule out Earth)"
    if not checks.get("Drifts"):
        return False, "zero drift (almost always terrestrial)"
    if checks.get("Outside known satellite / RFI bands") is False:
        return False, "inside a known RFI band"
    if checks.get("Not a mirror image about the coarse-channel centre") is False:
        return False, "mirror image about the coarse-channel centre (instrument artefact)"
    if checks.get("Not also seen in a different target") is False:
        return False, "also seen in a different target (interference)"
    if require_cadence and rec["assessment"]["cadence"] != "passed":
        return False, f"cadence {rec['assessment']['cadence']}"
    return True, "ok"


# --------------------------------------------------------------------------- formats

def title_line(rec: dict) -> str:
    s, o = rec["signal"], rec["observation"]
    return (f"[candidate] {s['frequency_mhz']:.6f} MHz, {s['drift_rate_hz_s']:+.3f} Hz/s, "
            f"SNR {s['snr']:.1f}, {o.get('target') or 'unknown target'} ({rec['finding_id']})")


def markdown(rec: dict) -> str:
    s, a, o = rec["signal"], rec["assessment"], rec["observation"]
    mark = {True: "yes", False: "no", None: "not done"}
    lines = [
        f"**{rec['finding_id']}**, status: `{rec['status']}`",
        "",
        "| | |", "|---|---|",
        f"| Frequency | {s['frequency_mhz']:.6f} MHz |",
        f"| Drift rate | {s['drift_rate_hz_s']:+.4f} Hz/s |",
        f"| SNR | {s['snr']:.1f} |",
        f"| AI class | {a['ai_class']} (p ET-like "
        f"{'n/a' if a['p_technosignature_like'] is None else round(a['p_technosignature_like'], 2)}) |",
        f"| Interest | {a['interest']} / 100 |",
        f"| Target / telescope | {o.get('target')} / {o.get('telescope')} |",
        f"| MJD | {o.get('mjd')} |",
        f"| Data | {o.get('data_url') or 'local file (not public)'} |",
        "", "**Checklist**", "",
    ]
    lines += [f"- {c['check']}: {mark[c['result']]}" for c in rec["checklist"]]
    lines += ["", f"_{rec['disclaimer']}_", "",
              "<details><summary>Full record</summary>", "", "```json",
              json.dumps(rec, indent=1, default=str), "```", "</details>"]
    return "\n".join(lines)


def plot_finding(rec: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2), constrained_layout=True)
    if rec.get("snippet_16x16"):
        ax.imshow(np.asarray(rec["snippet_16x16"]), aspect="auto", cmap="viridis",
                  origin="upper", interpolation="nearest")
    ax.set_xlabel("Frequency (de-drifted, binned)")
    ax.set_ylabel("Time")
    s = rec["signal"]
    ax.set_title(f"{rec['finding_id']}\n{s['frequency_mhz']:.6f} MHz  "
                 f"{s['drift_rate_hz_s']:+.3f} Hz/s  SNR {s['snr']:.1f}", fontsize=9)
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- sinks

@dataclass
class ShareResult:
    finding_id: str
    destination: str
    ok: bool
    detail: str


def to_bundle(recs: list[dict], out_dir: Path) -> list[ShareResult]:
    out_dir = Path(out_dir)
    results = []
    for rec in recs:
        d = out_dir / rec["finding_id"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "finding.json").write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        (d / "finding.md").write_text(f"# {title_line(rec)}\n\n{markdown(rec)}\n", encoding="utf-8")
        plot_finding(rec, d / "finding.png")
        zpath = out_dir / f"{rec['finding_id']}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("finding.json", "finding.md", "finding.png"):
                z.write(d / f, f"{rec['finding_id']}/{f}")
        results.append(ShareResult(rec["finding_id"], "bundle", True, str(zpath)))
    return results


def _post_json(url: str, payload: dict, headers: dict | None = None, timeout: float = 30.0):
    req = urllib.request.Request(url, data=json.dumps(payload, default=str).encode(),
                                 method="POST",
                                 headers={"Content-Type": "application/json",
                                          "User-Agent": f"ai-seti/{__version__}",
                                          **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
        return resp.status, (json.loads(body) if body.strip().startswith(b"{") else {})


def _get_json(url: str, headers: dict | None = None, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": f"ai-seti/{__version__}",
                                               **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def to_github(recs: list[dict], repo: str, token: str | None = None,
              api: str = "https://api.github.com") -> list[ShareResult]:
    """Open one issue per finding in `repo` (owner/name); skip if the ID is already there."""
    token = token or os.environ.get("AI_SETI_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        return [ShareResult(r["finding_id"], "github", False,
                            "set AI_SETI_GITHUB_TOKEN (issues:write)") for r in recs]
    hdr = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
           "X-GitHub-Api-Version": "2022-11-28"}
    results = []
    for rec in recs:
        try:
            q = urllib.request.quote(f"repo:{repo} \"{rec['finding_id']}\" in:title,body")
            found = _get_json(f"{api}/search/issues?q={q}", hdr)
            if found.get("total_count", 0):
                url = found["items"][0].get("html_url", "")
                results.append(ShareResult(rec["finding_id"], "github", True,
                                           f"already reported: {url}"))
                continue
            _, resp = _post_json(f"{api}/repos/{repo}/issues",
                                 {"title": title_line(rec), "body": markdown(rec),
                                  "labels": ["candidate", "unverified"]}, hdr)
            results.append(ShareResult(rec["finding_id"], "github", True,
                                       resp.get("html_url", "created")))
        except Exception as exc:
            results.append(ShareResult(rec["finding_id"], "github", False, str(exc)))
    return results


def to_webhook(recs: list[dict], url: str, fmt: str = "json") -> list[ShareResult]:
    results = []
    for rec in recs:
        if fmt == "slack":
            payload = {"text": f"*{title_line(rec)}*\n{markdown(rec).split('<details>')[0]}"}
        elif fmt == "discord":
            payload = {"content": (f"**{title_line(rec)}**\n"
                                   + markdown(rec).split("<details>")[0])[:1990]}
        else:
            payload = rec
        try:
            status, _ = _post_json(url, payload)
            results.append(ShareResult(rec["finding_id"], f"webhook:{fmt}", 200 <= status < 300,
                                       f"HTTP {status}"))
        except Exception as exc:
            results.append(ShareResult(rec["finding_id"], f"webhook:{fmt}", False, str(exc)))
    return results


def share(recs: list[dict], destinations: list[str], *, bundle_dir: Path, ledger=None,
          repo: str | None = None, webhook: str | None = None, webhook_format: str = "json",
          allow_synthetic: bool = False, require_cadence: bool | None = None,
          github_api: str = "https://api.github.com") -> tuple[list[ShareResult], list[tuple]]:
    """Gate, dedupe against the ledger, then send. Returns (results, skipped).

    `require_cadence=None` requires a cadence pass for remote destinations only: an
    uncadenced candidate may go into the local bundle but is never published (B8).
    """
    sendable, skipped = [], []
    for rec in recs:
        ok, why = passes_gate(rec, bool(require_cadence))
        if not ok:
            skipped.append((rec["finding_id"], why))
            continue
        sendable.append(rec)
    results: list[ShareResult] = []
    already = (ledger.shared if ledger is not None else {})
    for dest in destinations:
        # Key on finding_id, not on record equality: records carry a 16x16 snippet, so
        # `rec in batch` would deep-compare nested lists and scale as O(n^2).
        batch = [r for r in sendable if dest not in already.get(r["finding_id"], [])]
        batch_ids = {r["finding_id"] for r in batch}
        skipped += [(r["finding_id"], f"already shared to {dest}")
                    for r in sendable if r["finding_id"] not in batch_ids]
        if dest != "bundle" and require_cadence is None:
            uncad = [r for r in batch if r["assessment"]["cadence"] != "passed"]
            skipped += [(r["finding_id"], f"no cadence pass, not sent to {dest}") for r in uncad]
            batch = [r for r in batch if r["assessment"]["cadence"] == "passed"]
        if dest != "bundle":
            synth = [r for r in batch if r["observation"].get("synthetic")]
            if synth and not allow_synthetic:
                skipped += [(r["finding_id"], f"synthetic data not sent to {dest}") for r in synth]
                batch = [r for r in batch if not r["observation"].get("synthetic")]
        if not batch:
            continue
        if dest == "bundle":
            results += to_bundle(batch, bundle_dir)
        elif dest == "github":
            if not repo:
                raise ValueError("--repo owner/name is required for github")
            results += to_github(batch, repo, api=github_api)
        elif dest == "webhook":
            if not webhook:
                raise ValueError("--webhook URL is required for webhook")
            results += to_webhook(batch, webhook, webhook_format)
        else:
            raise ValueError(f"Unknown destination {dest}")
    if ledger is not None:
        for res in results:
            if res.ok:
                ledger.shared.setdefault(res.finding_id, [])
                if res.destination.split(":")[0] not in ledger.shared[res.finding_id]:
                    ledger.shared[res.finding_id].append(res.destination.split(":")[0])
    return results, skipped
