"""Render the cross-run summary page.

Visually consistent with `report.HTML_TEMPLATE` so the overview reads as part of the same
product: same night palette, same display font, same gold/teal interest bar. Self-contained
(no CDN beyond the same font links) because a summary is most useful opened from disk.

Author: Inventions4All - github:TWeb79
"""
from __future__ import annotations

import contextlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from . import __version__

SUMMARY_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AI-SETI run overview</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@500;700&family=Atkinson+Hyperlegible:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{
  --night:#0c1748; --steel:#22306b; --line:#3a4a8f; --paper:#f2ead8; --dim:#b9b3a3;
  --gold:#ffcc33; --hot:#ff5a3c; --teal:#36c2b4;
  --display:"Chakra Petch","Segoe UI",system-ui,sans-serif;
  --body:"Atkinson Hyperlegible",system-ui,-apple-system,sans-serif;
  box-sizing:border-box;
  padding-top:env(safe-area-inset-top,0px); padding-bottom:env(safe-area-inset-bottom,0px);
}
*,*::before,*::after{box-sizing:inherit}
body{margin:0;background:var(--night);color:var(--paper);font:16px/1.55 var(--body);
  font-variant-numeric:tabular-nums}
main{max-width:1280px;margin:0 auto;padding:28px 20px 64px}
h1,h2{font-family:var(--display);font-weight:700;margin:0}
h1{font-size:clamp(26px,4.4vw,44px);line-height:1.05}
h2{font-size:21px;margin-bottom:12px}
p{margin:0 0 10px;max-width:74ch}
.badge{position:absolute;top:10px;right:12px;font-family:var(--display);font-size:12px;
  letter-spacing:.04em;color:var(--gold);border:1px solid var(--gold);border-radius:4px;
  padding:2px 8px;background:rgba(12,23,72,.6)}
.hero{position:relative;display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);
  gap:28px;align-items:end;border-bottom:1px solid var(--steel);padding-bottom:24px}
.lede{color:var(--dim);margin:10px 0 18px}
.score{font-family:var(--display);font-size:64px;line-height:1;color:var(--gold)}
.score small{font-size:18px;color:var(--dim);margin-left:6px}
.readout{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;font-size:17px}
.readout dt{color:var(--dim)} .readout dd{margin:0;font-weight:700}
.stats{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:28px 0 8px}
.stats b{display:block;font-family:var(--display);font-size:24px}
.stats span{color:var(--dim);font-size:14px}
.warn{margin:10px 0;padding:8px 12px;border-left:3px solid var(--gold);background:rgba(255,204,51,.10);font-size:15px}
.warns{margin:20px 0 8px}
.warns h2{margin-bottom:8px}
.table{overflow-x:auto;margin-top:36px}
table{border-collapse:collapse;width:100%;min-width:1080px;font-size:15px}
th{text-align:left;color:var(--dim);font-weight:400;padding:8px 10px;border-bottom:1px solid var(--line);
  position:sticky;top:0;background:var(--night)}
th.sortable{cursor:pointer;user-select:none} th.sortable:hover{color:var(--gold)}
td{padding:7px 10px;border-bottom:1px solid #18245c}
tbody tr:hover{background:#17266a}
tbody tr.total{font-weight:700;color:var(--gold)}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:13px;background:var(--steel)}
.tag.et{background:var(--gold);color:var(--night);font-weight:700}
.flag{color:var(--hot);font-size:13px}
.ok{color:var(--teal)}
a{color:var(--teal)}
.note{margin-top:36px;color:var(--dim);font-size:14px}
@media (max-width:820px){.hero{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
@media (prefers-color-scheme:light){:root:not([data-theme="dark"]){--night:#eef0f8;--paper:#141a3a;--dim:#4b5275;--steel:#c9cfe6;--line:#aab3d6}
  :root:not([data-theme="dark"]) th{background:var(--night)}
  :root:not([data-theme="dark"]) tbody tr:hover{background:#dde3f6}
  :root:not([data-theme="dark"]) td{border-bottom-color:#d5daee}
  :root:not([data-theme="dark"]) .score{color:#b07800}
  :root:not([data-theme="dark"]) .badge{border-color:#b07800;color:#8a5c00;background:rgba(255,255,255,.75)}
  :root:not([data-theme="dark"]) .warn{background:rgba(176,120,0,.10)}}
</style></head>
<body><main>
<section class="hero">
  <span class="badge" id="badge"></span>
  <div>
    <h1 id="headline">Best signal across every run</h1>
    <p class="lede" id="lede"></p>
    <div class="score" id="score"></div>
    <dl class="readout" id="readout"></dl>
  </div>
</section>

<section class="stats" id="stats"></section>
<section class="warns" id="warns" hidden><h2>Before you read the table</h2><div id="warnlist"></div></section>

<section class="table"><h2>Every run</h2>
<table><thead><tr>
  <th class="sortable" data-k="run">Run</th>
  <th class="sortable" data-k="target">Target</th>
  <th class="sortable" data-k="hits">Hits</th>
  <th class="sortable" data-k="best_interest">Best interest</th>
  <th class="sortable" data-k="best_frequency_mhz">Best freq (MHz)</th>
  <th class="sortable" data-k="best_drift_hz_s">Drift (Hz/s)</th>
  <th class="sortable" data-k="best_snr">SNR</th>
  <th>AI class</th>
  <th>Flags</th>
  <th class="sortable" data-k="wall_seconds">Wall s</th>
  <th class="sortable" data-k="channels_per_second">ch/s</th>
</tr></thead><tbody id="rows"></tbody></table></section>
<p class="note" id="foot"></p>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const num = v => (v===null||v===undefined||v===""||Number.isNaN(Number(v))) ? null : Number(v);
const fmt = (v,d=2) => { const n=num(v); return n===null ? "–" : n.toFixed(d); };
const cell = v => String(v).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");

// Worst flags first so a misleading candidate is impossible to miss by scrolling.
function flagsOf(r){
  const out=[];
  if(r.best_mirror_image) out.push("mirror image");
  if(r.best_multi_target) out.push("multi-target");
  if(r.drift_degraded) out.push("drift degraded");
  if(r.errors>0) out.push((r.errors||0)+" failed unit(s)");
  if(r.ai_note) out.push("AI inactive");
  return out;
}
function rowHtml(r, isTotal){
  const f = flagsOf(r);
  const link = isTotal || !r.report ? cell(r.run)
    : `<a href="${cell(r.report)}" title="${cell(r.run)}">${cell(r.run)}</a>`;
  const cls = (r.best_ai_class||"").replaceAll("_"," ");
  const clsCell = cls ? `<span class="tag ${cls==="technosignature like"?"et":""}">${cell(cls)}</span>` : "–";
  return `<tr class="${isTotal?"total":""}"><td>${link}</td><td>${cell(r.target||"–")}</td>
    <td>${fmt(r.hits,0)}</td><td>${fmt(r.best_interest,1)}</td>
    <td>${fmt(r.best_frequency_mhz,4)}</td><td>${fmt(r.best_drift_hz_s,3)}</td>
    <td>${fmt(r.best_snr,1)}</td><td>${clsCell}</td>
    <td>${f.length?`<span class="flag">${cell(f.join(", "))}</span>`:'<span class="ok">–</span>'}</td>
    <td>${fmt(r.wall_seconds,1)}</td><td>${num(r.channels_per_second)===null?"–":Math.round(r.channels_per_second).toLocaleString()}</td></tr>`;
}
function render(order){
  const runs = D.rows.filter(r=>!r.is_total);
  const idx = runs.map((_,i)=>i);
  if(order){ const d = order.dir==="asc"?1:-1;
    idx.sort((a,b)=>{ const x=runs[a][order.k], y=runs[b][order.k];
      const nx=num(x), ny=num(y);
      if(nx!==null&&ny!==null) return (nx-ny)*d;
      return String(x??"").localeCompare(String(y??""))*d; }); }
  $("rows").innerHTML = idx.map(i=>rowHtml(runs[i],false)).join("")
    + (D.total ? rowHtml(D.total,true) : "");
}
const b = D.best;
if(b){
  $("headline").textContent = "Best signal across every run";
  $("lede").textContent = `${b.target||"unknown target"}${b.telescope?" ("+b.telescope+")":""} — from ${b.run}. `
    + "Every candidate stays unverified until it survives an ON/OFF cadence check and a re-observation.";
  $("score").innerHTML = `${fmt(b.best_interest,1)}<small>/ 100 interest</small>`;
  $("readout").innerHTML = [["Run",b.run],["Frequency",fmt(b.best_frequency_mhz,6)+" MHz"],
    ["Drift rate",fmt(b.best_drift_hz_s,3)+" Hz/s"],["De-Doppler SNR",fmt(b.best_snr,1)],
    ["AI says",(b.best_ai_class||"–").replaceAll("_"," ")]]
    .map(([k,v])=>`<dt>${k}</dt><dd>${cell(v)}</dd>`).join("");
} else {
  $("score").innerHTML = `–<small>/ 100 interest</small>`;
  $("lede").textContent = "No run has produced a candidate yet.";
}
$("stats").innerHTML = D.totals.map(([k,v])=>`<div><b>${cell(v)}</b><span>${cell(k)}</span></div>`).join("");
if(D.warnings.length){ $("warns").hidden=false;
  $("warnlist").innerHTML = D.warnings.map(w=>`<p class="warn">${cell(w)}</p>`).join(""); }
render(null);
document.querySelectorAll("th.sortable").forEach(th=>th.onclick=()=>{
  const k=th.dataset.k; order = (order&&order.k===k) ? {k,dir:order.dir==="asc"?"desc":"asc"} : {k,dir:"desc"};
  render(order);
});
$("foot").textContent = `${D.run_count} run(s) under ${D.root}, summarised ${D.built} UTC with AI-SETI ${D.version}. `
  + "This is an index over existing reports; each run's own report.html remains the detail. "
  + "Candidates are statistical detections, not claims of extraterrestrial origin.";
$("badge").textContent = `v${D.version||"?"} · built ${D.built.slice(0,16).replace("T"," ")} UTC`;
</script></body></html>
"""

# (label, totals-row key, formatting); `run_count` is derived rather than stored.
TOTAL_FIELDS = [("runs", "run_count", "int"), ("channels", "channels", "int"),
                ("hits", "hits", "int"), ("spikes", "spikes", "int"),
                ("pulses", "pulses", "int"), ("CPU hours", "cpu_hours", "float"),
                ("wall hours", "wall_hours", "float")]


def _clean(value):
    """Make a pandas/numpy cell JSON-safe.

    Missing values (`None`, `NaN`, `NaT`, `pd.NA`) all collapse to `null` so the page can
    print an em dash. Numbers must stay numbers: the totals strip formats them, and turning
    a float into a string here would break that arithmetic.
    """
    if value is None:
        return None
    # numpy scalars carry .item(); use it before any type test so they are seen as real ints
    # and floats rather than falling through to str().
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes)):
        with contextlib.suppress(AttributeError, ValueError):
            value = item()
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return None if (math.isnan(value) or math.isinf(value)) else value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, str):
        return value
    try:
        if value != value:      # pandas NA: fails the self-comparison
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def render_summary_html(df, best: dict, warnings: list[str], root: Path, out_dir: Path) -> str:
    """Build the standalone overview page for every run under `root`."""
    totals_row = df.iloc[-1] if len(df) else None
    rows = []
    out_dir = Path(out_dir)
    for _, r in df.iterrows():
        is_total = bool(totals_row is not None and r["run"] == totals_row["run"])
        row = {k: _clean(v) for k, v in r.items()}
        row["is_total"] = is_total
        # Link relative to the summary page so the file works opened straight from disk.
        # Resolve both ends first: macOS /tmp is a symlink to /private/tmp, so a lexically
        # correct "../.." would land in /private/Users/... and 404 in the browser.
        # os.path.relpath also copes with an outdir outside the reports tree, which
        # Path.relative_to cannot (it raises unless outdir is an ancestor of the run).
        run_folder = (Path(root) / str(r["run"])) if not is_total else None
        rel = None
        if run_folder is not None:
            try:
                rel = os.path.relpath((run_folder / "report.html").resolve(),
                                      Path(out_dir).resolve()).replace(os.sep, "/")
            except (ValueError, OSError):
                rel = None
        row["report"] = rel
        rows.append(row)

    totals = []
    if totals_row is not None:
        # `collect()` already expresses cpu_hours/wall_hours in hours; do not divide again.
        for label, key, kind in TOTAL_FIELDS:
            value = totals_row.get(key)
            if key == "run_count":
                totals.append((label, str(max(len(df) - 1, 0))))
                continue
            v = _clean(value)
            if v is None:
                totals.append((label, "–"))
            elif kind == "int":
                totals.append((label, f"{int(v):,}"))
            else:
                totals.append((label, f"{float(v):.2f}"))

    best_clean = {k: _clean(v) for k, v in best.items()} if best else {}
    payload = {
        "rows": rows, "total": rows[-1] if rows else None,
        "best": best_clean, "totals": totals, "warnings": warnings,
        "root": str(root), "run_count": max(len(df) - 1, 0), "version": __version__,
        "built": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    html = SUMMARY_TEMPLATE.replace("__DATA__", json.dumps(payload, default=str).replace("</", "<\\/"))
    return html
