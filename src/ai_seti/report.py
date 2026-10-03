"""Write everything a run produces: candidates.csv, metadata.json, PNGs and report.html."""
from __future__ import annotations

import contextlib
import json
import platform
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__

REPORT_TOP = 25


def _versions() -> dict:
    out = {"python": platform.python_version(), "ai_seti": __version__}
    for pkg in ("numpy", "scipy", "scikit-learn", "pandas", "h5py"):
        with contextlib.suppress(importlib_metadata.PackageNotFoundError):
            out[pkg] = importlib_metadata.version(pkg)
    return out


def band_profile(results: list[dict]) -> list[dict]:
    segs = [r for r in results if r.get("snr_profile")]
    segs.sort(key=lambda r: r["f_lo"])
    return [{"f_lo": r["f_lo"], "f_hi": r["f_hi"], "snr": r["snr_profile"]} for r in segs]


def plot_overview(results: list[dict], candidates: pd.DataFrame, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    prof = band_profile(results)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1.3]})
    for seg in prof:
        x = np.linspace(seg["f_lo"], seg["f_hi"], len(seg["snr"]))
        axes[0].plot(x, seg["snr"], lw=0.7, color="#2b3a78")
    if not candidates.empty:
        top = candidates.head(REPORT_TOP)
        axes[0].scatter(top["frequency_mhz"], top["snr"], c=top["interest"], cmap="plasma",
                        s=28, zorder=3, label="top candidates")
        axes[0].legend(loc="upper right")
    axes[0].set_ylabel("Best de-Doppler SNR")
    axes[0].set_title(title)
    if not candidates.empty:
        sc = axes[1].scatter(candidates["frequency_mhz"], candidates["drift_rate_hz_s"],
                             s=np.clip(candidates["snr"], 5, 80), c=candidates["interest"],
                             cmap="plasma", alpha=0.85)
        fig.colorbar(sc, ax=axes[1], label="Interest score")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Drift rate (Hz/s)")
    fig.savefig(out, dpi=130)
    plt.close(fig)


def write_outputs(results: list[dict], candidates: pd.DataFrame, outdir: Path, cfg,
                  title: str, extra_meta: dict | None = None, wall_seconds: float = 0.0) -> dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    flat = candidates.drop(columns=[c for c in candidates.columns if c.startswith("_")],
                           errors="ignore")
    flat.to_csv(outdir / "candidates.csv", index=False)
    if "_snippet" in candidates:   # de-drifted 16x16 snippets, row-aligned with candidates.csv
        snips = [np.asarray(s, dtype=np.float32) for s in candidates["_snippet"].head(200)]
        if snips:
            np.save(outdir / "snippets.npy", np.stack(snips))
    from .dsp.detectors import isolated_spikes
    raw_spikes = [s for r in results for s in r.get("spikes", [])]
    spikes = isolated_spikes(raw_spikes, candidates)
    pulses = [p for r in results for p in r.get("pulses", [])]
    pd.DataFrame(spikes).to_csv(outdir / "spikes.csv", index=False)
    pd.DataFrame(pulses).to_csv(outdir / "pulses.csv", index=False)

    channels = int(sum(r.get("n_channels", 0) for r in results))
    cpu = float(sum(sum(r.get("timings", {}).values()) for r in results))
    stage: dict[str, float] = {}
    for r in results:
        for k, v in r.get("timings", {}).items():
            stage[k] = stage.get(k, 0.0) + v
    stats = {
        "work_units": len(results), "errors": sum(1 for r in results if r.get("error")),
        "channels": channels, "hits": len(candidates), "spikes": len(spikes),
        "spikes_on_tones": len(raw_spikes) - len(spikes), "pulses": len(pulses),
        "pulses_undispersed": sum(1 for p in pulses if p.get("likely_rfi")),
        "cpu_seconds": round(cpu, 2),
        "wall_seconds": round(wall_seconds, 2),
        "channels_per_second": round(channels / wall_seconds, 1) if wall_seconds else None,
        "stage_seconds": {k: round(v, 2) for k, v in stage.items()},
        "workers": len({r.get("pid") for r in results if r.get("pid")}),
        # Smallest drift range any unit covered: below max_drift_rate_hz_s means a degraded run.
        "drift_searched_hz_s": min((r["drift_searched_hz_s"] for r in results
                                    if "drift_searched_hz_s" in r), default=None),
    }
    # Record a classifier failure in the report itself, so a run that quietly lost its
    # AI layer cannot be mistaken later for a run in which the AI layer found nothing.
    ai_note = candidates.attrs.get("ai_note") if isinstance(candidates, pd.DataFrame) else None
    if not ai_note and not candidates.empty:
        classes = set(candidates.get("ai_class", pd.Series(dtype=str)).astype(str))
        if classes & {"model_unavailable", "ai_disabled"}:
            ai_note = ("hit classifier produced no labels for this run "
                       f"({'model_unavailable' if 'model_unavailable' in classes else 'ai_disabled'}); "
                       "interest fell back to the heuristic")
    if ai_note:
        stats["ai_note"] = ai_note
    meta = {
        "title": title, "processed_utc": datetime.now(UTC).isoformat(),
        "versions": _versions(), "config": cfg.to_dict(), "stats": stats,
        "detectors": ["taylor-tree de-Doppler", "single-sample spikes",
                      "sub-band dedispersion + boxcar pulses",
                      "gradient-boosted hit classifier", "isolation-forest anomaly"],
        "note": "Candidates are unverified statistical detections, not technosignature claims.",
        **(extra_meta or {}),
    }
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    plot_overview(results, candidates, outdir / "band_overview.png", title)
    write_html(outdir / "report.html", candidates, results, meta)
    return stats


def write_html(path: Path, candidates: pd.DataFrame, results: list[dict], meta: dict) -> None:
    keep = ["frequency_mhz", "drift_rate_hz_s", "snr", "interest", "ai_class", "anomaly",
            "bandwidth_ch", "on_fraction", "zero_drift", "stationary", "drift_unresolved", "known_rfi_band", "mirror_image", "cadence",
            "multi_target", "unit_id",
            "p_technosignature_like", "_snippet"]
    top = candidates.head(REPORT_TOP)
    cands = []
    for _, r in top.iterrows():
        cands.append({k: (r[k].item() if hasattr(r[k], "item") else r[k])
                      for k in keep if k in r.index})
    first_meta: dict = {}
    for r in results:
        candidate = r.get("meta")
        if candidate:
            first_meta = candidate
            break
    payload = {"candidates": cands, "band": band_profile(results), "stats": meta["stats"],
               "title": meta["title"], "processed": meta["processed_utc"],
               "built": datetime.now(UTC).isoformat(timespec="seconds"),
               "target": first_meta.get("target") or "unknown target",
               "telescope": first_meta.get("telescope") or "",
               "version": meta["versions"].get("ai_seti")}
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload, default=str).replace("</", "<\\/"))
    path.write_text(html, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>AI-SETI run report</title>
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
html{scroll-padding-top:env(safe-area-inset-top,0px)}
body{margin:0;background:var(--night);color:var(--paper);font:16px/1.55 var(--body);
  font-variant-numeric:tabular-nums}
main{max-width:1180px;margin:0 auto;padding:28px 20px 64px}
h1,h2{font-family:var(--display);font-weight:700;letter-spacing:.01em;margin:0}
h1{font-size:clamp(28px,5vw,52px);line-height:1.05}
h2{font-size:22px;margin-bottom:12px}
p{margin:0 0 10px;max-width:72ch}
.hero{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr);gap:28px;align-items:end}
.sky{position:relative;background:radial-gradient(ellipse at 50% 120%,#1c2c7a 0%,var(--night) 70%);
  border:1px solid var(--line);border-radius:6px;aspect-ratio:16/10;overflow:hidden}
.sky canvas{width:100%;height:100%;display:block}
.sky .axis{position:absolute;font-size:12px;color:var(--dim)}
.sky .axis.x{bottom:8px;left:14px}.sky .axis.y{bottom:8px;right:14px}
.lede{color:var(--dim);margin:10px 0 18px}
.readout{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;font-size:17px}
.readout dt{color:var(--dim)} .readout dd{margin:0;font-weight:700}
.score{font-family:var(--display);font-size:64px;line-height:1;color:var(--gold)}
.score small{font-size:18px;color:var(--dim);margin-left:6px}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-top:40px}
.panel{border-top:2px solid var(--steel);padding-top:14px}
.stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}
.stats b{display:block;font-family:var(--display);font-size:26px}
.stats span{color:var(--dim);font-size:14px}
.band{margin-top:40px}
.band canvas{width:100%;height:150px;display:block;background:#0a1440;border:1px solid var(--line);border-radius:6px}
.table{overflow-x:auto;margin-top:40px}
table{border-collapse:collapse;width:100%;min-width:760px;font-size:15px}
th{text-align:left;color:var(--dim);font-weight:400;padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:8px 10px;border-bottom:1px solid #18245c}
tbody tr{cursor:pointer}
tbody tr:hover,tbody tr[aria-selected=true]{background:#17266a}
tbody tr:focus-visible{outline:2px solid var(--gold);outline-offset:-2px}
.tag{display:inline-block;padding:1px 8px;border-radius:10px;font-size:13px;background:var(--steel)}
.tag.et{background:var(--gold);color:var(--night);font-weight:700}
.iv{display:flex;align-items:center;gap:10px}
.bar{display:block;height:6px;background:var(--steel);border-radius:3px;width:90px}
.bar i{display:block;height:100%;border-radius:3px;background:linear-gradient(90deg,var(--teal),var(--gold),var(--hot))}
.note{margin-top:36px;color:var(--dim);font-size:14px}
.badge{position:absolute;top:10px;right:12px;font-family:var(--display);font-size:12px;
  letter-spacing:.04em;color:var(--gold);border:1px solid var(--gold);border-radius:4px;
  padding:2px 8px;background:rgba(12,23,72,.6);white-space:nowrap}
.hero{position:relative}
.warn{margin-top:12px;padding:8px 12px;border-left:3px solid var(--hot);
  background:rgba(255,90,60,.12);font-size:14px}
@media (max-width:820px){.hero,.cols{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
@media (prefers-color-scheme:light){:root:not([data-theme="dark"]){--night:#eef0f8;--paper:#141a3a;--dim:#4b5275;--steel:#c9cfe6;--line:#aab3d6}
  :root:not([data-theme="dark"]) .sky{background:radial-gradient(ellipse at 50% 120%,#c6cff0 0%,#eef0f8 70%)}
  :root:not([data-theme="dark"]) .band canvas{background:#e3e7f5}
  :root:not([data-theme="dark"]) tbody tr:hover,:root:not([data-theme="dark"]) tbody tr[aria-selected=true]{background:#dde3f6}
  :root:not([data-theme="dark"]) td{border-bottom-color:#d5daee}
  :root:not([data-theme="dark"]) .score{color:#b07800}
  :root:not([data-theme="dark"]) .badge{border-color:#b07800;color:#8a5c00;background:rgba(255,255,255,.75)}
  :root:not([data-theme="dark"]) .warn{background:rgba(200,40,20,.08)}}
</style></head>
<body><main>
<section class="hero">
  <span class="badge" id="badge"></span>
  <div>
    <div class="sky"><canvas id="sky" aria-label="3D power plot of the selected signal"></canvas>
      <span class="axis x">frequency, de-drifted along the track</span><span class="axis y">time</span></div>
  </div>
  <div>
    <h1 id="headline">Best signal of this run</h1>
    <p class="lede" id="lede"></p>
    <div class="score" id="score">–<small>/ 100 interest</small></div>
    <dl class="readout" id="readout"></dl>
  </div>
</section>
<p class="warn" id="warn" hidden></p>

<section class="cols">
  <div class="panel"><h2>What was searched</h2><div class="stats" id="stats"></div></div>
  <div class="panel"><h2>Why it ranks here</h2><p id="why"></p>
    <p style="color:var(--dim)">Every candidate is unverified until it survives an ON/OFF cadence check and a re-observation.</p></div>
</section>

<section class="band"><h2>Band overview</h2><canvas id="band" aria-label="Best SNR across the searched band"></canvas></section>

<section class="table"><h2>Top candidates</h2>
<table><thead><tr><th>#</th><th>Frequency (MHz)</th><th>Drift (Hz/s)</th><th>SNR</th><th>AI class</th><th>Anomaly</th><th>Interest</th></tr></thead>
<tbody id="rows"></tbody></table></section>
<p class="note" id="foot"></p>
</main>
<script>
const D = __DATA__;
const $ = id => document.getElementById(id);
const fmt = (v,d=2) => (v===null||v===undefined||Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();

function heightColor(h){ // SETI@home-style: blue -> teal -> gold -> red
  const stops=[[0,[43,58,140]],[.35,[54,194,180]],[.7,[255,204,51]],[1,[255,90,60]]];
  for(let i=1;i<stops.length;i++){ if(h<=stops[i][0]){const [a,ca]=stops[i-1],[b,cb]=stops[i];const t=(h-a)/(b-a);
    return `rgb(${ca.map((c,j)=>Math.round(c+(cb[j]-c)*t)).join(",")})`;}}
  return "rgb(255,90,60)";
}
let anim=null;
function drawSky(snip){
  const cv=$("sky"), r=cv.getBoundingClientRect(), dpr=devicePixelRatio||1;
  cv.width=r.width*dpr; cv.height=r.height*dpr; const g=cv.getContext("2d"); g.scale(dpr,dpr);
  const W=r.width,H=r.height, rows=snip?snip.length:16, cols=snip?snip[0].length:16;
  const sx=0.55, sy=0.42;                       // isometric depth offsets per time row
  const cell=Math.min(W*0.86/(cols+rows*sx), H*0.38/(rows*sy));
  const ox=(W-(cols+rows*sx)*cell)/2, oy=H*0.9, maxH=H*0.5;
  const start=performance.now(); cancelAnimationFrame(anim);
  function frame(now){
    const p=reduce?1:Math.min(1,(now-start)/1100), ease=1-Math.pow(1-p,3);
    g.clearRect(0,0,W,H);
    g.strokeStyle="rgba(120,140,220,.25)"; g.lineWidth=1;   // floor grid
    for(let t=0;t<=rows;t++){g.beginPath();g.moveTo(ox+t*cell*sx,oy-t*cell*sy);g.lineTo(ox+cols*cell+t*cell*sx,oy-t*cell*sy);g.stroke();}
    for(let f=0;f<=cols;f++){g.beginPath();g.moveTo(ox+f*cell,oy);g.lineTo(ox+f*cell+rows*cell*sx,oy-rows*cell*sy);g.stroke();}
    for(let t=rows-1;t>=0;t--){ for(let f=0;f<cols;f++){
      const v=snip?Math.max(0,snip[t][f]):0, h=Math.max(1.5,v*maxH*ease);
      const x=ox+f*cell+t*cell*sx, y=oy-t*cell*sy;
      g.fillStyle=heightColor(v); g.fillRect(x+cell*0.08,y-h,cell*0.8,h);
      g.fillStyle="rgba(0,0,0,.28)"; g.fillRect(x+cell*0.68,y-h,cell*0.2,h);   // side shade
      g.fillStyle="rgba(255,255,255,.22)"; g.fillRect(x+cell*0.08,y-h,cell*0.8,Math.min(2,h));
    }}
    if(p<1) anim=requestAnimationFrame(frame);
  }
  anim=requestAnimationFrame(frame);
}
function select(i){
  const c=D.candidates[i]; if(!c) return;
  document.querySelectorAll("#rows tr").forEach((tr,j)=>tr.setAttribute("aria-selected", j===i));
  $("headline").textContent = i===0 ? "Best signal of this run" : `Candidate ${i+1}`;
  $("lede").textContent = `${D.target}${D.telescope? " ("+D.telescope+")":""} — a ${c.drift_unresolved? "narrowband tone (drift not measurable at this resolution)" : ((c.stationary ?? c.zero_drift)? "non-drifting narrowband tone":"drifting narrowband tone")}`;
  $("score").innerHTML = `${fmt(c.interest,0)}<small>/ 100 interest</small>`;
  $("readout").innerHTML = [["Frequency",fmt(c.frequency_mhz,6)+" MHz"],["Drift rate",fmt(c.drift_rate_hz_s,3)+" Hz/s"],
    ["De-Doppler SNR",fmt(c.snr,1)],["AI says",(c.ai_class||"–").replaceAll("_"," ")],
    ["P(technosignature-like)",fmt(c.p_technosignature_like,2)],["Unusualness",fmt(c.anomaly,2)]]
    .map(([k,v])=>`<dt>${k}</dt><dd>${v}</dd>`).join("");
  const reasons=[];
  if(c.drift_unresolved) reasons.push("this data's channels are too wide to see any drift, so it cannot tell a moving source from one on Earth");
  else if(!(c.stationary ?? c.zero_drift)) reasons.push("it drifts, as a transmitter on a rotating, orbiting planet would");
  else reasons.push("it does not drift, which usually means a transmitter on Earth (score reduced)");
  if(c.bandwidth_ch<=3) reasons.push("it is only "+fmt(c.bandwidth_ch,0)+(c.bandwidth_ch>=1.5?" channels":" channel")+" wide, and nature rarely makes tones this narrow");
  if(c.on_fraction>=.75) reasons.push("it is present in "+Math.round(c.on_fraction*100)+"% of time samples");
  if(c.known_rfi_band) reasons.push("but it sits in a band crowded with satellites (score halved)");
  if(c.cadence==="failed") reasons.push("but it failed the ON/OFF test: it was there while pointing away, or missing in a pointing at the target (score cut to a fifth)");
  if(c.cadence==="passed") reasons.push("and it passed the ON/OFF test: present in every pointing at the target, absent when pointing away");
  if(c.multi_target) reasons.push("but the same frequency turned up in a different target, so it cannot come from this one (score cut to a fifth)");
  if(c.mirror_image) reasons.push("but it is mirrored across the coarse-channel centre by a partner of equal or greater strength, an instrument artefact (score cut to a fifth)");
  $("why").textContent = "It scores "+fmt(c.interest,0)+" because "+reasons.join("; ")+".";
  drawSky(c._snippet);
}
function drawBand(){
  const cv=$("band"), r=cv.getBoundingClientRect(), dpr=devicePixelRatio||1;
  cv.width=r.width*dpr; cv.height=r.height*dpr; const g=cv.getContext("2d"); g.scale(dpr,dpr);
  if(!D.band.length) return;
  const lo=Math.min(...D.band.map(b=>b.f_lo)), hi=Math.max(...D.band.map(b=>b.f_hi));
  const mx=Math.max(10,...D.band.flatMap(b=>b.snr));
  for(const b of D.band){ b.snr.forEach((s,i)=>{
    const f=b.f_lo+(b.f_hi-b.f_lo)*i/b.snr.length, x=(f-lo)/(hi-lo||1)*r.width, h=Math.max(0,s)/mx*(r.height-14);
    g.fillStyle=heightColor(Math.min(1,Math.max(0,s)/mx)); g.fillRect(x,r.height-h,Math.max(1,r.width/(D.band.length*b.snr.length)),h);
  });}
  g.fillStyle=css("--dim"); g.font="12px system-ui"; g.fillText(fmt(lo,4)+" MHz",6,12); g.textAlign="right"; g.fillText(fmt(hi,4)+" MHz",r.width-6,12);
}
const s=D.stats;
$("stats").innerHTML=[["work units",s.work_units],["channels",Number(s.channels).toLocaleString()],
 ["hits",s.hits],["lone spikes",s.spikes??"–"],["pulses",s.pulses??"–"],["CPU seconds",fmt(s.cpu_seconds,1)],["wall seconds",fmt(s.wall_seconds,1)],
 ["channels / s",s.channels_per_second? Number(s.channels_per_second).toLocaleString():"–"]]
 .map(([k,v])=>`<div><b>${v}</b><span>${k}</span></div>`).join("");
$("rows").innerHTML=D.candidates.map((c,i)=>`<tr tabindex="0"><td>${i+1}</td><td>${fmt(c.frequency_mhz,6)}</td><td>${fmt(c.drift_rate_hz_s,3)}</td>
 <td>${fmt(c.snr,1)}</td><td><span class="tag ${c.ai_class==="technosignature_like"?"et":""}">${(c.ai_class||"–").replaceAll("_"," ")}</span></td>
 <td>${fmt(c.anomaly,2)}</td><td><span class="iv"><span class="bar"><i style="width:${Math.max(2,c.interest)}%"></i></span>${fmt(c.interest,0)}</span></td></tr>`).join("");
document.querySelectorAll("#rows tr").forEach((tr,i)=>{tr.onclick=()=>select(i);tr.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();select(i);}}});
$("foot").textContent=`${D.title}, processed ${String(D.processed).slice(0,16).replace('T',' ')} UTC with AI-SETI ${D.version}. Candidates are statistical detections, not claims of extraterrestrial origin.`;
$("badge").textContent=`v${D.version||"?"} · built ${String(D.built||"").slice(0,16).replace("T"," ")} UTC`;
if(s.ai_note){$("warn").hidden=false; $("warn").textContent="AI layer inactive: "+s.ai_note;}
if(D.candidates.length) select(0); else {$("headline").textContent="No candidates above threshold"; $("lede").textContent="The band was quiet at this sensitivity. Lower --snr or crunch another work unit."; drawSky(null);}
drawBand(); addEventListener("resize",()=>{drawBand(); const sel=[...document.querySelectorAll("#rows tr")].findIndex(t=>t.getAttribute("aria-selected")==="true"); if(sel>=0) drawSky(D.candidates[sel]._snippet);});
</script></body></html>
"""
