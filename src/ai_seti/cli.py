"""ai-seti command line: demo, analyze, crunch (auto-loading client), cadence, train, benchmark."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.table import Table

from .config import SearchConfig

app = typer.Typer(no_args_is_help=True,
                  help="AI-SETI: parallel, AI-assisted radio technosignature search.")
console = Console()

CONFIG_OPT = typer.Option(Path("configs/default.json"), help="JSON config file.")


def _cfg(config: Path, workers: int | None = None, snr: float | None = None) -> SearchConfig:
    cfg = SearchConfig.load(config)
    if workers is not None:
        cfg.workers = workers
    if snr is not None:
        cfg.snr_threshold = snr
    return cfg


def _run_cadence(cands, cadence: list[dict], location, header, cfg: SearchConfig,
                 chan_range, ledger=None) -> tuple:
    """Search the same channels in the other scans of this cadence and test each hit (B7)."""
    import pandas as pd

    from .io.filterbank import read_header
    from .io.remote import remote_header
    from .pipeline import results_version, run_units, score_candidates
    from .rfi import apply_cadence, cadence_filter, cadence_is_testable
    from .sources import split

    version = results_version(cfg)

    me = next((s for s in cadence if s["url"] == str(location)), None)
    on_target = next((s["target"] for s in cadence if s["is_on"]), "?")
    if me is None or not me["is_on"]:
        print(f"[dim]Cadence: this scan is an OFF pointing of the {on_target} cadence, so its "
              "hits can't be cadence-tested.[/dim]")
        return apply_cadence(cands, "untestable"), {"status": "untestable", "on_target": on_target}
    empty = pd.DataFrame(columns=["frequency_mhz", "drift_rate_hz_s", "snr"])
    scans, fetched = [], 0
    for s in cadence:
        name = s["url"].rsplit("/", 1)[-1]
        cached = (ledger.cached_results(s["url"], chan_range, cfg, version)
                  if ledger is not None and s is not me else None)
        if s is me:
            hits = cands
        elif cached is not None:
            hits = score_candidates(cached, cfg)
        else:
            fetched += 1
            print(f"[dim]Cadence: searching {s['target']} ({'ON' if s['is_on'] else 'OFF'} "
                  f"scan, {fetched} of up to {len(cadence) - 1} others)…[/dim]")
            try:
                hdr = remote_header(s["url"]) if s["kind"] == "remote" else read_header(Path(s["url"]))
                if hdr.nchans != header.nchans:
                    raise ValueError("different channelisation")
                units = split(s["url"], hdr, cfg, s["kind"], chan_range=chan_range, meta={})
                unit_results = list(run_units(units, cfg))
                hits = score_candidates(unit_results, cfg)
                if ledger is not None:
                    ledger.store_results(s["url"], chan_range, cfg, version, unit_results)
            except Exception as exc:   # one unreadable scan makes the test incomplete, not a crash
                print(f"[yellow]Cadence: could not search {name}: {exc}[/yellow]")
                continue
        scans.append({"name": name, "mjd": s["mjd"], "is_on": s["is_on"],
                      "hits": hits if not hits.empty else empty})
        print(f"[dim]Cadence: {s['target']} ({'ON ' if s['is_on'] else 'OFF'}) "
              f"{len(hits):,} signal(s){' (cached)' if cached is not None else ''}[/dim]")
    testable, reason = cadence_is_testable(scans)
    if not testable:
        print(f"[yellow]Cadence not testable: {reason}[/yellow]")
        return apply_cadence(cands, "untestable"), {"status": "untestable", "reason": reason}
    events = cadence_filter(scans, reference=me["url"].rsplit("/", 1)[-1])
    cands = apply_cadence(cands, "tested", events)
    n_pass = int((cands["cadence"] == "passed").sum()) if "cadence" in cands else 0
    print(f"Cadence ({on_target}, {sum(s['is_on'] for s in scans)} ON / "
          f"{sum(not s['is_on'] for s in scans)} OFF): {n_pass:,} of {len(cands):,} signal(s) "
          "passed.")
    return cands, {"status": "tested", "on_target": on_target, "passed": n_pass,
                   "scans": [s["name"] for s in scans]}


def crunch_observation(location, kind, header, meta, cfg: SearchConfig, outdir: Path,
                       live: bool = True, ledger=None, chan_range=None, cadence=None):
    """Split -> parallel process (with live dashboard) -> score -> write outputs."""
    from .dashboard import CrunchDashboard
    from .pipeline import SCORING_VERSION, results_version, run_units, score_candidates
    from .report import write_outputs
    from .sources import drift_resolvable, split

    resolvable = drift_resolvable(header, cfg)
    if not resolvable:
        print(f"[yellow]Drift is not measurable here: {abs(header.foff) * 1e6:,.0f} Hz channels, "
              f"{header.nsamples * header.tsamp:,.0f} s scan, {cfg.max_drift_rate_hz_s:g} Hz/s "
              "limit. Hits are labelled drift_unresolved; prefer the high-resolution product."
              "[/yellow]")
    units = split(location, header, cfg, kind, chan_range=chan_range, meta=meta)
    dash = CrunchDashboard(len(units), ledger)
    dash.meta = meta
    results = []
    t0 = time.time()
    version = results_version(cfg)
    cached = (ledger.cached_results(str(location), chan_range, cfg, version)
              if ledger is not None else None)
    if cached is not None:
        # Searched already, as part of a cadence: reuse it instead of downloading again (B41).
        print(f"[dim]Reusing {len(cached)} work unit(s) already searched for a cadence; "
              "nothing to download.[/dim]")
        results = cached
        for res in results:
            dash.update(res)
    elif live and console.is_terminal:
        from rich.live import Live
        with Live(dash.render(), console=console, refresh_per_second=4) as lv:
            for res in run_units(units, cfg):
                results.append(res)
                dash.update(res)
                lv.update(dash.render())
    else:
        # Plain status (B39): what is about to happen, then download and analysis per unit.
        bytes_per_ch = header.nsamples * header.nifs * header.bytes_per_value
        total_mb = sum(u.chan_stop - u.chan_start for u in units) * bytes_per_ch / 2**20
        print(f"[dim]{'downloading' if kind == 'remote' else 'reading'} and analysing "
              f"{len(units)} work unit(s), ~{total_mb:,.0f} MB, "
              f"{min(cfg.workers or os.cpu_count() or 1, len(units))} at a time…[/dim]")
        for res in run_units(units, cfg):
            results.append(res)
            dash.update(res)
            best = max((h["snr"] for h in res.get("hits", [])), default=0)
            tm = res.get("timings", {})
            unit_mb = res.get("n_channels", 0) * bytes_per_ch / 2**20
            load_s = tm.get("load", 0.0)
            phases = (f"  {'downloaded' if kind == 'remote' else 'read'} {unit_mb:,.0f} MB in "
                      f"{load_s:.1f} s "
                      f"({unit_mb / load_s:.1f} MB/s), analysed in "
                      f"{sum(v for k, v in tm.items() if k != 'load'):.1f} s" if load_s else "")
            # Units run `n` at a time, so time left goes by batches, not by single units.
            n = min(cfg.workers or os.cpu_count() or 1, len(units))
            per_batch = (time.time() - t0) / -(-dash.done // n)
            left = -(-(len(units) - dash.done) // n) * per_batch
            a, b = (int(x) for x in res["unit_id"].rsplit(":", 1)[-1].split("-"))
            chans = f"{a:,}–{b:,}"
            print(f"[dim]{dash.done}/{len(units)} ({100 * dash.done // len(units)}%, "
                  f"~{left:.0f} s left)[/dim]  channels {chans}  "
                  f"signals={len(res.get('hits', []))}  strongest SNR={best:,.1f}{phases}"
                  + (f"  [red]{res['error']}[/red]" if res.get("error") else ""))
    wall = time.time() - t0
    cands = score_candidates(results, cfg)
    if ledger is not None and cadence and cached is None:
        # A sibling's cadence step, or this scan's own later crunch, can reuse these (B41).
        ledger.store_results(str(location), chan_range, cfg, version, results)
    target = str(meta.get("target") or header.source_name)
    if ledger is not None and not cands.empty:
        from .rfi import flag_multi_target
        cands = flag_multi_target(cands, ledger.signals, target, cfg.multi_target_tol_khz,
                                  cfg.multi_target_band_khz, cfg.multi_target_band_hits)
    cadence_meta = None
    if cadence and not cands.empty:
        cands, cadence_meta = _run_cadence(cands, cadence, location, header, cfg, chan_range,
                                           ledger)
    title = f"AI-SETI search of {meta.get('target') or Path(str(location)).name}"
    stats = write_outputs(results, cands, outdir, cfg, title,
                          {"location": str(location), "observation": meta,
                           "header": header.to_dict(), "drift_resolvable": resolvable,
                           "cadence": cadence_meta}, wall)
    if ledger is not None:
        for r in results:
            ledger.record_unit(r)
        if not cands.empty:
            ledger.remember_signals(cands["frequency_mhz"], target)
        if not cands.empty and ledger.record_best({**cands.iloc[0].to_dict(),
                                                   "location": str(location)},
                                                  SCORING_VERSION):
            print("[bold #ffcc33]New lifetime best signal![/bold #ffcc33]")
    return cands, stats


def _summary(cands, stats, outdir: Path, n: int = 8):
    t = Table(title=f"Top candidates ({stats['work_units']} units, {stats['channels']:,} channels, "
                    f"{stats['wall_seconds']:.1f} s wall, {stats['workers']} worker(s))")
    for col in ("#", "MHz", "Hz/s", "SNR", "AI class", "anomaly", "interest"):
        t.add_column(col, justify="left" if col == "AI class" else "right")
    for i, r in cands.head(n).iterrows():
        an = r["anomaly"]
        t.add_row(str(i + 1), f"{r['frequency_mhz']:.6f}", f"{r['drift_rate_hz_s']:+.3f}",
                  f"{r['snr']:.1f}", str(r.get("ai_class", "")),
                  "–" if an != an else f"{an:.2f}", f"{r['interest']:.0f}")
    console.print(t)
    print(f"Report: {outdir / 'report.html'}")


FOREVER_MAX_UNITS = 128

# Why a signal is set aside, strongest evidence first; each signal is counted once, under the
# first reason that applies. The wording is for people who don't read SNR tables.
SET_ASIDE = [
    ("mirror_image", "receiver artefacts: mirror pairs around a channel centre"),
    ("cadence_failed", "failed the ON/OFF test: seen while pointing away, or not every time"),
    ("multi_target", "seen (or in a band busy with signals) at another star, so not from this one"),
    ("known_rfi_band", "inside a band used by satellites or radio services"),
    ("stationary", "do not drift, so almost certainly transmitted from Earth"),
    ("out_of_distribution", "far stronger than anything the AI was trained on (strong transmitters)"),
    ("drift_unresolved", "drift cannot be measured in this file's wide channels"),
    ("ai_rfi", "shaped like interference, according to the AI"),
    ("ai_noise", "probably just noise"),
]


def _set_aside_masks(cands) -> dict:
    import pandas as pd
    none = pd.Series(False, index=cands.index)

    def flag(name: str):
        return cands[name].fillna(False).astype(bool) if name in cands else none

    cls = cands["ai_class"].astype(str) if "ai_class" in cands else pd.Series("", cands.index)
    unresolved = flag("drift_unresolved")
    stationary = (flag("stationary") if "stationary" in cands
                  else cands["zero_drift"] > 0 if "zero_drift" in cands else none)
    return {
        "mirror_image": flag("mirror_image"), "multi_target": flag("multi_target"),
        "cadence_failed": (cands["cadence"] == "failed") if "cadence" in cands else none,
        "known_rfi_band": flag("known_rfi_band"),
        "stationary": stationary & ~unresolved,
        "out_of_distribution": cls == "out_of_distribution",
        "drift_unresolved": unresolved,
        "ai_rfi": cls.str.startswith("rfi_"),
        "ai_noise": cls == "noise",
    }


def explain_run(cands, snr_threshold: float, rng: tuple[int, int] | None = None,
                nchans: int | None = None, stats: dict | None = None) -> list[str]:
    """Plain-language lines: where we are in the file, what was seen, and the verdict."""
    import pandas as pd
    lines = []
    if rng and nchans:
        lo, hi = rng
        line = (f"Progress in this file: channels {lo:,}–{hi:,} of {nchans:,} "
                f"({100 * hi / nchans:.1f}% done)")
        if hi < nchans:
            runs = -(-(nchans - hi) // max(hi - lo, 1))
            line += f"; about {runs:,} more run(s) like this to finish it."
        else:
            line += "; this file is finished."
        lines.append(line)
    if stats and (stats.get("spikes") or stats.get("pulses")):
        # Secondary detectors (B37): spikes on a known tone are dropped; say what is left.
        lines.append(f"Also: {stats.get('spikes', 0):,} lone spike(s) (single bright samples, "
                     f"not part of any tone) and {stats.get('pulses', 0):,} broadband pulse(s), "
                     f"{stats.get('pulses_undispersed', 0):,} of them undispersed, so from Earth. "
                     "See spikes.csv / pulses.csv.")
    if cands.empty:
        lines.append(f"Nothing rose above the noise (SNR {snr_threshold:g}). That is the usual result.")
        return lines
    lines.append(f"Seen: {len(cands):,} signal(s) above the noise. Set aside:")
    masks = _set_aside_masks(cands)
    remaining = pd.Series(True, index=cands.index)
    for key, text in SET_ASIDE:
        hit = masks[key] & remaining
        if hit.any():
            lines.append(f"  {int(hit.sum()):>5,}  {text}")
        remaining &= ~hit
    survivors = cands[remaining]
    if survivors.empty:
        lines.append("Verdict: nothing here needs follow-up; everything has an ordinary explanation.")
    else:
        b = survivors.iloc[0]
        tested = b.get("cadence") == "passed"
        lines.append(f"Verdict: {len(survivors):,} signal(s) passed every automatic check. Best: "
                     f"{b['frequency_mhz']:.6f} MHz, drifting {b['drift_rate_hz_s']:+.3f} Hz/s, "
                     f"interest {b['interest']:.0f}/100. "
                     + ("It also passed the ON/OFF cadence; it needs a re-observation."
                        if tested else "Still unverified until an ON/OFF cadence check and a "
                                       "re-observation."))
    return lines


@app.command()
def demo(outdir: Path = typer.Option(Path("reports/demo")),
         workers: int = typer.Option(None, help="Worker processes (default: all cores)."),
         n_chan: int = typer.Option(1 << 20, help="Channels in the synthetic coarse channel."),
         snr: float = typer.Option(10.0), live: bool = typer.Option(True),
         config: Path = CONFIG_OPT):
    """Crunch a synthetic GBT-like coarse channel with hidden ET-like signals and RFI."""
    from .sources import SyntheticSource
    cfg = _cfg(config, workers, snr)
    src = SyntheticSource(Path("data/raw/synthetic"), n=1, n_chan=n_chan)
    location, kind, header, meta = next(src.observations(set()))
    cands, stats = crunch_observation(location, kind, header, meta, cfg, outdir, live)
    _summary(cands, stats, outdir)
    inj = [i for i in src.injections[location] if i["kind"] == "technosignature_like"]
    found = 0
    for i in inj:
        span = abs(i["drift_ch_per_step"]) * header.nsamples + 4
        m = cands[(cands["channel"] - i["start_channel"]).abs() <= span] if len(cands) else cands
        rank = int(m.index.min()) + 1 if len(m) else None
        found += rank is not None
        print(f"  hidden ET-like tone at channel {i['start_channel']:.0f} "
              f"(drift {i['drift_ch_per_step']:+.2f} ch/step): "
              + (f"[green]recovered, rank #{rank}[/green]" if rank else "[red]missed[/red]"))
    print(f"[bold]{found}/{len(inj)} hidden signals recovered.[/bold]")


@app.command()
def inspect(path: str):
    """Print header metadata for a local .fil/.h5 file or a remote .fil URL."""
    from .io.filterbank import read_header
    from .io.remote import remote_header
    hdr = remote_header(path) if path.startswith("http") else read_header(Path(path))
    t = Table(title=f"Filterbank: {path}")
    t.add_column("Property")
    t.add_column("Value")
    for k, v in hdr.to_dict().items():
        t.add_row(k, str(v))
    t.add_row("frequency range MHz", f"{hdr.f_min:.6f} – {hdr.f_max:.6f}")
    t.add_row("fine channels / coarse", str(hdr.fine_per_coarse()))
    size = hdr.nsamples * hdr.nchans * hdr.nifs * hdr.bytes_per_value / 2**30
    t.add_row("data size", f"{size:.2f} GB")
    console.print(t)


@app.command()
def analyze(path: str, outdir: Path = typer.Option(Path("reports/analysis")),
            f_start_mhz: float = typer.Option(None), f_stop_mhz: float = typer.Option(None),
            workers: int = typer.Option(None), snr: float = typer.Option(None),
            live: bool = typer.Option(True), config: Path = CONFIG_OPT):
    """Search a local .fil/.h5 file or a remote .fil URL (streamed, never fully downloaded)."""
    from .io.filterbank import read_header
    from .io.remote import remote_header
    cfg = _cfg(config, workers, snr)
    if path.startswith("http"):
        hdr, kind = remote_header(path), "remote"
    else:
        if not Path(path).is_file():
            raise typer.BadParameter(f"File does not exist: {path}")
        hdr, kind = read_header(Path(path)), "local"
    from .sources import unit_too_large
    if reason := unit_too_large(hdr, cfg):
        print(f"[red]Not searched:[/red] {reason}")
        raise typer.Exit(1)
    rng = hdr.channel_range(f_start_mhz, f_stop_mhz)
    meta = {"target": hdr.source_name, "source": kind}
    cands, stats = crunch_observation(path, kind, hdr, meta, cfg, outdir, live, chan_range=rng)
    _summary(cands, stats, outdir)


@app.command()
def status():
    """Check the SETI@home and Breakthrough Listen data servers."""
    from .sources import BL_API, SAH_STATUS, BreakthroughListenSource, SetiAtHomeSource
    print(f"[dim]Checking {SAH_STATUS} …[/dim]")
    s = SetiAtHomeSource.probe(timeout=10.0)
    colour = "green" if s["distributing"] else "yellow" if s["reachable"] else "red"
    print(f"SETI@home: [{colour}]{s['detail']}[/{colour}]")
    if not s["reachable"]:
        print("[dim]  SETI@home has been hibernating since 2020; AI-SETI does not need it.[/dim]")
    # An empty target matches nothing, so it could not tell a working archive from an empty
    # answer (backlog B13). Ask for one row of a target that is known to exist.
    print(f"[dim]Checking {BL_API} …[/dim]")
    try:
        rows = BreakthroughListenSource(target="HIP", limit=1).query(timeout=10.0, retries=1)
        if rows:
            print("Breakthrough Listen Open Data: [green]reachable[/green] (query returned data)")
        else:
            print("Breakthrough Listen Open Data: [yellow]answered, but returned no rows[/yellow]")
    except Exception as exc:
        print(f"Breakthrough Listen Open Data: [red]unreachable[/red] ({exc})")


@app.command()
def crunch(source: str = typer.Option("auto", help="auto | bl | local | synthetic"),
           target: str = typer.Option("", help="BL target filter, e.g. 'HIP' or '!Voyager1'."),
           telescope: str = typer.Option("GBT"),
           file_types: str = typer.Option("filterbank", help="BL file types: filterbank, HDF5"),
           limit: int = typer.Option(10, help="Observations per query."),
           watch_dir: Path = typer.Option(Path("data/raw"), help="Directory for --source local."),
           forever: bool = typer.Option(False, help="Keep polling for new data like a BOINC client."),
           poll_seconds: int = typer.Option(600),
           max_units: int = typer.Option(None, help="Crunch at most N units per observation per "
                                         "visit; the next visit continues where this one stopped. "
                                         "Default: whole files, or 128 (~2 GB) with --forever."),
           outdir: Path = typer.Option(Path("reports/crunch")),
           state: Path = typer.Option(Path("data/state.json")),
           workers: int = typer.Option(None), snr: float = typer.Option(None),
           live: bool = typer.Option(True),
           auto_share: bool = typer.Option(False, help="After each observation, share findings "
                                           "that pass the gate to the configured share_to sinks."),
           retry_failed: bool = typer.Option(False, help="Forget recorded failures and try "
                                             "those files again (e.g. after an update)."),
           cadence: bool = typer.Option(True, help="For archive scans of a cadence's target, also "
                                        "search the same channels in its other scans and run the "
                                        "ON/OFF test (about 6x the work per run)."),
           include_unresolvable: bool = typer.Option(
               False, help="Also crunch products whose channels are too wide to measure drift "
                           "(BL mid-res .0002). Skipped by default: ~2 min each, never shareable."),
           config: Path = CONFIG_OPT):
    """Auto-load new data and crunch it continuously, SETI@home style."""
    from .sources import (
        BreakthroughListenSource,
        LocalSource,
        SetiAtHomeSource,
        SyntheticSource,
        cadence_siblings,
        drift_resolvable,
        unit_too_large,
    )
    from .state import MAX_ATTEMPTS, Ledger
    cfg = _cfg(config, workers, snr)
    ledger = Ledger.load(state)
    if max_units is None:
        # A high-res file is 64 GB (~2 h at ~8 MB/s): --forever takes it in ~2 GB visits so
        # work is saved often and new targets keep coming in (backlog B44).
        max_units = FOREVER_MAX_UNITS if forever else 0
    from .pipeline import SCORING_VERSION
    if old := ledger.drop_stale_best(SCORING_VERSION):
        print(f"[dim]Lifetime best (interest {old.get('interest')}) was scored under older rules "
              "and has been reset; the next run sets a new one.[/dim]")
    if retry_failed and ledger.failed:
        print(f"Retrying {len(ledger.failed)} previously failed file(s).")
        ledger.failed.clear()

    if source == "auto":
        s = ledger.seti_at_home_status(lambda: SetiAtHomeSource.probe(timeout=5.0))
        print(f"SETI@home server: {s['detail']}")
        if s["distributing"]:
            print("[yellow]SETI@home is sending work again: run the official BOINC client to "
                  "contribute there. AI-SETI continues on Breakthrough Listen data.[/yellow]")
        source = "bl"
    src: BreakthroughListenSource | LocalSource | SyntheticSource
    if source == "bl":
        src = BreakthroughListenSource(target, telescope, file_types, limit,
                                       Path("data/raw/bl"), cfg.max_download_mb)
    elif source == "local":
        src = LocalSource(watch_dir)
    elif source == "synthetic":
        src = SyntheticSource(Path("data/raw/synthetic"), n=limit, n_chan=1 << 18,
                              seed=int(time.time()) % 100000)
    else:
        raise typer.BadParameter(f"Unknown source {source}")

    n_pass = 0
    while True:
        n_new = n_failed = 0
        n_pass += 1
        t_pass, mb_pass = time.time(), 0.0
        for location, kind, header, meta in src.observations(ledger.skip(), set(ledger.progress)):
            if kind == "error":
                n_failed += 1
                ledger.fail(str(meta.get("url")), str(meta.get("error")))
                ledger.save()
                print(f"[red]skip[/red] {meta.get('url')}: {meta.get('error')}")
                continue
            key = str(meta.get("url") or location)
            too_large = unit_too_large(header, cfg)
            if not too_large and not include_unresolvable and not drift_resolvable(header, cfg):
                too_large = (f"drift is not measurable in its {abs(header.foff) * 1e6:,.0f} Hz "
                             "channels, so no hit could ever pass the drift check; use the "
                             "high-resolution product, or --include-unresolvable.")
            if too_large:
                n_failed += 1
                ledger.fail(key, too_large, permanent=True)
                ledger.save()
                print(f"[yellow]skip[/yellow] {Path(key).name}: {too_large}")
                continue
            n_new += 1
            print(f"\n[bold]Pass {n_pass} · file {n_new}[/bold]")
            stem = Path(str(location).split("?")[0]).stem
            print(f"\n[bold #36c2b4]New work:[/bold #36c2b4] {meta.get('target')}  {stem}  "
                  f"({header.f_min:.3f}–{header.f_max:.3f} MHz, {header.nchans:,} ch)")
            rng = ledger.next_range(key, header.nchans, max_units * cfg.channels_per_unit)
            partial = rng != (0, header.nchans)
            report_dir = outdir / (f"{stem}_ch{rng[0]}-{rng[1]}" if partial else stem)
            if partial:
                print(f"Channels {rng[0]:,}–{rng[1]:,} of {header.nchans:,}")
            sibs = cadence_siblings(key) if cadence and kind == "remote" else None
            cands, stats = crunch_observation(location, kind, header, meta, cfg, report_dir,
                                              live, ledger, chan_range=rng, cadence=sibs)
            mb_pass += ((rng[1] - rng[0]) * header.nsamples * header.nifs
                        * header.bytes_per_value / 2**20)
            _summary(cands, stats, report_dir, n=5)
            for line in explain_run(cands, cfg.snr_threshold, rng, header.nchans, stats):
                print(line)
            if auto_share and not cands.empty:
                _share_run(report_dir, cfg, cfg.share_to, ledger, cfg.share_top,
                           cfg.share_min_interest, None, True, False, cfg.share_require_cadence)
            if stats.get("errors"):
                # Don't mark channels searched that weren't: retry this range next run.
                ledger.fail(key, f"{stats['errors']} work unit(s) failed")
            else:
                ledger.advance(key, rng[1], header.nchans)
            ledger.save()
        took = time.time() - t_pass
        print(f"[bold]Pass {n_pass} done:[/bold] {n_new} file(s), ~{mb_pass:,.0f} MB in "
              f"{took // 60:.0f}m{took % 60:02.0f}s"
              + (f" ({mb_pass / took:.1f} MB/s)" if took > 0 and mb_pass else "")
              + (f"; {len(ledger.progress)} file(s) partly searched" if ledger.progress else ""))
        gave_up = sum(v["attempts"] >= MAX_ATTEMPTS for v in ledger.failed.values())
        print(f"{n_new} crunched, {n_failed} failed or skipped this pass; {gave_up} file(s) set "
              f"aside for good (unsearchable here, or failed {MAX_ATTEMPTS} times; see 'failed' "
              f"in {state}).")
        if not forever:
            if n_new == 0:
                print("No new observations matched. Try another --target or --source.")
            break
        if n_new:
            # Work may be waiting (partly searched files, more archive rows): don't idle (B38).
            print("[dim]Continuing at once: there may be more work waiting.[/dim]")
            continue
        print(f"[dim]Nothing new this pass; next poll in {poll_seconds // 60}:"
              f"{poll_seconds % 60:02d}.[/dim]")
        time.sleep(poll_seconds)
    print(f"Lifetime: {ledger.work_units:,} work units, {ledger.channels:,} channels, "
          f"{ledger.cpu_seconds / 3600:.2f} CPU-hours. Best interest so far: "
          f"{ledger.best.get('interest', '—')}")


@app.command()
def cadence(files: list[Path], outdir: Path = typer.Option(Path("reports/cadence")),
            workers: int = typer.Option(None), snr: float = typer.Option(None),
            config: Path = CONFIG_OPT):
    """ON/OFF cadence filter over several scans (OFF scans have '_OFF' in their name)."""
    from .io.filterbank import read_header
    from .rfi import cadence_filter, cadence_is_testable, is_off_scan
    cfg = _cfg(config, workers, snr)
    scans = []
    for f in files:
        hdr = read_header(f)
        cands, _ = crunch_observation(str(f), "local", hdr, {"target": hdr.source_name},
                                      cfg, outdir / f.stem, live=False)
        on = not (is_off_scan(hdr.source_name) or is_off_scan(f.name))
        scans.append({"name": f.name, "mjd": hdr.tstart, "is_on": on, "hits": cands})
        print(f"{f.name}: {'ON ' if on else 'OFF'} {len(cands)} hits")
    testable, reason = cadence_is_testable(scans)
    if not testable:
        print(f"[yellow]Cadence test not possible: {reason}. Nothing can be confirmed or "
              f"rejected, so events.csv will be empty.[/yellow]")
    events = cadence_filter(scans)
    outdir.mkdir(parents=True, exist_ok=True)
    events.to_csv(outdir / "events.csv", index=False)
    print(f"[bold]{len(events)} event(s) passed the cadence filter[/bold] -> {outdir / 'events.csv'}")


def _share_run(report_dir: Path, cfg: SearchConfig, destinations: list[str], ledger,
               top: int, min_interest: float, cadence_events: Path | None, send: bool,
               allow_synthetic: bool, require_cadence: bool | None):
    from .share import build_findings, passes_gate, share
    recs = build_findings(report_dir, top, min_interest, cfg.reporter_handle, cadence_events)
    if not recs:
        print(f"No candidates with interest >= {min_interest:g} in {report_dir}.")
        return
    t = Table(title="Findings ready to share")
    for col in ("ID", "MHz", "Hz/s", "SNR", "interest", "cadence", "gate"):
        t.add_column(col)
    remote = [d for d in destinations if d != "bundle"]
    gate_cadence = bool(remote) if require_cadence is None else require_cadence
    for r in recs:
        ok, why = passes_gate(r, gate_cadence)
        t.add_row(r["finding_id"], f"{r['signal']['frequency_mhz']:.6f}",
                  f"{r['signal']['drift_rate_hz_s']:+.3f}", f"{r['signal']['snr']:.1f}",
                  f"{r['assessment']['interest']:.0f}", r["assessment"]["cadence"],
                  "[green]ok[/green]" if ok else f"[yellow]{why}[/yellow]")
    console.print(t)
    if remote and not send:
        print(f"[yellow]Preview only: not sent to {', '.join(remote)}. Add --yes to send.[/yellow]")
        destinations = [d for d in destinations if d == "bundle"]
    results, skipped = share(recs, destinations, bundle_dir=Path("reports/share"), ledger=ledger,
                             repo=cfg.github_repo, webhook=cfg.webhook_url,
                             webhook_format=cfg.webhook_format, allow_synthetic=allow_synthetic,
                             require_cadence=require_cadence)
    for res in results:
        colour = "green" if res.ok else "red"
        print(f"[{colour}]{res.destination}[/{colour}] {res.finding_id}: {res.detail}")
    for fid, why in skipped:
        print(f"[dim]skipped {fid}: {why}[/dim]")
    if ledger is not None:
        ledger.save()


@app.command("share")
def share_cmd(report_dir: Path = typer.Argument(..., help="A run folder containing candidates.csv"),
              to: list[str] = typer.Option(None, help="bundle | github | webhook (repeatable)"),
              repo: str = typer.Option(None, help="GitHub owner/name for --to github"),
              webhook: str = typer.Option(None, help="Endpoint URL for --to webhook"),
              webhook_format: str = typer.Option(None, help="json | slack | discord"),
              handle: str = typer.Option(None, help="Public name to credit (optional)"),
              top: int = typer.Option(None), min_interest: float = typer.Option(None),
              cadence_events: Path = typer.Option(None, help="events.csv from `ai-seti cadence`"),
              require_cadence: bool = typer.Option(None, help="Only share cadence-passing finds"),
              allow_synthetic: bool = typer.Option(False, help="Allow demo data to remote sinks"),
              yes: bool = typer.Option(False, "--yes", help="Actually send to remote destinations"),
              state: Path = typer.Option(Path("data/state.json")), config: Path = CONFIG_OPT):
    """Package findings as `ai-seti-finding/1` records and share them back."""
    from .state import Ledger
    cfg = _cfg(config)
    for attr, val in (("github_repo", repo), ("webhook_url", webhook),
                      ("webhook_format", webhook_format), ("reporter_handle", handle)):
        if val:
            setattr(cfg, attr, val)
    if not (report_dir / "candidates.csv").exists():
        raise typer.BadParameter(f"No candidates.csv in {report_dir}")
    _share_run(report_dir, cfg, to or cfg.share_to, Ledger.load(state),
               top or cfg.share_top,
               cfg.share_min_interest if min_interest is None else min_interest,
               cadence_events, yes, allow_synthetic,
               cfg.share_require_cadence if require_cadence is None else require_cadence)


@app.command()
def train(per_class: int = typer.Option(500), seed: int = typer.Option(0),
          out: Path = typer.Option(None, help="Model path (default: bundled model).")):
    """Retrain the hit classifier on freshly simulated, pipeline-detected hits."""
    from rich.progress import Progress

    from .ai.model import DEFAULT_MODEL, LABELS
    from .ai.model import train as do_train
    with Progress() as prog:
        task = prog.add_task("Simulating + detecting training hits", total=per_class * len(LABELS))
        meta = do_train(per_class, seed, out or DEFAULT_MODEL,
                        progress=lambda d, t: prog.update(task, completed=d))
    print(f"Test accuracy: [bold]{meta['test_accuracy']:.3f}[/bold] on {len(meta['labels'])} classes")
    print(json.dumps({"labels": meta["labels"], "confusion_matrix": meta["confusion_matrix"]}))


@app.command()
def benchmark(trials: int = typer.Option(20), out: Path = typer.Option(Path("reports/benchmark"))):
    """Injection/recovery: v0.1 detector vs v0.2 at an equal false-alarm rate."""
    from rich.progress import Progress

    from .benchmark import run
    with Progress() as prog:
        task = prog.add_task("Injecting and recovering", total=None)
        res = run(trials=trials, progress=lambda d, t: prog.update(task, completed=d, total=t))
    out.mkdir(parents=True, exist_ok=True)
    res["table"].to_csv(out / "recovery.csv", index=False)
    (out / "summary.json").write_text(json.dumps({k: v for k, v in res.items() if k != "table"},
                                                 indent=2))
    console.print(res["table"].to_string(index=False))
    print(f"Thresholds at equal false-alarm rate: {res['thresholds']}")
    print(f"Seconds per {res['chunk']} chunk: {res['seconds_per_chunk']}")


@app.command()
def summary(root: Path = typer.Option(Path("reports"), help="Reports tree to summarise."),
            outdir: Path = typer.Option(None, help="Where to write; defaults to <root>."),
            top: int = typer.Option(10, help="Rows to print to the terminal.")):
    """Combine every existing run report under a tree into one overview page.

    Reads what previous runs already wrote; it never re-runs a search and never changes
    how an individual run reports itself.
    """
    import pandas as pd

    from .summary import best_overall, warnings_for, write_summary

    root = Path(root)
    out = Path(outdir) if outdir else root
    if not root.is_dir():
        raise typer.BadParameter(f"No such reports directory: {root}")
    html, df = write_summary(root, out)
    runs = df.iloc[:-1] if len(df) else df
    if runs.empty:
        console.print(f"[yellow]No completed runs found under[/yellow] {root}")
        return

    best = best_overall(df)
    t = Table(title=f"AI-SETI overview of {len(runs)} run(s) under {root}")
    t.add_column("target")
    t.add_column("hits", justify="right")
    t.add_column("best interest", justify="right")
    t.add_column("best MHz")
    t.add_column("AI class")
    t.add_column("flags", overflow="fold")

    def shown(value, spec: str) -> str:
        """Format a possibly-missing number. NaN is truthy, so test it explicitly."""
        return spec.format(value) if pd.notna(value) else "–"

    for _, r in runs.nlargest(top, "best_interest").iterrows():
        flags = []
        if bool(r.get("best_mirror_image")):
            flags.append("mirror image")
        if bool(r.get("best_multi_target")):
            flags.append("multi-target")
        if bool(r.get("drift_degraded")):
            flags.append("drift degraded")
        if r.get("ai_note"):
            flags.append("AI inactive")
        if pd.notna(r.get("errors")) and r["errors"] > 0:
            flags.append(f"{int(r['errors'])} failed unit(s)")
        t.add_row(str(r.get("target") or "–"), shown(r.get("hits"), "{:.0f}"),
                  shown(r.get("best_interest"), "{:.1f}"),
                  shown(r.get("best_frequency_mhz"), "{:.4f}"),
                  str(r.get("best_ai_class") or "–"), ", ".join(flags) or "–")
    console.print(t)
    for w in warnings_for(df):
        console.print(f"[gold]![/gold] {w}")
    if best:
        console.print(f"\nBest across all runs: [bold]{best['best_interest']:.1f}[/bold] interest "
                      f"at {best['best_frequency_mhz']:.6f} MHz ({best['run']})")
    console.print(f"\nOverview: [bold]{html}[/bold]\nData:    {out / 'summary.csv'}")


if __name__ == "__main__":
    app()
