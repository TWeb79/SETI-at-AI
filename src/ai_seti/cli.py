"""ai-seti command line: demo, analyze, crunch (auto-loading client), cadence, train, benchmark."""
from __future__ import annotations

import json
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


def crunch_observation(location, kind, header, meta, cfg: SearchConfig, outdir: Path,
                       live: bool = True, ledger=None, chan_range=None):
    """Split -> parallel process (with live dashboard) -> score -> write outputs."""
    from .dashboard import CrunchDashboard
    from .pipeline import run_units, score_candidates
    from .report import write_outputs
    from .sources import split

    units = split(location, header, cfg, kind, chan_range=chan_range, meta=meta)
    dash = CrunchDashboard(len(units), ledger)
    dash.meta = meta
    results = []
    t0 = time.time()
    if live and console.is_terminal:
        from rich.live import Live
        with Live(dash.render(), console=console, refresh_per_second=4) as lv:
            for res in run_units(units, cfg):
                results.append(res)
                dash.update(res)
                lv.update(dash.render())
    else:
        for res in run_units(units, cfg):
            results.append(res)
            dash.update(res)
            best = max((h["snr"] for h in res.get("hits", [])), default=0)
            print(f"[dim]{dash.done}/{len(units)}[/dim] {res['unit_id']}  "
                  f"hits={len(res.get('hits', []))}  best SNR={best:.1f}"
                  + (f"  [red]{res['error']}[/red]" if res.get("error") else ""))
    wall = time.time() - t0
    cands = score_candidates(results, cfg)
    title = f"AI-SETI search of {meta.get('target') or Path(str(location)).name}"
    stats = write_outputs(results, cands, outdir, cfg, title,
                          {"location": str(location), "observation": meta,
                           "header": header.to_dict()}, wall)
    if ledger is not None:
        for r in results:
            ledger.record_unit(r)
        if not cands.empty and ledger.record_best({**cands.iloc[0].to_dict(),
                                                   "location": str(location)}):
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
    rng = hdr.channel_range(f_start_mhz, f_stop_mhz)
    meta = {"target": hdr.source_name, "source": kind}
    cands, stats = crunch_observation(path, kind, hdr, meta, cfg, outdir, live, chan_range=rng)
    _summary(cands, stats, outdir)


@app.command()
def status():
    """Check the SETI@home and Breakthrough Listen data servers."""
    from .sources import BreakthroughListenSource, SetiAtHomeSource
    s = SetiAtHomeSource.probe()
    colour = "green" if s["distributing"] else "yellow"
    print(f"SETI@home: [{colour}]{s['detail']}[/{colour}]")
    try:
        rows = BreakthroughListenSource(target="", limit=3).query()
        print(f"Breakthrough Listen Open Data: [green]reachable[/green] ({len(rows)} sample rows)")
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
           max_units: int = typer.Option(0, help="Crunch only the first N units per observation."),
           outdir: Path = typer.Option(Path("reports/crunch")),
           state: Path = typer.Option(Path("data/state.json")),
           workers: int = typer.Option(None), snr: float = typer.Option(None),
           live: bool = typer.Option(True),
           auto_share: bool = typer.Option(False, help="After each observation, share findings "
                                           "that pass the gate to the configured share_to sinks."),
           config: Path = CONFIG_OPT):
    """Auto-load new data and crunch it continuously, SETI@home style."""
    from .sources import BreakthroughListenSource, LocalSource, SetiAtHomeSource, SyntheticSource
    from .state import Ledger
    cfg = _cfg(config, workers, snr)
    ledger = Ledger.load(state)

    if source == "auto":
        s = SetiAtHomeSource.probe()
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

    while True:
        n_new = 0
        for location, kind, header, meta in src.observations(set(ledger.observations_done)):
            if kind == "error":
                print(f"[red]skip[/red] {meta.get('url')}: {meta.get('error')}")
                continue
            n_new += 1
            stem = Path(str(location).split("?")[0]).stem
            print(f"\n[bold #36c2b4]New work:[/bold #36c2b4] {meta.get('target')}  {stem}  "
                  f"({header.f_min:.3f}–{header.f_max:.3f} MHz, {header.nchans:,} ch)")
            rng = (0, min(header.nchans, max_units * cfg.channels_per_unit)) if max_units else None
            cands, stats = crunch_observation(location, kind, header, meta, cfg,
                                              outdir / stem, live, ledger, chan_range=rng)
            _summary(cands, stats, outdir / stem, n=5)
            if auto_share and not cands.empty:
                _share_run(outdir / stem, cfg, cfg.share_to, ledger, cfg.share_top,
                           cfg.share_min_interest, None, True, False, cfg.share_require_cadence)
            ledger.done(str(meta.get("url") or location))
            ledger.save()
        if not forever:
            if n_new == 0:
                print("No new observations matched. Try another --target or --source.")
            break
        print(f"[dim]Waiting {poll_seconds}s for new data…[/dim]")
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
               allow_synthetic: bool, require_cadence: bool):
    from .share import build_findings, passes_gate, share
    recs = build_findings(report_dir, top, min_interest, cfg.reporter_handle, cadence_events)
    if not recs:
        print(f"No candidates with interest >= {min_interest:g} in {report_dir}.")
        return
    t = Table(title="Findings ready to share")
    for col in ("ID", "MHz", "Hz/s", "SNR", "interest", "cadence", "gate"):
        t.add_column(col)
    for r in recs:
        ok, why = passes_gate(r, require_cadence)
        t.add_row(r["finding_id"], f"{r['signal']['frequency_mhz']:.6f}",
                  f"{r['signal']['drift_rate_hz_s']:+.3f}", f"{r['signal']['snr']:.1f}",
                  f"{r['assessment']['interest']:.0f}", r["assessment"]["cadence"],
                  "[green]ok[/green]" if ok else f"[yellow]{why}[/yellow]")
    console.print(t)
    remote = [d for d in destinations if d != "bundle"]
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


if __name__ == "__main__":
    app()
