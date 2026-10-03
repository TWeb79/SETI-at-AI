"""A live terminal 'screensaver' for the cruncher, modelled on the SETI@home client:
data info, user info, analysis progress, the current power spectrum, a scrolling
waterfall and — the part everyone watched — the best signal found so far.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

RAMP = [(0.00, "#22306b"), (0.25, "#2f5fa8"), (0.45, "#36c2b4"),
        (0.70, "#ffcc33"), (1.00, "#ff5a3c")]
BLOCKS = " ▁▂▃▄▅▆▇█"


def _color(v: float) -> str:
    v = float(np.clip(v, 0, 1))
    for (a, ca), (b, cb) in itertools.pairwise(RAMP):
        if v <= b:
            t = (v - a) / (b - a)
            c0 = [int(ca[i:i + 2], 16) for i in (1, 3, 5)]
            c1 = [int(cb[i:i + 2], 16) for i in (1, 3, 5)]
            return "#" + "".join(f"{int(x + (y - x) * t):02x}"
                                 for x, y in zip(c0, c1, strict=True))
    return RAMP[-1][1]


class CrunchDashboard:
    def __init__(self, total_units: int, ledger=None, title: str = "AI-SETI@home"):
        self.total = total_units
        self.done = 0
        self.ledger = ledger
        self.title = title
        self.started = time.time()
        self.current: dict | None = None
        self.best: dict | None = None
        self.hits = 0
        self.channels = 0
        self.meta: dict = {}
        self.status = "warming up the receiver…"

    def update(self, result: dict) -> None:
        self.done += 1
        self.current = result
        self.meta = result.get("meta") or self.meta
        self.channels += int(result.get("n_channels", 0))
        self.hits += len(result.get("hits", []))
        for h in result.get("hits", []):
            score = h.get("p_technosignature_like", 0) or 0
            key = (0 if h.get("zero_drift") else 1, score, h["snr"])
            if self.best is None or key > self.best["_key"]:
                self.best = {**h, "_key": key}
        self.status = f"crunched {result['unit_id']}"

    # ------------------------------------------------------------------ panels
    def _spectrum(self, width: int = 72) -> Text:
        prof = np.asarray(self.current.get("snr_profile", []) if self.current else [])
        if prof.size == 0:
            return Text("no data yet", style="dim")
        x = np.interp(np.linspace(0, prof.size - 1, width), np.arange(prof.size), prof)
        top = max(10.0, float(np.max(x)))
        txt = Text()
        for row in range(5, 0, -1):
            for v in x:
                level = np.clip(v / top * 5 - (row - 1), 0, 1)
                txt.append(BLOCKS[int(level * 8)], style=_color(v / top))
            txt.append("\n")
        return txt

    def _waterfall(self, width: int = 72) -> Text:
        thumb = np.asarray(self.current.get("thumb", []) if self.current else [])
        if thumb.size == 0:
            return Text("")
        cols = np.linspace(0, thumb.shape[1] - 1, width).astype(int)
        rows = np.linspace(0, thumb.shape[0] - 1, min(8, thumb.shape[0])).astype(int)
        txt = Text()
        for r in rows:
            for c in cols:
                txt.append("█", style=_color(thumb[r, c] / 6.0))
            txt.append("\n")
        return txt

    def _info(self) -> Table:
        t = Table.grid(padding=(0, 2))
        t.add_column(style="#b9b3a3")
        t.add_column(style="bold #f2ead8")
        m = self.meta or {}
        t.add_row("Target", str(m.get("target") or "—"))
        t.add_row("Telescope", str(m.get("telescope") or "—"))
        if m.get("ra") is not None:
            t.add_row("RA / Dec", f"{m.get('ra')} / {m.get('decl')}")
        if self.current and "f_lo" in self.current:
            t.add_row("Band", f"{self.current['f_lo']:.6f} – {self.current['f_hi']:.6f} MHz")
        t.add_row("Source", str(m.get("source") or "—"))
        return t

    def _user(self) -> Table:
        el = time.time() - self.started
        t = Table.grid(padding=(0, 2))
        t.add_column(style="#b9b3a3")
        t.add_column(style="bold #f2ead8")
        t.add_row("Work units", f"{self.done} / {self.total}")
        t.add_row("Channels", f"{self.channels:,}")
        t.add_row("Rate", f"{self.channels / el:,.0f} ch/s" if el > 0 else "—")
        t.add_row("Hits", f"{self.hits:,}")
        if self.ledger is not None:
            t.add_row("Lifetime units", f"{self.ledger.work_units + self.done:,}")
            t.add_row("Lifetime CPU", f"{self.ledger.cpu_seconds / 3600:,.2f} h")
        return t

    def _best(self) -> Group:
        if not self.best:
            return Group(Text("listening…", style="dim italic"))
        b = self.best
        t = Table.grid(padding=(0, 2))
        t.add_column(style="#b9b3a3")
        t.add_column(style="bold #ffcc33")
        t.add_row("Frequency", f"{b.get('frequency_mhz', float('nan')):.6f} MHz")
        t.add_row("Drift", f"{b.get('drift_rate_hz_s', 0):+.3f} Hz/s")
        t.add_row("SNR", f"{b['snr']:.1f}")
        if "p_technosignature_like" in b:
            t.add_row("AI: ET-like", f"{b['p_technosignature_like']:.2f}")
        bar = Text("▮" * int(min(b["snr"], 60) / 2), style="#ff5a3c")
        return Group(t, bar)

    def render(self) -> Group:
        frac = self.done / self.total if self.total else 0
        prog = Text()
        filled = int(frac * 50)
        prog.append("█" * filled, style="#ffcc33")
        prog.append("░" * (50 - filled), style="#22306b")
        prog.append(f"  {frac * 100:5.1f}%   {self.status}", style="#b9b3a3")

        # A Group is as tall as its content. A Layout fills the whole terminal and leaves
        # ~15 blank lines behind when Live stops (backlog B15).
        top = Table.grid(expand=True)
        for _ in range(3):
            top.add_column(ratio=1)
        top.add_row(Panel(self._info(), title="Data info", border_style="#3a4a8f"),
                    Panel(self._user(), title="User info", border_style="#3a4a8f"),
                    Panel(self._best(), title="Best signal so far", border_style="#ffcc33"))
        return Group(
            top,
            Panel(self._spectrum(), title="De-Doppler power spectrum (current unit)",
                  border_style="#3a4a8f"),
            Panel(self._waterfall(), title="Waterfall", border_style="#3a4a8f"),
            Panel(prog, title=self.title, border_style="#36c2b4"),
        )
