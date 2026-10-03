"""Regenerate the README's example figures: `python docs/make_figures.py`.

Author: Inventions4All - github:TWeb79

Two raw time x frequency waterfalls, each run through the program's own pipeline steps:

* signal_normal.png  - a real terrestrial carrier from Breakthrough Listen open data
  (HIP2579, 7990.78 MHz, GBT). Streams a 16 x 256-channel window, about 16 kB.
* signal_unusual.png - the kind of signal the search is built for: a narrow tone that drifts.
  Simulated (the demo's injection), because no real one has ever been found.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai_seti.ai.features import feature_vector, hit_features
from ai_seti.ai.model import LABELS, HitScorer
from ai_seti.ai.simulate import Injection, inject, noise_waterfall
from ai_seti.dsp.dedoppler import drift_search
from ai_seti.dsp.preprocess import normalize

OUT = Path(__file__).with_name("images")
REAL_URL = ("http://blpd0.ssl.berkeley.edu/LHS1140/C/spliced_blc00010203040506071011121314151617"
            "_guppi_57774_77843_HIP2579_0004.gpuspec.0000.fil")
REAL_CHANNEL = 3824217          # 7990.780103 MHz, found by `ai-seti crunch` on 2026-10-03
TSAMP_HI, FOFF_HI = 18.253611008, -2.7939677238464355e-06   # GBT high-res product


def _measure(z: np.ndarray, near: int, tsamp: float, foff_mhz: float) -> dict:
    hits, _ = drift_search(z, tsamp=tsamp, foff_mhz=foff_mhz, max_drift_hz_s=4.0,
                           snr_threshold=8.0)
    h = min(hits, key=lambda x: abs(x.start_channel - near))
    feats, img = hit_features(z, h.start_channel, h.drift_ch_per_step, h.snr)
    probs = HitScorer().predict(feature_vector(feats, img)[None, :])[0]
    return {"hit": h, "feats": feats, "p": dict(zip(LABELS, probs, strict=True))}


def _figure(z: np.ndarray, m: dict, foff_hz: float, title: str, verdict, path: Path,
            colour: str) -> None:
    h, f, p = m["hit"], m["feats"], m["p"]
    t_n, f_n = z.shape
    # Signed: BL files run from high to low frequency, so channel order is reversed.
    hz = (np.arange(f_n) - h.start_channel) * foff_hz
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(10, 4.2), width_ratios=[1.35, 1],
                                 constrained_layout=True)
    ax.imshow(np.clip(z, -2, 8), aspect="auto", cmap="magma", origin="upper",
              extent=(hz[0], hz[-1], t_n - 0.5, -0.5), interpolation="nearest")
    track = h.drift_ch_per_step * np.arange(t_n) * foff_hz
    ax.plot(track, np.arange(t_n), "--", color="#36c2b4", lw=1.2, label="track found by the search")
    ax.set_xlabel("frequency offset from the hit (Hz)")
    ax.set_ylabel("time sample (18.25 s each)")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, fontsize=10)
    best = max(p, key=lambda k: p[k])
    lines = [
        f"drift            {h.drift_rate_hz_s:+.3f} Hz/s",
        f"SNR              {h.snr:,.1f}",
        f"bandwidth        {f['bandwidth_ch']:.0f} channel(s)",
        f"present in       {100 * f['on_fraction']:.0f}% of samples",
        f"AI class         {best} (p = {p[best]:.2f})",
        "",
        *verdict(h, f),
    ]
    tx.axis("off")
    tx.text(0.06, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    tx.add_patch(plt.Rectangle((0, 0), 0.02, 1, color=colour, transform=tx.transAxes))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def normal() -> None:
    from ai_seti.io import remote
    hdr = remote.remote_header(REAL_URL)
    c0 = REAL_CHANNEL - 128
    raw = remote.remote_window(REAL_URL, hdr, c0, c0 + 256)
    z = normalize(raw, block=64)
    m = _measure(z, 128, hdr.tsamp, hdr.foff)
    _figure(z, m, hdr.foff * 1e6,
            "Normal: a terrestrial carrier (real data, HIP2579, 7990.78 MHz)",
            lambda h, f: [
                "Why it is not ET:",
                "- a vertical line: it does not drift, so it",
                "  moves with the telescope -> it is on Earth",
                f"- far stronger (SNR {h.snr:,.0f}) than anything the",
                "  AI was trained on",
                "- the same frequency also appeared while",
                "  pointing at another star (HIP3249)"],
            OUT / "signal_normal.png", "#8a8a8a")


def unusual() -> None:
    rng = np.random.default_rng(7)
    data = noise_waterfall(16, 512, rng)
    inject(data, Injection("technosignature_like", 200.0, 3.0, 4.0, 1.0), rng)
    z = normalize(data, block=256)
    m = _measure(z, 200, TSAMP_HI, FOFF_HI)
    _figure(z, m, FOFF_HI * 1e6,
            "Unusual: a narrow drifting tone (simulated, as in `ai-seti demo`)",
            lambda h, f: [
                "Why it is worth a look:",
                "- one channel wide: nature rarely makes that",
                "- a slanted line: it drifts steadily, as a",
                "  transmitter on a rotating planet would",
                "- it lasts the whole observation",
                "",
                "Still not a detection: it needs the ON/OFF",
                "cadence check and a re-observation."],
            OUT / "signal_unusual.png", "#ff5a3c")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    unusual()
    normal()
