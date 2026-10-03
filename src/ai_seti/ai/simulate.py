"""Synthetic radio data: noise with a realistic bandpass, ET-like signals and RFI.

Used for (1) the demo "observation", (2) injection/recovery benchmarks and
(3) generating labelled training data for the hit classifier. Every injected signal
is described by an `Injection` so recovery can be scored exactly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

CLASSES = ["technosignature_like", "rfi_zero_drift", "rfi_wideband",
           "rfi_nonlinear", "rfi_intermittent", "rfi_strong_carrier"]
# The interference kinds the demo hides among its ET-like tones (weights in synthetic_observation).
DEMO_RFI = ("rfi_zero_drift", "rfi_wideband", "rfi_nonlinear", "rfi_intermittent")


@dataclass
class Injection:
    kind: str
    start_channel: float
    drift_ch_per_step: float
    snr_per_sample: float
    width_channels: float = 1.0

    def to_dict(self):
        return asdict(self)


def noise_waterfall(n_time: int, n_chan: int, rng: np.random.Generator,
                    bandpass: bool = True, fine_per_coarse: int | None = None,
                    chan_start: int = 0, dof: int = 512) -> np.ndarray:
    """Chi-square-like power noise times a polyphase-filterbank-style bandpass."""
    noise = rng.standard_normal((n_time, n_chan), dtype=np.float32)
    noise = 1.0 + noise * np.float32(np.sqrt(2.0 / dof))
    if bandpass:
        x = np.arange(chan_start, chan_start + n_chan, dtype=np.float64)
        period = fine_per_coarse or max(n_chan, 2)
        phase = (x % period) / period
        shape = 1.0 - 0.35 * (2 * phase - 1) ** 8 + 0.02 * np.sin(2 * np.pi * x / 4096.0)
        noise *= shape.astype(np.float32)[None, :]
        if fine_per_coarse:
            dc = (x % fine_per_coarse) == fine_per_coarse // 2
            noise[:, dc] *= 25.0
    return noise.astype(np.float32) * 1e6


def _profile(n_chan: int, centre: float, width: float) -> tuple[np.ndarray, np.ndarray]:
    half = int(np.ceil(3 * width)) + 1
    lo, hi = int(np.floor(centre)) - half, int(np.floor(centre)) + half + 1
    idx = np.arange(max(lo, 0), min(hi, n_chan))
    if width <= 1.0:
        prof = np.sinc(idx - centre) ** 2          # tone in a single FFT bin
    else:
        prof = np.exp(-0.5 * ((idx - centre) / (width / 2.355)) ** 2)
    s = prof.sum()
    return idx, (prof / s if s > 0 else prof)


def inject(data: np.ndarray, inj: Injection, rng: np.random.Generator,
           sigma: np.ndarray | None = None) -> None:
    """Add a signal in place. Amplitude is per-sample SNR relative to local noise."""
    t_n, f_n = data.shape
    if sigma is None:
        sigma = data.std(axis=0)
    on: np.ndarray = np.ones(t_n, dtype=bool)
    if inj.kind == "rfi_intermittent":
        on = np.asarray(rng.random(t_n) < 0.35)
        on[rng.integers(t_n)] = True
    for t in range(t_n):
        if not on[t]:
            continue
        centre = inj.start_channel + inj.drift_ch_per_step * t
        if inj.kind == "rfi_nonlinear":
            centre += 3.0 * inj.width_channels * np.sin(2 * np.pi * t / max(t_n / 2.5, 1)
                                                        + inj.snr_per_sample)
        width = max(inj.width_channels, 12.0) if inj.kind == "rfi_wideband" else inj.width_channels
        idx, prof = _profile(f_n, centre, width)
        if idx.size and prof.max() > 0:
            prof = prof / prof.max()
            data[t, idx] += (inj.snr_per_sample * sigma[idx] * prof).astype(data.dtype)


def random_injection(kind: str, n_time: int, n_chan: int, rng: np.random.Generator,
                     snr_range=(1.5, 6.0), max_drift: float = 4.0) -> Injection:
    margin = int(max_drift * n_time) + 40
    start = float(rng.uniform(margin, n_chan - margin))
    snr = float(rng.uniform(*snr_range))
    if kind == "technosignature_like":
        # Log-uniform from two search steps (one step is within the noise of zero drift, B36)
        # up to the cap, so slow and fast drifts are equally well represented (B3).
        lo = min(2.0 / max(n_time - 1, 1), max_drift)
        drift = float(np.exp(rng.uniform(np.log(lo), np.log(max_drift)))) * rng.choice([-1, 1])
        return Injection(kind, start, drift, snr, float(rng.uniform(0.5, 2.0)))
    if kind == "rfi_strong_carrier":
        # A bright, steady, slowly drifting transmitter: the commonest real RFI, and the one
        # the classifier used to call technosignature_like with p = 1 (former B8).
        bright = float(np.exp(rng.uniform(np.log(10.0), np.log(1000.0))))
        return Injection(kind, start, float(rng.uniform(-0.3, 0.3)), bright,
                         float(rng.uniform(0.5, 2.0)))
    if kind == "rfi_zero_drift":
        return Injection(kind, start, 0.0, snr, float(rng.uniform(0.5, 3.0)))
    if kind == "rfi_wideband":
        return Injection(kind, start, float(rng.uniform(-0.3, 0.3)), snr * 0.6,
                         float(rng.uniform(12, 40)))
    if kind == "rfi_nonlinear":
        return Injection(kind, start, float(rng.uniform(-1, 1)), snr,
                         float(rng.uniform(1.0, 3.0)))
    if kind == "rfi_intermittent":
        return Injection(kind, start, float(rng.uniform(-2, 2)), snr * 1.6,
                         float(rng.uniform(0.5, 2.0)))
    raise ValueError(kind)


def synthetic_observation(n_time: int = 16, n_chan: int = 1 << 20, seed: int = 42,
                          fch1: float = 1420.0, foff: float = -2.7939677238464355e-06,
                          tsamp: float = 18.253611008, n_rfi: int = 60,
                          n_et: int = 3) -> tuple[np.ndarray, dict, list[Injection]]:
    """A full 'coarse channel' of synthetic GBT-like data with RFI and a few ET-like tones."""
    rng = np.random.default_rng(seed)
    data = noise_waterfall(n_time, n_chan, rng, fine_per_coarse=n_chan)
    sigma = data.std(axis=0)
    injections: list[Injection] = []
    k = abs(4.0 * tsamp / (foff * 1e6))
    for _ in range(n_rfi):
        kind = rng.choice(DEMO_RFI, p=[0.45, 0.2, 0.15, 0.2])
        injections.append(random_injection(str(kind), n_time, n_chan, rng,
                                           snr_range=(2.0, 12.0), max_drift=min(k, 6)))
    for _ in range(n_et):
        inj = random_injection("technosignature_like", n_time, n_chan, rng,
                               snr_range=(2.5, 5.0), max_drift=min(k, 12))
        injections.append(inj)
    for inj in injections:
        inject(data, inj, rng, sigma)
    header = {"fch1": fch1, "foff": foff, "nchans": n_chan, "tsamp": tsamp, "nsamples": n_time,
              "nifs": 1, "nbits": 32, "tstart": 60000.0, "source_name": "SYNTHETIC-TAU-CETI-ISH",
              "extra": {"src_raj": 14320.0, "src_dej": -155618.0, "telescope_id": 6}}
    return data, header, injections
