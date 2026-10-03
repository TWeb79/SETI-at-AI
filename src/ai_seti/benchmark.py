"""Injection / recovery benchmark: the v0.1 median-spectrum detector vs v0.2 de-Doppler.

Both detectors are first calibrated on noise-only data to the SAME false-alarm rate
(at most `fa_rate` of noise chunks produce any detection), so the comparison is fair:
we compare sensitivity at equal false alarms, not at whatever defaults happen to be.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .ai.simulate import Injection, inject, noise_waterfall
from .core import robust_candidates
from .dsp.dedoppler import drift_search
from .dsp.preprocess import normalize

T, F = 16, 16384
TSAMP, FOFF = 1.0, 1e-6      # 1 s, 1 Hz channels -> drift in ch/step == Hz/s
MAX_DRIFT = 9.0


def _legacy_max(data) -> tuple[float, np.ndarray]:
    c = robust_candidates(data, threshold=-1e9)
    return float(c["robust_z"].max()), c


def _new_scores(data):
    z = normalize(data, block=256, fine_per_coarse=F)
    hits, snr = drift_search(z, TSAMP, FOFF, MAX_DRIFT, snr_threshold=0.0, max_hits=50)
    return snr, hits


def calibrate(n_noise: int, rng, fa_rate: float) -> tuple[float, float]:
    leg, new = [], []
    for _ in range(n_noise):
        d = noise_waterfall(T, F, rng, fine_per_coarse=F)
        spec = np.median(d, axis=0)
        med = np.median(spec)
        mad = 1.4826 * np.median(np.abs(spec - med))
        z = (spec - med) / mad
        z[F // 2] = -np.inf   # ignore the (always-present) DC spike for calibration
        leg.append(z.max())
        s, _ = _new_scores(d)
        new.append(float(s.max()))
    q = 1 - fa_rate
    return float(np.quantile(leg, q)), float(np.quantile(new, q))


def run(trials: int = 20, snrs=(0.75, 1.0, 1.5, 2.0, 3.0), drifts=(0.0, 0.5, 2.0, 8.0),
        n_noise: int = 60, fa_rate: float = 0.05, seed: int = 3, progress=None) -> dict:
    rng = np.random.default_rng(seed)
    thr_leg, thr_new = calibrate(n_noise, rng, fa_rate)
    rows = []
    t_leg = t_new = 0.0
    n_runs = 0
    total = len(snrs) * len(drifts) * trials
    for a in snrs:
        for dr in drifts:
            rec_l = rec_n = 0
            for _ in range(trials):
                d = noise_waterfall(T, F, rng, fine_per_coarse=F)
                start = float(rng.uniform(200, F - 200 - MAX_DRIFT * T))
                if F // 2 - 40 < start < F // 2 + 40:
                    start += 200
                inject(d, Injection("technosignature_like", start, dr, a, 1.0), rng)
                lo, hi = sorted([start, start + dr * (T - 1)])
                t0 = time.perf_counter()
                c = robust_candidates(d, threshold=thr_leg)
                t_leg += time.perf_counter() - t0
                if ((c["channel_index"] >= lo - 3) & (c["channel_index"] <= hi + 3)).any():
                    rec_l += 1
                t0 = time.perf_counter()
                z = normalize(d, block=256, fine_per_coarse=F)
                hits, _ = drift_search(z, TSAMP, FOFF, MAX_DRIFT, snr_threshold=thr_new)
                t_new += time.perf_counter() - t0
                if any(abs(h.start_channel - start) <= 3 and abs(h.drift_ch_per_step - dr) <= 0.5
                       for h in hits):
                    rec_n += 1
                n_runs += 1
                if progress:
                    progress(n_runs, total)
            rows.append({"snr_per_sample": a, "drift_ch_per_step": dr,
                         "v0.1_recovery": rec_l / trials, "v0.2_recovery": rec_n / trials})
    df = pd.DataFrame(rows)
    return {
        "table": df,
        "thresholds": {"v0.1_robust_z": thr_leg, "v0.2_snr": thr_new},
        "false_alarm_rate_per_chunk": fa_rate,
        "chunk": {"n_time": T, "n_chan": F},
        "seconds_per_chunk": {"v0.1": t_leg / max(n_runs, 1), "v0.2": t_new / max(n_runs, 1)},
    }
