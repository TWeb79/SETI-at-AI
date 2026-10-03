"""Turn a hit into a de-drifted image snippet plus physically meaningful features."""
from __future__ import annotations

import numpy as np

SNIPPET_HALF = 32
IMG_T, IMG_F = 16, 16
SCALAR_FEATURES = ["snr", "abs_drift_ch", "zero_drift", "bandwidth_ch", "on_fraction",
                   "modulation", "kurtosis", "wobble_rms", "sidelobe_ratio",
                   "off_track_level", "peak_to_mean"]


def dedrifted_snippet(z: np.ndarray, start: float, drift: float,
                      half: int = SNIPPET_HALF) -> np.ndarray:
    t_n, f_n = z.shape
    out = np.zeros((t_n, 2 * half), dtype=np.float32)
    for t in range(t_n):
        c = round(start + drift * t)
        lo, hi = c - half, c + half
        a, b = max(lo, 0), min(hi, f_n)
        if a < b:
            out[t, a - lo: b - lo] = z[t, a:b]
    return out


def _resample_rows(img: np.ndarray, rows: int) -> np.ndarray:
    t = img.shape[0]
    if t == rows:
        return img
    edges = np.linspace(0, t, rows + 1)
    return np.stack([img[int(edges[i]):max(int(edges[i + 1]), int(edges[i]) + 1)].mean(0)
                     for i in range(rows)])


def hit_features(z: np.ndarray, start: float, drift: float, snr: float) -> tuple[dict, np.ndarray]:
    snip = dedrifted_snippet(z, start, drift)
    c = SNIPPET_HALF
    profile = snip.mean(axis=0)
    peak = float(np.max(profile[c - 1:c + 2]))
    above = profile > 0.5 * max(peak, 1e-6)
    lo = c
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = c
    while hi < len(profile) - 1 and above[hi + 1]:
        hi += 1
    on_track = snip[:, c - 1:c + 2].max(axis=1)
    mean_on = float(on_track.mean())
    local = snip[:, c - 6:c + 7]
    offsets = local.argmax(axis=1) - 6
    strong = on_track > 1.5
    wobble = float(np.sqrt(np.mean(offsets[strong] ** 2))) if strong.sum() >= 2 else 6.0
    side = np.r_[profile[c - 20:c - 4], profile[c + 5:c + 21]]
    off_mask = np.ones(snip.shape[1], dtype=bool)
    off_mask[c - 4:c + 5] = False
    centred = on_track - mean_on
    var = float(np.mean(centred ** 2)) or 1e-9
    feats = {
        "snr": float(snr),
        "abs_drift_ch": float(abs(drift)),
        "zero_drift": float(abs(drift) < 0.02),
        "bandwidth_ch": float(hi - lo + 1),
        "on_fraction": float(np.mean(on_track > 2.0)),
        "modulation": float(np.std(on_track) / (abs(mean_on) + 1e-3)),
        "kurtosis": float(np.mean(centred ** 4) / var ** 2),
        "wobble_rms": wobble,
        "sidelobe_ratio": float(side.mean() / (peak + 1e-3)) if side.size else 0.0,
        "off_track_level": float(snip[:, off_mask].mean()),
        "peak_to_mean": float(on_track.max() / (abs(mean_on) + 1e-3)),
    }
    img = _resample_rows(snip, IMG_T).reshape(IMG_T, IMG_F, -1).mean(axis=2)
    img = img / (np.abs(img).max() + 1e-6)
    return feats, img.astype(np.float32)


def feature_vector(feats: dict, img: np.ndarray) -> np.ndarray:
    return np.concatenate([[feats[k] for k in SCALAR_FEATURES], img.ravel()]).astype(np.float32)
