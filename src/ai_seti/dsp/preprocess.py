"""Bandpass flattening and robust normalisation.

The v0.1 detector compared each channel to the global median of the spectrum, so the
polyphase-filterbank bandpass ripple (and the DC spike in the middle of every GBT
coarse channel) dominated the "outliers". Here we fit a robust piecewise bandpass
(median per block of channels, linearly interpolated), divide out a robust per-block
noise level, and repair DC bins, giving data that is ~N(0, 1) per sample.
"""
from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 1.4826


def _block_stat(arr2d: np.ndarray, block: int, fn) -> np.ndarray:
    """Apply `fn` over blocks of channels and interpolate back to every channel."""
    f = arr2d.shape[1]
    nb = max(1, int(np.ceil(f / block)))
    knots = np.empty(nb, dtype=np.float64)
    centers = np.empty(nb, dtype=np.float64)
    for i in range(nb):
        a, b = i * block, min(f, (i + 1) * block)
        knots[i] = fn(arr2d[:, a:b])
        centers[i] = 0.5 * (a + b - 1)
    if nb == 1:
        return np.full(f, knots[0], dtype=np.float32)
    return np.interp(np.arange(f), centers, knots).astype(np.float32)


def dc_channels(chan_start: int, n: int, fine_per_coarse: int | None) -> np.ndarray:
    """Relative indices of DC-spike bins (centre of each coarse channel) inside a window."""
    if not fine_per_coarse:
        return np.empty(0, dtype=int)
    half = fine_per_coarse // 2
    first = chan_start + ((half - chan_start) % fine_per_coarse)
    abs_idx = np.arange(first, chan_start + n, fine_per_coarse)
    return (abs_idx - chan_start).astype(int)


def normalize(data: np.ndarray, block: int = 512, chan_start: int = 0,
              fine_per_coarse: int | None = None, remove_dc: bool = True) -> np.ndarray:
    """Return float32 [time, chan] with bandpass removed and unit robust noise."""
    x = np.array(data, dtype=np.float32, copy=True)
    if x.ndim == 3:
        x = x.mean(axis=1)
    if x.ndim != 2:
        raise ValueError(f"Expected [time, chan] data, got shape {x.shape}")
    np.nan_to_num(x, copy=False)
    f = x.shape[1]

    if remove_dc:
        for c in dc_channels(chan_start, f, fine_per_coarse):
            lo, hi = max(c - 1, 0), min(c + 1, f - 1)
            x[:, c] = 0.5 * (x[:, lo] + x[:, hi])

    # Robust bandpass: block medians of the time-mean spectrum.
    spec = x.mean(axis=0, keepdims=True)
    bandpass = _block_stat(spec, block, np.median)
    x -= bandpass[None, :]

    # Robust noise per block (MAD over time x channels).
    def mad(a):
        m = np.median(a)
        return MAD_TO_SIGMA * np.median(np.abs(a - m))

    sigma = _block_stat(x, block, mad)
    fallback = float(np.std(x)) or 1.0
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
    x /= sigma[None, :]
    return x
