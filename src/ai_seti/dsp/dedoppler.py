"""De-Doppler (drift-rate) search with a vectorised Taylor tree.

A transmitter on another planet is accelerating relative to the telescope, so its
narrowband tone drifts in frequency: a straight, slanted line in the waterfall.
v0.1 took the time-median of every channel, which smears any line that drifts more
than ~1 channel over the observation into noise. Here we integrate along every
linear path instead:

* Taylor tree: all drifts 0..T-1 channels (0..1 ch/step) in O(T log T · F) instead of
  O(T² · F) brute force, each stage fully vectorised over frequency.
* Shearing: larger drift rates are handled by pre-shifting row t by k·t channels and
  re-running the tree, covering k(T-1) + d for d in [0, T-1].
* Negative drifts: the same search on the frequency-reversed band.
* SNR per drift row from a robust (median/MAD) estimate, then frequency-domain
  non-maximum suppression so one signal gives one hit.

Works with numpy or cupy (`xp`) — every op is array-API style.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

MAD_TO_SIGMA = 1.4826


@dataclass
class Hit:
    start_channel: int          # channel at t = 0 (relative to the array)
    drift_ch_per_step: float    # channels per time sample (signed)
    snr: float
    drift_rate_hz_s: float = 0.0
    frequency_mhz: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


def taylor_tree(data, xp=np):
    """Sum along every drift d in [0, T-1] channels over T rows (T must be a power of 2).

    Returns array [d, f]: sum of path starting at channel f with total drift d.
    Paths that leave the band on the right treat the missing samples as zero
    (data is assumed to be zero-mean noise after normalisation).
    """
    t, f = data.shape
    if t & (t - 1):
        raise ValueError("taylor_tree needs a power-of-two number of rows")
    cur = data[:, None, :]           # blocks of length L=1, each with 1 drift
    length = 1
    while length < t:
        a, b = cur[0::2], cur[1::2]  # (nb, L, F)
        nb = a.shape[0]
        new = xp.empty((nb, 2 * length, f), dtype=data.dtype)
        for d in range(2 * length):
            half = d // 2            # drift inside each half
            off = d - half           # second half starts this many channels later
            if off:
                new[:, d, :f - off] = a[:, half, :f - off] + b[:, half, off:]
                new[:, d, f - off:] = a[:, half, f - off:]
            else:
                new[:, d, :] = a[:, half, :] + b[:, half, :]
        cur = new
        length *= 2
    return cur[0]


def _sheared_view(padded, k: int, width: int, xp=np):
    """Zero-copy view where row r starts k*r channels later (shear for drift k ch/step)."""
    if xp is np:
        s0, s1 = padded.strides
        return np.lib.stride_tricks.as_strided(padded, shape=(padded.shape[0], width),
                                               strides=(s0 + k * s1, s1), writeable=False)
    out = xp.empty((padded.shape[0], width), dtype=padded.dtype)
    for row in range(padded.shape[0]):
        out[row] = padded[row, k * row: k * row + width]
    return out


def _search_one_sign(z, k_max: int, xp=np, k_limit_ch: float | None = None):
    """Best path sum / drift(ch per step) per start channel for non-negative drifts.

    `k_limit_ch` caps the drift in channels per step. Rows beyond it are skipped, so the
    search cannot report a drift rate the caller explicitly excluded. Without this, the
    `for d in range(tp)` sweep always reaches drift 1 ch/step even when `k_max == 0`, which
    on a coarse product is thousands of Hz/s past the configured ceiling (backlog B9).
    """
    t_real, f = z.shape
    tp = 1 << max(1, int(np.ceil(np.log2(max(t_real, 2)))))
    span = tp - 1
    if k_limit_ch is not None and k_limit_ch < 1.0 / span:
        # Only zero drift is in range, so the column sum is the entire answer and the tree
        # is pure overhead. `tree[0]` is exactly this sum.
        return xp.asarray(z.sum(axis=0), dtype=xp.float32), xp.zeros(f, dtype=xp.float32)
    pad = (k_max + 1) * span + 1
    padded = xp.zeros((tp, f + pad), dtype=xp.float32)
    padded[:t_real, :f] = z
    best = xp.full(f, -xp.inf, dtype=xp.float32)
    best_drift = xp.zeros(f, dtype=xp.float32)
    mask = xp.empty(f, dtype=bool)
    for k in range(k_max + 1):
        tree = taylor_tree(_sheared_view(padded, k, f + span + 1, xp), xp)
        for d in range(tp):
            if k_limit_ch is not None and (k + d / span) > k_limit_ch + 1e-9:
                continue
            row = tree[d, :f]
            xp.greater(row, best, out=mask)
            xp.copyto(best, row, where=mask)
            xp.copyto(best_drift, xp.float32((k * span + d) / span), where=mask)
    return best, best_drift


def _noise_scale(z, xp=np) -> tuple[float, float]:
    """Robust mean/sigma of a zero-drift path sum (one estimate for all drifts)."""
    col = z.sum(axis=0)[::3]
    med = float(xp.median(col))
    mad = float(xp.median(xp.abs(col - med))) * MAD_TO_SIGMA
    return med, (mad if mad > 0 else float(np.sqrt(z.shape[0])))


def drift_search(z: np.ndarray, tsamp: float = 1.0, foff_mhz: float = 1e-6,
                 max_drift_hz_s: float = 4.0, max_ch_per_step: int | None = None,
                 snr_threshold: float = 10.0, max_hits: int = 500,
                 freqs_mhz: np.ndarray | None = None, use_gpu: bool = False):
    """Search normalised data [time, chan] for linearly drifting narrowband signals.

    Returns (hits, snr_map) where snr_map is the best SNR per start channel
    (useful for the dashboard power plot).
    """
    xp = np
    if use_gpu:
        try:
            import cupy as xp  # type: ignore
        except ImportError:
            xp = np
    t, f = z.shape
    foff_hz = foff_mhz * 1e6
    k_needed = abs(max_drift_hz_s * tsamp / foff_hz) if foff_hz else 1
    k_max = max(0, int(np.ceil(k_needed)) - 1)
    if max_ch_per_step is not None:
        k_max = min(k_max, max(0, max_ch_per_step - 1))
    k_max = min(k_max, max(0, f // max(t, 1) - 1))

    zx = xp.asarray(z, dtype=xp.float32)
    med, scale = _noise_scale(zx, xp)
    pos_snr, pos_drift = _search_one_sign(zx, k_max, xp, k_needed)
    neg_snr, neg_drift = _search_one_sign(xp.ascontiguousarray(zx[:, ::-1]), k_max, xp, k_needed)
    pos_snr = (pos_snr - med) / scale
    neg_snr = (neg_snr - med) / scale
    neg_snr, neg_drift = neg_snr[::-1], -neg_drift[::-1]
    # Reversed search indexes the path by its start in the reversed band, which is the
    # same physical channel: element f' of the reversed array is channel F-1-f'.
    use_neg = neg_snr > pos_snr
    snr = xp.where(use_neg, neg_snr, pos_snr)
    drift = xp.where(use_neg, neg_drift, pos_drift)
    if xp is not np:
        snr, drift = xp.asnumpy(snr), xp.asnumpy(drift)
    snr = np.asarray(snr, dtype=np.float32)
    drift = np.asarray(drift, dtype=np.float32)

    # Non-maximum suppression across the channels a drifting track can touch.
    win = int(min(f, max(8, (k_max + 1) * (t - 1) + 4)))
    peaks = (snr >= snr_threshold) & (snr == maximum_filter1d(snr, size=2 * win + 1,
                                                              mode="nearest"))
    idx = np.flatnonzero(peaks)
    if idx.size > max_hits:
        idx = idx[np.argpartition(snr[idx], -max_hits)[-max_hits:]]
    idx = idx[np.argsort(-snr[idx])]
    hits = []
    for c in idx:
        d = float(drift[c])
        hits.append(Hit(
            start_channel=int(c), drift_ch_per_step=d, snr=float(snr[c]),
            drift_rate_hz_s=d * foff_hz / tsamp if tsamp else float("nan"),
            frequency_mhz=float(freqs_mhz[c]) if freqs_mhz is not None else float("nan"),
        ))
    return hits, snr
