"""The cruncher: process work units in parallel and rank what they find.

Per work unit (runs in a worker process, one BLAS thread each):
  load window (memmap / HDF5 chunk / HTTP range)  ->  bandpass + DC repair
  ->  Taylor-tree de-Doppler (+ spikes, + dispersed pulses)  ->  AI features & class
Across the run (main process):
  anomaly detection over all hits  ->  RFI flags  ->  0-100 interest score.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .config import SearchConfig
from .sources import WorkUnit, load_unit

THUMB_COLS = 160


def _limit_threads():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _pool_cols(arr: np.ndarray, cols: int, fn=np.max) -> np.ndarray:
    n = arr.shape[-1]
    if n <= cols:
        return arr
    usable = (n // cols) * cols
    return fn(arr[..., :usable].reshape(*arr.shape[:-1], cols, -1), axis=-1)


def process_work_unit(wu: WorkUnit, cfg: SearchConfig) -> dict:
    from .ai.features import feature_vector, hit_features
    from .ai.model import LABELS, HitScorer
    from .dsp.dedoppler import drift_search, searched_drift_hz_s
    from .dsp.detectors import find_pulses, find_spikes
    from .dsp.preprocess import normalize
    from .sources import drift_resolvable

    timings = {}
    t = time.perf_counter()
    raw = load_unit(wu)
    timings["load"] = time.perf_counter() - t

    hdr = wu.header
    unresolved = not drift_resolvable(hdr, cfg)
    freqs = hdr.frequencies(wu.chan_start, wu.chan_stop)
    t = time.perf_counter()
    nfine = cfg.fine_channels_per_coarse or hdr.fine_per_coarse()
    z = normalize(raw, block=cfg.bandpass_block, chan_start=wu.chan_start,
                  fine_per_coarse=nfine, remove_dc=cfg.remove_dc_spike)
    timings["preprocess"] = time.perf_counter() - t

    t = time.perf_counter()
    # max_ch_per_step enforces SearchConfig.max_drift_ch_per_step, which the same config
    # used to size the guard band in sources.drift_padding. Both limits must agree or a
    # signal drifting near the edge of a core is searched for but has no data under it.
    hits, snr_map = drift_search(z, tsamp=hdr.tsamp, foff_mhz=hdr.foff,
                                 max_drift_hz_s=cfg.max_drift_rate_hz_s,
                                 max_ch_per_step=cfg.max_drift_ch_per_step,
                                 snr_threshold=cfg.snr_threshold,
                                 max_hits=cfg.max_hits_per_unit, freqs_mhz=freqs,
                                 use_gpu=cfg.use_gpu)
    timings["dedoppler"] = time.perf_counter() - t
    n_t, n_f = z.shape
    drift_searched = searched_drift_hz_s(n_t, n_f, hdr.tsamp, hdr.foff, cfg.max_drift_rate_hz_s,
                                         cfg.max_drift_ch_per_step)

    t = time.perf_counter()
    spikes = find_spikes(z, cfg.spike_threshold, freqs_mhz=freqs)
    pulses = find_pulses(z, freqs, hdr.tsamp, cfg.pulse_threshold, cfg.max_dm, cfg.n_dm_trials)
    timings["detectors"] = time.perf_counter() - t

    t = time.perf_counter()
    rows, vectors = [], []
    for h in hits:
        abs_start = wu.chan_start + h.start_channel
        if not (wu.core_start <= abs_start < wu.core_stop):
            continue                      # belongs to the neighbouring unit
        feats, img = hit_features(z, h.start_channel, h.drift_ch_per_step, h.snr)
        row = {"unit_id": wu.unit_id, "channel": abs_start, **h.to_dict(), **feats}
        row.pop("start_channel")
        # Start channel of this tone's spectral image about the coarse-channel DC bin (B6).
        dc = (abs_start // nfine) * nfine + nfine // 2 if nfine else None
        row["mirror_channel"] = 2 * dc - abs_start if dc is not None else -1
        row["drift_unresolved"] = unresolved
        rows.append(row)
        vectors.append(feature_vector(feats, img))
        row["_snippet"] = img.round(3).tolist()
    if cfg.use_ai and rows:
        probs = HitScorer(cfg.model_path).predict(np.vstack(vectors))
        for row, p in zip(rows, probs, strict=True):
            for label, val in zip(LABELS, p, strict=True):
                row[f"p_{label}"] = float(val)
    timings["ai"] = time.perf_counter() - t

    for s in spikes:
        s["channel"] = s["channel"] + wu.chan_start
    lo = wu.core_start - wu.chan_start
    hi = lo + (wu.core_stop - wu.core_start)
    return {
        "unit_id": wu.unit_id, "location": wu.location, "meta": wu.meta,
        "n_channels": wu.core_stop - wu.core_start, "n_time": int(z.shape[0]),
        "f_lo": float(min(freqs[lo], freqs[hi - 1])), "f_hi": float(max(freqs[lo], freqs[hi - 1])),
        "timings": timings, "hits": rows, "drift_searched_hz_s": drift_searched,
        "spikes": [s for s in spikes if wu.core_start <= s["channel"] < wu.core_stop][:50],
        "pulses": pulses,
        "thumb": _pool_cols(z[:, lo:hi], THUMB_COLS).round(2).tolist(),
        "snr_profile": _pool_cols(snr_map[lo:hi], THUMB_COLS).round(2).tolist(),
        "pid": os.getpid(),
    }


def _failed_result(wu: WorkUnit, exc: BaseException) -> dict:
    return {"unit_id": wu.unit_id, "error": repr(exc), "hits": [], "spikes": [],
            "pulses": [], "timings": {}, "n_channels": 0, "meta": wu.meta}


def _worker_init():
    _limit_threads()


def run_units(units: list[WorkUnit], cfg: SearchConfig, on_result=None,
              workers: int | None = None):
    """Process units in parallel; yields results as they complete (for live UIs)."""
    n = workers or cfg.workers or os.cpu_count() or 1
    n = max(1, min(n, len(units)))
    if n == 1:
        _limit_threads()
        for wu in units:
            try:
                res = process_work_unit(wu, cfg)
            except Exception as exc:    # same contract as the pool path below
                res = _failed_result(wu, exc)
            if on_result:
                on_result(res)
            yield res
        return
    with ProcessPoolExecutor(max_workers=n, initializer=_worker_init) as pool:
        futures = {pool.submit(process_work_unit, wu, cfg): wu for wu in units}
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as exc:
                res = _failed_result(futures[fut], exc)
            if on_result:
                on_result(res)
            yield res


# Bump when the interest score changes meaning (new flags, penalties, caps), so a lifetime
# best recorded under older rules is not shown as if it were comparable (backlog B27).
SCORING_VERSION = 2


def score_candidates(results: list[dict], cfg: SearchConfig) -> pd.DataFrame:
    """Merge hits from all units, add anomaly + RFI flags, compute interest score."""
    from .ai.features import SCALAR_FEATURES
    from .ai.model import anomaly_scores
    from .rfi import flag_mirror_images, in_known_rfi_band

    rows = [h for r in results for h in r.get("hits", [])]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["anomaly"] = anomaly_scores(df[SCALAR_FEATURES].to_numpy(), cfg.anomaly_contamination)
    df["known_rfi_band"] = in_known_rfi_band(df["frequency_mhz"], cfg.known_rfi_bands_mhz)
    df = flag_mirror_images(df)

    thr = cfg.snr_threshold
    s_snr = np.clip(1 - np.exp(-(df["snr"] - thr) / thr), 0, 1)
    heuristic = ((df["zero_drift"] == 0) & (df["bandwidth_ch"] <= 4)
                 & (df["on_fraction"] >= 0.5)).astype(float) * 0.7
    from .ai.model import HitScorer, ood_snr_limit
    # Far above the training SNRs the classifier saturates (B8): don't trust p there. The
    # interest cap below applies in every mode, the heuristic is no better at these SNRs.
    ood = df["snr"] > ood_snr_limit(HitScorer(cfg.model_path).meta if cfg.use_ai else {})
    if "p_technosignature_like" in df and df["p_technosignature_like"].notna().any():
        p_et = df["p_technosignature_like"].fillna(0.0).where(~ood, heuristic)
    else:   # heuristic fallback when no model is available
        p_et = heuristic
    anomaly = df["anomaly"].fillna(0.5)
    score = 0.55 * p_et + 0.20 * s_snr + 0.25 * anomaly
    # "Doesn't drift" is evidence of an Earth source only where drift could have been seen (B19).
    unresolved = df.get("drift_unresolved", pd.Series(False, index=df.index)).fillna(False)
    score *= np.where((df["zero_drift"] > 0) & ~unresolved.astype(bool), 0.3, 1.0)
    score *= np.where(df["known_rfi_band"], 0.5, 1.0)
    score *= np.where(df["bandwidth_ch"] > 8, 0.4, 1.0)
    score *= np.where(df["mirror_image"], 0.2, 1.0)
    # Strong out-of-range carriers are usually terrestrial: demote them like a non-drifting
    # tone so one never heads a quiet list (B28), and cap them until a cadence clears them.
    score *= np.where(ood, 0.3, 1.0)
    df["interest"] = (100 * score).round(1)
    df["interest"] = df["interest"].where(~ood, df["interest"].clip(upper=50.0))

    prob_cols = [c for c in df.columns if c.startswith("p_")]
    if prob_cols and df[prob_cols].notna().any().any():
        df["ai_class"] = df[prob_cols].idxmax(axis=1).str[2:]
        df.loc[ood, "ai_class"] = "out_of_distribution"
    elif not cfg.use_ai:
        df["ai_class"] = "ai_disabled"
    else:
        # Name the cause rather than a bland "unscored": a missing or version-incompatible
        # model file is an actionable setup problem, not a property of the signal.
        from .ai.model import DEFAULT_MODEL, HitScorer
        detail = HitScorer(cfg.model_path).load_error
        df["ai_class"] = "model_unavailable"
        df.attrs["ai_note"] = (
            f"hit classifier unavailable ({detail or f'missing {DEFAULT_MODEL}'}); "
            "class labels and interest fell back to the heuristic")
    df["status"] = "unverified"
    df = df.sort_values("interest", ascending=False).reset_index(drop=True)
    return df
