"""Radio-frequency-interference rejection.

The single most powerful SETI filter is not a neural net — it's the sky. Breakthrough
Listen observes in ABACAD cadences (target ON, three different OFF positions). A real
signal from the target appears in every ON scan, follows its drift between scans,
and is absent from every OFF. Terrestrial interference enters the sidelobes and shows
up everywhere. `cadence_filter` implements that logic (as turboSETI's find_event does).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def in_known_rfi_band(freq_mhz: np.ndarray, bands) -> np.ndarray:
    f = np.asarray(freq_mhz, dtype=float)
    out = np.zeros(f.shape, dtype=bool)
    for lo, hi in bands:
        out |= (f >= lo) & (f <= hi)
    return out


# Below this strong/weak SNR ratio a pair is symmetric: both halves are the instrument's,
# neither is a sky tone with its echo. Real BL pairs measured 1.00-1.10 (backlog B18).
SYMMETRIC_PAIR_RATIO = 2.0


def flag_mirror_images(df: pd.DataFrame, tol_ch: int = 3,
                       symmetric_ratio: float = SYMMETRIC_PAIR_RATIO) -> pd.DataFrame:
    """Flag spectral images mirrored about a coarse-channel DC bin (backlog B6).

    GBT/BL coarse channels show a weaker copy of a strong tone at ``2*f_c - f`` with the
    opposite drift. ``mirror_channel`` (set per hit by the pipeline, -1 when the coarse
    channelisation is unknown) is where that partner would start. Within one file, both hits
    of a matching pair get ``mirror_pair``. ``mirror_image`` goes to the weaker one, or to both
    when their SNRs are within ``symmetric_ratio`` of each other (B18).
    """
    df["mirror_pair"] = False
    df["mirror_image"] = False
    if "mirror_channel" not in df or df.empty:
        return df
    stem = df["unit_id"].astype(str).str.rsplit(":", n=1).str[0]
    for _, g in df.groupby(stem):
        order = g["channel"].to_numpy().argsort()
        chans = g["channel"].to_numpy()[order]
        idx = g.index.to_numpy()[order]
        for i, mc, d, snr in zip(g.index, g["mirror_channel"], g["drift_ch_per_step"], g["snr"],
                                 strict=True):
            if mc < 0:
                continue
            lo = int(np.searchsorted(chans, mc - tol_ch, side="left"))
            hi = int(np.searchsorted(chans, mc + tol_ch, side="right"))
            for j in idx[lo:hi]:
                dj = df.at[j, "drift_ch_per_step"]
                if j == i or abs(d + dj) > max(0.25, 0.1 * abs(d)):
                    continue
                df.loc[[i, j], "mirror_pair"] = True
                snr_j = df.at[j, "snr"]
                if max(snr, snr_j) < symmetric_ratio * min(snr, snr_j):
                    df.loc[[i, j], "mirror_image"] = True
                else:
                    df.at[j if (snr_j, j) < (snr, i) else i, "mirror_image"] = True
    return df


def flag_multi_target(df: pd.DataFrame, seen: list, target: str, tol_khz: float = 2.0,
                      band_khz: float = 25.0, band_hits: int = 2) -> pd.DataFrame:
    """Flag hits seen, or sitting in an interference-busy band, at a *different* target.

    A signal present in several pointings cannot come from any one of them; it is the
    strongest interference evidence there is. `seen` is the ledger's [freq_mhz, target] list.
    A hit is flagged when another target had a signal within `tol_khz` (the same tone, B7), or
    at least `band_hits` signals within ±`band_khz` (the same comb or forest, whose tones
    land a few kHz apart from scan to scan, B21). Flagged hits get `multi_target` and a fifth
    of their interest, then the frame is re-ranked.
    """
    df["multi_target"] = False
    other = np.array(sorted(f for f, t in seen if t != target), dtype=float)
    if df.empty or not len(other):
        return df
    f = df["frequency_mhz"].to_numpy(dtype=float)

    def within(khz: float) -> np.ndarray:
        lo = np.searchsorted(other, f - khz / 1e3, side="left")
        return np.searchsorted(other, f + khz / 1e3, side="right") - lo

    df["multi_target"] = (within(tol_khz) > 0) | (within(band_khz) >= band_hits)
    df["interest"] = np.where(df["multi_target"], (df["interest"] * 0.2).round(1), df["interest"])
    return df.sort_values("interest", ascending=False).reset_index(drop=True)


def apply_cadence(df: pd.DataFrame, status: str, events: pd.DataFrame | None = None,
                  tol_hz: float = 50.0) -> pd.DataFrame:
    """Write the ON/OFF result onto each hit as `cadence` (backlog B7).

    `status` is "tested" (use `events` from `cadence_filter`), "untestable" (this scan is an
    OFF pointing, or there aren't enough scans) or "not_run". A tested hit that is not among
    the events failed: it showed up when pointing away, or not in every ON scan. That is the
    strongest interference evidence there is, so it keeps a fifth of its interest.
    """
    if status != "tested":
        df["cadence"] = status
        return df
    ev = (events["frequency_mhz"].to_numpy(dtype=float) if events is not None and not events.empty
          else np.empty(0))
    f = df["frequency_mhz"].to_numpy(dtype=float)
    passed = (np.abs(f[:, None] - ev[None, :]) <= tol_hz / 1e6).any(axis=1) if ev.size else \
        np.zeros(len(df), dtype=bool)
    df["cadence"] = np.where(passed, "passed", "failed")
    df["interest"] = np.where(passed, df["interest"], (df["interest"] * 0.2).round(1))
    return df.sort_values("interest", ascending=False).reset_index(drop=True)


def is_off_scan(name: str) -> bool:
    return "_OFF" in name.upper()


def cadence_is_testable(scans: list[dict], min_on_scans: int = 2) -> tuple[bool, str]:
    """Whether these scans can actually falsify a signal, and why not if they cannot.

    A single ON scan with no OFF scan confirms nothing: the reference hit matches itself,
    giving on_found/len(ons) == 1.0 and off_found == 0, so an unconditional pass. Since the
    cadence filter is the strongest filter this tool has, it must refuse that case rather
    than hand out a free "passed".
    """
    ons = [s for s in scans if s["is_on"]]
    offs = [s for s in scans if not s["is_on"]]
    if len(ons) < min_on_scans:
        return False, f"only {len(ons)} ON scan(s); a cadence test needs >= {min_on_scans}"
    if not offs:
        return False, "no OFF scan provided; pointing-away data is what rejects RFI"
    return True, f"{len(ons)} ON / {len(offs)} OFF scans"


def cadence_filter(scans: list[dict], min_on_fraction: float = 1.0,
                   freq_tol_hz: float = 20.0, min_on_scans: int = 2,
                   reference: str | None = None) -> pd.DataFrame:
    """scans: [{'name', 'mjd', 'is_on', 'hits': DataFrame(frequency_mhz, drift_rate_hz_s, snr)}].

    Returns events from the first ON scan that (a) re-appear along the extrapolated
    drift in >= min_on_fraction of ON scans and (b) never appear in an OFF scan.

    Returns an empty frame unless `cadence_is_testable` is satisfied, so a single
    observation can never be reported as a cadence pass.
    """
    scans = sorted(scans, key=lambda s: s["mjd"])
    testable, reason = cadence_is_testable(scans, min_on_scans)
    if not testable:
        logger.info("Cadence filter skipped: %s", reason)
        return pd.DataFrame()
    ons = [s for s in scans if s["is_on"]]
    if not ons or ons[0]["hits"].empty:
        return pd.DataFrame()
    # Events are this scan's hits; by default the first ON scan, or the one named `reference`.
    ref = next((s for s in ons if s["name"] == reference), ons[0])
    if ref["hits"].empty:
        return pd.DataFrame()
    t0 = ref["mjd"]
    events = []
    for _, hit in ref["hits"].iterrows():
        rate = float(hit["drift_rate_hz_s"])
        on_found, off_found, matches = 1, 0, []
        for s in scans:
            if s is ref:
                continue
            dt = (s["mjd"] - t0) * 86400.0
            pred = float(hit["frequency_mhz"]) + rate * dt / 1e6
            tol = (freq_tol_hz + abs(rate) * 60.0) / 1e6
            h = s["hits"]
            if h.empty:
                continue
            m = h[np.abs(h["frequency_mhz"] - pred) <= tol]
            if s["is_on"]:
                m = m[np.abs(m["drift_rate_hz_s"] - rate) <= max(0.2, 0.5 * abs(rate))]
                if len(m):
                    on_found += 1
                    matches.append(s["name"])
            else:
                if len(m):
                    off_found += 1
        frac = on_found / len(ons)
        if frac >= min_on_fraction and off_found == 0:
            events.append({**hit.to_dict(), "on_scans_matched": on_found,
                           "n_on_scans": len(ons), "off_scans_checked": len(scans) - len(ons),
                           "matched_in": ";".join(matches),
                           "status": "passed_cadence"})
    return pd.DataFrame(events)
