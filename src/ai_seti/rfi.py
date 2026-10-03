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
                   freq_tol_hz: float = 20.0, min_on_scans: int = 2) -> pd.DataFrame:
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
    ref = ons[0]
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
