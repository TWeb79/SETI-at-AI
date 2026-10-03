"""Secondary detectors in the SETI@home tradition.

* Spikes   – single (time, channel) power excursions, SETI@home's simplest signal type.
* Pulses   – broadband, dispersed bursts (the Astropulse idea): incoherent
             sub-band dedispersion + boxcar matched filters. Only meaningful for
             products with many time samples (e.g. BL mid/high-time-resolution);
             skipped automatically on 16-sample high-frequency-resolution data.
"""
from __future__ import annotations

import numpy as np

MAD_TO_SIGMA = 1.4826
KDM = 4.148808e3  # s MHz^2 pc^-1 cm^3


def find_spikes(z: np.ndarray, threshold: float = 9.0, max_spikes: int = 200,
                freqs_mhz: np.ndarray | None = None) -> list[dict]:
    flat = z.ravel()
    idx = np.flatnonzero(flat >= threshold)
    if idx.size > max_spikes:
        idx = idx[np.argpartition(flat[idx], -max_spikes)[-max_spikes:]]
    idx = idx[np.argsort(-flat[idx])]
    out = []
    for i in idx:
        t, c = divmod(int(i), z.shape[1])
        out.append({"time_index": t, "channel": c, "power_sigma": float(flat[i]),
                    "frequency_mhz": float(freqs_mhz[c]) if freqs_mhz is not None
                    else float("nan")})
    return out


def find_pulses(z: np.ndarray, freqs_mhz: np.ndarray, tsamp: float,
                threshold: float = 8.0, max_dm: float = 1000.0, n_dm: int = 64,
                widths=(1, 2, 4, 8, 16), nsub: int = 64, min_samples: int = 32,
                max_pulses: int = 50) -> list[dict]:
    t, f = z.shape
    if t < min_samples:
        return []
    nsub = max(1, min(nsub, f))
    usable = (f // nsub) * nsub
    sub = z[:, :usable].reshape(t, nsub, -1).mean(axis=2)
    sub_freq = freqs_mhz[:usable].reshape(nsub, -1).mean(axis=1)
    med = np.median(sub, axis=0)
    sig = np.median(np.abs(sub - med), axis=0) * MAD_TO_SIGMA
    sub = (sub - med) / np.where(sig > 0, sig, 1.0)

    f_ref = np.max(sub_freq)
    delay_per_dm = KDM * (sub_freq ** -2 - f_ref ** -2) / tsamp   # samples per unit DM
    max_useful_dm = (t / 2) / np.maximum(np.max(delay_per_dm), 1e-12)
    dms = np.linspace(0.0, min(max_dm, max_useful_dm), n_dm if max_useful_dm > 1 else 1)

    found: dict[int, dict] = {}
    for dm in dms:
        shifts = np.round(delay_per_dm * dm).astype(int)
        series = np.zeros(t, dtype=np.float64)
        for s, col in zip(shifts, sub.T, strict=True):
            if s == 0:
                series += col
            elif s < t:
                series[:t - s] += col[s:]
        series /= np.sqrt(nsub)
        cs = np.concatenate(([0.0], np.cumsum(series)))
        for w in widths:
            if w > t:
                break
            box = (cs[w:] - cs[:-w]) / np.sqrt(w)
            m = np.median(box)
            sd = np.median(np.abs(box - m)) * MAD_TO_SIGMA or 1.0
            snr = (box - m) / sd
            i = int(np.argmax(snr))
            if snr[i] >= threshold:
                key = i // max(w, 4)
                if key not in found or snr[i] > found[key]["snr"]:
                    found[key] = {"time_index": i, "width_samples": int(w), "dm": float(dm),
                                  "snr": float(snr[i]), "time_s": float(i * tsamp)}
    pulses = sorted(found.values(), key=lambda p: -p["snr"])[:max_pulses]
    for p in pulses:
        p["likely_rfi"] = p["dm"] < 1.0   # undispersed → terrestrial
    return pulses
