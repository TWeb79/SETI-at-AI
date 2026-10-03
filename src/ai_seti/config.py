"""Search configuration shared by the CLI, the pipeline and the dashboard."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class SearchConfig:
    # --- work-unit splitting -------------------------------------------------
    channels_per_unit: int = 262_144        # ~0.73 MHz at 2.79 Hz resolution
    max_drift_rate_hz_s: float = 4.0        # Earth rotation+orbit stays well under this
    # Absolute ceiling on drift in channels per time step, bounding compute and the
    # false-alarm rate. It must not undercut max_drift_rate_hz_s for the data in hand:
    # a GBT coarse channel integrates ~18 s at ~2.79 Hz, so 4 Hz/s is already ~26 ch/step
    # and a cap of 4 would throw away exactly the signals the rate limit allows. Lower it
    # to trade sensitivity for speed; the guard band in sources.drift_padding follows it.
    max_drift_ch_per_step: int = 32

    # --- preprocessing --------------------------------------------------------
    bandpass_block: int = 512               # channels per robust bandpass knot
    remove_dc_spike: bool = True
    fine_channels_per_coarse: int | None = None  # auto-detect from foff when None

    # --- detection thresholds -------------------------------------------------
    snr_threshold: float = 10.0             # dedoppler (turboSETI default is 10)
    spike_threshold: float = 9.0            # single-sample spikes (SETI@home "spikes")
    pulse_threshold: float = 8.0            # broadband dispersed pulses (Astropulse-like)
    max_hits_per_unit: int = 500
    max_dm: float = 1000.0
    n_dm_trials: int = 64

    # --- AI / scoring ----------------------------------------------------------
    use_ai: bool = True
    anomaly_contamination: float = 0.05
    model_path: str | None = None           # default: bundled model

    # --- interference seen at other targets (ledger, across runs) -----------------
    multi_target_tol_khz: float = 2.0       # the same tone at another star
    multi_target_band_khz: float = 25.0     # ...or a busy band there:
    multi_target_band_hits: int = 2         # this many signals within ±band_khz

    # --- sharing findings back ------------------------------------------------
    share_to: list[str] = field(default_factory=lambda: ["bundle"])  # bundle | github | webhook
    share_min_interest: float = 70.0
    share_top: int = 3
    share_require_cadence: bool | None = None   # None: required for github/webhook only
    github_repo: str | None = None          # e.g. "your-org/ai-seti-findings"
    webhook_url: str | None = None
    webhook_format: str = "json"            # json | slack | discord
    reporter_handle: str | None = None      # opt-in; nothing personal is sent otherwise

    # --- execution --------------------------------------------------------------
    workers: int = 0                        # 0 = all CPU cores
    use_gpu: bool = False                   # experimental, needs cupy
    max_download_mb: int = 2048
    known_rfi_bands_mhz: list[tuple[float, float]] = field(default_factory=lambda: [
        (1164.0, 1189.0),   # GPS L5 / Galileo E5a
        (1215.0, 1240.0),   # GPS L2
        (1200.0, 1341.0),   # GBT L-band notch filter region
        (1559.0, 1591.0),   # GPS L1 / Galileo E1 / BeiDou B1
        (1597.0, 1607.0),   # GLONASS L1
        (1616.0, 1626.5),   # Iridium
        (1525.0, 1559.0),   # Inmarsat downlink
    ])

    @classmethod
    def load(cls, path: Path | None) -> SearchConfig:
        cfg = cls()
        if path is None or not Path(path).is_file():
            return cfg
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        names = {f.name for f in fields(cls)}
        for key, value in raw.items():
            if key in names:
                if key == "known_rfi_bands_mhz":
                    value = [tuple(v) for v in value]
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)
