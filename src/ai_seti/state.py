"""A SETI@home-style ledger: what has been crunched, lifetime stats, best signal ever.

Lets `ai-seti crunch` resume after a restart and never re-process an observation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3    # a file that fails this often is skipped until the ledger is reset
SIGNALS_PER_OBS = 2000    # signals remembered per observation; weak comb tones matter (B21)
# ponytail: flat list, capped; a sorted on-disk index if lifetime runs outgrow it.
MAX_SIGNALS = 50_000


@dataclass
class Ledger:
    path: Path
    observations_done: list[str] = field(default_factory=list)
    work_units: int = 0
    channels: int = 0
    cpu_seconds: float = 0.0
    hits: int = 0
    best: dict = field(default_factory=dict)
    shared: dict = field(default_factory=dict)      # finding_id -> [destinations]
    progress: dict = field(default_factory=dict)    # observation -> next unsearched channel
    failed: dict = field(default_factory=dict)      # observation -> {"attempts", "reason"}
    signals: list = field(default_factory=list)     # [frequency_mhz, target] seen so far
    sah_status: dict = field(default_factory=dict)  # last SETI@home probe {"at", "result"}
    started_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @classmethod
    def load(cls, path: Path) -> Ledger:
        path = Path(path)
        if path.is_file():
            raw = json.loads(path.read_text(encoding="utf-8"))
            # Tolerate a ledger written by another version instead of dying on an
            # unexpected keyword: unknown keys are dropped, known ones keep their values.
            known = {f.name for f in fields(cls)} - {"path"}
            dropped = sorted(set(raw) - known)
            if dropped:
                logger.warning("Ignoring unknown ledger keys in %s: %s", path, ", ".join(dropped))
            return cls(path=path, **{k: v for k, v in raw.items() if k in known})
        return cls(path=path)

    def record_unit(self, result: dict) -> None:
        self.work_units += 1
        self.channels += int(result.get("n_channels", 0))
        self.cpu_seconds += float(sum(result.get("timings", {}).values()))
        self.hits += len(result.get("hits", []))

    def record_best(self, row: dict, scoring_version: int | None = None) -> bool:
        if row and row.get("interest", -1) > self.best.get("interest", -1):
            self.best = {k: v for k, v in row.items() if not str(k).startswith("_")}
            if scoring_version is not None:
                self.best["scoring_version"] = scoring_version
            return True
        return False

    def drop_stale_best(self, scoring_version: int) -> dict | None:
        """Forget a best signal scored under other rules; returns it so the caller can say so."""
        if self.best and self.best.get("scoring_version") != scoring_version:
            old, self.best = self.best, {}
            return old
        return None

    def done(self, obs: str) -> None:
        if obs not in self.observations_done:
            self.observations_done.append(obs)

    def seti_at_home_status(self, probe, max_age_h: float = 24.0) -> dict:
        """The SETI@home probe result, reusing one up to `max_age_h` old (backlog B43).

        The probe only feeds a status line, but an unreachable server cost every `crunch` start
        its full timeout. `probe` is called only when the cached answer is missing or stale.
        """
        at = self.sah_status.get("at")
        if at and (datetime.now(UTC) - datetime.fromisoformat(at)).total_seconds() < max_age_h * 3600:
            return self.sah_status["result"]
        result = probe()
        self.sah_status = {"at": datetime.now(UTC).isoformat(), "result": result}
        return result

    # --- unit results per (scan, channel range), so no range is downloaded twice (B41) ---
    CACHE_KEYS = ("snr_threshold", "max_drift_rate_hz_s", "max_drift_ch_per_step",
                  "channels_per_unit", "bandpass_block", "remove_dc_spike",
                  "fine_channels_per_coarse", "use_ai", "model_path", "spike_threshold",
                  "pulse_threshold", "max_dm", "n_dm_trials", "max_hits_per_unit")

    def _results_file(self, url: str, rng, cfg, version: str) -> Path:
        key = json.dumps([url, list(rng or ()), version,
                          {k: getattr(cfg, k) for k in self.CACHE_KEYS}], default=str)
        name = hashlib.sha1(key.encode()).hexdigest()[:20] + ".pkl"
        return self.path.parent / "cache" / "units" / name

    def cached_results(self, url: str, rng, cfg, version: str) -> list | None:
        """Work-unit results of this scan's channel range from an earlier search, or None.

        `version` must change whenever the results would (code or model), see
        `pipeline.results_version`. The cache is a local file this program wrote itself.
        """
        import pickle
        f = self._results_file(url, rng, cfg, version)
        if not f.exists():
            return None
        try:
            return pickle.loads(f.read_bytes())
        except Exception:   # truncated or from another version: search again
            return None

    def store_results(self, url: str, rng, cfg, version: str, results: list) -> None:
        import pickle
        f = self._results_file(url, rng, cfg, version)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(results))
        os.replace(tmp, f)

    def remember_signals(self, freqs_mhz, target: str) -> None:
        """Keep the run's strongest frequencies so a later target can recognise them."""
        self.signals += [[float(f), target] for f in list(freqs_mhz)[:SIGNALS_PER_OBS]]
        self.signals = self.signals[-MAX_SIGNALS:]

    def fail(self, obs: str, reason: str, permanent: bool = False) -> None:
        """Record a failed attempt; `permanent` gives up at once (retrying cannot help)."""
        entry = self.failed.setdefault(obs, {"attempts": 0, "reason": ""})
        entry["attempts"] = MAX_ATTEMPTS if permanent else entry["attempts"] + 1
        entry["reason"] = reason

    def skip(self) -> set[str]:
        """Observations not to fetch again: finished ones and ones that keep failing."""
        gave_up = {k for k, v in self.failed.items() if v.get("attempts", 0) >= MAX_ATTEMPTS}
        return set(self.observations_done) | gave_up

    def next_range(self, obs: str, nchans: int, max_chans: int = 0) -> tuple[int, int]:
        """Channels to search this run: resume where a partial (--max-units) run stopped."""
        start = int(self.progress.get(obs, 0))
        return start, min(nchans, start + max_chans) if max_chans else nchans

    def advance(self, obs: str, stop: int, nchans: int) -> None:
        """Record that channels up to `stop` are searched; done only once all of them are."""
        self.failed.pop(obs, None)
        if stop >= nchans:
            self.progress.pop(obs, None)
            self.done(obs)
        else:
            self.progress[obs] = stop

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data.pop("path")
        # Unique temp name: two crunchers sharing one ledger would otherwise race on a
        # single fixed ".tmp" and could publish a truncated file.
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
