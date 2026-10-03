"""A SETI@home-style ledger: what has been crunched, lifetime stats, best signal ever.

Lets `ai-seti crunch` resume after a restart and never re-process an observation.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


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

    def record_best(self, row: dict) -> bool:
        if row and row.get("interest", -1) > self.best.get("interest", -1):
            self.best = {k: v for k, v in row.items() if not str(k).startswith("_")}
            return True
        return False

    def done(self, obs: str) -> None:
        if obs not in self.observations_done:
            self.observations_done.append(obs)

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
