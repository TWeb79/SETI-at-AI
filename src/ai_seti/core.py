from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class AnalysisMetadata:
    source: str
    sha256: str
    processed_utc: str
    n_time: int
    n_frequency: int
    frequency_start_mhz: float | None
    frequency_stop_mhz: float | None
    detector: str
    threshold: float


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def make_synthetic_waterfall(
    n_time: int = 256,
    n_frequency: int = 1024,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a noise-only waterfall with two injected narrowband test signals."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n_time, n_frequency)).astype(np.float32)
    # Persistent narrowband feature.
    # (v0.1 hard-coded channels 310/700 and crashed for n_frequency < 701.)
    data[:, int(n_frequency * 310 / 1024)] += 7.0
    # A weak line drifting by one channel every 32 integrations.
    base = int(n_frequency * 700 / 1024)
    for t in range(n_time):
        channel = min(base + t // 32, n_frequency - 1)
        data[t, channel] += 5.0
    frequencies_mhz = np.linspace(1400.0, 1401.0, n_frequency, endpoint=False)
    return data, frequencies_mhz


def robust_candidates(
    data: np.ndarray,
    frequencies_mhz: np.ndarray | None = None,
    threshold: float = 6.0,
) -> pd.DataFrame:
    """
    LEGACY v0.1 detector, kept as a benchmark baseline.

    Flags positive outliers in the time-median spectrum using median/MAD. It cannot
    see signals that drift by more than ~1 channel over the observation, and the
    bandpass shape biases its noise estimate. See ai_seti.dsp.dedoppler.
    """
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = np.nanmean(arr, axis=2)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D [time, frequency] or 3-D data; got {arr.shape}")
    if arr.shape[0] < 2 or arr.shape[1] < 8:
        raise ValueError("Need at least 2 time samples and 8 frequency channels")

    spectrum = np.nanmedian(arr, axis=0)
    center = np.nanmedian(spectrum)
    mad = np.nanmedian(np.abs(spectrum - center))
    # Gaussian-consistent robust sigma. Fall back to standard deviation if MAD is zero.
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = float(np.nanstd(spectrum))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = np.finfo(float).eps

    scores = (spectrum - center) / robust_sigma
    indices = np.flatnonzero(scores >= threshold)
    frame = pd.DataFrame({
        "channel_index": indices.astype(int),
        "median_intensity": spectrum[indices].astype(float),
        "robust_z": scores[indices].astype(float),
        "frequency_mhz": (
            frequencies_mhz[indices].astype(float)
            if frequencies_mhz is not None else np.full(len(indices), np.nan)
        ),
        "candidate_type": "persistent_narrowband_outlier",
        "status": "unverified",
    })
    if not frame.empty:
        frame = frame.sort_values("robust_z", ascending=False).reset_index(drop=True)
    return frame


def plot_waterfall(
    data: np.ndarray,
    output: Path,
    frequencies_mhz: np.ndarray | None = None,
    title: str = "AI-SETI waterfall",
) -> None:
    arr = np.asarray(data)
    if arr.ndim == 3:
        arr = np.nanmean(arr, axis=2)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    extent = None
    if frequencies_mhz is not None and len(frequencies_mhz) == arr.shape[1]:
        extent = (float(frequencies_mhz[0]), float(frequencies_mhz[-1]),
                  float(arr.shape[0]), 0.0)
    image = ax.imshow(arr, aspect="auto", origin="upper", interpolation="nearest",
                      extent=extent, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Frequency channel" if extent is None else "Frequency (MHz)")
    ax.set_ylabel("Time integration")
    fig.colorbar(image, ax=ax, label="Intensity (arbitrary units)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_analysis(
    data: np.ndarray,
    frequencies_mhz: np.ndarray | None,
    source: str,
    output_dir: Path,
    threshold: float = 6.0,
    file_hash: str = "synthetic",
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = robust_candidates(data, frequencies_mhz, threshold)
    candidates.to_csv(output_dir / "candidates.csv", index=False)
    plot_waterfall(data, output_dir / "waterfall.png", frequencies_mhz, f"AI-SETI: {source}")

    meta = AnalysisMetadata(
        source=source,
        sha256=file_hash,
        processed_utc=datetime.now(UTC).isoformat(),
        n_time=int(data.shape[0]),
        n_frequency=int(data.shape[1]),
        frequency_start_mhz=float(frequencies_mhz[0]) if frequencies_mhz is not None else None,
        frequency_stop_mhz=float(frequencies_mhz[-1]) if frequencies_mhz is not None else None,
        detector="time-median spectrum + median/MAD robust outlier threshold",
        threshold=float(threshold),
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(asdict(meta), indent=2), encoding="utf-8"
    )
    return candidates


def load_filterbank(
    path: Path,
    max_load_mb: int = 256,
    f_start_mhz: float | None = None,
    f_stop_mhz: float | None = None,
):
    """Read a bounded .fil/.h5 selection; returns data [time, chan], freq axis, header dict.

    v0.2: uses the built-in memory-mapped reader (no blimpy needed). Kept for
    backwards compatibility with v0.1 scripts and the Streamlit app.
    """
    from .io.filterbank import read_header, read_window

    header = read_header(Path(path))
    c0, c1 = header.channel_range(f_start_mhz, f_stop_mhz)
    budget = int(max_load_mb * 1024 * 1024 // (4 * max(header.nsamples, 1)))
    if c1 - c0 > budget:
        raise ValueError(
            f"Selection needs {(c1 - c0) * header.nsamples * 4 / 2**20:.0f} MB > max_load_mb="
            f"{max_load_mb}. Narrow --f-start/--f-stop or use `ai-seti analyze` which streams "
            "work units.")
    data = read_window(Path(path), c0, c1, header)
    return data, header.frequencies(c0, c1), header.to_dict()
