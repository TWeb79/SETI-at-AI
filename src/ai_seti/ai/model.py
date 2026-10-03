"""AI layer: a supervised hit classifier + an unsupervised anomaly detector.

* Classifier — gradient-boosted trees on physical features + a 16x16 de-drifted
  snippet. Trained on simulated hits that went through the *real* detection
  pipeline, so it sees the same artefacts it will see in production. Classes:
  technosignature_like, four RFI morphologies, and noise.
* Anomaly — IsolationForest fitted per run on all hits: anything unlike the bulk of
  that observation's (mostly RFI) population gets a high score. This is how you find
  signals nobody thought to simulate.

CPU-only, no torch needed; trains in under a minute.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import ClassVar

import numpy as np

from .features import SCALAR_FEATURES, feature_vector, hit_features
from .simulate import CLASSES, inject, noise_waterfall, random_injection

LABELS = [*CLASSES, "noise"]
DEFAULT_MODEL = Path(__file__).with_name("hit_classifier.joblib")

logger = logging.getLogger(__name__)

# Highest hit SNR in the bundled model's training set (`ai-seti train` defaults: seed 0,
# 500 per class; reproduced 2026-10-03). Used for models whose metadata predates the
# recorded "snr_max". Above OOD_SNR_FACTOR x this, the trees only extrapolate flat, so
# p(technosignature_like) means nothing there (backlog B8).
TRAINED_SNR_MAX = 40.24
OOD_SNR_FACTOR = 3.0


def ood_snr_limit(meta: dict) -> float:
    """SNR above which a hit is outside what the classifier was trained on."""
    return OOD_SNR_FACTOR * float(meta.get("snr_max", TRAINED_SNR_MAX))


def _sklearn_version() -> str:
    """Installed scikit-learn version, or a placeholder if sklearn is unavailable."""
    try:
        import sklearn
        return sklearn.__version__
    except ImportError:
        return "not installed"


# Production drift range on GBT high-res data: 4 Hz/s is ~26 channels/step, capped at
# max_drift_ch_per_step = 32. Training only up to 6 left the classifier guessing (B3).
TRAIN_MAX_DRIFT = 32.0


def _training_example(kind: str, rng, n_time=16, n_chan=4096, max_drift=TRAIN_MAX_DRIFT):
    from ..dsp.dedoppler import drift_search
    from ..dsp.preprocess import normalize
    data = noise_waterfall(n_time, n_chan, rng, bandpass=True)
    inj = None
    if kind != "noise":
        inj = random_injection(kind, n_time, n_chan, rng, snr_range=(1.2, 8.0),
                               max_drift=max_drift)
        inject(data, inj, rng)
    z = normalize(data, block=256)
    hits, _ = drift_search(z, tsamp=1.0, foff_mhz=1e-6, max_drift_hz_s=max_drift + 1,
                           snr_threshold=5.0, max_hits=20)
    if not hits:
        return None
    if inj is None:
        h = hits[0]
    else:
        near = [h for h in hits if abs(h.start_channel - inj.start_channel) < 40]
        if not near:
            return None
        h = near[0]
    feats, img = hit_features(z, h.start_channel, h.drift_ch_per_step, h.snr)
    return feature_vector(feats, img)


# Pure noise yields a usable (>5 sigma) hit on ~4% of tries at 4 ms each, so filling the
# noise class needs ~25 tries per example; the old x4 budget stopped at ~110 of 500 (B20).
TRIES_PER_EXAMPLE = 30


def build_training_set(n_per_class: int = 400, seed: int = 0, progress=None):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    total = n_per_class * len(LABELS)
    for li, label in enumerate(LABELS):
        got, tries = 0, 0
        while got < n_per_class and tries < n_per_class * TRIES_PER_EXAMPLE:
            tries += 1
            v = _training_example(label, rng)
            if v is not None:
                xs.append(v)
                ys.append(li)
                got += 1
                if progress:
                    progress(len(xs), total)
        if got < n_per_class:
            logger.warning("Training class %s has %d of %d requested examples after %d tries; "
                           "the classifier will under-learn it.", label, got, n_per_class, tries)
    return np.vstack(xs), np.asarray(ys)


def train(n_per_class: int = 400, seed: int = 0, out: Path = DEFAULT_MODEL, progress=None):
    import joblib
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    t0 = time.time()
    x, y = build_training_set(n_per_class, seed, progress)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                         early_stopping=True, random_state=seed)
    clf.fit(xtr, ytr)
    pred = clf.predict(xte)
    acc = float((pred == yte).mean())
    # B3: report accuracy as a function of drift, so a gap in the training range shows up.
    drift = np.abs(xte[:, SCALAR_FEATURES.index("abs_drift_ch")])
    et = LABELS.index("technosignature_like")
    by_drift = {}
    for lo, hi in ((0, 1), (1, 4), (4, 12), (12, TRAIN_MAX_DRIFT + 1)):
        band = (drift >= lo) & (drift < hi)
        et_band = band & (yte == et)
        by_drift[f"{lo}-{hi:g} ch/step"] = {
            "n": int(band.sum()),
            "accuracy": round(float((pred[band] == yte[band]).mean()), 3) if band.any() else None,
            "et_recall": round(float((pred[et_band] == et).mean()), 3) if et_band.any() else None}
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte, clf.predict(xte), labels=range(len(LABELS))).tolist()
    meta = {"labels": LABELS, "scalar_features": SCALAR_FEATURES, "test_accuracy": acc,
            "accuracy_by_drift": by_drift, "train_max_drift_ch": TRAIN_MAX_DRIFT,
            "confusion_matrix": cm, "n_train": len(ytr), "train_seconds": time.time() - t0,
            "snr_max": float(x[:, SCALAR_FEATURES.index("snr")].max()),
            "class_counts": {lab: int((y == i).sum()) for i, lab in enumerate(LABELS)},
            # Recorded so a load-time mismatch can be reported precisely instead of guessed.
            "sklearn_version": sklearn.__version__, "numpy_version": np.__version__}
    joblib.dump({"model": clf, "meta": meta}, out)
    return meta


class HitScorer:
    """Loads the classifier lazily (once per worker process).

    A pickled estimator cannot cross a scikit-learn minor version, so a mismatch is the
    usual cause of failure here. That must never fail silently: every candidate would be
    written out as "unscored" and the run would look clean while the classifier did nothing.
    Load failures are therefore logged and surfaced through `load_error`.
    """
    _cache: ClassVar[dict] = {}

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else DEFAULT_MODEL
        self.model = None
        self.meta: dict = {}
        self.load_error: str | None = None
        key = str(self.path)
        if key in self._cache:
            self.model, self.meta, self.load_error = self._cache[key]
            return
        if self.path.is_file():
            try:
                import joblib
                bundle = joblib.load(self.path)
                self.model, self.meta = bundle["model"], bundle["meta"]
                if list(self.meta.get("labels", LABELS)) != list(LABELS):
                    # A model trained for other classes would misalign every probability column.
                    raise ValueError(f"model classes {self.meta.get('labels')} differ from this "
                                     f"version's {LABELS}; run `ai-seti train`")
            except Exception as exc:   # version mismatch etc. — degrade, but say so
                self.model = None
                self.load_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "Hit classifier at %s could not be loaded (%s). Hits will be labelled "
                    "model_unavailable; interest falls back to the heuristic. Run `ai-seti train` "
                    "to rebuild the model for scikit-learn %s.",
                    self.path, self.load_error, _sklearn_version())
        else:
            self.load_error = f"model file not found: {self.path}"
            logger.warning(
                "No hit classifier at %s. Hits will be labelled model_unavailable and interest falls back "
                "to the heuristic. Run `ai-seti train` to create one.", self.path)
        self._cache[key] = (self.model, self.meta, self.load_error)

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        model = self.model
        if model is None or len(vectors) == 0:
            return np.full((len(vectors), len(LABELS)), np.nan)
        return model.predict_proba(vectors)


def anomaly_scores(scalar_matrix: np.ndarray, contamination: float = 0.05,
                   seed: int = 0) -> np.ndarray:
    """0..1, higher = more unusual relative to this run's hit population."""
    n = len(scalar_matrix)
    if n < 20:
        return np.full(n, np.nan)
    from sklearn.ensemble import IsolationForest
    x = np.nan_to_num(scalar_matrix)
    x = (x - np.median(x, 0)) / (np.std(x, 0) + 1e-6)
    forest = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
    forest.fit(x)
    s = -forest.score_samples(x)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)
