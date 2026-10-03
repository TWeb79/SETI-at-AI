import numpy as np

from ai_seti.core import make_synthetic_waterfall, robust_candidates


def test_synthetic_data_shape():
    data, freqs = make_synthetic_waterfall(64, 256)
    assert data.shape == (64, 256)
    assert freqs.shape == (256,)


def test_legacy_detector_flags_injected_persistent_line():
    data, freqs = make_synthetic_waterfall(256, 1024)
    candidates = robust_candidates(data, freqs, threshold=6.0)
    assert 310 in set(candidates["channel_index"].tolist())


def test_legacy_detector_rejects_bad_dimensions():
    try:
        robust_candidates(np.ones(10))
    except ValueError:
        return
    raise AssertionError("Expected ValueError for 1-D input")
