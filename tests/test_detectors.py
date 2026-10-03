"""Tests for the secondary detectors (spikes and dedispersed pulses).

`find_pulses` is the most intricate untested code in the project, so these build a real
dispersed burst in the pulsar band (100-200 MHz, where dispersion across the band is
actually measurable) rather than asserting only that it returns a list.
"""
import numpy as np
import pytest

from ai_seti.dsp.detectors import KDM, MAD_TO_SIGMA, find_pulses, find_spikes
from ai_seti.dsp.preprocess import normalize
from ai_seti.io.filterbank import FilterbankHeader


def _noise(t: int, f: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return normalize(rng.normal(size=(t, f)).astype(np.float32), block=256)


# ------------------------------------------------------------------------- spikes

def test_find_spikes_locates_single_sample_excursions():
    z = _noise(32, 4096, seed=1)
    z[5, 100] = 25.0
    z[7, 900] = 40.0
    spikes = find_spikes(z, threshold=9.0)
    found = {(s["time_index"], s["channel"]) for s in spikes}
    assert (5, 100) in found and (7, 900) in found
    top = max(spikes, key=lambda s: s["power_sigma"])
    assert top["power_sigma"] == pytest.approx(40.0, rel=1e-3)


def test_find_spikes_reports_absolute_frequency():
    z = _noise(16, 4096, seed=2)
    z[3, 2048] = 30.0
    freqs = 1420.0 - 2.7939677238464355e-06 * np.arange(4096)
    spikes = find_spikes(z, threshold=9.0, freqs_mhz=freqs)
    hit = next(s for s in spikes if s["channel"] == 2048)
    assert hit["frequency_mhz"] == pytest.approx(freqs[2048])


def test_find_spikes_is_empty_on_quiet_data_and_capped():
    z = _noise(16, 4096, seed=3)
    assert find_spikes(z, threshold=12.0) == []
    for t in range(16):
        for c in range(0, 4096, 64):
            z[t, c] = 20.0
    assert len(find_spikes(z, threshold=9.0, max_spikes=50)) == 50


def test_find_spikes_without_frequency_axis_is_nan():
    z = _noise(16, 1024, seed=4)
    z[2, 7] = 30.0
    spikes = find_spikes(z, threshold=9.0)
    assert np.isnan(next(s for s in spikes if s["channel"] == 7)["frequency_mhz"])


# ------------------------------------------------------------------------- pulses

# Pulsar band: dispersion across 100-200 MHz is milliseconds-to-seconds, which is what makes
# a de-Doppler-style dedispersion search meaningful. At L-band it is nanoseconds and any
# detected pulse is necessarily undispersed (hence flagged likely_rfi).
PULSAR_T, PULSAR_F, PULSAR_TSAMP = 256, 8192, 1.0


def _pulsar_band() -> tuple[np.ndarray, np.ndarray]:
    freqs = np.linspace(200.0, 100.0, PULSAR_F)   # descending, like real filterbank data
    return _noise(PULSAR_T, PULSAR_F, seed=5), freqs


def _inject_dispersed(z: np.ndarray, freqs: np.ndarray, dm_true: float, t0: int,
                      nsub: int = 64, amp: float = 6.0) -> None:
    """Add one dispersed burst using the same delay law find_pulses assumes."""
    f = z.shape[1]
    usable = (f // nsub) * nsub
    sub_freq = freqs[:usable].reshape(nsub, -1).mean(axis=1)
    delay_per_dm = KDM * (sub_freq ** -2 - np.max(sub_freq) ** -2) / PULSAR_TSAMP
    for si, d in enumerate(delay_per_dm):
        cols = np.arange(si * (usable // nsub), (si + 1) * (usable // nsub))
        tt = t0 + round(d * dm_true)
        if 0 <= tt < z.shape[0]:
            z[tt, cols] += amp


def test_find_pulses_skips_short_records():
    """Below min_samples there is no dispersion to search for."""
    z, freqs = _noise(16, 4096, seed=6), np.linspace(200.0, 100.0, 4096)
    assert find_pulses(z, freqs, PULSAR_TSAMP) == []


def test_find_pulses_recovers_a_dispersed_burst():
    z, freqs = _pulsar_band()
    _inject_dispersed(z, freqs, dm_true=40.0, t0=40)
    pulses = find_pulses(z, freqs, PULSAR_TSAMP, threshold=8.0, max_dm=200.0, n_dm=64)
    assert pulses, "a strong dispersed burst should be detected"
    best = max(pulses, key=lambda p: p["snr"])
    assert best["snr"] >= 8.0
    assert best["dm"] > 1.0, "a dispersed burst must not be labelled undispersed RFI"
    assert best["likely_rfi"] is False
    # The recovered DM must be the injected one, not merely "some positive number".
    assert best["dm"] == pytest.approx(40.0, rel=0.35)
    assert 35 <= best["time_index"] <= 60, "arrival time should track the injected t0"


def test_find_pulses_flags_undispersed_as_rfi():
    """A burst with no dispersion across the band is terrestrial by construction.

    Other DM trials can still surface a noise peak, so the assertion is on the strongest
    detection, which for an undispersed burst must land at ~0 DM.
    """
    z, freqs = _pulsar_band()
    usable = PULSAR_F // 64
    for si in range(64):
        cols = np.arange(si * usable, (si + 1) * usable)
        z[40, cols] += 6.0
    pulses = find_pulses(z, freqs, PULSAR_TSAMP, threshold=8.0, max_dm=200.0)
    assert pulses
    best = max(pulses, key=lambda p: p["snr"])
    assert best["likely_rfi"] is True, "zero-DM pulses must be flagged"
    assert best["dm"] < 1.0


def test_find_pulses_reports_time_in_seconds():
    z, freqs = _pulsar_band()
    _inject_dispersed(z, freqs, dm_true=30.0, t0=50)
    pulses = find_pulses(z, freqs, PULSAR_TSAMP, threshold=8.0, max_dm=150.0)
    assert pulses
    for p in pulses:
        assert p["time_s"] == pytest.approx(p["time_index"] * PULSAR_TSAMP)
        assert p["width_samples"] in (1, 2, 4, 8, 16)
        assert 0.0 <= p["dm"] <= 150.0


def test_find_pulses_is_quiet_on_noise_only():
    """No bursts: the robust per-subband normalisation must keep the noise floor low."""
    z, freqs = _pulsar_band()
    pulses = find_pulses(z, freqs, PULSAR_TSAMP, threshold=8.0, max_dm=200.0)
    assert all(p["snr"] < 8.0 for p in pulses)


def test_find_pulses_limits_subbands_to_the_channel_count():
    """nsub is clamped to f, so a narrow record must not reshape into nothing."""
    z = _noise(PULSAR_T, 32, seed=7)
    freqs = np.linspace(200.0, 100.0, 32)
    find_pulses(z, freqs, PULSAR_TSAMP, nsub=64)   # must not raise


# ------------------------------------------------------------------- channel ranges

def test_channel_range_whole_file_and_subselection():
    hdr = FilterbankHeader(fch1=1420.0, foff=-2.7939677238464355e-06, nchans=262144,
                           tsamp=18.25, nsamples=16)
    assert hdr.channel_range(None, None) == (0, 262144)
    c0, c1 = hdr.channel_range(1420.0, 1420.1)
    assert 0 <= c0 < c1 <= hdr.nchans
    assert hdr.frequencies(c0, c0 + 3)[0] == pytest.approx(1420.0, abs=3e-6)
    assert hdr.channel_range(0.0, 1e6) == (0, 262144)


def test_channel_range_out_of_band_is_empty_never_inverted():
    """A request wholly outside the band must not return c1 < c0.

    The old clamping produced (0, -35283156335) for a far-away request, which would reach
    read_window as a negative-width slice.
    """
    hdr = FilterbankHeader(fch1=1420.0, foff=-2.7939677238464355e-06, nchans=262144,
                           tsamp=18.25, nsamples=16)
    for lo, hi in ((1e5, 1e6), (0.0, 100.0), (5000.0, 6000.0)):
        c0, c1 = hdr.channel_range(lo, hi)
        assert c1 >= c0, f"channel_range({lo}, {hi}) inverted: {(c0, c1)}"
        assert 0 <= c0 <= c1 <= hdr.nchans
        assert c1 == c0, "a wholly out-of-band request selects nothing"


def test_channel_range_one_sided_bounds():
    """foff is negative here, so channel 0 is the HIGH-frequency end of the band."""
    hdr = FilterbankHeader(fch1=1420.0, foff=-2.7939677238464355e-06, nchans=262144,
                           tsamp=18.25, nsamples=16)
    # "down to 1419.9 MHz" starts at channel 0 and stops where 1419.9 is reached
    c0, c1 = hdr.channel_range(1419.9, None)
    assert (c0, c1) == (0, 35793)
    assert hdr.frequencies(c0, c1 - 1)[-1] == pytest.approx(1419.9, abs=2e-6)
    # "up from the bottom of the band" starts where 1419.9 is and runs to the end
    c0, c1 = hdr.channel_range(None, 1419.9)
    assert c1 == hdr.nchans and c0 > 0
    assert hdr.frequencies(c0, c0 + 1)[0] == pytest.approx(1419.9, abs=2e-6)


def test_fine_per_coarse_detects_the_gbt_ratio():
    coarse = FilterbankHeader(fch1=1420.0, foff=-2.7939677238464355e-06, nchans=1 << 20,
                              tsamp=18.25, nsamples=16)
    assert coarse.fine_per_coarse() == 1 << 20
    odd = FilterbankHeader(fch1=1420.0, foff=-1.0e-5, nchans=1024, tsamp=1.0, nsamples=4)
    assert odd.fine_per_coarse() is None
    assert FilterbankHeader(fch1=1420.0, foff=0.0, nchans=8, tsamp=1.0,
                            nsamples=4).fine_per_coarse() is None


def test_mad_to_sigma_constant_is_the_gaussian_factor():
    assert pytest.approx(1.4826) == MAD_TO_SIGMA
