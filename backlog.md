# AI-SETI backlog — pending bugs and changes

Author: Inventions4All — github:TWeb79

Applies to: **v0.3.0**. Opened at the end of the 0.3.0 hardening pass
(see [implementationplan.md](implementationplan.md)).

Fixed items are removed from this file once done (last cleanup 2026-10-03: B29, B19, B28, B10, B31, B33, B34, B35, B32, B30, B21, B27, B15, B1, B2, B22, B23, B24, B25, B26, B16, B14, B17, B12, B4, B5, B6, B8,
B9, B11, B13, B18, B20, B7(b) and B19's labelling). Their write-ups, the two real-data runs of
2026-10-03 and the regression tests that pin them are in git history.

Items are sorted by severity, then by the order they should be fixed in. **IDs are stable and
must not be renumbered** — other documents and the tables below cross-reference them.

Severity: **high** = can produce a wrong or misleading result · **med** = wrong behaviour in
a plausible configuration · **low** = quality, debt or hygiene.

## Triage

| ID | Severity | Summary | Area |
|---|---|---|---|
| B7 | high | Archive cadences are never grouped, so the ON/OFF filter never runs automatically | `cli.py`, `sources.py` |
| B3 | high | The classifier never sees the drift regime it must score in production | `ai/model.py` |
| B36 | med | A stationary tone is reported as drifting 6% of the time and escapes the Earth-source penalty | `ai/features.py`, `pipeline.py`, `share.py` |
| B37 | low | Spikes and pulses are computed and saved, but never scored, summarised or shared, and spikes repeat de-Doppler hits | `dsp/detectors.py`, `report.py` |

**Suggested order.** B36 (a stationary tone can pass as drifting), then B37 (use or drop the
spike and pulse by-products), then B3 (retrain). B7 needs a design for inferring cadences from
the archive's metadata before any code.

---

# Bugs

## High

### B7 — archive cadences are never grouped, so the ON/OFF filter never runs automatically

Part (b) of this item is done: `rfi.flag_multi_target` flags a frequency already seen in a
different target. What remains is (a). The HIP2579, HIP2586 and HIP3249 files under
`LHS1140/C/` were recorded on the same day (MJD 57774) in consecutive scans: very likely the
OFF pointings of one LHS 1140 cadence. `BreakthroughListenSource` already reads `cadence_url`
(`src/ai_seti/sources.py`), but `crunch` never uses it, and `ai-seti cadence` only accepts
local files. The cadence filter, the strongest RFI test there is, therefore never runs on
archive data.

**Blocked on design (checked live 2026-10-03).** The archive's `query-files` rows carry **no
`cadence_url` field**. Each row has only `target`, `ra`, `decl`, `mjd`, `center_freq`, `size`,
`md5sum`, `utc` and `url`, so the plan below cannot work as written. A cadence has to be
inferred instead: same day (`mjd`), same receiver (`center_freq` band), scans a few minutes
apart. ON vs OFF also can't be read from file names here: the LHS 1140 files are each named after
their own HIP star. One option is to treat the target the user asked for as ON and the
neighbouring scans as OFF.

**Fix (original plan).** Group archive files by `cadence_url`; when a group is complete (≥ 2 ON,
≥ 1 OFF), run `cadence_filter` and feed its `events.csv` into scoring and sharing.

**Test.** A fake source yielding one ON/OFF/ON/OFF group with a tone only in the ONs: `crunch`
writes an event that passes; the same tone in an OFF fails it.

### B3 — the classifier never sees the drift regime it has to score in production

`ai/model.py:40` builds training examples with `n_chan=4096, max_drift=6.0`, and
`ai/simulate.py:89` draws `technosignature_like` drift from `uniform(0.05, max_drift)`, i.e.
**0.05–6 channels/step**. Production on a GBT coarse channel is ~2.794 Hz / 18.25 s, where
the 4 Hz/s rate limit is already **~26 channels/step** (README documents this).

So the model is trained exclusively on slow drift and is asked to classify fast drift. Its
class labels are least reliable in exactly the regime the search spends most of its time in.
The 95.7% held-out accuracy figure in the README is measured on the training distribution and
does not transfer.

**Fix.** Widen `max_drift` (and probably `n_chan`) in `_training_example` to cover the
production range, then retrain and re-measure. Report accuracy as a function of drift.

**Note.** A sweep over a production-regime training set was started and abandoned — the run
cost is high (27 Taylor-tree passes per sign per example, ~0.27 s/example, and rejected
examples are retried up to `n_per_class * 4` times). Budget for a reduced sweep before
committing to it.

**Also in this retrain.** The bundled model predates the B20 fix, so its noise class has ~110
examples instead of 500; a retrain with the current code fills it (55 s, accuracy 0.963 vs
0.957, measured 2026-10-03). Add a strong-carrier RFI class to the simulator (high SNR, slow
drift), the open part of the former B8. Strong, slowly drifting carriers are among the most
common terrestrial RFI, and the classifier has never seen one. Until then the SNR guard in
`ai.model.ood_snr_limit` labels such hits `out_of_distribution`.

## Medium

### B36 — a stationary tone is reported as drifting 6% of the time and escapes the Earth-source penalty

Found in the B30 review. `zero_drift` is `|drift| < 0.02` channels/step (`ai/features.py`), but
the Taylor tree's smallest non-zero step with 16 samples is 1/15 ≈ 0.067 channels/step,
±0.010 Hz/s on high-res BL data. That is one channel over the whole observation, within the
measurement noise of "not drifting". Measured: 200 perfectly stationary tones split across two
channels at random sub-channel positions (SNR ~16) were detected 194 times; **12 (6%) came back
with a one-step drift and were labelled drifting**. They then escape the ×0.3 zero-drift penalty,
pass the share gate's "Drifts" check, and count as "drifting, as a transmitter on a rotating
planet would" in the report. Real data is full of ±0.010 Hz/s hits.

**Fix.** Treat `|drift| ≤ 1 tree step` (≤ 1 channel over the observation) as stationary for
scoring, the gate and the wording. Keep the model feature `zero_drift` as it is, because the
bundled classifier was trained on it, or change both together at the B3 retrain.

**Test.** The 200-tone experiment above: no stationary tone is scored or gated as drifting.

## Low

### B37 — spikes and pulses are computed and saved, but never used

Found in the B30 review. Every work unit also runs the SETI@home-style spike detector and the
Astropulse-style dispersed-pulse search (`dsp/detectors.py`). The results go only to
`spikes.csv` and `pulses.csv`, and their counts into `metadata.json`. Nothing scores them,
flags them, puts them in the HTML report, the terminal summary or the share gate. On high-res
files 100% of spikes sit within 1 kHz of a de-Doppler hit: strong carriers re-reported sample
by sample, ~140–220 rows per file with 19–68 distinct channels. On mid-res files every pulse
found had DM < 1 (`likely_rfi`).

**Fix.** Either drop spikes that coincide with a de-Doppler hit and show the rest in the
summary and report, or say plainly in the README that they are raw by-products. The
documentation pass (B31) does the latter for now.

## Test debt

Total coverage is **65%** against a **60%** CI floor, so the floor does not currently bite.
These are the real gaps, largest first:

| Module | Coverage | Uncovered | Why it is open |
|---|---|---|---|
| `cli.py` | **0%** | 238 stmts | Covered by the CI smoke job instead of unit tests. Every argument parser, subcommand dispatch and terminal render path is unexercised by pytest.  |
| `dashboard.py` | **0%** | 124 stmts | Same. The live terminal view has no unit coverage at all. Where B15 lives. |
| `ai/model.py` | 48% | 58 stmts | `build_training_set` / `_training_example` (86-102) never run under test — the expensive path. This is where B3 lives, so a cheap version of it needs to exist before B3 can be fixed safely. |
| `sources.py` | 50% | 59 stmts | `split()` and the drift guard band (96-187) are barely tested outside the integration tests. Also where B7 lives. |
| `core.py` | 58% | 39 stmts | Legacy detector paths (112-183) untested. |
| `ai/simulate.py` | 58% | 36 stmts | Injection classes 85-128 largely untested — notably the ones B3 depends on. |
| `io/remote.py` | 61% | 30 stmts | Retry/backoff branches (43-48, 86-105) only partly covered. |
| `share.py` | 93% | 13 stmts | The GitHub/webhook sink bodies (280-289, 336-343) are not exercised end to end. |

**Recommendation.** Raise the floor above 60% only after `cli.py` or `dashboard.py` gets real
tests; today 65% passes partly because two large modules are entirely unmeasured, which makes
the percentage flattering. `io/remote.py` and `sources.py` should come first — they hold the
highest-severity defects and are the easiest to test against a local server.

---

## Limitations carried forward (need verification, not code)

These are documented honestly in the README but have no plan attached:

- **BL archive tested, but only against a few targets** (LHS 1140 / HIP files). Every archive
  interaction has now run live at least once; B12's cause was found that way.
- **GPU (CuPy) path unverified and excluded from CI.** The array-API style in
  `dsp/dedoppler.py` exists so it *can* work; it has never run on a GPU.
- **Multi-core speedup worse than linear and unexplained.** The demo reports 2.4 s on 1 core
  and 1.8 s on 4 workers. Real data did scale (125–137 k channels/s on 8 workers), so the demo
  figure is more likely a measurement artefact than a scaling failure — but it is unresolved.
- **Python 3.14 unsupported** by `requires-python`.
- **No turboSETI cross-check has ever been run**, despite the `[compare]` extra existing for
  exactly that purpose. This is the cheapest outstanding item and the most load-bearing: an
  independent implementation agreeing on the benchmark set would validate the headline result.

---

## Verified — *not* bugs

Recorded so they are not re-investigated.

**"RA / Dec 8.20833 / −13.2575 in the dashboard is wrong."** False. The archive reports RA in
degrees; HIP 2586 is at RA 00h 32m 50s = 8.21°. The value is right but unlabelled (cosmetic).

**"Opposite drift signs in the top candidate pairs point to a sign bug."** False. The pairs are
spectral images around the coarse-channel centre (now flagged `mirror_image`); an image of a tone drifting up drifts down.

**"The de-Doppler detector fails to find fast-drifting tones at production resolution."**
False. `_search_one_sign` integrates the true track exactly (`tree[d=0][8000] = 78.8`,
matching the manual path sum), and `drift_search` returns `(8000, 13.0 ch/step, snr 19.4)`.
The apparent failure was B2 (Hz passed as MHz) in the measurement script. The detector was
correct throughout.