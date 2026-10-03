# AI-SETI backlog — pending bugs and changes

Author: Inventions4All — github:TWeb79

Applies to: **v0.3.0**. Opened at the end of the 0.3.0 hardening pass
(see [implementationplan.md](implementationplan.md)). Everything below is *pending*;
nothing on this page is fixed.

Severity: **high** = can produce a wrong or misleading result · **med** = wrong behaviour in
a plausible configuration · **low** = quality, debt or hygiene.

---

## Bugs

### B1 — `drift_search` silently degenerates to a zero-drift-only search — high

`src/ai_seti/dsp/dedoppler.py:129-132` clamps the searched drift range with no warning:

```python
k_needed = abs(max_drift_hz_s * tsamp / foff_hz) if foff_hz else 1
k_max = max(0, int(np.ceil(k_needed)) - 1)
k_max = min(k_max, max(0, max_ch_per_step - 1))
k_max = min(k_max, max(0, f // max(t, 1) - 1))
```

If any of these drive `k_max` to `0` while `max_drift_hz_s > 0`, the search evaluates only
drift `0` and reports **no hits** for anything drifting. That is the worst possible failure
shape for this program: "I found nothing" instead of "I could not look there".

Measured: a GBT coarse channel (32768 channels, `tsamp=18.253611008`,
`foff=2.7939677238464355e-06` MHz = 2.794 Hz) carrying a tone that drifts 13 channels/step at
SNR 6/sample returned **0 hits** — while `_search_one_sign` had correctly integrated the track
to a path sum of `78.8` (`_noise_scale` → `med=0.039`, `scale=4.054`, so SNR ≈ 19.4). The same
call with the range intact returns `(start_channel=8000, drift=13.0, snr=19.4)` exactly.

**Fix.** Refuse to be quiet. When a non-zero `max_drift_hz_s` cannot be represented at the
given `tsamp`/`foff_hz`, log a warning naming the achievable rate and either raise or record
it in `metadata.json`. A `k_max == 0` run should be visibly a degraded run.

**Test.** Assert the warning/failure when the requested rate is unsearchable.

### B2 — `foff_mhz` has no unit validation; Hz passed where MHz is expected blinds the search — high

`drift_search(..., foff_mhz=...)` multiplies by `1e6` internally
(`src/ai_seti/dsp/dedoppler.py:127`). Passing `foff` already in Hz inflates it a
million-fold, `k_needed` collapses, and B1 fires silently. Measured: `foff_mhz=2.794`
(Hz) → `k_max=0`, 0 hits; `foff_mhz=2.794e-06` (MHz) → 1 hit, recovered exactly.

This exact mistake was made during the 0.3.0 review and cost ~30 minutes, because the
failure looks like a detector bug rather than a units bug.

**Fix.** Take `foff_hz` and derive MHz internally, or validate the magnitude of `foff_mhz`
and reject values that imply an implausible channel width for the time resolution.

**Test.** Assert a wrong-unit argument is rejected rather than silently accepted.

### B3 — the classifier never sees the drift regime it has to score in production — high

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

---

## Test debt

Total coverage is **65%** against a **60%** CI floor, so the floor does not currently bite.
These are the real gaps, largest first:

| Module | Coverage | Uncovered | Why it is open |
|---|---|---|---|
| `cli.py` | **0%** | 238 stmts | Covered by the CI smoke job instead of unit tests. Every argument parser, subcommand dispatch and terminal render path is unexercised by pytest. |
| `dashboard.py` | **0%** | 124 stmts | Same. The live terminal view has no unit coverage at all. |
| `ai/model.py` | 48% | 58 stmts | `build_training_set` / `_training_example` (86-102) never run under test — the expensive path. This is where B3 lives, so a cheap version of it needs to exist before B3 can be fixed safely. |
| `sources.py` | 50% | 59 stmts | `split()` and the drift guard band (96-187) are barely tested outside the integration tests. |
| `core.py` | 58% | 39 stmts | Legacy detector paths (112-183) untested. |
| `ai/simulate.py` | 58% | 36 stmts | Injection classes 85-128 largely untested — notably the ones B3 depends on. |
| `io/remote.py` | 61% | 30 stmts | Retry/backoff branches (43-48, 86-105) only partly covered. |
| `share.py` | 93% | 13 stmts | The GitHub/webhook sink bodies (280-289, 336-343) are not exercised end to end. |

**Recommendation.** Raise the floor above 60% only after `cli.py` or `dashboard.py` gets real
tests; today 65% passes partly because two large modules are entirely unmeasured, which makes
the percentage flattering.

---

## Limitations carried forward (need verification, not code)

These are documented honestly in the README but have no plan attached:

- **BL archive untested in reality.** The archive API and HTTP Range streaming were never
  exercised against the real servers (development sandbox was network-restricted). Range
  streaming is tested against a local server only.
- **GPU (CuPy) path unverified and excluded from CI.** The array-API style in
  `dsp/dedoppler.py` exists so it *can* work; it has never run on a GPU.
- **Multi-core speedup never measured.** The demo reports 2.4 s on 1 core and 1.8 s on 4
  workers — worse than linear, which is unexplained and worth a profile.
- **Python 3.14 unsupported** by `requires-python`.
- **No turboSETI cross-check has ever been run**, despite the `[compare]` extra existing for
  exactly that purpose. This is the cheapest outstanding item and the most load-bearing: an
  independent implementation agreeing on the benchmark set would validate the headline result.

---

## Verified — *not* a bug

Recorded so it is not re-investigated.

**"The de-Doppler detector fails to find fast-drifting tones at production resolution."**
False. `_search_one_sign` integrates the true track exactly (`tree[d=0][8000] = 78.8`,
matching the manual path sum), and `drift_search` returns `(8000, 13.0 ch/step, snr 19.4)`.
The apparent failure was B2 (Hz passed as MHz) in the measurement script. The detector was
correct throughout.