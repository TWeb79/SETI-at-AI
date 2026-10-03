# Implementation plan — AI-SETI 0.3.0 hardening pass

Author: Inventions4All — github:TWeb79

Scope: close the gaps found in the 0.3.0 review. No new signal-processing features; this
pass is about correctness, honesty about what the software actually did, and making the
quality gates enforceable.

## Task list

| # | Task | Module | Status |
|---|---|---|---|
| 1 | Pin `scikit-learn<1.9` so the bundled pickled estimator loads | `pyproject.toml` | done |
| 2 | Record the sklearn/numpy version in the trained model metadata | `ai/model.py` | done |
| 3 | Stop swallowing classifier load failures: log + expose `load_error` | `ai/model.py` | done |
| 4 | Distinguish `model_unavailable` / `ai_disabled` from `unscored` | `pipeline.py` | done |
| 5 | Write the AI-layer failure into `metadata.json` and the HTML report | `report.py` | done |
| 6 | Make `max_drift_ch_per_step` an effective cap, shared with the guard band | `sources.py`, `pipeline.py` | done |
| 7 | Refuse a cadence "pass" from a single ON scan with no OFF scan | `rfi.py` | done |
| 8 | Re-check cadence status when grading a finding for sharing | `share.py` | done |
| 9 | Key the share dedupe on `finding_id` instead of record equality | `share.py` | done |
| 10 | Unique temp file for the ledger; tolerate unknown ledger keys | `state.py` | done |
| 11 | Declare `ruff` lint rules and a `mypy` configuration | `pyproject.toml` | done |
| 12 | Fix every ruff finding (76 at the start) | all | done |
| 13 | Fix the 10 genuine mypy errors | all | done |
| 14 | Version + build datetime stamp in the report UI | `report.py` | done |
| 15 | CI: lint, typecheck, tests on 3.11/3.12/3.13 + smoke job | `.github/workflows/ci.yml` | done |
| 16 | `.gitignore`, `ARCHITECTURE.md`, `requirements.txt` | repo root | done |
| 17 | Regression tests for tasks 1, 6, 7, 8, 9, 10 | `tests/` | done |
| 18 | Correct stale README claims | `README.md` | done |
| 19 | Close the coverage gaps the review exposed: `dsp/detectors.py` 28%, `benchmark.py` 0%, `io/filterbank.py` 69%, `state.py` 51% | `tests/` | done |
| 20 | Fix `channel_range` returning an inverted range for out-of-band requests | `io/filterbank.py` | done |
| 21 | Backlog the defects found after the pass and link it from the docs | `backlog.md` | done |
| 22 | Regression test: drift search at real GBT coarse-channel resolution | `tests/test_model_and_limits.py` | done |
| 23 | Fix the stale `tests/test_model.py` path cited in the architecture notes | `ARCHITECTURE.md` | done |
| 24 | Record the unhandled drift-range collapse in the failure-handling table | `ARCHITECTURE.md` | done |
| 25 | State the drift-search and classifier-drift limitations in the README | `README.md` | done |
| 26 | B4: stop `remote_header` re-requesting a range that returns no bytes | `io/remote.py` | done |
| 27 | B9: never report a drift rate above `max_drift_rate_hz_s` | `dsp/dedoppler.py` | done |
| 28 | B6: flag and demote mirror images about the coarse-channel DC bin | `rfi.py`, `pipeline.py`, `share.py`, `report.py` | done |
| 29 | B8: out-of-distribution SNR guard, interest cap, cadence required for remote shares | `ai/model.py`, `pipeline.py`, `share.py`, `config.py` | done |
| 30 | B5: `--max-units` resumes where the last run stopped; done only when fully searched | `state.py`, `cli.py` | done |
| 31 | B11: record failed files (give up after 3), widen the archive query past seen rows | `state.py`, `sources.py`, `cli.py` | done |
| 32 | Review fixes for 28–31: config default, chunk-edge padding, SNR cap without a model, unit errors | `configs/default.json`, `sources.py`, `pipeline.py`, `cli.py` | done |
| 33 | B7(b): flag frequencies already seen in a different target (lifetime ledger index) | `rfi.py`, `state.py`, `cli.py`, `share.py`, `report.py` | done |
| 34 | B18: flag both halves of a symmetric mirror pair | `rfi.py`, `report.py`, `share.py` | done |
| 35 | B19: label hits `drift_unresolved` where drift cannot move a tone one channel | `sources.py`, `pipeline.py`, `share.py`, `report.py`, `cli.py` | done |

## Test plan

One-liner per task, mapping to the test that would fail if the task were reverted.

1. **Task 1/2 — bundled model loads.** `test_bundled_classifier_loads`: `HitScorer().available`
   is true, and `predict` returns finite probabilities instead of NaN. Reverting the pin or
   the `except Exception` fallback fails this.
2. **Task 3/4 — failure is visible.** `test_model_load_failure_is_reported`: point the scorer at
   a corrupt file and assert `load_error` is set; `test_score_candidates_flags_unavailable_model`
   asserts `ai_class == "model_unavailable"` rather than a silent `unscored`.
3. **Task 5 — failure reaches the report.** `test_write_outputs_records_ai_note`: a candidate
   frame with `model_unavailable` produces `stats["ai_note"]`.
4. **Task 6 — drift cap is real.** `test_drift_cap_limits_searched_paths`: raising
   `max_drift_ch_per_step` raises the recovered drift of a fast tone; `test_drift_padding_matches_searched_drift`
   asserts the guard band is at least the drift the search can cover, so the two never drift apart.
5. **Task 7 — cadence needs a real test.** `test_cadence_rejects_single_on_scan_without_off`:
   one ON scan, no OFF, yields no events. `test_cadence_requires_off_scan` and
   `test_cadence_passes_with_on_and_off` pin the accept/reject boundaries.
6. **Task 8 — share grading.** `test_cadence_status_rejects_thin_events`: a hand-written
   `events.csv` claiming one ON scan grades as `failed`, not `passed`.
7. **Task 9 — dedupe keys.** `test_webhook_and_ledger_dedup` (existing) covers ledger reuse;
   `test_share_dedupes_by_id_not_record_value` sends two records with identical payloads but
   distinct IDs and asserts both are delivered.
8. **Task 10 — ledger robustness.** `test_ledger_roundtrip_and_best`,
   `test_ledger_ignores_unknown_keys`, `test_ledger_save_is_atomic`.
9. **Task 20 — inverted channel range.** `test_channel_range_out_of_band_is_empty_never_inverted`:
   a wholly out-of-band request must not return `c1 < c0`.
10. **Task 19 — detector coverage.** `test_find_pulses_recovers_a_dispersed_burst` builds a real
    dispersed burst in the 100–200 MHz pulsar band and asserts the recovered DM is the
    injected one, not merely positive. Plus spike localisation, IF-axis selection for 3-D
    HDF5, config load/round-trip, and two benchmark assertions including the headline claim
    that v0.2 beats v0.1 on a drifting tone.
11. **Regression suite.** All 15 pre-existing tests still pass, including
    `test_pipeline_finds_injected_signal_once`, whose drift expectations the cap still satisfies.
12. **Task 22 — production resolution.** `test_drift_search_recovers_a_tone_at_gbt_coarse_resolution`
    builds a 32768-channel coarse channel at 2.794 Hz / 18.25 s, injects a 13 ch/step tone and
    asserts the start channel, drift and SNR are all recovered. Every other drift test uses a
    narrower synthetic grid, so none of them would have caught a wrong-unit `foff` collapsing
    the searchable drift to zero — the failure mode recorded as backlog B1/B2.
13. **Task 26 — B4.** `test_remote_header_gives_up_when_the_server_stops_sending` serves an
    empty but well-formed `206` and asserts `remote_header` raises within a second after at
    most one wasted request. Measured against the pre-fix loop: 17,681 requests in 3 s, never
    terminating.
14. **Task 27 — B9.** `test_drift_search_never_reports_beyond_the_configured_rate` injects a
    0.5 ch/step tone into a 2.86 kHz mid-resolution-shaped block (1 ch/step = 2,672 Hz/s),
    asserts nothing is reported under a 4 Hz/s ceiling, and asserts the same tone *is*
    recovered once the ceiling is raised above its own rate — so the cap filters rather than
    disables the search. `test_drift_search_skips_the_tree_when_only_zero_drift_is_in_range`
    pins the zero-drift shortcut.

## Verification performed

- `pytest` — 72 tests, all green.
- `pytest --cov=ai_seti` — **65%** (was 50%). `dsp/detectors.py` 28%→98%, `benchmark.py`
  0%→94%, `io/filterbank.py` 69%→91%, `config.py` 77%→100%, `report.py` 0%→85%,
  `state.py` 51%→93%, `rfi.py` 95%, `share.py` 93%, `pipeline.py` 88%.
  `cli.py` and `dashboard.py` remain at 0% by design — the CI smoke job covers them instead.
- `ruff check .` — clean (was 76 findings).
- `mypy` — clean, 23 source files (was 21 errors).
- CLI smoke: `demo`, `analyze`, `inspect`, `status`, `crunch`, `cadence`, `benchmark`, `share`.

## Outcome worth noting

With the classifier actually loading, `ai-seti demo` now recovers the three hidden ET-like
tones at ranks **#1, #2 and #3**, all classified `technosignature_like`, ahead of the
strongest RFI (interest 74–82 vs 28–32). Before the fix the same run reported `unscored`
for every candidate and the README's "#1–#3" claim was false — the AI layer was doing
nothing. Single-core demo time is 2.4 s wall / 2.3 s CPU.

## Deferred — see [backlog.md](backlog.md)

This pass did not close everything it found. Three defects are documented and left open
rather than fixed, because each needs a decision about intended behaviour, not just a patch:

- **B1** — `drift_search` clamps the searchable drift range silently. An unsearchable drift
  rate produces "no hits" rather than a degraded run, which is the wrong failure shape for a
  null-result instrument.
- **B2** — `foff_mhz` is unvalidated, so a channel width in Hz passed as MHz quietly blinds
  the search. This was hit for real during this pass and cost ~30 minutes of misdiagnosis.
- **B3** — the classifier trains on 0.05–6 channels/step but must score up to ~26 in
  production. Fixing it means regenerating and re-measuring the model, which is a separate
  piece of work.

Two module gaps are also recorded as test debt: `cli.py` and `dashboard.py` are at 0% and
are currently covered only by the CI smoke job, which makes the headline 65% coverage flatter
than it looks. The floor should not be raised until one of them has real tests.
