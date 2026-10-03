# AI-SETI backlog — pending bugs and changes

Author: Inventions4All — github:TWeb79

Applies to: **v0.3.0**. Opened at the end of the 0.3.0 hardening pass
(see [implementationplan.md](implementationplan.md)).

Updated 2026-10-03: B4–B15 added after a code review and the first runs against real
Breakthrough Listen data (`ai-seti crunch --target HIP --max-units 8`). Those runs are
summarised in [Real-data run 2026-10-03](#real-data-run-2026-10-03). B16–B17 added the same
day after tracing every reader of `SearchConfig`'s drift limits. B18–B21 added after a
second real-data run with the B4–B11 fixes in place, summarised in
[Real-data run 2026-10-03, second pass](#real-data-run-2026-10-03-second-pass).

Items are sorted by severity, then by the order they should be fixed in. **IDs are stable and
must not be renumbered** — other documents and the tables below cross-reference them.

Severity: **high** = can produce a wrong or misleading result · **med** = wrong behaviour in
a plausible configuration · **low** = quality, debt or hygiene.

## Triage

| ID | Severity | Summary | Area |
|---|---|---|---|
| ~~B4~~ | high — **fixed** | ~~`remote_header` looped forever on an empty 206~~ | `io/remote.py` |
| ~~B6~~ | high — **fixed**, but see B18 | ~~Coarse-channel mirror images ranked as top `technosignature_like` candidates~~ | `rfi.py`, `pipeline.py` |
| ~~B8~~ | high — **fixed** (simulator class open, see B3) | ~~Classifier saturates at p=1.0 at any SNR — publishes RFI as a technosignature~~ | `ai/model.py`, `pipeline.py`, `share.py` |
| ~~B18~~ | high — **fixed** | ~~Mirror pairs about the coarse-channel centre have equal SNR; B6 demoted only one half~~ | `rfi.py` |
| B7 | high — **(b) fixed**, (a) open | The same signal in several targets is not recognised as interference | `rfi.py`, `state.py`, `cli.py` |
| ~~B5~~ | high — **fixed** | ~~`--max-units` took the top of the band, then marked the whole file done~~ | `cli.py`, `state.py` |
| ~~B9~~ | high — **fixed** | ~~Drifts far beyond the configured limit were searched and reported~~ | `dsp/dedoppler.py` |
| B1 | high | `drift_search` silently degenerates to a zero-drift-only search | `dsp/dedoppler.py` |
| B2 | high | `foff_mhz` is unvalidated; Hz passed where MHz is expected blinds the search | `dsp/dedoppler.py` |
| B3 | high | The classifier never sees the drift regime it must score in production | `ai/model.py` |
| B10 | med | Non-power-of-two time samples roughly double the Taylor-tree cost | `dsp/dedoppler.py` |
| ~~B11~~ | med — **fixed** | ~~Failed files were never recorded and the archive query never paged~~ | `cli.py`, `sources.py`, `state.py` |
| B12 | med | Every `.gpuspec.8.0001.fil` fails with "Corrupt SIGPROC header string" | `io/filterbank.py` |
| ~~B19~~ | med — **fixed** (labelling; speed open) | ~~Mid-res products cannot resolve drift, yet every hit was reported as "does not drift"~~ | `pipeline.py`, `sources.py` |
| B20 | med | Classifier training silently gets ~110 of 500 requested noise examples | `ai/model.py` |
| B16 | med | Streamlit drift slider above ~4.9 Hz/s has no effect on high-res GBT data | `app.py` |
| B13 | low | `ai-seti status` reports "reachable (0 sample rows)" and can hang silently | `cli.py` |
| B14 | low | README commands fail when pasted into zsh | `README.md` |
| B15 | low | The terminal dashboard leaves blank lines and torn panels | `dashboard.py` |
| B21 | low | Multi-target match (B7b) misses the DC forest: neighbours across targets sit 4–7 kHz apart | `rfi.py` |
| B17 | low | Shared findings omit `max_drift_ch_per_step`, the drift cap that can bind | `share.py` |

**Suggested order.** Done: B4, B5, B6, B8, B9, B11, B18, B19 and B7(b).
Next is B20 (noise-class shortfall in training).
Then B7(a), which needs cadence plumbing. Then B1 + B2 + B3, then the rest.

---

# Bugs

## High

### B4 — `remote_header` loops forever on an empty Range response, hammering the server

> **Status: fixed.** `remote_header` now takes `max_requests` and `chunk`, stops when a
> response adds no bytes, stops once `len(buf)` reaches the advertised `Content-Range` total,
> and raises `ValueError` naming the URL instead of the old 1 MB guard. Regression test:
> `test_remote_header_gives_up_when_the_server_stops_sending`.

`src/ai_seti/io/remote.py:51-65`. If a `206` response carries fewer bytes than the header
needs (empty body, truncated proxy reply), `buf` never grows, `parse_sigproc_header` keeps
raising `EOFError`, and the `len(buf) > 1 << 20` exit is never reached:

```python
while True:
    body, hdrs, _ = _get(url, len(buf), len(buf) + chunk)
    buf += body
    ...
    try:
        return parse_sigproc_header(buf, total)
    except EOFError as exc:
        if len(buf) > 1 << 20:                      # never true: buf cannot grow
            raise ValueError("SIGPROC header larger than 1 MB?") from exc
```

Measured against a local server that returns an empty `206`: **10,798 requests in 5 s**, still
looping. Against the real archive this is an unthrottled request flood from every volunteer
whose connection hiccups, and `crunch` hangs with no output. Note that `total` is parsed but
never used as a completion condition.

**Fix.** Break when a response adds no bytes, stop once `len(buf)` reaches the `Content-Range`
total, cap the number of header requests (e.g. 64), and raise `ValueError` naming the URL.

**Test.** Local server returning an empty `206`: `remote_header` raises within 1 s and sends
at most a handful of requests.

### B9 — on coarse products, drifts far beyond `max_drift_rate_hz_s` are searched and reported

> **Status: fixed.** `_search_one_sign` now takes `k_limit_ch` (the ceiling in channels per
> step) and skips every `(k, d)` row beyond it, so an out-of-range drift is never evaluated
> and therefore never reported. When the ceiling is below one tree resolution step the tree
> is skipped entirely and the zero-drift column sum is returned, which is exactly `tree[0]`.
> Regression tests: `test_drift_search_never_reports_beyond_the_configured_rate` and
> `test_drift_search_skips_the_tree_when_only_zero_drift_is_in_range`.

The opposite failure of B1. `_search_one_sign` (`src/ai_seti/dsp/dedoppler.py:93-99`) loops
`for d in range(tp)` regardless of `k_max`, so drift `d/span` is always searched up to
**1 channel/step** even when `k_max == 0`:

On the BL mid-resolution product (`.0002`: 2.86 kHz × 1.07 s × 279 samples) one channel per
step is **±2,665 Hz/s**.

Log: candidates reported at −276, −370, −485, −568 and **−1,225 Hz/s** with
`max_drift_rate_hz_s = 4`. Reproduced: a tone sweeping 0.5 ch/step is reported at
**−1,329.7 Hz/s**.

**Fix.** Discard drift rows whose rate exceeds `max_drift_rate_hz_s`. When the maximum drift
over the whole observation is below one channel (`max_drift × T × tsamp < |foff|`), skip the
tree and use the zero-drift column sum. Record in `metadata.json` which drift range was
actually searched (shares the reporting requirement with B1).

**Test.** Mid-res-shaped input with a 0.5 ch/step tone and `max_drift_hz_s = 4`: no hit with
`|drift_rate_hz_s| > 4`.

### B6 — mirror images around the coarse-channel DC bin rank as top "technosignature_like" candidates

> **Status: fixed.** Each hit now carries `mirror_channel`, the start channel of its image
> about the coarse-channel DC bin (`pipeline.process_work_unit`, -1 when the channelisation is
> unknown). `rfi.flag_mirror_images` matches partners within ±3 channels and opposite drift in
> the same file, sets `mirror_pair` on both and `mirror_image` on the weaker, and the interest
> of a `mirror_image` is multiplied by 0.2. The flag is in `candidates.csv`, explained in the
> HTML report, listed in the share checklist, and blocks sharing in `passes_gate`.
> **Real data contradicts the "weaker image" premise: see B18.**
> Regression test: `test_mirror_image_about_coarse_centre_is_flagged_and_demoted`, which
> checks the demotion factor itself, not only that the image is weaker.

GBT/BL coarse channels show spectral images: a strong tone at `f_c + Δ` produces a weaker copy
at `f_c − Δ` with the opposite drift sign, where `f_c` is the coarse-channel centre (DC bin).
Nothing in the pipeline knows this.

Every top-ranked candidate from the high-resolution files in the 2026-10-03 run is such a pair
(sums are exactly `2·f_c` to 3 decimal places, drifts are mirrored):

| file | pair (MHz, Hz/s) | f_c |
|---|---|---|
| HIP2586 `.0000` | 8000.282770 / −0.031 and 7999.717227 / +0.031 | 8000.0000 |
| HIP2586 `.0000` | 8000.302443 / −0.041 and 7999.697554 / +0.041 | 8000.0000 |
| HIP3249 `.0000` | 7999.701315 / +0.082 and 8000.298682 / −0.082 | 8000.0000 |
| HIP3249 `.0007` | 2250.282793 / −0.061 and 2249.717204 / +0.061 | 2250.0000 |

All were classed `technosignature_like` with interest 78–94.

**Fix.** After scoring, for each hit compute its coarse-channel centre from `fch1`, `foff` and
`fine_per_coarse()`; look for a partner at `2·f_c − f` (± a few channels) with drift ≈ −drift.
Flag the weaker one `mirror_image` (and the pair `mirror_pair`), and multiply interest by
≤ 0.2. Report the flag in the CSV, HTML report and share checklist.

**Test.** Inject a tone and its mirror (opposite drift, 10× weaker) around a synthetic coarse
centre; the weaker hit is flagged and drops below the stronger one by interest.

### B8 — the classifier saturates at p = 1.0 for any narrow, steady, drifting tone — at any SNR

> **Status: fixed, except the simulator class.** A hit with SNR above 3× the classifier's
> training maximum (`ai.model.ood_snr_limit`; `snr_max` is now recorded by `train()`, and the
> bundled model falls back to 40.24, reproduced by retraining with the defaults) gets `ai_class = "out_of_distribution"`, its
> p is replaced by the heuristic, and its interest is capped at 50. `share_require_cadence`
> now defaults to `None`, meaning a cadence pass is required for github/webhook but not for
> the local bundle (`--no-require-cadence` opts out). Regression tests:
> `test_far_out_of_range_snr_is_not_trusted_as_technosignature`,
> `test_uncadenced_finding_goes_to_bundle_but_not_remote_by_default`,
> `test_far_out_of_range_snr_is_capped_without_a_classifier`. The SNR cap applies whether or
> not a classifier is loaded, and `configs/default.json` no longer pins
> `share_require_cadence: false`, which had kept the new sharing default from ever applying.
> **Still open:** a strong-carrier RFI class in the simulator. It needs a retrain, so it goes
> with B3. The guard is on SNR only: checking every feature range would currently mark almost
> every production hit out of range, because of the drift gap in B3.

Related to B3 (training distribution), but a separate failure: the classifier has no notion of
being outside its training range. Measured with the bundled model on a narrow tone drifting
0.2 channels/step:

| per-sample amplitude | SNR | class | p(technosignature_like) |
|---|---|---|---|
| 3 | 11 | technosignature_like | 1.00 |
| 300 | 1,099 | technosignature_like | 1.00 |
| 3000 | 11,424 | technosignature_like | 1.00 |
| 3000, drift 0 | 10,691 | rfi_zero_drift | 0.00 |

Training SNRs are roughly 5–40. In the real-data run the hit with **SNR 144,777** was ranked #1,
interest 94, and stored as the lifetime best (`Best interest so far: 94.2`). Strong, slowly
drifting carriers are among the most common kinds of terrestrial RFI. Also, `zero_drift` is
`|drift| < 0.02` ch/step (`ai/features.py:61`), below the tree's drift resolution of 1/15
ch/step, so a single resolution step of drift is enough to count as "drifting".

Consequence for sharing: these hits pass `passes_gate` (`share.py:150`), and
`share_require_cadence` defaults to `False` (`config.py:43`), so `crunch --auto-share` would
publish them.

**Fix.** Add an out-of-distribution guard (SNR and feature ranges recorded in the model
metadata; outside them → `ai_class = "out_of_distribution"`, p not used in the score). Add a
strong-carrier RFI class with high SNR and slow drift to the simulator. Cap interest (e.g. ≤ 50)
until a cadence pass exists. Default `share_require_cadence = True` for remote destinations.

**Test.** A hit with SNR 10,000 gets `out_of_distribution` and interest ≤ 50; auto-share does not
send uncadenced candidates to remote sinks by default.

### B7 — the same signal in different targets is not recognised as interference

> **Status: (b) fixed, (a) open.** The ledger keeps `signals`: the frequencies of each
> observation's 50 strongest candidates and their target (capped at 50,000 entries).
> `rfi.flag_multi_target` flags a hit within 2 kHz of a frequency seen in a *different*
> target, cuts its interest to a fifth and re-ranks. A repeat sighting in the same target, such
> as a cadence ON scan, doesn't count. The flag is in the CSV, explained in the HTML report,
> listed in the share checklist, and blocks sharing. Regression tests:
> `test_same_signal_in_a_second_target_is_flagged_multi_target`,
> `test_same_target_seen_twice_is_not_multi_target`.
> **Still open:** (a), grouping archive files by `cadence_url` and running `cadence_filter`
> automatically. The 2 kHz tolerance is a guess (see the comment in `rfi.py`). In a band dense
> with interference it can flag a genuine hit that happens to sit near someone else's RFI.

The signals at ~7999.70 and ~8000.28–8000.30 MHz appear in **HIP2579, HIP2586 and HIP3249**.
These files sit under `LHS1140/C/` in the archive and were recorded on the same day
(MJD 57774) in consecutive scans, so they are very likely the OFF pointings of an LHS 1140
cadence. A signal present in several targets cannot come from any one of them; this is the
strongest RFI evidence available and the pipeline ignores it.

`BreakthroughListenSource` already reads `cadence_url` (`src/ai_seti/sources.py:155`), but
`crunch` never uses it, and `ai-seti cadence` only accepts local files.

**Fix.** (a) Group archive files by `cadence_url` and run `cadence_filter` automatically when a
group is complete. (b) Independently, keep a lifetime frequency index in the ledger: a hit within
tolerance of a hit already seen in a *different* target is flagged `multi_target`, and interest
is reduced.

**Test.** Two scans with different `source_name` and the same tone: the second scan's hit is
flagged `multi_target`.

### B5 — `--max-units` always crunches the top of the band, then marks the whole file done

> **Status: fixed.** The ledger keeps `progress` (observation → next unsearched channel).
> `Ledger.next_range` gives each run the next `max_units` units, and `Ledger.advance` marks the
> observation done only when its last channel is searched. Partial runs write to
> `<outdir>/<stem>_ch<start>-<stop>` so they don't overwrite each other's reports. Not done: the
> optional "spread units across the band". Regression test:
> `test_max_units_resumes_and_marks_done_only_when_whole_file_is_searched`, the first test that
> drives `crunch` through the CLI. Follow-up from review: `split` now pads units into channels
> outside `chan_range` (a chunk edge is not a data edge), and a range with a failed work unit
> is recorded as a failure instead of being advanced past; `run_units` with one worker now
> records unit errors like the process pool does instead of crashing. Tests:
> `test_partial_range_units_are_padded_beyond_the_range_edges`,
> `test_failed_work_unit_is_not_marked_searched`.

`src/ai_seti/cli.py:219` builds `rng = (0, min(header.nchans, max_units * cfg.channels_per_unit))`
— always the first units — and `cli.py:226` calls `ledger.done(url)` unconditionally afterwards.

Effect in the 2026-10-03 run: every 1,073,741,824-channel high-resolution file had
**2,097,152 channels (0.2%) searched**, always the same 5.9 MHz at 7995.6–8001.5 MHz (the
first coarse channel, at the band edge), and the file was then recorded as finished. The other
99.8% is never revisited, and the lifetime stats read as if whole observations were analysed.

**Fix.** Keep per-observation progress in the ledger (done unit ids or next channel offset);
mark an observation done only when all its units are done; on the next run continue where the
last one stopped. Optionally spread `--max-units` across the band instead of taking the first N.

**Test.** Two consecutive `crunch --max-units 2` runs on a 6-unit local file process units
0–1 then 2–3, and the observation is only marked done after the third run.

### B1 — `drift_search` silently degenerates to a zero-drift-only search

`src/ai_seti/dsp/dedoppler.py:129-132` clamps the searched drift range with no warning:

```python
k_needed = abs(max_drift_hz_s * tsamp / foff_hz) if foff_hz else 1
k_max = max(0, int(np.ceil(k_needed)) - 1)
k_max = min(k_max, max(0, max_ch_per_step - 1))
k_max = min(k_max, max(0, f // max(t, 1) - 1))
```

If any of these drive `k_max` to `0` while `max_drift_hz_s > 0`, the search evaluates only
drift `0` and reports **no hits** for anything drifting. That is the worst possible failure
shape for this program: "I found nothing" instead of "I could not look there". (The mirror
image of this — drift `0…1` ch/step always searched regardless of `k_max` — is B9.)

Measured: a GBT coarse channel (32768 channels, `tsamp=18.253611008`,
`foff=2.7939677238464355e-06` MHz = 2.794 Hz) carrying a tone that drifts 13 channels/step at
SNR 6/sample returned **0 hits** — while `_search_one_sign` had correctly integrated the track
to a path sum of `78.8` (`_noise_scale` → `med=0.039`, `scale=4.054`, so SNR ≈ 19.4). The same
call with the range intact returns `(start_channel=8000, drift=13.0, snr=19.4)` exactly.

**Fix.** Refuse to be quiet. When a non-zero `max_drift_hz_s` cannot be represented at the
given `tsamp`/`foff_hz`, log a warning naming the achievable rate and either raise or record
it in `metadata.json`. A `k_max == 0` run should be visibly a degraded run.

**Test.** Assert the warning/failure when the requested rate is unsearchable.

### B2 — `foff_mhz` has no unit validation; Hz passed where MHz is expected blinds the search

`drift_search(..., foff_mhz=...)` multiplies by `1e6` internally
(`src/ai_seti/dsp/dedoppler.py:127`). Passing `foff` already in Hz inflates it a
million-fold, `k_needed` collapses, and B1 fires silently. Measured: `foff_mhz=2.794`
(Hz) → `k_max=0`, 0 hits; `foff_mhz=2.794e-06` (MHz) → 1 hit, recovered exactly.

This exact mistake was made during the 0.3.0 review and cost ~30 minutes, because the
failure looks like a detector bug rather than a units bug.

**Fix.** Take `foff_hz` and derive MHz internally, or validate the magnitude of `foff_mhz`
and reject values that imply an implausible channel width for the time resolution.

**Test.** Assert a wrong-unit argument is rejected rather than silently accepted.

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

### B18 — mirror pairs about the coarse-channel centre have equal SNR, so B6 leaves one half at #1

> **Status: fixed.** `rfi.flag_mirror_images` now flags *both* halves `mirror_image` when their
> SNRs are within `SYMMETRIC_PAIR_RATIO` = 2 of each other, and keeps the weaker-only rule for
> unequal pairs. Re-flagging the saved candidates from the second real-data run: images 81 → 162
> (HIP2579) and 77 → 154 (HIP3249). The best interest in either file falls from 70.7 to ≤ 16,
> and every top-10 candidate is now a flagged image. Report and share-gate wording no longer
> say "weaker". Regression test: `test_symmetric_mirror_pair_flags_both_halves` (symmetric pair,
> unequal pair, unpaired tone). The density flag suggested below is not done; B21 covers it.

B6 assumed an image is a *weaker* copy ("a strong tone at `f_c + Δ` produces a weaker copy at
`f_c − Δ`") and demotes only the weaker hit of each pair. The second real-data run shows this is
wrong for these products. In the first two coarse channels of HIP2579 `.0000` and HIP3249
`.0000`:

| file | hits | pairs found | SNR ratio strong/weak (median, p90) | hits within 0.6 MHz of `f_c` |
|---|---|---|---|---|
| HIP2579 `.0000` | 166 | 81 | 1.02, 1.08 | 93% |
| HIP3249 `.0000` | 158 | 77 | 1.02, 1.07 | 92% |

Every one of the top 10 candidates in both files is the marginally stronger half of such a pair
(ratios 1.00–1.10). So the #1 candidate is still a symmetric artefact. In HIP2579 it is
8000.344291 MHz, −0.071 Hz/s, SNR 32, `technosignature_like`, **interest 70.7**: above
`share_min_interest`, the new lifetime best, and it goes into the local share bundle.
`f_c` = 7999.999999 MHz (`fch1 + foff · 2^19`) in both files, and each partner sits within one
channel of `2·f_c − f`, so the pairing itself is right. Only the "weaker" rule is wrong.

**Fix.** When a pair's SNR ratio is below a threshold (say 2), flag *both* halves
`mirror_image`: a symmetric pair is an instrument product, not a sky tone with its echo. Keep
the weaker-only rule for strongly unequal pairs. Consider also a density flag: 92–93% of hits
within ±0.6 MHz of `f_c` says that region is a spur forest in this receiver chain.

**Test.** Inject a tone and a mirror of equal amplitude; both must be flagged and drop below an
unpaired tone of the same SNR.

## Medium

### B19 — mid-res products cannot resolve drift, yet every hit is called "does not drift"

> **Status: fixed (labelling); speed open.** `sources.drift_resolvable(header, cfg)` checks
> whether `max_drift_rate_hz_s × nsamples × tsamp` reaches one channel. When it doesn't, every hit
> gets `drift_unresolved = True`. The ×0.3 zero-drift penalty is not applied. The report says
> "drift not measurable at this resolution" instead of "does not drift". The share checklist
> shows the drift check as unknown, and the gate refuses with "drift not measurable at this
> resolution (cannot rule out Earth)". `crunch` prints a notice, and `metadata.json` records
> `drift_resolvable`. Regression test: `test_midres_hits_are_drift_unresolved_not_stationary`.
> **Still open:** these products still cost ~2 min each (load-bound). Skipping them
> automatically in `crunch` is a behaviour decision and has not been made.

The `.0002` product is 2,861 Hz × 1.07 s × 272 samples. At the 4 Hz/s limit, the largest drift
over the whole observation is 4 × 292 s ≈ 1,170 Hz, which is less than one channel. So drift is
*unmeasurable*, not zero. Every hit in both `.0002` files came out `zero_drift` (13/13 and
16/16). It got the ×0.3 zero-drift penalty (all interest ≤ 10). The HTML report says "it does
not drift, which usually means a transmitter on Earth", and the share gate rejects it as "zero
drift (almost always terrestrial)". Those statements are not supported by the data.

The files are also expensive. Each took 107–121 s for 1 M channels (8.7–9.8 k channels/s,
against 118–130 k on `.0000`). One remote 66 k-channel unit timed by stage: **load 48.4 s**,
preprocess 0.25 s, de-Doppler 0.01 s, detectors 0.01 s, AI 0.38 s. So the time is the HTTP
Range streaming of 272 rows per unit, not the search (see B10).

**Fix.** Compute "maximum resolvable drift" (`foff / (nsamples · tsamp)`) per file. When the
configured rate cannot reach one channel over the observation, either skip the product with a
clear message (preferring `.0000`), or label hits `drift_unresolved` and leave out the
zero-drift penalty and the "does not drift" wording. Record the choice in `metadata.json`.

**Test.** A `.0002`-shaped header with an injected tone: the hit is not labelled `zero_drift`,
and the report text does not claim it is stationary.

### B20 — classifier training silently gets ~110 of 500 requested noise examples

`ai-seti train --per-class 500` (seed 0, 22 s) completed with test accuracy 0.957, but built only
2,611 examples instead of 3,000. The confusion matrix has 28 noise test rows against 125 for
every other class, so about 110 noise examples in total. `_training_example("noise")`
(`ai/model.py:52`) keeps an example only if pure noise produces a hit above SNR 5, which is rare.
`build_training_set` gives up after `n_per_class × 4` tries without saying so, and the progress
bar never reaches 100%. The bundled model has the same shortfall: same `n_train` = 1,958 and
identical accuracy.

**Fix.** Lower the hit threshold for noise examples only, or take the strongest noise path from
the SNR map, so the class can be filled. Log a warning, and record per-class counts in the
model metadata, when any class falls short.

**Test.** `build_training_set(n_per_class=50)` returns 50 rows of every label, or warns and
reports the shortfall.

### B10 — non-power-of-two time samples double the Taylor-tree cost

`_search_one_sign` pads time to the next power of two (`src/ai_seti/dsp/dedoppler.py:85`).
279 samples become 512, nearly half of them zeros. Measured on 32k channels: **0.70 s at
T = 279 vs 0.31 s at T = 256**. In the log, mid-res units ran at 8–9 k channels/s (110–124 s
per file), versus 125–137 k channels/s on high-resolution data.

**Update after the second run.** On mid-res data the tree no longer costs anything after B9
(de-Doppler 0.01 s per unit). The 107–121 s per file is Range streaming (load 48.4 s of 49.2 s
per unit, see B19). This item now only matters for products where a tree is needed and T is not
a power of two.

**Fix.** After B9 most mid-res runs need no tree at all. Where a tree is needed, split time into
power-of-two segments and combine, or decimate to the largest power of two below T.

**Test.** Timing guard: T = 279 costs ≤ 1.2× T = 256 on the same width.

### B11 — failed files are never recorded, and the archive query never pages

> **Status: fixed.** Checked against the live API on 2026-10-03: `offset`, `page` and `skip`
> are ignored, but the order is stable across `limit` values. So
> `BreakthroughListenSource.new_rows` widens the query (×4, up to 2,000 rows) until it holds
> `limit` rows not yet done. Failures go into the ledger's `failed` map (attempt count and last
> reason). After `state.MAX_ATTEMPTS` = 3 the file is skipped, and a later success clears the
> entry. Each pass prints "N crunched, M failed; K skipped". Regression tests:
> `test_bl_query_widens_past_rows_already_seen`,
> `test_failing_file_is_recorded_and_given_up_after_max_attempts`.

`cli.py:212-214` prints `skip` and continues, but nothing is written to the ledger, so the same
three files are fetched and fail on every run. `BreakthroughListenSource.query()`
(`src/ai_seti/sources.py:139-145`) sends no offset, so it asks for the same first
`limit = 10` rows every time (`cli.py:172`). Once those are done, the client reports
`No new observations matched` even though the archive has more data for the target.

**Fix.** Record failures in the ledger with reason and attempt count; skip after N attempts and
list them in the end-of-run summary. Page through results (check whether the API supports an
offset; otherwise widen `limit` and iterate per target from the target listing). End-of-run line:
"N done, M failed, K new".

**Test.** A source yielding one error row: the second run does not retry it after N attempts and
the summary counts it as failed.

### B12 — every `.gpuspec.8.0001.fil` fails with "Corrupt SIGPROC header string" — needs investigation

All three high-time-resolution 8-bit files in the run failed; `src/ai_seti/io/filterbank.py:100-101`
rejects a string length outside 1–255 (`if not 0 < n < 256`). Could not be reproduced from the
development sandbox (no access to `blpd0.ssl.berkeley.edu`). Hypotheses: the response is not
SIGPROC at all (HTML error page or redirect body served with `206`), or the header of this
product contains a longer or unexpected string field.

**Next step.** On failure, log the URL, HTTP status, `Content-Type` and the first 64 bytes (hex);
compare with `watutil -i` from blimpy on the same file. Fix depends on the finding. Independently,
reject non-SIGPROC responses early with a clear message rather than a parser error.

### B16 — the Streamlit drift slider above ~4.9 Hz/s has no effect on high-res GBT data

`app.py:21` offers `max_drift_rate_hz_s` from 0.5 to 10 Hz/s, but leaves
`max_drift_ch_per_step` at its default of 32 (`config.py:18`). On a GBT high-resolution product
(`tsamp` ≈ 18.25 s, `foff` ≈ 2.794 Hz) one channel/step is ≈ 0.153 Hz/s, so the cap limits
the search to ≈ 4.9 Hz/s. Both `sources.drift_ceil_ch_per_step` and `drift_search` take the
smaller of the two limits, so slider values 5–10 Hz/s silently search the same space as 4.9.

**Fix.** After the header is read, show the effective limit next to the slider
(`min(rate, cap · foff / tsamp)`), and either raise `max_drift_ch_per_step` with the slider or
clamp the slider's maximum to the effective limit.

**Test.** For a high-res header and `max_drift_rate_hz_s = 10`, the reported effective rate equals
`32 · foff_hz / tsamp`.

## Low

### B13 — `ai-seti status` says "reachable (0 sample rows)" and can hang ~3 min silently

`cli.py:161` queries with `target=""`, which returns no rows, so the check cannot tell a working
archive from an empty answer. `_get` uses `timeout = 60`, 3 retries and back-off with no output;
the first `status` call in the log was interrupted with Ctrl-C.

**Fix.** Query a known target with `limit = 1`, use a 10 s timeout for the status probe, and print
which endpoint is being tried.

### B14 — README commands fail when pasted into zsh, and `python app.py` is documented nowhere as wrong

README lines 56–88 put `# comments` after commands. macOS zsh has `interactive_comments` off by
default, so `ai-seti status   # is … reachable?` fails with `no matches found: reachable?`, and
`ai-seti crunch … # live dashboard …` fails with typer's `unexpected extra argument`. Both
happened in the log. Also `python app.py` fails (Streamlit is an optional extra, and the app must
be started with `streamlit run app.py`).

**Fix.** Put comments on their own line above each command. In the Streamlit section, state
`pip install -e ".[dashboard]"` and `streamlit run app.py`, and make `app.py` print that hint
when run with plain `python`.

### B15 — the terminal dashboard leaves blank lines and torn panels

`dashboard.py` renders a fixed-height `Layout` (32 rows). After `Live` stops, ~15 empty lines
remain, and `skip` messages printed while `Live` is active tear the region (the log shows a
half-drawn "no data yet" panel followed by a full redraw).

**Fix.** Print through `live.console.print(...)` while live, and render a `Group` sized to
content instead of a full-height `Layout`.

### B21 — multi-target matching (B7b) misses the spur forest around the coarse-channel centre

In the second run B7(b) flagged 2 of 166 hits in HIP2579 `.0000` and 59 of 158 in HIP3249
`.0000`. But the top candidates were not flagged: HIP3249's 7999.583600 and 8000.348557 MHz
sit 4–7 kHz from the nearest hits in the other targets, beyond the 2 kHz tolerance. The same
±0.6 MHz forest around `f_c` appears in all three targets (B18), so it is the same
interference. It just doesn't land on the same frequencies. Exact-frequency matching is the
wrong granularity for it.

**Fix.** Mostly resolved by B18. Optionally add a band-level check (hit density per 100 kHz
seen in other targets), and make the tolerance a config value.

### B17 — shared findings omit `max_drift_ch_per_step`, the drift cap that can bind

`share.py:36` `CONFIG_KEYS` records `max_drift_rate_hz_s` but not `max_drift_ch_per_step`.
With the defaults the rate limit binds on high-res data (≈ 26 < 32 ch/step), but a user who
lowers the cap, or searches at > 4.9 Hz/s (see B16), publishes a finding whose recorded config
does not show the limit that was actually applied, so it cannot be reproduced from the record.

**Fix.** Add `"max_drift_ch_per_step"` to `CONFIG_KEYS`.

**Test.** `build_findings()` output contains `max_drift_ch_per_step` under the config block.

---

## Real-data run 2026-10-03

`ai-seti crunch --target HIP --max-units 8` on macOS, 4–8 workers, against the Breakthrough
Listen archive.

**Worked.** Archive query, HTTP Range streaming of remote `.fil` files, header parsing for
`.0000` and `.0002` products, parallel processing, reports, ledger. High-resolution throughput
125–137 k channels/s on 8 workers.

**Did not work or misled.**

| Symptom in the log | Cause |
|---|---|
| Drifts of −276 to −1,225 Hz/s with a 4 Hz/s limit | B9 — **fixed** |
| Mid-res files take 110–124 s for 1 M channels | B10 (and B9) |
| Top candidates `technosignature_like`, interest 78–94 | B6 mirror pairs, B7 multi-target, B8 saturation |
| SNR 144,777 becomes the lifetime best | B8 |
| Every high-res file analysed at 7995.6–8001.5 MHz only | B5 |
| `.8.0001.fil` → "Corrupt SIGPROC header string", every run | B12, B11 |
| Later runs: "No new observations matched" | B5 (files marked done), B11 (no paging) |
| `status` hung once; "reachable (0 sample rows)" | B13 |
| `zsh: no matches found`, typer extra-argument error, `python app.py` | B14 |

**Bottom line.** None of the candidates from this run is credible: the top ones are a
coarse-channel image artefact present in several targets. The run is still useful as the first
end-to-end test on real data, and it exposed failure modes the synthetic benchmarks cannot.

## Real-data run 2026-10-03, second pass

Same target filter, with B4–B9, B11 and the review fixes in place:
`ai-seti crunch --target HIP --max-units 8 --limit 4 --no-live` with a fresh ledger, 10 cores.
Four observations: HIP2586 `.0002`, HIP2579 `.0000` and `.0002`, HIP3249 `.0000`. Exit 0.

**Worked.**
- **Errors:** none. 4 crunched, 0 failed, and no warnings in the log.
- **`--max-units`:** stored `progress` = 2,097,152 for each high-res file, and did not mark it
  done (B5).
- **Classifier:** loaded in every file (no `ai_note`) and scored all 353 hits.
- **Out-of-distribution guard (B8):** labelled 4–33 hits per file `out_of_distribution` and
  capped them. The SNR 144,777-style lifetime best is gone.
- **Mirror flag (B6):** fires on 162 and 154 hits. **Multi-target flag (B7b):** fires on 2 and
  59.
- **High-res throughput:** 118–130 k channels/s, unchanged.

**AI training.** `ai-seti train --per-class 500` to a scratch path: 22 s, test accuracy 0.957,
`snr_max` = 40.24 now recorded. This showed that B8's fallback constant (34.0, measured with
400 per class) was wrong for the bundled model; it is corrected to 40.24. Training also short-
changes the noise class (B20).

| Symptom | Cause |
|---|---|
| #1 candidate HIP2579 8000.344291 MHz, `technosignature_like`, interest 70.7 | B18: symmetric mirror pairs, only one half demoted |
| 92–93% of high-res hits within ±0.6 MHz of the coarse-channel centre | B18 (spur forest) |
| Every mid-res hit `zero_drift`, interest ≤ 10, "does not drift" in the report | B19 |
| Mid-res files 107–121 s each; 98% of it is loading | B19, B10 (updated) |
| Top HIP3249 hits not flagged multi-target | B21 |

**Bottom line.** The fixes did what they were meant to, but the top candidates are still
artefacts, now for a reason B6 did not anticipate (B18). Mid-res products should be skipped or
labelled honestly (B19) rather than crunched for 2 minutes each.

---

## Test debt

Total coverage is **65%** against a **60%** CI floor, so the floor does not currently bite.
These are the real gaps, largest first:

| Module | Coverage | Uncovered | Why it is open |
|---|---|---|---|
| `cli.py` | **0%** | 238 stmts | Covered by the CI smoke job instead of unit tests. Every argument parser, subcommand dispatch and terminal render path is unexercised by pytest. Also where B5, B11 and B13 live. |
| `dashboard.py` | **0%** | 124 stmts | Same. The live terminal view has no unit coverage at all. Where B15 lives. |
| `ai/model.py` | 48% | 58 stmts | `build_training_set` / `_training_example` (86-102) never run under test — the expensive path. This is where B3 lives, so a cheap version of it needs to exist before B3 can be fixed safely. |
| `sources.py` | 50% | 59 stmts | `split()` and the drift guard band (96-187) are barely tested outside the integration tests. Also where B7 and B11 live. |
| `core.py` | 58% | 39 stmts | Legacy detector paths (112-183) untested. |
| `ai/simulate.py` | 58% | 36 stmts | Injection classes 85-128 largely untested — notably the ones B3 and B8 depend on. |
| `io/remote.py` | 61% | 30 stmts | Retry/backoff branches (43-48, 86-105) only partly covered. Where B4 lives. |
| `share.py` | 93% | 13 stmts | The GitHub/webhook sink bodies (280-289, 336-343) are not exercised end to end. |

**Recommendation.** Raise the floor above 60% only after `cli.py` or `dashboard.py` gets real
tests; today 65% passes partly because two large modules are entirely unmeasured, which makes
the percentage flattering. `io/remote.py` and `sources.py` should come first — they hold the
highest-severity defects and are the easiest to test against a local server.

---

## Limitations carried forward (need verification, not code)

These are documented honestly in the README but have no plan attached:

- **BL archive tested, but only against a few targets.** B12 still needs a live reproduction;
  every other archive interaction has now run at least once.
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
spectral images around the coarse-channel centre (B6); an image of a tone drifting up drifts down.

**"The de-Doppler detector fails to find fast-drifting tones at production resolution."**
False. `_search_one_sign` integrates the true track exactly (`tree[d=0][8000] = 78.8`,
matching the manual path sum), and `drift_search` returns `(8000, 13.0 ch/step, snr 19.4)`.
The apparent failure was B2 (Hz passed as MHz) in the measurement script. The detector was
correct throughout.