# AI-SETI backlog — pending bugs and changes

Author: Inventions4All — github:TWeb79

Applies to: **v0.3.0**. Opened at the end of the 0.3.0 hardening pass
(see [implementationplan.md](implementationplan.md)).

Fixed items are removed from this file once done. As of 2026-10-03 everything from B1 to B45 is
fixed, or closed with a measured reason (B10, B40). B46–B52 (added 2026-10-03) close the gaps
to the original SETI@home analysis. The write-ups, the real-data runs of
2026-10-03 and the regression tests that pin them are in git history; the performance
measurements are kept below because they still describe how `crunch` spends its time.

Items are sorted by severity, then by the order they should be fixed in. **IDs are stable and
must not be renumbered** — other documents and the tables below cross-reference them.

Severity: **high** = can produce a wrong or misleading result · **med** = wrong behaviour in
a plausible configuration · **low** = quality, debt or hygiene.

## Triage

| ID | Severity | Summary | Area |
|---|---|---|---|
| B48 | high | No minimum drift rate: hits drifting 0.01–0.1 Hz/s count as "drifting" (SETI@home rejected < 0.086 Hz/s) | `pipeline.py`, `config.py` |
| B46 | high | Simulator and benchmarks ignore drift smearing, so sensitivity above 1 channel/step is overstated | `ai/simulate.py`, `benchmark.py`, README |
| B49 | high | No back-end birdies: nobody measures how many ET-like signals the RFI filters and scoring throw away | new `birdies.py`, `pipeline.py`, `report.py` |
| B47 | med | Single frequency resolution; SETI@home searched 15 (0.075–1221 Hz) | `dsp/dedoppler.py`, `pipeline.py` |
| B51 | med | RFI zones are a static list; SETI@home learned them from the statistics of its own detections | `rfi.py`, `state.py` |
| B50 | med | No multiplets: repeat detections at the same target are never linked into stronger candidates | new `multiplets.py`, `state.py`, `report.py` |
| B52 | low | No narrowband pulse search (folding, triplets, autocorrelation) on high-time-resolution products | `dsp/detectors.py`, `sources.py` |

Recommended order: **B48 → B46 → B49 → B47 → B51 → B50 → B52.** B48 is a one-line rule with an
immediate effect on ranking. B46 makes the sensitivity claims true before anything is built on
them. B49 is the measuring instrument every later item needs (each later fix must report its
birdie loss). B50 depends on B49 for scoring and on B51 for clean inputs.

These seven items come from a comparison with the original SETI@home analysis, see
[Gap analysis vs SETI@home](#gap-analysis-vs-setihome-2026-10-03) below.

---

# Bugs

## High

### B48 — no minimum drift rate: near-stationary tones are treated as drifting — high

**Where.** `pipeline.py:97` sets `stationary` only when the total drift over the scan is ≤ 1
channel. At GBT high resolution (2.79 Hz × 18.25 s × 16 samples) that is **≈ 0.01 Hz/s**.

**Why it matters.** Terrestrial transmitters share the telescope's frame, so they show (almost)
no drift. SETI@home flagged every spike and Gaussian with |drift| < **0.086 Hz/s** as RFI
(Anderson et al. 2025, §6.4.1), and recent Breakthrough Listen searches use a minimum of
**±0.1 Hz/s** for the same reason. The mirror-image pairs from the 2026-10-03 run drifted at
0.031–0.082 Hz/s and were ranked as the best ET-like candidates before `mirror_image` existed;
both published rules would have rejected them outright.

A barycentric (Doppler-corrected) beacon still shows the Earth's own drift, ν · a⊕/c. With
a⊕ ≈ 0.034 m s⁻² (rotation at mid latitude) that is ≈ 0.16 Hz/s at 1.42 GHz (matches the
−0.16 Hz/s SETI@home measured at Arecibo) and ≈ 0.9 Hz/s at 8 GHz, scaled by the projection on
the line of sight. A fixed cut is therefore too strict at high frequency and too loose at low.

**Fix.**
- New config `min_drift_fraction` (default 0.5) and `min_drift_floor_hz_s` (default 0.05).
  Flag a hit `low_drift` when `|drift_rate_hz_s| < max(min_drift_floor_hz_s,
  min_drift_fraction × ν × 1.13e-10)`. At 1.42 GHz this gives 0.08 Hz/s, close to SETI@home.
- `low_drift` multiplies interest like `stationary` does (0.3), is listed in the share
  checklist, and fails the share gate. Flag, don't delete, so B49 can count it.
- `drift_unresolved` hits (mid-res) keep their current handling.

**Test.** Hits at 1420 MHz with 0.05 Hz/s → `low_drift`; 0.2 Hz/s → not. At 8 GHz, 0.3 Hz/s →
`low_drift`; 1.0 Hz/s → not. Replay the 2026-10-03 mirror pairs: all `low_drift`.

### B46 — drift smearing is not modelled, so sensitivity above 1 channel/step is overstated — high

**Where.** `ai/simulate.py:61` (`inject`) puts each injected tone into one channel per time step,
however fast it drifts. `benchmark.py` and the classifier training inherit this.

**Why it matters.** A real tone drifting *d* channels per step sweeps across *d* channels *within*
each integration, so each channel gets ≈ 1/*d* of the power. The incoherent Taylor tree adds one
channel per step and loses that power; this is a known limitation of tree de-Doppler searches
above one pixel per time step (turboSETI has the same weakness). SETI@home avoided it with
*coherent* de-drifting of raw voltages before its FFTs, which is impossible on filterbank data.

**Measured (2026-10-03, `drift_search` on `main@6d4f1c3`, T = 16, GBT high-res, mean of 6 trials,
same total power):**

| drift (ch/step) | ≈ Hz/s | SNR, unsmeared (as simulated today) | SNR, smeared (realistic) | smeared, channels binned × d |
|---|---|---|---|---|
| 2 | 0.31 | 12.1 | 6.5 | 7.8 |
| 4 | 0.61 | 12.3 | 4.2 | 5.8 |
| 8 | 1.22 | 11.7 | 3.7 | 4.4 |

So a tone that the benchmark finds at SNR 12 is **below threshold** in reality as soon as it drifts
more than ~1 channel/step (≈ 0.15 Hz/s at high resolution). At the advertised 4 Hz/s (≈ 26
ch/step) only very strong signals survive. The README recovery table and the "±4 Hz/s" range
describe the search grid, not the sensitivity.

**Fix.**
1. `inject`: integrate the line profile over the frequency swept during each sample (sum of the
   sinc²/Gaussian profile over `[c(t), c(t)+d]`, normalised to the same total power). Keep the
   old behaviour behind `smear=False` for regression comparison only.
2. Rerun `ai-seti benchmark` with smearing; add a "sensitivity vs drift rate" table (minimum
   SNR per sample for 50% and 90% recovery) to the README and `metadata.json`.
3. Retrain the classifier on smeared injections (B3 range) and re-measure accuracy per drift bin.
4. Partial remedy is B47 (binning recovers ≈ √d of the loss, see last column).

**Test.** `inject` with drift 4 ch/step spreads power over ≥ 4 channels per row with unchanged
total; benchmark recovery at drift 8 ch/step drops versus today's numbers (pins the new
physics).

### B49 — no back-end birdies: filter and scoring losses are unmeasured — high

**Why it matters.** Every RFI rule added since 0.3.0 (`mirror_image`, `multi_target`,
`out_of_distribution`, cadence, B48's `low_drift`, B51's zones) can also remove a real signal, and
nothing measures how often. SETI@home's answer was "candidate birdies": synthetic persistent ET
signals injected *before* RFI removal, then tracked to see whether they survive and rank highly.
They kept the fraction of birdie detections removed by RFI filtering below about 10%, and used
birdies to develop and choose the scoring functions (Anderson et al. 2025, §5, §6.6, §9). A
search that cannot say how much it throws away cannot say what "nothing found" means.

**Fix.** New module `birdies.py` and option `crunch/analyze --birdies N`:
1. **Inject into real data.** After `load_unit`, add N smeared (B46) ET-like tones per
   observation at random frequencies and drifts drawn from a planetary-motion range (not just
   uniform), with powers on a grid around threshold. Seeded and logged with ids.
2. **Track** each birdie through every stage: detected? which flags (`mirror_image`,
   `multi_target`, `out_of_distribution`, `low_drift`, `known_rfi_band`, zone, cadence)? final
   rank and interest?
3. **Keep them out of the science outputs**: birdie hits go to `birdies.csv`, never into
   `candidates.csv`, the ledger's `signals`, sharing, or the lifetime best.
4. **Report** per run and cumulatively: recovery vs power, loss per filter, fraction of birdies
   in the top 10 / top 100. Store in `metadata.json` and a report panel.
5. **Gate future work on it.** Any new filter or score change must report its birdie loss;
   target ≤ 10% total filter loss at powers above the 90%-detection level (the SETI@home figure).
6. Also add the mirror image of a few birdies on purpose, to confirm `mirror_image` flags the
   weaker copy and not the original.

**Test.** A run with 20 birdies on a local synthetic file: `birdies.csv` has 20 rows with stage
outcomes; `candidates.csv` contains none of them; the ledger is unchanged by birdie hits.

## Performance of `crunch --forever` (analysed 2026-10-03)

Measured on this machine (10 cores) against the Breakthrough Listen archive, with the
per-stage timings every run stores in `metadata.json` (25 real runs) plus targeted probes.

| What | Measured |
|---|---|
| Share of run time spent **loading data** | **92–99%** (high-res 92–97%, mid-res 96–99%); preprocessing 0–1%, de-Doppler 2–5%, AI 0–4% |
| High-res throughput (`.0000`, 16 samples) | 117–138 k channels/s ≈ **7–8 MB/s**; 8 units of 262,144 channels in 15–18 s |
| One HTTP Range request | ~400 ms to the first byte; 1 MB at **0.2 MB/s**, 4 MB at 0.5 MB/s, 16 MB at 1.9 MB/s |
| Parallel 1 MB requests | 1: 0.5 MB/s · 4: 2.0 · 8: 4.3 · 16: 6.6 · 32: **7.2 MB/s** (levels off: a link or server ceiling) |
| Requests in flight today | 8 worker processes × 8 threads = **64**, each a new connection |
| Process pool + model load per observation | **~2 s** (8 workers, 1.6–1.7 s each) |
| File sizes (HIP2579, one scan) | `.0000` **64 GB** (1,073,741,824 ch × 16 samples × float32) · `.0002` 1.06 GB · `.8.0001` 6.4 GB. No HDF5 version is offered for these scans. |
| Time to stream one whole `.0000` file at ~8 MB/s | **~2.2 h**; `--max-units 8` reads 128 MB (0.2%) per visit, ~512 visits per file |

The search itself is not the bottleneck: the CPU workers wait on the network ~95% of the
time. The levers are, in order: don't sleep while work is waiting (B38), make each byte cheaper
to fetch (B40), don't fetch the same bytes twice (B41), and stop paying fixed costs per
observation (B42, B43).

## Medium

### B47 — single frequency resolution; SETI@home searched a ladder of 15 — med

**Why it matters.** SETI@home analysed every work unit at 15 DFT lengths, from 1221 Hz down to
0.075 Hz resolution, because long DFTs suit continuous narrow signals and short ones suit short or
pulsed signals (Anderson et al. 2025, §3.3). This repo searches only the file's native resolution
(2.79 Hz on `.0000`). A signal wider than a channel loses SNR roughly as √(bandwidth / 2.79 Hz):
a 50 Hz-wide carrier is found at ~¼ of its possible SNR, and B46 shows fast drifts smear the same
way.

**Fix.**
- In `process_work_unit`, after `normalize`, run `drift_search` on the native spectrum and on
  spectra binned ×2, ×4, …, ×64 in frequency (sum b channels, divide by √b; pass `foff × b`).
  Total extra cost is a geometric series, < 1× the native search (each level has 1/b of the
  channels and 1/b of the drift trials).
- Optionally also sum adjacent time samples for very slow drifts (SETI@home's longest DFTs are
  the equivalent); skip when it would leave < 4 samples.
- Merge hits across levels: cluster by frequency (± b channels) and drift; keep the level with
  the highest SNR; record `resolution_hz` and `bandwidth_hz` for the hit.
- Scoring: a hit found best at a coarse level is "wide"; feed `resolution_hz` to the classifier
  (retrain) instead of today's `bandwidth_ch` at native resolution only.
- Cache-friendly: run levels per tile so data stays hot (same pattern as the existing tiling).

**Test.** Inject a 40 Hz-wide tone: ladder SNR ≥ 2× native-only, reported `resolution_hz` within
×2 of 40 Hz. Inject a 1-channel tone: the native level wins and no duplicate hit appears at other
levels. Noise only: false-alarm rate per run rises by no more than the extra trials predict
(calibrate thresholds per level like `benchmark.calibrate`).

### B51 — RFI zones are a hand-written list; SETI@home learned them from data — med

**Where.** `config.py:58` `known_rfi_bands_mhz` (static, L-band centric). The ledger already keeps
`signals: [frequency_mhz, target]` (`state.py:35`) but only `flag_multi_target` reads it.

**Why it matters.** SETI@home split the band into small zones, measured for each zone how often it
showed a statistical excess of detections, and removed the worst zones (Anderson et al. 2025,
§6.1.1). That catches every local transmitter, including ones nobody listed, and adapts per
telescope and per year. The static list knows nothing about 4–8 GHz, where most of the
2026-10-03 runs were.

**Fix.**
- `state.py`: keep a compact histogram of hit counts per 1 kHz zone per observation (not raw
  hits), plus observation counts per zone, so the ledger stays small.
- `rfi.py: learned_zones(ledger, min_observations, p_threshold, max_band_fraction)`: zones whose
  hit rate across *different targets* is improbably high under a Poisson model with the median
  zone rate; cap the removed fraction of band (e.g. 5%).
- Flag hits `rfi_zone` (interest × 0.3), shown in the report and share checklist; keep the static
  list as a seed.
- Export/import zones as JSON so volunteers can share them (fits the share feature).
- Measure the birdie loss with B49 before enabling by default.

**Test.** A ledger where one 1 kHz zone has hits in 8 of 10 targets and others in ≤ 1: that zone
is learned, a new hit there is flagged `rfi_zone`; the removed band fraction stays under the cap.

### B50 — no multiplets: repeat detections at the same target are never combined — med

**Why it matters.** SETI@home's candidates were never single detections but *multiplets*: groups
of detections from one sky position whose frequencies and drifts are consistent with one source,
possibly spread over 14 years, scored by power, density and time coverage. Single-detection
"singlets" turned out to be essentially all RFI, so a multiplet needed at least two detections
(Anderson et al. 2025, §7). This repo ranks each observation in isolation, and the ledger is
used only to *reject* signals seen at other targets (B7 / `multi_target`), never to *confirm*
signals seen again at the *same* target. Breakthrough Listen targets are often observed more than
once (different days, bands, cadences), so the data supports it.

**Fix.**
1. Ledger: store each surviving candidate (not flagged RFI) with target, sky position, MJD,
   topocentric frequency, drift, SNR, resolution (B47) and observation id.
2. Barycentric correction for grouping: convert frequency to the barycentre using target RA/Dec,
   site and time (astropy as an optional extra; skip grouping when unavailable). SETI@home
   grouped barycentric signals within ~250 Hz and non-barycentric ones within a wider window with
   drift-consistency rules.
3. `multiplets.py`: per target, link detections from **different observations** within the
   frequency tolerance and with consistent drift (same sign, |Δdrift| bounded). Require ≥ 2
   observations.
4. Score like SETI@home: power (median normalised SNR), density (Poisson probability of that
   many detections in that frequency span given the target's hit rate), time (fraction of the
   observed time the signal is present). Report each factor and the combinations, not one opaque
   number (SETI@home found each variant best for different birdies).
5. Output `multiplets.csv`, a report section, and a share record type `multiplet` (stronger
   evidence than a single candidate; still `unverified`).
6. Later, optional: an AI clustering step for grouping, which the SETI@home authors name as the
   natural next improvement. Validate on held-out birdie families only (B49); their attempt to
   learn score weights with neural networks failed because weights fitted to one birdie set did
   not transfer.

**Test.** Two synthetic observations of one target a week apart with the same barycentric tone →
one multiplet; the same tone at two different targets → none (it is `multi_target`); a
single-observation candidate never forms a multiplet.

## Low

### B52 — no narrowband pulse search (folding, triplets, autocorrelation) — low

**Why it matters.** Besides spikes and Gaussians, SETI@home searched for narrowband pulses with a
folding algorithm, triplets of equally spaced pulses, and autocorrelation for repeating structure
(Anderson et al. 2025, §3.3). This repo's `find_pulses` looks for *broadband dispersed* bursts
(the Astropulse idea), a different signal class. On `.0000` files (16 samples of 18 s) periodic
narrowband pulses can't be searched at all; it is only possible on the high-time-resolution
`.8.0001` product, which `sources.py` currently skips for size (6.4 GB per scan).

**Fix.** Only after B47–B50.
- Per channel group on high-time-res data: fast folding over a period grid (e.g. 0.05–10 s),
  plus a triplet detector (three equal-power peaks equally spaced) and a short-lag
  autocorrelation.
- Stream `.8.0001` per channel window like `.0000` (header-aware), or support a user-supplied
  local file only.
- Add period-zone RFI (SETI@home saw radar, calibration and computer-interrupt periods) via the
  B51 machinery keyed on period instead of frequency.

**Test.** Injected narrowband pulse train (period 0.8 s, duty 10%) in a synthetic high-time-res
window is found within 2% of its period; noise-only false alarms stay at the calibrated rate.

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

- **No coherent de-drifting.** SETI@home de-drifted raw complex voltages *before* its FFTs, at
  about 123,000 drift rates over ±100 Hz/s, which keeps a fast-drifting tone in one bin. Public
  Breakthrough Listen filterbank files carry power only (no phase), so this repo can only do the
  incoherent Taylor tree. B46 measures the cost, B47 recovers part of it. Revisit only if
  baseband (raw voltage) data becomes available.

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

## Gap analysis vs SETI@home (2026-10-03)

Compared against the published description of SETI@home's front end (Korpela et al. 2025, AJ,
doi:10.3847/1538-3881/ade5a7) and back end (Anderson et al. 2025, "SETI@home: Data Analysis and
Findings", arXiv:2506.14737), on `main@6d4f1c3`.

| SETI@home | AI-SETI today | Status |
|---|---|---|
| Raw complex voltages, 2.5 MHz | BL filterbank power spectra | Different by necessity (no phase) |
| Work units with time overlap | Channel windows with drift guard band | Equivalent |
| Coherent de-drift, ~123k rates, ±100 Hz/s | Incoherent Taylor tree, ±4 Hz/s (BL/turboSETI standard) | Modern standard; loss measured in **B46** |
| 15 resolutions, 0.075–1221 Hz | Native resolution only | **B47** |
| Spikes | Spikes + de-Doppler hits | Covered |
| Gaussians (beam passing over source) | ON/OFF cadence (BL tracks targets) | Equivalent role |
| Pulses (folding), triplets, autocorrelation | Broadband dispersed pulses only | **B52** |
| Low-drift RFI cut at 0.086 Hz/s | `stationary` ≈ 0.01 Hz/s | **B48** |
| Learned frequency-zone RFI | Static band list | **B51** |
| "Same signal at different sky positions" RFI rule | `multi_target`, cadence discovery | Covered |
| Multiplets across observations, scored by power/density/time | Per-observation ranking only | **B50** |
| Candidate birdies through the whole back end | Front-end injection + AI training only | **B49** |
| Manual review + re-observation | Share records with checklist | Analogous |
| No AI in the client; NN score weighting tried and failed | Per-hit classifier + isolation forest | Different place; validate with **B49** |

---

## Verified — *not* bugs


**"Non-power-of-two time samples double the Taylor-tree cost" (B10).** True (0.70 s at T = 279
vs 0.31 s at T = 256 on 32k channels), but closed 2026-10-03: no Breakthrough Listen product
reaches that path any more. High-res files have T = 16, and mid-res files skip the tree
because drift is unmeasurable there (and `crunch` now skips them). Revisit only if a product
with a non-power-of-two T *and* measurable drift turns up.

**"Keep-alive connections, fewer threads or prefetching will speed up `crunch`."** Tried and
measured 2026-10-03 (B40), then reverted. A kept-alive connection carries about twice the data
of a fresh one (0.22 vs 0.125 MB/s), but the total levels off at the same **~7–8 MB/s** link or
server ceiling, which today's 8 workers × 8 fresh connections already reach. Four units in a
row in one worker: 62 s old vs 74 s with 4 kept-alive threads. Two full 8-unit runs with 8
kept-alive threads: 20.4 s and 23.8 s, against 15–18 s before. Analysis is ≤ 5% of run time,
so prefetching can't win more than that. Gains have to come from not idling (B38, done), not
downloading twice (B41) and fixed costs (B42, B43).
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

# AI-SETI backlog — pending bugs and changes

Author: Inventions4All — github:TWeb79

Applies to: **v0.3.0**. Opened at the end of the 0.3.0 hardening pass
(see [implementationplan.md](implementationplan.md)).

Fixed items are removed from this file once done. As of 2026-10-03 everything from B1 to B45 is
fixed, or closed with a measured reason (B10, B40). B46–B52 (added 2026-10-03) close the gaps
to the original SETI@home analysis. The write-ups, the real-data runs of
2026-10-03 and the regression tests that pin them are in git history; the performance
measurements are kept below because they still describe how `crunch` spends its time.

Items are sorted by severity, then by the order they should be fixed in. **IDs are stable and
must not be renumbered** — other documents and the tables below cross-reference them.

Severity: **high** = can produce a wrong or misleading result · **med** = wrong behaviour in
a plausible configuration · **low** = quality, debt or hygiene.

## Triage

| ID | Severity | Summary | Area |
|---|---|---|---|
| B48 | high | No minimum drift rate: hits drifting 0.01–0.1 Hz/s count as "drifting" (SETI@home rejected < 0.086 Hz/s) | `pipeline.py`, `config.py` |
| B46 | high | Simulator and benchmarks ignore drift smearing, so sensitivity above 1 channel/step is overstated | `ai/simulate.py`, `benchmark.py`, README |
| B49 | high | No back-end birdies: nobody measures how many ET-like signals the RFI filters and scoring throw away | new `birdies.py`, `pipeline.py`, `report.py` |
| B47 | med | Single frequency resolution; SETI@home searched 15 (0.075–1221 Hz) | `dsp/dedoppler.py`, `pipeline.py` |
| B51 | med | RFI zones are a static list; SETI@home learned them from the statistics of its own detections | `rfi.py`, `state.py` |
| B50 | med | No multiplets: repeat detections at the same target are never linked into stronger candidates | new `multiplets.py`, `state.py`, `report.py` |
| B52 | low | No narrowband pulse search (folding, triplets, autocorrelation) on high-time-resolution products | `dsp/detectors.py`, `sources.py` |

Recommended order: **B48 → B46 → B49 → B47 → B51 → B50 → B52.** B48 is a one-line rule with an
immediate effect on ranking. B46 makes the sensitivity claims true before anything is built on
them. B49 is the measuring instrument every later item needs (each later fix must report its
birdie loss). B50 depends on B49 for scoring and on B51 for clean inputs.

These seven items come from a comparison with the original SETI@home analysis, see
[Gap analysis vs SETI@home](#gap-analysis-vs-setihome-2026-10-03) below.

---

# Bugs

## High

### B48 — no minimum drift rate: near-stationary tones are treated as drifting — high

**Where.** `pipeline.py:97` sets `stationary` only when the total drift over the scan is ≤ 1
channel. At GBT high resolution (2.79 Hz × 18.25 s × 16 samples) that is **≈ 0.01 Hz/s**.

**Why it matters.** Terrestrial transmitters share the telescope's frame, so they show (almost)
no drift. SETI@home flagged every spike and Gaussian with |drift| < **0.086 Hz/s** as RFI
(Anderson et al. 2025, §6.4.1), and recent Breakthrough Listen searches use a minimum of
**±0.1 Hz/s** for the same reason. The mirror-image pairs from the 2026-10-03 run drifted at
0.031–0.082 Hz/s and were ranked as the best ET-like candidates before `mirror_image` existed;
both published rules would have rejected them outright.

A barycentric (Doppler-corrected) beacon still shows the Earth's own drift, ν · a⊕/c. With
a⊕ ≈ 0.034 m s⁻² (rotation at mid latitude) that is ≈ 0.16 Hz/s at 1.42 GHz (matches the
−0.16 Hz/s SETI@home measured at Arecibo) and ≈ 0.9 Hz/s at 8 GHz, scaled by the projection on
the line of sight. A fixed cut is therefore too strict at high frequency and too loose at low.

**Fix.**
- New config `min_drift_fraction` (default 0.5) and `min_drift_floor_hz_s` (default 0.05).
  Flag a hit `low_drift` when `|drift_rate_hz_s| < max(min_drift_floor_hz_s,
  min_drift_fraction × ν × 1.13e-10)`. At 1.42 GHz this gives 0.08 Hz/s, close to SETI@home.
- `low_drift` multiplies interest like `stationary` does (0.3), is listed in the share
  checklist, and fails the share gate. Flag, don't delete, so B49 can count it.
- `drift_unresolved` hits (mid-res) keep their current handling.

**Test.** Hits at 1420 MHz with 0.05 Hz/s → `low_drift`; 0.2 Hz/s → not. At 8 GHz, 0.3 Hz/s →
`low_drift`; 1.0 Hz/s → not. Replay the 2026-10-03 mirror pairs: all `low_drift`.

### B46 — drift smearing is not modelled, so sensitivity above 1 channel/step is overstated — high

**Where.** `ai/simulate.py:61` (`inject`) puts each injected tone into one channel per time step,
however fast it drifts. `benchmark.py` and the classifier training inherit this.

**Why it matters.** A real tone drifting *d* channels per step sweeps across *d* channels *within*
each integration, so each channel gets ≈ 1/*d* of the power. The incoherent Taylor tree adds one
channel per step and loses that power; this is a known limitation of tree de-Doppler searches
above one pixel per time step (turboSETI has the same weakness). SETI@home avoided it with
*coherent* de-drifting of raw voltages before its FFTs, which is impossible on filterbank data.

**Measured (2026-10-03, `drift_search` on `main@6d4f1c3`, T = 16, GBT high-res, mean of 6 trials,
same total power):**

| drift (ch/step) | ≈ Hz/s | SNR, unsmeared (as simulated today) | SNR, smeared (realistic) | smeared, channels binned × d |
|---|---|---|---|---|
| 2 | 0.31 | 12.1 | 6.5 | 7.8 |
| 4 | 0.61 | 12.3 | 4.2 | 5.8 |
| 8 | 1.22 | 11.7 | 3.7 | 4.4 |

So a tone that the benchmark finds at SNR 12 is **below threshold** in reality as soon as it drifts
more than ~1 channel/step (≈ 0.15 Hz/s at high resolution). At the advertised 4 Hz/s (≈ 26
ch/step) only very strong signals survive. The README recovery table and the "±4 Hz/s" range
describe the search grid, not the sensitivity.

**Fix.**
1. `inject`: integrate the line profile over the frequency swept during each sample (sum of the
   sinc²/Gaussian profile over `[c(t), c(t)+d]`, normalised to the same total power). Keep the
   old behaviour behind `smear=False` for regression comparison only.
2. Rerun `ai-seti benchmark` with smearing; add a "sensitivity vs drift rate" table (minimum
   SNR per sample for 50% and 90% recovery) to the README and `metadata.json`.
3. Retrain the classifier on smeared injections (B3 range) and re-measure accuracy per drift bin.
4. Partial remedy is B47 (binning recovers ≈ √d of the loss, see last column).

**Test.** `inject` with drift 4 ch/step spreads power over ≥ 4 channels per row with unchanged
total; benchmark recovery at drift 8 ch/step drops versus today's numbers (pins the new
physics).

### B49 — no back-end birdies: filter and scoring losses are unmeasured — high

**Why it matters.** Every RFI rule added since 0.3.0 (`mirror_image`, `multi_target`,
`out_of_distribution`, cadence, B48's `low_drift`, B51's zones) can also remove a real signal, and
nothing measures how often. SETI@home's answer was "candidate birdies": synthetic persistent ET
signals injected *before* RFI removal, then tracked to see whether they survive and rank highly.
They kept the fraction of birdie detections removed by RFI filtering below about 10%, and used
birdies to develop and choose the scoring functions (Anderson et al. 2025, §5, §6.6, §9). A
search that cannot say how much it throws away cannot say what "nothing found" means.

**Fix.** New module `birdies.py` and option `crunch/analyze --birdies N`:
1. **Inject into real data.** After `load_unit`, add N smeared (B46) ET-like tones per
   observation at random frequencies and drifts drawn from a planetary-motion range (not just
   uniform), with powers on a grid around threshold. Seeded and logged with ids.
2. **Track** each birdie through every stage: detected? which flags (`mirror_image`,
   `multi_target`, `out_of_distribution`, `low_drift`, `known_rfi_band`, zone, cadence)? final
   rank and interest?
3. **Keep them out of the science outputs**: birdie hits go to `birdies.csv`, never into
   `candidates.csv`, the ledger's `signals`, sharing, or the lifetime best.
4. **Report** per run and cumulatively: recovery vs power, loss per filter, fraction of birdies
   in the top 10 / top 100. Store in `metadata.json` and a report panel.
5. **Gate future work on it.** Any new filter or score change must report its birdie loss;
   target ≤ 10% total filter loss at powers above the 90%-detection level (the SETI@home figure).
6. Also add the mirror image of a few birdies on purpose, to confirm `mirror_image` flags the
   weaker copy and not the original.

**Test.** A run with 20 birdies on a local synthetic file: `birdies.csv` has 20 rows with stage
outcomes; `candidates.csv` contains none of them; the ledger is unchanged by birdie hits.

## Performance of `crunch --forever` (analysed 2026-10-03)

Measured on this machine (10 cores) against the Breakthrough Listen archive, with the
per-stage timings every run stores in `metadata.json` (25 real runs) plus targeted probes.

| What | Measured |
|---|---|
| Share of run time spent **loading data** | **92–99%** (high-res 92–97%, mid-res 96–99%); preprocessing 0–1%, de-Doppler 2–5%, AI 0–4% |
| High-res throughput (`.0000`, 16 samples) | 117–138 k channels/s ≈ **7–8 MB/s**; 8 units of 262,144 channels in 15–18 s |
| One HTTP Range request | ~400 ms to the first byte; 1 MB at **0.2 MB/s**, 4 MB at 0.5 MB/s, 16 MB at 1.9 MB/s |
| Parallel 1 MB requests | 1: 0.5 MB/s · 4: 2.0 · 8: 4.3 · 16: 6.6 · 32: **7.2 MB/s** (levels off: a link or server ceiling) |
| Requests in flight today | 8 worker processes × 8 threads = **64**, each a new connection |
| Process pool + model load per observation | **~2 s** (8 workers, 1.6–1.7 s each) |
| File sizes (HIP2579, one scan) | `.0000` **64 GB** (1,073,741,824 ch × 16 samples × float32) · `.0002` 1.06 GB · `.8.0001` 6.4 GB. No HDF5 version is offered for these scans. |
| Time to stream one whole `.0000` file at ~8 MB/s | **~2.2 h**; `--max-units 8` reads 128 MB (0.2%) per visit, ~512 visits per file |

The search itself is not the bottleneck: the CPU workers wait on the network ~95% of the
time. The levers are, in order: don't sleep while work is waiting (B38), make each byte cheaper
to fetch (B40), don't fetch the same bytes twice (B41), and stop paying fixed costs per
observation (B42, B43).

## Medium

### B47 — single frequency resolution; SETI@home searched a ladder of 15 — med

**Why it matters.** SETI@home analysed every work unit at 15 DFT lengths, from 1221 Hz down to
0.075 Hz resolution, because long DFTs suit continuous narrow signals and short ones suit short or
pulsed signals (Anderson et al. 2025, §3.3). This repo searches only the file's native resolution
(2.79 Hz on `.0000`). A signal wider than a channel loses SNR roughly as √(bandwidth / 2.79 Hz):
a 50 Hz-wide carrier is found at ~¼ of its possible SNR, and B46 shows fast drifts smear the same
way.

**Fix.**
- In `process_work_unit`, after `normalize`, run `drift_search` on the native spectrum and on
  spectra binned ×2, ×4, …, ×64 in frequency (sum b channels, divide by √b; pass `foff × b`).
  Total extra cost is a geometric series, < 1× the native search (each level has 1/b of the
  channels and 1/b of the drift trials).
- Optionally also sum adjacent time samples for very slow drifts (SETI@home's longest DFTs are
  the equivalent); skip when it would leave < 4 samples.
- Merge hits across levels: cluster by frequency (± b channels) and drift; keep the level with
  the highest SNR; record `resolution_hz` and `bandwidth_hz` for the hit.
- Scoring: a hit found best at a coarse level is "wide"; feed `resolution_hz` to the classifier
  (retrain) instead of today's `bandwidth_ch` at native resolution only.
- Cache-friendly: run levels per tile so data stays hot (same pattern as the existing tiling).

**Test.** Inject a 40 Hz-wide tone: ladder SNR ≥ 2× native-only, reported `resolution_hz` within
×2 of 40 Hz. Inject a 1-channel tone: the native level wins and no duplicate hit appears at other
levels. Noise only: false-alarm rate per run rises by no more than the extra trials predict
(calibrate thresholds per level like `benchmark.calibrate`).

### B51 — RFI zones are a hand-written list; SETI@home learned them from data — med

**Where.** `config.py:58` `known_rfi_bands_mhz` (static, L-band centric). The ledger already keeps
`signals: [frequency_mhz, target]` (`state.py:35`) but only `flag_multi_target` reads it.

**Why it matters.** SETI@home split the band into small zones, measured for each zone how often it
showed a statistical excess of detections, and removed the worst zones (Anderson et al. 2025,
§6.1.1). That catches every local transmitter, including ones nobody listed, and adapts per
telescope and per year. The static list knows nothing about 4–8 GHz, where most of the
2026-10-03 runs were.

**Fix.**
- `state.py`: keep a compact histogram of hit counts per 1 kHz zone per observation (not raw
  hits), plus observation counts per zone, so the ledger stays small.
- `rfi.py: learned_zones(ledger, min_observations, p_threshold, max_band_fraction)`: zones whose
  hit rate across *different targets* is improbably high under a Poisson model with the median
  zone rate; cap the removed fraction of band (e.g. 5%).
- Flag hits `rfi_zone` (interest × 0.3), shown in the report and share checklist; keep the static
  list as a seed.
- Export/import zones as JSON so volunteers can share them (fits the share feature).
- Measure the birdie loss with B49 before enabling by default.

**Test.** A ledger where one 1 kHz zone has hits in 8 of 10 targets and others in ≤ 1: that zone
is learned, a new hit there is flagged `rfi_zone`; the removed band fraction stays under the cap.

### B50 — no multiplets: repeat detections at the same target are never combined — med

**Why it matters.** SETI@home's candidates were never single detections but *multiplets*: groups
of detections from one sky position whose frequencies and drifts are consistent with one source,
possibly spread over 14 years, scored by power, density and time coverage. Single-detection
"singlets" turned out to be essentially all RFI, so a multiplet needed at least two detections
(Anderson et al. 2025, §7). This repo ranks each observation in isolation, and the ledger is
used only to *reject* signals seen at other targets (B7 / `multi_target`), never to *confirm*
signals seen again at the *same* target. Breakthrough Listen targets are often observed more than
once (different days, bands, cadences), so the data supports it.

**Fix.**
1. Ledger: store each surviving candidate (not flagged RFI) with target, sky position, MJD,
   topocentric frequency, drift, SNR, resolution (B47) and observation id.
2. Barycentric correction for grouping: convert frequency to the barycentre using target RA/Dec,
   site and time (astropy as an optional extra; skip grouping when unavailable). SETI@home
   grouped barycentric signals within ~250 Hz and non-barycentric ones within a wider window with
   drift-consistency rules.
3. `multiplets.py`: per target, link detections from **different observations** within the
   frequency tolerance and with consistent drift (same sign, |Δdrift| bounded). Require ≥ 2
   observations.
4. Score like SETI@home: power (median normalised SNR), density (Poisson probability of that
   many detections in that frequency span given the target's hit rate), time (fraction of the
   observed time the signal is present). Report each factor and the combinations, not one opaque
   number (SETI@home found each variant best for different birdies).
5. Output `multiplets.csv`, a report section, and a share record type `multiplet` (stronger
   evidence than a single candidate; still `unverified`).
6. Later, optional: an AI clustering step for grouping, which the SETI@home authors name as the
   natural next improvement. Validate on held-out birdie families only (B49); their attempt to
   learn score weights with neural networks failed because weights fitted to one birdie set did
   not transfer.

**Test.** Two synthetic observations of one target a week apart with the same barycentric tone →
one multiplet; the same tone at two different targets → none (it is `multi_target`); a
single-observation candidate never forms a multiplet.

## Low

### B52 — no narrowband pulse search (folding, triplets, autocorrelation) — low

**Why it matters.** Besides spikes and Gaussians, SETI@home searched for narrowband pulses with a
folding algorithm, triplets of equally spaced pulses, and autocorrelation for repeating structure
(Anderson et al. 2025, §3.3). This repo's `find_pulses` looks for *broadband dispersed* bursts
(the Astropulse idea), a different signal class. On `.0000` files (16 samples of 18 s) periodic
narrowband pulses can't be searched at all; it is only possible on the high-time-resolution
`.8.0001` product, which `sources.py` currently skips for size (6.4 GB per scan).

**Fix.** Only after B47–B50.
- Per channel group on high-time-res data: fast folding over a period grid (e.g. 0.05–10 s),
  plus a triplet detector (three equal-power peaks equally spaced) and a short-lag
  autocorrelation.
- Stream `.8.0001` per channel window like `.0000` (header-aware), or support a user-supplied
  local file only.
- Add period-zone RFI (SETI@home saw radar, calibration and computer-interrupt periods) via the
  B51 machinery keyed on period instead of frequency.

**Test.** Injected narrowband pulse train (period 0.8 s, duty 10%) in a synthetic high-time-res
window is found within 2% of its period; noise-only false alarms stay at the calibrated rate.

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

- **No coherent de-drifting.** SETI@home de-drifted raw complex voltages *before* its FFTs, at
  about 123,000 drift rates over ±100 Hz/s, which keeps a fast-drifting tone in one bin. Public
  Breakthrough Listen filterbank files carry power only (no phase), so this repo can only do the
  incoherent Taylor tree. B46 measures the cost, B47 recovers part of it. Revisit only if
  baseband (raw voltage) data becomes available.

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

## Gap analysis vs SETI@home (2026-10-03)

Compared against the published description of SETI@home's front end (Korpela et al. 2025, AJ,
doi:10.3847/1538-3881/ade5a7) and back end (Anderson et al. 2025, "SETI@home: Data Analysis and
Findings", arXiv:2506.14737), on `main@6d4f1c3`.

| SETI@home | AI-SETI today | Status |
|---|---|---|
| Raw complex voltages, 2.5 MHz | BL filterbank power spectra | Different by necessity (no phase) |
| Work units with time overlap | Channel windows with drift guard band | Equivalent |
| Coherent de-drift, ~123k rates, ±100 Hz/s | Incoherent Taylor tree, ±4 Hz/s (BL/turboSETI standard) | Modern standard; loss measured in **B46** |
| 15 resolutions, 0.075–1221 Hz | Native resolution only | **B47** |
| Spikes | Spikes + de-Doppler hits | Covered |
| Gaussians (beam passing over source) | ON/OFF cadence (BL tracks targets) | Equivalent role |
| Pulses (folding), triplets, autocorrelation | Broadband dispersed pulses only | **B52** |
| Low-drift RFI cut at 0.086 Hz/s | `stationary` ≈ 0.01 Hz/s | **B48** |
| Learned frequency-zone RFI | Static band list | **B51** |
| "Same signal at different sky positions" RFI rule | `multi_target`, cadence discovery | Covered |
| Multiplets across observations, scored by power/density/time | Per-observation ranking only | **B50** |
| Candidate birdies through the whole back end | Front-end injection + AI training only | **B49** |
| Manual review + re-observation | Share records with checklist | Analogous |
| No AI in the client; NN score weighting tried and failed | Per-hit classifier + isolation forest | Different place; validate with **B49** |

---

## Verified — *not* bugs


**"Non-power-of-two time samples double the Taylor-tree cost" (B10).** True (0.70 s at T = 279
vs 0.31 s at T = 256 on 32k channels), but closed 2026-10-03: no Breakthrough Listen product
reaches that path any more. High-res files have T = 16, and mid-res files skip the tree
because drift is unmeasurable there (and `crunch` now skips them). Revisit only if a product
with a non-power-of-two T *and* measurable drift turns up.

**"Keep-alive connections, fewer threads or prefetching will speed up `crunch`."** Tried and
measured 2026-10-03 (B40), then reverted. A kept-alive connection carries about twice the data
of a fresh one (0.22 vs 0.125 MB/s), but the total levels off at the same **~7–8 MB/s** link or
server ceiling, which today's 8 workers × 8 fresh connections already reach. Four units in a
row in one worker: 62 s old vs 74 s with 4 kept-alive threads. Two full 8-unit runs with 8
kept-alive threads: 20.4 s and 23.8 s, against 15–18 s before. Analysis is ≤ 5% of run time,
so prefetching can't win more than that. Gains have to come from not idling (B38, done), not
downloading twice (B41) and fixed costs (B42, B43).
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