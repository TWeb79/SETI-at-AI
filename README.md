# AI-SETI v0.3 — parallel, AI-assisted technosignature search

In the spirit of SETI@home: your computer downloads small pieces of real radio-telescope
data, searches them for narrowband signals that drift like a transmitter on another
planet would, and keeps a running tally of the best signal it has ever found.

> Candidates are statistical detections. A real technosignature claim needs an ON/OFF
> cadence pass, independent re-observation and a lot of humility.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module responsibilities and data flow,
[implementationplan.md](implementationplan.md) for the change log, and
[backlog.md](backlog.md) for known defects and deferred work.

## Data: where the work comes from

SETI@home stopped distributing work units on 31 March 2020 and has been hibernating since.
`ai-seti crunch` checks the SETI@home server status on every start and tells you if it
ever wakes up. Meanwhile it auto-loads from the **Breakthrough Listen Open Data archive**
(Berkeley SETI Research Center, the same group): `.fil` files are streamed per work unit
with HTTP Range requests (a 4 GB file is never downloaded whole); `.h5` files are cached
with a size guard.

## Install

Requires Python 3.11–3.13.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[radio,dev]"          # add ,dashboard for Streamlit; ,gpu for CuPy (experimental)
```
blimpy/turboSETI are no longer required (optional `[compare]` extra for cross-checks).

> `pip install -e ".[compare]"` only. If you installed a scikit-learn outside the
> `>=1.8,<1.9` window in [pyproject.toml](pyproject.toml), the bundled
> `hit_classifier.joblib` will not unpickle. AI-SETI tells you when this happens
> (`model_unavailable` in the candidate table, plus a banner in `report.html`) instead of
> quietly reporting every hit as `unscored`. Rebuild the model with `ai-seti train`.

## Development and quality gates

```bash
pytest                                   # 72 tests
pytest --cov=ai_seti --cov-report=term-missing
ruff check .                             # lint
mypy                                     # typecheck (configured in pyproject.toml)
```

CI (`.github/workflows/ci.yml`) runs lint, typecheck and tests on Python 3.11, 3.12 and
3.13, enforces a 60% coverage floor, and adds a smoke job that asserts the bundled
classifier loads, `demo` produces a report with real AI classes, and `share` writes a
bundle.

## How to use this

Three steps: install it, point it at some data, read the report. You can stop after any of
them, and nothing is sent anywhere unless you explicitly ask for it.

### 1. Try it on data it makes up

```bash
ai-seti demo
```

This builds a fake radio observation — a GBT-like coarse channel with **3 signals hidden in
it** plus 60 pieces of interference — searches it, and writes a report. The three hidden
signals should come back ranked #1, #2 and #3, all labelled `technosignature_like`. Nothing
is downloaded and nothing leaves the machine.

### 2. Point it at real data

Pick whichever fits:

```bash
ai-seti analyze data/raw/obs.fil                        # a file already on disk
ai-seti analyze https://example.org/obs.0000.fil        # a URL, streamed not downloaded
ai-seti crunch --target HIP --max-units 8               # fetch new BL observations, keep going
```

`analyze` searches one file, once. `crunch` is the SETI@home-style loop: it asks the
Breakthrough Listen archive for work, processes everything it has not seen, and remembers
what it already did in `data/state.json`. `--max-units N` caps each observation at N units
per run; the next run continues at the next channel, and an observation only counts as done
once every channel has been searched.

Before hunting for aliens, it is worth checking the file you have is sane:

```bash
ai-seti inspect data/raw/obs.fil    # header only: size, sample rate, channel width, source
```

### 3. Read the report

Each run writes a folder — `reports/analysis` or `reports/crunch` by default:

| File | What it is |
|---|---|
| `report.html` | **Start here.** Animated 3-D power plot plus the ranked candidate table. |
| `candidates.csv` | Every hit, one row per signal, ranked by interest score. |
| `spikes.csv`, `pulses.csv` | Single-sample spikes and dispersed pulses. |
| `band_overview.png` | Where in the band the power sits. |
| `metadata.json` | Config, package versions and timings for this run. |

The single number that ranks candidates is **interest**, 0–100. It blends how ET-like the
signal looks, its SNR, and how unusual it is against every other hit in the same run.

## The AI part: running with or without it

The AI is one optional layer on top of the search. A classifier looks at each hit and
labels it — `technosignature_like`, `rfi_zero_drift`, `rfi_intermittent`, and so on — and
feeds a 0–1 "this looks like a technosignature" probability into the interest score.

### With AI (the default)

The bundled `hit_classifier.joblib` scores every hit. You get an `ai_class` label per row
and per-class probabilities as `p_*` columns in `candidates.csv`. The demo's #1–#3 ranking
depends on this.

### Without AI

Set `"use_ai": false` in a config file and pass it with `--config`:

```bash
echo '{"use_ai": false}' > no-ai.json
ai-seti analyze data/raw/obs.fil --config no-ai.json
```

The search itself is untouched — same detections, same SNRs, same spikes and pulses. Only
the labelling and the ranking change:

| | AI on | AI off |
|---|---|---|
| `ai_class` | `technosignature_like`, `rfi_*`, … | `ai_disabled` on every row |
| `p_*` probability columns | present | absent |
| interest score | uses the classifier probability, a soft 0–1 value | uses a fixed rule: narrow, drifting and steady scores 0.7, everything else 0.0 |

On one demo file, the top candidate scored **81.5 with AI and 70.3 without** — and the
ranking itself changes, because the fallback rule can only award 0.7 or 0. It cannot put a
faint signal above a strong one the way a real probability can.

You might turn it off to check whether the classifier is flattering your results, to see
what the raw detector finds on its own, or to save a little time on a slow machine.

### If the AI is on but the model will not load

You get `ai_class = model_unavailable` and a warning banner in `report.html`, never a
silent `unscored`. That distinction is the point: a run that fell back to the heuristic
should not look like a clean one. The usual cause is a scikit-learn version outside the
`>=1.8,<1.9` window the bundled model was pickled with — see
[Install](#install) and `ai-seti train`.

## Quick start

```bash
ai-seti demo                       # synthetic GBT coarse channel, 3 hidden ET-like tones + 60 RFI
ai-seti status                     # is SETI@home sending work? is the BL archive reachable?
ai-seti crunch --target HIP --max-units 8          # auto-load real BL data, live dashboard
ai-seti crunch --forever --poll-seconds 900         # BOINC-style: keep crunching new data
ai-seti crunch --source local --watch-dir data/raw  # your own files
ai-seti analyze data/raw/obs.fil --f-start-mhz 1420 --f-stop-mhz 1421
ai-seti analyze http://…/obs.0000.fil               # remote, streamed
ai-seti cadence A.fil B_OFF.fil A2.fil C_OFF.fil A3.fil D_OFF.fil
ai-seti benchmark                  # v0.1 vs v0.2 injection/recovery
ai-seti train                      # retrain the hit classifier (~10 s, CPU)
streamlit run app.py
```

Each run writes `candidates.csv`, `spikes.csv`, `pulses.csv`, `metadata.json`
(config, versions, timings), `band_overview.png` and `report.html` — an animated 3D
power plot of the best signal in the style of the old SETI@home screensaver, stamped with
the software version and build time. `data/state.json` keeps lifetime stats and lets the
client resume.

## Sharing findings back

SETI@home clients reported every result to Berkeley; that server no longer accepts
results. AI-SETI packages each finding as a self-describing `ai-seti-finding/1` record
(signal, AI assessment, observation, software versions, de-drifted snippet, verification
checklist, SHA-256) and sends it where you choose:

```bash
ai-seti share reports/crunch/<obs>                       # preview + local bundle (zip, PNG, Markdown)
ai-seti share reports/crunch/<obs> --to github --repo your-org/ai-seti-findings --yes
ai-seti share reports/crunch/<obs> --to webhook --webhook https://discord.com/api/webhooks/… \
        --webhook-format discord --handle "your-name" --yes
ai-seti share reports/cadence/A --cadence-events reports/cadence/events.csv --require-cadence --yes
ai-seti crunch --auto-share        # share after every observation, using share_* in the config
```

- **GitHub:** one issue per finding, labelled `candidate`/`unverified`; token from
  `AI_SETI_GITHUB_TOKEN` (fine-grained, Issues: write). Searches the repo first, so a
  signal someone already reported is linked, not duplicated.
- **Webhook:** raw JSON, or Slack / Discord message format, for a team or community server.
- **Same signal, same ID:** finding IDs come from the data file + frequency (10 Hz bins) +
  drift (0.05 Hz/s bins), so independent volunteers' reports of one signal collapse together.

Guard rails: remote destinations need `--yes` (or `--auto-share`), and only receive findings
that passed an ON/OFF cadence (`--cadence-events`) unless you pass `--no-require-cadence`;
zero-drift signals, signals in known RFI bands and coarse-channel mirror images are never sent;
hits far above the classifier's training SNR are labelled `out_of_distribution` and capped at
interest 50; synthetic/demo data stays local unless
`--allow-synthetic`; local paths are stripped and your name is only included with
`--handle`; the ledger never sends the same finding to the same place twice. Every record
is marked `unverified_candidate`. If something ever survives cadence, an independent
pipeline and re-observation, that is the point to contact the Breakthrough Listen team
directly rather than post publicly.

## Review of v0.1 → what changed

| v0.1 problem | v0.2 fix |
|---|---|
| Time-median spectrum: any tone drifting > ~1 channel over the observation is erased. Real ET signals always drift (planet rotation/orbit). | Taylor-tree de-Doppler search over ±`max_drift_rate_hz_s`, both signs; shear trick for drifts > 1 ch/step; zero-copy strided views. |
| Global median/MAD: bandpass ripple inflates the noise estimate; DC spike of every GBT coarse channel flagged. | Robust piecewise bandpass + per-block MAD noise; DC bins repaired (coarse width auto-detected from `foff`). |
| Whole selection loaded via blimpy, single core. | Memory-mapped SIGPROC / chunked HDF5 reader; overlapping work units; process pool (1 BLAS thread per worker); HTTP Range streaming. |
| No RFI handling, no AI. | Known-band flags, zero-drift penalty, coarse-channel mirror-image flag, ON/OFF cadence filter; gradient-boosted hit classifier + isolation-forest anomaly score → 0–100 interest score. |
| Only one signal type. | + SETI@home-style spikes, + Astropulse-style dispersed pulses (for high-time-resolution products). |
| Test fixture crashed for < 701 channels. | Fixed; 72 tests incl. a local Range-server streaming test, mocked GitHub/webhook sharing, and a drift search at real GBT coarse-channel resolution. |

## Review of v0.3 — what changed

The 0.3.0 pass is a correctness and honesty pass; no new signal processing.

| Problem found in review | Fix |
|---|---|
| The bundled classifier was pickled with scikit-learn 1.8 while `pyproject.toml` allowed `>=1.4`, so every fresh install silently ran with no AI at all: all hits came out `unscored` and interest fell back to a heuristic. | Pinned `scikit-learn>=1.8,<1.9`, recorded the training versions in the model metadata, and made load failures **loud** — logged, exposed as `load_error`, surfaced as `model_unavailable` in the candidate table, `stats.ai_note` in `metadata.json` and a banner in `report.html`. |
| `max_drift_ch_per_step` was dead config: never passed to `drift_search`, and its one other reader computed `min(max(k,1), max(cap, k))`, which is always `k`. The knob did nothing while costing 27 Taylor-tree passes per sign. | The cap is now enforced, shared with the guard band in `drift_padding` so the two cannot disagree, and the default raised to 32 — at a GBT coarse channel 4 Hz/s is already ~26 channels/step, so the old value of 4 silently blinded the search to signals the rate limit allows. |
| `cadence_filter` returned `passed_cadence` for a single ON scan with no OFF scan (the reference hit matched itself, giving 1.0/1.0), so `--require-cadence` was free to satisfy. | Requires ≥2 ON and ≥1 OFF scan; `ai-seti cadence` states why a test is impossible instead of writing an empty file silently, and `share` re-checks the scan counts before granting a pass. |
| Share dedupe tested record *equality* over dicts carrying a 16×16 snippet — O(n²) and semantically wrong. | Keyed on `finding_id`. |
| Ledger wrote a fixed `.tmp` sibling (concurrent crunchers raced) and crashed on unknown keys. | Unique per-PID temp file, `os.replace`, cleanup on failure, unknown keys dropped with a warning. |
| No CI; ruff and mypy ran on defaults with no config and failed (76 and 21 findings). | Declared `[tool.ruff.lint]` and `[tool.mypy]`; both are clean. Added CI across 3.11/3.12/3.13 plus a smoke job that fails if the bundled classifier stops loading. |
| Report carried no version or build stamp. | `report.html` shows `v<version> · built <UTC>` and an AI-layer banner. |
| First run against real Breakthrough Listen data (2026-10-03) exposed two more: a `206` with an empty body made `remote_header` re-request the same range forever (**17,681 requests in 3 s**, never terminating — an unthrottled flood of a public research server), and the Taylor tree reported drifts far past the configured ceiling (−276 to −1,225 Hz/s under a 4 Hz/s limit) because its `for d in range(tp)` sweep always reaches 1 ch/step. | `remote_header` now stops on lack of progress, bounded by `max_requests`. `drift_search` skips tree rows beyond `max_drift_rate_hz_s`, so a reported rate is always inside the configured range. |

## Results (measured on the development machine, 1 CPU core)

**Injection/recovery at equal false-alarm rate** (both detectors calibrated so ≤5% of
noise-only 16×16384 chunks raise any detection; 20 trials per cell; `ai-seti benchmark`):

| SNR per sample | drift 0 | drift 0.5 ch/step | drift 2 | drift 8 |
|---|---|---|---|---|
| 1.5 | 35% → **60%** | 0% → **80%** | 0% → **45%** | 10% → **40%** |
| 2.0 | 55% → **80%** | 0% → **90%** | 0% → **100%** | 0% → **85%** |
| 3.0 | 80% → **100%** | 0% → **100%** | 0% → **100%** | 0% → **100%** |

(v0.1 → v0.2). v0.2 costs ~5× more CPU per chunk (it searches ~20× more paths).

**Demo:** 1,048,576 channels in **2.4 s wall / 2.3 s CPU on one core** (1.8 s wall on 4
workers); the three hidden ET-like tones are recovered and rank **#1, #2 and #3**, all
classified `technosignature_like`, ahead of the strongest RFI (interest 74–82 vs 28–32).
Classifier: **95.7%** held-out accuracy on 6 classes (simulated; 2500 examples).

## How a work unit is processed

```
load window (memmap | HDF5 chunks | HTTP Range)
  → DC-spike repair → robust bandpass flattening → N(0,1) normalisation
  → Taylor-tree de-Doppler (+ spikes, + dedispersed pulses)
  → non-max suppression → keep hits whose start lies in this unit's core
  → features (bandwidth, on-fraction, wobble, modulation, …) + 16×16 de-drifted snippet
  → classifier probabilities
run level: isolation-forest anomaly → RFI flags → interest score → report
```

## Honest limitations

- The classifier is trained on simulated signals and RFI. Real RFI is richer; retrain or
  fine-tune on labelled real hits before trusting class labels. The anomaly score and the
  cadence filter matter more than the class label.
- The cadence filter **requires at least 2 ON scans and 1 OFF scan** and reports nothing
  otherwise: a single observation cannot confirm or reject anything, and
  `ai-seti cadence` says so explicitly rather than returning an empty file silently.
- `max_drift_ch_per_step` (default 32) bounds the search. At a GBT coarse channel's
  ~18 s / 2.79 Hz resolution, 4 Hz/s is already ~26 channels per step, so lowering this
  below that number silently throws away signals the rate limit allows.
- **`drift_search` will not tell you when it could not search the range you asked for.** If
  the searched drift clamps to zero — a `foff` passed in Hz instead of MHz, a cap below the
  rate, or a resolution too coarse for the requested drift — it reports *no hits* instead of
  flagging a degraded run. Absence of detections is not evidence of absence until you have
  checked the range was searchable. Tracked as [backlog B1/B2](backlog.md).
- **The classifier is trained on drift of 0.05–6 channels/step but production needs up to
  ~26.** Its class labels are least reliable in the fast-drift regime the search spends most
  of its time in, and the published 95.7% accuracy is measured on the training distribution,
  not on production drift. Tracked as [backlog B3](backlog.md).
- BL archive API and Range streaming against the real servers were not tested from the
  development sandbox (network-restricted); Range streaming is tested locally.
- GPU (CuPy) path is experimental and untested, and excluded from CI. Multi-core speedup
  was not benchmarked on the development machine, though the multiprocess path is
  exercised by tests and the smoke job.
- Cross-check hits against turboSETI (`pip install -e ".[compare]"`) before publishing.

## Data policy

Keep raw observations immutable; every report records config, package versions and
timings. Respect the archive's terms and cite the observation in any publication.
