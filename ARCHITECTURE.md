# AI-SETI architecture

Parallel, AI-assisted radio technosignature search in the spirit of SETI@home.
This document describes how a work unit becomes a ranked candidate, where each
responsibility lives, and which boundaries the design deliberately enforces.

Known defects and deferred work are tracked in [backlog.md](backlog.md).

## System shape

```
              ┌──────────────────────── one observation ────────────────────────┐
              │                                                                │
  source ──► split ──► work unit ──► [worker process] ──► result dict ──┐        │
              │          │            load → normalize →                    │        │
              │          │            de-Doppler → spikes →                │        │
              │          │            pulses → features → AI              │        │
              │          │                                     (1 BLAS     │        │
              │          │                                      thread ea.) │        │
              └──────────┴──────────────────────────────────────────────┼────────┘
                                                                        ▼
                                              score_candidates (main process)
                                       anomaly → RFI bands → interest score
                                                                        │
                                                                        ▼
                                     write_outputs → candidates.csv, metadata.json,
                                     band_overview.png, report.html, snippets.npy
                                                                        │
                                                          share → bundle / github / webhook
```

Data sources:

| Source | Kind | Transport | Guard |
|---|---|---|---|
| `BreakthroughListenSource` | `.fil` | HTTP Range, one request per time step | falls back to download if the server ignores `Range` |
| `BreakthroughListenSource` | `.h5` | streamed download into `data/raw/bl` | `--max-download-mb` size guard |
| `LocalSource` | `.fil` / `.h5` | `np.memmap` (SIGPROC) or `h5py` slicing | never reads the whole selection |
| `SyntheticSource` | `.fil` | generated, written as a real SIGPROC file | tagged `source="synthetic"`; blocked from remote sharing |
| `SetiAtHomeSource` | — | status probe only | no work is distributed; it reports if that changes |

## Module responsibilities

| Module | Responsibility | Must not |
|---|---|---|
| `config.py` | `SearchConfig` dataclass; JSON load/dump | contain logic |
| `io/filterbank.py` | SIGPROC header parse/serialise, memmap window read, HDF5 read | touch business logic |
| `io/remote.py` | HTTP Range GET with retry/backoff, remote header + window, size-guarded download | interpret signal data |
| `sources.py` | Data sources; `split()` into overlapping work units; drift guard band | run detection |
| `dsp/preprocess.py` | DC-spike repair, robust piecewise bandpass, per-block MAD normalisation | decide what a hit is |
| `dsp/dedoppler.py` | Taylor-tree de-Doppler search, SNR map, non-max suppression | know about RFI or scoring |
| `dsp/detectors.py` | Single-sample spikes; dedispersed pulses (Astropulse-style) | drift-search |
| `ai/simulate.py` | Synthetic noise + labelled injections | import the pipeline (would be circular) |
| `ai/features.py` | De-drifted 16×16 snippet + physical scalar features | know the model |
| `ai/model.py` | Hit classifier (`HitScorer`) and per-run IsolationForest anomaly score | persist state |
| `rfi.py` | Known-band flags; ON/OFF cadence filter | score interest |
| `pipeline.py` | `process_work_unit` (worker side); `run_units` (pool); `score_candidates` (run side) | write files |
| `report.py` | CSV/JSON/PNG/HTML output; version + build stamp | re-run detection |
| `share.py` | `ai-seti-finding/1` records, gate, dedupe ledger, bundle/github/webhook sinks | bypass the gate |
| `state.py` | `Ledger`: resume, lifetime stats, best-ever signal, share dedupe | store observations |
| `dashboard.py` / `app.py` | Terminal live view; Streamlit front end | alter results |

## Data flow, per work unit

1. **Split.** `split()` cuts the observation into cores of `channels_per_unit` and widens
   each into a padded window. The guard band is `drift_ceil_ch_per_step × nsamples + 8`
   channels, computed by the same rule `drift_search` uses to cap drift — if those two
   disagree, a signal near a core edge is searched for but has no data under it.
2. **Load.** Only the padded window is touched: `np.memmap` for SIGPROC, chunked `h5py`
   slicing for HDF5, concurrent Range requests for remote `.fil`.
3. **Normalise.** DC bins at the centre of each GBT coarse channel are replaced by the mean
   of their neighbours; a robust piecewise bandpass (block medians, interpolated) is
   divided out; per-block MAD gives unit robust noise. Result is ~N(0,1) per sample.
4. **De-Doppler.** `taylor_tree()` sums along every drift 0…T−1 in O(T log T · F); a
   shear pass covers k·(T−1)+d for integer k; the reversed band covers negative drift.
   SNR comes from a median/MAD estimate of a zero-drift path sum. Non-maximum suppression
   over the channels a track can touch keeps one signal to one hit.
5. **Secondary detectors.** Spikes (single sample ≥ 9σ); pulses (sub-band dedispersion plus
   boxcar matched filters, only for products with ≥ 32 time samples).
6. **Core filter.** A hit is kept only if its start channel lies in the unit's *core*, so
   overlapping windows never report the same signal twice.
7. **Features + AI.** 11 scalar features plus a 16×16 de-drifted snippet feed the bundled
   gradient-boosted classifier, giving per-class probabilities.
8. **Run level.** IsolationForest anomaly score over this run's hit population, known-RFI-band
   flag, and the interest score:
   `score = (0.55·p_ET + 0.20·f(SNR) + 0.25·anomaly) × zero-drift × RFI-band × bandwidth`
   scaled to 0–100.

## Service boundaries

AI-SETI is a local CLI. There is no server, no database and no always-on daemon.

- **CLI ↔ core.** `ai_seti.cli` owns argument parsing, terminal rendering and ledger
  bookkeeping; it delegates to `pipeline`, `report` and `share`. All commands are pure
  functions over files on disk, which is what makes the resume-after-restart model work.
- **Worker ↔ main process.** Workers return plain JSON-serialisable dicts. Nothing is
  shared through memory; `ProcessPoolExecutor` with one BLAS thread per worker keeps
  oversubscription from degrading throughput.
- **Core ↔ AI.** `ai/features.py` and `dsp/*` do not import scikit-learn. A model failure
  degrades to the documented heuristic in `pipeline.score_candidates` and is reported in
  `metadata.json` as `stats.ai_note` and in `candidates.ai_class` as `model_unavailable` —
  it is never allowed to look like a clean run.
- **Core ↔ network.** Only `io/remote.py` and `sources.py` open sockets. Tests exercise the
  Range path against a local HTTP server, so no test needs the internet.

## Failure handling

| Stage | Behaviour |
|---|---|
| One work unit raises | captured in `run_units`, returned as `{"error": ...}` with empty hits; the run continues and `stats.errors` counts it |
| Network hiccup | `io/remote._get` retries 3× with exponential backoff, then raises `ConnectionError` |
| Server ignores `Range` | `RangeNotSupported` → the source falls back to a size-guarded download |
| Classifier will not load | logged, `model_unavailable`, heuristic fallback, `ai_note` in metadata |
| Ledger write | unique per-PID temp file then `os.replace`, so a partial file is never published |
| Searched drift range collapses to zero | **not handled.** `drift_search` clamps `k_max` silently, so an unsearchable drift rate yields "no hits" rather than a degraded run. See [backlog.md](backlog.md) B1/B2. |
| Reported drift exceeds the ceiling | `drift_search` skips tree rows past `max_drift_rate_hz_s`, so a reported rate is always inside the configured range. |
| Server stops sending a Range body | `remote_header` stops on lack of progress, bounded by `max_requests`, and raises `ValueError` naming the URL |

## External dependencies

`numpy`, `scipy` (non-max filter), `pandas` (tables), `scikit-learn` (classifier, IsolationForest),
`joblib` (model artifact), `typer` + `rich` (CLI), `matplotlib` (plots).
Optional: `h5py` + `hdf5plugin` (`.h5`), `streamlit` (dashboard), `cupy` (experimental GPU),
`turbo_seti`/`blimpy` (independent cross-check).

## Known deliberate deviations

- **Python 3.11–3.13 only**, per `requires-python`. No support for 3.14 yet.
- **The bundled model pins scikit-learn to `<1.9`.** A pickled estimator is not portable
  across minor versions; the upper bound and `tests/test_model_and_limits.py` keep this
  honest.
- **GPU (CuPy) path is experimental and untested**, and is excluded from CI. The array-API
  style in `dsp/dedoppler.py` exists so it can work, not because it is verified.
