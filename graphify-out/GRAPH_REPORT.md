# Graph Report - 61-SETI_AI  (2026-10-03)

## Corpus Check
- Corpus is ~32,342 words - fits in a single context window. You may not need a graph.

## Summary
- 528 nodes · 1174 edges · 21 communities (15 shown, 6 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 83 edges (avg confidence: 0.92)
- Token cost: 66,609 input · 0 output

## Community Hubs (Navigation)
- AI Hit Classifier
- Architecture & CI Docs
- RFI & Cadence Filtering
- CLI Commands
- De-Doppler Taylor Search
- Findings Sharing
- Candidate Scoring & Config
- Streamlit App & Config
- Legacy Core Waterfall
- Hit Features & Anomaly
- Crunch Dashboard
- Filterbank & Remote IO
- Filterbank Header
- Range Request Tests
- HDF5 & Benchmark Tests
- Window Readers
- Remote Range Fetch
- Share HTTP Capture Stub
- Port Rules
- Package Entry

## God Nodes (most connected - your core abstractions)
1. `FilterbankHeader` - 31 edges
2. `SearchConfig` - 26 edges
3. `read_header()` - 23 edges
4. `drift_search()` - 20 edges
5. `normalize()` - 20 edges
6. `AI-SETI backlog` - 20 edges
7. `share()` - 17 edges
8. `noise_waterfall()` - 16 edges
9. `inject()` - 16 edges
10. `Ledger` - 16 edges

## Surprising Connections (you probably didn't know these)
- `test_config_roundtrips_through_dict()` --uses--> `SearchConfig`  [INFERRED]
  tests/test_io_and_config.py → src/ai_seti/config.py
- `test_channel_range_whole_file_and_subselection()` --uses--> `FilterbankHeader`  [INFERRED]
  tests/test_detectors.py → src/ai_seti/io/filterbank.py
- `test_fine_per_coarse_detects_the_gbt_ratio()` --uses--> `FilterbankHeader`  [INFERRED]
  tests/test_detectors.py → src/ai_seti/io/filterbank.py
- `_write_fil()` --uses--> `Injection`  [INFERRED]
  tests/test_v02.py → src/ai_seti/ai/simulate.py
- `test_benchmark_calibrates_and_reports_a_table()` --calls--> `run()`  [INFERRED]
  tests/test_io_and_config.py → src/ai_seti/benchmark.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Work unit to ranked candidate flow** — architecture_work_unit_pipeline, architecture_taylor_tree_dedoppler, architecture_hitscorer, architecture_score_candidates, architecture_interest_score [EXTRACTED 1.00]
- **Bugs undermining candidate credibility (B6, B7, B8)** — backlog_b6, backlog_b7, backlog_b8, backlog_real_data_run_2026_10_03 [EXTRACTED 1.00]
- **scikit-learn pin enforcement for bundled model** — architecture_sklearn_pin, requirements, _github_workflows_ci_smoke, architecture_hitscorer [INFERRED 0.95]

## Communities (21 total, 6 thin omitted)

### Community 0 - "AI Hit Classifier"
Cohesion: 0.06
Nodes (31): build_training_set(), HitScorer, _sklearn_version(), train(), _training_example(), inject(), Injection, noise_waterfall() (+23 more)

### Community 1 - "Architecture & CI Docs"
Cohesion: 0.07
Nodes (44): CI workflow (ci.yml), CI smoke job (bundled classifier load, demo, share, benchmark), CI verify job (lint, mypy, pytest, 60% coverage floor), AI-SETI Architecture, Data sources (BreakthroughListen, Local, Synthetic, SETI@home probe), HitScorer gradient-boosted hit classifier (ai/model.py), IsolationForest anomaly score, Ledger (state.py): resume, lifetime stats, share dedupe (+36 more)

### Community 2 - "RFI & Cadence Filtering"
Cohesion: 0.06
Nodes (20): cadence_filter(), Ledger, _hits(), _rec(), server(), _shift(), test_cadence_passes_with_on_and_off_and_rejects_off_persistence(), test_cadence_rejects_signal_also_present_in_off_scan() (+12 more)

### Community 3 - "CLI Commands"
Cohesion: 0.07
Nodes (21): analyze(), benchmark(), cadence(), _cfg(), crunch(), crunch_observation(), demo(), inspect() (+13 more)

### Community 4 - "De-Doppler Taylor Search"
Cohesion: 0.07
Nodes (25): Hit, _noise_scale(), _search_one_sign(), _sheared_view(), taylor_tree(), find_pulses(), find_spikes(), _inject_dispersed() (+17 more)

### Community 5 - "Findings Sharing"
Cohesion: 0.12
Nodes (23): build_findings(), _cadence_status(), checklist(), _clean(), finding_id(), _get_json(), markdown(), passes_gate() (+15 more)

### Community 6 - "Candidate Scoring & Config"
Cohesion: 0.11
Nodes (16): SearchConfig, score_candidates(), in_known_rfi_band(), drift_ceil_ch_per_step(), drift_padding(), split(), test_config_load_defaults_and_overrides(), test_config_shipped_default_is_loadable() (+8 more)

### Community 7 - "Streamlit App & Config"
Cohesion: 0.12
Nodes (5): band_profile(), plot_overview(), _versions(), write_html(), write_outputs()

### Community 8 - "Legacy Core Waterfall"
Cohesion: 0.13
Nodes (12): _legacy_max(), AnalysisMetadata, load_filterbank(), make_synthetic_waterfall(), plot_waterfall(), robust_candidates(), save_analysis(), sha256_file() (+4 more)

### Community 9 - "Hit Features & Anomaly"
Cohesion: 0.14
Nodes (12): dedrifted_snippet(), feature_vector(), hit_features(), _resample_rows(), anomaly_scores(), _limit_threads(), _pool_cols(), process_work_unit() (+4 more)

### Community 11 - "Filterbank & Remote IO"
Cohesion: 0.17
Nodes (3): parse_sigproc_header(), _read_string(), download()

### Community 13 - "Range Request Tests"
Cohesion: 0.15
Nodes (7): _EmptyRangeHandler, _RangeHandler, test_drift_search_recovers_positive_and_negative_drifts(), test_noise_only_has_no_hits_at_default_threshold(), test_normalize_removes_bandpass_and_dc_spike(), test_pipeline_finds_injected_signal_once(), _write_fil()

### Community 14 - "HDF5 & Benchmark Tests"
Cohesion: 0.18
Nodes (5): test_benchmark_calibrates_and_reports_a_table(), test_benchmark_detects_strong_drift_better_than_legacy(), test_config_roundtrips_through_dict(), test_h5_header_roundtrip(), _write_h5()

### Community 15 - "Window Readers"
Cohesion: 0.33
Nodes (6): _h5_header(), read_header(), read_window(), test_h5_three_dimensional_with_if_axis(), test_h5_window_read_matches_source(), test_sigproc_roundtrip_and_window()

### Community 16 - "Remote Range Fetch"
Cohesion: 0.25
Nodes (5): _get(), RangeNotSupported, remote_window(), fetch(), test_remote_range_streaming()

## Knowledge Gaps
- **7 isolated node(s):** `ai-seti`, `turboSETI cross-check ([compare] extra)`, `Port Instructions v2.0`, `B12 gpuspec.8.0001.fil corrupt SIGPROC header`, `B13 status hangs / 0 sample rows` (+2 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 191 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `FilterbankHeader` connect `Filterbank Header` to `CLI Commands`, `De-Doppler Taylor Search`, `Candidate Scoring & Config`, `Hit Features & Anomaly`, `Filterbank & Remote IO`, `Range Request Tests`, `Window Readers`, `Remote Range Fetch`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `CrunchDashboard` connect `Crunch Dashboard` to `CLI Commands`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `read_header()` connect `Window Readers` to `AI Hit Classifier`, `CLI Commands`, `Legacy Core Waterfall`, `Filterbank & Remote IO`, `Filterbank Header`, `Range Request Tests`, `HDF5 & Benchmark Tests`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Are the 13 inferred relationships involving `FilterbankHeader` (e.g. with `remote_header()` and `remote_window()`) actually correct?**
  _`FilterbankHeader` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 20 inferred relationships involving `SearchConfig` (e.g. with `_cfg()` and `crunch_observation()`) actually correct?**
  _`SearchConfig` has 20 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ai-seti`, `turboSETI cross-check ([compare] extra)`, `Port Instructions v2.0` to the rest of the system?**
  _7 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `AI Hit Classifier` be split into smaller, more focused modules?**
  _Cohesion score 0.05672926447574335 - nodes in this community are weakly interconnected._