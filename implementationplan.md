# Implementation plan — AI-SETI

Author: Inventions4All — github:TWeb79

Pending work only. Completed tasks are removed once done; the last cleanup (2026-10-03)
removed tasks 1–35 from the 0.3.0 hardening pass and the backlog fixes B4, B5, B6, B8, B9,
B11, B13, B18, B20 and B7(b). Their test plans and verification notes are in git history. Each
task below links to its full write-up in [backlog.md](backlog.md).

## Task list

In the backlog's suggested order.

| # | Task | Backlog | Module | Status |
|---|---|---|---|---|
| 1 | Accept an empty string value in the SIGPROC header parser | B12 | `io/filterbank.py` | pending |
| 2 | Group archive files by `cadence_url` and run the ON/OFF filter automatically | B7 | `cli.py`, `sources.py` | pending |
| 3 | Web UI: remote URL and archive inputs | B22 | `app.py` | pending |
| 4 | Web UI: port 8061 via `.streamlit/config.toml` and `ports.env` | B23 | `app.py`, `README.md` | pending |
| 5 | Web UI: bind to 127.0.0.1, restrict the path field | B24 | `app.py` | pending |
| 6 | Report "could not search this drift range" instead of silently searching zero drift | B1 | `dsp/dedoppler.py` | pending |
| 7 | Validate `foff_mhz` units | B2 | `dsp/dedoppler.py` | pending |
| 8 | Retrain on the production drift range, plus a strong-carrier RFI class (also refills the bundled model's noise class) | B3 | `ai/model.py`, `ai/simulate.py` | pending |
| 9 | Decide whether `crunch` skips drift-unresolvable (mid-res) products | B19 | `cli.py`, `sources.py` | needs decision |
| 10 | Decide how `out_of_distribution` hits rank against the cap | B28 | `pipeline.py` | needs decision |
| 11 | Decide how resumed files share `--limit` with new ones | B29 | `cli.py`, `sources.py` | needs decision |
| 12 | Taylor tree without padding T to a power of two | B10 | `dsp/dedoppler.py` | pending |
| 13 | Streamlit drift slider shows the effective limit | B16 | `app.py` | pending |
| 14 | Web UI report: pass wall time and metadata | B25 | `app.py` | pending |
| 15 | Web UI: version and build stamp | B26 | `app.py` | pending |
| 16 | Re-score or label a stale lifetime best | B27 | `state.py`, `cli.py` | pending |
| 17 | README commands safe to paste into zsh | B14 | `README.md` | pending |
| 18 | Terminal dashboard without blank lines or torn panels | B15 | `dashboard.py` | pending |
| 19 | Band-level multi-target check | B21 | `rfi.py` | pending |
| 20 | Record `max_drift_ch_per_step` in shared findings | B17 | `share.py` | pending |

Also open, outside the backlog:

- `README.md` omits `reports/demo` from the list of output folders.
- The warnings in `ai/model.py` still say hits will be "unscored"; the label is
  `model_unavailable`.

## Test debt

`cli.py` and `dashboard.py` have the least unit coverage (see backlog "Test debt"). Don't raise
the 60% CI floor until one of them has real tests.
