# Implementation plan — AI-SETI

Author: Inventions4All — github:TWeb79

Pending work only. Completed tasks are removed once done; the last cleanup (2026-10-03)
removed tasks 1–35 from the 0.3.0 hardening pass and the backlog fixes B39, B44, B43, B42, B40, B38, B7, B37, B36, B29, B19, B28, B10, B31, B33, B34, B35, B32, B30, B21, B27, B15, B1, B2, B22, B23, B24, B25, B26, B16, B14, B17, B12, B4, B5, B6, B8, B9,
B11, B13, B18, B20 and B7(b). Their test plans and verification notes are in git history. Each
task below links to its full write-up in [backlog.md](backlog.md).

## Task list

In the backlog's suggested order. Tasks 1–6 are the documentation pass, high priority.

| # | Task | Backlog | Module | Status |
|---|---|---|---|---|
| 1 | Reuse cadence-searched ranges when an OFF scan is crunched on its own | B41 | `cli.py`, `state.py` | pending |
| 2 | Retrain on the production drift range, plus a strong-carrier RFI class (also refills the bundled model's noise class) | B3 | `ai/model.py`, `ai/simulate.py` | pending |

## Test debt

`cli.py` and `dashboard.py` have the least unit coverage (see backlog "Test debt"). Don't raise
the 60% CI floor until one of them has real tests.
