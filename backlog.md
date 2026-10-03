# AI-SETI backlog — pending bugs and changes

Author: Inventions4All — github:TWeb79

Applies to: **v0.3.0**. Opened at the end of the 0.3.0 hardening pass
(see [implementationplan.md](implementationplan.md)).

Fixed items are removed from this file once done. As of 2026-10-03 everything from B1 to B44 is
fixed, or closed with a measured reason (B10, B40). The write-ups, the real-data runs of
2026-10-03 and the regression tests that pin them are in git history; the performance
measurements are kept below because they still describe how `crunch` spends its time.

Items are sorted by severity, then by the order they should be fixed in. **IDs are stable and
must not be renumbered** — other documents and the tables below cross-reference them.

Severity: **high** = can produce a wrong or misleading result · **med** = wrong behaviour in
a plausible configuration · **low** = quality, debt or hygiene.

## Triage

| ID | Severity | Summary | Area |
|---|---|---|---|

**Nothing is pending.** Every item found so far is fixed or closed; add new ones above.

---

# Bugs

## High

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

## Low

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