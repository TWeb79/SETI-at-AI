# AI-SETI backlog — pending bugs and changes

Author: Inventions4All — github:TWeb79

Applies to: **v0.3.0**. Opened at the end of the 0.3.0 hardening pass
(see [implementationplan.md](implementationplan.md)).

Fixed items are removed from this file once done (last cleanup 2026-10-03: B4, B5, B6, B8,
B9, B11, B13, B18, B20, B7(b) and B19's labelling). Their write-ups, the two real-data runs of
2026-10-03 and the regression tests that pin them are in git history.

Items are sorted by severity, then by the order they should be fixed in. **IDs are stable and
must not be renumbered** — other documents and the tables below cross-reference them.

Severity: **high** = can produce a wrong or misleading result · **med** = wrong behaviour in
a plausible configuration · **low** = quality, debt or hygiene.

## Triage

| ID | Severity | Summary | Area |
|---|---|---|---|
| B7 | high | Archive cadences are never grouped, so the ON/OFF filter never runs automatically | `cli.py`, `sources.py` |
| B1 | high | `drift_search` silently degenerates to a zero-drift-only search | `dsp/dedoppler.py` |
| B2 | high | `foff_mhz` is unvalidated; Hz passed where MHz is expected blinds the search | `dsp/dedoppler.py` |
| B3 | high | The classifier never sees the drift regime it must score in production | `ai/model.py` |
| B10 | med | Non-power-of-two time samples roughly double the Taylor-tree cost | `dsp/dedoppler.py` |
| B12 | med | Every `.gpuspec.8.0001.fil` fails: the header parser rejects a legal empty string field | `io/filterbank.py` |
| B19 | med | Mid-res products cost ~2 min each although drift is unmeasurable in them: skip or keep? | `cli.py`, `sources.py` |
| B22 | med | The web UI cannot use live data: no URL, no Breakthrough Listen archive | `app.py` |
| B23 | med | The web UI starts on Streamlit's 8501, not 8061 as `RULES_ports.md` assigns | `app.py`, `README.md` |
| B24 | med | The web UI listens on every interface, and its path field reads any file on the host | `app.py` |
| B16 | med | Streamlit drift slider above ~4.9 Hz/s has no effect on high-res GBT data | `app.py` |
| B14 | low | README commands fail when pasted into zsh | `README.md` |
| B15 | low | The terminal dashboard leaves blank lines and torn panels | `dashboard.py` |
| B21 | low | Multi-target matching is too exact for the spur forest around the coarse-channel centre | `rfi.py` |
| B25 | low | Web UI reports show "wall seconds 0.0" and no channels/s | `app.py` |
| B26 | low | The web UI page has no version or build stamp | `app.py` |
| B27 | low | The ledger's lifetime best is never re-scored, so a known artefact stays "best so far" | `state.py`, `cli.py` |
| B28 | low | An `out_of_distribution` carrier can still top the list at the interest cap | `pipeline.py` |
| B29 | low | Partly searched observations take every `--limit` slot, so new files wait | `cli.py`, `sources.py` |
| B17 | low | Shared findings omit `max_drift_ch_per_step`, the drift cap that can bind | `share.py` |

**Suggested order.** B12 (now a one-line fix, it costs every `.8.0001` file), then B7 (cadence
plumbing), then B22 + B23 + B24 together (one pass over `app.py`), then B1 + B2 + B3, then the
rest. B19 and B28 need a decision before any code.

---

# Bugs

## High

### B7 — archive cadences are never grouped, so the ON/OFF filter never runs automatically

Part (b) of this item is done: `rfi.flag_multi_target` flags a frequency already seen in a
different target. What remains is (a). The HIP2579, HIP2586 and HIP3249 files under
`LHS1140/C/` were recorded on the same day (MJD 57774) in consecutive scans: very likely the
OFF pointings of one LHS 1140 cadence. `BreakthroughListenSource` already reads `cadence_url`
(`src/ai_seti/sources.py`), but `crunch` never uses it, and `ai-seti cadence` only accepts
local files. The cadence filter, the strongest RFI test there is, therefore never runs on
archive data.

**Fix.** Group archive files by `cadence_url`; when a group is complete (≥ 2 ON, ≥ 1 OFF), run
`cadence_filter` and feed its `events.csv` into scoring and sharing.

**Test.** A fake source yielding one ON/OFF/ON/OFF group with a tone only in the ONs: `crunch`
writes an event that passes; the same tone in an OFF fails it.

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
image of this — drift `0…1` ch/step always searched regardless of `k_max` — was B9, now fixed.)

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

**Also in this retrain.** The bundled model predates the B20 fix, so its noise class has ~110
examples instead of 500; a retrain with the current code fills it (55 s, accuracy 0.963 vs
0.957, measured 2026-10-03). Add a strong-carrier RFI class to the simulator (high SNR, slow
drift), the open part of the former B8. Strong, slowly drifting carriers are among the most
common terrestrial RFI, and the classifier has never seen one. Until then the SNR guard in
`ai.model.ood_snr_limit` labels such hits `out_of_distribution`.

## Medium

### B19 — mid-res products cost ~2 min each although drift is unmeasurable in them

The labelling half is done: hits from products where drift cannot move a tone one channel are
`drift_unresolved` (see `sources.drift_resolvable`). What remains is a decision. A `.0002`
file (2,861 Hz × 1.07 s × 272 samples) still takes 107–121 s per 1 M channels, and 98% of it
is HTTP Range loading (one 66 k-channel unit: load 48.4 s of 49.2 s). The result can never pass
the share gate, because drift cannot be checked.

**Options.** (1) `crunch` skips products where `drift_resolvable` is false, with a message,
and prefers the `.0000` file of the same scan. (2) Keep crunching them for the zero-drift and
broadband detectors only, but say so in the summary. (1) is the cheaper default; (2) keeps
data that the spike and pulse detectors can still use.

**Test.** For option (1): a mid-res-shaped local file is skipped, with the reason in the summary
and the ledger.

### B10 — non-power-of-two time samples double the Taylor-tree cost

`_search_one_sign` pads time to the next power of two (`src/ai_seti/dsp/dedoppler.py`).
279 samples become 512, nearly half of them zeros. Measured on 32k channels: **0.70 s at
T = 279 vs 0.31 s at T = 256**. On BL mid-res products the tree is now skipped entirely
(de-Doppler 0.01 s per unit; their slowness is Range loading, see B19), so this only matters
for products where a tree is needed and T is not a power of two.

**Fix.** Split time into power-of-two segments and combine, or decimate to the largest power
of two below T.

**Test.** Timing guard: T = 279 costs ≤ 1.2× T = 256 on the same width.

### B12 — every `.gpuspec.8.0001.fil` fails: the header parser rejects a legal empty string field

Reproduced live on 2026-10-03 (HIP2579 `…gpuspec.8.0001.fil`, again in the third run, now
recorded as a ledger failure). The server answers correctly: `206 Partial Content`,
`application/octet-stream`, `bytes 0-383/6861881685`. The bytes are valid SIGPROC. The cause is
the first keyword after `HEADER_START`, `rawdatafile`: its string value has **length 0**
(offset 0x1f: `00 00 00 00`). `parse_sigproc_header` (`src/ai_seti/io/filterbank.py`) accepts
string lengths only in 1–255 (`if not 0 < n < 256`), so it rejects a legal empty string as
"Corrupt SIGPROC header string". The other fields (`source_name` HIP2579, `nbits` 8, `nifs` 1)
parse normally after it.

**Fix.** Accept `n == 0` (an empty string value) and keep the upper bound.

**Test.** A SIGPROC header written with `rawdatafile = ""` parses, and its other fields are
correct.

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

### B22 — the web UI cannot use live data

`app.py` offers only "Synthetic demo" and "Local filterbank (.fil / .h5)". A user of the web UI
cannot stream a remote `.fil` URL or pull work from the Breakthrough Listen archive. The CLI
does both (`ai-seti analyze <url>`, `ai-seti crunch --target …`). The web UI is the obvious
entry point for a non-CLI user, and it can only show simulated data or a file they already
downloaded.

**Fix.** Add two inputs that reuse the CLI paths: "Remote file URL" (`remote.remote_header`
plus `split(..., kind="remote")`), and "Breakthrough Listen archive" (target filter → list from
`BreakthroughListenSource.new_rows` → pick one). Keep a channel-range or `--max-units`-style cap
so a 1-billion-channel file doesn't start a multi-hour job from a button.

**Test.** With `BreakthroughListenSource.query` and the remote reads monkeypatched, the app
logic for the archive option builds remote work units for the chosen file.

### B23 — the web UI starts on Streamlit's 8501, not 8061 as `RULES_ports.md` assigns

`RULES_ports.md` gives project NN the web dashboard port 80NN, so this project (61) gets
**8061**. There is no `.streamlit/config.toml` and no `ports.env`, and `README.md` documents a
bare `streamlit run app.py`. So the app starts on Streamlit's default 8501. The 2026-10-03 run
only used 8061 because the port was passed by hand.

**Fix.** Add `.streamlit/config.toml` with `[server] port = 8061` (and `address`, see B24). Add a
`ports.env` (`PROJECT_ID=61`, `DASHBOARD_PORT=8061`) as the rules suggest, and state the port in
the README.

**Test.** Loading `.streamlit/config.toml` yields port 8061 (a two-line test that guards against
drift).

### B24 — the web UI listens on every interface, and its path field reads any file on the host

`app.py`'s docstring says "Runs locally; nothing is uploaded". But Streamlit's default binds
every interface: the 2026-10-03 start printed a Network URL (`192.168.6.140:8061`) and an
External URL (`217.246.214.110:8061`). Anyone who can reach the port gets the "Local filterbank"
text field. It opens any path on the host with the server user's permissions. Non-filterbank
files fail in the parser, but the server still reads them, and the error text is shown to the
remote user.

**Fix.** Bind to `127.0.0.1` by default (`server.address` in `.streamlit/config.toml`, together
with B23). Optionally restrict the path field to a data directory (e.g. `data/raw`) and show a
generic error instead of the parser message.

**Test.** The shipped config sets `server.address = "127.0.0.1"`; a path outside the allowed
directory is refused.

## Low

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

### B21 — multi-target matching is too exact for the spur forest around the coarse-channel centre

`rfi.flag_multi_target` matches frequencies within 2 kHz across targets. On the 2026-10-03
real data it flagged 2 of 166 hits in HIP2579 `.0000` and 59 of 158 in HIP3249 `.0000`. The
±0.6 MHz forest around the coarse-channel centre appears in all three LHS 1140 targets, but
its tones sit 4–7 kHz apart from target to target, beyond the tolerance. The mirror-pair flag
now demotes most of that forest anyway; this item is about the multi-target check alone being
the wrong granularity.

**Fix.** Add a band-level check (hit density per 100 kHz already seen in other targets), and
make the tolerance a config value.

**Test.** Two targets with dense, non-identical tone forests in the same 100 kHz: hits in the
second are flagged.

### B25 — web UI reports show "wall seconds 0.0" and no channels/s

`app.py` calls `write_outputs(results, cands, out, cfg, title)` without `extra_meta` or
`wall_seconds`. The embedded report therefore shows **wall seconds 0.0** and channels/s "–"
(seen in the 2026-10-03 run). Its `metadata.json` also lacks the observation, the header and
`drift_resolvable`, so the B19 notice never appears in the web UI.

**Fix.** Time the run and pass `wall_seconds`. Pass the same `extra_meta` the CLI builds. Better,
call `cli.crunch_observation` (or a shared helper) instead of repeating its steps.

### B26 — the web UI page has no version or build stamp

`RULES_coding.md` requires a version number and deployment datetime in the UI. The CLI report
footer has one ("AI-SETI 0.3.0, processed … UTC"), but the web page itself shows only the title
and a caption until a run finishes.

**Fix.** Show `ai_seti.__version__` and the build/start time in the sidebar or caption.

### B27 — the ledger's lifetime best is never re-scored, so a known artefact stays "best so far"

`Ledger.best` keeps the highest-interest row ever seen. After B18 the 8000.344291 MHz HIP2579 hit
is a flagged mirror image with interest ≤ 16. But the scratch ledger from the second run still
prints "Best interest so far: 70.7" for it, because the stored copy predates the fix and is
never revisited. The same happens after any scoring change.

**Fix.** Store the scoring version (or a hash of the scoring config) with `best`, and drop or
re-flag the entry when it differs. At least print it as "best under scoring vX".

### B28 — an `out_of_distribution` carrier can still top the list at the interest cap

In the third run, the #1 candidate in HIP2579 channels 2.1–4.2 M was 7990.780103 MHz,
SNR 339.6, `out_of_distribution`, at exactly the **interest cap of 50**, ahead of every
classified hit. The B8 guard caps such hits but doesn't demote them, so a strong terrestrial
carrier can still head the list, and be the "best" of a quiet range. (The same frequency in
HIP3249 was then correctly cut to 10 by the multi-target flag.)

**Decision needed.** Either demote OOD hits (e.g. ×0.3, like zero drift), or keep the cap and
sort them below classified hits of equal interest. Either way, say in the report why they're
there.

### B29 — partly searched observations take every `--limit` slot, so new files wait

Since B5, a high-resolution file is searched `--max-units` units per run. At `--max-units 8`
(2.1 M channels) a 1,073,741,824-channel file needs **512 runs** to finish. `new_rows` returns
unfinished files first, so with a small `--limit` the same few files are revisited and new
targets are never fetched. In the third run 2 of the 4 slots went to resumed files.

**Decision needed.** Spread units across files (round-robin), give new files a share of each
pass, or default `--max-units` larger for archive runs.

### B17 — shared findings omit `max_drift_ch_per_step`, the drift cap that can bind

`share.py:36` `CONFIG_KEYS` records `max_drift_rate_hz_s` but not `max_drift_ch_per_step`.
With the defaults the rate limit binds on high-res data (≈ 26 < 32 ch/step), but a user who
lowers the cap, or searches at > 4.9 Hz/s (see B16), publishes a finding whose recorded config
does not show the limit that was actually applied, so it cannot be reproduced from the record.

**Fix.** Add `"max_drift_ch_per_step"` to `CONFIG_KEYS`.

**Test.** `build_findings()` output contains `max_drift_ch_per_step` under the config block.

---

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