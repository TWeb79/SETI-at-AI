"""Streamlit front end: `streamlit run app.py` (port 8061 from .streamlit/config.toml).

Binds to 127.0.0.1 by default, so only this machine can reach it. Searches synthetic data,
a local file, a remote .fil URL or the Breakthrough Listen archive; results stay local.
"""
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ai_seti import __version__
from ai_seti.cli import explain_run
from ai_seti.config import SearchConfig
from ai_seti.io import remote
from ai_seti.io.filterbank import read_header
from ai_seti.pipeline import run_units, score_candidates
from ai_seti.report import write_outputs
from ai_seti.sources import (
    BreakthroughListenSource,
    SyntheticSource,
    drift_ceil_ch_per_step,
    drift_resolvable,
    split,
    unit_too_large,
)

if __name__ == "__main__" and not st.runtime.exists():
    sys.exit("This is a Streamlit app. Start it with:  streamlit run app.py")

MODES = ["Synthetic demo", "Breakthrough Listen archive", "Remote file URL",
         "Local filterbank (.fil / .h5)"]


@st.cache_resource
def _started() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


@st.cache_data(ttl=600, show_spinner="Asking the Breakthrough Listen archive…")
def _archive_files(target: str, limit: int) -> list[dict]:
    return BreakthroughListenSource(target=target, limit=limit).query(limit, timeout=20, retries=1)


@st.cache_data(show_spinner="Reading the file header…")
def _remote_header(url: str):
    return remote.remote_header(url)


@st.cache_resource
def _synthetic():
    return next(SyntheticSource(Path(tempfile.mkdtemp()), n_chan=1 << 19).observations(set()))


st.set_page_config(page_title="AI-SETI", layout="wide")
st.title("AI-SETI signal search")
st.caption(f"AI-SETI {__version__} · started {_started()} · Candidates are unverified "
           "statistical detections, not technosignature claims.")

mode = st.sidebar.radio("Input", MODES)
cfg = SearchConfig()
cfg.snr_threshold = st.sidebar.slider("De-Doppler SNR threshold", 6.0, 25.0, 10.0, 0.5)
cfg.max_drift_rate_hz_s = st.sidebar.slider("Max drift (Hz/s)", 0.5, 10.0, 4.0, 0.5)
cfg.workers = st.sidebar.number_input("Workers (0 = all cores)", 0, 256, 0)

try:
    if mode == "Synthetic demo":
        location, kind, header, meta = _synthetic()
    elif mode == "Local filterbank (.fil / .h5)":
        path = st.sidebar.text_input("Path to file")
        if not path or not Path(path).expanduser().is_file():
            st.info("Enter the path to a local Breakthrough Listen filterbank file.")
            st.stop()
        location, kind = str(Path(path).expanduser()), "local"
        header = read_header(Path(location))
        meta = {"target": header.source_name, "source": "local"}
    else:
        if mode == "Remote file URL":
            location = st.sidebar.text_input("URL of a .fil file")
            meta = {"source": "remote", "url": location}
        else:
            target = st.sidebar.text_input("Target filter", "HIP")
            rows = [r for r in _archive_files(target, 50) if r["url"].endswith(".fil")]
            if not rows:
                st.info("No filterbank files for this target. Try another filter.")
                st.stop()
            row = st.sidebar.selectbox("File", rows, format_func=lambda r: Path(r["url"]).name)
            location = row["url"]
            meta = {"target": row.get("target"), "telescope": row.get("telescope"),
                    "mjd": row.get("mjd"), "source": "breakthrough_listen", "url": location}
        if not location.startswith("http"):
            st.info("Enter or choose a remote .fil file. Only the chosen channels are streamed.")
            st.stop()
        kind, header = "remote", _remote_header(location)
        meta.setdefault("target", header.source_name)
except Exception as exc:   # bad path, unreachable server, not a filterbank file
    st.error(f"Could not open this input: {exc}")
    st.stop()

if reason := unit_too_large(header, cfg):
    st.error(f"Not searched: {reason}")
    st.stop()

# Units cap the work: a high-resolution BL file is a billion channels.
n_max = -(-header.nchans // cfg.channels_per_unit)
n_units = st.sidebar.number_input(f"Work units to search (file has {n_max:,})", 1, n_max,
                                  min(8, n_max))
first = st.sidebar.number_input("Start at unit", 0, n_max - 1, 0)
chan_range = (first * cfg.channels_per_unit,
              min(header.nchans, (first + n_units) * cfg.channels_per_unit))

foff_hz = abs(header.foff) * 1e6
effective = drift_ceil_ch_per_step(header, cfg) * foff_hz / header.tsamp if header.tsamp else 0
st.sidebar.caption(f"{meta.get('target')}: {header.f_min:.3f}–{header.f_max:.3f} MHz, "
                   f"{header.nchans:,} channels of {foff_hz:,.2f} Hz. Drift actually searched: "
                   f"±{min(cfg.max_drift_rate_hz_s, effective):.2f} Hz/s.")
if not drift_resolvable(header, cfg):
    st.sidebar.warning("Drift is not measurable in this file's wide channels; hits will be "
                       "labelled drift_unresolved.")

if st.sidebar.button("Start crunching", type="primary"):
    units = split(location, header, cfg, kind, chan_range=chan_range, meta=meta)
    bar = st.progress(0.0, text="Crunching work units")
    results, t0 = [], time.time()
    for i, res in enumerate(run_units(units, cfg), 1):
        results.append(res)
        bar.progress(i / len(units), text=f"{i}/{len(units)} work units")
    cands = score_candidates(results, cfg)
    out = Path(tempfile.mkdtemp(prefix="ai_seti_"))
    stats = write_outputs(results, cands, out, cfg, f"AI-SETI search of {meta['target']}",
                  {"location": str(location), "observation": meta, "header": header.to_dict(),
                   "drift_resolvable": drift_resolvable(header, cfg)}, time.time() - t0)
    st.text("\n".join(explain_run(cands, cfg.snr_threshold, chan_range, header.nchans, stats)))
    components.html((out / "report.html").read_text(), height=1500, scrolling=True)
    st.download_button("Download candidates CSV", (out / "candidates.csv").read_bytes(),
                       file_name="candidates.csv", mime="text/csv")
