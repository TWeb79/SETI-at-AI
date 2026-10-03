"""Streamlit front end: `streamlit run app.py`. Runs locally; nothing is uploaded."""
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ai_seti.config import SearchConfig
from ai_seti.io.filterbank import read_header
from ai_seti.pipeline import run_units, score_candidates
from ai_seti.report import write_outputs
from ai_seti.sources import SyntheticSource, split

st.set_page_config(page_title="AI-SETI", layout="wide")
st.title("AI-SETI signal search")
st.caption("Candidates are unverified statistical detections, not technosignature claims.")

mode = st.sidebar.radio("Input", ["Synthetic demo", "Local filterbank (.fil / .h5)"])
cfg = SearchConfig()
cfg.snr_threshold = st.sidebar.slider("De-Doppler SNR threshold", 6.0, 25.0, 10.0, 0.5)
cfg.max_drift_rate_hz_s = st.sidebar.slider("Max drift (Hz/s)", 0.5, 10.0, 4.0, 0.5)
cfg.workers = st.sidebar.number_input("Workers (0 = all cores)", 0, 256, 0)

if mode == "Synthetic demo":
    src = SyntheticSource(Path(tempfile.mkdtemp()), n_chan=1 << 19)
    location, kind, header, meta = next(src.observations(set()))
    chan_range = None
else:
    path = st.sidebar.text_input("Path to file")
    if not path or not Path(path).expanduser().is_file():
        st.info("Enter the path to a local Breakthrough Listen filterbank file.")
        st.stop()
    location, kind = str(Path(path).expanduser()), "local"
    header = read_header(Path(location))
    meta = {"target": header.source_name, "source": "local"}
    f0 = st.sidebar.number_input("Start MHz (0 = whole file)", value=0.0, format="%.6f")
    f1 = st.sidebar.number_input("Stop MHz (0 = whole file)", value=0.0, format="%.6f")
    chan_range = header.channel_range(f0 or None, f1 or None)

if st.sidebar.button("Start crunching", type="primary"):
    units = split(location, header, cfg, kind, chan_range=chan_range, meta=meta)
    bar = st.progress(0.0, text="Crunching work units")
    results = []
    for i, res in enumerate(run_units(units, cfg), 1):
        results.append(res)
        bar.progress(i / len(units), text=f"{i}/{len(units)} work units")
    cands = score_candidates(results, cfg)
    out = Path(tempfile.mkdtemp(prefix="ai_seti_"))
    write_outputs(results, cands, out, cfg, f"AI-SETI search of {meta['target']}")
    components.html((out / "report.html").read_text(), height=1500, scrolling=True)
    st.download_button("Download candidates CSV", (out / "candidates.csv").read_bytes(),
                       file_name="candidates.csv", mime="text/csv")
