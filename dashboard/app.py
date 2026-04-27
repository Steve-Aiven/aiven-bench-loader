"""
Aiven OpenSearch k-NN benchmark — Streamlit entry point.

This file is the main page of a multi-page Streamlit app.  Sub-pages live in
dashboard/pages/.  This module also starts the background job runner thread
exactly once.

Pages:
  1 Login     — paste Aiven token, verify, store in session state
  2 Services  — pick project + OpenSearch services, tag with version labels
  3 New Test  — build the benchmark matrix and submit jobs
  4 Queue     — live job queue with log streaming
  5 Results   — charts grouped by version × engine
"""

from __future__ import annotations

import os
import sys

import streamlit as st

# Ensure the package is importable when running streamlit directly from the
# dashboard/ directory (e.g. `streamlit run app.py`).
_pkg_root = os.path.join(os.path.dirname(__file__), "..", "src")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from aiven_semantic_search_bench.job_runner import ensure_started

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

# Start the runner daemon once per container process.
ensure_started(RESULTS_DIR)

st.set_page_config(
    page_title="Aiven k-NN Benchmark",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _is_logged_in() -> bool:
    return bool(st.session_state.get("aiven_token"))


def _has_services() -> bool:
    return bool(st.session_state.get("selected_services"))


st.title("Aiven OpenSearch k-NN Benchmark")

if not _is_logged_in():
    st.info("Use the **Login** page in the sidebar to connect your Aiven account.")
elif not _has_services():
    st.info("Use the **Services** page to select OpenSearch services to benchmark.")
else:
    services = st.session_state.get("selected_services", [])
    st.success(
        f"Connected — {len(services)} service(s) selected: "
        + ", ".join(f"{s['label']} ({s['name']})" for s in services)
    )
    st.markdown(
        "Use the sidebar to navigate: **New Test** to submit a benchmark matrix, "
        "**Queue** to watch progress, **Results** to view charts."
    )

st.caption(
    "Token lives in browser session only — it is never written to disk or logged."
)
