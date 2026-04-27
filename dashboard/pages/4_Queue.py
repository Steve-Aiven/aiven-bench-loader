"""
Queue page — live view of the benchmark job queue with log streaming.

Auto-refreshes every 3 seconds while jobs are running.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_pkg_root = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import streamlit as st

from aiven_semantic_search_bench.job_queue import JobQueue

st.set_page_config(page_title="Queue — Aiven k-NN Bench", layout="wide")
st.title("Job Queue")

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

q = JobQueue(RESULTS_DIR)
jobs = q.all_jobs()

# ── Summary metrics ───────────────────────────────────────────────────────────

counts = {"pending": 0, "running": 0, "ok": 0, "failed": 0}
for j in jobs:
    counts[j.state] = counts.get(j.state, 0) + 1

m1, m2, m3, m4 = st.columns(4)
m1.metric("Pending", counts["pending"])
m2.metric("Running", counts["running"])
m3.metric("Completed", counts["ok"])
m4.metric("Failed", counts["failed"])

# Auto-refresh while something is running or pending.
active = counts["pending"] + counts["running"]
if active > 0:
    st.info(f"{active} job(s) active — auto-refreshing every 3 s.")
    time.sleep(3)
    st.rerun()

# ── Job table ─────────────────────────────────────────────────────────────────

if not jobs:
    st.info("No jobs in queue. Use **New Test** to submit a benchmark matrix.")
    st.stop()

import pandas as pd

_STATUS_ICONS = {"pending": "⏳", "running": "⚙️", "ok": "✅", "failed": "❌"}

rows = []
for j in jobs:
    rows.append(
        {
            "Status":    _STATUS_ICONS.get(j.state, j.state),
            "Job":       j.display_label(),
            "Service":   j.service_label,
            "Type":      j.bench_type,
            "Submitted": j.submitted_at,
            "Started":   j.started_at,
            "Finished":  j.finished_at,
            "job_id":    j.job_id,
            "log_path":  j.log_path,
            "state":     j.state,
        }
    )

df = pd.DataFrame(rows)
st.dataframe(
    df[["Status", "Job", "Service", "Type", "Submitted", "Started", "Finished"]],
    use_container_width=True,
    hide_index=True,
)

# ── Log viewer ────────────────────────────────────────────────────────────────

st.subheader("Log viewer")
running_or_done = [j for j in jobs if j.state in ("running", "ok", "failed")]
if not running_or_done:
    st.info("Logs appear here once a job starts.")
else:
    job_options = {j.display_label(): j for j in reversed(running_or_done)}
    selected_label = st.selectbox("Select job", options=list(job_options.keys()))
    selected_job = job_options[selected_label]
    log_file = Path(selected_job.log_path) if selected_job.log_path else None

    if log_file and log_file.exists():
        log_text = log_file.read_text(errors="replace")
        st.code(log_text, language="text")
    else:
        st.info("Log file not yet available.")

# ── Clear finished ────────────────────────────────────────────────────────────

st.divider()
if st.button("Clear completed / failed jobs from queue"):
    removed = q.clear_finished()
    st.success(f"Removed {removed} finished job(s).")
    st.rerun()
