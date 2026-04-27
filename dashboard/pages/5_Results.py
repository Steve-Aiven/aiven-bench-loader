"""
Results page — charts grouped by version × engine.

Reads every JSON report in RESULTS_DIR and renders tabs for each benchmark
type.  The ``plan_label`` field is parsed to extract ``(version, engine, mode,
compression)`` tuples for grouped colour axes.

Label convention produced by BenchmarkJob.display_label():
    {service_label}/{engine}/{mode}[/{compression}][/{data_type}][/derived]/{bench_type}

For backwards compatibility with plain string labels (e.g. ``free-tier``),
unparseable labels are shown as-is.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_pkg_root = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Results — Aiven k-NN Bench", layout="wide")
st.title("Results")

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_reports(results_dir: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    p = Path(results_dir)
    if not p.exists():
        return {}
    for path in sorted(p.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data["_filename"] = path.name
        grouped[data.get("name", "unknown")].append(data)
    return dict(grouped)


def _plan_label(report: dict[str, Any]) -> str:
    return report.get("params", {}).get("plan_label", "unlabeled")


def _version_engine(label: str) -> tuple[str, str]:
    """
    Parse a display_label like 'v2.17/faiss/in_memory/32x/index' into
    (version, engine).  Falls back to (label, 'unknown') for unrecognised formats.
    """
    parts = label.split("/")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return label, "unknown"


def _select_reports(
    reports: list[dict[str, Any]], key: str
) -> list[dict[str, Any]]:
    options = [_plan_label(r) for r in reports]
    chosen = st.sidebar.multiselect(
        f"Runs ({key})", options=options, default=options, key=f"sel-{key}"
    )
    return [r for r, lbl in zip(reports, options) if lbl in chosen]


# ── Render helpers ────────────────────────────────────────────────────────────

def render_bench_index(reports: list[dict[str, Any]]) -> None:
    st.header("bench-index — indexing throughput")
    chosen = _select_reports(reports, "bench-index")
    if not chosen:
        st.info("No runs selected.")
        return

    rows = []
    for r in chosen:
        lbl = _plan_label(r)
        version, engine = _version_engine(lbl)
        for entry in r.get("results", []):
            rows.append({"label": lbl, "version": version, "engine": engine, **entry})
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No result rows.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(
            df, x="batch_size", y="docs_per_sec", color="label",
            markers=True, log_x=True, title="Throughput: docs/sec vs batch size",
        )
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.line(
            df, x="batch_size", y="p95_ms", color="label",
            markers=True, log_x=True, title="p95 latency vs batch size",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)


def render_bench_search(reports: list[dict[str, Any]]) -> None:
    st.header("bench-search — query latency over rounds")
    chosen = _select_reports(reports, "bench-search")
    if not chosen:
        st.info("No runs selected.")
        return

    rows = []
    for r in chosen:
        lbl = _plan_label(r)
        for entry in r.get("results", []):
            rows.append({"label": lbl, **entry})
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No result rows.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(df, x="round", y="p95_ms", color="label", markers=True, title="p95 ms per round")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        agg = df.groupby("label", as_index=False)["p95_ms"].mean()
        fig = px.bar(agg, x="label", y="p95_ms", title="Average p95 ms", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)


def render_bench_recall(reports: list[dict[str, Any]]) -> None:
    st.header("bench-recall — recall@K vs latency")
    chosen = _select_reports(reports, "bench-recall")
    if not chosen:
        st.info("No runs selected.")
        return

    rows = []
    for r in chosen:
        lbl = _plan_label(r)
        version, engine = _version_engine(lbl)
        for entry in r.get("results", []):
            rows.append({"label": lbl, "version": version, "engine": engine, **entry})
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No result rows.")
        return

    recall_cols = [c for c in df.columns if c.startswith("recall@")]
    if recall_cols:
        recall_df = df.melt(id_vars=["label", "p95_ms"], value_vars=recall_cols, var_name="K", value_name="recall")
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(recall_df, x="K", y="recall", color="label", barmode="group",
                         title="Recall@K by configuration", text_auto=".3f")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Recall@10 vs p95 scatter
            r10 = df[["label", "p95_ms"]].copy()
            r10["recall@10"] = df.get("recall@10", 0.0)
            fig = px.scatter(r10, x="p95_ms", y="recall@10", color="label",
                             title="Recall@10 vs p95 ms (Pareto frontier)", size_max=15)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)


def render_bench_hybrid(reports: list[dict[str, Any]]) -> None:
    st.header("bench-hybrid — hybrid query latency + recall")
    chosen = _select_reports(reports, "bench-hybrid")
    if not chosen:
        st.info("No runs selected.")
        return

    rows = []
    for r in chosen:
        lbl = _plan_label(r)
        for entry in r.get("results", []):
            rows.append({"label": lbl, **entry})
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No result rows.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="label", y="p95_ms", color="filter_selectivity",
                     barmode="group", title="p95 ms by filter selectivity", text_auto=".0f")
        st.plotly_chart(fig, use_container_width=True)
    if "recall@10" in df.columns:
        with col2:
            fig = px.bar(df, x="label", y="recall@10", color="filter_selectivity",
                         barmode="group", title="Recall@10 by filter selectivity", text_auto=".3f")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)


def render_bench_recover(reports: list[dict[str, Any]]) -> None:
    st.header("bench-recover — cold-start cost")
    chosen = _select_reports(reports, "bench-recover")
    if not chosen:
        st.info("No runs selected.")
        return

    rows = []
    for r in chosen:
        lbl = _plan_label(r)
        for entry in r.get("results", []):
            rows.append({"label": lbl, **entry})
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No result rows.")
        return

    fig = px.bar(df, x="label", y="latency_ms", color="type", barmode="group",
                 title="Cold-start vs warm latency", text_auto=".0f")
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)


def render_bench_plan_change(reports: list[dict[str, Any]]) -> None:
    st.header("bench-plan-change — upgrade impact")
    chosen = _select_reports(reports, "bench-plan-change")
    if not chosen:
        st.info("No runs selected.")
        return

    rows = []
    for r in chosen:
        lbl = _plan_label(r)
        for entry in r.get("results", []):
            rows.append({"label": lbl, **entry})
    df = pd.DataFrame(rows)
    if df.empty:
        st.warning("No result rows.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(df, x="label", y="p95_ms", color="phase", barmode="group",
                     title="p95 ms by phase", text_auto=".0f")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(df, x="label", y="errors", color="phase", barmode="group",
                     title="Errors by phase")
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("Raw data"):
        st.dataframe(df, use_container_width=True)


# ── Main ──────────────────────────────────────────────────────────────────────

grouped = load_reports(RESULTS_DIR)

if not grouped:
    st.warning(
        f"No reports found in `{Path(RESULTS_DIR).resolve()}`. "
        "Run a benchmark from the **New Test** page or via the CLI."
    )
    st.stop()

st.sidebar.header("Filter runs")
st.sidebar.caption("Runs are grouped by their `plan_label` tag.")

tab_names = [
    "bench-index",
    "bench-search",
    "bench-recall",
    "bench-hybrid",
    "bench-recover",
    "bench-plan-change",
]
tabs = st.tabs(tab_names)

with tabs[0]:
    render_bench_index(grouped.get("bench-index", []))
with tabs[1]:
    render_bench_search(grouped.get("bench-search", []))
with tabs[2]:
    render_bench_recall(grouped.get("bench-recall", []))
with tabs[3]:
    render_bench_hybrid(grouped.get("bench-hybrid", []))
with tabs[4]:
    render_bench_recover(grouped.get("bench-recover", []))
with tabs[5]:
    render_bench_plan_change(grouped.get("bench-plan-change", []))
