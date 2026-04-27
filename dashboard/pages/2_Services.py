"""
Services page — pick a project and select OpenSearch services to benchmark.

Up to three services may be selected and each tagged with a label
(e.g. ``v2.17``, ``v2.19``, ``v3.3``).  The label is used in report
filenames and chart legends so runs from different services appear side-by-side.
"""

from __future__ import annotations

import os
import sys

_pkg_root = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import streamlit as st
from aiven_semantic_search_bench.aiven_client import AivenDiscovery

st.set_page_config(page_title="Services — Aiven k-NN Bench", layout="wide")
st.title("Services")

if not st.session_state.get("aiven_token"):
    st.warning("Please log in on the **Login** page first.")
    st.stop()

token: str = st.session_state["aiven_token"]
projects: list[str] = st.session_state.get("aiven_projects", [])
discovery = AivenDiscovery(api_token=token)

# ── Project selection ─────────────────────────────────────────────────────────
if not projects:
    with st.spinner("Loading projects…"):
        try:
            projects = discovery.project_names()
            st.session_state["aiven_projects"] = projects
        except Exception as exc:
            st.error(f"Could not load projects: {exc}")
            st.stop()

selected_project = st.selectbox(
    "Project",
    options=projects,
    index=0,
    key="project_selector",
)

# ── Service table ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner="Loading services…")
def _load_services(project: str, token: str):
    d = AivenDiscovery(api_token=token)
    return d.list_services(project)


services = _load_services(selected_project, token)

if not services:
    st.info(
        f"No OpenSearch services found in project **{selected_project}**. "
        "Create one at [console.aiven.io](https://console.aiven.io)."
    )
    st.stop()

import pandas as pd

df = pd.DataFrame(
    [
        {
            "Name": s.name,
            "Version": s.opensearch_version or "?",
            "Plan": s.plan,
            "Cloud": s.cloud_name,
            "State": s.state,
        }
        for s in services
    ]
)

st.dataframe(df, use_container_width=True, hide_index=True)

# ── Service selection ─────────────────────────────────────────────────────────
st.subheader("Select services to benchmark")
st.caption("Pick up to 3 services. Assign each a short version label for charts.")

existing_selected: list[dict] = st.session_state.get("selected_services", [])
max_services = 3

svc_map = {s.name: s for s in services}

with st.form("service_selection_form"):
    cols = st.columns([3, 2])
    selected_names = cols[0].multiselect(
        "Services (max 3)",
        options=[s.name for s in services],
        default=[s["name"] for s in existing_selected if s["name"] in svc_map],
        max_selections=max_services,
    )

    labels_raw = cols[1].text_input(
        "Labels (comma-separated, one per service)",
        value=", ".join(s["label"] for s in existing_selected)
        if existing_selected
        else "v2.17, v2.19, v3.3",
        help="Short labels used in chart legends, e.g. 'v2.17, v2.19, v3.3'",
    )

    confirm = st.form_submit_button("Confirm selection")

if confirm:
    labels = [l.strip() for l in labels_raw.split(",") if l.strip()]
    if len(labels) < len(selected_names):
        labels += [f"svc{i+1}" for i in range(len(labels), len(selected_names))]

    selection = []
    for name, label in zip(selected_names, labels):
        svc = svc_map[name]
        if not svc.service_uri:
            st.error(
                f"Service '{name}' has no service_uri — it may still be provisioning."
            )
            continue
        selection.append(
            {
                "name":    name,
                "label":   label,
                "version": svc.opensearch_version or "?",
                "plan":    svc.plan,
                "cloud":   svc.cloud_name,
                "uri":     svc.service_uri,
            }
        )

    st.session_state["selected_services"] = selection
    if selection:
        st.success(
            f"Selected {len(selection)} service(s): "
            + ", ".join(f"{s['label']} → {s['name']}" for s in selection)
        )
        st.info("Go to **New Test** to configure the benchmark matrix.")
    else:
        st.warning("No services selected.")

# ── Current selection summary ─────────────────────────────────────────────────
current = st.session_state.get("selected_services", [])
if current:
    st.divider()
    st.subheader("Current selection")
    sel_df = pd.DataFrame(
        [
            {
                "Label":   s["label"],
                "Service": s["name"],
                "Version": s["version"],
                "Plan":    s["plan"],
                "Cloud":   s["cloud"],
            }
            for s in current
        ]
    )
    st.dataframe(sel_df, use_container_width=True, hide_index=True)
