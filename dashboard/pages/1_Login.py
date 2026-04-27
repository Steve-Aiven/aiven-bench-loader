"""
Login page — paste an Aiven personal API token and verify it.

The token is stored only in ``st.session_state["aiven_token"]``.
It is never written to disk and is not logged anywhere.
"""

from __future__ import annotations

import os
import sys

_pkg_root = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import streamlit as st
from aiven_semantic_search_bench.aiven_client import AivenDiscovery

st.set_page_config(page_title="Login — Aiven k-NN Bench", layout="wide")
st.title("Login")
st.markdown(
    "Enter your Aiven personal API token. "
    "You can create one at [console.aiven.io/account/tokens](https://console.aiven.io/account/tokens)."
)

current_token = st.session_state.get("aiven_token", "")

with st.form("login_form"):
    token_input = st.text_input(
        "API Token",
        value="",
        type="password",
        placeholder="aiven1:…",
        help="Your Aiven personal access token. Never stored to disk.",
    )
    submitted = st.form_submit_button("Verify & Connect")

if submitted:
    token = token_input.strip()
    if not token:
        st.error("Token cannot be empty.")
    else:
        with st.spinner("Verifying token…"):
            discovery = AivenDiscovery(api_token=token)
            try:
                projects = discovery.project_names()
                st.session_state["aiven_token"] = token
                st.session_state["aiven_projects"] = projects
                # Clear stale selections from a previous login.
                st.session_state.pop("selected_services", None)
                st.success(
                    f"Connected! Found {len(projects)} project(s): "
                    + ", ".join(projects[:5])
                    + (" …" if len(projects) > 5 else "")
                )
                st.info("Navigate to **Services** in the sidebar to pick your OpenSearch services.")
            except Exception as exc:
                st.error(f"Authentication failed: {exc}")

if current_token:
    st.divider()
    st.success("You are already logged in.")
    if st.button("Log out"):
        for key in ("aiven_token", "aiven_projects", "selected_services"):
            st.session_state.pop(key, None)
        st.rerun()
