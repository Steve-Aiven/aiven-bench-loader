"""Aiven OpenSearch k-NN Benchmark — NiceGUI single-page UI.

Five tabs in a single browser page — no page switching, no lost form state.
The Queue tab auto-refreshes every 3 s via WebSocket without disturbing other tabs.
Session (token + selected services) is persisted to results/.bench_session.json.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import sys
import time

from pathlib import Path
from typing import Any

_pkg_root = Path(__file__).parent.parent / "src"
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import pandas as pd
import plotly.express as px
from nicegui import ui, app as _app

from aiven_semantic_search_bench.aiven_client import AivenDiscovery
from aiven_semantic_search_bench.hf_embedder import RECOMMENDED_MODELS
from aiven_semantic_search_bench.job_queue import JobQueue
from aiven_semantic_search_bench.job_runner import ensure_started
from aiven_semantic_search_bench.job_spec import BenchmarkJob
from aiven_semantic_search_bench.opensearch_client import KnnSpec

import experiments as _experiments

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
MATRICES_DIR = Path(RESULTS_DIR) / "matrices"
MATRICES_DIR.mkdir(parents=True, exist_ok=True)
_SESSION_FILE = Path(RESULTS_DIR) / ".bench_session.json"

ensure_started(RESULTS_DIR)

# One-time clean-slate migration: remove top-level results/*.json/.md files
_experiments.migrate_clean_slate(RESULTS_DIR)

# ── Session helpers ────────────────────────────────────────────────────────────

def _load_session() -> dict:
    try:
        return json.loads(_SESSION_FILE.read_text())
    except Exception:
        return {}


def _save_session(sess: dict) -> None:
    _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    _SESSION_FILE.write_text(json.dumps(sess, indent=2))


# ── Domain helpers ─────────────────────────────────────────────────────────────

_STATUS_ICONS = {"pending": "⏳", "running": "⚙️", "ok": "✅", "failed": "❌", "skipped": "⏭️"}


def _slugify(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.strip().lower())


def _parse_ints(raw: str, fallback: list[int]) -> list[int]:
    out = [int(p.strip()) for p in raw.split(",") if p.strip().isdigit()]
    return out or fallback


def _skip_reason(engine: str, method: str, mode: str, compression: str, data_type: str) -> str | None:
    if method == "ivf" and engine != "faiss":
        return "ivf requires faiss engine"
    if mode == "on_disk" and engine != "faiss":
        return "on_disk requires faiss engine"
    if compression != "none" and mode != "on_disk":
        return "compression requires on_disk mode"
    if data_type in ("byte", "fp16", "binary") and engine != "faiss":
        return f"data_type={data_type} requires faiss engine"
    return None


def _corpus_status(corpus_dir: str) -> dict:
    p = Path(corpus_dir)
    manifest_file = p / "manifest.json"
    qrels_file = p / "qrels.npy"
    if not manifest_file.exists():
        return {"exists": False}
    try:
        m = json.loads(manifest_file.read_text())
        return {
            "exists": True,
            "docs": m.get("actual_docs", "?"),
            "queries": m.get("actual_queries", "?"),
            "model": m.get("embed_model", "?"),
            "source_dim": m.get("source_dim", "?"),
            "preset": m.get("preset", "?"),
            "has_metadata": m.get("has_metadata", False),
            "created_at": m.get("created_at_utc", "?"),
            "has_groundtruth": qrels_file.exists(),
        }
    except Exception:
        return {"exists": False}


def _load_reports_for_experiment(results_dir: str, slug: str) -> dict[str, list[dict[str, Any]]]:
    """Load benchmark reports grouped by bench type for one experiment slug."""
    return _experiments.load_reports(results_dir, slug)


SAMPLE_MATRICES: dict[str, dict] = {
    "Quick Pilot": {
        "_description": "10k docs, Faiss + Lucene HNSW float in-memory. Fast sanity-check (~10 min). Uses batch_size=50,500 only.",
        "corpus": {"doc_count": 10000, "query_count": 1000, "embed_dim": 384, "corpus_dir": "corpus", "opensearch_index": "bench"},
        "knn_axes": {"engines": ["faiss", "lucene"], "methods": ["hnsw"], "modes": ["in_memory"],
                     "compressions": ["none"], "data_types": ["float"],
                     "m": 16, "ef_construction": 128, "ef_search": "256",
                     "with_text": False, "with_metadata": False, "derived_source": False},
        "query_types": {"index": True, "search": True, "recall": False, "hybrid": False, "filter_selectivity": "none"},
        "bench_params": {"batch_sizes": "50,500", "k": 10, "rounds": 3},
    },
    "Full Comparison": {
        "_description": "100k docs, Faiss + Lucene × float + byte, index/search/recall (~1–2 h).",
        "corpus": {"doc_count": 100_000, "query_count": 2000, "embed_dim": 384, "corpus_dir": "corpus", "opensearch_index": "bench"},
        "knn_axes": {"engines": ["faiss", "lucene"], "methods": ["hnsw"], "modes": ["in_memory"],
                     "compressions": ["none"], "data_types": ["float", "byte"],
                     "m": 16, "ef_construction": 128, "ef_search": "256",
                     "with_text": False, "with_metadata": False, "derived_source": False},
        "query_types": {"index": True, "search": True, "recall": True, "hybrid": False, "filter_selectivity": "none"},
        "bench_params": {"batch_sizes": "1,5,10,20,50", "k": 10, "rounds": 5},
    },
    "Disk-Optimized (2.17+)": {
        "_description": "Faiss on_disk with binary quantisation. Cost/memory savings vs in-memory (~45 min).",
        "corpus": {"doc_count": 50_000, "query_count": 1000, "embed_dim": 384, "corpus_dir": "corpus", "opensearch_index": "bench"},
        "knn_axes": {"engines": ["faiss"], "methods": ["hnsw"],
                     "modes": ["in_memory", "on_disk"],
                     "compressions": ["none", "4x", "32x"], "data_types": ["float", "binary"],
                     "m": 16, "ef_construction": 128, "ef_search": "256",
                     "with_text": False, "with_metadata": False, "derived_source": False},
        "query_types": {"index": True, "search": True, "recall": True, "hybrid": False, "filter_selectivity": "none"},
        "bench_params": {"batch_sizes": "10,50", "k": 10, "rounds": 3},
    },
    "Hybrid Search": {
        "_description": "BM25 + k-NN with metadata filter selectivity. Text + metadata enabled (~30 min).",
        "corpus": {"doc_count": 20_000, "query_count": 500, "embed_dim": 384, "corpus_dir": "corpus", "opensearch_index": "bench"},
        "knn_axes": {"engines": ["faiss", "lucene"], "methods": ["hnsw"], "modes": ["in_memory"],
                     "compressions": ["none"], "data_types": ["float"],
                     "m": 16, "ef_construction": 128, "ef_search": "256",
                     "with_text": True, "with_metadata": True, "derived_source": False},
        "query_types": {"index": True, "search": False, "recall": False, "hybrid": True, "filter_selectivity": "low"},
        "bench_params": {"batch_sizes": "10,50", "k": 10, "rounds": 3},
    },
    "Chaos / Saturation": {
        "_description": "High-workload stress test. Requires a pre-populated index. 8 index threads + 16 search threads at k=100 for 5 minutes.",
        "corpus": {"doc_count": 10_000, "query_count": 1000, "embed_dim": 384, "corpus_dir": "corpus", "opensearch_index": "bench"},
        "knn_axes": {"engines": ["faiss"], "methods": ["hnsw"], "modes": ["in_memory"],
                     "compressions": ["none"], "data_types": ["float"],
                     "m": 16, "ef_construction": 128, "ef_search": "256",
                     "with_text": False, "with_metadata": False, "derived_source": False},
        "query_types": {"index": False, "search": False, "recall": False, "hybrid": False, "stress": True, "filter_selectivity": "none"},
        "bench_params": {"batch_sizes": "50", "k": 10, "rounds": 3},
        "stress_params": {"stress_index_clients": 8, "stress_search_clients": 16, "stress_duration": 300, "stress_batch_size": 100, "stress_k": 100},
    },
    "FP16 / BFloat16 (3.3+)": {
        "_description": "Faiss fp16 vs float — ~16% latency improvement from native SIMD FP16 in OpenSearch 3.3.",
        "corpus": {"doc_count": 50_000, "query_count": 1000, "embed_dim": 384, "corpus_dir": "corpus", "opensearch_index": "bench"},
        "knn_axes": {"engines": ["faiss"], "methods": ["hnsw"], "modes": ["in_memory"],
                     "compressions": ["none"], "data_types": ["float", "fp16"],
                     "m": 16, "ef_construction": 128, "ef_search": "256",
                     "with_text": False, "with_metadata": False, "derived_source": False},
        "query_types": {"index": True, "search": True, "recall": True, "hybrid": False, "filter_selectivity": "none"},
        "bench_params": {"batch_sizes": "10,50", "k": 10, "rounds": 5},
    },
}


# ── Page ───────────────────────────────────────────────────────────────────────

@ui.page("/")
async def main() -> None:  # noqa: C901 — long but linear; split would hurt readability
    sess = _load_session()
    env_token = os.environ.get("AIVEN_API_TOKEN", "")
    init_token = sess.get("aiven_token", env_token)
    init_projects: list[str] = sess.get("aiven_projects", [])
    init_services: list[dict] = sess.get("selected_services", [])
    init_experiment: str = sess.get("current_experiment", "")

    # Per-connection mutable state shared across closures
    _conn: dict[str, Any] = {
        "token": init_token,
        "projects": init_projects,
        "selected_services": init_services,
        "log_job_id": None,
        "current_experiment": init_experiment,
    }

    # ── Header ─────────────────────────────────────────────────────────────────
    with ui.header().classes("bg-blue-900 text-white items-center px-6 py-2 shadow-md gap-4"):
        ui.label("Aiven OpenSearch k-NN Benchmark").classes("text-lg font-bold")
        ui.space()
        hdr_status = ui.label("").classes("text-sm opacity-75")

    def _refresh_header() -> None:
        svcs = _conn["selected_services"]
        if not _conn["token"]:
            hdr_status.set_text("Not logged in")
        elif not svcs:
            hdr_status.set_text("Logged in — no services selected")
        else:
            hdr_status.set_text("Benchmarking: " + " · ".join(s["label"] for s in svcs))

    _refresh_header()

    # ── Tab bar ────────────────────────────────────────────────────────────────
    with ui.tabs().classes("w-full bg-blue-800 text-white") as main_tabs:
        tab_svcs    = ui.tab("Services",  icon="cloud")
        tab_corpus  = ui.tab("Corpus",    icon="storage")
        tab_test    = ui.tab("New Test",  icon="play_arrow")
        tab_queue   = ui.tab("Queue",     icon="queue")
        tab_results = ui.tab("Results",   icon="bar_chart")

    with ui.tab_panels(main_tabs, value=tab_svcs).classes("w-full"):

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — SERVICES
        # ══════════════════════════════════════════════════════════════════════
        with ui.tab_panel(tab_svcs):
            with ui.card().classes("max-w-4xl mx-auto w-full mt-4"):
                ui.label("Connect to Aiven").classes("text-lg font-semibold")
                ui.markdown(
                    "Paste your [personal API token](https://console.aiven.io/account/tokens). "
                    "It is saved to `results/.bench_session.json` so you won't need to re-enter it on restart."
                )

                token_input = ui.input(
                    "API Token", value=init_token, password=True, placeholder="aiven1:…",
                ).classes("w-full mt-2")
                conn_label = ui.label(
                    "Saved session token loaded." if init_token else ""
                ).classes("text-sm text-green-600")

                project_col = ui.column().classes("w-full gap-2 mt-2")

                def _build_project_area(token: str, projects: list[str]) -> None:
                    project_col.clear()
                    if not projects:
                        return
                    saved_project = _load_session().get("selected_project", projects[0])
                    init_project = saved_project if saved_project in projects else projects[0]
                    with project_col:
                        ui.separator()
                        ui.label("Select services to benchmark").classes("font-semibold mt-1")
                        project_sel = ui.select(
                            projects, value=init_project, label="Project",
                            with_input=True,
                        ).classes("w-96")
                        svcs_col = ui.column().classes("w-full")

                        def _render_services(project: str) -> None:
                            svcs_col.clear()
                            with svcs_col:
                                spin = ui.spinner(size="md")
                            try:
                                discovery = AivenDiscovery(api_token=token)
                                svcs = discovery.list_services(project)
                            except Exception as exc:
                                svcs_col.clear()
                                with svcs_col:
                                    ui.label(f"Error loading services: {exc}").classes("text-red-600")
                                return

                            svcs_col.clear()
                            with svcs_col:
                                if not svcs:
                                    ui.label(f"No OpenSearch services in '{project}'.").classes("text-gray-500")
                                    return

                                cols = [
                                    {"name": "name",    "label": "Name",    "field": "name",    "align": "left", "sortable": True},
                                    {"name": "version", "label": "Version", "field": "version", "align": "left"},
                                    {"name": "plan",    "label": "Plan",    "field": "plan",    "align": "left"},
                                    {"name": "cloud",   "label": "Cloud",   "field": "cloud",   "align": "left"},
                                    {"name": "state",   "label": "State",   "field": "state",   "align": "left"},
                                ]
                                rows = [
                                    {"name": s.name, "version": s.opensearch_version or "?",
                                     "plan": s.plan, "cloud": s.cloud_name, "state": s.state}
                                    for s in svcs
                                ]
                                ui.table(columns=cols, rows=rows, row_key="name").classes("w-full")

                                ui.separator()
                                svc_map = {s.name: s for s in svcs}
                                svc_names = [s.name for s in svcs]
                                existing = _conn["selected_services"]
                                existing_names = [s["name"] for s in existing if s["name"] in svc_map]
                                existing_labels_str = ", ".join(s["label"] for s in existing) if existing else "v2.17, v2.19, v3.3"

                                ui.label("Pick up to 3 services and give each a short label for chart legends:").classes("text-sm text-gray-600")
                                with ui.row().classes("w-full gap-4 mt-1"):
                                    svc_pick = ui.select(
                                        svc_names, multiple=True, value=existing_names,
                                        label="Services (max 3)",
                                    ).classes("flex-1")
                                    labels_fld = ui.input(
                                        "Labels (comma-separated, one per service)", value=existing_labels_str,
                                    ).classes("flex-1")

                                sel_status = ui.label("").classes("text-sm")
                                if existing:
                                    sel_status.set_text("Current: " + ", ".join(f"{s['label']} → {s['name']}" for s in existing))
                                    sel_status.classes("text-green-700")

                                def _apply_selection() -> None:
                                    picked: list[str] = list(svc_pick.value or [])[:3]
                                    if not picked:
                                        ui.notify("Select at least one service.", type="warning")
                                        return
                                    raw = labels_fld.value or ""
                                    labels = [lbl.strip() for lbl in raw.split(",") if lbl.strip()]
                                    while len(labels) < len(picked):
                                        labels.append(f"svc{len(labels) + 1}")
                                    selection = []
                                    for name, label in zip(picked, labels):
                                        svc = svc_map.get(name)
                                        if not svc or not svc.service_uri:
                                            ui.notify(f"'{name}' has no URI (still provisioning?).", type="warning")
                                            continue
                                        selection.append({
                                            "name": name, "label": label,
                                            "version": svc.opensearch_version or "?",
                                            "plan": svc.plan, "cloud": svc.cloud_name,
                                            "uri": svc.service_uri,
                                        })
                                    if selection:
                                        _conn["selected_services"] = selection
                                        current_sess = _load_session()
                                        current_sess.update({
                                            "aiven_token": _conn["token"],
                                            "aiven_projects": _conn["projects"],
                                            "selected_project": project_sel.value,
                                            "selected_services": selection,
                                        })
                                        _save_session(current_sess)
                                        _refresh_header()
                                        sel_status.set_text("Saved: " + ", ".join(f"{s['label']} → {s['name']}" for s in selection))
                                        sel_status.classes("text-green-700")
                                        ui.notify(f"{len(selection)} service(s) saved.", type="positive")

                                ui.button("Apply selection", on_click=_apply_selection, color="blue").classes("mt-2")

                        def _on_project_change(e) -> None:
                            current_sess = _load_session()
                            current_sess["selected_project"] = e.value
                            _save_session(current_sess)
                            _render_services(e.value)

                        project_sel.on_value_change(_on_project_change)
                        _render_services(init_project)

                def _connect() -> None:
                    token = token_input.value.strip()
                    if not token:
                        conn_label.set_text("Token cannot be empty.")
                        conn_label.classes("text-red-600", remove="text-green-600")
                        return
                    conn_label.set_text("Verifying…")
                    conn_label.classes("text-gray-500", remove="text-green-600 text-red-600")
                    try:
                        discovery = AivenDiscovery(api_token=token)
                        projects = discovery.project_names()
                        _conn["token"] = token
                        _conn["projects"] = projects
                        current_sess = _load_session()
                        current_sess.update({"aiven_token": token, "aiven_projects": projects})
                        _save_session(current_sess)
                        conn_label.set_text(f"Connected — {len(projects)} project(s) found.")
                        conn_label.classes("text-green-600", remove="text-gray-500 text-red-600")
                        _refresh_header()
                        _build_project_area(token, projects)
                    except Exception as exc:
                        conn_label.set_text(f"Authentication failed: {exc}")
                        conn_label.classes("text-red-600", remove="text-gray-500 text-green-600")

                def _logout() -> None:
                    _conn.update({"token": "", "projects": [], "selected_services": []})
                    _save_session({})
                    token_input.value = ""
                    project_col.clear()
                    conn_label.set_text("Logged out.")
                    conn_label.classes("text-gray-500", remove="text-green-600 text-red-600")
                    _refresh_header()
                    ui.notify("Logged out.", type="info")

                with ui.row().classes("gap-2 mt-2"):
                    ui.button("Connect", on_click=_connect, color="blue")
                    ui.button("Log out", on_click=_logout).props("flat")

                if init_token and init_projects:
                    _build_project_area(init_token, init_projects)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — CORPUS
        # ══════════════════════════════════════════════════════════════════════
        with ui.tab_panel(tab_corpus):
            with ui.card().classes("max-w-4xl mx-auto w-full mt-4 gap-4"):
                ui.label("Corpus Management").classes("text-lg font-semibold")

                corpus_dir_in = ui.input("Corpus directory", value="corpus").classes("w-full mt-2")

                @ui.refreshable
                def corpus_status_view() -> None:
                    cs = _corpus_status(corpus_dir_in.value or "corpus")
                    with ui.row().classes("w-full gap-4 mt-2"):
                        with ui.card().classes("flex-1 items-center text-center"):
                            if cs["exists"]:
                                ui.icon("check_circle", color="positive", size="xl")
                                ui.label("Corpus ready").classes("font-semibold text-green-700 mt-1")
                                ui.label(f"{cs['docs']:,} docs / {cs['queries']:,} queries").classes("text-sm")
                                ui.label(f"dim={cs['source_dim']}  model={cs['model']}").classes("text-sm text-gray-500")
                                ui.label(f"Built: {cs['created_at']}").classes("text-xs text-gray-400 mt-1")
                            else:
                                ui.icon("warning", color="warning", size="xl")
                                ui.label("Corpus missing").classes("font-semibold text-orange-700 mt-1")
                                ui.label("Benchmarks will fail until you build the corpus.").classes("text-sm")

                        with ui.card().classes("flex-1 items-center text-center"):
                            if cs.get("has_groundtruth"):
                                ui.icon("check_circle", color="positive", size="xl")
                                ui.label("Ground truth ready").classes("font-semibold text-green-700 mt-1")
                                ui.label("Recall benchmarks are enabled.").classes("text-sm")
                            elif cs["exists"]:
                                ui.icon("info", color="info", size="xl")
                                ui.label("Ground truth missing").classes("font-semibold text-blue-700 mt-1")
                                ui.label("Enable below or recall benchmarks will be skipped.").classes("text-sm")
                            else:
                                ui.label("Build corpus first.").classes("text-sm text-gray-400")

                corpus_status_view()
                ui.button(
                    "Refresh status", icon="refresh", on_click=corpus_status_view.refresh,
                ).props("flat size=sm color=grey").classes("mt-1")

                ui.separator()
                ui.label("Build corpus").classes("font-semibold mt-2")
                ui.label(
                    "Downloads the HuggingFace dataset and embeds locally — no API key needed. "
                    "Model weights are cached after the first run."
                ).classes("text-sm text-gray-500")

                _model_labels = [m[0] for m in RECOMMENDED_MODELS]
                _model_ids    = [m[1] for m in RECOMMENDED_MODELS]
                _model_dims   = [m[2] for m in RECOMMENDED_MODELS]
                _model_descs  = [m[3] for m in RECOMMENDED_MODELS]

                with ui.grid(columns=2).classes("w-full gap-3 mt-3"):
                    bc_preset  = ui.select(["mixed", "msmarco", "quora", "fiqa", "scifact"], value="mixed", label="Dataset preset").classes("w-full")
                    bc_model   = ui.select(_model_labels, value=_model_labels[0], label="Embedding model").classes("w-full")
                    bc_docs    = ui.number("Doc count",   value=10_000, min=100, max=1_000_000, step=1000).classes("w-full")
                    bc_queries = ui.number("Query count", value=1_000,  min=10,  max=100_000,   step=100).classes("w-full")
                    bc_meta    = ui.checkbox("Include metadata columns (for filter tests)", value=False)
                    bc_gt      = ui.checkbox("Build ground truth after corpus (for recall)", value=True)

                bc_model_desc = ui.label("").classes("text-xs text-gray-500 mt-1")

                def _update_model_desc(_e: Any = None) -> None:
                    lbl = bc_model.value
                    if lbl in _model_labels:
                        bc_model_desc.set_text(_model_descs[_model_labels.index(lbl)])

                bc_model.on_value_change(_update_model_desc)
                _update_model_desc()

                with ui.expansion("Speed tip — native build on Mac Apple Silicon", icon="bolt").classes("mt-3 w-full"):
                    ui.markdown(
                        "The Docker container uses **Linux ARM64 CPU** for inference — no MPS/Metal access.  "
                        "On an M4 Pro, running the build natively is **5–10× faster**:\n\n"
                        "```bash\n"
                        "pip install -e '.[build]'\n"
                        "HF_EMBED_MODEL='BAAI/bge-small-en-v1.5' \\\n"
                        "python -m aiven_semantic_search_bench bench-build-corpus \\\n"
                        "    --dataset mixed --doc-count 10000 --query-count 1000\n"
                        "```\n\n"
                        "The `corpus/` folder is bind-mounted into Docker and immediately available to the container."
                    )

                def _queue_corpus_build() -> None:
                    lbl = bc_model.value
                    idx = _model_labels.index(lbl) if lbl in _model_labels else 0
                    job = BenchmarkJob(
                        bench_type="build-corpus",
                        service_label="corpus-build",
                        opensearch_uri="",
                        opensearch_version="",
                        opensearch_index="",
                        spec=KnnSpec(embed_dim=_model_dims[idx]),
                        embed_dim=_model_dims[idx],
                        doc_count=int(bc_docs.value),
                        query_count=int(bc_queries.value),
                        corpus_dir=corpus_dir_in.value or "corpus",
                        out_dir=RESULTS_DIR,
                        corpus_preset=bc_preset.value,
                        corpus_with_metadata=bc_meta.value,
                        corpus_with_groundtruth=bc_gt.value,
                        corpus_embed_model=_model_ids[idx],
                    )
                    JobQueue(RESULTS_DIR).enqueue(job)
                    ui.notify(
                        f"Queued corpus build: {bc_preset.value} / {int(bc_docs.value):,} docs / "
                        f"{_model_ids[idx]} (dim={_model_dims[idx]}). Switch to Queue tab to watch.",
                        type="positive",
                    )

                ui.button(
                    "Queue corpus build", on_click=_queue_corpus_build, color="blue", icon="add_task",
                ).classes("mt-4")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — NEW TEST
        # ══════════════════════════════════════════════════════════════════════
        with ui.tab_panel(tab_test):
            with ui.column().classes("max-w-5xl mx-auto w-full mt-4 gap-4"):

                # ── Shared helpers ─────────────────────────────────────────────
                def _corpus_embed_dims(corpus_dir: str) -> tuple[list[int], int]:
                    try:
                        m = json.loads((Path(corpus_dir) / "manifest.json").read_text())
                        dims = sorted(m.get("supported_dims") or [256, 384, 512, 768])
                        return dims, dims[-1]
                    except Exception:
                        return [256, 384, 512, 768], 384

                _dim_opts, _dim_default = _corpus_embed_dims("corpus")

                def _no_services_warning() -> bool:
                    if not _conn["selected_services"]:
                        ui.notify("No services selected — go to the Services tab first.", type="warning")
                        return True
                    return False

                def _current_out_dir() -> str:
                    """Return the out_dir for the currently selected experiment, or RESULTS_DIR."""
                    slug = _conn.get("current_experiment", "")
                    if slug:
                        d = _experiments.experiment_dir(RESULTS_DIR, slug)
                        d.mkdir(parents=True, exist_ok=True)
                        return str(d)
                    return RESULTS_DIR

                # ── Shared experiment selector (rendered per card) ──────────────
                def _render_experiment_selector(
                    refresh_callbacks: list | None = None,
                ) -> None:
                    """
                    Render the experiment selector row.  Each card in the New Test tab
                    calls this so the user always sees which experiment is active.
                    ``refresh_callbacks`` is a list of callables to invoke after the
                    experiment changes (e.g. to refresh the Results tab).
                    """
                    exps = _experiments.list_experiments(RESULTS_DIR)
                    exp_names = [e["name"] for e in exps]
                    slug_by_name = {e["name"]: e["slug"] for e in exps}

                    current_slug = _conn.get("current_experiment", "")
                    current_name = next((e["name"] for e in exps if e["slug"] == current_slug), "")

                    ui.label("Experiment").classes("text-xs font-semibold uppercase text-gray-400 mt-2")

                    with ui.row().classes("w-full gap-3 items-end"):
                        exp_sel = ui.select(
                            exp_names or ["(no experiments yet)"],
                            value=current_name or (exp_names[0] if exp_names else "(no experiments yet)"),
                            label="Active experiment",
                        ).classes("flex-1")

                        def _on_exp_change(e: Any) -> None:
                            name = e.value
                            slug = slug_by_name.get(name, "")
                            _conn["current_experiment"] = slug
                            curr = _load_session()
                            curr["current_experiment"] = slug
                            _save_session(curr)
                            for cb in (refresh_callbacks or []):
                                try:
                                    cb()
                                except Exception:
                                    pass

                        exp_sel.on_value_change(_on_exp_change)

                        def _open_new_exp_dialog() -> None:
                            with ui.dialog() as dlg, ui.card().classes("min-w-80"):
                                ui.label("New experiment").classes("font-semibold text-lg")
                                new_name_input = ui.input("Name", placeholder="e.g. v2.17 baseline").classes("w-full")
                                new_desc_input = ui.input("Description (optional)").classes("w-full")

                                def _create() -> None:
                                    name = new_name_input.value.strip()
                                    if not name:
                                        ui.notify("Name cannot be empty.", type="warning")
                                        return
                                    meta = _experiments.create_experiment(RESULTS_DIR, name, new_desc_input.value.strip())
                                    slug = meta["slug"]
                                    _conn["current_experiment"] = slug
                                    curr = _load_session()
                                    curr["current_experiment"] = slug
                                    _save_session(curr)
                                    dlg.close()
                                    ui.notify(f"Experiment '{name}' created.", type="positive")

                                with ui.row().classes("gap-2 mt-3"):
                                    ui.button("Create", on_click=_create, color="blue")
                                    ui.button("Cancel", on_click=dlg.close).props("flat")
                            dlg.open()

                        ui.button("New experiment", on_click=_open_new_exp_dialog, icon="add").props("outline size=sm")

                # ══════════════════════════════════════════════════════════════
                # QUICK TEST 1 — BENCHMARK
                # ══════════════════════════════════════════════════════════════
                with ui.card().classes("w-full border-l-4 border-blue-500"):
                    with ui.row().classes("items-center gap-3 mb-1"):
                        ui.icon("speed", color="blue", size="md")
                        ui.label("Benchmark").classes("text-lg font-bold text-blue-800")
                    ui.label(
                        "Index + search with optimal defaults (Faiss HNSW · batch 50 · k=10 · 3 rounds · OSB warmup). "
                        "Runs sequentially for each selected service."
                    ).classes("text-sm text-gray-500 mb-3")

                    _render_experiment_selector()
                    with ui.grid(columns=3).classes("w-full gap-3 mt-2"):
                        qb_doc_count   = ui.number("Doc count",   value=10_000, min=100, max=10_000_000, step=1000).classes("w-full")
                        qb_query_count = ui.number("Query count", value=500,    min=10,  max=100_000,    step=100).classes("w-full")
                        qb_embed_dim   = ui.select(_dim_opts, value=_dim_default, label="Embed dim").classes("w-full")
                        qb_index_name  = ui.input("Index name",   value="bench").classes("w-full")
                        qb_corpus_dir  = ui.input("Corpus dir",   value="corpus").classes("w-full")

                    with ui.row().classes("gap-6 mt-1"):
                        qb_do_index  = ui.checkbox("Index",  value=True)
                        qb_do_search = ui.checkbox("Search", value=True)
                        qb_do_recall = ui.checkbox("Recall  (needs qrels.npy)", value=False)

                    def _submit_benchmark() -> None:
                        if _no_services_warning():
                            return
                        if not qb_do_index.value and not qb_do_search.value and not qb_do_recall.value:
                            ui.notify("Select at least Index or Search.", type="warning")
                            return
                        if not _conn.get("current_experiment"):
                            ui.notify("Select or create an experiment first.", type="warning")
                            return
                        embed_dim   = int(qb_embed_dim.value)
                        doc_count   = int(qb_doc_count.value)
                        query_count = int(qb_query_count.value)
                        index_name  = qb_index_name.value or "bench"
                        corpus_dir  = qb_corpus_dir.value or "corpus"
                        out_dir     = _current_out_dir()
                        # Optimal fixed defaults
                        spec = KnnSpec(
                            embed_dim=embed_dim,
                            engine="faiss", method="hnsw",
                            space_type="cosinesimil",
                            mode="in_memory", compression="none",
                            data_type="float",
                            m=16, ef_construction=128, ef_search=256,
                        )
                        bench_types = (
                            (["index"]  if qb_do_index.value  else [])
                            + (["search"] if qb_do_search.value else [])
                            + (["recall"] if qb_do_recall.value else [])
                        )
                        q = JobQueue(RESULTS_DIR)
                        n = 0
                        for svc in _conn["selected_services"]:
                            index_job_id = ""
                            for bt in bench_types:
                                job = BenchmarkJob(
                                    bench_type=bt,
                                    service_label=svc["label"],
                                    opensearch_uri=svc["uri"],
                                    opensearch_version=svc["version"],
                                    opensearch_index=index_name,
                                    spec=spec,
                                    embed_dim=embed_dim,
                                    doc_count=doc_count,
                                    query_count=query_count,
                                    corpus_dir=corpus_dir,
                                    out_dir=out_dir,
                                    rounds=3, k=10,
                                    batch_sizes=[50],
                                    warmup_queries=50,
                                    search_clients=1,
                                    depends_on=index_job_id if bt != "index" else "",
                                )
                                q.enqueue(job)
                                if bt == "index":
                                    index_job_id = job.job_id
                                n += 1
                        ui.notify(
                            f"Queued {n} job(s) across {len(_conn['selected_services'])} service(s). "
                            f"Switch to Queue tab to watch.",
                            type="positive",
                        )

                    ui.button(
                        "Run benchmark", on_click=_submit_benchmark,
                        color="blue", icon="play_arrow",
                    ).classes("mt-3")

                # ══════════════════════════════════════════════════════════════
                # QUICK TEST 2 — STRESS + PLAN CHANGE
                # ══════════════════════════════════════════════════════════════
                with ui.card().classes("w-full border-l-4 border-red-500"):
                    with ui.row().classes("items-center gap-3 mb-1"):
                        ui.icon("whatshot", color="red", size="md")
                        ui.label("Stress + Plan Change").classes("text-lg font-bold text-red-800")
                    ui.label(
                        "Hammers one service with concurrent bulk-indexing + k-NN search. "
                        "Optionally triggers a plan change mid-run via the Aiven API. "
                        "Run bench-index first to populate the index."
                    ).classes("text-sm text-gray-500 mb-3")

                    # ── Service selector (single) ──────────────────────────────
                    stress_svc_col = ui.column().classes("w-full")

                    @ui.refreshable
                    def _stress_svc_selector() -> None:
                        stress_svc_col.clear()
                        svcs = _conn["selected_services"]
                        with stress_svc_col:
                            if not svcs:
                                ui.label("No services — go to Services tab first.").classes("text-orange-500 text-sm")
                            else:
                                svc_names = [s["label"] for s in svcs]
                                st_svc_sel._opts = svc_names  # type: ignore[attr-defined]

                    svc_labels_for_stress = [s["label"] for s in _conn["selected_services"]] or ["(none)"]
                    st_svc_sel = ui.select(
                        svc_labels_for_stress,
                        value=svc_labels_for_stress[0],
                        label="Service to stress",
                    ).classes("w-full max-w-xs")

                    _render_experiment_selector()
                    with ui.grid(columns=3).classes("w-full gap-3 mt-2"):
                        st_duration       = ui.number("Duration (s)",           value=120,  min=30,  max=3600, step=30).classes("w-full")
                        st_index_clients  = ui.number("Index clients",           value=8,    min=1,   max=64,   step=1).classes("w-full")
                        st_search_clients = ui.number("Search clients",          value=16,   min=1,   max=64,   step=1).classes("w-full")
                        st_batch_size     = ui.number("Index batch size",         value=100,  min=1,   max=2000, step=50).classes("w-full")
                        st_k              = ui.number("k (search pressure)",      value=100,  min=1,   max=1000, step=10).classes("w-full")
                        st_corpus_dir     = ui.input("Corpus dir",               value="corpus").classes("w-full")
                        st_index_name     = ui.input("Index name",               value="bench").classes("w-full")
                        st_embed_dim      = ui.select(_dim_opts, value=_dim_default, label="Embed dim").classes("w-full")

                    # ── Plan change section ────────────────────────────────────
                    ui.separator().classes("mt-3")
                    with ui.row().classes("items-center gap-3 mt-2"):
                        ui.icon("swap_vert", color="orange")
                        ui.label("Plan change (optional)").classes("font-semibold text-orange-700")

                    ui.label(
                        "Select a target plan and the test will trigger the change via the Aiven API "
                        "at the specified delay. Leave blank to run stress-only."
                    ).classes("text-xs text-gray-500 mb-2")

                    # Plan dropdown + fetch button
                    _conn["available_plans"] = []
                    plan_status = ui.label("").classes("text-xs text-gray-400")

                    with ui.row().classes("w-full gap-3 items-end flex-wrap"):
                        st_plan_sel = ui.select(
                            ["(no plan change)"],
                            value="(no plan change)",
                            label="Target plan",
                            with_input=True,
                        ).classes("flex-1 min-w-48")
                        st_plan_after = ui.number(
                            "Change after (s)", value=60, min=10, max=3600, step=10,
                        ).classes("w-40")
                        st_post_settle = ui.number(
                            "Run after settled (s)", value=60, min=10, max=600, step=10,
                        ).classes("w-44").tooltip(
                            "Continue the test this many seconds after the plan change settles to RUNNING"
                        )

                        def _fetch_plans() -> None:
                            token = _conn.get("token", "")
                            if not token:
                                plan_status.set_text("Not logged in — connect in Services tab first.")
                                return
                            svcs = _conn["selected_services"]
                            sel_label = st_svc_sel.value
                            svc = next((s for s in svcs if s["label"] == sel_label), None)
                            if not svc:
                                plan_status.set_text("No service selected.")
                                return
                            plan_status.set_text("Fetching plans…")
                            sess = _load_session()
                            project = sess.get("selected_project", "")
                            try:
                                from aiven_semantic_search_bench.aiven_client import AivenDiscovery
                                disc = AivenDiscovery(api_token=token)
                                plans = disc.list_available_plans(project, svc["name"])
                                if plans:
                                    options = ["(no plan change)"] + plans
                                    st_plan_sel.options = options
                                    st_plan_sel.value = svc.get("plan", plans[0]) if svc.get("plan") in plans else plans[0]
                                    plan_status.set_text(f"{len(plans)} plans available. Current: {svc.get('plan', '?')}")
                                    plan_status.classes("text-green-600", remove="text-gray-400 text-red-600")
                                else:
                                    plan_status.set_text("No plans returned. Enter a plan name manually.")
                                    plan_status.classes("text-orange-500", remove="text-gray-400")
                            except Exception as exc:
                                plan_status.set_text(f"Error: {exc}")
                                plan_status.classes("text-red-600", remove="text-gray-400")

                        ui.button("Fetch plans", on_click=_fetch_plans, icon="refresh").props("outline size=sm")

                    plan_status.classes("mt-1")

                    # ── Thanos metrics ─────────────────────────────────────────
                    ui.separator().classes("mt-3")
                    with ui.row().classes("items-center gap-3 mt-2"):
                        ui.icon("monitoring", color="purple")
                        ui.label("Thanos metrics (optional)").classes("font-semibold text-purple-700")

                    ui.label(
                        "Detects or creates an Aiven for Metrics (Thanos) service so JVM heap, GC, "
                        "indexing rate and search rate are recorded during the test. "
                        "Existing integrations in the same project are reused automatically."
                    ).classes("text-xs text-gray-500 mb-2")

                    _conn["thanos_uri"] = ""
                    _conn["thanos_svc_name"] = ""

                    # Restore from session
                    _sess_thanos = _load_session()
                    if _sess_thanos.get("thanos_uri"):
                        _conn["thanos_uri"] = _sess_thanos["thanos_uri"]
                        _conn["thanos_svc_name"] = _sess_thanos.get("thanos_svc_name", "")

                    thanos_status = ui.label(
                        f"Thanos configured: {_conn['thanos_uri'][:50]}…"
                        if _conn["thanos_uri"]
                        else "Not configured — click Detect to scan for existing integrations."
                    ).classes(
                        "text-xs text-green-600" if _conn["thanos_uri"] else "text-xs text-gray-400"
                    )

                    # Integration detail cards (populated on detect)
                    thanos_cards = ui.column().classes("w-full gap-1 mt-1")

                    with ui.row().classes("gap-3 items-center flex-wrap mt-1"):
                        st_thanos_name = ui.input(
                            "New Thanos service name", value="bench-metrics",
                        ).classes("w-48").tooltip("Used only when creating a new Thanos service")
                        st_thanos_plan = ui.select(
                            ["startup-4", "startup-8", "business-4"],
                            value="startup-4",
                            label="Plan",
                        ).classes("w-32")

                        def _detect_thanos() -> None:
                            token = _conn.get("token", "")
                            if not token:
                                thanos_status.set_text("Not logged in.")
                                return
                            svcs = _conn["selected_services"]
                            sel_label = st_svc_sel.value
                            svc = next((s for s in svcs if s["label"] == sel_label), None)
                            if not svc:
                                thanos_status.set_text("No service selected.")
                                return
                            sess = _load_session()
                            project = sess.get("selected_project", "")
                            thanos_status.set_text("Scanning integrations…")
                            thanos_status.classes("text-blue-600", remove="text-gray-400 text-green-600 text-red-600")
                            thanos_cards.clear()
                            try:
                                from aiven_semantic_search_bench.aiven_client import detect_thanos_integrations
                                integrations = detect_thanos_integrations(
                                    api_token=token,
                                    project=project,
                                    opensearch_service_name=svc["name"],
                                )
                                if not integrations:
                                    thanos_status.set_text(
                                        "No metrics integrations found — use 'Create + Connect' to set one up."
                                    )
                                    thanos_status.classes("text-amber-600", remove="text-blue-600")
                                    return

                                chosen_uri = ""
                                chosen_svc = ""
                                with thanos_cards:
                                    for integ in integrations:
                                        state     = integ["thanos_state"].upper()
                                        state_col = "green" if state == "RUNNING" else ("red" if state == "POWEROFF" else "orange")
                                        proj_tag  = "(this project)" if integ["same_project"] else f"(project: {integ['thanos_project']})"
                                        active_tag = "active" if integ["active"] else "pending"
                                        uri_short  = integ["query_uri"][:55] + "…" if integ["query_uri"] else "—"

                                        with ui.card().classes("w-full p-2 bg-gray-50"):
                                            with ui.row().classes("items-center gap-2 w-full justify-between"):
                                                with ui.row().classes("items-center gap-2"):
                                                    ui.badge(state, color=state_col).props("rounded")
                                                    ui.label(
                                                        f"{integ['thanos_service']} {proj_tag}  [{active_tag}]"
                                                    ).classes("font-mono text-xs")
                                                if integ["query_uri"]:
                                                    def _use_this(u=integ["query_uri"], n=integ["thanos_service"], p=integ["thanos_project"]) -> None:
                                                        _conn["thanos_uri"] = u
                                                        _conn["thanos_svc_name"] = n
                                                        curr = _load_session()
                                                        curr["thanos_uri"] = u
                                                        curr["thanos_svc_name"] = n
                                                        _save_session(curr)
                                                        thanos_status.set_text(
                                                            f"Using {n} ({p}) — {u[:50]}…"
                                                        )
                                                        thanos_status.classes("text-green-600", remove="text-blue-600 text-amber-600")
                                                        ui.notify(f"Thanos set to {n}", type="positive")
                                                    ui.button("Use this", on_click=_use_this).props("flat size=xs color=purple")
                                            ui.label(uri_short).classes("text-xs text-gray-500 font-mono break-all")

                                # Auto-select the first RUNNING same-project one
                                best = next(
                                    (i for i in integrations
                                     if i["same_project"] and i["thanos_state"] == "RUNNING" and i["query_uri"]),
                                    None
                                )
                                if best:
                                    chosen_uri = best["query_uri"]
                                    chosen_svc = best["thanos_service"]
                                    _conn["thanos_uri"] = chosen_uri
                                    _conn["thanos_svc_name"] = chosen_svc
                                    curr = _load_session()
                                    curr["thanos_uri"] = chosen_uri
                                    curr["thanos_svc_name"] = chosen_svc
                                    _save_session(curr)
                                    thanos_status.set_text(
                                        f"Auto-selected {chosen_svc} (same project, RUNNING) — {chosen_uri[:50]}…"
                                    )
                                    thanos_status.classes("text-green-600", remove="text-blue-600")
                                    ui.notify(f"Thanos auto-selected: {chosen_svc}", type="positive")
                                else:
                                    thanos_status.set_text(
                                        f"Found {len(integrations)} integration(s) — none RUNNING in this project. "
                                        "Use 'Create + Connect' or click 'Use this' on one above."
                                    )
                                    thanos_status.classes("text-amber-600", remove="text-blue-600")
                            except Exception as exc:
                                thanos_status.set_text(f"Detection error: {exc}")
                                thanos_status.classes("text-red-600", remove="text-blue-600")

                        def _setup_thanos() -> None:
                            token = _conn.get("token", "")
                            if not token:
                                thanos_status.set_text("Not logged in.")
                                return
                            svcs = _conn["selected_services"]
                            sel_label = st_svc_sel.value
                            svc = next((s for s in svcs if s["label"] == sel_label), None)
                            if not svc:
                                thanos_status.set_text("No service selected.")
                                return
                            sess = _load_session()
                            project = sess.get("selected_project", "")
                            thanos_status.set_text("Creating Thanos + waiting for RUNNING… (may take 2–3 min)")
                            thanos_status.classes("text-blue-600", remove="text-gray-400 text-green-600 text-red-600 text-amber-600")
                            try:
                                from aiven_semantic_search_bench.aiven_client import setup_thanos_for_opensearch
                                uri, svc_name = setup_thanos_for_opensearch(
                                    api_token=token,
                                    project=project,
                                    opensearch_service_name=svc["name"],
                                    thanos_service_name=st_thanos_name.value or "bench-metrics",
                                    thanos_plan=st_thanos_plan.value,
                                )
                                _conn["thanos_uri"] = uri
                                _conn["thanos_svc_name"] = svc_name
                                curr = _load_session()
                                curr["thanos_uri"] = uri
                                curr["thanos_svc_name"] = svc_name
                                _save_session(curr)
                                thanos_status.set_text(
                                    f"Thanos ready ({svc_name}) — metrics will appear ~2 min after test starts."
                                )
                                thanos_status.classes("text-green-600", remove="text-blue-600 text-gray-400")
                                ui.notify("Thanos configured and connected.", type="positive")
                                # Refresh the detect cards
                                _detect_thanos()
                            except Exception as exc:
                                thanos_status.set_text(f"Error: {exc}")
                                thanos_status.classes("text-red-600", remove="text-blue-600 text-gray-400")

                        ui.button(
                            "Detect", on_click=_detect_thanos,
                            icon="search",
                        ).props("outline size=sm color=purple")
                        ui.button(
                            "Create + Connect", on_click=_setup_thanos,
                            icon="add_circle",
                        ).props("outline size=sm color=purple")

                    # ── Also queue an index job first? ─────────────────────────
                    ui.separator().classes("mt-3")
                    st_do_index_first = ui.checkbox(
                        "Queue an index job first (populates the index before stressing)",
                        value=False,
                    )

                    def _submit_stress() -> None:
                        if _no_services_warning():
                            return
                        if not _conn.get("current_experiment"):
                            ui.notify("Select or create an experiment first.", type="warning")
                            return
                        svcs = _conn["selected_services"]
                        sel_label = st_svc_sel.value
                        svc = next((s for s in svcs if s["label"] == sel_label), None)
                        if not svc:
                            ui.notify("Could not find selected service.", type="warning")
                            return

                        embed_dim  = int(st_embed_dim.value)
                        index_name = st_index_name.value or "bench"
                        corpus_dir = st_corpus_dir.value or "corpus"
                        out_dir    = _current_out_dir()
                        spec = KnnSpec(
                            embed_dim=embed_dim,
                            engine="faiss", method="hnsw",
                            space_type="cosinesimil",
                            mode="in_memory", compression="none",
                            data_type="float",
                            m=16, ef_construction=128, ef_search=256,
                        )

                        plan_target = st_plan_sel.value
                        if plan_target == "(no plan change)":
                            plan_target = ""

                        sess = _load_session()
                        aiven_token    = _conn.get("token", "")
                        aiven_project  = sess.get("selected_project", "")
                        aiven_svc_name = svc.get("name", "")

                        q = JobQueue(RESULTS_DIR)
                        n = 0
                        index_job_id = ""

                        if st_do_index_first.value:
                            idx_job = BenchmarkJob(
                                bench_type="index",
                                service_label=svc["label"],
                                opensearch_uri=svc["uri"],
                                opensearch_version=svc["version"],
                                opensearch_index=index_name,
                                spec=spec,
                                embed_dim=embed_dim,
                                doc_count=int(qb_doc_count.value),
                                query_count=int(qb_query_count.value),
                                corpus_dir=corpus_dir,
                                out_dir=out_dir,
                                batch_sizes=[50],
                            )
                            q.enqueue(idx_job)
                            index_job_id = idx_job.job_id
                            n += 1

                        stress_job = BenchmarkJob(
                            bench_type="stress",
                            service_label=svc["label"],
                            opensearch_uri=svc["uri"],
                            opensearch_version=svc["version"],
                            opensearch_index=index_name,
                            spec=spec,
                            embed_dim=embed_dim,
                            doc_count=int(qb_doc_count.value),
                            query_count=int(qb_query_count.value),
                            corpus_dir=corpus_dir,
                            out_dir=out_dir,
                            stress_index_clients=int(st_index_clients.value),
                            stress_search_clients=int(st_search_clients.value),
                            stress_duration=int(st_duration.value),
                            stress_batch_size=int(st_batch_size.value),
                            stress_k=int(st_k.value),
                            plan_change_target=plan_target,
                            plan_change_after_s=int(st_plan_after.value),
                            post_settle_s=int(st_post_settle.value),
                            aiven_api_token=aiven_token,
                            aiven_project=aiven_project,
                            aiven_service_name=aiven_svc_name,
                            thanos_uri=_conn.get("thanos_uri", ""),
                            depends_on=index_job_id,
                        )
                        q.enqueue(stress_job)
                        n += 1

                        plan_note = f" → plan change to '{plan_target}' after {int(st_plan_after.value)}s (+{int(st_post_settle.value)}s post-settle)" if plan_target else ""
                        thanos_note = " + Thanos metrics" if _conn.get("thanos_uri") else ""
                        ui.notify(
                            f"Queued {n} job(s) for {sel_label}{plan_note}{thanos_note}. "
                            f"Switch to Queue tab to watch.",
                            type="positive",
                        )

                    ui.button(
                        "Run stress test", on_click=_submit_stress,
                        color="red", icon="whatshot",
                    ).classes("mt-3")

                # ══════════════════════════════════════════════════════════════
                # ADVANCED — full matrix form (collapsed by default)
                # ══════════════════════════════════════════════════════════════
                with ui.expansion("Advanced matrix (expert mode)", icon="tune").classes("w-full mt-2"):
                    ui.label(
                        "Build a full benchmark matrix across multiple k-NN configurations, "
                        "query types, and services."
                    ).classes("text-sm text-gray-400 mb-3")

                    with ui.card().classes("w-full"):
                        ui.label("Matrix configuration").classes("font-semibold text-base")

                        _render_experiment_selector()
                        ui.label("Corpus").classes("text-xs font-semibold uppercase text-gray-400 mt-3")
                        with ui.grid(columns=3).classes("w-full gap-3"):
                            nt_corpus_dir  = ui.input("Corpus dir", value="corpus").classes("w-full")
                            nt_doc_count   = ui.number("Doc count",   value=10_000, min=100, max=10_000_000, step=1000).classes("w-full")
                            nt_query_count = ui.number("Query count", value=500,    min=10,  max=100_000,    step=100).classes("w-full")
                            nt_embed_dim   = ui.select(_dim_opts, value=_dim_default, label="Embed dim").classes("w-full")
                            nt_index_name  = ui.input("Index name", value="bench").classes("w-full")

                        ui.label("k-NN axes").classes("text-xs font-semibold uppercase text-gray-400 mt-4")
                        with ui.grid(columns=3).classes("w-full gap-3"):
                            nt_engines      = ui.select(["faiss", "lucene"],                        multiple=True, value=["faiss", "lucene"], label="Engines").classes("w-full")
                            nt_methods      = ui.select(["hnsw", "ivf"],                             multiple=True, value=["hnsw"],            label="Methods").classes("w-full")
                            nt_modes        = ui.select(["in_memory", "on_disk"],                   multiple=True, value=["in_memory"],        label="Modes").classes("w-full")
                            nt_compressions = ui.select(["none","1x","2x","4x","8x","16x","32x"],  multiple=True, value=["none"],             label="Compression (on_disk only)").classes("w-full")
                            nt_data_types   = ui.select(["float","byte","fp16","binary"],           multiple=True, value=["float"],            label="Data types").classes("w-full")

                        ui.label("Flags").classes("text-xs font-semibold uppercase text-gray-400 mt-3")
                        with ui.row().classes("gap-6"):
                            nt_with_text      = ui.checkbox("Text field (for hybrid)",      value=False)
                            nt_with_metadata  = ui.checkbox("Metadata (for filter tests)",  value=False)
                            nt_derived_source = ui.checkbox("derived_source (2.19+)",       value=False)

                        ui.label("HNSW parameters").classes("text-xs font-semibold uppercase text-gray-400 mt-4")
                        with ui.grid(columns=3).classes("w-full gap-3"):
                            nt_m               = ui.number("m",              value=16,  min=4,  max=128,  step=4).classes("w-full")
                            nt_ef_construction = ui.number("ef_construction", value=128, min=32, max=2048, step=32).classes("w-full")
                            nt_ef_search       = ui.input("ef_search (comma-separated)", value="256").classes("w-full")

                        ui.label("Query types").classes("text-xs font-semibold uppercase text-gray-400 mt-4")
                        with ui.row().classes("gap-6"):
                            nt_do_index  = ui.checkbox("index",                           value=True)
                            nt_do_search = ui.checkbox("search",                          value=True)
                            nt_do_recall = ui.checkbox("recall  (needs qrels.npy)",       value=False)
                            nt_do_hybrid = ui.checkbox("hybrid  (needs text + metadata)", value=False)
                            nt_do_stress = ui.checkbox("stress  (chaos / saturation)",    value=False)
                        nt_filter_sel = ui.select(["none", "low", "high"], value="none", label="Filter selectivity (hybrid)").classes("w-48 mt-2")

                        ui.label("Benchmark parameters").classes("text-xs font-semibold uppercase text-gray-400 mt-4")
                        with ui.grid(columns=3).classes("w-full gap-3"):
                            nt_batch_sizes = ui.input("Batch sizes (comma-separated)", value="1,5,10,20,50").classes("w-full")
                            nt_k           = ui.number("k (nearest neighbours)", value=10, min=1, max=500).classes("w-full")
                            nt_rounds      = ui.number("Rounds",                 value=3,  min=1, max=20).classes("w-full")

                    with ui.expansion("Advanced search settings (OSB-inspired)", icon="science").classes("w-full mt-2"):
                        ui.markdown(
                            "These settings are modelled on [OpenSearch Benchmark](https://opensearch.org/docs/latest/benchmark/workloads/vectorsearch/) "
                            "(`vectorsearch` workload). They only affect **search** jobs."
                        ).classes("text-sm text-gray-500 mb-2")
                        with ui.grid(columns=3).classes("w-full gap-3"):
                            nt_warmup_queries = ui.number(
                                "Warmup queries/round (0 = skip)",
                                value=50, min=0, max=500, step=10,
                            ).classes("w-full")
                            nt_search_clients = ui.number(
                                "Search clients (parallel workers)",
                                value=1, min=1, max=32, step=1,
                            ).classes("w-full")
                            nt_force_merge    = ui.number(
                                "Force-merge segments (0 = skip)",
                                value=0, min=0, max=32, step=1,
                            ).classes("w-full")
                        ui.label("Sustained-throughput mode (both > 0 to activate):").classes("text-xs text-gray-500 mt-2")
                        with ui.grid(columns=2).classes("w-full gap-3"):
                            nt_target_throughput = ui.number(
                                "Target throughput (ops/s, 0 = rounds mode)",
                                value=0.0, min=0.0, step=1.0,
                            ).classes("w-full")
                            nt_time_period = ui.number(
                                "Time period (seconds, 0 = rounds mode)",
                                value=0, min=0, max=3600, step=30,
                            ).classes("w-full")

                    with ui.expansion("Stress test settings", icon="whatshot").classes("w-full mt-2"):
                        with ui.grid(columns=3).classes("w-full gap-3"):
                            nt_stress_index_clients = ui.number(
                                "Index clients", value=8, min=1, max=64, step=1,
                            ).classes("w-full")
                            nt_stress_search_clients = ui.number(
                                "Search clients", value=16, min=1, max=64, step=1,
                            ).classes("w-full")
                            nt_stress_duration = ui.number(
                                "Duration (s)", value=120, min=30, max=3600, step=30,
                            ).classes("w-full")
                            nt_stress_batch_size = ui.number(
                                "Batch size", value=100, min=1, max=2000, step=50,
                            ).classes("w-full")
                            nt_stress_k = ui.number(
                                "k for search", value=100, min=1, max=1000, step=10,
                            ).classes("w-full")

                    # Helper: read all form widgets → dict
                    def _read_form() -> dict:
                        return {
                            "corpus": {
                                "doc_count":        int(nt_doc_count.value),
                                "query_count":      int(nt_query_count.value),
                                "embed_dim":        int(nt_embed_dim.value),
                                "corpus_dir":       nt_corpus_dir.value,
                                "opensearch_index": nt_index_name.value,
                            },
                            "knn_axes": {
                                "engines":         list(nt_engines.value or []),
                                "methods":         list(nt_methods.value or []),
                                "modes":           list(nt_modes.value or []),
                                "compressions":    list(nt_compressions.value or []),
                                "data_types":      list(nt_data_types.value or []),
                                "m":               int(nt_m.value),
                                "ef_construction": int(nt_ef_construction.value),
                                "ef_search":       nt_ef_search.value,
                                "with_text":       nt_with_text.value,
                                "with_metadata":   nt_with_metadata.value,
                                "derived_source":  nt_derived_source.value,
                            },
                            "query_types": {
                                "index":              nt_do_index.value,
                                "search":             nt_do_search.value,
                                "recall":             nt_do_recall.value,
                                "hybrid":             nt_do_hybrid.value,
                                "stress":             nt_do_stress.value,
                                "filter_selectivity": nt_filter_sel.value,
                            },
                            "bench_params": {
                                "batch_sizes": nt_batch_sizes.value,
                                "k":           int(nt_k.value),
                                "rounds":      int(nt_rounds.value),
                            },
                            "search_params": {
                                "warmup_queries":       int(nt_warmup_queries.value),
                                "search_clients":       int(nt_search_clients.value),
                                "target_throughput":    float(nt_target_throughput.value),
                                "time_period":          int(nt_time_period.value),
                                "force_merge_segments": int(nt_force_merge.value),
                            },
                            "stress_params": {
                                "stress_index_clients":  int(nt_stress_index_clients.value),
                                "stress_search_clients": int(nt_stress_search_clients.value),
                                "stress_duration":       int(nt_stress_duration.value),
                                "stress_batch_size":     int(nt_stress_batch_size.value),
                                "stress_k":              int(nt_stress_k.value),
                            },
                        }

                    def _apply_matrix(m: dict) -> None:
                        c = m.get("corpus", {})
                        nt_doc_count.value       = int(c.get("doc_count", 10_000))
                        nt_query_count.value     = int(c.get("query_count", 500))
                        nt_embed_dim.value       = int(c.get("embed_dim", _dim_default))
                        nt_corpus_dir.value      = c.get("corpus_dir", "corpus")
                        nt_index_name.value      = c.get("opensearch_index", "bench")
                        k = m.get("knn_axes", {})
                        nt_engines.value         = list(k.get("engines", ["faiss"]))
                        nt_methods.value         = list(k.get("methods", ["hnsw"]))
                        nt_modes.value           = list(k.get("modes", ["in_memory"]))
                        nt_compressions.value    = list(k.get("compressions", ["none"]))
                        nt_data_types.value      = list(k.get("data_types", ["float"]))
                        nt_m.value               = int(k.get("m", 16))
                        nt_ef_construction.value = int(k.get("ef_construction", 128))
                        nt_ef_search.value       = str(k.get("ef_search", "256"))
                        nt_with_text.value       = bool(k.get("with_text", False))
                        nt_with_metadata.value   = bool(k.get("with_metadata", False))
                        nt_derived_source.value  = bool(k.get("derived_source", False))
                        q = m.get("query_types", {})
                        nt_do_index.value        = bool(q.get("index", True))
                        nt_do_search.value       = bool(q.get("search", True))
                        nt_do_recall.value       = bool(q.get("recall", False))
                        nt_do_hybrid.value       = bool(q.get("hybrid", False))
                        nt_do_stress.value       = bool(q.get("stress", False))
                        nt_filter_sel.value      = q.get("filter_selectivity", "none")
                        p = m.get("bench_params", {})
                        nt_batch_sizes.value     = str(p.get("batch_sizes", "1,5,10,20,50"))
                        nt_k.value               = int(p.get("k", 10))
                        nt_rounds.value          = int(p.get("rounds", 3))
                        sp = m.get("search_params", {})
                        nt_warmup_queries.value      = int(sp.get("warmup_queries", 50))
                        nt_search_clients.value      = int(sp.get("search_clients", 1))
                        nt_target_throughput.value   = float(sp.get("target_throughput", 0.0))
                        nt_time_period.value         = int(sp.get("time_period", 0))
                        nt_force_merge.value         = int(sp.get("force_merge_segments", 0))
                        stp = m.get("stress_params", {})
                        nt_stress_index_clients.value  = int(stp.get("stress_index_clients", 8))
                        nt_stress_search_clients.value = int(stp.get("stress_search_clients", 16))
                        nt_stress_duration.value       = int(stp.get("stress_duration", 120))
                        nt_stress_batch_size.value     = int(stp.get("stress_batch_size", 100))
                        nt_stress_k.value              = int(stp.get("stress_k", 100))

                    def _compute_cells() -> tuple[list[dict], list[str]]:
                        bench_types = (
                            (["index"]  if nt_do_index.value  else [])
                            + (["search"] if nt_do_search.value else [])
                            + (["recall"] if nt_do_recall.value else [])
                            + (["hybrid"] if nt_do_hybrid.value else [])
                            + (["stress"] if nt_do_stress.value else [])
                        )
                        ef_search_vals = _parse_ints(nt_ef_search.value, [256])
                        cells = []
                        for svc in _conn["selected_services"]:
                            for eng in (nt_engines.value or []):
                                for meth in (nt_methods.value or []):
                                    for mode in (nt_modes.value or []):
                                        for comp in (nt_compressions.value or []):
                                            for dt in (nt_data_types.value or []):
                                                for ef_s in ef_search_vals:
                                                    reason = _skip_reason(eng, meth, mode, comp, dt)
                                                    cells.append({
                                                        "service": svc["label"],
                                                        "engine": eng, "method": meth,
                                                        "mode": mode, "compression": comp,
                                                        "data_type": dt, "ef_search": ef_s,
                                                        "skip_reason": reason or "",
                                                        "svc": svc,
                                                    })
                        return cells, bench_types

                    # ── Sample matrices ────────────────────────────────────────
                    with ui.expansion("Sample matrices", icon="view_list").classes("w-full"):
                        sample_desc = ui.label("").classes("text-sm text-gray-500 mb-2")
                        sample_sel = ui.select(
                            list(SAMPLE_MATRICES.keys()),
                            value=list(SAMPLE_MATRICES.keys())[0],
                            label="Choose sample",
                        ).classes("w-full mb-2")

                        def _update_sample_desc(_e: Any = None) -> None:
                            sample_desc.set_text(SAMPLE_MATRICES.get(sample_sel.value, {}).get("_description", ""))

                        sample_sel.on_value_change(_update_sample_desc)
                        _update_sample_desc()

                        def _load_sample() -> None:
                            key = sample_sel.value
                            if key in SAMPLE_MATRICES:
                                _apply_matrix(SAMPLE_MATRICES[key])
                                ui.notify(f"Loaded sample: {key}", type="positive")

                        ui.button("Load sample", on_click=_load_sample, icon="download")

                    # ── Save / load matrices ───────────────────────────────────
                    with ui.expansion("Save / load matrices", icon="folder").classes("w-full"):

                        @ui.refreshable
                        def saved_matrices_section() -> None:
                            saved = sorted(p.stem for p in MATRICES_DIR.glob("*.json"))
                            with ui.row().classes("w-full gap-4"):
                                with ui.card().classes("flex-1"):
                                    ui.label("Save current configuration").classes("font-semibold text-sm")
                                    save_name = ui.input("Matrix name", value="my_matrix").classes("w-full")

                                    def _save_matrix() -> None:
                                        slug = _slugify(save_name.value)
                                        if not slug:
                                            ui.notify("Name cannot be empty.", type="warning")
                                            return
                                        m = _read_form()
                                        m["_name"] = save_name.value
                                        m["_saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                                        (MATRICES_DIR / f"{slug}.json").write_text(json.dumps(m, indent=2))
                                        ui.notify(f"Saved as {slug}.json", type="positive")
                                        saved_matrices_section.refresh()

                                    ui.button("Save", on_click=_save_matrix, icon="save")

                                with ui.card().classes("flex-1"):
                                    ui.label("Load saved configuration").classes("font-semibold text-sm")
                                    if not saved:
                                        ui.label("No saved matrices yet.").classes("text-gray-400 text-sm")
                                    else:
                                        load_sel = ui.select(saved, value=saved[0], label="Saved matrix").classes("w-full")
                                        with ui.row().classes("gap-2 mt-1"):
                                            def _load_saved(name: str = saved[0]) -> None:
                                                sel_name = load_sel.value
                                                path = MATRICES_DIR / f"{sel_name}.json"
                                                try:
                                                    _apply_matrix(json.loads(path.read_text()))
                                                    ui.notify(f"Loaded {sel_name}", type="positive")
                                                except Exception as exc:
                                                    ui.notify(f"Error: {exc}", type="negative")

                                            def _delete_saved() -> None:
                                                sel_name = load_sel.value
                                                (MATRICES_DIR / f"{sel_name}.json").unlink(missing_ok=True)
                                                ui.notify(f"Deleted {sel_name}", type="info")
                                                saved_matrices_section.refresh()

                                            ui.button("Load",   on_click=_load_saved)
                                            ui.button("Delete", on_click=_delete_saved, color="negative").props("flat")

                        saved_matrices_section()

                    # ── Preview area ───────────────────────────────────────────
                    preview_col = ui.column().classes("w-full")

                    def _preview() -> None:
                        preview_col.clear()
                        cells, bench_types = _compute_cells()
                        if not cells:
                            with preview_col:
                                ui.label("No cells — select at least one value in each axis.").classes("text-orange-600")
                            return
                        preview_rows = []
                        for c in cells:
                            for bt in bench_types:
                                preview_rows.append({
                                    "Service": c["service"], "Engine": c["engine"],
                                    "Method": c["method"],   "Mode": c["mode"],
                                    "Compression": c["compression"], "DataType": c["data_type"],
                                    "ef_search": c["ef_search"], "BenchType": bt,
                                    "Status": ("SKIP: " + c["skip_reason"]) if c["skip_reason"] else "queued",
                                })
                        valid = [r for r in preview_rows if r["Status"] == "queued"]
                        with preview_col:
                            with ui.row().classes("gap-3 mb-2"):
                                ui.badge(f"{len(valid)} valid", color="positive")
                                ui.badge(f"{len(preview_rows) - len(valid)} skipped (invalid combos)", color="warning")
                            cols = [{"name": k, "label": k, "field": k, "align": "left"} for k in list(preview_rows[0].keys())]
                            ui.table(columns=cols, rows=preview_rows[:300], row_key="BenchType").classes("w-full")
                            if len(preview_rows) > 300:
                                ui.label(f"Showing first 300 of {len(preview_rows)}.").classes("text-xs text-gray-400")

                    def _submit() -> None:
                        svcs = _conn["selected_services"]
                        if not svcs:
                            ui.notify("No services selected. Go to the Services tab first.", type="warning")
                            return
                        if not _conn.get("current_experiment"):
                            ui.notify("Select or create an experiment first.", type="warning")
                            return
                        cells, bench_types = _compute_cells()
                        if not bench_types:
                            ui.notify("Select at least one query type (index / search / recall / hybrid).", type="warning")
                            return
                        valid_cells = [c for c in cells if not c["skip_reason"]]
                        if not valid_cells:
                            ui.notify("All matrix cells are invalid. Adjust your axis selections.", type="warning")
                            return

                        embed_dim             = int(nt_embed_dim.value)
                        corpus_dir            = nt_corpus_dir.value
                        out_dir               = _current_out_dir()
                        index_name            = nt_index_name.value
                        k_val                 = int(nt_k.value)
                        rounds_val            = int(nt_rounds.value)
                        batch_sizes           = _parse_ints(nt_batch_sizes.value, [10, 50])
                        filter_sel            = nt_filter_sel.value
                        m_val                 = int(nt_m.value)
                        ef_constr             = int(nt_ef_construction.value)
                        doc_count             = int(nt_doc_count.value)
                        query_count           = int(nt_query_count.value)
                        derived_src           = nt_derived_source.value
                        with_text             = nt_with_text.value
                        with_meta             = nt_with_metadata.value
                        warmup_queries_val       = int(nt_warmup_queries.value)
                        search_clients_val       = int(nt_search_clients.value)
                        target_throughput_val    = float(nt_target_throughput.value)
                        time_period_val          = int(nt_time_period.value)
                        force_merge_val          = int(nt_force_merge.value)
                        stress_index_clients_val = int(nt_stress_index_clients.value)
                        stress_search_clients_val = int(nt_stress_search_clients.value)
                        stress_duration_val      = int(nt_stress_duration.value)
                        stress_batch_size_val    = int(nt_stress_batch_size.value)
                        stress_k_val             = int(nt_stress_k.value)

                        q = JobQueue(RESULTS_DIR)
                        n_submitted = 0
                        for c in valid_cells:
                            svc = c["svc"]
                            spec = KnnSpec(
                                embed_dim=embed_dim,
                                engine=c["engine"], method=c["method"],
                                space_type="cosinesimil",
                                mode=c["mode"], compression=c["compression"],
                                data_type=c["data_type"],
                                derived_source=derived_src,
                                m=m_val, ef_construction=ef_constr, ef_search=c["ef_search"],
                                with_text=with_text or "hybrid" in bench_types,
                                with_metadata=with_meta or "hybrid" in bench_types,
                            )
                            index_job_id = ""
                            for bt in bench_types:
                                job = BenchmarkJob(
                                    bench_type=bt,
                                    service_label=svc["label"],
                                    opensearch_uri=svc["uri"],
                                    opensearch_version=svc["version"],
                                    opensearch_index=index_name,
                                    spec=spec,
                                    embed_dim=embed_dim,
                                    doc_count=doc_count,
                                    query_count=query_count,
                                    corpus_dir=corpus_dir,
                                    out_dir=out_dir,
                                    rounds=rounds_val,
                                    k=k_val,
                                    batch_sizes=batch_sizes,
                                    filter_selectivity=filter_sel,
                                    warmup_queries=warmup_queries_val,
                                    search_clients=search_clients_val,
                                    target_throughput=target_throughput_val,
                                    time_period=time_period_val,
                                    force_merge_segments=force_merge_val,
                                    stress_index_clients=stress_index_clients_val,
                                    stress_search_clients=stress_search_clients_val,
                                    stress_duration=stress_duration_val,
                                    stress_batch_size=stress_batch_size_val,
                                    stress_k=stress_k_val,
                                    depends_on=index_job_id if bt != "index" else "",
                                )
                                q.enqueue(job)
                                if bt == "index":
                                    index_job_id = job.job_id
                                n_submitted += 1

                        ui.notify(
                            f"Queued {n_submitted} job(s). Switch to the Queue tab to watch progress.",
                            type="positive",
                        )

                    with ui.row().classes("gap-2 mt-2"):
                        ui.button("Preview matrix", on_click=_preview, icon="preview").props("outline")
                        ui.button("Submit to queue", on_click=_submit, color="blue", icon="send")

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 — QUEUE (auto-refreshes every 3 s, never disturbs other tabs)
        # ══════════════════════════════════════════════════════════════════════
        with ui.tab_panel(tab_queue):
            ui.label("Job Queue").classes("text-lg font-semibold mt-2")

            @ui.refreshable
            def queue_view() -> None:
                q = JobQueue(RESULTS_DIR)
                jobs = q.all_jobs()

                counts: dict[str, int] = {"pending": 0, "running": 0, "ok": 0, "failed": 0, "skipped": 0}
                for j in jobs:
                    counts[j.state] = counts.get(j.state, 0) + 1

                with ui.row().classes("gap-4 mb-4 flex-wrap"):
                    for state, label, color in [
                        ("pending", "Pending",   "orange-600"),
                        ("running", "Running",   "blue-600"),
                        ("ok",      "Completed", "green-600"),
                        ("failed",  "Failed",    "red-600"),
                        ("skipped", "Skipped",   "gray-500"),
                    ]:
                        with ui.card().classes("px-6 py-3 text-center min-w-24"):
                            ui.label(str(counts[state])).classes(f"text-3xl font-bold text-{color}")
                            ui.label(label).classes("text-xs text-gray-500")

                active = counts["pending"] + counts["running"]
                if active:
                    ui.badge(f"{active} active — refreshing every 3 s", color="blue")

                if not jobs:
                    ui.label("No jobs yet — submit a benchmark matrix from the New Test tab.").classes("text-gray-500 mt-2")
                    return

                cols = [
                    {"name": "status",    "label": "",          "field": "status",    "align": "center"},
                    {"name": "job",       "label": "Job",       "field": "job",       "align": "left", "sortable": True},
                    {"name": "type",      "label": "Type",      "field": "type",      "align": "left"},
                    {"name": "submitted", "label": "Submitted", "field": "submitted", "align": "left"},
                    {"name": "started",   "label": "Started",   "field": "started",   "align": "left"},
                    {"name": "finished",  "label": "Finished",  "field": "finished",  "align": "left"},
                ]
                rows = [
                    {
                        "id":        j.job_id,
                        "status":    _STATUS_ICONS.get(j.state, j.state),
                        "job":       j.display_label(),
                        "type":      j.bench_type,
                        "submitted": j.submitted_at or "",
                        "started":   j.started_at   or "",
                        "finished":  j.finished_at  or "",
                        "state":     j.state,
                        "log_path":  j.log_path or "",
                    }
                    for j in jobs
                ]
                ui.table(columns=cols, rows=rows, row_key="id").classes("w-full")

                # ── Log viewer ────────────────────────────────────────────────
                ui.separator()
                ui.label("Log viewer").classes("font-semibold")
                done_jobs = [j for j in jobs if j.state in ("running", "ok", "failed", "skipped")]

                if not done_jobs:
                    ui.label("Logs appear here once a job starts.").classes("text-gray-400 text-sm")
                else:
                    log_options = {j.display_label(): j for j in reversed(done_jobs)}

                    # Restore selected job across refreshes
                    init_log_label = None
                    if _conn["log_job_id"]:
                        for lbl, j in log_options.items():
                            if j.job_id == _conn["log_job_id"]:
                                init_log_label = lbl
                                break
                    if not init_log_label:
                        init_log_label = next(iter(log_options))

                    log_sel = ui.select(list(log_options.keys()), value=init_log_label, label="Job").classes("w-full max-w-2xl")
                    log_container = ui.column().classes("w-full")

                    def _show_log(label: str) -> None:
                        job = log_options.get(label)
                        if not job:
                            return
                        _conn["log_job_id"] = job.job_id
                        log_container.clear()
                        with log_container:
                            if job.state == "skipped":
                                ui.label(f"Job skipped: {job.error_message or 'dependency failed'}").classes("text-orange-600")
                            else:
                                lf = Path(job.log_path) if job.log_path else None
                                if lf and lf.exists():
                                    text = lf.read_text(errors="replace")
                                    ui.html(
                                        f'<pre style="background:#1e1e1e;color:#d4d4d4;padding:12px;'
                                        f'border-radius:4px;overflow:auto;max-height:400px;'
                                        f'font-size:12px;font-family:monospace;">'
                                        f"{_html.escape(text)}</pre>"
                                    )
                                else:
                                    ui.label("Log not yet available.").classes("text-gray-400")

                    log_sel.on_value_change(lambda e: _show_log(e.value))
                    _show_log(init_log_label)

                # ── Controls ──────────────────────────────────────────────────
                ui.separator()

                def _clear_finished() -> None:
                    removed = JobQueue(RESULTS_DIR).clear_finished()
                    ui.notify(f"Removed {removed} finished job(s).", type="info")
                    queue_view.refresh()

                def _cancel_pending() -> None:
                    cancelled = JobQueue(RESULTS_DIR).cancel_pending()
                    ui.notify(
                        f"Cancelled {cancelled} pending job(s)." if cancelled
                        else "No pending jobs to cancel.",
                        type="warning" if cancelled else "info",
                    )
                    queue_view.refresh()

                with ui.row().classes("gap-2"):
                    ui.button(
                        "Clear completed / failed / skipped", on_click=_clear_finished, icon="delete_sweep",
                    ).props("flat color=grey")
                    ui.button(
                        "Cancel all pending", on_click=_cancel_pending, icon="cancel",
                    ).props("flat color=negative")

            queue_view()
            ui.timer(3.0, queue_view.refresh)

        # ══════════════════════════════════════════════════════════════════════
        # TAB 5 — RESULTS
        # ══════════════════════════════════════════════════════════════════════
        with ui.tab_panel(tab_results):

            @ui.refreshable
            def results_view() -> None:  # noqa: C901
                exps = _experiments.list_experiments(RESULTS_DIR)
                slug_by_name = {e["name"]: e["slug"] for e in exps}
                name_by_slug = {e["slug"]: e["name"] for e in exps}
                exp_names = [e["name"] for e in exps]

                current_slug = _conn.get("current_experiment", "")
                # Fall back to first experiment if stored slug is gone
                if current_slug and current_slug not in name_by_slug:
                    current_slug = ""
                if not current_slug and exps:
                    current_slug = exps[0]["slug"]
                    _conn["current_experiment"] = current_slug

                # ── Header: experiment selector + management ───────────────────
                with ui.row().classes("w-full items-end gap-3 flex-wrap mt-2"):
                    ui.label("Experiment").classes("text-lg font-semibold")

                    current_name = name_by_slug.get(current_slug, "")
                    res_exp_sel = ui.select(
                        exp_names or ["(no experiments)"],
                        value=current_name or (exp_names[0] if exp_names else "(no experiments)"),
                        label="",
                    ).classes("min-w-48")

                    def _on_res_exp_change(e: Any) -> None:
                        slug = slug_by_name.get(e.value, "")
                        _conn["current_experiment"] = slug
                        curr = _load_session()
                        curr["current_experiment"] = slug
                        _save_session(curr)
                        results_view.refresh()

                    res_exp_sel.on_value_change(_on_res_exp_change)

                    def _open_rename_dialog() -> None:
                        slug = _conn.get("current_experiment", "")
                        if not slug:
                            return
                        with ui.dialog() as dlg, ui.card().classes("min-w-80"):
                            ui.label("Rename experiment").classes("font-semibold text-lg")
                            old_meta = next((e for e in exps if e["slug"] == slug), {})
                            rn_name = ui.input("Name", value=old_meta.get("name", "")).classes("w-full")
                            rn_desc = ui.input("Description", value=old_meta.get("description", "")).classes("w-full")

                            def _do_rename() -> None:
                                n = rn_name.value.strip()
                                if not n:
                                    ui.notify("Name cannot be empty.", type="warning")
                                    return
                                _experiments.rename_experiment(RESULTS_DIR, slug, n, rn_desc.value.strip())
                                dlg.close()
                                results_view.refresh()
                                ui.notify(f"Renamed to '{n}'.", type="positive")

                            with ui.row().classes("gap-2 mt-3"):
                                ui.button("Save", on_click=_do_rename, color="blue")
                                ui.button("Cancel", on_click=dlg.close).props("flat")
                        dlg.open()

                    def _open_delete_exp_dialog() -> None:
                        slug = _conn.get("current_experiment", "")
                        if not slug:
                            return
                        exp_name = name_by_slug.get(slug, slug)
                        with ui.dialog() as dlg, ui.card().classes("min-w-80"):
                            ui.label("Delete experiment?").classes("font-semibold text-lg text-red-700")
                            ui.label(
                                f"This will permanently delete '{exp_name}' and all its reports. "
                                "This cannot be undone."
                            ).classes("text-sm text-gray-600")

                            def _do_delete() -> None:
                                _experiments.delete_experiment(RESULTS_DIR, slug)
                                _conn["current_experiment"] = ""
                                curr = _load_session()
                                curr["current_experiment"] = ""
                                _save_session(curr)
                                dlg.close()
                                results_view.refresh()
                                ui.notify(f"Deleted experiment '{exp_name}'.", type="info")

                            with ui.row().classes("gap-2 mt-3"):
                                ui.button("Delete", on_click=_do_delete, color="negative")
                                ui.button("Cancel", on_click=dlg.close).props("flat")
                        dlg.open()

                    def _create_exp_from_results() -> None:
                        with ui.dialog() as dlg, ui.card().classes("min-w-80"):
                            ui.label("New experiment").classes("font-semibold text-lg")
                            ne_name = ui.input("Name", placeholder="e.g. v2.17 baseline").classes("w-full")
                            ne_desc = ui.input("Description (optional)").classes("w-full")

                            def _do_create() -> None:
                                name = ne_name.value.strip()
                                if not name:
                                    ui.notify("Name cannot be empty.", type="warning")
                                    return
                                meta = _experiments.create_experiment(RESULTS_DIR, name, ne_desc.value.strip())
                                _conn["current_experiment"] = meta["slug"]
                                curr = _load_session()
                                curr["current_experiment"] = meta["slug"]
                                _save_session(curr)
                                dlg.close()
                                results_view.refresh()
                                ui.notify(f"Experiment '{name}' created.", type="positive")

                            with ui.row().classes("gap-2 mt-3"):
                                ui.button("Create", on_click=_do_create, color="blue")
                                ui.button("Cancel", on_click=dlg.close).props("flat")
                        dlg.open()

                    with ui.row().classes("gap-1"):
                        ui.button("New", icon="add", on_click=_create_exp_from_results).props("outline size=sm color=blue")
                        ui.button("Rename", icon="edit", on_click=_open_rename_dialog).props("outline size=sm color=grey")
                        ui.button("Delete", icon="delete", on_click=_open_delete_exp_dialog).props("outline size=sm color=negative")
                        ui.button("Refresh", icon="refresh", on_click=results_view.refresh).props("flat size=sm color=grey")

                if not current_slug:
                    with ui.card().classes("w-full mt-4 items-center text-center py-8"):
                        ui.icon("science", size="xl", color="grey")
                        ui.label("No experiments yet.").classes("text-lg text-gray-500 mt-2")
                        ui.label(
                            "Create an experiment from the New Test tab, then submit a benchmark."
                        ).classes("text-sm text-gray-400")
                    return

                # ── Summary chips ──────────────────────────────────────────────
                summ = _experiments.experiment_summary(RESULTS_DIR, current_slug)
                meta_info = next((e for e in exps if e["slug"] == current_slug), {})
                exp_dir_path = _experiments.experiment_dir(RESULTS_DIR, current_slug)

                with ui.row().classes("gap-3 flex-wrap mt-1 mb-2 items-center"):
                    ui.badge(f"{summ['run_count']} runs", color="blue")
                    if summ["engines"]:
                        ui.badge("engines: " + ", ".join(summ["engines"]), color="teal")
                    if summ["bench_types"]:
                        ui.badge("types: " + ", ".join(t.replace("bench-", "") for t in summ["bench_types"]), color="purple")
                    if summ["date_from"]:
                        date_str = summ["date_from"] if summ["date_from"] == summ["date_to"] else f"{summ['date_from']} – {summ['date_to']}"
                        ui.badge(date_str, color="grey")
                    if meta_info.get("description"):
                        ui.label(meta_info["description"]).classes("text-xs text-gray-500 italic")
                    ui.label(str(exp_dir_path)).classes("text-xs text-gray-400 font-mono break-all")

                if summ["run_count"] == 0:
                    with ui.card().classes("w-full mt-4 items-center text-center py-6"):
                        ui.icon("hourglass_empty", size="xl", color="grey")
                        ui.label("No benchmark reports in this experiment yet.").classes("text-gray-500 mt-2")
                        ui.label("Submit jobs from the New Test tab — they will appear here automatically.").classes("text-sm text-gray-400")
                    return

                # ── All reports for this experiment ────────────────────────────
                grouped = _load_reports_for_experiment(RESULTS_DIR, current_slug)
                all_reports = _experiments.list_reports(RESULTS_DIR, current_slug)

                # ── Filter row ─────────────────────────────────────────────────
                all_engines    = sorted({r.get("params", {}).get("knn_spec", {}).get("engine", "") for r in all_reports if r.get("params", {}).get("knn_spec", {}).get("engine")})
                all_modes      = sorted({r.get("params", {}).get("knn_spec", {}).get("mode", "") for r in all_reports if r.get("params", {}).get("knn_spec", {}).get("mode")})
                all_data_types = sorted({r.get("params", {}).get("knn_spec", {}).get("data_type", "") for r in all_reports if r.get("params", {}).get("knn_spec", {}).get("data_type")})
                all_labels     = sorted({r.get("params", {}).get("plan_label", "") for r in all_reports if r.get("params", {}).get("plan_label")})

                with ui.row().classes("w-full gap-3 flex-wrap items-end mb-1"):
                    ui.label("Filters:").classes("text-xs font-semibold text-gray-400 self-center")
                    flt_engine = ui.select(["(all)"] + all_engines, value="(all)", label="Engine").classes("min-w-28")
                    flt_mode   = ui.select(["(all)"] + all_modes,   value="(all)", label="Mode").classes("min-w-28")
                    flt_dtype  = ui.select(["(all)"] + all_data_types, value="(all)", label="Data type").classes("min-w-28")
                    flt_label  = ui.select(["(all)"] + all_labels,  value="(all)", label="Plan label").classes("min-w-48")

                def _apply_filters(reps: list[dict]) -> list[dict]:
                    out = []
                    for r in reps:
                        s = r.get("params", {}).get("knn_spec", {})
                        if flt_engine.value != "(all)" and s.get("engine") != flt_engine.value:
                            continue
                        if flt_mode.value != "(all)" and s.get("mode") != flt_mode.value:
                            continue
                        if flt_dtype.value != "(all)" and s.get("data_type") != flt_dtype.value:
                            continue
                        lbl = r.get("params", {}).get("plan_label", "")
                        if flt_label.value != "(all)" and lbl != flt_label.value:
                            continue
                        out.append(r)
                    return out

                def _plan_label(r: dict) -> str:
                    return r.get("params", {}).get("plan_label", "unlabeled")

                # ── Result + Manage sub-tabs ───────────────────────────────────
                with ui.tabs().classes("w-full") as result_tabs:
                    rt_idx     = ui.tab("bench-index")
                    rt_search  = ui.tab("bench-search")
                    rt_recall  = ui.tab("bench-recall")
                    rt_hybrid  = ui.tab("bench-hybrid")
                    rt_stress  = ui.tab("bench-stress")
                    rt_recover = ui.tab("bench-recover")
                    rt_manage  = ui.tab("Manage", icon="tune")

                with ui.tab_panels(result_tabs, value=rt_idx).classes("w-full"):

                    # ── bench-index ────────────────────────────────────────────
                    with ui.tab_panel(rt_idx):
                        reps = _apply_filters(grouped.get("bench-index", []))
                        if not reps:
                            ui.label("No index benchmark reports yet.").classes("text-gray-400")
                        else:
                            rows = [{"label": _plan_label(r), **e} for r in reps for e in r.get("results", [])]
                            df = pd.DataFrame(rows)
                            if not df.empty and "batch_size" in df.columns:
                                with ui.row().classes("w-full gap-4"):
                                    fig = px.line(df, x="batch_size", y="docs_per_sec", color="label",
                                                  markers=True, log_x=True, title="Throughput: docs/sec vs batch size")
                                    ui.plotly(fig).classes("flex-1")
                                    fig2 = px.line(df, x="batch_size", y="p95_ms", color="label",
                                                   markers=True, log_x=True, title="p95 latency vs batch size")
                                    ui.plotly(fig2).classes("flex-1")

                    # ── bench-search ───────────────────────────────────────────
                    with ui.tab_panel(rt_search):
                        reps = _apply_filters(grouped.get("bench-search", []))
                        if not reps:
                            ui.label("No search benchmark reports yet.").classes("text-gray-400")
                        else:
                            rows = [{"label": _plan_label(r), **e} for r in reps for e in r.get("results", [])]
                            df = pd.DataFrame(rows)
                            if not df.empty:
                                if "ops_per_sec" in df.columns and df["ops_per_sec"].notna().any():
                                    with ui.row().classes("w-full gap-4"):
                                        fig_ops = px.bar(
                                            df.dropna(subset=["ops_per_sec"]),
                                            x="label", y="ops_per_sec",
                                            title="Sustained throughput (ops/s)", text_auto=".1f",
                                        )
                                        ui.plotly(fig_ops).classes("flex-1")
                                        fig_lat = px.bar(
                                            df.dropna(subset=["ops_per_sec"]),
                                            x="label", y="p95_ms",
                                            title="p95 ms (sustained mode)", text_auto=".1f",
                                        )
                                        ui.plotly(fig_lat).classes("flex-1")
                                rounds_df = df[~df.get("mode", pd.Series(dtype=str)).eq("sustained")] if "mode" in df.columns else df
                                if not rounds_df.empty and "round" in rounds_df.columns:
                                    with ui.row().classes("w-full gap-4"):
                                        fig = px.line(rounds_df, x="round", y="p95_ms", color="label",
                                                      markers=True, title="p95 ms per round")
                                        ui.plotly(fig).classes("flex-1")
                                        if "p90_ms" in rounds_df.columns:
                                            pct_cols = [c for c in ["p50_ms", "p90_ms", "p95_ms", "p99_ms", "p999_ms"] if c in rounds_df.columns]
                                            melt = rounds_df.melt(id_vars=["label"], value_vars=pct_cols,
                                                                   var_name="percentile", value_name="latency_ms")
                                            fig_pct = px.bar(
                                                melt.groupby(["label", "percentile"], as_index=False)["latency_ms"].mean(),
                                                x="percentile", y="latency_ms", color="label",
                                                barmode="group", title="Latency percentiles (avg across rounds)",
                                                text_auto=".1f",
                                            )
                                            ui.plotly(fig_pct).classes("flex-1")
                                        else:
                                            agg = rounds_df.groupby("label", as_index=False)["p95_ms"].mean()
                                            fig2 = px.bar(agg, x="label", y="p95_ms",
                                                          title="Average p95 ms", text_auto=".1f")
                                            ui.plotly(fig2).classes("flex-1")

                    # ── bench-recall ───────────────────────────────────────────
                    with ui.tab_panel(rt_recall):
                        reps = _apply_filters(grouped.get("bench-recall", []))
                        if not reps:
                            ui.label("No recall benchmark reports yet.").classes("text-gray-400")
                        else:
                            rows = [{"label": _plan_label(r), **e} for r in reps for e in r.get("results", [])]
                            df = pd.DataFrame(rows)
                            if not df.empty:
                                recall_cols = [c for c in df.columns if c.startswith("recall@")]
                                if recall_cols and "p95_ms" in df.columns:
                                    recall_df = df.melt(
                                        id_vars=["label", "p95_ms"], value_vars=recall_cols,
                                        var_name="K", value_name="recall",
                                    )
                                    with ui.row().classes("w-full gap-4"):
                                        fig = px.bar(recall_df, x="K", y="recall", color="label",
                                                     barmode="group", title="Recall@K by configuration", text_auto=".3f")
                                        ui.plotly(fig).classes("flex-1")
                                        if "recall@10" in df.columns:
                                            fig2 = px.scatter(df, x="p95_ms", y="recall@10", color="label",
                                                              title="Recall@10 vs p95 ms (Pareto)")
                                            ui.plotly(fig2).classes("flex-1")

                    # ── bench-hybrid ───────────────────────────────────────────
                    with ui.tab_panel(rt_hybrid):
                        reps = _apply_filters(grouped.get("bench-hybrid", []))
                        if not reps:
                            ui.label("No hybrid benchmark reports yet.").classes("text-gray-400")
                        else:
                            rows = [{"label": _plan_label(r), **e} for r in reps for e in r.get("results", [])]
                            df = pd.DataFrame(rows)
                            if not df.empty and "p95_ms" in df.columns:
                                color_col = "filter_selectivity" if "filter_selectivity" in df.columns else "label"
                                fig = px.bar(df, x="label", y="p95_ms", color=color_col,
                                             barmode="group", title="p95 ms by filter selectivity", text_auto=".0f")
                                ui.plotly(fig).classes("w-full")

                    # ── bench-stress ───────────────────────────────────────────
                    with ui.tab_panel(rt_stress):
                        reps = _apply_filters(grouped.get("bench-stress", []))
                        if not reps:
                            ui.label("No stress test reports yet.").classes("text-gray-400")
                        else:
                            interval_rows = [
                                {"label": _plan_label(r), **e}
                                for r in reps for e in r.get("results", [])
                                if e.get("mode") != "summary"
                            ]
                            summary_rows = [
                                {"label": _plan_label(r), **e}
                                for r in reps for e in r.get("results", [])
                                if e.get("mode") == "summary"
                            ]
                            if interval_rows:
                                df_int = pd.DataFrame(interval_rows)
                                with ui.row().classes("w-full gap-4"):
                                    if "search_p95_ms" in df_int.columns:
                                        fig_lat = px.line(
                                            df_int.dropna(subset=["search_p95_ms"]),
                                            x="interval_s", y="search_p95_ms", color="label",
                                            markers=True, title="Search p95 ms over time",
                                        )
                                        ui.plotly(fig_lat).classes("flex-1")
                                    if "error_rate_pct" in df_int.columns:
                                        fig_err = px.line(
                                            df_int, x="interval_s", y="error_rate_pct", color="label",
                                            markers=True, title="Error rate % over time",
                                        )
                                        ui.plotly(fig_err).classes("flex-1")
                                with ui.row().classes("w-full gap-4"):
                                    if "index_ops" in df_int.columns and "search_ops" in df_int.columns:
                                        df_ops = df_int.copy()
                                        df_ops["combined_ops"] = df_ops["index_ops"] + df_ops["search_ops"]
                                        fig_ops = px.line(
                                            df_ops, x="interval_s", y="combined_ops", color="label",
                                            markers=True, title="Combined ops per interval",
                                        )
                                        ui.plotly(fig_ops).classes("flex-1")
                                    if "cluster_status" in df_int.columns:
                                        status_map = {"green": 2, "yellow": 1, "red": 0, "unreachable": -1}
                                        df_int["cluster_health_numeric"] = df_int["cluster_status"].map(status_map)
                                        fig_health = px.line(
                                            df_int.dropna(subset=["cluster_health_numeric"]),
                                            x="interval_s", y="cluster_health_numeric", color="label",
                                            markers=True, title="Cluster health (2=green, 1=yellow, 0=red)",
                                        )
                                        ui.plotly(fig_health).classes("flex-1")
                                if "nodes" in df_int.columns and df_int["nodes"].gt(0).any():
                                    with ui.row().classes("w-full gap-4"):
                                        fig_nodes = px.line(
                                            df_int[df_int["nodes"] > 0],
                                            x="interval_s", y="nodes", color="label",
                                            markers=True, title="Node count over time",
                                        )
                                        ui.plotly(fig_nodes).classes("flex-1")
                                        if "relocating_shards" in df_int.columns:
                                            fig_reloc = px.line(
                                                df_int, x="interval_s", y="relocating_shards", color="label",
                                                markers=True, title="Relocating shards",
                                            )
                                            ui.plotly(fig_reloc).classes("flex-1")
                            _plan_events = {"plan_change_triggered", "plan_change_settled",
                                            "plan_change_failed", "node_count_change"}
                            event_rows = [
                                {"label": _plan_label(r), **e}
                                for r in reps for e in r.get("results", [])
                                if e.get("event") in _plan_events
                            ]
                            failed_events = [e for e in event_rows if e.get("event") == "plan_change_failed"]
                            if failed_events:
                                for fe in failed_events:
                                    with ui.card().classes("w-full bg-amber-50 border border-amber-300 p-3"):
                                        with ui.row().classes("items-center gap-2"):
                                            ui.icon("warning", color="amber")
                                            ui.label(
                                                f"Plan change to '{fe.get('new_plan')}' FAILED "
                                                f"at t={fe.get('interval_s')}s: {fe.get('error', 'unknown error')}"
                                            ).classes("text-amber-800 font-medium text-sm")
                            if event_rows:
                                ui.separator()
                                ui.label("Plan-change & topology events").classes("font-semibold mt-2")
                                ev_cols_all = ["label", "event", "interval_s", "new_plan",
                                               "nodes_before", "nodes_after", "nodes_delta", "error"]
                                ev_cols = [{"name": c, "label": c, "field": c, "align": "left"} for c in ev_cols_all]
                                ui.table(columns=ev_cols, rows=event_rows, row_key="interval_s").classes("w-full")
                            thanos_all: list[dict] = []
                            for r in reps:
                                tm = r.get("params", {}).get("thanos_metrics") or {}
                                if tm:
                                    lbl = _plan_label(r)
                                    for metric_name, pts in tm.items():
                                        for pt in pts:
                                            thanos_all.append({"label": lbl, "metric": metric_name,
                                                               "t": pt.get("t"), "v": pt.get("v")})
                            if thanos_all:
                                import pandas as _pd
                                df_th = _pd.DataFrame(thanos_all)
                                df_th["ts"] = _pd.to_datetime(df_th["t"], unit="s", utc=True)
                                ui.separator()
                                ui.label("Thanos metrics").classes("font-semibold mt-2 text-purple-700")
                                thanos_metrics_to_show = [
                                    ("jvm_heap_pct",       "JVM Heap Used %"),
                                    ("gc_old_count",       "GC Old gen rate (events/s)"),
                                    ("index_rate",         "Index ops/s"),
                                    ("search_rate",        "Search ops/s"),
                                    ("search_latency_ms",  "Avg search latency (ms)"),
                                ]
                                with ui.row().classes("w-full gap-4 flex-wrap"):
                                    for metric_key, metric_title in thanos_metrics_to_show:
                                        sub = df_th[df_th["metric"] == metric_key]
                                        if sub.empty:
                                            continue
                                        fig_th = px.line(sub, x="ts", y="v", color="label",
                                                         markers=False, title=metric_title)
                                        fig_th.update_layout(xaxis_title="Time", yaxis_title="")
                                        ui.plotly(fig_th).classes("flex-1 min-w-96")
                            if summary_rows:
                                df_sum = pd.DataFrame(summary_rows)
                                ui.separator()
                                ui.label("Summary").classes("font-semibold mt-2")
                                cols_show = [c for c in [
                                    "label", "duration_s", "combined_ops_s",
                                    "total_index_ops", "total_index_errors",
                                    "total_search_ops", "total_search_errors",
                                    "overall_error_pct", "search_p95_ms",
                                    "time_to_first_error_s", "final_cluster_status",
                                ] if c in df_sum.columns]
                                tbl_cols = [{"name": c, "label": c, "field": c, "align": "left"} for c in cols_show]
                                ui.table(columns=tbl_cols, rows=df_sum[cols_show].to_dict("records"), row_key="label").classes("w-full")

                    # ── bench-recover ──────────────────────────────────────────
                    with ui.tab_panel(rt_recover):
                        reps = _apply_filters(grouped.get("bench-recover", []))
                        if not reps:
                            ui.label("No recover benchmark reports yet.").classes("text-gray-400")
                        else:
                            rows = [{"label": _plan_label(r), **e} for r in reps for e in r.get("results", [])]
                            df = pd.DataFrame(rows)
                            if not df.empty and "latency_ms" in df.columns:
                                color_col = "type" if "type" in df.columns else "label"
                                fig = px.bar(df, x="label", y="latency_ms", color=color_col,
                                             barmode="group", title="Cold-start vs warm latency", text_auto=".0f")
                                ui.plotly(fig).classes("w-full")

                    # ── Manage ─────────────────────────────────────────────────
                    with ui.tab_panel(rt_manage):
                        ui.label("Manage reports").classes("font-semibold text-base mt-2")
                        ui.label(
                            "Rename plan labels, delete runs, view raw JSON, or export results as CSV. "
                            "Changes take effect immediately — refresh the chart tabs to see updates."
                        ).classes("text-sm text-gray-500 mb-3")

                        manage_reps = _experiments.list_reports(RESULTS_DIR, current_slug)

                        if not manage_reps:
                            ui.label("No reports yet.").classes("text-gray-400")
                        else:
                            # Track checked state per row: filename → bool
                            checked: dict[str, bool] = {r["_filename"]: False for r in manage_reps}

                            with ui.column().classes("w-full gap-2"):
                                for rep in manage_reps:
                                    fname = rep["_filename"]
                                    path_str = rep["_path"]
                                    tags = rep.get("_tags", [])
                                    size_kb = rep.get("_size_kb", 0)
                                    generated = rep.get("generated_at", "")[:16].replace("T", " ")
                                    bench_name = rep.get("name", "?")
                                    plan_lbl = rep.get("params", {}).get("plan_label", "")

                                    with ui.card().classes("w-full p-3"):
                                        with ui.row().classes("w-full items-start gap-3 flex-wrap"):
                                            # Left: metadata
                                            with ui.column().classes("flex-1 min-w-48 gap-1"):
                                                with ui.row().classes("items-center gap-2 flex-wrap"):
                                                    ui.badge(bench_name.replace("bench-", ""), color="blue")
                                                    for tag in tags[1:]:  # skip bench_name (already shown)
                                                        ui.badge(tag, color="teal").props("outline")
                                                    ui.label(f"{size_kb} KB").classes("text-xs text-gray-400")

                                                ui.label(plan_lbl).classes("text-sm font-mono text-gray-700")
                                                ui.label(generated).classes("text-xs text-gray-400")
                                                ui.label(fname).classes("text-xs text-gray-300 font-mono")

                                            # Right: action buttons
                                            with ui.row().classes("gap-1 flex-shrink-0"):
                                                def _open_relabel(p=path_str, cur=plan_lbl) -> None:
                                                    with ui.dialog() as dlg, ui.card().classes("min-w-80"):
                                                        ui.label("Rename plan label").classes("font-semibold")
                                                        new_lbl = ui.input("Plan label", value=cur).classes("w-full")

                                                        def _do_relabel(dp=p, dlg_ref=dlg) -> None:
                                                            _experiments.rename_report_label(dp, new_lbl.value.strip())
                                                            dlg_ref.close()
                                                            results_view.refresh()
                                                            ui.notify("Label updated.", type="positive")

                                                        with ui.row().classes("gap-2 mt-2"):
                                                            ui.button("Save", on_click=_do_relabel, color="blue")
                                                            ui.button("Cancel", on_click=dlg.close).props("flat")
                                                    dlg.open()

                                                def _open_raw(r=rep) -> None:
                                                    raw_copy = {k: v for k, v in r.items() if not k.startswith("_")}
                                                    with ui.dialog() as dlg, ui.card().classes("w-full max-w-3xl"):
                                                        ui.label(r.get("_filename", "")).classes("font-mono text-sm font-semibold")
                                                        ui.separator()
                                                        ui.code(json.dumps(raw_copy, indent=2), language="json").classes("w-full max-h-96 overflow-auto text-xs")
                                                        ui.button("Close", on_click=dlg.close).props("flat").classes("mt-2")
                                                    dlg.open()

                                                def _do_export_csv(r=rep) -> None:
                                                    csv_text = _experiments.export_report_csv(r)
                                                    if csv_text:
                                                        filename = r["_filename"].replace(".json", ".csv")
                                                        ui.download.content(csv_text, filename)
                                                    else:
                                                        ui.notify("No result rows to export.", type="warning")

                                                def _confirm_delete(p=path_str, fn=fname) -> None:
                                                    with ui.dialog() as dlg, ui.card():
                                                        ui.label(f"Delete {fn}?").classes("font-semibold")
                                                        ui.label("Both .json and .md files will be removed.").classes("text-sm text-gray-500")

                                                        def _do_del(dp=p, dlg_ref=dlg) -> None:
                                                            _experiments.delete_report(dp)
                                                            dlg_ref.close()
                                                            results_view.refresh()
                                                            ui.notify("Report deleted.", type="info")

                                                        with ui.row().classes("gap-2 mt-2"):
                                                            ui.button("Delete", on_click=_do_del, color="negative")
                                                            ui.button("Cancel", on_click=dlg.close).props("flat")
                                                    dlg.open()

                                                ui.button("Rename", icon="edit", on_click=_open_relabel).props("flat size=sm color=grey")
                                                ui.button("Raw", icon="code", on_click=_open_raw).props("flat size=sm color=grey")
                                                ui.button("CSV", icon="download", on_click=_do_export_csv).props("flat size=sm color=grey")
                                                ui.button("Delete", icon="delete", on_click=_confirm_delete).props("flat size=sm color=negative")

            results_view()


# ── Entry point ────────────────────────────────────────────────────────────────

ui.run(
    title="Aiven k-NN Benchmark",
    host="0.0.0.0",
    port=8080,
    favicon="🔍",
    storage_secret="aiven-semantic-search-bench-ui",
    dark=False,
    show=False,
    reload=False,
)
