"""
New Test page — configure and submit a benchmark matrix.

The user picks corpus settings, k-NN axes (engine, mode, compression,
data type, HNSW params, query types) and the UI previews the final cell
count with skip-rule annotations before submitting.
"""

from __future__ import annotations

import os
import sys
import time

_pkg_root = os.path.join(os.path.dirname(__file__), "..", "..", "src")
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import streamlit as st

from aiven_semantic_search_bench.job_queue import JobQueue
from aiven_semantic_search_bench.job_spec import BenchmarkJob
from aiven_semantic_search_bench.opensearch_client import KnnSpec

st.set_page_config(page_title="New Test — Aiven k-NN Bench", layout="wide")
st.title("New Test")

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")

if not st.session_state.get("aiven_token"):
    st.warning("Please log in on the **Login** page first.")
    st.stop()

selected_services = st.session_state.get("selected_services", [])
if not selected_services:
    st.warning("No services selected. Go to **Services** first.")
    st.stop()

st.markdown(
    f"Benchmarking **{len(selected_services)}** service(s): "
    + ", ".join(f"`{s['label']}` ({s['name']})" for s in selected_services)
)

# ── Skip-rule validation ──────────────────────────────────────────────────────

def _skip_reason(
    engine: str,
    method: str,
    mode: str,
    compression: str,
    data_type: str,
) -> str | None:
    if method == "ivf" and engine != "faiss":
        return "ivf requires faiss engine"
    if mode == "on_disk" and engine != "faiss":
        return "on_disk requires faiss engine"
    if compression != "none" and mode != "on_disk":
        return "compression requires on_disk mode"
    if data_type in ("byte", "fp16", "binary") and engine != "faiss":
        return f"data_type={data_type} requires faiss engine"
    return None


# ── Form ──────────────────────────────────────────────────────────────────────

with st.form("matrix_form"):
    st.subheader("Corpus settings")
    col1, col2, col3 = st.columns(3)
    corpus_dir = col1.text_input("Corpus dir", value="corpus")
    doc_count = col2.number_input("Doc count", min_value=100, max_value=10_000_000, value=10_000, step=1000)
    query_count = col3.number_input("Query count", min_value=10, max_value=100_000, value=500, step=100)
    embed_dim = col1.selectbox("Embed dim", options=[256, 512, 768, 1536, 3072], index=2)
    out_dir = col2.text_input("Results dir", value=RESULTS_DIR)
    opensearch_index = col3.text_input("Index name", value="bench")

    st.subheader("k-NN axes")
    cA, cB, cC = st.columns(3)
    engines = cA.multiselect("Engines", ["faiss", "lucene"], default=["faiss", "lucene"])
    methods = cB.multiselect("Methods", ["hnsw", "ivf"], default=["hnsw"])
    modes = cC.multiselect("Modes", ["in_memory", "on_disk"], default=["in_memory"])

    cD, cE, cF = st.columns(3)
    compressions = cD.multiselect(
        "Compression (on_disk only)",
        ["none", "1x", "2x", "4x", "8x", "16x", "32x"],
        default=["none"],
    )
    data_types = cE.multiselect(
        "Data types",
        ["float", "byte", "fp16", "binary"],
        default=["float"],
    )
    with_text = cF.checkbox("Include text field (for hybrid)", value=False)
    with_metadata = cF.checkbox("Include metadata field (for filter tests)", value=False)
    derived_source = cF.checkbox("derived_source (2.19+)", value=False)

    st.subheader("HNSW parameters")
    hA, hB, hC = st.columns(3)
    m_val = hA.number_input("m", min_value=4, max_value=128, value=16, step=4)
    ef_construction = hB.number_input("ef_construction", min_value=32, max_value=2048, value=128, step=32)
    ef_search_vals_raw = hC.text_input(
        "ef_search values (comma-separated)",
        value="256",
        help="Multiple values sweep latency/recall trade-off.",
    )

    st.subheader("Query types")
    qA, qB, qC, qD = st.columns(4)
    do_index = qA.checkbox("index", value=True)
    do_search = qB.checkbox("search", value=True)
    do_recall = qC.checkbox("recall (needs qrels.npy)", value=False)
    do_hybrid = qD.checkbox("hybrid (needs text/metadata fields)", value=False)

    filter_sel = st.selectbox(
        "Filter selectivity (hybrid only)",
        ["none", "low", "high"],
        index=0,
    )

    batch_sizes_raw = st.text_input(
        "Batch sizes for index benchmark (comma-separated)",
        value="1,5,10,20,50",
    )
    k_val = st.number_input("k (nearest neighbours)", min_value=1, max_value=500, value=10)
    rounds_val = st.number_input("Rounds (search)", min_value=1, max_value=20, value=3)

    preview = st.form_submit_button("Preview matrix")
    submit = st.form_submit_button("Submit to queue")


# ── Parse inputs ──────────────────────────────────────────────────────────────

def _parse_ints(raw: str) -> list[int]:
    out = []
    for piece in raw.split(","):
        piece = piece.strip()
        if piece.isdigit():
            out.append(int(piece))
    return out or [256]


ef_search_vals = _parse_ints(ef_search_vals_raw)
batch_sizes = _parse_ints(batch_sizes_raw)
bench_types = (
    (["index"] if do_index else [])
    + (["search"] if do_search else [])
    + (["recall"] if do_recall else [])
    + (["hybrid"] if do_hybrid else [])
)

# ── Matrix expansion ──────────────────────────────────────────────────────────

cells = []
for svc in selected_services:
    for engine in engines:
        for method in methods:
            for mode in modes:
                for compression in compressions:
                    for data_type in data_types:
                        for ef_s in ef_search_vals:
                            reason = _skip_reason(engine, method, mode, compression, data_type)
                            cells.append(
                                {
                                    "service":     svc["label"],
                                    "engine":      engine,
                                    "method":      method,
                                    "mode":        mode,
                                    "compression": compression,
                                    "data_type":   data_type,
                                    "ef_search":   ef_s,
                                    "skip_reason": reason or "",
                                    "svc":         svc,
                                }
                            )

valid_cells = [c for c in cells if not c["skip_reason"]]
skipped = [c for c in cells if c["skip_reason"]]

if preview or submit:
    st.subheader("Matrix preview")
    import pandas as pd

    preview_rows = []
    for c in cells:
        for bt in bench_types:
            preview_rows.append(
                {
                    "Service":     c["service"],
                    "Engine":      c["engine"],
                    "Method":      c["method"],
                    "Mode":        c["mode"],
                    "Compression": c["compression"],
                    "DataType":    c["data_type"],
                    "ef_search":   c["ef_search"],
                    "BenchType":   bt,
                    "Status":      "SKIP: " + c["skip_reason"] if c["skip_reason"] else "queued",
                }
            )

    pf = pd.DataFrame(preview_rows)
    st.dataframe(pf, use_container_width=True, hide_index=True)

    active = pf[pf["Status"] == "queued"]
    st.metric("Valid cells to queue", len(active))
    st.metric("Skipped (invalid combos)", len(pf) - len(active))

if submit:
    if not bench_types:
        st.error("Select at least one query type.")
        st.stop()
    if not valid_cells:
        st.error("All matrix cells were skipped. Adjust your axis selections.")
        st.stop()

    q = JobQueue(RESULTS_DIR)
    n_submitted = 0

    progress = st.progress(0.0, text="Submitting jobs…")
    total = len(valid_cells) * len(bench_types)

    for i, c in enumerate(valid_cells):
        svc = c["svc"]
        spec = KnnSpec(
            embed_dim=int(embed_dim),
            engine=c["engine"],
            method=c["method"],
            space_type="cosinesimil",
            mode=c["mode"],
            compression=c["compression"],
            data_type=c["data_type"],
            derived_source=derived_source,
            m=int(m_val),
            ef_construction=int(ef_construction),
            ef_search=c["ef_search"],
            with_text=with_text or do_hybrid,
            with_metadata=with_metadata or do_hybrid,
        )
        for bt in bench_types:
            job = BenchmarkJob(
                bench_type=bt,
                service_label=svc["label"],
                opensearch_uri=svc["uri"],
                opensearch_version=svc["version"],
                opensearch_index=opensearch_index,
                spec=spec,
                embed_dim=int(embed_dim),
                doc_count=int(doc_count),
                query_count=int(query_count),
                corpus_dir=corpus_dir,
                out_dir=RESULTS_DIR,
                rounds=int(rounds_val),
                k=int(k_val),
                batch_sizes=batch_sizes,
                filter_selectivity=filter_sel,
            )
            q.enqueue(job)
            n_submitted += 1
        progress.progress((i + 1) / len(valid_cells), text=f"Submitted {n_submitted}/{total}")

    st.success(f"Queued {n_submitted} job(s). Go to **Queue** to watch progress.")
