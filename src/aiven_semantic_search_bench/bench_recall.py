"""
bench-recall: recall@K accuracy measurement.

What this measures
-------------------
Issue the same k-NN searches as ``bench-search`` but compare each result set
against the brute-force ground truth in ``corpus/qrels.npy`` to compute
recall@K for K ∈ {1, 5, 10, 50, 100}.

recall@K = fraction of queries where at least one of the K ground-truth
           nearest neighbours appears in the K results returned by OpenSearch.

Run ``bench-build-groundtruth`` once before this command to build the qrels
file.  If the file is missing the command exits with an informative error.

Setup
------
Run ``bench-index`` first to populate the index, then ``bench-recall``
against the same index.  The ``--embed-dim`` and ``--k`` must match.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import Settings
from .corpus import load_corpus
from .opensearch_client import KnnSpec, get_index_stats, get_opensearch_client
from .reporter import write_report
from .stats import percentiles_ms

_QRELS_NAME = "qrels.npy"
_RECALL_KS = (1, 5, 10, 50, 100)


def _knn_search_ids(
    client, index: str, vector: np.ndarray, k: int
) -> list[str]:
    """Return the ordered list of document IDs from a k-NN search."""
    resp = client.search(
        index=index,
        body={
            "size": k,
            "query": {
                "knn": {
                    "description_vector": {
                        "vector": vector.tolist(),
                        "k": k,
                    }
                }
            },
        },
    )
    return [h["_id"] for h in resp.get("hits", {}).get("hits", [])]


def _load_doc_id_map(corpus_dir: str, doc_count: int) -> dict[int, str]:
    """Build a mapping row_index → doc_id from docs.parquet."""
    import pandas as pd
    docs_df = pd.read_parquet(Path(corpus_dir) / "docs.parquet")
    docs_df = docs_df.iloc[:doc_count]
    return {i: str(row["doc_id"]) for i, row in enumerate(docs_df.itertuples())}


def _recall_at_k(retrieved_ids: list[str], true_ids: set[str], k: int) -> float:
    """Recall@k: 1.0 if any of the top-k retrieved are in true_ids, else 0.0."""
    return 1.0 if any(rid in true_ids for rid in retrieved_ids[:k]) else 0.0


def cmd_bench_recall(
    settings: Settings,
    *,
    query_count: int,
    k: int,
    embed_dim: int,
    spec: KnnSpec | None = None,
    corpus_dir: str,
    out_dir: str,
    label: str,
    opensearch_uri: str | None = None,
) -> int:
    uri = opensearch_uri or settings.opensearch_uri
    knn = spec or KnnSpec(embed_dim=embed_dim)
    client = get_opensearch_client(uri)

    if not client.indices.exists(index=settings.opensearch_index):
        print(f"[bench-recall] ERROR: index '{settings.opensearch_index}' does not exist.")
        print("[bench-recall] Run 'bench-index' first.")
        return 1

    stats = get_index_stats(client, settings.opensearch_index)
    doc_count = stats["doc_count"]
    if doc_count == 0:
        print(f"[bench-recall] ERROR: index '{settings.opensearch_index}' is empty.")
        return 1

    # Load ground truth.
    qrels_path = Path(corpus_dir) / _QRELS_NAME
    if not qrels_path.exists():
        print(
            f"[bench-recall] ERROR: {qrels_path} not found. "
            "Run 'bench-build-groundtruth' first."
        )
        return 1

    qrels = np.load(str(qrels_path))  # shape (Q, K_gt)
    gt_k = qrels.shape[1]
    print(f"[bench-recall] Ground truth: {qrels.shape[0]:,} queries × top-{gt_k}")

    # Build row-index → doc_id mapping so we can convert integer qrels to IDs.
    print("[bench-recall] Building doc ID map...")
    idx_to_id = _load_doc_id_map(corpus_dir, int(doc_count))

    print(f"[bench-recall] Loading corpus from {corpus_dir} at dim={embed_dim}...")
    bundle = load_corpus(corpus_dir, embed_dim)
    available = min(len(bundle.queries), qrels.shape[0])
    if query_count > available:
        print(
            f"[bench-recall] WARNING: requested {query_count} queries, "
            f"available {available}. Using {available}."
        )
        query_count = available

    query_vectors = bundle.query_vectors[:query_count]

    max_k = min(k, gt_k)
    recall_sums = {rk: 0.0 for rk in _RECALL_KS if rk <= max_k}
    latencies_ms: list[float] = []

    print(
        f"[bench-recall] Running {query_count} queries (k={k}, dim={embed_dim})..."
    )
    for q_idx, vec in enumerate(query_vectors):
        true_row_indices = qrels[q_idx][:max_k]
        true_ids = {idx_to_id[int(i)] for i in true_row_indices if int(i) in idx_to_id}

        t0 = time.perf_counter()
        retrieved = _knn_search_ids(client, settings.opensearch_index, vec, k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        for rk in recall_sums:
            recall_sums[rk] += _recall_at_k(retrieved, true_ids, rk)

    lat = percentiles_ms(latencies_ms)
    recall_results = {
        f"recall@{rk}": round(recall_sums[rk] / query_count, 4)
        for rk in recall_sums
    }

    result_row = {
        "queries":    query_count,
        "k":          k,
        "p50_ms":     round(lat["p50_ms"], 1),
        "p95_ms":     round(lat["p95_ms"], 1),
        "p99_ms":     round(lat["p99_ms"], 1),
        "max_ms":     round(lat["max_ms"], 1),
        "mean_ms":    round(lat["mean_ms"], 1),
        **recall_results,
    }

    print(
        f"[bench-recall] p50={lat['p50_ms']:.1f}ms  p95={lat['p95_ms']:.1f}ms  "
        + "  ".join(f"{rk}={v:.3f}" for rk, v in recall_results.items())
    )

    json_path, md_path = write_report(
        "bench-recall",
        params={
            "plan_label":    label,
            "index":         settings.opensearch_index,
            "doc_count":     doc_count,
            "query_count":   query_count,
            "k":             k,
            "embed_dim":     embed_dim,
            "knn_spec":      knn.to_dict(),
            "corpus_preset": bundle.manifest.get("preset"),
            "corpus_dir":    corpus_dir,
            "groundtruth_k": int(gt_k),
        },
        results=[result_row],
        notes=[
            "recall@K = fraction of queries where at least one of the K ground-truth "
            "nearest neighbours appears in the OpenSearch top-K results.",
            f"Ground truth computed at dim={bundle.manifest.get('groundtruth_dim', 'MAX_DIM')} "
            f"(brute-force cosine similarity).",
            f"k-NN spec: {knn.label()}",
        ],
        out_dir=out_dir,
    )
    print(f"[bench-recall] Wrote: {json_path}")
    print(f"[bench-recall] Wrote: {md_path}")
    return 0
