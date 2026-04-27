"""
bench-search: semantic search latency benchmark.

What this measures
-------------------
Issue ``--query-count`` k-NN searches per round for ``--rounds`` rounds, using
queries pulled directly from the pre-built corpus. Each query already has its
embedding cached; the benchmark only times the OpenSearch round-trip. Reports
per-round and aggregate p50/p95/p99 client-side latency.

Why pre-embedded queries?
--------------------------
If we embedded queries with Gemini at benchmark time, the measurement would
include Vertex AI throughput and Google round-trip latency. We don't want
that — we want to measure OpenSearch and only OpenSearch.

Setup
------
Run ``bench-index`` first to populate the index with corpus documents at the
same ``--embed-dim`` you intend to query at.
"""

from __future__ import annotations

import time

import numpy as np

from .config import Settings
from .corpus import load_corpus
from .opensearch_client import KnnSpec, get_index_stats, get_opensearch_client
from .reporter import write_report
from .stats import percentiles_ms


def _knn_search(client, index: str, vector: np.ndarray, k: int, ef_search: int) -> list[str]:
    """Issue one k-NN search and return the list of returned doc IDs."""
    resp = client.search(
        index=index,
        body={
            "size": k,
            "query": {
                "knn": {
                    "description_vector": {
                        "vector": vector.tolist(),
                        "k": k,
                        "filter": {"match_all": {}},
                    }
                }
            },
            # ef_search controls recall vs latency at query time (HNSW).
            "search_pipeline": None,
        },
        params={"search_type": "query_then_fetch"},
    )
    hits = resp.get("hits", {}).get("hits", [])
    return [h["_id"] for h in hits]


def cmd_bench_search(
    settings: Settings,
    *,
    rounds: int,
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
        print(f"[bench-search] ERROR: index '{settings.opensearch_index}' does not exist.")
        print("[bench-search] Run 'bench-index' first to populate the index.")
        return 1

    stats = get_index_stats(client, settings.opensearch_index)
    doc_count = stats["doc_count"]
    if doc_count == 0:
        print(f"[bench-search] ERROR: index '{settings.opensearch_index}' is empty.")
        print("[bench-search] Run 'bench-index' first to populate the index.")
        return 1

    print(f"[bench-search] Loading corpus from {corpus_dir} at dim={embed_dim}...")
    bundle = load_corpus(corpus_dir, embed_dim)
    available = len(bundle.queries)
    if query_count > available:
        print(
            f"[bench-search] WARNING: requested {query_count} queries, "
            f"corpus has {available}. Using {available}."
        )
        query_count = available

    query_vectors = bundle.query_vectors[:query_count]

    print(
        f"[bench-search] Index '{settings.opensearch_index}' has {doc_count:,} docs. "
        f"Running {rounds} round(s) x {query_count} queries (k={k}, dim={embed_dim}, "
        f"ef_search={knn.ef_search})..."
    )

    results: list[dict] = []
    all_latencies_ms: list[float] = []

    for r in range(1, rounds + 1):
        round_latencies: list[float] = []
        for vec in query_vectors:
            t0 = time.perf_counter()
            _knn_search(client, settings.opensearch_index, vec, k=k, ef_search=knn.ef_search)
            round_latencies.append((time.perf_counter() - t0) * 1000)
        all_latencies_ms.extend(round_latencies)

        s = percentiles_ms(round_latencies)
        results.append(
            {
                "round":   r,
                "queries": len(round_latencies),
                "p50_ms":  round(s["p50_ms"], 1),
                "p95_ms":  round(s["p95_ms"], 1),
                "p99_ms":  round(s["p99_ms"], 1),
                "max_ms":  round(s["max_ms"], 1),
                "mean_ms": round(s["mean_ms"], 1),
            }
        )
        print(
            f"[bench-search] round {r}/{rounds}: "
            f"p50={s['p50_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms  "
            f"p99={s['p99_ms']:.1f}ms  max={s['max_ms']:.1f}ms"
        )

    overall = percentiles_ms(all_latencies_ms)
    print(
        f"[bench-search] Overall ({len(all_latencies_ms)} queries): "
        f"p50={overall['p50_ms']:.1f}ms  p95={overall['p95_ms']:.1f}ms  "
        f"p99={overall['p99_ms']:.1f}ms"
    )

    json_path, md_path = write_report(
        "bench-search",
        params={
            "plan_label":    label,
            "index":         settings.opensearch_index,
            "doc_count":     doc_count,
            "rounds":        rounds,
            "k":             k,
            "query_count":   query_count,
            "embed_dim":     embed_dim,
            "knn_spec":      knn.to_dict(),
            "corpus_preset": bundle.manifest.get("preset"),
            "corpus_dir":    corpus_dir,
            "embed_model":   bundle.manifest.get("embed_model"),
        },
        results=results,
        notes=[
            "Queries are loaded from the pre-built corpus; embeddings are NOT recomputed.",
            "Latency is client-side round-trip including network to the Aiven service.",
            "Per-round rows show whether latency is stable or drifts under sustained load.",
            f"Embedding dimension: {embed_dim} (Matryoshka-truncated from "
            f"{bundle.source_dim} stored).",
            f"k-NN spec: {knn.label()}  ef_search={knn.ef_search}",
            f"Label '{label}' — re-run with a different label to compare configurations.",
        ],
        out_dir=out_dir,
    )
    print(f"[bench-search] Wrote: {json_path}")
    print(f"[bench-search] Wrote: {md_path}")
    return 0
