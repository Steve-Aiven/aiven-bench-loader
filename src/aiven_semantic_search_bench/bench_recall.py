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

Rounds
------
``--rounds N`` (default 1) repeats the full query set N times and reports:

- Aggregate recall@K across all N×query_count evaluations.
- Per-round recall@K in the results table — variance across rounds is a
  **recall stability** signal.  For exact (Lucene) indexes every round
  should be identical; for approximate (Faiss HNSW) indexes occasional
  small variance is expected and worth quantifying.
- A stability note: max − min recall@K deviation across rounds.

With ``--rounds 200 --query-count 500`` you get 100,000 latency samples
(p99.9 reliable) and 200 independent recall measurements.

Concurrency
-----------
``--search-clients N`` runs N worker threads in parallel, identical to
``bench-search --search-clients N``.  Use 1 (default) for cleanest serial
baseline; raise to stress recall accuracy under concurrent load.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .clickhouse_sink import get_sink
from .config import Settings
from .corpus import load_corpus
from .opensearch_client import KnnSpec, encode_vector, get_index_stats, get_opensearch_client
from .report_context import benchmark_report_extras
from .reporter import raw_samples_enabled, write_report
from .stats import confidence_note, percentiles_ms

_QRELS_NAME = "qrels.npy"
_RECALL_KS = (1, 5, 10, 50, 100)


def _knn_search_ids(
    client, index: str, vector: np.ndarray, k: int, data_type: str = "float"
) -> list[str]:
    """Return the ordered list of document IDs from a k-NN search."""
    resp = client.search(
        index=index,
        body={
            "size": k,
            "query": {
                "knn": {
                    "description_vector": {
                        "vector": encode_vector(vector, data_type),
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
    return dict(enumerate(docs_df["doc_id"].astype(str)))


def _recall_at_k(retrieved_ids: list[str], true_ids: set[str], k: int) -> float:
    """Recall@k: 1.0 if any of the top-k retrieved are in true_ids, else 0.0."""
    return 1.0 if any(rid in true_ids for rid in retrieved_ids[:k]) else 0.0


def _run_recall_round(
    client,
    index: str,
    work_items: list[tuple[int, np.ndarray, set[str]]],
    k: int,
    max_k: int,
    data_type: str,
    search_clients: int,
    recall_ks: tuple[int, ...],
    raw_samples: list | None,
    round_num: int,
) -> tuple[dict[int, float], list[float]]:
    """
    Execute one pass over ``work_items`` with ``search_clients`` parallel
    threads.  Returns (recall_sums, latencies_ms) for this round.

    When ``raw_samples`` is provided, each sample dict includes ``ts``,
    ``round``, ``query_idx``, ``latency_ms``, and per-K recall indicators.
    """
    recall_sums = {rk: 0.0 for rk in recall_ks}
    latencies_ms: list[float] = []
    lock = threading.Lock()

    def _worker(items: list[tuple[int, np.ndarray, set[str]]]) -> None:
        local: list[tuple[int, float, str, list[str], set[str]]] = []
        for q_idx, vec, true_ids in items:
            ts_utc = datetime.now(timezone.utc).isoformat()
            t0 = time.perf_counter()
            retrieved = _knn_search_ids(client, index, vec, k=k, data_type=data_type)
            lat_ms = (time.perf_counter() - t0) * 1000
            local.append((q_idx, lat_ms, ts_utc, retrieved, true_ids))
        with lock:
            for q_idx, lat_ms, ts_utc, retrieved, true_ids in local:
                latencies_ms.append(lat_ms)
                for rk in recall_ks:
                    recall_sums[rk] += _recall_at_k(retrieved, true_ids, rk)
                if raw_samples is not None:
                    row: dict[str, Any] = {
                        "ts": ts_utc,
                        "round": round_num,
                        "query_idx": q_idx,
                        "latency_ms": round(lat_ms, 3),
                    }
                    for rk in recall_ks:
                        row[f"recall@{rk}"] = float(_recall_at_k(retrieved, true_ids, rk))
                    raw_samples.append(row)

    chunk_size = max(1, (len(work_items) + search_clients - 1) // search_clients)
    chunks = [work_items[i : i + chunk_size] for i in range(0, len(work_items), chunk_size)]
    with ThreadPoolExecutor(max_workers=search_clients) as pool:
        futs = [pool.submit(_worker, chunk) for chunk in chunks]
        for fut in as_completed(futs):
            fut.result()

    return recall_sums, latencies_ms


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
    search_clients: int = 1,
    rounds: int = 1,
) -> int:
    uri = opensearch_uri or settings.opensearch_uri
    knn = spec or KnnSpec(embed_dim=embed_dim)
    deployment_ctx, preflight_ctx = benchmark_report_extras(settings, uri)
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

    max_k = min(k, gt_k)
    recall_ks = tuple(rk for rk in _RECALL_KS if rk <= max_k)

    # Pre-build work items (query_idx, vector, true_ids) — reused each round.
    query_vectors = bundle.query_vectors[:query_count]
    work_items: list[tuple[int, np.ndarray, set[str]]] = []
    for q_idx in range(query_count):
        true_ids = {
            idx_to_id[int(i)]
            for i in qrels[q_idx][:max_k]
            if int(i) in idx_to_id
        }
        work_items.append((q_idx, query_vectors[q_idx], true_ids))

    save_raw = raw_samples_enabled()
    raw_samples: list[dict[str, Any]] = [] if save_raw else None  # type: ignore[assignment]

    print(
        f"[bench-recall] Running {rounds} round(s) × {query_count} queries "
        f"(k={k}, dim={embed_dim}, search_clients={search_clients}, "
        f"total_samples={rounds * query_count:,})..."
    )

    # ── Round loop ─────────────────────────────────────────────────────────────
    all_latencies: list[float] = []
    agg_recall_sums = {rk: 0.0 for rk in recall_ks}
    results: list[dict[str, Any]] = []
    per_round_recall: dict[int, dict[str, float]] = {}  # for stability analysis
    sink = get_sink()

    for r in range(1, rounds + 1):
        round_recall_sums, round_lats = _run_recall_round(
            client=client,
            index=settings.opensearch_index,
            work_items=work_items,
            k=k,
            max_k=max_k,
            data_type=knn.data_type,
            search_clients=search_clients,
            recall_ks=recall_ks,
            raw_samples=raw_samples,
            round_num=r,
        )
        all_latencies.extend(round_lats)
        for rk in recall_ks:
            agg_recall_sums[rk] += round_recall_sums[rk]

        s = percentiles_ms(round_lats)
        round_recalls = {
            f"recall@{rk}": round(round_recall_sums[rk] / query_count, 4)
            for rk in recall_ks
        }
        per_round_recall[r] = round_recalls

        row: dict[str, Any] = {
            "round":   r,
            "queries": query_count,
            "clients": search_clients,
            "p50_ms":  round(s["p50_ms"], 1),
            "p95_ms":  round(s["p95_ms"], 1),
            "p99_ms":  round(s["p99_ms"], 1),
            "max_ms":  round(s["max_ms"], 1),
            **round_recalls,
        }
        results.append(row)

        round_label = {"round": str(r)}
        sink.metric("search_p50_ms", float(s["p50_ms"]), labels=round_label)
        sink.metric("search_p99_ms", float(s["p99_ms"]), labels=round_label)
        for rk_key, rk_val in round_recalls.items():
            k_num = rk_key.split("@", 1)[1]
            sink.metric("recall_at_k", float(rk_val), labels={"k": k_num, "round": str(r)})

        print(
            f"[bench-recall] round {r}/{rounds}: "
            f"p50={s['p50_ms']:.1f}ms  p99={s['p99_ms']:.1f}ms  "
            + "  ".join(f"{rk}={v:.4f}" for rk, v in round_recalls.items())
        )

    # ── Aggregate across all rounds ────────────────────────────────────────────
    total_queries = rounds * query_count
    agg_lat = percentiles_ms(all_latencies)
    agg_recall = {
        f"recall@{rk}": round(agg_recall_sums[rk] / total_queries, 4)
        for rk in recall_ks
    }

    # Add aggregate row when rounds > 1
    if rounds > 1:
        agg_row: dict[str, Any] = {
            "round":   "AGGREGATE",
            "queries": total_queries,
            "clients": search_clients,
            "p50_ms":  round(agg_lat["p50_ms"], 1),
            "p95_ms":  round(agg_lat["p95_ms"], 1),
            "p99_ms":  round(agg_lat["p99_ms"], 1),
            "max_ms":  round(agg_lat["max_ms"], 1),
            **agg_recall,
        }
        results.append(agg_row)

    # ── Recall stability (rounds > 1) ──────────────────────────────────────────
    stability_notes: list[str] = []
    if rounds > 1:
        for rk in recall_ks:
            key = f"recall@{rk}"
            per_round_vals = [per_round_recall[r][key] for r in range(1, rounds + 1)]
            mn, mx = min(per_round_vals), max(per_round_vals)
            deviation = round(mx - mn, 4)
            sink.metric("recall_stability_range", deviation, labels={"k": str(rk)})
            if deviation > 0:
                stability_notes.append(
                    f"recall@{rk} range across {rounds} rounds: "
                    f"{mn:.4f}–{mx:.4f} (Δ={deviation:.4f}); "
                    + ("deterministic index" if deviation == 0 else
                       "expected HNSW non-determinism" if deviation < 0.01 else
                       "HIGH — consider raising ef_search")
                )
            else:
                stability_notes.append(f"recall@{rk}: perfectly stable across {rounds} rounds.")

    # ── Emit aggregate metrics ─────────────────────────────────────────────────
    for rk_key, rk_val in agg_recall.items():
        k_num = rk_key.split("@", 1)[1]
        sink.metric("recall_at_k", float(rk_val), labels={"k": k_num})
    sink.metric("search_p50_ms", float(agg_lat["p50_ms"]), labels={"phase": "recall"})
    sink.metric("search_p99_ms", float(agg_lat["p99_ms"]), labels={"phase": "recall"})

    print(
        f"[bench-recall] aggregate ({total_queries:,} queries, {rounds} round(s)): "
        f"p50={agg_lat['p50_ms']:.1f}ms  p99={agg_lat['p99_ms']:.1f}ms  "
        + "  ".join(f"{rk}={v:.4f}" for rk, v in agg_recall.items())
    )

    raw_payload = {"queries": raw_samples} if save_raw and raw_samples else None

    json_path, md_path, raw_path = write_report(
        "bench-recall",
        params={
            "plan_label":    label,
            "index":         settings.opensearch_index,
            "doc_count":     doc_count,
            "query_count":   query_count,
            "rounds":        rounds,
            "k":             k,
            "search_clients": search_clients,
            "embed_dim":     embed_dim,
            "knn_spec":      knn.to_dict(),
            "corpus_preset": bundle.manifest.get("preset"),
            "corpus_dir":    corpus_dir,
            "groundtruth_k": int(gt_k),
        },
        results=results,
        deployment=deployment_ctx,
        preflight=preflight_ctx,
        notes=[
            "recall@K = fraction of queries where at least one of the K ground-truth "
            "nearest neighbours appears in the OpenSearch top-K results.",
            f"Ground truth computed at dim={bundle.manifest.get('groundtruth_dim', 'MAX_DIM')} "
            f"(brute-force cosine similarity).",
            f"k-NN spec: {knn.label()}",
            f"Latency confidence: {confidence_note(total_queries)}",
            *stability_notes,
        ],
        out_dir=out_dir,
        raw_data=raw_payload,
    )
    print(f"[bench-recall] Wrote: {json_path}")
    print(f"[bench-recall] Wrote: {md_path}")
    if raw_path:
        print(f"[bench-recall] Wrote: {raw_path}")
    return 0
