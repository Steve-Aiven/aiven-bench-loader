"""
bench-search: semantic search latency + sustained-throughput benchmark.

What this measures
-------------------
Two modes, selected by ``target_throughput`` / ``time_period``:

Rounds mode (default):
  Issue ``query_count`` k-NN searches per round for ``rounds`` rounds,
  optionally in parallel across ``search_clients`` worker threads.  Reports
  per-round and aggregate latency percentiles (p50/p90/p95/p99/p99.9).

Sustained-throughput mode (``target_throughput`` > 0 and ``time_period`` > 0):
  Drive ``search_clients`` workers at ``target_throughput`` ops/s for
  ``time_period`` seconds.  Reports measured ops/s and the full latency
  distribution over the run.  If the cluster is slower than the target rate
  the measured ops/s falls below the target — that gap is itself a signal.
  Modelled on OSB's ``target_throughput`` + ``time_period`` parameters.

OSB-inspired pre-measurement steps
------------------------------------
1. Force-merge (``force_merge_segments`` > 0):
   Merge the index to N segments before any search so measured latency
   reflects a fully-optimised production index rather than a freshly-
   ingested one with many small segments.  OSB does the same in its
   ``force-merge-index`` procedure.

2. Warmup (``warmup_queries`` > 0):
   Issue queries until p95 latency changes less than 10 % between
   consecutive warmup rounds (up to 5 rounds), then discard those samples.
   This removes cold-cache effects from the first measured round, mirroring
   OSB's ``warmup-knn-indices`` custom runner.

Why pre-embedded queries?
--------------------------
Query vectors come from the pre-built corpus.  No embedding model is called
at benchmark time, so measured latency reflects OpenSearch + network only.
"""

from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .config import Settings
from .corpus import load_corpus
from .opensearch_client import (
    KnnSpec,
    force_merge as os_force_merge,
    get_index_stats,
    get_opensearch_client,
)
from .reporter import write_report
from .stats import percentiles_ms

# ── Warmup constants (mirroring OSB's warmup-knn-indices runner) ──────────────
_WARMUP_STABILITY_THRESHOLD = 0.10  # <10 % Δp95 ⇒ latency is stable
_WARMUP_MAX_ROUNDS = 5


# ── Core search call ──────────────────────────────────────────────────────────

def _knn_search(client, index: str, vector: np.ndarray, k: int, ef_search: int) -> None:
    """Issue one k-NN search.  Return value is discarded; we only time the call."""
    client.search(
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
        },
        params={"search_type": "query_then_fetch"},
    )


# ── Warmup phase ──────────────────────────────────────────────────────────────

def _warmup_until_stable(
    client,
    index: str,
    query_vectors: np.ndarray,
    k: int,
    ef_search: int,
    queries_per_round: int,
) -> None:
    """
    Issue warmup queries until p95 latency changes less than
    ``_WARMUP_STABILITY_THRESHOLD`` between consecutive rounds, or until
    ``_WARMUP_MAX_ROUNDS`` rounds have been completed.

    Mirrors OSB's ``warmup-knn-indices`` custom runner which retries the
    warmup operation until the index is ready and latency has stabilised.
    Measurements are discarded — they represent the cost of warming the
    JVM JIT, OS page cache, and HNSW graph memory, not the steady-state
    query latency we care about.
    """
    vecs = query_vectors[:queries_per_round]
    prev_p95: float | None = None

    for rnd in range(1, _WARMUP_MAX_ROUNDS + 1):
        lats: list[float] = []
        for v in vecs:
            t0 = time.perf_counter()
            _knn_search(client, index, v, k=k, ef_search=ef_search)
            lats.append((time.perf_counter() - t0) * 1000)

        p95 = percentiles_ms(lats)["p95_ms"]
        print(f"[bench-search] warmup round {rnd}/{_WARMUP_MAX_ROUNDS}: p95={p95:.1f} ms")

        if prev_p95 is not None:
            delta = abs(p95 - prev_p95) / max(prev_p95, 1.0)
            if delta < _WARMUP_STABILITY_THRESHOLD:
                print(
                    f"[bench-search] warmup stable (Δp95={delta:.1%} < 10%) "
                    f"after {rnd} round(s)."
                )
                return
        prev_p95 = p95

    print(f"[bench-search] warmup reached max {_WARMUP_MAX_ROUNDS} rounds; proceeding.")


# ── Rounds mode ───────────────────────────────────────────────────────────────

def _run_rounds(
    client,
    index: str,
    query_vectors: np.ndarray,
    k: int,
    ef_search: int,
    *,
    rounds: int,
    search_clients: int,
) -> list[dict]:
    """
    Rounds mode: for each round dispatch all ``len(query_vectors)`` queries
    across ``search_clients`` parallel workers using a thread pool.

    Using ``search_clients > 1`` applies concurrency pressure to the cluster
    and is closer to a real serving workload than strictly serial queries.
    """
    all_latencies: list[float] = []
    results: list[dict] = []

    for r in range(1, rounds + 1):
        round_lats: list[float] = []
        lock = threading.Lock()

        # Partition query vectors across workers
        n = len(query_vectors)
        chunk_size = max(1, (n + search_clients - 1) // search_clients)
        chunks = [
            query_vectors[i : i + chunk_size]
            for i in range(0, n, chunk_size)
        ]

        def _worker(vecs: np.ndarray) -> None:
            for v in vecs:
                t0 = time.perf_counter()
                _knn_search(client, index, v, k=k, ef_search=ef_search)
                with lock:
                    round_lats.append((time.perf_counter() - t0) * 1000)

        with ThreadPoolExecutor(max_workers=search_clients) as pool:
            futs = [pool.submit(_worker, chunk) for chunk in chunks]
            for f in as_completed(futs):
                f.result()  # re-raise any exception from a worker thread

        all_latencies.extend(round_lats)
        s = percentiles_ms(round_lats)
        results.append({
            "round":   r,
            "queries": len(round_lats),
            "p50_ms":  round(s["p50_ms"],  1),
            "p90_ms":  round(s["p90_ms"],  1),
            "p95_ms":  round(s["p95_ms"],  1),
            "p99_ms":  round(s["p99_ms"],  1),
            "p999_ms": round(s["p999_ms"], 1),
            "max_ms":  round(s["max_ms"],  1),
            "mean_ms": round(s["mean_ms"], 1),
        })
        print(
            f"[bench-search] round {r}/{rounds}: "
            f"p50={s['p50_ms']:.1f}  p90={s['p90_ms']:.1f}  "
            f"p95={s['p95_ms']:.1f}  p99={s['p99_ms']:.1f}  "
            f"p99.9={s['p999_ms']:.1f}  max={s['max_ms']:.1f} ms"
        )

    overall = percentiles_ms(all_latencies)
    n_total = len(all_latencies)
    print(
        f"[bench-search] overall ({n_total} queries across {rounds} round(s)): "
        f"p50={overall['p50_ms']:.1f}  p95={overall['p95_ms']:.1f}  "
        f"p99={overall['p99_ms']:.1f}  p99.9={overall['p999_ms']:.1f} ms"
    )
    return results


# ── Sustained-throughput mode ─────────────────────────────────────────────────

def _run_sustained(
    client,
    index: str,
    query_vectors: np.ndarray,
    k: int,
    ef_search: int,
    *,
    search_clients: int,
    target_throughput: float,
    time_period: int,
) -> list[dict]:
    """
    Sustained-throughput mode, modelled on OSB's ``target_throughput`` +
    ``time_period`` workload parameters.

    A rate-limiting feeder thread dispatches query vectors at
    ``target_throughput`` ops/s for ``time_period`` seconds.
    ``search_clients`` worker threads drain the queue and record latency.

    If the cluster is slower than the target rate the queue fills up and the
    measured ops/s falls below the target — that gap is itself a useful
    signal (the cluster is saturated at this concurrency level).
    """
    interval = 1.0 / target_throughput          # seconds between dispatch slots
    deadline = time.monotonic() + time_period
    work_q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=search_clients * 8)
    latencies: list[float] = []
    lock = threading.Lock()
    n_vecs = len(query_vectors)

    def feeder() -> None:
        idx = 0
        next_t = time.monotonic()
        while time.monotonic() < deadline:
            sleep_for = next_t - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            work_q.put(query_vectors[idx % n_vecs])
            idx += 1
            next_t += interval
        for _ in range(search_clients):
            work_q.put(None)  # sentinel: tell each worker to stop

    def worker() -> None:
        while True:
            vec = work_q.get()
            if vec is None:
                break
            t0 = time.perf_counter()
            _knn_search(client, index, vec, k=k, ef_search=ef_search)
            with lock:
                latencies.append((time.perf_counter() - t0) * 1000)

    t_start = time.perf_counter()
    feeder_thread = threading.Thread(target=feeder, daemon=True)
    feeder_thread.start()

    with ThreadPoolExecutor(max_workers=search_clients) as pool:
        futs = [pool.submit(worker) for _ in range(search_clients)]
        for f in as_completed(futs):
            f.result()

    feeder_thread.join(timeout=5)
    elapsed = time.perf_counter() - t_start

    s = percentiles_ms(latencies)
    ops_per_sec = round(len(latencies) / elapsed, 1) if elapsed > 0 else 0.0

    print(
        f"[bench-search] sustained: {len(latencies)} queries in {elapsed:.1f}s "
        f"= {ops_per_sec} ops/s (target={target_throughput:.1f})"
    )
    print(
        f"[bench-search] p50={s['p50_ms']:.1f}  p90={s['p90_ms']:.1f}  "
        f"p95={s['p95_ms']:.1f}  p99={s['p99_ms']:.1f}  "
        f"p99.9={s['p999_ms']:.1f}  max={s['max_ms']:.1f} ms"
    )

    return [{
        "mode":              "sustained",
        "queries":           len(latencies),
        "elapsed_s":         round(elapsed, 1),
        "ops_per_sec":       ops_per_sec,
        "target_throughput": target_throughput,
        "search_clients":    search_clients,
        "p50_ms":            round(s["p50_ms"],  1),
        "p90_ms":            round(s["p90_ms"],  1),
        "p95_ms":            round(s["p95_ms"],  1),
        "p99_ms":            round(s["p99_ms"],  1),
        "p999_ms":           round(s["p999_ms"], 1),
        "max_ms":            round(s["max_ms"],  1),
        "mean_ms":           round(s["mean_ms"], 1),
    }]


# ── Public entry point ────────────────────────────────────────────────────────

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
    # OSB-inspired parameters
    warmup_queries: int = 50,
    search_clients: int = 1,
    target_throughput: float = 0.0,
    time_period: int = 0,
    force_merge_segments: int = 0,
) -> int:
    uri = opensearch_uri or settings.opensearch_uri
    knn = spec or KnnSpec(embed_dim=embed_dim)
    client = get_opensearch_client(uri)

    if not client.indices.exists(index=settings.opensearch_index):
        print(f"[bench-search] ERROR: index '{settings.opensearch_index}' does not exist.")
        print("[bench-search] Run 'bench-index' first to populate the index.")
        return 1

    idx_stats = get_index_stats(client, settings.opensearch_index)
    doc_count = idx_stats["doc_count"]
    if doc_count == 0:
        print(f"[bench-search] ERROR: index '{settings.opensearch_index}' is empty.")
        print("[bench-search] Run 'bench-index' first to populate the index.")
        return 1

    # ── Step 1: optional force-merge (OSB force-merge-index procedure) ────
    if force_merge_segments > 0:
        print(
            f"[bench-search] Force-merging index to {force_merge_segments} segment(s) "
            f"(mirrors OSB force-merge-index procedure)..."
        )
        os_force_merge(client, settings.opensearch_index, max_num_segments=force_merge_segments)
        print("[bench-search] Force-merge complete.")

    # ── Load corpus ───────────────────────────────────────────────────────
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

    sustained_mode = target_throughput > 0 and time_period > 0
    mode_desc = (
        f"sustained {target_throughput:.1f} ops/s for {time_period}s "
        f"via {search_clients} client(s)"
        if sustained_mode
        else f"{rounds} round(s) × {query_count} queries via {search_clients} client(s)"
    )
    print(
        f"[bench-search] Index '{settings.opensearch_index}' has {doc_count:,} docs. "
        f"Running {mode_desc} (k={k}, dim={embed_dim}, ef_search={knn.ef_search})..."
    )

    # ── Step 2: warmup (OSB warmup-knn-indices) ───────────────────────────
    if warmup_queries > 0:
        print(
            f"[bench-search] Warmup: up to {_WARMUP_MAX_ROUNDS} rounds of "
            f"{min(warmup_queries, len(query_vectors))} queries until p95 "
            f"stabilises (OSB-inspired)..."
        )
        _warmup_until_stable(
            client,
            settings.opensearch_index,
            query_vectors,
            k=k,
            ef_search=knn.ef_search,
            queries_per_round=min(warmup_queries, len(query_vectors)),
        )

    # ── Step 3: measure ───────────────────────────────────────────────────
    if sustained_mode:
        results = _run_sustained(
            client,
            settings.opensearch_index,
            query_vectors,
            k=k,
            ef_search=knn.ef_search,
            search_clients=search_clients,
            target_throughput=target_throughput,
            time_period=time_period,
        )
    else:
        results = _run_rounds(
            client,
            settings.opensearch_index,
            query_vectors,
            k=k,
            ef_search=knn.ef_search,
            rounds=rounds,
            search_clients=search_clients,
        )

    # ── Report ────────────────────────────────────────────────────────────
    json_path, md_path = write_report(
        "bench-search",
        params={
            "plan_label":            label,
            "index":                 settings.opensearch_index,
            "doc_count":             doc_count,
            "mode":                  "sustained" if sustained_mode else "rounds",
            "rounds":                rounds,
            "k":                     k,
            "query_count":           query_count,
            "embed_dim":             embed_dim,
            "knn_spec":              knn.to_dict(),
            "corpus_preset":         bundle.manifest.get("preset"),
            "corpus_dir":            corpus_dir,
            "embed_model":           bundle.manifest.get("embed_model"),
            "search_clients":        search_clients,
            "warmup_queries":        warmup_queries,
            "target_throughput":     target_throughput,
            "time_period":           time_period,
            "force_merge_segments":  force_merge_segments,
        },
        results=results,
        notes=[
            "Queries are loaded from the pre-built corpus; embeddings are NOT recomputed.",
            "Latency is client-side round-trip including network to the Aiven service.",
            f"Embedding dimension: {embed_dim} (Matryoshka-truncated from "
            f"{bundle.source_dim} stored).",
            f"k-NN spec: {knn.label()}  ef_search={knn.ef_search}",
            f"Label '{label}' — re-run with a different label to compare configurations.",
            "p90 and p99.9 added for finer tail-latency visibility (matches OSB output).",
            *(
                [f"Warmup: {min(warmup_queries, len(query_vectors))} queries/round "
                 f"until p95 stable (OSB-inspired)."]
                if warmup_queries > 0
                else ["Warmup: disabled."]
            ),
            *(
                [f"Force-merge: max_num_segments={force_merge_segments} "
                 f"(mirrors OSB force-merge-index procedure)."]
                if force_merge_segments > 0
                else []
            ),
            *(
                [f"Sustained mode: target={target_throughput:.1f} ops/s, "
                 f"time_period={time_period}s, clients={search_clients}."]
                if sustained_mode
                else [f"Rounds mode: {rounds} round(s), {search_clients} client(s)."]
            ),
        ],
        out_dir=out_dir,
    )
    print(f"[bench-search] Wrote: {json_path}")
    print(f"[bench-search] Wrote: {md_path}")
    return 0
