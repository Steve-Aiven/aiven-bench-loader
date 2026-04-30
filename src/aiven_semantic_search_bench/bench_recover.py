"""
bench-recover: auto-pause wake-up latency benchmark.

What this measures
-------------------
Aiven for OpenSearch on the free (Hobbyist) tier automatically pauses the
service after a period of inactivity. The first request after a pause triggers
a warm-up cycle that can take 10-40 seconds. This benchmark:

1. Verifies the index has data (or seeds a small dataset from the corpus).
2. Picks a single test query embedding from the corpus (no Gemini call).
3. Issues one "warm" request to confirm the service is healthy and responsive.
4. Waits ``--idle-minutes`` minutes with a live countdown.
5. Issues the test query and measures exactly how long the first
   post-idle request takes.
6. Sends two more back-to-back requests to show the warm-up curve.

Migration story
---------------
On paid Aiven plans, auto-pause is disabled. If your application has irregular
traffic (overnight lulls, weekend drops), users may be silently absorbing
10-40 second "cold start" delays on the free tier.

Run on the Hobbyist plan with ``--label free-tier`` and record the cold row.
Then run on a Business plan with ``--label business-plan``. The cold-start
latency should collapse to a normal query latency, proving the upgrade
eliminated the problem.

Tip: use ``--idle-minutes 6`` or longer. The free tier typically pauses after
~5 minutes of no activity, so 6 minutes gives the service time to fully idle.
"""

from __future__ import annotations

import time

import numpy as np

from .config import Settings
from .corpus import CorpusBundle, load_corpus
from .opensearch_client import get_index_stats, get_opensearch_client, reset_index
from .report_context import benchmark_report_extras
from .reporter import write_report


def _seed_index(
    client,
    settings: Settings,
    bundle: CorpusBundle,
    doc_count: int,
    embed_dim: int,
) -> None:
    """Delete and recreate the index, then populate it from the corpus."""
    print(f"[bench-recover] Seeding index with {doc_count} docs from corpus...")
    reset_index(client, settings.opensearch_index, embed_dim=embed_dim)

    docs_df = bundle.docs.iloc[:doc_count]
    vectors = bundle.doc_vectors[:doc_count]

    body: list[dict] = []
    for (_, row), vec in zip(docs_df.iterrows(), vectors, strict=True):
        body.append({"index": {"_index": settings.opensearch_index, "_id": str(row["doc_id"])}})
        body.append({
            "description":        row["text"],
            "source":             row["source"],
            "description_vector": vec.tolist(),
        })
    resp = client.bulk(body=body)
    if resp.get("errors"):
        raise RuntimeError("Bulk indexing failed during seed phase")

    client.indices.refresh(index=settings.opensearch_index)
    print(f"[bench-recover] Index seeded with {doc_count} documents.")


def _knn_search(client, index: str, vector: np.ndarray) -> float:
    """Issue one k-NN search and return elapsed milliseconds."""
    t0 = time.perf_counter()
    client.search(
        index=index,
        body={
            "size": 5,
            "query": {
                "knn": {
                    "description_vector": {
                        "vector": vector.tolist(),
                        "k": 5,
                    }
                }
            },
        },
    )
    return (time.perf_counter() - t0) * 1000


def _countdown(total_seconds: int, description: str) -> None:
    """Block for ``total_seconds``, printing a live MM:SS countdown."""
    for remaining in range(total_seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(
            f"\r[bench-recover] {description} - {mins:02d}:{secs:02d} remaining...",
            end="",
            flush=True,
        )
        time.sleep(1)
    print()


def cmd_bench_recover(
    settings: Settings,
    *,
    idle_minutes: int,
    doc_count: int,
    embed_dim: int,
    corpus_dir: str,
    out_dir: str,
    label: str,
) -> int:
    print(f"[bench-recover] Loading corpus from {corpus_dir} at dim={embed_dim}...")
    bundle = load_corpus(corpus_dir, embed_dim)

    deployment_ctx, preflight_ctx = benchmark_report_extras(
        settings, settings.opensearch_uri
    )

    # Use a longer timeout - the first post-pause request may need to wait for
    # the service to finish waking up before it can respond.
    client = get_opensearch_client(settings.opensearch_uri, timeout=120)

    needs_seed = False
    if not client.indices.exists(index=settings.opensearch_index):
        needs_seed = True
    else:
        stats = get_index_stats(client, settings.opensearch_index)
        needs_seed = stats["doc_count"] == 0

    if needs_seed:
        _seed_index(client, settings, bundle, doc_count, embed_dim)
    else:
        existing_count = get_index_stats(client, settings.opensearch_index)["doc_count"]
        print(f"[bench-recover] Using existing index with {existing_count:,} documents.")

    # Pick a deterministic test query from the corpus. Index 0 keeps results
    # comparable across runs even when we reseed.
    test_query = str(bundle.queries.iloc[0]["text"])
    query_vector = bundle.query_vectors[0]
    print(f"[bench-recover] Test query: {test_query[:80]!r} (dim={embed_dim})")

    print("[bench-recover] Sending warm-up request to confirm service is healthy...")
    warmup_ms = _knn_search(client, settings.opensearch_index, query_vector)
    print(f"[bench-recover] Warm-up latency: {warmup_ms:.0f}ms - service is responsive.")

    print(
        f"\n[bench-recover] Starting {idle_minutes}-minute idle window.\n"
        f"  On the free tier the service auto-pauses after ~5 minutes of inactivity.\n"
        f"  On paid plans this countdown is just a wait - no pause will occur.\n"
    )
    _countdown(idle_minutes * 60, f"Idle window ({idle_minutes} min)")

    print("[bench-recover] Issuing first post-idle search request...")
    results: list[dict] = []
    for req_num in range(1, 4):
        latency_ms = _knn_search(client, settings.opensearch_index, query_vector)
        if req_num == 1:
            req_type = "cold (first after idle)"
        else:
            req_type = f"warm #{req_num - 1}"
        print(f"[bench-recover] Request {req_num} ({req_type}): {latency_ms:.0f}ms")
        results.append(
            {
                "request":    req_num,
                "type":       req_type,
                "latency_ms": round(latency_ms, 0),
            }
        )

    cold_ms = results[0]["latency_ms"]
    warm_ms = results[-1]["latency_ms"]
    print(
        f"\n[bench-recover] Cold start: {cold_ms:.0f}ms  |  "
        f"Warm (request 3): {warm_ms:.0f}ms  |  "
        f"Overhead: {cold_ms - warm_ms:.0f}ms"
    )

    json_path, md_path, _raw_path = write_report(
        "bench-recover",
        params={
            "plan_label":     label,
            "index":          settings.opensearch_index,
            "idle_minutes":   idle_minutes,
            "warmup_ms":      round(warmup_ms, 1),
            "test_query":     test_query,
            "embed_dim":      embed_dim,
            "corpus_preset":  bundle.manifest.get("preset"),
            "corpus_dir":     corpus_dir,
        },
        results=results,
        deployment=deployment_ctx,
        preflight=preflight_ctx,
        notes=[
            f"Request 1 is the 'cold start' after {idle_minutes} minutes of inactivity.",
            "Requests 2 and 3 show how quickly the service recovers once woken.",
            "On paid Aiven plans, auto-pause is disabled - request 1 should match requests 2-3.",
            "Test query and embedding come from the pre-built corpus (no Gemini calls).",
            f"Label '{label}' - re-run with --label after upgrading to compare cold-start behavior.",
        ],
        out_dir=out_dir,
    )
    print(f"[bench-recover] Wrote: {json_path}")
    print(f"[bench-recover] Wrote: {md_path}")
    return 0
