"""
bench-index: indexing throughput at varying batch sizes.

What this measures
-------------------
For each batch size in the sweep, this benchmark:
1. Resets the OpenSearch index to an empty state (using the provided KnnSpec).
2. Indexes the same N documents using ``_bulk`` with ``batch_size`` operations
   per request.
3. Times each individual ``_bulk`` request.
4. Reports docs/sec, p50/p95/p99 request latency, and total wall-clock time.

Why batch size matters
-----------------------
OpenSearch ingestion has fixed per-request overhead (HTTP, auth, parsing).
With batch_size=1 you pay that overhead for every document. With larger
batches the overhead amortises across many docs, but very large batches can
hit request-size limits or stress the receiving node's memory. This sweep
gives you the curve so you can pick a value that fits your real workload.

Why we load from a pre-built corpus
-------------------------------------
Documents and embeddings come from a corpus built once by
``bench-build-corpus``. That means this command makes ZERO calls to Vertex
AI — it only talks to OpenSearch. The numbers it reports reflect Aiven
OpenSearch + network RTT exclusively.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .clickhouse_sink import get_sink
from .config import Settings
from .corpus import load_corpus
from .opensearch_client import KnnSpec, encode_vector, get_opensearch_client, reset_index
from .report_context import benchmark_report_extras
from .reporter import raw_samples_enabled, write_report
from .stats import chunked, percentiles_ms, stopwatch

_DEFAULT_SPEC = KnnSpec(embed_dim=768)



def _bulk_index_request(
    client,
    index: str,
    batch: list[tuple[str, str, str, np.ndarray]],
    *,
    spec: KnnSpec,
) -> None:
    """Send one ``_bulk`` request containing N (action, source) pairs."""
    body: list[dict] = []
    for doc_id, description, source, vector in batch:
        body.append({"index": {"_index": index, "_id": doc_id}})
        doc: dict = {
            "description":        description,
            "source":             source,
            "description_vector": encode_vector(vector, spec.data_type),
        }
        if spec.with_text:
            doc["content"] = description
        body.append(doc)

    resp = client.bulk(body=body)
    if resp.get("errors"):
        first_error = next(
            (
                item["index"]["error"]
                for item in resp.get("items", [])
                if "error" in item.get("index", {})
            ),
            None,
        )
        raise RuntimeError(f"_bulk reported errors: {first_error}")


def cmd_bench_index(
    settings: Settings,
    *,
    doc_count: int,
    batch_sizes: list[int],
    embed_dim: int,
    spec: KnnSpec | None = None,
    corpus_dir: str,
    label: str = "unlabeled",
    out_dir: str,
    opensearch_uri: str | None = None,
) -> int:
    """
    Run the indexing throughput benchmark.

    Parameters
    ----------
    opensearch_uri:
        When provided (by the job runner), overrides ``settings.opensearch_uri``
        so each job targets the correct version-specific service.
    spec:
        Full k-NN index specification. Defaults to a plain float/HNSW/faiss
        index at the requested ``embed_dim`` if not provided.
    """
    if doc_count <= 0:
        raise ValueError("doc-count must be > 0")
    if not batch_sizes:
        raise ValueError("batch-sizes must contain at least one value")

    uri = opensearch_uri or settings.opensearch_uri
    knn = spec or KnnSpec(embed_dim=embed_dim)
    deployment_ctx, preflight_ctx = benchmark_report_extras(settings, uri)

    print(f"[bench-index] Loading corpus from {corpus_dir} at dim={embed_dim}...")
    bundle = load_corpus(corpus_dir, embed_dim)
    available = len(bundle.docs)
    if doc_count > available:
        print(
            f"[bench-index] WARNING: requested {doc_count} docs, "
            f"corpus has {available}. Using {available}."
        )
        doc_count = available

    docs_df = bundle.docs.iloc[:doc_count]
    vectors = bundle.doc_vectors[:doc_count]
    print(
        f"[bench-index] Using {doc_count} documents at dim={embed_dim} "
        f"({bundle.manifest.get('preset', 'unknown')} corpus, "
        f"sourced from {bundle.manifest.get('doc_sources')})."
    )
    print(f"[bench-index] k-NN spec: {knn.label()}")

    payload = list(
        zip(
            docs_df["doc_id"].astype(str).tolist(),
            docs_df["text"].tolist(),
            docs_df["source"].astype(str).tolist(),
            list(vectors),
            strict=True,
        )
    )

    # Inject synthetic metadata if the spec calls for it and the corpus has it.
    if knn.with_metadata and "category" in docs_df.columns:
        payload_meta = []
        for (doc_id, desc, src, vec), row in zip(
            payload, docs_df.itertuples(index=False)
        ):
            payload_meta.append((doc_id, desc, src, vec, row))
        payload_with_meta = payload_meta
    else:
        payload_with_meta = None

    client = get_opensearch_client(uri)
    sink = get_sink()

    results: list[dict] = []
    raw_bulk: dict[str, list[dict[str, Any]]] = {}
    save_raw = raw_samples_enabled()

    for bs in batch_sizes:
        print(f"[bench-index] Resetting index and indexing at batch_size={bs}...")
        reset_index(client, settings.opensearch_index, spec=knn)

        bs_label = {"batch_size": str(bs)}
        request_latencies_ms: list[float] = []
        bulk_samples: list[dict[str, Any]] = []
        bulk_failures = 0
        req_idx = 0
        with stopwatch() as sw:
            for batch in chunked(payload, bs):
                t0 = time.perf_counter()
                ok_req = True
                try:
                    _bulk_index_request(
                        client, settings.opensearch_index, batch, spec=knn
                    )
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    request_latencies_ms.append(elapsed_ms)
                    sink.metric("index_bulk_ms", elapsed_ms, labels=bs_label)
                except Exception:
                    ok_req = False
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    request_latencies_ms.append(elapsed_ms)
                    bulk_failures += 1
                    sink.metric("bulk_failures_total", 1.0, labels=bs_label)
                if save_raw:
                    bulk_samples.append(
                        {
                            "request_index": req_idx,
                            "latency_ms": round(elapsed_ms, 3),
                            "ok": ok_req,
                            "docs_in_batch": len(batch),
                        }
                    )
                req_idx += 1
        if save_raw:
            raw_bulk[str(bs)] = bulk_samples

        client.indices.refresh(index=settings.opensearch_index)
        wall_s = sw["elapsed_s"]
        docs_per_sec = doc_count / wall_s if wall_s > 0 else 0.0

        latency_stats = percentiles_ms(request_latencies_ms)
        results.append(
            {
                "batch_size":   bs,
                "documents":    doc_count,
                "wall_seconds": round(wall_s, 3),
                "docs_per_sec": round(docs_per_sec, 1),
                "requests":     latency_stats["count"],
                "p50_ms":       round(latency_stats["p50_ms"], 1),
                "p95_ms":       round(latency_stats["p95_ms"], 1),
                "p99_ms":       round(latency_stats["p99_ms"], 1),
                "max_ms":       round(latency_stats["max_ms"], 1),
                "mean_ms":      round(latency_stats["mean_ms"], 1),
            }
        )
        sink.metric("docs_per_sec", float(docs_per_sec), labels=bs_label)
        sink.metric("index_p50_ms", float(latency_stats["p50_ms"]), labels=bs_label)
        sink.metric("index_p95_ms", float(latency_stats["p95_ms"]), labels=bs_label)
        sink.metric("index_p99_ms", float(latency_stats["p99_ms"]), labels=bs_label)
        print(
            f"[bench-index] batch_size={bs}: "
            f"{docs_per_sec:.1f} docs/sec, p50={latency_stats['p50_ms']:.1f}ms, "
            f"p95={latency_stats['p95_ms']:.1f}ms"
            + (f", FAILURES={bulk_failures}" if bulk_failures else "")
        )

    raw_payload = {"bulk_requests_by_batch_size": raw_bulk} if save_raw and raw_bulk else None

    json_path, md_path, raw_path = write_report(
        "bench-index",
        params={
            "plan_label":    label,
            "documents":     doc_count,
            "batch_sizes":   batch_sizes,
            "embed_dim":     embed_dim,
            "knn_spec":      knn.to_dict(),
            "corpus_preset": bundle.manifest.get("preset"),
            "corpus_dir":    corpus_dir,
            "embed_model":   bundle.manifest.get("embed_model"),
            "index":         settings.opensearch_index,
        },
        results=results,
        deployment=deployment_ctx,
        preflight=preflight_ctx,
        notes=[
            "Documents come from the pre-built corpus; embeddings are NOT recomputed at runtime.",
            "Latency is per `_bulk` HTTP request, including OpenSearch processing time.",
            "Index is reset between batch sizes to avoid cache-warming effects.",
            f"Embedding dimension: {embed_dim} (Matryoshka-truncated from "
            f"{bundle.source_dim} stored).",
            f"k-NN spec: {knn.label()}",
            f"Label '{label}' — re-run with a different label to compare configurations.",
        ],
        out_dir=out_dir,
        raw_data=raw_payload,
    )
    print(f"[bench-index] Wrote: {json_path}")
    print(f"[bench-index] Wrote: {md_path}")
    if raw_path:
        print(f"[bench-index] Wrote: {raw_path}")
    return 0
