"""
bench-plan-change: query latency and error rate during a live plan change.

What this measures
-------------------
Runs a continuous query loop while the OpenSearch service plan is changed
out-of-band via the Aiven REST API. Captures every query's latency and
errors, tags each by phase (``before_change``, ``during_migration``,
``after_change``) and writes a JSON + Markdown report.

Why this matters for a migration story
---------------------------------------
The other benchmarks compare two static plans. This one shows the customer
exactly what their users experience *during* the upgrade itself: how long
queries take while OpenSearch is rebalancing, how many connections fail,
and how quickly latency settles after the new plan is applied.

When --and-back is supplied, the benchmark runs both directions:
``from_plan`` -> ``to_plan`` -> ``from_plan`` so a single run produces both
the upgrade cost and the downgrade cost.

Setup
-----
Requires AIVEN_API_TOKEN, AIVEN_PROJECT, AIVEN_SERVICE_NAME in the
environment, plus a pre-built corpus (run ``bench-build-corpus`` first).
The OpenSearch index must exist; it is auto-seeded with a small slice of
the corpus if empty.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import httpx
import numpy as np

from .aiven_client import AivenClient
from .config import Settings
from .corpus import CorpusBundle, load_corpus
from .opensearch_client import get_index_stats, get_opensearch_client, reset_index
from .report_context import benchmark_report_extras
from .reporter import write_report
from .stats import percentiles_ms

_DURING_PHASE_MAX_SECONDS = 900  # 15-minute safety bound per plan change
_PLAN_POLL_INTERVAL_S = 5.0      # how often to ask the Aiven API "are we done?"

# How many queries we cycle through during the live phase loop. We don't
# want to issue every one of the corpus's 100k queries during a 30-second
# warm-up phase; a small ring of varied queries is enough to drive
# representative load.
_QUERY_RING_SIZE = 50


def _ensure_seeded(
    client,
    settings: Settings,
    bundle: CorpusBundle,
    doc_count: int,
    embed_dim: int,
) -> int:
    """Return the doc count, seeding from the corpus if the index is empty."""
    if client.indices.exists(index=settings.opensearch_index):
        existing = get_index_stats(client, settings.opensearch_index)["doc_count"]
        if existing > 0:
            print(f"[bench-plan-change] Using existing index with {existing:,} documents.")
            return int(existing)

    print(f"[bench-plan-change] Seeding index with {doc_count} docs from corpus...")
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
    print(f"[bench-plan-change] Seeded {doc_count} documents.")
    return doc_count


def _one_query(client, index: str, vector: np.ndarray) -> tuple[float, str | None]:
    """Issue one k-NN search. Returns (latency_ms, error_kind_or_None)."""
    t0 = time.perf_counter()
    try:
        client.search(
            index=index,
            body={
                "size": 5,
                "query": {
                    "knn": {
                        "description_vector": {"vector": vector.tolist(), "k": 5}
                    }
                },
            },
        )
        return (time.perf_counter() - t0) * 1000, None
    except Exception as exc:
        return (time.perf_counter() - t0) * 1000, type(exc).__name__


def _run_phase(
    *,
    client,
    index: str,
    vectors: list[np.ndarray],
    phase: str,
    stop_when: Callable[[], bool],
    run_start: float,
    timeline: list[dict],
) -> None:
    """Issue queries continuously until ``stop_when`` returns True."""
    print(f"[bench-plan-change] Phase '{phase}' started.")
    issued = 0
    while not stop_when():
        for vec in vectors:
            latency_ms, error = _one_query(client, index, vec)
            timeline.append(
                {
                    "phase":      phase,
                    "elapsed_s":  round(time.monotonic() - run_start, 2),
                    "latency_ms": round(latency_ms, 1),
                    "error":      error,
                }
            )
            issued += 1
            if stop_when():
                break
    print(f"[bench-plan-change] Phase '{phase}' complete: {issued} queries issued.")


def _run_change_phase(
    *,
    client,
    index: str,
    vectors: list[np.ndarray],
    aiven: AivenClient,
    target_plan: str,
    run_start: float,
    timeline: list[dict],
) -> None:
    """
    Trigger a plan change, then keep querying until the service is RUNNING
    on the new plan (or the safety deadline elapses).
    """
    print(f"[bench-plan-change] Triggering plan change to '{target_plan}'...")
    try:
        aiven.update_plan(target_plan)
    except httpx.HTTPStatusError as exc:
        # Avoid forwarding the full response body (may contain account details).
        request_id = exc.response.headers.get("x-request-id", "n/a")
        raise RuntimeError(
            f"Aiven API rejected plan change to '{target_plan}': "
            f"HTTP {exc.response.status_code} (request-id: {request_id})"
        ) from exc

    deadline = time.monotonic() + _DURING_PHASE_MAX_SECONDS
    last_poll = 0.0
    settled = False

    def _is_settled() -> bool:
        nonlocal last_poll, settled
        if settled:
            return True
        if time.monotonic() >= deadline:
            print("[bench-plan-change] WARNING: hit max migration timeout.")
            return True
        if time.monotonic() - last_poll >= _PLAN_POLL_INTERVAL_S:
            last_poll = time.monotonic()
            try:
                state, plan = aiven.get_state_and_plan()
                print(f"[bench-plan-change]   poll: state={state} plan={plan}")
                if state == "RUNNING" and plan == target_plan:
                    settled = True
                    return True
            except httpx.HTTPError as exc:
                print(f"[bench-plan-change]   poll error (will retry): {exc}")
        return False

    _run_phase(
        client=client,
        index=index,
        vectors=vectors,
        phase=f"during_migration_to_{target_plan}",
        stop_when=_is_settled,
        run_start=run_start,
        timeline=timeline,
    )


def _run_timed_phase(
    *,
    client,
    index: str,
    vectors: list[np.ndarray],
    phase: str,
    duration_s: int,
    run_start: float,
    timeline: list[dict],
) -> None:
    deadline = time.monotonic() + duration_s
    _run_phase(
        client=client,
        index=index,
        vectors=vectors,
        phase=phase,
        stop_when=lambda: time.monotonic() >= deadline,
        run_start=run_start,
        timeline=timeline,
    )


def _phase_summary(phase: str, samples: list[dict]) -> dict[str, Any]:
    successes = [s["latency_ms"] for s in samples if s["error"] is None]
    errors = sum(1 for s in samples if s["error"] is not None)
    if successes:
        stats = percentiles_ms(successes)
        elapsed_seconds = (
            samples[-1]["elapsed_s"] - samples[0]["elapsed_s"]
            if len(samples) > 1 else 0.0
        )
        return {
            "phase":       phase,
            "duration_s":  round(elapsed_seconds, 1),
            "queries":     len(samples),
            "successful":  len(successes),
            "errors":      errors,
            "p50_ms":      round(stats["p50_ms"], 1),
            "p95_ms":      round(stats["p95_ms"], 1),
            "p99_ms":      round(stats["p99_ms"], 1),
            "max_ms":      round(stats["max_ms"], 1),
            "mean_ms":     round(stats["mean_ms"], 1),
        }
    return {
        "phase":       phase,
        "duration_s":  0.0,
        "queries":     len(samples),
        "successful":  0,
        "errors":      errors,
        "p50_ms":      0.0,
        "p95_ms":      0.0,
        "p99_ms":      0.0,
        "max_ms":      0.0,
        "mean_ms":     0.0,
    }


def cmd_bench_plan_change(
    settings: Settings,
    *,
    from_plan: str,
    to_plan: str,
    and_back: bool,
    pre_load_seconds: int,
    post_settle_seconds: int,
    doc_count: int,
    embed_dim: int,
    corpus_dir: str,
    label: str,
    out_dir: str,
) -> int:
    settings.require_aiven_api_credentials()

    aiven = AivenClient(
        project=settings.aiven_project,
        service_name=settings.aiven_service_name,
        api_token=settings.aiven_api_token,
    )

    print(f"[bench-plan-change] Checking current Aiven service state for "
          f"'{settings.aiven_service_name}'...")
    try:
        current_state, current_plan = aiven.get_state_and_plan()
    except httpx.HTTPError as exc:
        print(f"[bench-plan-change] ERROR contacting Aiven API: {exc}")
        return 1

    if current_state != "RUNNING":
        print(f"[bench-plan-change] ERROR: service is in state '{current_state}', "
              f"need 'RUNNING' to start.")
        return 1
    if current_plan != from_plan:
        print(f"[bench-plan-change] WARNING: service is on plan '{current_plan}' "
              f"but --from-plan is '{from_plan}'. Continuing anyway.")

    print(f"[bench-plan-change] Loading corpus from {corpus_dir} at dim={embed_dim}...")
    bundle = load_corpus(corpus_dir, embed_dim)

    deployment_ctx, preflight_ctx = benchmark_report_extras(
        settings,
        settings.opensearch_uri,
        aiven_api_token=settings.aiven_api_token,
        aiven_project=settings.aiven_project,
    )

    client = get_opensearch_client(settings.opensearch_uri, timeout=120)
    seeded_doc_count = _ensure_seeded(client, settings, bundle, doc_count, embed_dim)

    # Use the first _QUERY_RING_SIZE pre-embedded queries from the corpus.
    # The same ring is reused across phases so latency comparisons are
    # apples-to-apples within the run.
    ring_n = min(_QUERY_RING_SIZE, len(bundle.query_vectors))
    query_vectors: list[np.ndarray] = list(bundle.query_vectors[:ring_n])
    print(f"[bench-plan-change] Using {ring_n} pre-embedded queries from corpus.")

    timeline: list[dict] = []
    run_start = time.monotonic()

    _run_timed_phase(
        client=client, index=settings.opensearch_index, vectors=query_vectors,
        phase="before_change", duration_s=pre_load_seconds,
        run_start=run_start, timeline=timeline,
    )

    _run_change_phase(
        client=client, index=settings.opensearch_index, vectors=query_vectors,
        aiven=aiven, target_plan=to_plan,
        run_start=run_start, timeline=timeline,
    )

    _run_timed_phase(
        client=client, index=settings.opensearch_index, vectors=query_vectors,
        phase="after_change", duration_s=post_settle_seconds,
        run_start=run_start, timeline=timeline,
    )

    if and_back:
        _run_change_phase(
            client=client, index=settings.opensearch_index, vectors=query_vectors,
            aiven=aiven, target_plan=from_plan,
            run_start=run_start, timeline=timeline,
        )
        _run_timed_phase(
            client=client, index=settings.opensearch_index, vectors=query_vectors,
            phase="after_revert", duration_s=post_settle_seconds,
            run_start=run_start, timeline=timeline,
        )

    by_phase: dict[str, list[dict]] = {}
    for sample in timeline:
        by_phase.setdefault(sample["phase"], []).append(sample)
    summaries = [_phase_summary(phase, by_phase[phase]) for phase in by_phase]

    print("\n[bench-plan-change] Phase summary:")
    for s in summaries:
        print(
            f"  {s['phase']:<32} queries={s['queries']:>4}  "
            f"errors={s['errors']:>3}  "
            f"p50={s['p50_ms']:>7.1f}ms  p95={s['p95_ms']:>7.1f}ms  "
            f"p99={s['p99_ms']:>7.1f}ms"
        )

    json_path, md_path, _raw_path = write_report(
        "bench-plan-change",
        params={
            "plan_label":           label,
            "index":                settings.opensearch_index,
            "indexed_doc_count":    seeded_doc_count,
            "from_plan":            from_plan,
            "to_plan":              to_plan,
            "and_back":             and_back,
            "pre_load_seconds":     pre_load_seconds,
            "post_settle_seconds":  post_settle_seconds,
            "service":              settings.aiven_service_name,
            "embed_dim":            embed_dim,
            "corpus_preset":        bundle.manifest.get("preset"),
            "corpus_dir":           corpus_dir,
            "embed_model":          bundle.manifest.get("embed_model"),
            "timeline":             timeline,
        },
        results=summaries,
        deployment=deployment_ctx,
        preflight=preflight_ctx,
        notes=[
            f"Phase 'during_migration_to_{to_plan}' includes the full Aiven rebalance window.",
            "Errors and elevated p95 in the 'during_migration' row are the upgrade tax.",
            "after_change p50/p95 should match the destination plan's steady-state latency.",
            "Queries are pre-embedded from the corpus; no Gemini calls during the run.",
            f"Label '{label}' - re-run with different --from/--to plans to compare paths.",
        ],
        out_dir=out_dir,
    )
    print(f"[bench-plan-change] Wrote: {json_path}")
    print(f"[bench-plan-change] Wrote: {md_path}")
    return 0
