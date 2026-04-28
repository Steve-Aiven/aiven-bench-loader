"""
bench-stress: high-workload chaos / saturation test.

Intentionally drives the target OpenSearch service past its steady-state
capacity to discover failure modes: JVM OOM, GC storms, merge storms,
request timeouts, node restarts, or disk pressure.

Two concurrent thread pools run simultaneously:

  Index pool (``index_clients`` threads):
    Each thread round-robin bulk-indexes corpus documents continuously.
    Uses overwrite semantics (same doc IDs), so disk usage is bounded.
    Each thread gets its own OpenSearch client.

  Search pool (``search_clients`` threads):
    Each thread issues k-NN searches at full throttle with no rate limit.
    High ``k`` + large ef_search maximises HNSW traversal cost and JVM
    heap pressure.

Every ``sample_interval`` seconds a cluster-health snapshot is logged.

Duration
--------
The test runs for at least ``duration`` seconds.  If a plan change is
configured, after the change is triggered the test continues until the
service transitions back to RUNNING on the new plan, then runs for an
additional ``post_settle_s`` seconds (default 60) so that the full
recovery arc is captured.

Thanos / metrics
-----------------
If a Thanos Prometheus-compatible URI is supplied (``thanos_uri``), the
test queries JVM, GC, indexing, and search metrics from Thanos for the
full test window and appends them to the report as ``thanos_metrics``.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse, unquote

import numpy as np

from .config import Settings
from .corpus import load_corpus
from .opensearch_client import (
    KnnSpec,
    get_index_stats,
    get_opensearch_client,
)
from .reporter import write_report
from .stats import percentiles_ms

_SAMPLE_INTERVAL_S = 10
_ERROR_WARN_THRESHOLD_PCT = 20.0
_PLAN_POLL_INTERVAL_S = 15  # how often to poll Aiven API for service state


# ── Deadline manager ──────────────────────────────────────────────────────────

class _Deadline:
    """Thread-safe monotonic deadline."""

    def __init__(self, initial_s: float) -> None:
        self._t = time.perf_counter() + initial_s
        self._lock = threading.Lock()

    def extend_to(self, t: float) -> float:
        """Set deadline to max(current, t) — only moves deadline later."""
        with self._lock:
            if t > self._t:
                self._t = t
            return self._t

    def trim_to(self, t: float) -> float:
        """Set deadline to exactly t, regardless of current value."""
        with self._lock:
            self._t = t
            return self._t

    def remaining(self) -> float:
        with self._lock:
            return self._t - time.perf_counter()

    def expired(self) -> bool:
        return self.remaining() <= 0


# ── Thanos / Prometheus query ─────────────────────────────────────────────────

_THANOS_METRICS = [
    ("jvm_heap_pct",        "opensearch_jvm_mem_heap_used_percent"),
    ("gc_old_count",        "rate(opensearch_jvm_gc_collection_count_total{gc='old'}[1m])"),
    ("gc_young_count",      "rate(opensearch_jvm_gc_collection_count_total{gc='young'}[1m])"),
    ("index_rate",          "rate(opensearch_indices_indexing_index_total[1m])"),
    ("search_rate",         "rate(opensearch_indices_search_query_total[1m])"),
    ("search_latency_ms",   "opensearch_indices_search_query_time_in_millis / clamp_min(opensearch_indices_search_query_total, 1)"),
]


def _query_thanos(thanos_uri: str, start_ts: float, end_ts: float) -> dict[str, list[dict]]:
    """
    Query each metric in ``_THANOS_METRICS`` from the Thanos Prometheus API.

    ``thanos_uri`` is the full service URI including credentials, e.g.
    ``https://user:password@host:port``.  The Prometheus query_range
    endpoint is appended automatically.

    Returns a dict mapping metric name → list of {timestamp, value} dicts.
    Data older than ~1 min may not be visible yet; the caller should allow
    a short settle time before querying.
    """
    try:
        import httpx as _httpx  # imported here to keep it an optional dep at module level
    except ImportError:
        print("[bench-stress] httpx not available — skipping Thanos query.")
        return {}

    parsed = urlparse(thanos_uri)
    # Strip any trailing path (Aiven returns query_frontend_uri with /api/v1/ suffix)
    base = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
    auth = (unquote(parsed.username), unquote(parsed.password)) if parsed.username else None

    results: dict[str, list[dict]] = {}
    step = max(int((end_ts - start_ts) / 200), _SAMPLE_INTERVAL_S)

    for name, query in _THANOS_METRICS:
        try:
            with _httpx.Client(timeout=30, verify=True) as client:
                resp = client.get(
                    f"{base}/api/v1/query_range",
                    auth=auth,
                    params={
                        "query": query,
                        "start": int(start_ts),
                        "end":   int(end_ts),
                        "step":  step,
                    },
                )
            resp.raise_for_status()
            raw = resp.json()
            series = raw.get("data", {}).get("result", [])
            points: list[dict] = []
            for s in series:
                for ts, val in s.get("values", []):
                    try:
                        points.append({"t": ts, "v": float(val)})
                    except (ValueError, TypeError):
                        pass
            results[name] = points
            print(f"[bench-stress] Thanos {name}: {len(points)} data points.")
        except Exception as exc:
            print(f"[bench-stress] Thanos query '{name}' failed: {exc}")
            results[name] = []

    return results


# ── Plan change thread ────────────────────────────────────────────────────────

def _plan_change_thread(
    *,
    aiven_api_token: str,
    aiven_project: str,
    aiven_service_name: str,
    plan_change_target: str,
    plan_change_after_s: int,
    post_settle_s: int,
    deadline: _Deadline,
    stop_event: threading.Event,
    interval_rows: list[dict],
    t_start: float,
    admin_client: Any,
) -> None:
    """
    1. Wait ``plan_change_after_s`` seconds.
    2. Trigger a plan change via the Aiven API.
    3. Poll every ``_PLAN_POLL_INTERVAL_S`` seconds until the service returns
       to RUNNING on the new plan.
    4. Extend the test deadline to ``settled_time + post_settle_s`` so the
       full recovery arc is captured.
    """
    stop_event.wait(timeout=plan_change_after_s)
    if stop_event.is_set():
        return

    elapsed = time.perf_counter() - t_start
    print(
        f"\n[bench-stress] >>> Triggering plan change → '{plan_change_target}' "
        f"at t={elapsed:.0f}s via Aiven API <<<\n"
    )
    try:
        from .aiven_client import AivenClient
        ac = AivenClient(
            project=aiven_project,
            service_name=aiven_service_name,
            api_token=aiven_api_token,
        )
        ac.update_plan(plan_change_target)
    except Exception as exc:
        err_msg = str(exc)
        print(
            f"\n[bench-stress] ✗ Plan change to '{plan_change_target}' FAILED: {err_msg}\n"
            f"[bench-stress]   Common causes:\n"
            f"[bench-stress]   • Node quota exceeded (upgrading to more nodes requires"
            f" project quota headroom — check other running services)\n"
            f"[bench-stress]   • Plan not valid for this cloud region\n"
            f"[bench-stress]   • Service currently REBALANCING (another change in progress)\n"
            f"[bench-stress]   Test will continue to its original deadline.\n"
        )
        interval_rows.append({
            "event":      "plan_change_failed",
            "interval_s": round(elapsed, 1),
            "new_plan":   plan_change_target,
            "error":      err_msg,
        })
        return

    interval_rows.append({
        "event":      "plan_change_triggered",
        "interval_s": round(elapsed, 1),
        "new_plan":   plan_change_target,
    })

    # Extend the deadline immediately so the test keeps running while the plan
    # change completes.  Premium plan changes involving node additions can take
    # 30–60 minutes.  We set a 2-hour cap here; it will be trimmed down to
    # settled_time + post_settle_s as soon as the service reaches RUNNING.
    _MAX_PLAN_CHANGE_WAIT_S = 7200
    deadline.extend_to(time.perf_counter() + _MAX_PLAN_CHANGE_WAIT_S)
    print(
        "[bench-stress] Plan change accepted — service is REBALANCING.\n"
        f"[bench-stress] Deadline extended (cap: {_MAX_PLAN_CHANGE_WAIT_S // 60}h) — "
        f"will trim to {post_settle_s}s after service returns to RUNNING."
    )

    # Poll until the service is RUNNING on the new plan.
    # For same-machine-type plan changes (e.g. premium-6x → premium-9x),
    # Aiven rolls in additional nodes and migrates shards before decommissioning
    # old nodes — the cluster stays available throughout.  We log node-count
    # changes so the timeline shows exactly when nodes were added/removed.
    prev_node_count: int | None = None

    # We do NOT exit on stop_event here: the deadline has already been extended
    # by up to 2h so stop_event won't fire until the plan change settles or the
    # cap is reached.  Use stop_event.wait() as the sleep so we can react to an
    # external Ctrl-C quickly.
    while not stop_event.wait(timeout=_PLAN_POLL_INTERVAL_S):
        try:
            from .aiven_client import AivenClient
            ac = AivenClient(
                project=aiven_project,
                service_name=aiven_service_name,
                api_token=aiven_api_token,
            )
            state, current_plan = ac.get_state_and_plan()
        except Exception as exc:
            print(f"[bench-stress] Poll error: {exc}")
            continue

        elapsed = time.perf_counter() - t_start

        # Sample OpenSearch node count to detect node additions/removals.
        try:
            h = admin_client.cluster.health()
            node_count = h.get("number_of_nodes", -1)
            relocating = h.get("relocating_shards", 0)
        except Exception:
            node_count = -1
            relocating = 0

        node_note = ""
        if node_count >= 0:
            if prev_node_count is not None and node_count != prev_node_count:
                change = node_count - prev_node_count
                direction = "added" if change > 0 else "removed"
                node_note = f"  *** NODE COUNT {prev_node_count} → {node_count} ({abs(change)} {direction}) ***"
                interval_rows.append({
                    "event":            "node_count_change",
                    "interval_s":       round(elapsed, 1),
                    "nodes_before":     prev_node_count,
                    "nodes_after":      node_count,
                    "nodes_delta":      change,
                    "relocating_shards": relocating,
                })
            prev_node_count = node_count

        print(
            f"[bench-stress] Aiven: state={state!r} plan={current_plan!r} "
            f"| OS: nodes={node_count} relocating={relocating} "
            f"(t={elapsed:.0f}s)"
            + node_note
        )

        if state == "RUNNING" and current_plan == plan_change_target:
            interval_rows.append({
                "event":      "plan_change_settled",
                "interval_s": round(elapsed, 1),
                "new_plan":   current_plan,
                "final_nodes": node_count,
            })
            # Trim the deadline to exactly settled_time + post_settle_s.
            # We use trim_to (not extend_to) because we previously set a 2h
            # cap on trigger — this resets it to the correct shorter window.
            new_deadline = time.perf_counter() + post_settle_s
            deadline.trim_to(new_deadline)
            print(
                f"[bench-stress] Service RUNNING on new plan '{current_plan}' "
                f"with {node_count} nodes. "
                f"Test will continue {post_settle_s}s more (t={elapsed:.0f}s + {post_settle_s}s)."
            )
            return


# ── Worker: continuous bulk indexing ─────────────────────────────────────────

def _index_worker(
    uri: str,
    index: str,
    docs: list[tuple[str, str, str, np.ndarray]],
    *,
    batch_size: int,
    stop_event: threading.Event,
    counters: dict[str, int],
    lock: threading.Lock,
    spec: KnnSpec,
    thread_offset: int,
) -> None:
    client = get_opensearch_client(uri, timeout=120)
    n = len(docs)
    idx = thread_offset % n

    while not stop_event.is_set():
        batch = [docs[(idx + i) % n] for i in range(batch_size)]
        idx = (idx + batch_size) % n

        body: list[dict] = []
        for doc_id, text, source, vector in batch:
            body.append({"index": {"_index": index, "_id": doc_id}})
            doc: dict = {
                "description":        text,
                "source":             source,
                "description_vector": vector.tolist(),
            }
            if spec.with_text:
                doc["content"] = text
            body.append(doc)

        try:
            resp = client.bulk(body=body)
            had_errors = 1 if resp.get("errors") else 0
            with lock:
                counters["index_ops"] += len(batch)
                if had_errors:
                    n_err = sum(
                        1 for item in resp.get("items", [])
                        if "error" in item.get("index", {})
                    )
                    counters["index_errors"] += n_err or 1
        except Exception:
            with lock:
                counters["index_errors"] += batch_size


# ── Worker: continuous k-NN search ───────────────────────────────────────────

def _search_worker(
    uri: str,
    index: str,
    query_vectors: np.ndarray,
    *,
    k: int,
    ef_search: int,
    stop_event: threading.Event,
    counters: dict[str, int],
    lock: threading.Lock,
    lats: list[float],
    thread_offset: int,
) -> None:
    client = get_opensearch_client(uri, timeout=120)
    n = len(query_vectors)
    idx = thread_offset % n

    while not stop_event.is_set():
        v = query_vectors[idx % n]
        idx += 1
        t0 = time.perf_counter()
        try:
            client.search(
                index=index,
                body={
                    "size": k,
                    "query": {
                        "knn": {
                            "description_vector": {
                                "vector": v.tolist(),
                                "k": k,
                                "filter": {"match_all": {}},
                            }
                        }
                    },
                },
                params={"search_type": "query_then_fetch"},
            )
            lat_ms = (time.perf_counter() - t0) * 1000
            with lock:
                counters["search_ops"] += 1
                lats.append(lat_ms)
        except Exception:
            with lock:
                counters["search_errors"] += 1


# ── Cluster health snapshot ───────────────────────────────────────────────────

def _health_snapshot(client) -> dict[str, Any]:
    """
    Capture cluster health including node count.

    For Aiven plan changes between plans of the same machine type (e.g.
    premium-6x → premium-9x), Aiven adds new nodes to the cluster and
    migrates shards onto them before decommissioning old nodes.  This is
    NOT a full rebuild — the cluster stays available throughout.  The
    ``nodes`` field here lets you observe nodes being added and removed
    in real time.
    """
    try:
        h = client.cluster.health()
        return {
            "status":               h.get("status", "unknown"),
            "nodes":                h.get("number_of_nodes", -1),
            "data_nodes":           h.get("number_of_data_nodes", -1),
            "active_shards":        h.get("active_shards", -1),
            "relocating_shards":    h.get("relocating_shards", 0),
            "initializing_shards":  h.get("initializing_shards", 0),
            "unassigned_shards":    h.get("unassigned_shards", 0),
            "pending_tasks":        h.get("number_of_pending_tasks", 0),
        }
    except Exception:
        return {
            "status": "unreachable", "nodes": -1, "data_nodes": -1,
            "active_shards": -1, "relocating_shards": -1,
            "initializing_shards": -1, "unassigned_shards": -1, "pending_tasks": -1,
        }


# ── Public entry point ────────────────────────────────────────────────────────

def cmd_bench_stress(
    settings: Settings,
    *,
    embed_dim: int,
    spec: KnnSpec | None = None,
    corpus_dir: str,
    out_dir: str,
    label: str,
    opensearch_uri: str | None = None,
    index_clients: int = 8,
    search_clients: int = 16,
    duration: int = 120,
    batch_size: int = 100,
    k: int = 100,
    # Extended duration after plan change settles
    post_settle_s: int = 60,
    # Optional mid-run plan change via Aiven API
    plan_change_target: str = "",
    plan_change_after_s: int = 60,
    aiven_api_token: str = "",
    aiven_project: str = "",
    aiven_service_name: str = "",
    # Optional Thanos / Prometheus metrics query
    thanos_uri: str = "",
) -> int:
    uri = opensearch_uri or settings.opensearch_uri
    knn = spec or KnnSpec(embed_dim=embed_dim)
    admin_client = get_opensearch_client(uri, timeout=30)

    # ── Preflight ──────────────────────────────────────────────────────────
    if not admin_client.indices.exists(index=settings.opensearch_index):
        print(
            f"[bench-stress] ERROR: index '{settings.opensearch_index}' does not exist.\n"
            f"[bench-stress] Run a bench-index job first."
        )
        return 1

    idx_stats = get_index_stats(admin_client, settings.opensearch_index)
    doc_count = idx_stats["doc_count"]
    if doc_count == 0:
        print(
            f"[bench-stress] ERROR: index '{settings.opensearch_index}' is empty.\n"
            f"[bench-stress] Run a bench-index job first."
        )
        return 1

    plan_change_active = bool(
        plan_change_target and aiven_api_token and aiven_project and aiven_service_name
    )
    print(
        f"[bench-stress] Target: '{settings.opensearch_index}' ({doc_count:,} docs)\n"
        f"[bench-stress] Config: {index_clients} index thread(s) (batch={batch_size}) + "
        f"{search_clients} search thread(s) (k={k}, ef_search={knn.ef_search}) "
        f"× min {duration}s"
        + (
            f"\n[bench-stress] Plan change: → '{plan_change_target}' after {plan_change_after_s}s; "
            f"test extends {post_settle_s}s past settle"
            if plan_change_active else ""
        )
        + (f"\n[bench-stress] Thanos metrics: {thanos_uri[:40]}…" if thanos_uri else "")
    )

    # ── Load corpus ────────────────────────────────────────────────────────
    print(f"[bench-stress] Loading corpus from {corpus_dir} (dim={embed_dim})…")
    bundle = load_corpus(corpus_dir, embed_dim)
    docs: list[tuple[str, str, str, np.ndarray]] = list(zip(
        bundle.docs["doc_id"].astype(str).tolist(),
        bundle.docs["text"].tolist(),
        bundle.docs["source"].astype(str).tolist(),
        list(bundle.doc_vectors),
        strict=True,
    ))
    query_vectors = bundle.query_vectors
    print(
        f"[bench-stress] Corpus: {len(docs):,} docs / {len(query_vectors):,} queries. "
        "Starting stress run…"
    )

    # ── Shared state ───────────────────────────────────────────────────────
    counters: dict[str, int] = {
        "index_ops": 0, "index_errors": 0,
        "search_ops": 0, "search_errors": 0,
    }
    lats: list[float] = []
    lock = threading.Lock()
    stop_event = threading.Event()
    deadline = _Deadline(duration)
    interval_rows: list[dict] = []
    t_start = time.perf_counter()
    t_start_unix = time.time()
    first_error_s: float | None = None
    prev: dict[str, int] = {k2: 0 for k2 in counters}

    # ── Sampler thread ─────────────────────────────────────────────────────
    def _sampler() -> None:
        nonlocal first_error_s

        while not stop_event.is_set():
            time.sleep(_SAMPLE_INTERVAL_S)
            if stop_event.is_set():
                break

            elapsed = time.perf_counter() - t_start
            with lock:
                cur = dict(counters)
                snap_lats = list(lats)

            delta_io  = cur["index_ops"]     - prev["index_ops"]
            delta_ie  = cur["index_errors"]  - prev["index_errors"]
            delta_so  = cur["search_ops"]    - prev["search_ops"]
            delta_se  = cur["search_errors"] - prev["search_errors"]
            prev.update(cur)

            interval_ops  = delta_io + delta_so
            interval_errs = delta_ie + delta_se
            err_rate_pct  = (interval_errs / max(interval_ops + interval_errs, 1)) * 100

            if interval_errs > 0 and first_error_s is None:
                first_error_s = elapsed

            health     = _health_snapshot(admin_client)
            search_p95: float | None = None
            if snap_lats:
                search_p95 = round(percentiles_ms(snap_lats)["p95_ms"], 1)

            remaining = deadline.remaining()
            row: dict = {
                "interval_s":          round(elapsed, 1),
                "index_ops":           delta_io,
                "index_errors":        delta_ie,
                "search_ops":          delta_so,
                "search_errors":       delta_se,
                "error_rate_pct":      round(err_rate_pct, 1),
                "search_p95_ms":       search_p95,
                "cluster_status":      health["status"],
                "nodes":               health["nodes"],
                "data_nodes":          health["data_nodes"],
                "relocating_shards":   health["relocating_shards"],
                "initializing_shards": health["initializing_shards"],
                "unassigned_shards":   health["unassigned_shards"],
                "active_shards":       health["active_shards"],
                "pending_tasks":       health["pending_tasks"],
            }
            interval_rows.append(row)

            status_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(health["status"], "⚫")
            relocating_note = (
                f" relocating={health['relocating_shards']}"
                if health["relocating_shards"] > 0 else ""
            )
            initializing_note = (
                f" initializing={health['initializing_shards']}"
                if health["initializing_shards"] > 0 else ""
            )
            warn = "  ⚠️  HIGH ERRORS" if err_rate_pct > _ERROR_WARN_THRESHOLD_PCT else ""
            print(
                f"[bench-stress] t={elapsed:5.0f}s "
                f"(+{remaining:.0f}s left) | "
                f"cluster={health['status']}{status_icon} "
                f"nodes={health['nodes']} | "
                f"idx {delta_io:5}ops {delta_ie:3}err | "
                f"srch {delta_so:5}ops {delta_se:3}err | "
                f"err={err_rate_pct:.1f}% | "
                f"p95={search_p95 or '—':>7}ms"
                + relocating_note + initializing_note + warn
            )

    sampler_thread = threading.Thread(target=_sampler, daemon=True, name="stress-sampler")
    sampler_thread.start()

    # ── Plan-change thread ─────────────────────────────────────────────────
    pc_thread: threading.Thread | None = None
    if plan_change_active:
        pc_thread = threading.Thread(
            target=_plan_change_thread,
            kwargs=dict(
                aiven_api_token=aiven_api_token,
                aiven_project=aiven_project,
                aiven_service_name=aiven_service_name,
                plan_change_target=plan_change_target,
                plan_change_after_s=plan_change_after_s,
                post_settle_s=post_settle_s,
                deadline=deadline,
                stop_event=stop_event,
                interval_rows=interval_rows,
                t_start=t_start,
                admin_client=admin_client,
            ),
            daemon=True,
            name="stress-planchange",
        )
        pc_thread.start()

    # ── Worker pools ───────────────────────────────────────────────────────
    print("[bench-stress] Workers online…")
    n_docs  = len(docs)
    n_qvecs = len(query_vectors)

    with ThreadPoolExecutor(max_workers=index_clients + search_clients) as pool:
        futs = []
        for i in range(index_clients):
            futs.append(pool.submit(
                _index_worker, uri, settings.opensearch_index, docs,
                batch_size=batch_size, stop_event=stop_event,
                counters=counters, lock=lock, spec=knn,
                thread_offset=i * (n_docs // max(index_clients, 1)),
            ))
        for i in range(search_clients):
            futs.append(pool.submit(
                _search_worker, uri, settings.opensearch_index, query_vectors,
                k=k, ef_search=knn.ef_search, stop_event=stop_event,
                counters=counters, lock=lock, lats=lats,
                thread_offset=i * (n_qvecs // max(search_clients, 1)),
            ))

        # ── Main deadline loop ─────────────────────────────────────────────
        while not deadline.expired():
            time.sleep(min(deadline.remaining(), 1.0))

        stop_event.set()
        print(f"\n[bench-stress] Deadline reached — stopping workers…")

        for f in as_completed(futs, timeout=60):
            try:
                f.result()
            except Exception:
                pass

    sampler_thread.join(timeout=15)
    if pc_thread:
        pc_thread.join(timeout=10)

    t_end_unix = time.time()
    elapsed_total = time.perf_counter() - t_start

    # ── Final summary ──────────────────────────────────────────────────────
    with lock:
        total_index_ops     = counters["index_ops"]
        total_index_errors  = counters["index_errors"]
        total_search_ops    = counters["search_ops"]
        total_search_errors = counters["search_errors"]
        final_lats          = list(lats)

    total_ops    = total_index_ops + total_search_ops
    total_errors = total_index_errors + total_search_errors
    combined_ops_s    = round(total_ops / elapsed_total, 1) if elapsed_total > 0 else 0.0
    overall_error_pct = round((total_errors / max(total_ops + total_errors, 1)) * 100, 2)
    final_health      = _health_snapshot(admin_client)

    overall_p95: float | None = None
    if final_lats:
        overall_p95 = round(percentiles_ms(final_lats)["p95_ms"], 1)

    print(
        f"\n[bench-stress] ── SUMMARY ──\n"
        f"[bench-stress]   Duration:          {elapsed_total:.1f}s\n"
        f"[bench-stress]   Combined ops/s:    {combined_ops_s}\n"
        f"[bench-stress]   Index ops/errors:  {total_index_ops:,} / {total_index_errors:,}\n"
        f"[bench-stress]   Search ops/errors: {total_search_ops:,} / {total_search_errors:,}\n"
        f"[bench-stress]   Error rate:        {overall_error_pct:.2f}%\n"
        f"[bench-stress]   Search p95:        {overall_p95} ms\n"
        f"[bench-stress]   1st error at:      "
        + (f"{first_error_s:.1f}s" if first_error_s is not None else "none")
        + f"\n[bench-stress]   Final cluster:     {final_health['status']}"
    )

    summary_row: dict = {
        "mode":                  "summary",
        "duration_s":            round(elapsed_total, 1),
        "combined_ops_s":        combined_ops_s,
        "total_ops":             total_ops,
        "total_index_ops":       total_index_ops,
        "total_index_errors":    total_index_errors,
        "total_search_ops":      total_search_ops,
        "total_search_errors":   total_search_errors,
        "overall_error_pct":     overall_error_pct,
        "search_p95_ms":         overall_p95,
        "time_to_first_error_s": round(first_error_s, 1) if first_error_s is not None else None,
        "final_cluster_status":  final_health["status"],
    }

    results = interval_rows + [summary_row]

    # ── Thanos metrics ─────────────────────────────────────────────────────
    thanos_data: dict = {}
    if thanos_uri:
        # Allow ~90s for metrics to propagate to Thanos before querying.
        print("[bench-stress] Querying Thanos metrics (allow ~30s for propagation)…")
        time.sleep(30)
        thanos_data = _query_thanos(thanos_uri, t_start_unix, t_end_unix)
        if thanos_data:
            print(f"[bench-stress] Thanos: collected {len(thanos_data)} metric series.")
        else:
            print("[bench-stress] Thanos: no metrics returned (integration may still be catching up).")

    # ── Write report ───────────────────────────────────────────────────────
    json_path, md_path = write_report(
        "bench-stress",
        params={
            "plan_label":          label,
            "index":               settings.opensearch_index,
            "doc_count":           doc_count,
            "index_clients":       index_clients,
            "search_clients":      search_clients,
            "min_duration_s":      duration,
            "actual_duration_s":   round(elapsed_total, 1),
            "post_settle_s":       post_settle_s,
            "batch_size":          batch_size,
            "k":                   k,
            "ef_search":           knn.ef_search,
            "embed_dim":           embed_dim,
            "knn_spec":            knn.to_dict(),
            "corpus_dir":          corpus_dir,
            "plan_change_target":  plan_change_target or None,
            "plan_change_after_s": plan_change_after_s if plan_change_active else None,
            "thanos_metrics":      thanos_data if thanos_data else None,
        },
        results=results,
        notes=[
            "Stress test: concurrent bulk-indexing + k-NN search, no rate limit.",
            f"{index_clients} index thread(s) × batch_size={batch_size}, "
            f"{search_clients} search thread(s) × k={k}, ef_search={knn.ef_search}.",
            f"Min duration: {duration}s. Health sampled every {_SAMPLE_INTERVAL_S}s.",
            f"Actual duration: {elapsed_total:.1f}s.",
            "Index pool uses overwrite semantics (same doc IDs) — disk is bounded.",
            "Aiven plan change mechanics: for changes between plans of the same machine "
            "type (e.g. premium-6x-8 → premium-9x-8), Aiven adds new nodes to the "
            "running cluster and migrates shards to them via OpenSearch's live shard "
            "rebalancing, then decommissions the old nodes. The cluster remains "
            "available throughout — there is no full rebuild. Node additions and "
            "removals are logged as 'node_count_change' events in the interval "
            "timeline and are visible in relocating_shards spikes.",
            *(
                [
                    f"Plan change: '{plan_change_target}' triggered after {plan_change_after_s}s.",
                    f"Test extended {post_settle_s}s past plan-change settle.",
                ]
                if plan_change_active else []
            ),
            *(
                [f"Thanos metrics collected for {len(thanos_data)} series."]
                if thanos_data else []
            ),
        ],
        out_dir=out_dir,
    )
    print(f"[bench-stress] Wrote: {json_path}")
    print(f"[bench-stress] Wrote: {md_path}")
    return 0
