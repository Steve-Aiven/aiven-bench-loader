"""
Benchmark report writer.

Every benchmark produces two artifacts:

1. A JSON file - the raw, machine-readable result. Useful for diffing across
   runs, plotting, or feeding into another tool.
2. A Markdown file - a human-readable summary you can paste into a blog post
   or share with a teammate. Includes a configuration block, a results table,
   and a notes section.

Both files share a timestamped filename so re-runs do not overwrite previous
results. They live under `results/` by default (gitignored).
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any


def _field_guide_markdown(bench_name: str) -> str:
    """
    Human-readable definitions for top-level report sections, deployment,
    preflight, common params, and benchmark-specific results columns.
    """
    specific = _FIELD_GUIDE_BY_BENCH.get(bench_name, _FIELD_GUIDE_DEFAULT)

    return f"""## Report field guide

### Document structure (JSON and this Markdown report)

| Concept | Meaning |
|--------|--------|
| **`name`** | Which command produced this file (`{bench_name}`). |
| **`generated_at`** | When the report file was written. |
| **`params`** | Everything that describes *how* the run was configured (inputs, index name, k-NN spec, corpus paths, etc.). |
| **`results`** | Measured outputs — table rows below; meaning depends on the benchmark (see *This benchmark*). |
| **`notes`** | Free-form caveats, definitions, and reminders appended by the tool. |
| **`deployment`** | *(When present)* Where the runner and target OpenSearch service run, plus **plan** and **cloud** when resolved via env vars or the Aiven API. |
| **`preflight`** | *(When present)* Timings for repeated **HTTPS GET /** to the cluster **before** the benchmark (same URI/auth as the run). Surrogate for network + TLS + HTTP stack latency (not ICMP ping). |
| **`field_guide`** | *(JSON only)* This section as a single string for programmatic consumers. |

### `deployment.loader` (benchmark runner)

| Field | Meaning |
|--------|--------|
| **`hostname`** | Host running the benchmark (container or laptop). |
| **`service_name`** | Optional Aiven Application name or label. |
| **`plan`** | Aiven service **plan** when set manually or fetched from the API. |
| **`cloud_name`** | Aiven **cloud / region** identifier (e.g. `aws-eu-west-1`). |
| **`metadata_source`** | `env` (from `BENCH_LOADER_*`), `aiven_api`, `unknown`, or `aiven_api_error`. |

### `deployment.target_opensearch`

| Field | Meaning |
|--------|--------|
| **`host`** / **`port`** | OpenSearch endpoint parsed from the benchmark URI (credentials are never stored here). |
| **`service_name`** | Aiven OpenSearch service name when configured (`BENCH_TARGET_OPENSEARCH_SERVICE_NAME` or related settings). |
| **`plan`** / **`cloud_name`** | Target cluster plan and region when set or resolved via API. |
| **`metadata_source`** | Same idea as for the loader. |

### `preflight` keys

| Field | Meaning |
|--------|--------|
| **`ok`** | Whether all probe requests finished without transport errors. |
| **`method`** | Probe type — **`HTTPS GET /`** using the same scheme/host/port and auth as OpenSearch. |
| **`rounds`** | Number of timed probes (see `BENCH_PREFLIGHT_ROUNDS`). |
| **`target_base_url`** | Exact base URL that was timed. |
| **`min_ms`**, **`p50_ms`**, **`p95_ms`**, **`p99_ms`**, **`max_ms`**, **`mean_ms`** | Distribution of **end-to-end** probe latency in milliseconds. |
| **`error`** | Present when `ok` is false — failure reason. |

### Common `params` keys

| Param | Typical meaning |
|--------|------------------|
| **`plan_label`** | Your label for this run (e.g. OpenSearch version or configuration) for comparing reports. |
| **`index`** | OpenSearch index name used for this benchmark. |
| **`embed_dim`** | Vector dimension for this run (may truncate/normalize from the corpus stored dimension). |
| **`knn_spec`** | Serialized k-NN settings: engine (`faiss` / `lucene`), method, similarity space, `in_memory` vs `on_disk`, compression, vector dtype, HNSW **m** / **ef_construction** / **ef_search**, hybrid flags. |
| **`corpus_dir`** | On-disk corpus directory (parquet + NumPy embeddings). |
| **`corpus_preset`** | Dataset mix recorded in the corpus manifest (e.g. `mixed`). |

### This benchmark: `{bench_name}`

{specific}
"""


_FIELD_GUIDE_DEFAULT = """_No extended guide for this benchmark name — interpret **`results`** using **`params`** and the command-line help for this command._"""


_FIELD_GUIDE_BY_BENCH: dict[str, str] = {
    "bench-index": """**Extra params:** **`documents`** — docs indexed per batch-size sweep; **`batch_sizes`** — bulk `_bulk` sizes tested; **`embed_model`** — model id from the corpus manifest.

**Results rows:** One row per **`batch_size`**. **`docs_per_sec`** is throughput for that sweep; **`p50_ms`** / **`p95_ms`** / etc. are latencies **per `_bulk` HTTP request** (network + OpenSearch). **`wall_seconds`** is total time for indexing all documents at that batch size. The index is reset between batch sizes.""",
    "bench-recall": """**Extra params:** **`doc_count`** — documents present in the index at recall time (from cluster stats); **`query_count`** — queries evaluated; **`k`** — k-NN **`k`** requested from OpenSearch; **`groundtruth_k`** — width of the brute-force neighbour lists in `qrels.npy`.

**Results:** **`p*_ms`** — client-side round-trip latency per query. **`recall@1`**, **`recall@5`**, … — fraction of queries where **at least one** of the top-*K* ground-truth neighbour doc IDs appears in OpenSearch’s top-*K* hits (see **`notes`**). Low recall often means the index holds fewer docs than the corpus used to build ground truth.""",
    "bench-search": """**Extra params:** **`mode`** — `rounds` (fixed rounds × queries) vs `sustained` (target throughput × duration); **`rounds`**, **`query_count`**, **`search_clients`** — concurrency layout; **`warmup_queries`** — queries discarded while stabilising p95; **`force_merge_segments`** — optional merge before measurement; **`target_throughput`** / **`time_period`** — sustained-mode caps.

**Results:** In rounds mode, one row per measurement round plus optional aggregate; latencies are k-NN search round-trips. In sustained mode, summary rows include **`ops_per_sec`** and latency percentiles over all queries.""",
    "bench-hybrid": """**Extra params:** **`query_mode`** — native hybrid vs bool-should fallback; **`filter_selectivity`** — metadata filter strictness for hybrid + filter tests.

**Results:** Latency percentiles plus optional **`recall@*`** columns when `qrels.npy` exists (same interpretation as **`bench-recall`**).""",
    "bench-stress": """**Extra params:** **`index_clients`** / **`search_clients`** — concurrent bulk vs search threads; **`batch_size`**, **`k`**, **`ef_search`** — load intensity; **`min_duration_s`** / **`actual_duration_s`** — planned vs actual wall time; optional **`plan_change_*`** when mid-run resize is configured.

**Results:** Interval rows sample cluster health over time; the final **`mode: summary`** row aggregates throughput, errors, and search latency tails.""",
    "bench-recover": """**Extra params:** **`idle_minutes`** — wait before cold-start measurement; **`warmup_ms`** — latency of the health-check query before the idle window.

**Results:** One row per numbered request after idle — **`cold (first after idle)`** vs **`warm`** rows show auto-pause wake-up cost on tiers that pause.""",
    "bench-plan-change": """**Extra params:** **`from_plan`** / **`to_plan`** — Aiven plan migration under test; **`and_back`** — whether a revert leg runs; **`service`** — Aiven OpenSearch service name; **`timeline`** — per-query samples in JSON (phase, latency, errors).

**Results:** One summary row **per phase** (`before_change`, `during_migration`, `after_change`, etc.) with query counts, error counts, and latency percentiles for that phase.""",
}


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_format_value(val)}" for k, val in v.items())
    return str(v)


def _md_deployment_block(dep: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("## Deployment")
    lines.append("")
    loader = dep.get("loader") or {}
    lines.append("### Loader (benchmark runner)")
    lines.append("")
    for key, label in (
        ("hostname", "Hostname"),
        ("service_name", "Aiven service name"),
        ("plan", "Plan"),
        ("cloud_name", "Cloud / region"),
        ("metadata_source", "Metadata source"),
    ):
        v = loader.get(key)
        if v:
            lines.append(f"- **{label}:** `{v}`")
    lines.append("")
    tgt = dep.get("target_opensearch") or {}
    lines.append("### Target OpenSearch cluster")
    lines.append("")
    host, port = tgt.get("host"), tgt.get("port")
    if host:
        hp = f"{host}:{port}" if port else str(host)
        lines.append(f"- **Endpoint:** `{hp}`")
    for key, label in (
        ("service_name", "Aiven service name"),
        ("plan", "Plan"),
        ("cloud_name", "Cloud / region"),
        ("metadata_source", "Metadata source"),
    ):
        v = tgt.get(key)
        if v:
            lines.append(f"- **{label}:** `{v}`")
    if not host and not any(tgt.get(k) for k in ("service_name", "plan", "cloud_name")):
        lines.append("_No target metadata (URI host only unless `BENCH_TARGET_OPENSEARCH_*` or API lookup is configured)._")
    lines.append("")
    return lines


def _md_preflight_block(preflight: dict[str, Any]) -> list[str]:
    lines: list[str] = ["## Preflight (before benchmark)", ""]
    if preflight.get("ok"):
        lines.append(
            f"- **{preflight.get('method', 'probe')}** to `{preflight.get('target_base_url', '')}` "
            f"— {preflight.get('rounds', 0)} round(s): "
            f"min **{preflight.get('min_ms')} ms**, "
            f"p50 **{preflight.get('p50_ms')} ms**, "
            f"p95 **{preflight.get('p95_ms')} ms**, "
            f"mean **{preflight.get('mean_ms')} ms**"
        )
        lines.append("")
        lines.append(
            "_Uses HTTPS with the same credentials as the benchmark URI (TLS + HTTP stack), "
            "not ICMP ping — suitable from containers and matches real client paths._"
        )
    else:
        lines.append(f"- **Failed:** {preflight.get('error', 'unknown error')}")
        if preflight.get("target_base_url"):
            lines.append(f"- Target: `{preflight['target_base_url']}`")
    lines.append("")
    return lines


def _render_markdown(
    name: str,
    params: dict[str, Any],
    results: list[dict[str, Any]],
    notes: list[str],
    *,
    deployment: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_")
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: `{platform.python_version()}`")
    lines.append(f"- Platform: `{platform.platform()}`")
    lines.append("")

    if deployment:
        lines.extend(_md_deployment_block(deployment))
    if preflight:
        lines.extend(_md_preflight_block(preflight))

    lines.append("## Parameters")
    lines.append("")
    for k, v in params.items():
        lines.append(f"- `{k}`: {_format_value(v)}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    if not results:
        lines.append("_No results recorded._")
    else:
        headers = list(results[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in results:
            lines.append("| " + " | ".join(_format_value(row.get(h, "")) for h in headers) + " |")
    lines.append("")

    lines.append(_field_guide_markdown(name))
    lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def write_report(
    name: str,
    *,
    params: dict[str, Any],
    results: list[dict[str, Any]],
    notes: list[str] | None = None,
    out_dir: str | Path = "results",
    deployment: dict[str, Any] | None = None,
    preflight: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """
    Write JSON + Markdown reports under `out_dir/<name>-<timestamp>.{json,md}`.

    Returns the (json_path, md_path) tuple so the CLI can print where each
    artifact ended up.
    """
    notes = notes or []
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = p / f"{name}-{ts}.json"
    md_path = p / f"{name}-{ts}.md"

    payload: dict[str, Any] = {
        "name": name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": params,
        "results": results,
        "notes": notes,
    }
    if deployment is not None:
        payload["deployment"] = deployment
    if preflight is not None:
        payload["preflight"] = preflight
    payload["field_guide"] = _field_guide_markdown(name)
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(
        _render_markdown(
            name,
            params,
            results,
            notes,
            deployment=deployment,
            preflight=preflight,
        )
    )

    # Notify the (optional) ClickHouse sink so the loader API can surface the
    # actual report path on the SSE `result` event and so the orchestrator can
    # find the report referenced from `bench_runs.summary`. No-op when CH is
    # not configured.
    try:
        from .clickhouse_sink import get_sink

        get_sink().report_written(json_path, md_path)
    except Exception:
        pass

    return json_path, md_path
