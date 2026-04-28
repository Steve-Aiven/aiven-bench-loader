"""
BenchmarkJob — the unit of work queued and executed by the runner.

A job fully describes one cell in the benchmark matrix: which service to
target, which k-NN configuration to use, how many documents / queries to
process, and which benchmark type to run.  It is JSON-serializable so it
can round-trip through the on-disk queue file.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .opensearch_client import KnnSpec

BenchType = Literal["index", "search", "recall", "hybrid", "stress", "build-corpus", "build-groundtruth"]
JobState = Literal["pending", "running", "ok", "failed", "skipped"]

# Filter selectivity values used by bench-hybrid.
FilterSelectivity = Literal["none", "low", "high"]


@dataclass
class BenchmarkJob:
    """
    One benchmark cell.

    ``service_label`` is the user-supplied tag (e.g. ``"v2.17"``,
    ``"v3.3-prod"``).  It is used as the ``plan_label`` in report files so
    all the existing dashboard charts work without modification.

    ``opensearch_uri`` is resolved from the Aiven API by the UI and stored
    here so the runner never touches session state.  It is treated as
    sensitive: the runner validates it is non-empty but does not log it.
    """

    bench_type: BenchType
    service_label: str
    opensearch_uri: str
    opensearch_version: str          # e.g. "2.17", "2.19", "3.3"
    opensearch_index: str
    spec: KnnSpec
    embed_dim: int
    doc_count: int
    query_count: int
    corpus_dir: str = "corpus"
    out_dir: str = "results"
    # bench-search / recall settings
    rounds: int = 3
    k: int = 10
    # bench-index settings
    batch_sizes: list[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])
    # bench-hybrid settings
    filter_selectivity: FilterSelectivity = "none"
    # bench-search OSB-inspired settings
    warmup_queries: int = 50         # queries per warmup round (0 = skip warmup)
    search_clients: int = 1          # parallel search worker threads
    target_throughput: float = 0.0   # ops/s rate cap (0 = unlimited, rounds mode)
    time_period: int = 0             # seconds for sustained mode (0 = rounds mode)
    force_merge_segments: int = 0    # 0 = skip; N = merge to N segments before search
    # bench-stress settings
    stress_index_clients: int = 8    # concurrent bulk-index threads
    stress_search_clients: int = 16  # concurrent search threads
    stress_duration: int = 120       # total test duration in seconds
    stress_batch_size: int = 100     # docs per _bulk request
    stress_k: int = 100              # k for k-NN search (high = more JVM pressure)
    # Optional mid-stress plan change via Aiven API
    plan_change_target: str = ""     # plan name to change to (empty = no change)
    plan_change_after_s: int = 60    # seconds into the stress test before triggering
    post_settle_s: int = 60          # seconds to continue after plan change settles
    aiven_api_token: str = ""        # Aiven API token (stored like opensearch_uri)
    aiven_project: str = ""          # Aiven project name
    aiven_service_name: str = ""     # Aiven service name (for plan change API call)
    # Optional Thanos metrics collection
    thanos_uri: str = ""             # Prometheus-compatible URI for Thanos query
    # build-corpus settings (only used when bench_type == "build-corpus")
    corpus_preset: str = "mixed"
    corpus_with_metadata: bool = False
    corpus_with_groundtruth: bool = True  # auto-build ground truth after corpus
    corpus_embed_model: str = ""          # overrides HF_EMBED_MODEL env var
    # Dependency tracking: if set, this job will be skipped when the named
    # job_id ended in "failed".  Used to skip search/recall/hybrid when their
    # corresponding index job failed (e.g. due to missing corpus).
    depends_on: str = ""
    # Internal queue fields — set by the queue, not the submitter
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: JobState = "pending"
    submitted_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    report_path: str = ""
    log_path: str = ""
    error_message: str = ""

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id":                    self.job_id,
            "bench_type":                self.bench_type,
            "service_label":             self.service_label,
            "opensearch_uri":            self.opensearch_uri,
            "opensearch_version":        self.opensearch_version,
            "opensearch_index":          self.opensearch_index,
            "spec":                      self.spec.to_dict(),
            "embed_dim":                 self.embed_dim,
            "doc_count":                 self.doc_count,
            "query_count":               self.query_count,
            "corpus_dir":                self.corpus_dir,
            "out_dir":                   self.out_dir,
            "rounds":                    self.rounds,
            "k":                         self.k,
            "batch_sizes":               self.batch_sizes,
            "filter_selectivity":        self.filter_selectivity,
            "warmup_queries":            self.warmup_queries,
            "search_clients":            self.search_clients,
            "target_throughput":         self.target_throughput,
            "time_period":               self.time_period,
            "force_merge_segments":      self.force_merge_segments,
            "stress_index_clients":      self.stress_index_clients,
            "stress_search_clients":     self.stress_search_clients,
            "stress_duration":           self.stress_duration,
            "stress_batch_size":         self.stress_batch_size,
            "stress_k":                  self.stress_k,
            "plan_change_target":        self.plan_change_target,
            "plan_change_after_s":       self.plan_change_after_s,
            "post_settle_s":             self.post_settle_s,
            "aiven_api_token":           self.aiven_api_token,
            "aiven_project":             self.aiven_project,
            "aiven_service_name":        self.aiven_service_name,
            "thanos_uri":                self.thanos_uri,
            "corpus_preset":             self.corpus_preset,
            "corpus_with_metadata":      self.corpus_with_metadata,
            "corpus_with_groundtruth":   self.corpus_with_groundtruth,
            "corpus_embed_model":        self.corpus_embed_model,
            "depends_on":                self.depends_on,
            "state":                     self.state,
            "submitted_at":              self.submitted_at,
            "started_at":                self.started_at,
            "finished_at":               self.finished_at,
            "report_path":               self.report_path,
            "log_path":                  self.log_path,
            "error_message":             self.error_message,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "BenchmarkJob":
        job = BenchmarkJob(
            bench_type=d["bench_type"],
            service_label=d["service_label"],
            opensearch_uri=d["opensearch_uri"],
            opensearch_version=d["opensearch_version"],
            opensearch_index=d["opensearch_index"],
            spec=KnnSpec.from_dict(d["spec"]),
            embed_dim=int(d["embed_dim"]),
            doc_count=int(d["doc_count"]),
            query_count=int(d["query_count"]),
            corpus_dir=d.get("corpus_dir", "corpus"),
            out_dir=d.get("out_dir", "results"),
            rounds=int(d.get("rounds", 3)),
            k=int(d.get("k", 10)),
            batch_sizes=list(d.get("batch_sizes", [1, 5, 10, 20, 50])),
            filter_selectivity=d.get("filter_selectivity", "none"),
            warmup_queries=int(d.get("warmup_queries", 50)),
            search_clients=int(d.get("search_clients", 1)),
            target_throughput=float(d.get("target_throughput", 0.0)),
            time_period=int(d.get("time_period", 0)),
            force_merge_segments=int(d.get("force_merge_segments", 0)),
            stress_index_clients=int(d.get("stress_index_clients", 8)),
            stress_search_clients=int(d.get("stress_search_clients", 16)),
            stress_duration=int(d.get("stress_duration", 120)),
            stress_batch_size=int(d.get("stress_batch_size", 100)),
            stress_k=int(d.get("stress_k", 100)),
            plan_change_target=d.get("plan_change_target", ""),
            plan_change_after_s=int(d.get("plan_change_after_s", 60)),
            post_settle_s=int(d.get("post_settle_s", 60)),
            aiven_api_token=d.get("aiven_api_token", ""),
            aiven_project=d.get("aiven_project", ""),
            aiven_service_name=d.get("aiven_service_name", ""),
            thanos_uri=d.get("thanos_uri", ""),
            corpus_preset=d.get("corpus_preset", "mixed"),
            corpus_with_metadata=bool(d.get("corpus_with_metadata", False)),
            corpus_with_groundtruth=bool(d.get("corpus_with_groundtruth", True)),
            corpus_embed_model=d.get("corpus_embed_model", ""),
            depends_on=d.get("depends_on", ""),
            job_id=d.get("job_id", uuid.uuid4().hex),
            state=d.get("state", "pending"),
            submitted_at=d.get("submitted_at", ""),
            started_at=d.get("started_at", ""),
            finished_at=d.get("finished_at", ""),
            report_path=d.get("report_path", ""),
            log_path=d.get("log_path", ""),
            error_message=d.get("error_message", ""),
        )
        return job

    def display_label(self) -> str:
        """Short label used in the queue table."""
        return f"{self.service_label}/{self.spec.label()}/{self.bench_type}"
