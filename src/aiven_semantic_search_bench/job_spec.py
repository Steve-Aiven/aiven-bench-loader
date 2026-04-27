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

BenchType = Literal["index", "search", "recall", "hybrid"]
JobState = Literal["pending", "running", "ok", "failed"]

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
            "job_id":              self.job_id,
            "bench_type":          self.bench_type,
            "service_label":       self.service_label,
            "opensearch_uri":      self.opensearch_uri,
            "opensearch_version":  self.opensearch_version,
            "opensearch_index":    self.opensearch_index,
            "spec":                self.spec.to_dict(),
            "embed_dim":           self.embed_dim,
            "doc_count":           self.doc_count,
            "query_count":         self.query_count,
            "corpus_dir":          self.corpus_dir,
            "out_dir":             self.out_dir,
            "rounds":              self.rounds,
            "k":                   self.k,
            "batch_sizes":         self.batch_sizes,
            "filter_selectivity":  self.filter_selectivity,
            "state":               self.state,
            "submitted_at":        self.submitted_at,
            "started_at":          self.started_at,
            "finished_at":         self.finished_at,
            "report_path":         self.report_path,
            "log_path":            self.log_path,
            "error_message":       self.error_message,
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
