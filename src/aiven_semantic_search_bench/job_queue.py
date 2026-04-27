"""
JSON-backed benchmark job queue.

The queue persists to ``{results_dir}/queue/queue.json`` so it survives
container restarts.  Every mutation is file-locked (``fcntl.flock``) so the
Streamlit UI thread and the runner thread can safely read/write concurrently
without corrupting the file.

Queue file format: a JSON array of serialized ``BenchmarkJob`` dicts.
The file is rewritten atomically on every mutation (write-to-tmp + rename).
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Any

from .job_spec import BenchmarkJob


_QUEUE_NAME = "queue.json"


def _queue_dir(results_dir: str) -> Path:
    d = Path(results_dir) / "queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _queue_path(results_dir: str) -> Path:
    return _queue_dir(results_dir) / _QUEUE_NAME


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class JobQueue:
    """
    Thread-safe, file-backed FIFO queue of ``BenchmarkJob`` objects.

    All public methods acquire an exclusive lock on the queue file for the
    duration of the read-modify-write cycle.  The lock is an advisory
    ``flock``; all processes reading the file should use the same locking
    convention.

    Usage::

        q = JobQueue("results")
        q.enqueue(job)          # append, state=pending
        job = q.pop_pending()   # None if nothing to run
        q.mark_running(job_id)
        q.mark_done(job_id, report_path=..., log_path=...)
        rows = q.all_jobs()     # list for the queue table
    """

    def __init__(self, results_dir: str = "results") -> None:
        self._path = _queue_path(results_dir)

    # ── Internal I/O ─────────────────────────────────────────────────────────

    def _read_locked(self, fh) -> list[dict[str, Any]]:
        fh.seek(0)
        content = fh.read()
        if not content.strip():
            return []
        return json.loads(content)

    def _write_locked(self, fh, jobs: list[dict[str, Any]]) -> None:
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(jobs, indent=2))
        tmp.replace(self._path)
        # Truncate + rewrite in the open handle so the lock is still held.
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(jobs, indent=2))
        fh.flush()

    def _with_lock(self, fn):
        """Open the queue file, acquire an exclusive lock, run fn(fh), release."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                return fn(fh)
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    # ── Public API ────────────────────────────────────────────────────────────

    def enqueue(self, job: BenchmarkJob) -> None:
        """Append a job with state=pending and a submitted_at timestamp."""
        job.state = "pending"
        job.submitted_at = _utcnow()

        def _fn(fh):
            jobs = self._read_locked(fh)
            jobs.append(job.to_dict())
            self._write_locked(fh, jobs)

        self._with_lock(_fn)

    def pop_pending(self) -> BenchmarkJob | None:
        """
        Return the first pending job without removing it from the queue.

        The runner marks it ``running`` via ``mark_running`` immediately after
        so the same job is not returned twice.  Returns None if nothing is
        pending.
        """
        result: BenchmarkJob | None = None

        def _fn(fh):
            nonlocal result
            jobs = self._read_locked(fh)
            for job_dict in jobs:
                if job_dict.get("state") == "pending":
                    result = BenchmarkJob.from_dict(job_dict)
                    break

        self._with_lock(_fn)
        return result

    def mark_running(self, job_id: str, log_path: str = "") -> None:
        def _fn(fh):
            jobs = self._read_locked(fh)
            for j in jobs:
                if j["job_id"] == job_id:
                    j["state"] = "running"
                    j["started_at"] = _utcnow()
                    if log_path:
                        j["log_path"] = log_path
            self._write_locked(fh, jobs)

        self._with_lock(_fn)

    def mark_done(
        self,
        job_id: str,
        *,
        ok: bool,
        report_path: str = "",
        log_path: str = "",
        error_message: str = "",
    ) -> None:
        def _fn(fh):
            jobs = self._read_locked(fh)
            for j in jobs:
                if j["job_id"] == job_id:
                    j["state"] = "ok" if ok else "failed"
                    j["finished_at"] = _utcnow()
                    if report_path:
                        j["report_path"] = report_path
                    if log_path:
                        j["log_path"] = log_path
                    if error_message:
                        j["error_message"] = error_message
            self._write_locked(fh, jobs)

        self._with_lock(_fn)

    def all_jobs(self) -> list[BenchmarkJob]:
        """Return all jobs (any state) in queue order."""
        def _fn(fh):
            return self._read_locked(fh)

        raw = self._with_lock(_fn)
        return [BenchmarkJob.from_dict(d) for d in raw]

    def pending_count(self) -> int:
        return sum(1 for j in self.all_jobs() if j.state == "pending")

    def clear_finished(self) -> int:
        """Remove all ok/failed jobs; return how many were removed."""
        removed = 0

        def _fn(fh):
            nonlocal removed
            jobs = self._read_locked(fh)
            kept = [j for j in jobs if j.get("state") not in ("ok", "failed")]
            removed = len(jobs) - len(kept)
            if removed:
                self._write_locked(fh, kept)

        self._with_lock(_fn)
        return removed
