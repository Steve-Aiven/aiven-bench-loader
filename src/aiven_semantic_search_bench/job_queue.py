"""
JSON-backed benchmark job queue.

The queue persists to ``{results_dir}/queue/queue.json`` so it survives
container restarts.  Every mutation is file-locked (``fcntl.flock``) so the
NiceGUI UI thread and the runner thread can safely read/write concurrently
without corrupting the file.

Only used by the standalone NiceGUI dashboard. The loader API
(``dashboard/api.py``) bypasses the queue and dispatches each /run request
synchronously to the appropriate ``cmd_bench_*`` function — the orchestrator
owns scheduling on its side.

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

        Before returning a job, checks its ``depends_on`` field.  If the
        dependency job ended in ``failed``, this job is automatically marked
        ``skipped`` and the search continues to the next pending job.

        The runner marks the returned job ``running`` via ``mark_running``
        immediately after so the same job is not returned twice.
        Returns None if nothing is pending (or runnable).
        """
        result: BenchmarkJob | None = None

        def _fn(fh):
            nonlocal result
            jobs = self._read_locked(fh)
            # Build a quick state lookup for dependency resolution.
            state_by_id = {j["job_id"]: j.get("state", "pending") for j in jobs}
            changed = False
            for job_dict in jobs:
                if job_dict.get("state") != "pending":
                    continue
                dep = job_dict.get("depends_on", "")
                if dep and state_by_id.get(dep) == "failed":
                    # Dependency failed — skip this job automatically.
                    job_dict["state"] = "skipped"
                    job_dict["finished_at"] = _utcnow()
                    job_dict["error_message"] = (
                        f"Skipped: dependency job {dep[:8]}… failed."
                    )
                    changed = True
                    continue
                if dep and state_by_id.get(dep) in ("pending", "running"):
                    # Dependency not yet finished — skip for now, try later.
                    continue
                result = BenchmarkJob.from_dict(job_dict)
                break
            if changed:
                self._write_locked(fh, jobs)

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
        """Remove all ok/failed/skipped jobs; return how many were removed."""
        removed = 0

        def _fn(fh):
            nonlocal removed
            jobs = self._read_locked(fh)
            kept = [j for j in jobs if j.get("state") not in ("ok", "failed", "skipped")]
            removed = len(jobs) - len(kept)
            if removed:
                self._write_locked(fh, kept)

        self._with_lock(_fn)
        return removed

    def reset_stale_running(self) -> int:
        """
        Mark any jobs stuck in ``running`` state as ``failed``.

        Called once at runner startup.  Jobs left in "running" state after a
        process crash are orphaned — the runner thread that owned them is gone
        and they will never reach ``mark_done``.  Resetting them to ``failed``
        makes the situation visible in the UI and allows their dependents
        (search, recall, stress jobs) to be auto-skipped correctly rather than
        blocking forever waiting for a dependency that will never finish.

        Returns the number of jobs reset.
        """
        reset = 0

        def _fn(fh):
            nonlocal reset
            jobs = self._read_locked(fh)
            for j in jobs:
                if j.get("state") == "running":
                    j["state"] = "failed"
                    j["finished_at"] = _utcnow()
                    j["error_message"] = (
                        "Process was restarted while this job was running. "
                        "Re-submit from the New Test tab if needed."
                    )
                    reset += 1
            if reset:
                self._write_locked(fh, jobs)

        self._with_lock(_fn)
        return reset

    def cancel_pending(self, job_ids: list[str] | None = None) -> int:
        """
        Mark pending jobs as ``skipped`` so the runner ignores them.

        If ``job_ids`` is given only those jobs are cancelled; otherwise all
        pending jobs are cancelled.  Jobs that are already ``running`` are not
        touched — the runner owns them until they finish.

        Returns the number of jobs cancelled.
        """
        cancelled = 0

        def _fn(fh):
            nonlocal cancelled
            jobs = self._read_locked(fh)
            for j in jobs:
                if j.get("state") != "pending":
                    continue
                if job_ids is not None and j["job_id"] not in job_ids:
                    continue
                j["state"] = "skipped"
                j["finished_at"] = _utcnow()
                j["error_message"] = "Cancelled by user."
                cancelled += 1
            if cancelled:
                self._write_locked(fh, jobs)

        self._with_lock(_fn)
        return cancelled
