"""
Background job runner daemon.

The Streamlit app imports this module once at startup (via ``ensure_started``).
A single daemon thread pops jobs from the ``JobQueue`` and dispatches them to
the appropriate ``cmd_bench_*`` function, capturing stdout/stderr to a per-job
log file.

One job runs at a time — this is intentional so that network + CPU pressure
from one benchmark cell does not contaminate the latency measurements of the
next.

Thread safety:
  - ``ensure_started`` is idempotent and safe to call from the Streamlit
    main thread on every page render.
  - The runner thread and the Streamlit thread only share state via the
    ``JobQueue`` file (which uses ``flock`` for mutual exclusion).
"""

from __future__ import annotations

import io
import sys
import threading
import time
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from .job_queue import JobQueue
from .job_spec import BenchmarkJob

_started = False
_lock = threading.Lock()

_POLL_INTERVAL_S = 2.0


def _log_path(job: BenchmarkJob) -> Path:
    d = Path(job.out_dir) / "queue"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job.job_id}.log"


def _dispatch(job: BenchmarkJob, log_fh: io.TextIOWrapper) -> tuple[bool, str]:
    """
    Run the benchmark for this job, capturing all output to ``log_fh``.

    Returns ``(ok, report_path)``.
    """
    from .bench_index import cmd_bench_index
    from .bench_search import cmd_bench_search
    from .bench_recall import cmd_bench_recall
    from .bench_hybrid import cmd_bench_hybrid
    from .config import Settings

    # Build a minimal Settings object from the job — the runner never reads
    # environment variables for the service URI so that each job targets the
    # correct service regardless of what is in .env.
    settings = Settings(
        opensearch_uri=job.opensearch_uri,
        opensearch_index=job.opensearch_index,
        hf_embed_model="",      # not used by measurement commands
        hf_token="",
        hf_embed_max_dim=job.embed_dim,
        embed_dim=job.embed_dim,
        aiven_api_token="",
        aiven_project="",
        aiven_service_name="",
    )

    # Capture print output.
    buf = io.StringIO()
    report_path = ""
    ok = True

    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            if job.bench_type == "index":
                rc = cmd_bench_index(
                    settings,
                    doc_count=job.doc_count,
                    batch_sizes=job.batch_sizes,
                    embed_dim=job.embed_dim,
                    spec=job.spec,
                    corpus_dir=job.corpus_dir,
                    label=job.display_label(),
                    out_dir=job.out_dir,
                )
                ok = (rc == 0)

            elif job.bench_type == "search":
                rc = cmd_bench_search(
                    settings,
                    rounds=job.rounds,
                    query_count=job.query_count,
                    k=job.k,
                    embed_dim=job.embed_dim,
                    spec=job.spec,
                    corpus_dir=job.corpus_dir,
                    label=job.display_label(),
                    out_dir=job.out_dir,
                )
                ok = (rc == 0)

            elif job.bench_type == "recall":
                rc = cmd_bench_recall(
                    settings,
                    query_count=job.query_count,
                    k=job.k,
                    embed_dim=job.embed_dim,
                    spec=job.spec,
                    corpus_dir=job.corpus_dir,
                    label=job.display_label(),
                    out_dir=job.out_dir,
                )
                ok = (rc == 0)

            elif job.bench_type == "hybrid":
                rc = cmd_bench_hybrid(
                    settings,
                    query_count=job.query_count,
                    k=job.k,
                    embed_dim=job.embed_dim,
                    spec=job.spec,
                    filter_selectivity=job.filter_selectivity,
                    corpus_dir=job.corpus_dir,
                    label=job.display_label(),
                    out_dir=job.out_dir,
                )
                ok = (rc == 0)

            else:
                print(f"[runner] Unknown bench_type: {job.bench_type!r}")
                ok = False

    except Exception as exc:
        print(f"[runner] EXCEPTION: {exc}", file=buf)
        ok = False

    log_fh.write(buf.getvalue())
    log_fh.flush()
    return ok, report_path


def _runner_loop(results_dir: str) -> None:
    queue = JobQueue(results_dir)
    while True:
        job = queue.pop_pending()
        if job is None:
            time.sleep(_POLL_INTERVAL_S)
            continue

        log_file = _log_path(job)
        queue.mark_running(job.job_id, log_path=str(log_file))

        with open(log_file, "w") as log_fh:
            log_fh.write(
                f"[runner] Starting job {job.job_id}\n"
                f"[runner] type={job.bench_type}  label={job.display_label()}\n"
                f"[runner] service={job.service_label}  version={job.opensearch_version}\n\n"
            )
            log_fh.flush()
            ok, report_path = _dispatch(job, log_fh)
            log_fh.write(
                f"\n[runner] Job {'OK' if ok else 'FAILED'} — {job.job_id}\n"
            )

        queue.mark_done(
            job.job_id,
            ok=ok,
            report_path=report_path,
            log_path=str(log_file),
        )


def ensure_started(results_dir: str = "results") -> None:
    """
    Start the runner daemon thread if it has not already been started.

    Safe to call multiple times — subsequent calls are no-ops.  Designed to
    be called at Streamlit app startup (e.g. top of the main app file or in
    a shared ``utils.py`` imported by every page).
    """
    global _started
    with _lock:
        if _started:
            return
        t = threading.Thread(
            target=_runner_loop,
            args=(results_dir,),
            daemon=True,
            name="bench-runner",
        )
        t.start()
        _started = True
