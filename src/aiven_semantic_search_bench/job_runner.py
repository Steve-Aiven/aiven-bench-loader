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
from pathlib import Path

from .job_queue import JobQueue
from .job_spec import BenchmarkJob

_started = False
_lock = threading.Lock()

_POLL_INTERVAL_S = 2.0


class _FlushingTee(io.TextIOBase):
    """
    Write-only text stream that mirrors every write to two sinks
    and flushes both immediately.  Used to send job output to the
    log file in real-time while also keeping a copy in memory for
    error reporting.
    """

    def __init__(self, primary: io.IOBase, secondary: io.StringIO) -> None:
        self._primary = primary
        self._secondary = secondary

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._primary.flush()
        self._secondary.write(s)
        return len(s)

    def flush(self) -> None:
        self._primary.flush()
        self._secondary.flush()


def _log_path(job: BenchmarkJob) -> Path:
    d = Path(job.out_dir) / "queue"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{job.job_id}.log"


def _dispatch(job: BenchmarkJob, log_fh: io.TextIOWrapper) -> tuple[bool, str]:
    """
    Run the benchmark for this job, streaming all output to ``log_fh`` in
    real-time (flushed after every ``print`` call) so the Queue log viewer
    updates while the job is still running.

    Returns ``(ok, report_path)``.
    """
    from .bench_index import cmd_bench_index
    from .bench_search import cmd_bench_search
    from .bench_recall import cmd_bench_recall
    from .bench_hybrid import cmd_bench_hybrid
    from .bench_stress import cmd_bench_stress
    from .config import Settings

    settings = Settings(
        opensearch_uri=job.opensearch_uri,
        opensearch_index=job.opensearch_index,
        hf_embed_model="",
        hf_token="",
        hf_embed_max_dim=job.embed_dim,
        embed_dim=job.embed_dim,
        aiven_api_token="",
        aiven_project="",
        aiven_service_name="",
    )

    # Tee: every print goes to the log file immediately AND to an in-memory
    # buffer (used only for exception reporting below).
    buf = io.StringIO()
    tee = _FlushingTee(log_fh, buf)
    report_path = ""
    ok = True

    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    try:
        if job.bench_type == "build-corpus":
            ok = _run_build_corpus(job, settings)

        elif job.bench_type == "build-groundtruth":
            ok = _run_build_groundtruth(job)

        elif job.bench_type == "index":
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
                warmup_queries=job.warmup_queries,
                search_clients=job.search_clients,
                target_throughput=job.target_throughput,
                time_period=job.time_period,
                force_merge_segments=job.force_merge_segments,
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

        elif job.bench_type == "stress":
            rc = cmd_bench_stress(
                settings,
                embed_dim=job.embed_dim,
                spec=job.spec,
                corpus_dir=job.corpus_dir,
                label=job.display_label(),
                out_dir=job.out_dir,
                index_clients=job.stress_index_clients,
                search_clients=job.stress_search_clients,
                duration=job.stress_duration,
                batch_size=job.stress_batch_size,
                k=job.stress_k,
                plan_change_target=job.plan_change_target,
                plan_change_after_s=job.plan_change_after_s,
                post_settle_s=job.post_settle_s,
                aiven_api_token=job.aiven_api_token,
                aiven_project=job.aiven_project,
                aiven_service_name=job.aiven_service_name,
                thanos_uri=job.thanos_uri,
            )
            ok = (rc == 0)

        else:
            print(f"[runner] Unknown bench_type: {job.bench_type!r}")
            ok = False

    except Exception as exc:
        import traceback
        print(f"[runner] EXCEPTION: {type(exc).__name__}: {exc}")
        print(traceback.format_exc())
        ok = False
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    return ok, report_path


def _run_build_corpus(job: "BenchmarkJob", settings: "Settings") -> bool:  # noqa: F821
    from .corpus.builder import build_corpus

    # Rebuild Settings with HF model from env so corpus builds have access to
    # the model name (measurement jobs pass empty strings for HF fields).
    import os
    from .config import Settings as Cfg
    from aiven_semantic_search_bench.hf_embedder import RECOMMENDED_MODELS
    _default_model = os.environ.get("HF_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    _chosen_model  = job.corpus_embed_model or _default_model
    # Auto-derive max_dim from model if it's one of the known recommended models.
    _known_dims = {m[1]: m[2] for m in RECOMMENDED_MODELS}
    _default_max_dim = int(os.environ.get("HF_EMBED_MAX_DIM", "768"))
    _max_dim = _known_dims.get(_chosen_model, _default_max_dim)

    hf_settings = Cfg(
        opensearch_uri="",
        opensearch_index="",
        hf_embed_model=_chosen_model,
        hf_token=os.environ.get("HF_TOKEN", ""),
        hf_embed_max_dim=_max_dim,
        embed_dim=min(job.embed_dim, _max_dim),
        aiven_api_token="",
        aiven_project="",
        aiven_service_name="",
    )
    rc = build_corpus(
        settings=hf_settings,
        out_dir=job.corpus_dir,
        preset=job.corpus_preset,
        target_docs=job.doc_count,
        target_queries=job.query_count,
        with_metadata=job.corpus_with_metadata,
    )
    if rc != 0:
        return False
    if job.corpus_with_groundtruth:
        return _run_build_groundtruth(job)
    return True


def _run_build_groundtruth(job: "BenchmarkJob") -> bool:  # noqa: F821
    from .corpus.groundtruth import build_groundtruth
    rc = build_groundtruth(corpus_dir=job.corpus_dir)
    return rc == 0


def _runner_loop(results_dir: str) -> None:
    queue = JobQueue(results_dir)
    stale = queue.reset_stale_running()
    if stale:
        print(
            f"[runner] Reset {stale} stale 'running' job(s) to 'failed' "
            f"(left over from a previous process crash).",
            flush=True,
        )
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
                f"[runner] service={job.service_label}  version={job.opensearch_version}\n"
            )
            # Warn if the user-assigned label contains a version number that
            # clearly contradicts the service's reported OpenSearch version.
            # The Aiven API sometimes returns only the major version (e.g. "2")
            # so we only warn when the major versions differ, not when the
            # stored version is a prefix of the label version.
            import re as _re
            label_ver_match = _re.search(r'v?(\d+\.\d+)', job.service_label)
            if label_ver_match and job.opensearch_version:
                label_ver = label_ver_match.group(1)           # e.g. "2.17"
                label_major = label_ver.split(".")[0]           # e.g. "2"
                svc_ver = job.opensearch_version               # e.g. "2" or "2.17.1"
                svc_major = svc_ver.split(".")[0]
                # Only warn when major versions differ, or when the service
                # version is at least as specific as the label and they differ.
                major_mismatch = label_major != svc_major
                full_mismatch = (
                    len(svc_ver.split(".")) >= 2          # service has major.minor
                    and label_ver not in svc_ver          # and they don't match
                )
                if major_mismatch or full_mismatch:
                    log_fh.write(
                        f"[runner] ⚠️  VERSION MISMATCH: label suggests v{label_ver} "
                        f"but service is running OpenSearch {job.opensearch_version}. "
                        f"Go to Services tab and re-apply correct labels before submitting.\n"
                    )
            log_fh.write("\n")
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
