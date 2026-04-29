"""
Aiven Bench Loader — FastAPI shim.

Exposes the REST API defined in
``../aiven-bench-orchestrator/loader-api/openapi.yaml`` so the
aiven-bench-orchestrator can submit benchmark jobs, stream logs, and manage
the corpus remotely. This module is the entrypoint when the image is
deployed as an Aiven Application (LOADER_MODE=1).

Endpoints
---------
POST  /run               — submit a benchmark job (SSE log stream + result)
POST  /cancel/{job_id}   — cancel a running job
POST  /corpus/build      — build the corpus on this machine (SSE progress)
POST  /corpus/upload     — upload a pre-built corpus as a .tar.gz archive
GET   /corpus/status     — current corpus state
DELETE /corpus           — wipe the corpus directory
GET   /healthz           — health check (no auth)
GET   /metrics           — Prometheus exposition (no auth, scraped by Thanos)

Authentication
--------------
Every endpoint except /healthz and /metrics requires:
  Authorization: Bearer <LOADER_API_KEY>
LOADER_API_KEY is set as an environment variable when the Aiven Application
is deployed by the orchestrator.

Usage
-----
  uvicorn dashboard.api:app --host 0.0.0.0 --port 8080

Or override the CMD in the Dockerfile:
  docker run --rm -e LOADER_API_KEY=... image uvicorn dashboard.api:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator

import prometheus_client
from fastapi import Depends, FastAPI, HTTPException, Request, Security, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sse_starlette.sse import EventSourceResponse

# Generated Pydantic models from the OpenAPI spec — do not hand-edit api_models.py.
from dashboard.api_models import (
    CorpusConfig,
    CorpusStatus,
    HealthResponse,
    JobSpec,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

_CORPUS_DIR = Path(os.environ.get("CORPUS_DIR", "/data/corpus"))
_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "/data/results"))
_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Auth ──────────────────────────────────────────────────────────────────────

_API_KEY = os.environ.get("LOADER_API_KEY", "")
_bearer = HTTPBearer(auto_error=True)


def _require_auth(
    creds: HTTPAuthorizationCredentials = Security(_bearer),
) -> None:
    if not _API_KEY:
        raise HTTPException(status_code=500, detail="LOADER_API_KEY is not configured")
    if creds.credentials != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid Bearer token")


# ── Corpus state ──────────────────────────────────────────────────────────────

_corpus_lock = threading.Lock()
_corpus_state: dict[str, Any] = {"state": "missing"}  # in-memory corpus metadata


def _corpus_status_obj() -> CorpusStatus:
    with _corpus_lock:
        s = _corpus_state.copy()
    return CorpusStatus(
        state=s.get("state", "missing"),
        config=s.get("config"),
        built_at=s.get("built_at"),
        sha256=s.get("sha256"),
        doc_count=s.get("doc_count"),
        query_count=s.get("query_count"),
        embed_dim=s.get("embed_dim"),
    )


def _corpus_sha256() -> str:
    npy = _CORPUS_DIR / "docs_embeddings.npy"
    if not npy.exists():
        return ""
    h = hashlib.sha256()
    with open(npy, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _state_from_manifest(corpus_dir: Path, fallback_doc_count: int = 0, fallback_query_count: int = 0) -> dict[str, Any]:
    """Read manifest.json and return a partial _corpus_state dict."""
    doc_count = fallback_doc_count
    query_count = fallback_query_count
    embed_dim = 768
    try:
        manifest_path = corpus_dir / "manifest.json"
        if manifest_path.exists():
            m = json.loads(manifest_path.read_text())
            doc_count = m.get("actual_docs", m.get("doc_count", doc_count))
            query_count = m.get("actual_queries", m.get("query_count", query_count))
            embed_dim = m.get("source_dim", embed_dim)
    except Exception:
        pass
    return {"doc_count": doc_count, "query_count": query_count, "embed_dim": embed_dim}


_CORPUS_ALLOWED_FILES = frozenset({
    "manifest.json",
    "docs.parquet",
    "queries.parquet",
    "docs_embeddings.npy",
    "queries_embeddings.npy",
    "qrels.npy",
})


# ── Job cancellation registry ─────────────────────────────────────────────────

_cancel_events: dict[str, threading.Event] = {}
_cancel_lock = threading.Lock()


def _register_cancel(job_id: str) -> threading.Event:
    ev = threading.Event()
    with _cancel_lock:
        _cancel_events[job_id] = ev
    return ev


def _unregister_cancel(job_id: str) -> None:
    with _cancel_lock:
        _cancel_events.pop(job_id, None)


# ── Dispatch helper (runs in a worker thread) ─────────────────────────────────

_job_lock = threading.Lock()   # one benchmark job at a time


def _dispatch_job(
    spec: JobSpec,
    cancel_ev: threading.Event,
    queue: "asyncio.Queue[dict]",
) -> None:
    """
    Run the benchmark described by spec. Puts SSE event dicts into queue.
    Runs in a ThreadPoolExecutor worker; queue is consumed by the /run SSE generator.
    """
    import asyncio
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from aiven_semantic_search_bench.config import Settings
    from aiven_semantic_search_bench.opensearch_client import KnnSpec
    from aiven_semantic_search_bench.job_spec import BenchmarkJob

    def _put(event: str, data: str) -> None:
        # asyncio.Queue is not thread-safe, but we can use a thread-safe approach
        queue.put_nowait({"event": event, "data": data})

    def _log(msg: str) -> None:
        _put("log", msg)

    # Build KnnSpec from the received data
    raw_spec = spec.spec
    knn_spec = KnnSpec(
        engine=raw_spec.engine.value,
        method=raw_spec.method.value,
        space_type=raw_spec.space_type.value,
        mode=(raw_spec.mode.value if raw_spec.mode else "in_memory"),
        compression=(raw_spec.compression.value if raw_spec.compression else "none"),
        data_type=(raw_spec.data_type.value if raw_spec.data_type else "float"),
        hnsw_m=(raw_spec.hnsw_m or 16),
        hnsw_ef_construction=(raw_spec.hnsw_ef_construction or 100),
        with_text=(raw_spec.with_text or False),
        with_metadata=(raw_spec.with_metadata or False),
        derived_source=(raw_spec.derived_source or False),
    )

    target_uri = spec.target_opensearch.uri
    settings = Settings(
        opensearch_uri=target_uri,
        opensearch_index=spec.opensearch_index,
        hf_embed_model="",
        hf_token="",
        hf_embed_max_dim=spec.embed_dim,
        embed_dim=spec.embed_dim,
        aiven_api_token=(spec.aiven_api_token or ""),
        aiven_project=(spec.aiven_project or ""),
        aiven_service_name=(spec.aiven_service_name or ""),
    )

    # Redirect stdout/stderr to the SSE stream
    class _Tee(io.TextIOBase):
        def write(self, s: str) -> int:
            for line in s.splitlines():
                if line:
                    _log(line)
            return len(s)
        def flush(self) -> None:
            pass

    tee = _Tee()
    report_path = ""
    ok = False

    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    try:
        if cancel_ev.is_set():
            _log("[loader] Job cancelled before start.")
            _put("result", json.dumps({"ok": False, "error": "cancelled", "report_path": ""}))
            return

        bench_type = spec.bench_type.value if hasattr(spec.bench_type, 'value') else spec.bench_type
        _log(f"[loader] Starting {bench_type} job {spec.job_id}")

        if bench_type == "index":
            from aiven_semantic_search_bench.bench_index import cmd_bench_index
            rc = cmd_bench_index(
                settings,
                doc_count=spec.doc_count,
                batch_sizes=list(spec.batch_sizes) if spec.batch_sizes else [1, 5, 10, 20, 50],
                embed_dim=spec.embed_dim,
                spec=knn_spec,
                corpus_dir=str(_CORPUS_DIR),
                label=f"{spec.service_label}/{bench_type}",
                out_dir=str(_RESULTS_DIR),
            )
            ok = (rc == 0)

        elif bench_type == "search":
            from aiven_semantic_search_bench.bench_search import cmd_bench_search
            rc = cmd_bench_search(
                settings,
                rounds=(spec.rounds or 3),
                query_count=spec.query_count,
                k=(spec.k or 10),
                embed_dim=spec.embed_dim,
                spec=knn_spec,
                corpus_dir=str(_CORPUS_DIR),
                label=f"{spec.service_label}/{bench_type}",
                out_dir=str(_RESULTS_DIR),
                warmup_queries=(spec.warmup_queries or 50),
                search_clients=(spec.search_clients or 1),
                target_throughput=(spec.target_throughput or 0.0),
                time_period=(spec.time_period or 0),
                force_merge_segments=(spec.force_merge_segments or 0),
            )
            ok = (rc == 0)

        elif bench_type == "recall":
            from aiven_semantic_search_bench.bench_recall import cmd_bench_recall
            rc = cmd_bench_recall(
                settings,
                query_count=spec.query_count,
                k=(spec.k or 10),
                embed_dim=spec.embed_dim,
                spec=knn_spec,
                corpus_dir=str(_CORPUS_DIR),
                label=f"{spec.service_label}/{bench_type}",
                out_dir=str(_RESULTS_DIR),
            )
            ok = (rc == 0)

        elif bench_type == "hybrid":
            from aiven_semantic_search_bench.bench_hybrid import cmd_bench_hybrid
            fs_val = spec.filter_selectivity.value if spec.filter_selectivity else "none"
            rc = cmd_bench_hybrid(
                settings,
                query_count=spec.query_count,
                k=(spec.k or 10),
                embed_dim=spec.embed_dim,
                spec=knn_spec,
                filter_selectivity=fs_val,
                corpus_dir=str(_CORPUS_DIR),
                label=f"{spec.service_label}/{bench_type}",
                out_dir=str(_RESULTS_DIR),
            )
            ok = (rc == 0)

        elif bench_type == "stress":
            from aiven_semantic_search_bench.bench_stress import cmd_bench_stress
            rc = cmd_bench_stress(
                settings,
                embed_dim=spec.embed_dim,
                spec=knn_spec,
                corpus_dir=str(_CORPUS_DIR),
                label=f"{spec.service_label}/{bench_type}",
                out_dir=str(_RESULTS_DIR),
                index_clients=(spec.stress_index_clients or 8),
                search_clients=(spec.stress_search_clients or 16),
                duration=(spec.stress_duration or 120),
                batch_size=(spec.stress_batch_size or 100),
                k=(spec.stress_k or 100),
                plan_change_target=(spec.plan_change_target or ""),
                plan_change_after_s=(spec.plan_change_after_s or 60),
                post_settle_s=(spec.post_settle_s or 60),
                aiven_api_token=(spec.aiven_api_token or ""),
                aiven_project=(spec.aiven_project or ""),
                aiven_service_name=(spec.aiven_service_name or ""),
                thanos_uri="",
            )
            ok = (rc == 0)

        else:
            _log(f"[loader] Unknown bench_type: {bench_type!r}")
            ok = False

    except Exception as exc:
        import traceback
        _log(f"[loader] EXCEPTION: {type(exc).__name__}: {exc}")
        _log(traceback.format_exc())
        ok = False
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    _put("result", json.dumps({"ok": ok, "report_path": report_path}))


# ── Corpus build helper ───────────────────────────────────────────────────────

def _run_corpus_build(config: CorpusConfig, queue: "queue_module.Queue") -> None:
    """Runs in a worker thread; puts SSE-style dicts into queue."""
    import queue as queue_module_local

    def _put(event: str, data: str) -> None:
        queue.put({"event": event, "data": data})

    def _log(msg: str) -> None:
        _put("log", msg)

    class _Tee(io.TextIOBase):
        def write(self, s: str) -> int:
            for line in s.splitlines():
                if line:
                    _log(line)
            return len(s)
        def flush(self) -> None:
            pass

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from aiven_semantic_search_bench.config import Settings
    from aiven_semantic_search_bench.corpus.builder import build_corpus

    hf_model = config.model or "nomic-ai/nomic-embed-text-v1.5"
    hf_token = config.hf_token or ""
    settings = Settings(
        opensearch_uri="",
        opensearch_index="",
        hf_embed_model=hf_model,
        hf_token=hf_token,
        hf_embed_max_dim=768,
        embed_dim=768,
        aiven_api_token="",
        aiven_project="",
        aiven_service_name="",
    )

    tee = _Tee()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    try:
        with _corpus_lock:
            _corpus_state["state"] = "building"
            _corpus_state["config"] = config.model_dump()

        _log(f"[corpus] Building corpus: dataset={config.dataset} model={config.model} docs={config.docs_n} queries={config.queries_n}")

        if config.hf_token:
            os.environ["HF_TOKEN"] = config.hf_token

        rc = build_corpus(
            settings=settings,
            out_dir=str(_CORPUS_DIR),
            preset=config.dataset or "mixed",
            target_docs=config.docs_n,
            target_queries=config.queries_n,
            with_metadata=(config.with_metadata or False),
        )

        if rc != 0:
            raise RuntimeError(f"build_corpus returned exit code {rc}")

        # Build ground truth if requested
        if config.with_groundtruth:
            _log("[corpus] Building ground-truth nearest neighbours...")
            from aiven_semantic_search_bench.corpus.groundtruth import build_groundtruth
            rc2 = build_groundtruth(corpus_dir=str(_CORPUS_DIR))
            if rc2 != 0:
                raise RuntimeError(f"build_groundtruth returned exit code {rc2}")

        sha = _corpus_sha256()
        built_at = datetime.now(timezone.utc).isoformat()

        manifest_state = _state_from_manifest(
            _CORPUS_DIR,
            fallback_doc_count=config.docs_n,
            fallback_query_count=config.queries_n,
        )

        with _corpus_lock:
            _corpus_state.update({
                "state": "ready",
                "config": config.model_dump(),
                "built_at": built_at,
                "sha256": sha,
                **manifest_state,
            })

        status_obj = _corpus_status_obj()
        _put("ready", status_obj.model_dump_json())
        _log(f"[corpus] Corpus ready. sha256={sha[:12]}...")

    except Exception as exc:
        import traceback
        _log(f"[corpus] FAILED: {exc}")
        _log(traceback.format_exc())
        with _corpus_lock:
            _corpus_state["state"] = "missing"
        _put("error", str(exc))
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    # sentinel so the consumer knows the stream is done
    queue.put(None)


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Aiven Bench Loader API",
    description="Benchmark execution engine — deployed as an Aiven Application.",
    version="1.0.0",
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


@app.post("/run", dependencies=[Depends(_require_auth)])
async def run_job(spec: JobSpec) -> EventSourceResponse:
    """Submit a benchmark job; stream logs + result as SSE."""
    with _corpus_lock:
        corpus_ready = _corpus_state.get("state") == "ready"
    if not corpus_ready:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Corpus not ready. Submit POST /corpus/build first.",
        )

    import asyncio
    import queue as q_module

    sync_q: q_module.Queue = q_module.Queue()
    cancel_ev = _register_cancel(spec.job_id)

    loop = asyncio.get_event_loop()

    def _run() -> None:
        with _job_lock:
            _dispatch_job(spec, cancel_ev, sync_q)  # type: ignore[arg-type]

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def _generate() -> AsyncGenerator[dict, None]:
        try:
            while True:
                try:
                    item = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: sync_q.get(timeout=30)
                    )
                    if item is None:
                        break
                    yield item
                    if item.get("event") == "result":
                        break
                except Exception:
                    break
        finally:
            _unregister_cancel(spec.job_id)

    return EventSourceResponse(_generate())


@app.post("/cancel/{job_id}", dependencies=[Depends(_require_auth)])
async def cancel_job(job_id: str) -> dict:
    """Cancel a running job by setting its cancellation event."""
    with _cancel_lock:
        ev = _cancel_events.get(job_id)
    if ev is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
    ev.set()
    return {"cancelled": True}


@app.post("/corpus/build", dependencies=[Depends(_require_auth)])
async def corpus_build(config: CorpusConfig) -> EventSourceResponse:
    """Build the corpus; stream progress as SSE."""
    import asyncio
    import queue as q_module

    with _corpus_lock:
        if _corpus_state.get("state") == "building":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Corpus build already in progress.",
            )

    sync_q: q_module.Queue = q_module.Queue()

    thread = threading.Thread(
        target=_run_corpus_build, args=(config, sync_q), daemon=True
    )
    thread.start()

    async def _generate() -> AsyncGenerator[dict, None]:
        while True:
            try:
                item = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: sync_q.get(timeout=30)
                )
            except q_module.Empty:
                if not thread.is_alive():
                    break
                continue
            if item is None:
                break
            yield item
            if item.get("event") in ("ready", "error"):
                break

    return EventSourceResponse(_generate())


@app.get("/corpus/status", dependencies=[Depends(_require_auth)])
async def corpus_status() -> CorpusStatus:
    """Return current corpus state."""
    return _corpus_status_obj()


@app.delete("/corpus", dependencies=[Depends(_require_auth)])
async def delete_corpus() -> dict:
    """Wipe the corpus directory."""
    with _corpus_lock:
        if _corpus_state.get("state") in ("building",):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete corpus while build is in progress.",
            )
        if _job_lock.locked():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete corpus while a job is running.",
            )
        shutil.rmtree(str(_CORPUS_DIR), ignore_errors=True)
        _CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        _corpus_state.clear()
        _corpus_state["state"] = "missing"
    return {"deleted": True}


@app.post("/corpus/upload", dependencies=[Depends(_require_auth)])
async def corpus_upload(file: UploadFile) -> CorpusStatus:
    """
    Upload a pre-built corpus as a .tar.gz archive.

    The archive must contain ``manifest.json`` at its root.  Only the
    canonical corpus filenames are extracted; any other members are silently
    skipped.  Rejects while a build or benchmark job is in progress.
    """
    with _corpus_lock:
        if _corpus_state.get("state") == "building":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot upload corpus while a build is in progress.",
            )
    if _job_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot upload corpus while a benchmark job is running.",
        )

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    with tempfile.TemporaryDirectory(prefix="corpus_upload_") as tmp_str:
        tmp = Path(tmp_str)
        try:
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
                for member in tf.getmembers():
                    name = Path(member.name).name  # strip any directory prefix
                    if name not in _CORPUS_ALLOWED_FILES:
                        continue
                    member.name = name  # force flat extraction
                    tf.extract(member, path=tmp, filter="data")
        except tarfile.TarError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid tar.gz archive: {exc}",
            ) from exc

        if not (tmp / "manifest.json").exists():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Archive must contain manifest.json.",
            )
        if not (tmp / "docs_embeddings.npy").exists():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Archive must contain docs_embeddings.npy.",
            )

        # Atomically replace the corpus directory contents.
        shutil.rmtree(str(_CORPUS_DIR), ignore_errors=True)
        shutil.copytree(str(tmp), str(_CORPUS_DIR))

    sha = _corpus_sha256()
    built_at = datetime.now(timezone.utc).isoformat()
    manifest_state = _state_from_manifest(_CORPUS_DIR)

    with _corpus_lock:
        _corpus_state.update({
            "state": "ready",
            "config": None,
            "built_at": built_at,
            "sha256": sha,
            **manifest_state,
        })

    return _corpus_status_obj()


@app.get("/healthz")
async def healthz() -> HealthResponse:
    """Health check — no auth required."""
    with _corpus_lock:
        cs = _corpus_state.get("state", "missing")
    return HealthResponse(status="ok", corpus_state=cs)


@app.get("/metrics")
async def metrics() -> Any:
    """Prometheus text format exposition — no auth required, scraped by Thanos."""
    from fastapi.responses import PlainTextResponse
    data = prometheus_client.generate_latest()
    return PlainTextResponse(data.decode("utf-8"), media_type="text/plain; version=0.0.4")
