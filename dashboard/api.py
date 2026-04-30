"""
Aiven Bench Loader — FastAPI shim.

Exposes the REST API defined in
``../aiven-bench-orchestrator/loader-api/openapi.yaml`` so the
aiven-bench-orchestrator can submit benchmark jobs, stream logs, and manage
the corpus remotely. This module is the entrypoint when the image is
deployed as an Aiven Application (LOADER_MODE=1).

Endpoints
---------
POST  /run               — submit a benchmark job (SSE log stream + result).
                           Optional: omit or null ``embed_dim``, ``doc_count``, ``query_count``
                           to use values from the loaded corpus (GET /corpus/status).
POST  /cancel/{job_id}   — cancel a running job
POST  /corpus/build      — build the corpus on this machine (SSE progress)
POST  /corpus/upload     — upload a pre-built corpus as a .tar.gz archive
POST  /corpus/fetch      — fetch pre-built corpus from pre-signed URL
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
import queue
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator

import httpx
import prometheus_client
from fastapi import Depends, FastAPI, HTTPException, Request, Security, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
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


def _resolve_job_spec_with_corpus(spec: JobSpec) -> JobSpec:
    """Fill ``embed_dim`` / ``doc_count`` / ``query_count`` from the ready corpus when omitted."""
    with _corpus_lock:
        if _corpus_state.get("state") != "ready":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Corpus not ready. Submit POST /corpus/build first.",
            )
        c_embed = _corpus_state.get("embed_dim")
        c_docs = _corpus_state.get("doc_count")
        c_queries = _corpus_state.get("query_count")

    embed_dim = spec.embed_dim if spec.embed_dim is not None else c_embed
    doc_count = spec.doc_count if spec.doc_count is not None else c_docs
    query_count = spec.query_count if spec.query_count is not None else c_queries

    missing = [
        name
        for name, val in (
            ("embed_dim", embed_dim),
            ("doc_count", doc_count),
            ("query_count", query_count),
        )
        if val is None
    ]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Cannot resolve job fields {missing!r} from corpus metadata. "
                f"Pass them explicitly in the request body or reload the corpus (GET /corpus/status)."
            ),
        )
    if int(doc_count) <= 0 or int(query_count) <= 0 or int(embed_dim) <= 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Resolved doc_count, query_count, and embed_dim must be positive.",
        )
    return spec.model_copy(
        update={
            "embed_dim": int(embed_dim),
            "doc_count": int(doc_count),
            "query_count": int(query_count),
        }
    )


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
_CORPUS_REQUIRED_FILES = frozenset({
    "manifest.json",
    "docs.parquet",
    "queries.parquet",
    "docs_embeddings.npy",
    "queries_embeddings.npy",
})
_corpus_install_lock = threading.Lock()


class CorpusFetchRequest(BaseModel):
    url: str = Field(min_length=1)
    sha256: str | None = None


class _QueueReader(io.RawIOBase):
    """A file-like bridge for piping async stream chunks into tarfile."""

    def __init__(self) -> None:
        self._queue: "queue.Queue[bytes | None]" = queue.Queue(maxsize=16)
        self._buffer = bytearray()
        self._closed = False
        self._error: Exception | None = None

    def readable(self) -> bool:
        return True

    def feed(self, chunk: bytes) -> None:
        if chunk:
            while True:
                if self._error is not None:
                    err = self._error
                    self._error = None
                    raise err
                try:
                    self._queue.put(chunk, timeout=0.1)
                    return
                except queue.Full:
                    continue

    def finish(self) -> None:
        self._queue.put(None)

    def set_error(self, exc: Exception) -> None:
        self._error = exc

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        if self._closed:
            return b""
        if self._error is not None:
            err = self._error
            self._error = None
            raise err

        want_all = size is None or size < 0
        target = float("inf") if want_all else size
        while len(self._buffer) < target:
            item = self._queue.get()
            if item is None:
                self._closed = True
                break
            self._buffer.extend(item)
            if want_all:
                continue

        if want_all:
            out = bytes(self._buffer)
            self._buffer.clear()
            return out

        out = bytes(self._buffer[:size])
        del self._buffer[:size]
        return out


def _extract_allowed_tar_gz(fileobj: io.RawIOBase, dst_dir: Path) -> None:
    try:
        with tarfile.open(fileobj=fileobj, mode="r|gz") as tf:
            for member in tf:
                if member is None or not member.isfile():
                    continue
                name = Path(member.name).name
                if name not in _CORPUS_ALLOWED_FILES:
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                out_path = dst_dir / name
                with open(out_path, "wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
    except tarfile.TarError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid tar.gz archive: {exc}",
        ) from exc


def _validate_corpus_dir(corpus_dir: Path) -> None:
    missing = sorted(name for name in _CORPUS_REQUIRED_FILES if not (corpus_dir / name).exists())
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Archive missing required files: {', '.join(missing)}",
        )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_replace_corpus(extracted_dir: Path) -> None:
    """Replace the corpus directory with newly-extracted content.

    Uses shutil.move which falls back to copy+delete when os.rename fails
    across mount boundaries (common in containerised environments where
    /data/corpus may be on a different overlay layer than /data).
    """
    backup = _CORPUS_DIR.parent / f".corpus-backup-{uuid.uuid4().hex}"
    moved_old = False
    try:
        if _CORPUS_DIR.exists():
            shutil.move(str(_CORPUS_DIR), str(backup))
            moved_old = True
        shutil.move(str(extracted_dir), str(_CORPUS_DIR))
    except Exception:
        if moved_old and backup.exists() and not _CORPUS_DIR.exists():
            shutil.move(str(backup), str(_CORPUS_DIR))
        raise
    finally:
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)


@contextmanager
def _corpus_mutation_guard(action: str) -> Any:
    with _corpus_lock:
        if _corpus_state.get("state") == "building":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Cannot {action} corpus while a build is in progress.",
            )
    if _job_lock.locked():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Cannot {action} corpus while a benchmark job is running.",
        )
    if not _corpus_install_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot mutate corpus while another corpus operation is in progress.",
        )
    try:
        yield
    finally:
        _corpus_install_lock.release()


def _finalize_installed_corpus(corpus_dir: Path, expected_sha256: str | None = None) -> CorpusStatus:
    _validate_corpus_dir(corpus_dir)
    sha = _sha256_file(corpus_dir / "docs_embeddings.npy")
    if expected_sha256 and expected_sha256.strip() and sha.lower() != expected_sha256.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"docs_embeddings.npy sha256 mismatch: expected {expected_sha256.strip().lower()}, got {sha}",
        )
    _atomic_replace_corpus(corpus_dir)
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


async def _extract_streamed_tar_gz_to_dir(stream: AsyncIterator[bytes], dst_dir: Path) -> None:
    reader = _QueueReader()
    result: dict[str, Exception | None] = {"error": None}

    def _worker() -> None:
        try:
            _extract_allowed_tar_gz(reader, dst_dir)
        except Exception as exc:
            result["error"] = exc
            reader.set_error(exc)

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()

    try:
        async for chunk in stream:
            if result["error"] is not None:
                break
            reader.feed(chunk)
    finally:
        if result["error"] is None:
            reader.finish()
        worker.join()

    if result["error"] is not None:
        raise result["error"]


async def _upload_file_chunks(file: UploadFile, chunk_size: int = 1 << 20) -> AsyncIterator[bytes]:
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        yield chunk


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
    from aiven_semantic_search_bench.clickhouse_sink import get_sink

    sink = get_sink()

    def _put(event: str, data: str) -> None:
        # asyncio.Queue is not thread-safe, but we can use a thread-safe approach
        queue.put_nowait({"event": event, "data": data})

    def _log(msg: str) -> None:
        _put("log", msg)
        # Mirror the SSE log line into ClickHouse so the orchestrator can
        # query the structured event stream by job_id. No-op when CH is unset.
        sink.log("info", "stdout", msg)

    # Build KnnSpec from the received data
    raw_spec = spec.spec
    knn_spec = KnnSpec(
        embed_dim=spec.embed_dim,
        engine=str(raw_spec.engine),
        method=str(raw_spec.method),
        space_type=str(raw_spec.space_type),
        mode=str(raw_spec.mode) if raw_spec.mode else "in_memory",
        compression=str(raw_spec.compression) if raw_spec.compression else "none",
        data_type=str(raw_spec.data_type) if raw_spec.data_type else "float",
        m=(raw_spec.hnsw_m or 16),
        ef_construction=(raw_spec.hnsw_ef_construction or 100),
        with_text=(raw_spec.with_text or False),
        with_metadata=(raw_spec.with_metadata or False),
        derived_source=(raw_spec.derived_source or False),
    )

    target_uri = spec.target_opensearch.uri or os.environ.get("OPENSEARCH_URI", "")
    settings = Settings(
        opensearch_uri=target_uri,
        opensearch_index=spec.opensearch_index,
        hf_embed_model="",
        hf_token="",
        hf_embed_max_dim=spec.embed_dim,
        hf_embed_device=os.environ.get("HF_EMBED_DEVICE", "").strip(),
        corpus_embed_backend=os.environ.get(
            "CORPUS_EMBED_BACKEND", "hf"
        ).strip().lower(),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        gemini_embed_model=os.environ.get(
            "GEMINI_EMBED_MODEL", "models/text-embedding-004"
        ).strip(),
        gcp_project_id=(
            os.environ.get("GCP_PROJECT_ID", "").strip()
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        ),
        gcp_location=os.environ.get("GCP_LOCATION", "us-central1").strip(),
        vertex_embed_model=os.environ.get(
            "VERTEX_EMBED_MODEL", "gemini-embedding-001"
        ).strip(),
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

    bench_type = spec.bench_type.value if hasattr(spec.bench_type, 'value') else spec.bench_type
    sink.bind_job(
        spec.job_id,
        bench_type=str(bench_type),
        label=str(spec.service_label or ""),
        spec=spec.model_dump(mode="json") if hasattr(spec, "model_dump") else {},
    )

    sys.stdout = tee  # type: ignore[assignment]
    sys.stderr = tee  # type: ignore[assignment]
    try:
        if cancel_ev.is_set():
            _log("[loader] Job cancelled before start.")
            _put("result", json.dumps({"ok": False, "error": "cancelled", "report_path": ""}))
            sink.unbind_job(status="cancelled", summary={"error": "cancelled before start"})
            return

        _log(f"[loader] Starting {bench_type} job {spec.job_id}")
        _log(
            f"[loader] workload embed_dim={spec.embed_dim} "
            f"doc_count={spec.doc_count} query_count={spec.query_count}"
        )

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
            fs_val = str(spec.filter_selectivity) if spec.filter_selectivity else "none"
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

    # Pull the actual report path the bench module wrote (via reporter.write_report
    # → sink.report_written). Bench commands do not return paths directly, so this
    # is the canonical source. Falls back to "" when CH is not configured AND the
    # bench command did not run far enough to call write_report.
    last = sink.last_report_path
    if last is not None:
        report_path = str(last)

    summary = {
        "ok": ok,
        "report_path": report_path,
        "bench_type": str(bench_type),
    }
    sink.unbind_job(status=("ok" if ok else "failed"), summary=summary)

    _put("result", json.dumps({"ok": ok, "report_path": report_path}))


# ── Corpus build helper ───────────────────────────────────────────────────────

def _run_corpus_build(config: CorpusConfig, queue: "queue_module.Queue") -> None:
    """Runs in a worker thread; puts SSE-style dicts into queue."""
    import queue as queue_module_local

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from aiven_semantic_search_bench.clickhouse_sink import get_sink

    sink = get_sink()
    # Synthetic job_id so corpus-build progress shows up alongside benchmark
    # runs in the bench_logs table; the orchestrator can filter on this prefix.
    corpus_job_id = f"corpus-{int(time.time())}"
    sink.bind_job(
        corpus_job_id,
        bench_type="corpus_build",
        label=str(config.dataset or "mixed"),
        spec=config.model_dump() if hasattr(config, "model_dump") else {},
    )

    def _put(event: str, data: str) -> None:
        queue.put({"event": event, "data": data})

    def _log(msg: str) -> None:
        _put("log", msg)
        sink.log("info", "corpus", msg)

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
        hf_embed_max_dim=int(os.environ.get("HF_EMBED_MAX_DIM", "768")),
        hf_embed_device=os.environ.get("HF_EMBED_DEVICE", "").strip(),
        corpus_embed_backend=os.environ.get(
            "CORPUS_EMBED_BACKEND", "hf"
        ).strip().lower(),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", "").strip(),
        gemini_embed_model=os.environ.get(
            "GEMINI_EMBED_MODEL", "models/text-embedding-004"
        ).strip(),
        gcp_project_id=(
            os.environ.get("GCP_PROJECT_ID", "").strip()
            or os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
        ),
        gcp_location=os.environ.get("GCP_LOCATION", "us-central1").strip(),
        vertex_embed_model=os.environ.get(
            "VERTEX_EMBED_MODEL", "gemini-embedding-001"
        ).strip(),
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
        sink.unbind_job(
            status="ok",
            summary={
                "sha256": sha,
                "built_at": built_at,
                "doc_count": manifest_state.get("doc_count"),
                "query_count": manifest_state.get("query_count"),
            },
        )

    except Exception as exc:
        import traceback
        _log(f"[corpus] FAILED: {exc}")
        _log(traceback.format_exc())
        with _corpus_lock:
            _corpus_state["state"] = "missing"
        _put("error", str(exc))
        sink.unbind_job(status="failed", summary={"error": str(exc)})
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
    spec = _resolve_job_spec_with_corpus(spec)

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
        import queue as _q
        try:
            while True:
                try:
                    item = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: sync_q.get(timeout=30)
                    )
                except _q.Empty:
                    # Keep the stream alive while the job thread is still running.
                    if not thread.is_alive():
                        break
                    continue
                except Exception:
                    break
                if item is None:
                    break
                yield item
                if item.get("event") == "result":
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

    The archive must contain all required corpus artifacts at its root.
    Only canonical corpus filenames are extracted; any other members are
    silently skipped. Rejects while a build or benchmark job is in progress.
    """
    with _corpus_mutation_guard("upload"):
        tmp = Path(tempfile.mkdtemp(prefix="corpus-upload-", dir=str(_CORPUS_DIR.parent)))
        try:
            await _extract_streamed_tar_gz_to_dir(_upload_file_chunks(file), tmp)
            return _finalize_installed_corpus(tmp)
        except HTTPException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process uploaded archive: {exc}",
            ) from exc


@app.post("/corpus/fetch", dependencies=[Depends(_require_auth)], operation_id="fetchCorpus")
async def corpus_fetch(req: CorpusFetchRequest) -> CorpusStatus:
    url = req.url.strip()
    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Request field 'url' must be non-empty.",
        )

    with _corpus_mutation_guard("fetch"):
        tmp = Path(tempfile.mkdtemp(prefix="corpus-fetch-", dir=str(_CORPUS_DIR.parent)))
        timeout = httpx.Timeout(timeout=300.0, connect=15.0, read=30.0, write=30.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                max_redirects=3,
            ) as client:
                async with client.stream("GET", url) as resp:
                    if resp.status_code >= 400:
                        raise HTTPException(
                            status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Unable to download corpus archive from URL (status {resp.status_code}).",
                        )
                    await _extract_streamed_tar_gz_to_dir(resp.aiter_bytes(), tmp)
            return _finalize_installed_corpus(tmp, req.sha256)
        except HTTPException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        except httpx.HTTPError as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unable to fetch corpus archive: {exc}",
            ) from exc
        except Exception as exc:
            shutil.rmtree(tmp, ignore_errors=True)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to process fetched archive: {exc}",
            ) from exc


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


@app.get("/results", dependencies=[Depends(_require_auth)])
async def list_results() -> list[dict]:
    """List available benchmark result files (JSON and Markdown)."""
    files = []
    for p in sorted(_RESULTS_DIR.glob("*")):
        if p.is_file() and p.suffix in (".json", ".md"):
            stat = p.stat()
            files.append({
                "name": p.name,
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            })
    return files


@app.get("/results/{filename}", dependencies=[Depends(_require_auth)])
async def download_result(filename: str) -> Any:
    """Download a single benchmark result file by name."""
    from fastapi.responses import FileResponse
    safe = _RESULTS_DIR / Path(filename).name  # strip any path separators
    if not safe.exists() or not safe.is_file():
        raise HTTPException(status_code=404, detail="Result file not found")
    media_type = "application/json" if safe.suffix == ".json" else "text/markdown"
    return FileResponse(str(safe), filename=safe.name, media_type=media_type)
