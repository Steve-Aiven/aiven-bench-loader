"""
Build a benchmark corpus offline.

End-to-end: sample text from public IR datasets, embed every text once using
either a local sentence-transformers model or (optional) remote Google embeddings
(Gemini API key or Vertex AI with ADC), then persist parquet + npy + manifest.
Local HF runs need a one-time weight download; remote backends bill per cloud
pricing.

Resume support means a build can survive an interruption (e.g. laptop sleep)
without losing all progress: if the .npy file already has the expected shape
and its last row is non-zero the embedding phase is skipped entirely.

Cost note
---------
Using ``nomic-ai/nomic-embed-text-v1.5`` (default), the corpus build is
**free**.  Model weights (~270 MB) are downloaded once to the HuggingFace
cache; subsequent runs are fully offline.  CPU throughput is roughly
300–600 docs/sec on a modern laptop; a 100k-doc build takes 3–6 minutes.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Settings
from ..hf_embedder import HfEmbedder
from .io import (
    existing_embeddings,
    write_dataframe,
    write_embeddings,
    write_manifest,
)
from .sources import (
    MAX_DIM,
    SUPPORTED_DIMS,
    sample_corpus,
)

# How often we flush in-progress embeddings to disk.  Each checkpoint writes
# the full npy array; on a modern SSD this takes ~0.1s for 768-dim/100k rows.
_CHECKPOINT_EVERY = 5_000


def build_corpus(
    *,
    settings: Settings,
    out_dir: str | Path,
    preset: str,
    target_docs: int,
    target_queries: int,
    embed_batch_size: int = 64,
    seed: int = 42,
    dry_run: bool = False,
    resume: bool = True,
    with_metadata: bool = False,
) -> int:
    """
    Run the full sample → embed → persist pipeline.

    Returns 0 on success, 1 on user-visible error.
    """
    if preset == "":
        raise ValueError("preset must be non-empty")
    if target_docs <= 0 or target_queries <= 0:
        raise ValueError("target_docs and target_queries must be > 0")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_dim = settings.hf_embed_max_dim

    backend = (settings.corpus_embed_backend or "hf").strip().lower()
    if backend not in ("hf", "gemini", "vertex"):
        print(
            f"[build-corpus] ERROR: unknown CORPUS_EMBED_BACKEND={backend!r} "
            f"(use hf, gemini, or vertex)."
        )
        return 1

    _model_disp = (
        settings.vertex_embed_model
        if backend == "vertex"
        else settings.gemini_embed_model
        if backend == "gemini"
        else settings.hf_embed_model
    )
    print(
        f"[build-corpus] preset={preset}  target_docs={target_docs}  "
        f"target_queries={target_queries}  out_dir={out_dir}  "
        f"backend={backend}  model={_model_disp}  max_dim={max_dim}"
    )

    print("[build-corpus] Sampling text from Hugging Face datasets...")
    t0 = time.perf_counter()
    docs, queries = sample_corpus(
        preset=preset,
        target_docs=target_docs,
        target_queries=target_queries,
        seed=seed,
    )
    print(
        f"[build-corpus] Sampled {len(docs)} docs, {len(queries)} queries "
        f"in {time.perf_counter() - t0:.1f}s."
    )

    docs_df = pd.DataFrame(docs, columns=["doc_id", "text", "source"])
    queries_df = pd.DataFrame(queries, columns=["query_id", "text", "source"])

    if with_metadata:
        docs_df = _add_synthetic_metadata(docs_df, seed=seed)
        print("[build-corpus] Synthetic metadata columns added (category, tenant_id, created_at).")

    write_dataframe(out_dir, "docs", docs_df)
    write_dataframe(out_dir, "queries", queries_df)
    print(
        f"[build-corpus] Wrote docs.parquet and queries.parquet "
        f"({len(docs)} + {len(queries)} rows)."
    )

    if dry_run:
        print("[build-corpus] dry_run=True — skipping embedding and manifest write.")
        return 0

    if backend == "gemini":
        if not settings.gemini_api_key:
            print(
                "[build-corpus] ERROR: CORPUS_EMBED_BACKEND=gemini requires GEMINI_API_KEY."
            )
            return 1
        from ..gemini_embedder import GeminiCorpusEmbedder

        pause_s = float(os.environ.get("GEMINI_EMBED_PAUSE_S", "0") or "0")
        embedder = GeminiCorpusEmbedder(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_embed_model or "models/text-embedding-004",
            max_dim=max_dim,
            request_pause_s=pause_s,
        )
    elif backend == "vertex":
        if not settings.gcp_project_id:
            print(
                "[build-corpus] ERROR: CORPUS_EMBED_BACKEND=vertex requires "
                "GCP_PROJECT_ID or GOOGLE_CLOUD_PROJECT."
            )
            return 1
        from ..vertex_embedder import VertexCorpusEmbedder

        pause_s = float(os.environ.get("VERTEX_EMBED_PAUSE_S", "0") or "0")
        embedder = VertexCorpusEmbedder(
            project_id=settings.gcp_project_id,
            location=settings.gcp_location or "us-central1",
            model_name=settings.vertex_embed_model or "gemini-embedding-001",
            max_dim=max_dim,
            request_pause_s=pause_s,
        )
    else:
        _dev = settings.hf_embed_device.strip() if settings.hf_embed_device else None
        embedder = HfEmbedder(
            model_name=settings.hf_embed_model,
            max_dim=max_dim,
            hf_token=settings.hf_token or None,
            batch_size=embed_batch_size,
            device=_dev,
        )

    docs_vecs = _embed_phase_with_resume(
        embedder=embedder,
        kind="docs",
        texts=docs_df["text"].tolist(),
        batch_size=embed_batch_size,
        out_dir=out_dir,
        out_name="docs_embeddings",
        max_dim=max_dim,
        resume=resume,
    )
    queries_vecs = _embed_phase_with_resume(
        embedder=embedder,
        kind="queries",
        texts=queries_df["text"].tolist(),
        batch_size=embed_batch_size,
        out_dir=out_dir,
        out_name="queries_embeddings",
        max_dim=max_dim,
        resume=resume,
    )

    write_embeddings(out_dir, "docs_embeddings", docs_vecs)
    write_embeddings(out_dir, "queries_embeddings", queries_vecs)

    # Use the model's actual output dim (may be < max_dim for non-Matryoshka
    # models like bge-small-en-v1.5 which outputs 384d, not 768d).
    actual_dim = int(docs_vecs.shape[1])

    # Derive supported dims: every power-of-2 step up to actual_dim.
    supported = [d for d in SUPPORTED_DIMS if d <= actual_dim]
    if actual_dim not in supported:
        supported.append(actual_dim)

    manifest = {
        "preset":           preset,
        "target_docs":      target_docs,
        "target_queries":   target_queries,
        "actual_docs":      int(len(docs_df)),
        "actual_queries":   int(len(queries_df)),
        "source_dim":       actual_dim,
        "supported_dims":   sorted(supported),
        "embed_model": (
            settings.vertex_embed_model
            if backend == "vertex"
            else settings.gemini_embed_model
            if backend == "gemini"
            else settings.hf_embed_model
        ),
        "embed_backend":    backend,
        "embed_batch_size": int(embed_batch_size),
        "seed":             int(seed),
        "has_metadata":     with_metadata,
        "created_at_utc":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "doc_sources":      _source_counts(docs_df),
        "query_sources":    _source_counts(queries_df),
    }
    write_manifest(out_dir, manifest)
    print(f"[build-corpus] manifest.json written. Build complete: {out_dir}")
    return 0


def _embed_phase_with_resume(
    *,
    embedder: Any,
    kind: str,
    texts: list[str],
    batch_size: int,
    out_dir: Path,
    out_name: str,
    max_dim: int,
    resume: bool,
) -> np.ndarray:
    expected = (len(texts), max_dim)
    if resume:
        cached = existing_embeddings(out_dir, out_name, expected)
        if cached is not None:
            print(
                f"[build-corpus] Resume: {out_name}.npy already has shape {expected}, "
                f"skipping {len(texts):,} {kind} embeddings."
            )
            return cached
    return _embed_phase(
        embedder=embedder,
        kind=kind,
        texts=texts,
        batch_size=batch_size,
        out_dir=out_dir,
        out_name=out_name,
        max_dim=max_dim,
    )


def _embed_phase(
    *,
    embedder: Any,
    kind: str,
    texts: list[str],
    batch_size: int,
    out_dir: Path,
    out_name: str,
    max_dim: int,
) -> np.ndarray:
    n = len(texts)
    print(
        f"[build-corpus] Embedding {n:,} {kind} at dim={max_dim}, "
        f"batch_size={batch_size}..."
    )

    # Defer array allocation until we know the model's actual output dim.
    # If the model outputs fewer dims than max_dim (e.g. bge-small=384 vs
    # max_dim=768), we use the model's actual dim to avoid shape mismatches.
    out: np.ndarray | None = None
    actual_dim = max_dim
    last_checkpoint = 0
    t_start = time.perf_counter()
    last_print = t_start

    for i in range(0, n, batch_size):
        batch = texts[i : i + batch_size]
        if kind == "docs":
            vectors = embedder.embed_documents(batch)
        elif kind == "queries":
            vectors = embedder.embed_queries(batch)
        else:
            raise ValueError(f"unknown kind {kind!r}")

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch at offset {i}: "
                f"requested {len(batch)}, got {len(vectors)}."
            )

        arr = np.asarray(vectors, dtype=np.float32)
        if out is None:
            # First batch: discover actual output dimension.
            actual_dim = arr.shape[1]
            if actual_dim != max_dim:
                print(
                    f"[build-corpus] Model output dim={actual_dim} "
                    f"(max_dim configured as {max_dim}; using {actual_dim})."
                )
            out = np.zeros((n, actual_dim), dtype=np.float32)
        out[i : i + len(batch)] = arr

        now = time.perf_counter()
        if now - last_print >= 5.0 or (i + len(batch)) == n:
            done = i + len(batch)
            rate = done / max(now - t_start, 1e-9)
            eta_s = (n - done) / max(rate, 1e-9)
            print(
                f"[build-corpus]   {kind} {done:,}/{n:,} "
                f"({done / n * 100:.1f}%, {rate:.0f}/s, ETA {eta_s:.0f}s)"
            )
            last_print = now

        if (i + len(batch)) - last_checkpoint >= _CHECKPOINT_EVERY:
            filled = i + len(batch)
            write_embeddings(out_dir, out_name, out[:filled])
            last_checkpoint = filled

    elapsed = time.perf_counter() - t_start
    print(
        f"[build-corpus] {kind} embedded in {elapsed:.1f}s "
        f"({n / max(elapsed, 1e-9):.0f}/s)."
    )
    return out if out is not None else np.zeros((0, actual_dim), dtype=np.float32)


def _source_counts(df: pd.DataFrame) -> dict[str, int]:
    if "source" not in df.columns:
        return {}
    return {str(k): int(v) for k, v in df["source"].value_counts().items()}


def _add_synthetic_metadata(df: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    """
    Add synthetic metadata columns to a docs DataFrame for filter-selectivity tests.

    Columns: ``category`` (keyword), ``tenant_id`` (keyword), ``created_at`` (ISO-8601).
    Values are deterministic for a given seed.
    """
    import random as _random
    from datetime import datetime, timedelta

    rng = _random.Random(seed + 9999)

    categories = [
        "infrastructure", "application", "security", "network",
        "database", "storage", "compute", "identity",
        "monitoring", "backup", "compliance", "billing",
        "support", "release", "change", "incident",
    ]
    tenants = [f"tenant-{i:03d}" for i in range(10)]
    base_date = datetime(2022, 1, 1)
    span_days = 365 * 3

    n = len(df)
    df = df.copy()
    df["category"]   = [rng.choice(categories) for _ in range(n)]
    df["tenant_id"]  = [rng.choice(tenants) for _ in range(n)]
    df["created_at"] = [
        (base_date + timedelta(days=rng.randint(0, span_days))).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        for _ in range(n)
    ]
    return df
