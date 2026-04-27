"""
On-disk format and loader for the benchmark corpus.

Layout (under ``corpus_dir/``):
    manifest.json               metadata (preset, model, counts, dim)
    docs.parquet                doc_id, text, source
    queries.parquet             query_id, text, source
    docs_embeddings.npy         float32, shape (N, MAX_DIM)
    queries_embeddings.npy      float32, shape (M, MAX_DIM)

The embeddings are stored at MAX_DIM (3072) once. ``load_corpus`` slices
and L2-renormalizes the prefix at benchmark time to support 256, 512,
768, 1536, 3072 without re-embedding.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .sources import MAX_DIM, SUPPORTED_DIMS

_MANIFEST_NAME = "manifest.json"
_DOCS_PARQUET = "docs.parquet"
_QUERIES_PARQUET = "queries.parquet"
_DOCS_NPY = "docs_embeddings.npy"
_QUERIES_NPY = "queries_embeddings.npy"


def manifest_path(corpus_dir: str | Path) -> Path:
    return Path(corpus_dir) / _MANIFEST_NAME


@dataclass(frozen=True)
class CorpusBundle:
    """Everything a benchmark needs: text + pre-sliced + renormalized vectors."""

    docs: pd.DataFrame              # columns: doc_id, text, source
    doc_vectors: np.ndarray         # shape (N, embed_dim), float32, L2-normalized
    queries: pd.DataFrame           # columns: query_id, text, source
    query_vectors: np.ndarray       # shape (M, embed_dim), float32, L2-normalized
    embed_dim: int                  # the dimension benchmarks should use
    source_dim: int                 # the on-disk stored dim (typically MAX_DIM)
    manifest: dict[str, Any]


def load_corpus(corpus_dir: str | Path, embed_dim: int) -> CorpusBundle:
    """
    Read the persisted corpus and project to ``embed_dim``.

    ``embed_dim`` must be <= the stored ``source_dim`` and listed in the
    manifest's ``supported_dims``. Vectors are sliced and re-normalized
    to preserve cosine similarity at the target dimension (Matryoshka
    Representation Learning).

    Raises:
        FileNotFoundError if the corpus has not been built yet.
        ValueError if embed_dim is incompatible with the manifest.
    """
    corpus_dir = Path(corpus_dir)
    manifest_file = corpus_dir / _MANIFEST_NAME
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"No corpus manifest at {manifest_file}. "
            f"Run 'bench-build-corpus' first."
        )

    manifest = json.loads(manifest_file.read_text())
    source_dim = int(manifest["source_dim"])
    supported = manifest.get("supported_dims") or list(SUPPORTED_DIMS)

    if embed_dim > source_dim:
        raise ValueError(
            f"Requested embed_dim={embed_dim} exceeds stored source_dim={source_dim}. "
            f"Re-run bench-build-corpus with a higher --max-dim."
        )
    if embed_dim not in supported:
        raise ValueError(
            f"embed_dim={embed_dim} is not in this corpus's supported_dims={supported}. "
            f"Pick one of those, or rebuild the corpus."
        )

    docs_df = pd.read_parquet(corpus_dir / _DOCS_PARQUET)
    queries_df = pd.read_parquet(corpus_dir / _QUERIES_PARQUET)

    # mmap_mode='r' avoids reading the full file into memory when we only
    # need the prefix slice.
    docs_full = np.load(corpus_dir / _DOCS_NPY, mmap_mode="r")
    queries_full = np.load(corpus_dir / _QUERIES_NPY, mmap_mode="r")

    if docs_full.shape[1] != source_dim:
        raise ValueError(
            f"docs_embeddings.npy stores dim={docs_full.shape[1]} but manifest "
            f"claims source_dim={source_dim}. Corpus is corrupt."
        )

    doc_vecs = _matryoshka(docs_full, embed_dim)
    query_vecs = _matryoshka(queries_full, embed_dim)

    return CorpusBundle(
        docs=docs_df,
        doc_vectors=doc_vecs,
        queries=queries_df,
        query_vectors=query_vecs,
        embed_dim=embed_dim,
        source_dim=source_dim,
        manifest=manifest,
    )


def _matryoshka(vectors: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Slice each row to its first ``target_dim`` components and L2-renormalize.

    This is the standard MRL truncation: a Matryoshka-trained model
    encodes information hierarchically into the prefix, so
    ``vec[:k] / ||vec[:k]||`` is a valid k-dim representation that
    preserves cosine similarity for retrieval.

    Stored vectors are typically NOT pre-normalized (Vertex AI returns
    raw model outputs), so we normalize unconditionally even when
    target_dim == source_dim.
    """
    truncated = np.array(vectors[:, :target_dim], dtype=np.float32, copy=True)
    norms = np.linalg.norm(truncated, axis=1, keepdims=True)
    # Replace any zero norms with 1 to avoid NaN; an all-zero vector
    # stays all-zero, which OpenSearch handles fine.
    norms = np.where(norms == 0.0, 1.0, norms)
    return truncated / norms


def write_manifest(corpus_dir: str | Path, manifest: dict[str, Any]) -> None:
    """Atomically write the manifest as the LAST step of a successful build."""
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    tmp = corpus_dir / (_MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    tmp.replace(corpus_dir / _MANIFEST_NAME)


def write_dataframe(corpus_dir: str | Path, name: str, df: pd.DataFrame) -> None:
    """Write a parquet file by name (without extension)."""
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(corpus_dir / f"{name}.parquet", index=False)


def write_embeddings(corpus_dir: str | Path, name: str, vectors: np.ndarray) -> None:
    """
    Atomically save a numpy array to ``corpus_dir/{name}.npy``.

    Note on the file-handle dance: ``np.save(path, arr)`` silently appends
    ``.npy`` to a path that does not already end in ``.npy``, so a tmp
    path like ``foo.npy.tmp`` would actually be written to ``foo.npy.tmp.npy``
    and the subsequent rename would fail. Passing an open file handle
    bypasses that filename-mangling and gives us a real atomic-rename.
    """
    corpus_dir = Path(corpus_dir)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    target = corpus_dir / f"{name}.npy"
    tmp = corpus_dir / f"{name}.npy.tmp"
    with open(tmp, "wb") as f:
        np.save(f, vectors)
    tmp.replace(target)


def existing_embeddings(corpus_dir: str | Path, name: str, expected_shape: tuple[int, int]) -> np.ndarray | None:
    """
    Return previously-written embeddings for resume support, or None.

    Used by ``build_corpus`` to skip the docs or queries phase when its
    output file already exists with the expected ``(rows, dim)`` shape AND is
    fully populated. The checkpoint logic pre-allocates a zero-filled array
    and flushes it periodically, so a file with the right shape can still
    contain un-filled trailing rows (zero-norm vectors). We verify the last
    row has a non-trivial L2 norm as a proxy for "fully populated"; a
    legitimately zero embedding from Gemini is astronomically unlikely.
    """
    path = Path(corpus_dir) / f"{name}.npy"
    if not path.exists():
        return None
    arr = np.load(path, mmap_mode="r")
    # A partial checkpoint has the right number of columns but fewer rows.
    # A corrupt/mismatched build has a different column count — reject it.
    if arr.ndim != 2 or arr.shape[1] != expected_shape[1]:
        return None
    # Must be fully populated: all expected rows present and last row non-zero.
    if arr.shape[0] != expected_shape[0] or np.linalg.norm(arr[-1]) < 0.01:
        return None
    return np.array(arr, copy=True)
