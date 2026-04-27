"""
Brute-force ground truth computation.

``build_groundtruth`` loads the pre-built corpus embeddings, computes the
exact top-K nearest neighbours for every query against every document using
chunked NumPy cosine similarity (vectors are already L2-normalised so
cosine similarity = dot product), and writes the result to
``corpus/qrels.npy``.

The output is a uint32 array of shape (n_queries, K) where each row holds
the integer row-indices (into docs_embeddings.npy) of the K nearest
neighbours in descending similarity order.

Memory note
-----------
A full pairwise distance matrix for Q queries × D docs at MAX_DIM (3072
float32) is Q × D × 4 bytes.  For 100k × 100k that is ~37 GB — far too
large to hold in RAM.  We therefore process queries in chunks of
``chunk_q`` rows and keep only the top-K per chunk, which requires only
``chunk_q × D × 4`` bytes at once (~1.2 GB for chunk_q=100, D=100k).

The chunked approach is ~2–3× slower than the batched numpy approach but
works on an 8 GB laptop for up to ~1M docs / ~10k queries.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .io import manifest_path, write_manifest
from .sources import MAX_DIM

_QRELS_NAME = "qrels.npy"


def build_groundtruth(
    corpus_dir: str | Path,
    *,
    k: int = 100,
    chunk_q: int = 500,
) -> int:
    """
    Compute brute-force top-K nearest neighbours and write ``qrels.npy``.

    Parameters
    ----------
    corpus_dir:
        Directory produced by ``bench-build-corpus``.
    k:
        Number of nearest neighbours per query to store (default 100;
        recall@K for K ≤ 100 can be computed from the result).
    chunk_q:
        Number of query rows processed at once.  Reduce if you run out of
        RAM; increasing speeds up the computation.

    Returns 0 on success, 1 on error.
    """
    corpus_dir = Path(corpus_dir)
    mpath = manifest_path(corpus_dir)
    if not mpath.exists():
        print(
            f"[groundtruth] ERROR: no corpus manifest at {mpath}. "
            "Run bench-build-corpus first."
        )
        return 1

    manifest = json.loads(mpath.read_text())
    source_dim = int(manifest.get("source_dim", MAX_DIM))

    docs_npy = corpus_dir / "docs_embeddings.npy"
    queries_npy = corpus_dir / "queries_embeddings.npy"

    if not docs_npy.exists() or not queries_npy.exists():
        print("[groundtruth] ERROR: embedding .npy files not found in corpus dir.")
        return 1

    print(f"[groundtruth] Loading docs embeddings from {docs_npy} ...")
    docs = np.load(docs_npy, mmap_mode="r")       # shape (D, source_dim)
    print(f"[groundtruth] Loading queries embeddings from {queries_npy} ...")
    queries = np.load(queries_npy, mmap_mode="r") # shape (Q, source_dim)

    D, Q = docs.shape[0], queries.shape[0]
    print(f"[groundtruth] {D:,} docs × {Q:,} queries at dim={source_dim}  k={k}")

    # Vectors are stored raw (not L2-normalised).  Normalise in-memory
    # before computing cosine similarity.
    print("[groundtruth] Normalising doc vectors ...")
    docs_norm = np.array(docs, dtype=np.float32)
    doc_norms = np.linalg.norm(docs_norm, axis=1, keepdims=True)
    doc_norms = np.where(doc_norms == 0.0, 1.0, doc_norms)
    docs_norm /= doc_norms

    k_actual = min(k, D)
    qrels = np.zeros((Q, k_actual), dtype=np.uint32)

    t_start = time.perf_counter()
    last_print = t_start

    for q_start in range(0, Q, chunk_q):
        q_end = min(q_start + chunk_q, Q)
        chunk = np.array(queries[q_start:q_end], dtype=np.float32)

        # Normalise query chunk.
        q_norms = np.linalg.norm(chunk, axis=1, keepdims=True)
        q_norms = np.where(q_norms == 0.0, 1.0, q_norms)
        chunk /= q_norms

        # Cosine similarity = dot product of normalised vectors.
        # Shape: (chunk_size, D)
        sims = chunk @ docs_norm.T

        # Top-K indices per query (descending similarity).
        # np.argpartition is O(D) per row; argsort of the K selected is O(K log K).
        top_k_idx = np.argpartition(sims, -k_actual, axis=1)[:, -k_actual:]
        for i, row_idx in enumerate(top_k_idx):
            sorted_idx = row_idx[np.argsort(-sims[i, row_idx])]
            qrels[q_start + i] = sorted_idx.astype(np.uint32)

        now = time.perf_counter()
        if now - last_print >= 5.0 or q_end == Q:
            done = q_end
            rate = done / max(now - t_start, 1e-9)
            eta_s = (Q - done) / max(rate, 1e-9)
            print(
                f"[groundtruth]   {done:,}/{Q:,} queries "
                f"({done / Q * 100:.1f}%, {rate:.0f} q/s, ETA {eta_s:.0f}s)"
            )
            last_print = now

    out_path = corpus_dir / _QRELS_NAME
    np.save(str(out_path), qrels)
    elapsed = time.perf_counter() - t_start
    print(f"[groundtruth] Wrote {out_path}  shape={qrels.shape}  ({elapsed:.1f}s)")

    # Bump manifest with groundtruth metadata.
    manifest["groundtruth_k"] = k_actual
    manifest["groundtruth_dim"] = source_dim
    manifest["groundtruth_built_at_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    write_manifest(corpus_dir, manifest)
    print("[groundtruth] manifest.json updated.")
    return 0
