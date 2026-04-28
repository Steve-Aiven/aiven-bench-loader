"""
Hugging Face sentence-transformers embedding wrapper.

Replaces the Gemini/Vertex AI embedder.  The model runs locally — no API
token, no GCP project, no billing account required.  The first run downloads
model weights to the HuggingFace cache (``~/.cache/huggingface`` or
``HF_HOME``); subsequent runs are fully offline.

Chosen model: ``nomic-ai/nomic-embed-text-v1.5``
  - 768-dim output, Matryoshka Representation Learning (MRL) trained.
  - Apache 2.0 licence.
  - State-of-the-art quality for its size on MTEB retrieval benchmarks.
  - The MRL property means truncating the prefix to 256, 512, or 768 dims
    and L2-renormalising gives valid representations — exactly what the
    corpus loader does at benchmark time.

Task prefixes (required by nomic-embed):
  - Documents: ``"search_document: "``
  - Queries:   ``"search_query: "``

Other MRL-compatible models that work as drop-in replacements (set via
``HF_EMBED_MODEL`` env var / ``--embed-model`` flag):
  - ``mixedbread-ai/mxbai-embed-large-v1``  (1024-dim)
  - ``Snowflake/snowflake-arctic-embed-m-v2.0``  (768-dim)
  - ``sentence-transformers/all-mpnet-base-v2``  (768-dim, no prefix needed)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Imported lazily inside methods so the rest of the package imports cleanly
# even when sentence-transformers is not installed (e.g. measurement-only
# containers that never build a corpus).


_DOC_PREFIX = "search_document: "
_QUERY_PREFIX = "search_query: "

# Models that require task prefixes (nomic family).
_PREFIX_MODELS: set[str] = {
    "nomic-ai/nomic-embed-text-v1.5",
    "nomic-ai/nomic-embed-text-v1",
}

# Default maximum output dimensionality.  The corpus stores embeddings at
# this dimension; benchmarks truncate at load time via Matryoshka slicing.
DEFAULT_MAX_DIM = 768

# Models that do NOT need trust_remote_code.  Every other model defaults True
# only if the model_name is in _PREFIX_MODELS (i.e. nomic family).
_NO_REMOTE_CODE: set[str] = {
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-MiniLM-L12-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "mixedbread-ai/mxbai-embed-large-v1",
    "Snowflake/snowflake-arctic-embed-m-v2.0",
}

# Recommended models with their native output dimensions.
# Entries: (display_name, model_id, dim, description)
RECOMMENDED_MODELS: list[tuple[str, str, int, str]] = [
    (
        "nomic-embed-text-v1.5  (768-dim, best quality)",
        "nomic-ai/nomic-embed-text-v1.5",
        768,
        "MRL-trained, Apache 2.0.  Best retrieval quality in its size class.  "
        "Requires trust_remote_code=True.",
    ),
    (
        "bge-small-en-v1.5  (384-dim, fastest CPU/GPU)",
        "BAAI/bge-small-en-v1.5",
        384,
        "~4× faster than nomic on CPU, ~2× on MPS.  Good quality for benchmarking. "
        "No trust_remote_code needed.",
    ),
    (
        "all-MiniLM-L6-v2  (384-dim, very fast CPU)",
        "sentence-transformers/all-MiniLM-L6-v2",
        384,
        "Classic, widely-used 80 MB model.  Fastest CPU option. "
        "No trust_remote_code needed.",
    ),
    (
        "mxbai-embed-large-v1  (1024-dim, highest quality)",
        "mixedbread-ai/mxbai-embed-large-v1",
        1024,
        "Highest retrieval quality; MRL-trained.  Needs HF_EMBED_MAX_DIM=1024. "
        "Best on MPS / GPU.",
    ),
]


def _best_device() -> str:
    """
    Auto-detect the fastest available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.
    MPS is available on Mac M-series chips when running natively (not in Docker).
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _optimal_batch_size(device: str) -> int:
    """Larger batches saturate GPU/MPS faster; smaller batches avoid OOM on CPU."""
    if device == "cuda":
        return 512
    if device == "mps":
        return 256
    # CPU: use all available cores via torch inter-op threads.
    import os
    cores = os.cpu_count() or 4
    # 64 per core saturates most CPUs without excessive memory pressure.
    return min(256, cores * 16)


@dataclass
class HfEmbedder:
    """
    Local sentence-transformers embedding client with automatic device selection.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to ``nomic-ai/nomic-embed-text-v1.5``.
        See ``RECOMMENDED_MODELS`` for a curated list with speed/quality notes.
    max_dim:
        Maximum output dimensionality stored in the corpus.  Embeddings are
        L2-normalised at this dimension; benchmark commands Matryoshka-truncate
        to the requested ``--embed-dim``.
    hf_token:
        Optional HuggingFace Hub token (gated models only).
    batch_size:
        Texts per forward pass.  Defaults to ``None`` which auto-selects based
        on the device (512 for CUDA, 256 for MPS, 64–256 for CPU).
    trust_remote_code:
        Required for nomic-embed models.  Auto-set based on model name if None.
    device:
        PyTorch device string.  ``None`` auto-detects: CUDA > MPS > CPU.
        On Apple Silicon Macs running natively (not Docker), this will be
        ``"mps"`` giving ~5–10× speedup over CPU.
    """

    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    max_dim: int = DEFAULT_MAX_DIM
    hf_token: str | None = None
    batch_size: int | None = None
    trust_remote_code: bool | None = None
    device: str | None = None
    _model: object = field(init=False, repr=False, default=None)
    _device: str = field(init=False, repr=False, default="cpu")
    _batch_size: int = field(init=False, repr=False, default=64)

    def __post_init__(self) -> None:
        import os
        from sentence_transformers import SentenceTransformer

        self._device = self.device if self.device else _best_device()
        self._batch_size = self.batch_size if self.batch_size else _optimal_batch_size(self._device)

        # Tune PyTorch CPU thread count when running on CPU so we use all cores.
        if self._device == "cpu":
            try:
                import torch
                n_threads = os.cpu_count() or 4
                torch.set_num_threads(n_threads)
                torch.set_num_interop_threads(max(1, n_threads // 2))
            except Exception:
                pass

        needs_remote = self.model_name in _PREFIX_MODELS and self.model_name not in _NO_REMOTE_CODE
        trc = self.trust_remote_code if self.trust_remote_code is not None else needs_remote

        kwargs: dict = {"trust_remote_code": trc, "device": self._device}
        if self.hf_token:
            kwargs["token"] = self.hf_token

        print(
            f"[hf-embedder] Loading {self.model_name!r} "
            f"(device={self._device}, batch={self._batch_size}, max_dim={self.max_dim})"
        )
        self._model = SentenceTransformer(self.model_name, **kwargs)
        print(f"[hf-embedder] Model ready.")

    def _needs_prefix(self) -> bool:
        return self.model_name in _PREFIX_MODELS and self.model_name not in _NO_REMOTE_CODE

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Run the model and return float32 lists truncated to max_dim."""
        import numpy as np

        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        # Truncate to max_dim (Matryoshka prefix) and re-normalise.
        vecs = np.array(vecs, dtype=np.float32)[:, : self.max_dim]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        vecs /= norms
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._needs_prefix():
            texts = [_DOC_PREFIX + t for t in texts]
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        if self._needs_prefix():
            texts = [_QUERY_PREFIX + t for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self.embed_queries([text])[0]
