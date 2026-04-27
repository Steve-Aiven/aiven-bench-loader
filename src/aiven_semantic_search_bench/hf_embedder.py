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


@dataclass
class HfEmbedder:
    """
    Local sentence-transformers embedding client.

    Parameters
    ----------
    model_name:
        HuggingFace model ID.  Defaults to ``nomic-ai/nomic-embed-text-v1.5``.
    max_dim:
        Maximum output dimensionality.  The model must support at least this
        many dimensions.  Embeddings are L2-normalised at this dimension
        before being stored; benchmark commands then Matryoshka-truncate to
        the requested ``--embed-dim``.
    hf_token:
        Optional HuggingFace Hub token.  Required only for gated models.
        Set via ``HF_TOKEN`` env var or pass directly.
    batch_size:
        Texts per forward pass through the model.  The embedder's
        ``embed_documents`` / ``embed_queries`` methods further chunk their
        input into batches of this size.
    trust_remote_code:
        Required for ``nomic-ai/nomic-embed-text-v1.5`` which ships a custom
        pooling layer.  Defaults to True.
    device:
        PyTorch device string.  Defaults to ``"cpu"``; set to ``"cuda"`` or
        ``"mps"`` for GPU acceleration.
    """

    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    max_dim: int = DEFAULT_MAX_DIM
    hf_token: str | None = None
    batch_size: int = 64
    trust_remote_code: bool = True
    device: str = "cpu"
    _model: object = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        kwargs: dict = {
            "trust_remote_code": self.trust_remote_code,
            "device": self.device,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token

        print(
            f"[hf-embedder] Loading model {self.model_name!r} "
            f"(device={self.device}, max_dim={self.max_dim}) …"
        )
        self._model = SentenceTransformer(self.model_name, **kwargs)
        print(f"[hf-embedder] Model loaded.")

    def _needs_prefix(self) -> bool:
        return self.model_name in _PREFIX_MODELS

    def _encode(self, texts: list[str]) -> list[list[float]]:
        """Run the model and return float32 lists truncated to max_dim."""
        import numpy as np

        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,   # L2-normalise at the model's native dim
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
