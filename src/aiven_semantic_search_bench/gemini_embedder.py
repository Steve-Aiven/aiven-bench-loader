"""
Google Gemini embedding API — optional corpus-build backend.

Use when you want to avoid local GPU/CPU load during ``bench-build-corpus``.
Benchmark commands never call this; they only read ``corpus/*.npy``.

Requires: ``pip install 'google-generativeai>=0.8,<1'`` (see ``[build-gemini]`` extra).

Environment (via :class:`~aiven_semantic_search_bench.config.Settings`):

- ``CORPUS_EMBED_BACKEND=gemini``
- ``GEMINI_API_KEY`` — API key from Google AI Studio
- ``GEMINI_EMBED_MODEL`` — default ``models/text-embedding-004`` (768-d output)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


def _embedding_vector(resp: object) -> np.ndarray:
    """Normalize SDK responses (dict-like or EmbedContentResponse) to a 1-d float array."""
    if isinstance(resp, dict):
        emb = resp.get("embedding")
    else:
        emb = getattr(resp, "embedding", None)
    if emb is None:
        raise RuntimeError(f"Gemini embed response missing 'embedding': {resp!r}")
    if hasattr(emb, "__iter__") and not isinstance(emb, (str, bytes, dict)):
        vec = np.asarray(list(emb), dtype=np.float32)
    else:
        vec = np.asarray(emb, dtype=np.float32)
    if vec.ndim != 1:
        vec = vec.reshape(-1)
    return vec


@dataclass
class GeminiCorpusEmbedder:
    """Same surface as :class:`HfEmbedder` for ``build_corpus`` embedding phases."""

    api_key: str
    model_name: str = "models/text-embedding-004"
    max_dim: int = 768
    request_pause_s: float = 0.0
    _configured: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("GEMINI_API_KEY is empty")
        try:
            import google.generativeai as genai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "CORPUS_EMBED_BACKEND=gemini requires google-generativeai. "
                "Install with: pip install -e '.[build-gemini]'"
            ) from exc
        genai.configure(api_key=self.api_key.strip())
        self._genai = genai
        self._configured = True
        print(
            f"[gemini-embedder] model={self.model_name!r} "
            f"max_dim={self.max_dim} (remote API — local CPU/GPU mostly idle)"
        )

    def _truncate_text(self, text: str) -> str:
        # Stay under typical token limits; IR snippets are short anyway.
        if len(text) > 16_000:
            return text[:16_000]
        return text

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        assert self._configured
        genai = self._genai
        out: list[list[float]] = []
        for t in texts:
            tt = self._truncate_text(t)
            last_err: Exception | None = None
            for attempt in range(5):
                try:
                    resp = genai.embed_content(
                        model=self.model_name,
                        content=tt,
                        task_type=task_type,
                    )
                    vec = _embedding_vector(resp)
                    if vec.size > self.max_dim:
                        vec = vec[: self.max_dim]
                    n = float(np.linalg.norm(vec))
                    if n > 0:
                        vec = vec / n
                    out.append(vec.tolist())
                    break
                except Exception as exc:
                    last_err = exc
                    time.sleep(0.5 * (2**attempt))
            else:
                raise RuntimeError(f"Gemini embed failed after retries: {last_err}") from last_err
            if self.request_pause_s > 0:
                time.sleep(self.request_pause_s)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts, task_type="retrieval_document")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts, task_type="retrieval_query")
