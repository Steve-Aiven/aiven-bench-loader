"""
Vertex AI text embeddings — optional corpus-build backend.

Uses Application Default Credentials (``gcloud auth application-default login``),
same pattern as the sister ``aiven_semantic_search`` demo with OpenSearch.

``gemini-embedding-001`` accepts **one text per predict call**; we serialize
requests and optionally throttle with ``VERTEX_EMBED_PAUSE_S``.

Requires: ``pip install -e '.[build-vertex]'`` (``google-cloud-aiplatform``).

Environment (via :class:`~aiven_semantic_search_bench.config.Settings`):

- ``CORPUS_EMBED_BACKEND=vertex``
- ``GCP_PROJECT_ID`` or ``GOOGLE_CLOUD_PROJECT``
- ``GCP_LOCATION`` — region (default ``us-central1``)
- ``VERTEX_EMBED_MODEL`` — default ``gemini-embedding-001``
- ``HF_EMBED_MAX_DIM`` — passed as ``output_dimensionality`` (e.g. 768 or 3072)

See: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class VertexCorpusEmbedder:
    """Same surface as :class:`HfEmbedder` / :class:`GeminiCorpusEmbedder`."""

    project_id: str
    location: str = "us-central1"
    model_name: str = "gemini-embedding-001"
    max_dim: int = 3072
    request_pause_s: float = 0.0
    _model: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pid = self.project_id.strip()
        if not pid:
            raise ValueError("GCP project id is empty (set GCP_PROJECT_ID)")
        try:
            import vertexai  # type: ignore[import-untyped]
            from vertexai.language_models import TextEmbeddingModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "CORPUS_EMBED_BACKEND=vertex requires google-cloud-aiplatform. "
                "Install with: pip install -e '.[build-vertex]'"
            ) from exc

        vertexai.init(project=pid, location=self.location.strip() or "us-central1")
        self._model = TextEmbeddingModel.from_pretrained(self.model_name)
        print(
            f"[vertex-embedder] project={pid!r} location={self.location!r} "
            f"model={self.model_name!r} output_dimensionality={self.max_dim} "
            "(ADC — local CPU/GPU idle)"
        )

    def _truncate_text(self, text: str) -> str:
        if len(text) > 16_000:
            return text[:16_000]
        return text

    def _embed_batch(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        from vertexai.language_models import TextEmbeddingInput  # type: ignore[import-untyped]

        kwargs: dict[str, object] = {"auto_truncate": True}
        if self.max_dim > 0:
            kwargs["output_dimensionality"] = int(self.max_dim)

        out: list[list[float]] = []
        for t in texts:
            tt = self._truncate_text(t)
            inp = TextEmbeddingInput(tt, task_type)
            last_err: Exception | None = None
            for attempt in range(5):
                try:
                    emb_list = self._model.get_embeddings([inp], **kwargs)
                    if not emb_list:
                        raise RuntimeError("Vertex returned empty embeddings list")
                    vec = np.asarray(emb_list[0].values, dtype=np.float32)
                    if vec.ndim != 1:
                        vec = vec.reshape(-1)
                    n = float(np.linalg.norm(vec))
                    if n > 0:
                        vec = vec / n
                    out.append(vec.tolist())
                    break
                except Exception as exc:
                    last_err = exc
                    time.sleep(0.5 * (2**attempt))
            else:
                raise RuntimeError(
                    f"Vertex embed failed after retries: {last_err}"
                ) from last_err
            if self.request_pause_s > 0:
                time.sleep(self.request_pause_s)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._embed_batch(texts, task_type="RETRIEVAL_QUERY")
