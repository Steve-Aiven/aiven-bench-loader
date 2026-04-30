"""
Configuration for the benchmarking tool.

Environment variables are loaded into a frozen ``Settings`` dataclass.
Secrets are kept out of source control — copy ``.env.example`` to ``.env``
and fill in the values.

What's required and when
------------------------
- ``OPENSEARCH_URI`` / ``OPENSEARCH_INDEX`` — needed by CLI measurement
  commands when not going through one of the UIs. The standalone NiceGUI
  dashboard resolves the URI from the Aiven API automatically; the loader
  API receives the URI inline with each /run request from the orchestrator;
  the CLI falls back to this env var.
- ``HF_EMBED_MODEL`` — the sentence-transformers model used by
  ``bench-build-corpus``.  Defaults to ``nomic-ai/nomic-embed-text-v1.5``.
- ``HF_TOKEN`` — optional HuggingFace Hub token.  Required only for gated
  models.  Open models like nomic-embed-text-v1.5 work without it.
- ``HF_EMBED_MAX_DIM`` — maximum embedding dimension to store in the corpus
  (default 768, matching the nomic model output).
- ``HF_EMBED_DEVICE`` — optional: ``mps`` (Apple GPU), ``cuda``, or ``cpu``.
  When unset, auto-detects (CUDA, then MPS on Apple Silicon, else CPU).
- ``CORPUS_EMBED_BACKEND`` — ``hf`` (default), ``gemini`` (Google AI Studio API
  key), or ``vertex`` (Vertex AI embeddings via ADC — matches GCP console billing).
- ``GEMINI_API_KEY`` / ``GEMINI_EMBED_MODEL`` — used when backend is ``gemini``.
- ``GCP_PROJECT_ID`` (or ``GOOGLE_CLOUD_PROJECT``), ``GCP_LOCATION``,
  ``VERTEX_EMBED_MODEL`` — used when backend is ``vertex``.
- ``EMBED_DIM`` — the dimension used at benchmark time (must be ≤
  ``HF_EMBED_MAX_DIM``; Matryoshka-truncated from the stored max).
- ``AIVEN_API_TOKEN / PROJECT / SERVICE_NAME`` — optional; only needed by
  the CLI ``bench-plan-change`` and the stress benchmark when it triggers
  a mid-run plan change. The UI flows do not read these env vars.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    opensearch_uri: str
    opensearch_index: str
    hf_embed_model: str
    hf_token: str               # empty string = no token (open models)
    hf_embed_max_dim: int
    hf_embed_device: str  # "" = auto (cuda > mps > cpu)
    corpus_embed_backend: str  # hf | gemini | vertex
    gemini_api_key: str
    gemini_embed_model: str
    gcp_project_id: str
    gcp_location: str
    vertex_embed_model: str
    embed_dim: int
    # Aiven REST API — only required for bench-plan-change CLI command.
    aiven_api_token: str
    aiven_project: str
    aiven_service_name: str

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            opensearch_uri=os.environ.get("OPENSEARCH_URI", ""),
            opensearch_index=os.environ.get("OPENSEARCH_INDEX", "bench"),
            hf_embed_model=os.environ.get(
                "HF_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"
            ),
            hf_token=os.environ.get("HF_TOKEN", ""),
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
            embed_dim=int(os.environ.get("EMBED_DIM", "768")),
            aiven_api_token=os.environ.get("AIVEN_API_TOKEN", ""),
            aiven_project=os.environ.get("AIVEN_PROJECT", ""),
            aiven_service_name=os.environ.get("AIVEN_SERVICE_NAME", ""),
        )

    def require_opensearch_uri(self) -> str:
        """Return the URI or raise a clear error if not configured."""
        if not self.opensearch_uri:
            raise RuntimeError(
                "OPENSEARCH_URI is not set. Either set it in .env or pass "
                "--opensearch-uri on the command line."
            )
        return self.opensearch_uri

    def require_aiven_api_credentials(self) -> None:
        """Validate that AIVEN_API_TOKEN, AIVEN_PROJECT, AIVEN_SERVICE_NAME are set."""
        missing = [
            name
            for name, value in [
                ("AIVEN_API_TOKEN", self.aiven_api_token),
                ("AIVEN_PROJECT", self.aiven_project),
                ("AIVEN_SERVICE_NAME", self.aiven_service_name),
            ]
            if not value
        ]
        if missing:
            raise RuntimeError(
                "Missing required environment variables for Aiven API access: "
                + ", ".join(missing)
            )
