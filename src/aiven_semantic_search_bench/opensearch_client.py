"""
OpenSearch client + index helpers for benchmarking.

Key additions over the original minimal client:

- ``KnnSpec`` — frozen dataclass that encodes every axis of the k-NN
  configuration matrix (engine, method, mode, quantization, HNSW params,
  data type, derived_source, hybrid text/metadata fields).  One ``KnnSpec``
  fully describes an index; passing it to ``build_index_mapping`` and
  ``reset_index`` keeps all the version-feature logic in one place.

- ``build_index_mapping(spec)`` — produces the OpenSearch index body for
  any valid ``KnnSpec`` combination.  Invalid combos (e.g. ``ivf`` with
  ``lucene``, ``on_disk`` with ``lucene``) are rejected early with a clear
  message.

- ``reset_index`` / ``wait_for_status`` / ``get_index_stats`` are unchanged
  in semantics; ``reset_index`` now accepts a ``KnnSpec``.

``description_vector`` is kept as the knn_vector field name so existing
reports remain comparable.  The ``content`` (text) and ``metadata`` (object)
fields are additive and only present when ``spec.with_text`` /
``spec.with_metadata`` are True.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse, unquote

from opensearchpy import OpenSearch

# ── KnnSpec ───────────────────────────────────────────────────────────────────

Engine = Literal["faiss", "lucene"]
Method = Literal["hnsw", "ivf"]
SpaceType = Literal["cosinesimil", "innerproduct", "l2"]
Mode = Literal["in_memory", "on_disk"]
CompressionLevel = Literal["none", "1x", "2x", "4x", "8x", "16x", "32x"]
DataType = Literal["float", "byte", "fp16", "binary"]


@dataclass(frozen=True)
class KnnSpec:
    """
    Full specification for a k-NN index configuration.

    Every axis of the benchmark matrix lives here so that a single object
    can be passed around, serialized to JSON (see ``to_dict``), and used to
    drive both ``build_index_mapping`` and report params.

    Validation rules (enforced by ``validate()``, called by
    ``build_index_mapping``):
      - ``ivf`` requires ``engine=faiss``
      - ``on_disk`` requires ``engine=faiss``
      - ``compression`` != ``none`` requires ``mode=on_disk``
      - ``data_type=byte`` requires ``engine=faiss``
      - ``data_type=fp16`` / ``binary`` require ``engine=faiss``
      - ``derived_source`` is accepted on all versions but is a no-op on 2.17
    """

    embed_dim: int

    engine: Engine = "faiss"
    method: Method = "hnsw"
    space_type: SpaceType = "cosinesimil"
    mode: Mode = "in_memory"
    compression: CompressionLevel = "none"
    data_type: DataType = "float"
    derived_source: bool = False

    # HNSW graph parameters
    m: int = 16
    ef_construction: int = 128
    ef_search: int = 256

    # Additive fields for hybrid / filter tests
    with_text: bool = False
    with_metadata: bool = False

    def validate(self) -> None:
        """Raise ValueError for any invalid combination."""
        if self.method == "ivf" and self.engine != "faiss":
            raise ValueError("method='ivf' requires engine='faiss'")
        if self.mode == "on_disk" and self.engine != "faiss":
            raise ValueError("mode='on_disk' requires engine='faiss'")
        if self.compression != "none" and self.mode != "on_disk":
            raise ValueError(
                f"compression='{self.compression}' requires mode='on_disk'"
            )
        if self.data_type in ("byte", "fp16", "binary") and self.engine != "faiss":
            raise ValueError(
                f"data_type='{self.data_type}' requires engine='faiss'"
            )

    def label(self) -> str:
        """Short human-readable label for report filenames and chart axes."""
        parts = [self.engine, self.mode]
        if self.compression != "none":
            parts.append(self.compression)
        if self.data_type != "float":
            parts.append(self.data_type)
        if self.derived_source:
            parts.append("derived")
        return "/".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict for storing in report params."""
        return {
            "embed_dim":       self.embed_dim,
            "engine":          self.engine,
            "method":          self.method,
            "space_type":      self.space_type,
            "mode":            self.mode,
            "compression":     self.compression,
            "data_type":       self.data_type,
            "derived_source":  self.derived_source,
            "m":               self.m,
            "ef_construction": self.ef_construction,
            "ef_search":       self.ef_search,
            "with_text":       self.with_text,
            "with_metadata":   self.with_metadata,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "KnnSpec":
        return KnnSpec(
            embed_dim=int(d["embed_dim"]),
            engine=d.get("engine", "faiss"),
            method=d.get("method", "hnsw"),
            space_type=d.get("space_type", "cosinesimil"),
            mode=d.get("mode", "in_memory"),
            compression=d.get("compression", "none"),
            data_type=d.get("data_type", "float"),
            derived_source=bool(d.get("derived_source", False)),
            m=int(d.get("m", 16)),
            ef_construction=int(d.get("ef_construction", 128)),
            ef_search=int(d.get("ef_search", 256)),
            with_text=bool(d.get("with_text", False)),
            with_metadata=bool(d.get("with_metadata", False)),
        )


# ── URI parsing + client factory ─────────────────────────────────────────────


@dataclass(frozen=True)
class ParsedOpenSearchUri:
    host: str
    port: int
    username: str | None
    password: str | None
    use_ssl: bool


def parse_opensearch_uri(uri: str) -> ParsedOpenSearchUri:
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported OPENSEARCH_URI scheme: {parsed.scheme!r}")
    if not parsed.hostname or not parsed.port:
        raise ValueError("OPENSEARCH_URI must include hostname and port")

    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None

    return ParsedOpenSearchUri(
        host=parsed.hostname,
        port=int(parsed.port),
        username=username,
        password=password,
        use_ssl=(parsed.scheme == "https"),
    )


def get_opensearch_client(uri: str, *, timeout: int = 60) -> OpenSearch:
    """
    Build a TLS-verified OpenSearch client.

    The higher default timeout (60 s) covers benchmark workloads that
    occasionally hit slow operations — waking a paused service, large bulk
    requests, ``indices.stats`` on a very large index.
    """
    p = parse_opensearch_uri(uri)
    http_auth = (p.username, p.password) if p.username and p.password else None

    return OpenSearch(
        hosts=[{"host": p.host, "port": p.port}],
        http_auth=http_auth,
        use_ssl=p.use_ssl,
        verify_certs=True,
        ssl_show_warn=False,
        http_compress=True,
        timeout=timeout,
    )


# ── Index mapping ─────────────────────────────────────────────────────────────


def build_index_mapping(spec: KnnSpec) -> dict[str, Any]:
    """
    Build the OpenSearch index body for the given ``KnnSpec``.

    Raises ``ValueError`` for invalid combinations before attempting
    to create the index so failures surface during the UI matrix-preview
    stage rather than mid-run.
    """
    spec.validate()

    # ── knn_vector method block ───────────────────────────────────────────
    method_params: dict[str, Any] = {}
    if spec.method == "hnsw":
        method_params = {
            "ef_construction": spec.ef_construction,
            "m": spec.m,
        }
    elif spec.method == "ivf":
        # IVF needs nlist tuned per dataset; 256 is a safe default for 100k–1M.
        method_params = {"nlist": 256, "nprobes": 16}

    method_block: dict[str, Any] = {
        "name": spec.method,
        "engine": spec.engine,
        "space_type": spec.space_type,
        "parameters": method_params,
    }

    vector_field: dict[str, Any] = {
        "type": "knn_vector",
        "dimension": spec.embed_dim,
        "method": method_block,
    }

    # data_type is a top-level field attribute (not inside method).
    if spec.data_type != "float":
        vector_field["data_type"] = spec.data_type

    # on_disk mode + compression level (Faiss only, 2.17+)
    if spec.mode == "on_disk":
        vector_field["mode"] = "on_disk"
        if spec.compression != "none":
            vector_field["compression_level"] = spec.compression

    # ── index settings ────────────────────────────────────────────────────
    index_settings: dict[str, Any] = {
        "number_of_shards": "1",
        "number_of_replicas": "0",
        "knn": True,
    }

    # derived_source reduces storage overhead; available in 2.19+ (experimental)
    # and more mature in 3.x. We include it unconditionally when requested
    # and let OpenSearch version surface an error if not supported.
    if spec.derived_source:
        index_settings["knn.derived_source.enabled"] = True

    # ── properties ────────────────────────────────────────────────────────
    properties: dict[str, Any] = {
        "description":        {"type": "text"},
        "source":             {"type": "keyword"},
        "description_vector": vector_field,
    }

    # Additional fields for hybrid + filter benchmarks.
    if spec.with_text:
        # Full-text content for BM25 scoring in hybrid queries.
        properties["content"] = {"type": "text"}

    if spec.with_metadata:
        properties["metadata"] = {
            "type": "object",
            "properties": {
                "category":   {"type": "keyword"},
                "tenant_id":  {"type": "keyword"},
                "created_at": {"type": "date", "format": "strict_date_optional_time"},
            },
        }

    return {
        "settings": {"index": index_settings},
        "mappings": {"properties": properties},
    }


# ── Index lifecycle ───────────────────────────────────────────────────────────


def reset_index(client: OpenSearch, index: str, *, spec: KnnSpec) -> None:
    """
    Delete and recreate the benchmark index.

    Every benchmark starts from a clean index so that previous runs do not
    skew measurements (residual segments, warmed caches).
    """
    if client.indices.exists(index=index):
        client.indices.delete(index=index)
    client.indices.create(index=index, body=build_index_mapping(spec))


def wait_for_status(
    client: OpenSearch,
    index: str,
    *,
    status: str = "green",
    timeout: str = "30s",
) -> None:
    client.cluster.health(index=index, wait_for_status=status, timeout=timeout)


def get_index_stats(client: OpenSearch, index: str) -> dict[str, Any]:
    """
    Return a small subset of ``indices.stats`` useful for benchmark reports.

    Only doc count and primary store size are fetched to keep report payloads
    readable.
    """
    raw = client.indices.stats(index=index, metric=["docs", "store"])
    primaries = raw["_all"]["primaries"]
    return {
        "doc_count":        primaries["docs"]["count"],
        "deleted_docs":     primaries["docs"]["deleted"],
        "store_size_bytes": primaries["store"]["size_in_bytes"],
    }
