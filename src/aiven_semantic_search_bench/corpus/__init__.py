"""
Offline corpus pipeline.

Benchmarks must isolate OpenSearch behavior from third-party dependencies.
Calling Gemini at benchmark time conflates Vertex AI throughput, network
RTT to Google, and Aiven OpenSearch performance. This package solves that
by pre-computing every embedding once via ``bench-build-corpus`` and
persisting the result to disk; benchmarks then load the corpus and only
make OpenSearch calls during measurement.

Public API:
    SUPPORTED_DIMS, MAX_DIM        - the embedding dimensions the corpus supports
    SOURCES, SOURCE_NAMES, MIXED   - HF dataset adapters
    sample_corpus                   - draw deterministic samples from one or more sources
    build_corpus                    - run the full sample -> embed -> persist pipeline
    load_corpus                     - read a built corpus and slice/renormalize to target dim
    CorpusBundle                    - dataclass returned by load_corpus
"""

from .builder import build_corpus
from .groundtruth import build_groundtruth
from .io import CorpusBundle, load_corpus, manifest_path
from .sources import (
    MAX_DIM,
    MIXED,
    SOURCE_NAMES,
    SOURCES,
    SUPPORTED_DIMS,
    sample_corpus,
)

__all__ = [
    "MAX_DIM",
    "SUPPORTED_DIMS",
    "SOURCES",
    "SOURCE_NAMES",
    "MIXED",
    "sample_corpus",
    "build_corpus",
    "build_groundtruth",
    "load_corpus",
    "manifest_path",
    "CorpusBundle",
]
