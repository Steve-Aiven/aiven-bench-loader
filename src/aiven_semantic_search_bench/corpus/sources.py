"""
Hugging Face Hub adapters for the benchmark corpus.

We use the BeIR re-host of public retrieval datasets because:
  - It exposes a uniform schema (corpus + queries + qrels) across all four
    datasets we sample from, so the loader can be one piece of code.
  - The corpora are real text drawn from real systems (Bing search results,
    Quora questions, financial Q&A on StackExchange, scientific abstracts),
    which is far more representative than a synthetic generator.
  - No authentication is required to download.

We intentionally do NOT use qrels (relevance judgments). This benchmarking
suite measures latency and throughput, not retrieval quality, so we just
need diverse text to embed and search with.

The "mixed" preset draws ~25% from each of four sources, with deficits
rolled over to MS MARCO (the only source with >100k queries), so the
target counts are always achievable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# 768 is the output dimensionality of nomic-ai/nomic-embed-text-v1.5
# (the default local embedding model).  We embed every text once at MAX_DIM
# and rely on the Matryoshka property to derive 256 / 512 / 768 -dim vectors
# at benchmark time by truncating + L2-renormalising the prefix.
# One embed run serves every supported dimension with no extra API calls.
#
# If you switch to a 1024-dim model (e.g. mixedbread-ai/mxbai-embed-large-v1),
# rebuild the corpus after setting HF_EMBED_MAX_DIM=1024 in .env; the
# SUPPORTED_DIMS tuple is read from the corpus manifest at load time so older
# corpora continue to work at their original dimensions.
MAX_DIM = 768
SUPPORTED_DIMS: tuple[int, ...] = (256, 512, 768)


@dataclass(frozen=True)
class CorpusSource:
    """How to find a particular HF dataset's corpus and queries."""

    name: str
    hf_path: str
    docs_config: str
    docs_split: str
    docs_text_fields: tuple[str, ...]
    queries_config: str
    queries_split: str
    queries_text_fields: tuple[str, ...]


# All four sources share the BeIR schema: `_id`, `title`, `text` for the
# corpus side, `_id`, `text` for the query side. Title is empty on some
# datasets (e.g. queries) and that is fine - we strip and skip empty rows.
SOURCES: dict[str, CorpusSource] = {
    "msmarco": CorpusSource(
        name="msmarco",
        hf_path="BeIR/msmarco",
        docs_config="corpus", docs_split="corpus",
        docs_text_fields=("title", "text"),
        queries_config="queries", queries_split="queries",
        queries_text_fields=("text",),
    ),
    "quora": CorpusSource(
        name="quora",
        hf_path="BeIR/quora",
        docs_config="corpus", docs_split="corpus",
        docs_text_fields=("title", "text"),
        queries_config="queries", queries_split="queries",
        queries_text_fields=("text",),
    ),
    "fiqa": CorpusSource(
        name="fiqa",
        hf_path="BeIR/fiqa",
        docs_config="corpus", docs_split="corpus",
        docs_text_fields=("title", "text"),
        queries_config="queries", queries_split="queries",
        queries_text_fields=("text",),
    ),
    "scifact": CorpusSource(
        name="scifact",
        hf_path="BeIR/scifact",
        docs_config="corpus", docs_split="corpus",
        docs_text_fields=("title", "text"),
        queries_config="queries", queries_split="queries",
        queries_text_fields=("text",),
    ),
}

SOURCE_NAMES = tuple(SOURCES.keys())

# Order matters: msmarco comes LAST so that any deficit from the smaller
# datasets (scifact has only ~5k docs and ~1k queries) is absorbed by
# msmarco's effectively unlimited supply. This keeps the total count
# matching the user's --doc-count / --query-count target.
MIXED: tuple[str, ...] = ("quora", "fiqa", "scifact", "msmarco")


# ──────────────────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────────────────

# Row type: (id_str, text_str, source_name_str)
SampledRow = tuple[str, str, str]


def _row_text(row: dict, fields: tuple[str, ...]) -> str:
    """Concatenate the named fields with a space, ignoring None or empty values."""
    parts: list[str] = []
    for f in fields:
        v = row.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return " ".join(parts)


def _sample_one(
    source: CorpusSource,
    *,
    kind: str,
    n: int,
    seed: int,
) -> list[SampledRow]:
    """
    Deterministically sample ``n`` rows from one HF dataset using streaming
    mode so only the rows we need are downloaded.

    Strategy: shuffle the iterable dataset with a fixed seed and a shuffle
    buffer large enough to randomise the order, then take the first ``n``
    non-empty rows.  This avoids downloading the full corpus (MS MARCO is
    8+ GB) and is typically 10–100× faster for small sample sizes.

    The shuffle buffer is capped at 50 000 rows; we download at most
    ``n + buffer_size`` rows' worth of data regardless of corpus size.
    """
    if kind not in ("docs", "queries"):
        raise ValueError(f"kind must be 'docs' or 'queries', got {kind!r}")
    if n <= 0:
        return []

    from datasets import load_dataset

    config = source.docs_config if kind == "docs" else source.queries_config
    split = source.docs_split if kind == "docs" else source.queries_split
    fields = source.docs_text_fields if kind == "docs" else source.queries_text_fields

    # buffer_size controls how much we shuffle. Larger = more random but
    # requires downloading more data before the first row is emitted.
    # 10 000 is a good balance: randomises well for typical benchmarks and
    # only downloads ~10 k rows before we start receiving output.
    buffer_size = min(10_000, n * 3)

    ds = load_dataset(source.hf_path, config, split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=buffer_size)

    out: list[SampledRow] = []
    for i, row in enumerate(ds):
        if len(out) >= n:
            break
        text = _row_text(row, fields)
        if not text:
            continue
        row_id = str(row.get("_id") or row.get("id") or i)
        out.append((row_id, text, source.name))

    return out


def sample_corpus(
    *,
    preset: str,
    target_docs: int,
    target_queries: int,
    seed: int = 42,
) -> tuple[list[SampledRow], list[SampledRow]]:
    """
    Sample documents and queries according to a named preset.

    Presets:
        - 'msmarco', 'quora', 'fiqa', 'scifact'  - single-source
        - 'mixed'  - 25% from each of the four sources, with any deficit
                     (sources have varying capacity; scifact maxes out at
                     ~5k docs / 1k queries) rolled over to MS MARCO

    Returns (docs, queries) where each is a list of (id, text, source).
    """
    if preset != "mixed" and preset not in SOURCES:
        raise ValueError(
            f"Unknown preset {preset!r}; expected 'mixed' or one of {list(SOURCES)}"
        )

    if preset != "mixed":
        src = SOURCES[preset]
        docs = _sample_one(src, kind="docs", n=target_docs, seed=seed)
        # Use a different seed offset for queries so we don't accidentally
        # sample identical row indices on datasets where docs and queries
        # share a length.
        queries = _sample_one(src, kind="queries", n=target_queries, seed=seed + 1)
        return docs, queries

    return _sample_mixed(target_docs, target_queries, seed)


def _sample_mixed(
    target_docs: int,
    target_queries: int,
    seed: int,
) -> tuple[list[SampledRow], list[SampledRow]]:
    """
    Sample evenly across the MIXED sources, rolling over deficits to msmarco.

    The non-msmarco sources are sampled first so we know how much they
    actually returned (some are smaller than target/4); msmarco then takes
    everything else needed to hit the target totals.
    """
    n_src = len(MIXED)
    per_docs = target_docs // n_src
    per_queries = target_queries // n_src

    docs: list[SampledRow] = []
    queries: list[SampledRow] = []
    docs_deficit = 0
    queries_deficit = 0

    # Smaller datasets first: their per-source budget may be capped by their
    # own total row count; we accumulate any deficit into the msmarco pull.
    smaller = [s for s in MIXED if s != "msmarco"]
    for i, name in enumerate(smaller):
        src = SOURCES[name]
        d = _sample_one(src, kind="docs", n=per_docs, seed=seed + i * 17)
        q = _sample_one(src, kind="queries", n=per_queries, seed=seed + i * 17 + 1)
        docs.extend(d)
        queries.extend(q)
        docs_deficit += per_docs - len(d)
        queries_deficit += per_queries - len(q)

    # msmarco soaks up rounding remainder + deficits.
    msmarco_docs_n = per_docs + docs_deficit + (target_docs - per_docs * n_src)
    msmarco_queries_n = per_queries + queries_deficit + (target_queries - per_queries * n_src)
    msmarco = SOURCES["msmarco"]
    docs.extend(_sample_one(msmarco, kind="docs", n=msmarco_docs_n, seed=seed + 1000))
    queries.extend(_sample_one(msmarco, kind="queries", n=msmarco_queries_n, seed=seed + 1001))

    return docs, queries
