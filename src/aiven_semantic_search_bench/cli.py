"""
CLI for the OpenSearch k-NN benchmarking tool.

Subcommands
-----------
  bench-build-corpus        One-time: sample real text + embed once at 3072 dims.
  bench-build-groundtruth   One-time: brute-force top-100 nearest neighbours.
  bench-index               Indexing throughput at varying batch sizes.
  bench-search              k-NN query latency over multiple rounds.
  bench-recall              recall@K accuracy against brute-force ground truth.
  bench-hybrid              BM25 + k-NN hybrid queries with optional filter.
  bench-recover             Cold-start cost after auto-pause (free/hobbyist tier).
  bench-plan-change         Upgrade / downgrade impact while service is live.

Every measurement command accepts --engine, --method, --mode, --compression,
--data-type, --m, --ef-construction, --ef-search, and --with-metadata so the
same workload can be swept across the k-NN configuration matrix from the
command line (the UI generates equivalent jobs automatically).
"""

from __future__ import annotations

import argparse
import os
import sys

from .bench_index import cmd_bench_index
from .bench_plan_change import cmd_bench_plan_change
from .bench_recall import cmd_bench_recall
from .bench_recover import cmd_bench_recover
from .bench_search import cmd_bench_search
from .bench_hybrid import cmd_bench_hybrid
from .config import Settings
from .corpus import SOURCE_NAMES, SUPPORTED_DIMS, build_corpus, build_groundtruth
from .opensearch_client import KnnSpec

_PRESET_CHOICES = ("mixed",) + SOURCE_NAMES


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        n = int(piece)
        if n <= 0:
            raise argparse.ArgumentTypeError(f"values must be > 0 (got {n})")
        out.append(n)
    if not out:
        raise argparse.ArgumentTypeError("list cannot be empty")
    return out


def _add_corpus_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--embed-dim",
        type=int,
        choices=list(SUPPORTED_DIMS),
        default=None,
        help=(
            f"Embedding dimension at load time (Matryoshka-truncated from 3072). "
            f"Choices: {list(SUPPORTED_DIMS)}. Defaults to EMBED_DIM env var."
        ),
    )
    sp.add_argument(
        "--corpus-dir",
        default="corpus",
        help="Directory holding the pre-built corpus (default: corpus/)",
    )


def _add_knn_spec_args(sp: argparse.ArgumentParser) -> None:
    """Shared k-NN configuration flags used by every measurement command."""
    grp = sp.add_argument_group("k-NN configuration")
    grp.add_argument(
        "--engine",
        choices=["faiss", "lucene"],
        default="faiss",
        help="k-NN engine (default: faiss)",
    )
    grp.add_argument(
        "--method",
        choices=["hnsw", "ivf"],
        default="hnsw",
        help="k-NN method (default: hnsw; ivf requires faiss)",
    )
    grp.add_argument(
        "--space-type",
        choices=["cosinesimil", "innerproduct", "l2"],
        default="cosinesimil",
        help="Similarity space (default: cosinesimil)",
    )
    grp.add_argument(
        "--mode",
        choices=["in_memory", "on_disk"],
        default="in_memory",
        help="Index mode (default: in_memory; on_disk requires faiss, 2.17+)",
    )
    grp.add_argument(
        "--compression",
        choices=["none", "1x", "2x", "4x", "8x", "16x", "32x"],
        default="none",
        help="Binary quantization compression level (default: none; requires on_disk)",
    )
    grp.add_argument(
        "--data-type",
        choices=["float", "byte", "fp16", "binary"],
        default="float",
        help="Vector data type (default: float; byte/fp16 require faiss 2.17+/3.3+)",
    )
    grp.add_argument(
        "--derived-source",
        action="store_true",
        default=False,
        help="Enable derived_source (storage savings; 2.19+ experimental, 3.x recommended)",
    )
    grp.add_argument("--m", type=int, default=16, help="HNSW m parameter (default: 16)")
    grp.add_argument(
        "--ef-construction",
        type=int,
        default=128,
        help="HNSW ef_construction (default: 128)",
    )
    grp.add_argument(
        "--ef-search",
        type=int,
        default=256,
        help="HNSW ef_search (default: 256)",
    )
    grp.add_argument(
        "--with-text",
        action="store_true",
        default=False,
        help="Add a 'content' text field to the index mapping (required for hybrid)",
    )
    grp.add_argument(
        "--with-metadata",
        action="store_true",
        default=False,
        help="Add a 'metadata' object field (category, tenant_id, created_at) for filter tests",
    )
    grp.add_argument(
        "--opensearch-uri",
        default=None,
        help="Override OPENSEARCH_URI from .env (for headless multi-service runs)",
    )


def _knn_spec_from_args(args: argparse.Namespace, embed_dim: int) -> KnnSpec:
    return KnnSpec(
        embed_dim=embed_dim,
        engine=args.engine,
        method=args.method,
        space_type=args.space_type,
        mode=args.mode,
        compression=args.compression,
        data_type=args.data_type,
        derived_source=args.derived_source,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        with_text=args.with_text or getattr(args, "hybrid", False),
        with_metadata=args.with_metadata or getattr(args, "with_filter", False),
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aiven-semantic-search-bench",
        description="Benchmark Aiven for OpenSearch k-NN across versions and configurations.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── bench-build-corpus ───────────────────────────────────────────────────
    sp = sub.add_parser(
        "bench-build-corpus",
        help="One-time: sample real text from HF datasets + embed once at 3072 dims",
    )
    sp.add_argument("--dataset", choices=_PRESET_CHOICES, default="mixed")
    sp.add_argument("--doc-count", type=int, default=100_000)
    sp.add_argument("--query-count", type=int, default=100_000)
    sp.add_argument("--embed-batch-size", type=int, default=100)
    sp.add_argument("--seed", type=int, default=42)
    sp.add_argument("--out-dir", default="corpus")
    sp.add_argument("--dry-run", action="store_true")
    sp.add_argument(
        "--with-metadata",
        action="store_true",
        default=False,
        help="Add synthetic category/tenant_id/created_at columns to docs.parquet",
    )

    # ── bench-build-groundtruth ──────────────────────────────────────────────
    sp = sub.add_parser(
        "bench-build-groundtruth",
        help="One-time: compute brute-force top-K nearest neighbours (writes corpus/qrels.npy)",
    )
    sp.add_argument(
        "--corpus-dir",
        default="corpus",
        help="Corpus directory produced by bench-build-corpus (default: corpus/)",
    )
    sp.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of ground-truth nearest neighbours per query (default: 100)",
    )
    sp.add_argument(
        "--chunk-q",
        type=int,
        default=500,
        help="Query chunk size for memory-bounded computation (default: 500)",
    )

    # ── bench-index ──────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "bench-index",
        help="Indexing throughput at varying _bulk batch sizes",
    )
    sp.add_argument("--doc-count", type=int, default=1000)
    sp.add_argument(
        "--batch-sizes",
        type=_parse_int_list,
        default=[1, 5, 10, 20, 50],
        help="Comma-separated batch sizes (default: 1,5,10,20,50)",
    )
    sp.add_argument("--label", default="unlabeled")
    sp.add_argument("--out-dir", default="results")
    _add_corpus_args(sp)
    _add_knn_spec_args(sp)

    # ── bench-search ─────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "bench-search",
        help="k-NN query latency over multiple rounds",
    )
    sp.add_argument("--rounds", type=int, default=3)
    sp.add_argument("--query-count", type=int, default=100)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--label", default="unlabeled")
    sp.add_argument("--out-dir", default="results")
    _add_corpus_args(sp)
    _add_knn_spec_args(sp)

    # ── bench-recall ─────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "bench-recall",
        help="recall@K accuracy vs ground truth alongside latency",
    )
    sp.add_argument("--query-count", type=int, default=500)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument("--label", default="unlabeled")
    sp.add_argument("--out-dir", default="results")
    _add_corpus_args(sp)
    _add_knn_spec_args(sp)

    # ── bench-hybrid ─────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "bench-hybrid",
        help="BM25 + k-NN hybrid query with optional metadata filter",
    )
    sp.add_argument("--query-count", type=int, default=200)
    sp.add_argument("--k", type=int, default=10)
    sp.add_argument(
        "--filter-selectivity",
        choices=["none", "low", "high"],
        default="none",
        help="Metadata filter selectivity: none / low (~25%% match) / high (~6%% match)",
    )
    sp.add_argument("--label", default="unlabeled")
    sp.add_argument("--out-dir", default="results")
    _add_corpus_args(sp)
    _add_knn_spec_args(sp)

    # ── bench-recover ────────────────────────────────────────────────────────
    sp = sub.add_parser(
        "bench-recover",
        help="First-query latency after auto-pause idle window",
    )
    sp.add_argument("--idle-minutes", type=int, default=6)
    sp.add_argument("--doc-count", type=int, default=200)
    sp.add_argument("--label", default="unlabeled")
    sp.add_argument("--out-dir", default="results")
    _add_corpus_args(sp)

    # ── bench-plan-change ────────────────────────────────────────────────────
    sp = sub.add_parser(
        "bench-plan-change",
        help="Query latency and errors during a live Aiven plan change",
    )
    sp.add_argument("--from-plan", required=True)
    sp.add_argument("--to-plan", required=True)
    sp.add_argument("--and-back", action="store_true")
    sp.add_argument("--pre-load-seconds", type=int, default=30)
    sp.add_argument("--post-settle-seconds", type=int, default=30)
    sp.add_argument("--doc-count", type=int, default=200)
    sp.add_argument("--label", default="unlabeled")
    sp.add_argument("--out-dir", default="results")
    _add_corpus_args(sp)

    return p


def _resolve_embed_dim(arg_value: int | None, settings: Settings) -> int:
    if arg_value is not None:
        return int(arg_value)
    if settings.embed_dim not in SUPPORTED_DIMS:
        raise SystemExit(
            f"EMBED_DIM={settings.embed_dim} is not in {list(SUPPORTED_DIMS)}. "
            f"Pass --embed-dim explicitly."
        )
    return int(settings.embed_dim)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings.from_env()

    if args.command == "bench-build-corpus":
        return build_corpus(
            settings=settings,
            out_dir=str(args.out_dir),
            preset=str(args.dataset),
            target_docs=int(args.doc_count),
            target_queries=int(args.query_count),
            embed_batch_size=int(args.embed_batch_size),
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            with_metadata=bool(args.with_metadata),
        )

    if args.command == "bench-build-groundtruth":
        return build_groundtruth(
            corpus_dir=str(args.corpus_dir),
            k=int(args.k),
            chunk_q=int(args.chunk_q),
        )

    if args.command == "bench-index":
        embed_dim = _resolve_embed_dim(args.embed_dim, settings)
        spec = _knn_spec_from_args(args, embed_dim)
        return cmd_bench_index(
            settings,
            doc_count=int(args.doc_count),
            batch_sizes=list(args.batch_sizes),
            embed_dim=embed_dim,
            spec=spec,
            corpus_dir=str(args.corpus_dir),
            label=str(args.label),
            out_dir=str(args.out_dir),
            opensearch_uri=args.opensearch_uri,
        )

    if args.command == "bench-search":
        embed_dim = _resolve_embed_dim(args.embed_dim, settings)
        spec = _knn_spec_from_args(args, embed_dim)
        return cmd_bench_search(
            settings,
            rounds=int(args.rounds),
            query_count=int(args.query_count),
            k=int(args.k),
            embed_dim=embed_dim,
            spec=spec,
            corpus_dir=str(args.corpus_dir),
            label=str(args.label),
            out_dir=str(args.out_dir),
            opensearch_uri=args.opensearch_uri,
        )

    if args.command == "bench-recall":
        embed_dim = _resolve_embed_dim(args.embed_dim, settings)
        spec = _knn_spec_from_args(args, embed_dim)
        return cmd_bench_recall(
            settings,
            query_count=int(args.query_count),
            k=int(args.k),
            embed_dim=embed_dim,
            spec=spec,
            corpus_dir=str(args.corpus_dir),
            label=str(args.label),
            out_dir=str(args.out_dir),
            opensearch_uri=args.opensearch_uri,
        )

    if args.command == "bench-hybrid":
        embed_dim = _resolve_embed_dim(args.embed_dim, settings)
        spec = _knn_spec_from_args(args, embed_dim)
        # Hybrid always needs text and metadata fields.
        spec = KnnSpec(
            embed_dim=spec.embed_dim,
            engine=spec.engine,
            method=spec.method,
            space_type=spec.space_type,
            mode=spec.mode,
            compression=spec.compression,
            data_type=spec.data_type,
            derived_source=spec.derived_source,
            m=spec.m,
            ef_construction=spec.ef_construction,
            ef_search=spec.ef_search,
            with_text=True,
            with_metadata=True,
        )
        return cmd_bench_hybrid(
            settings,
            query_count=int(args.query_count),
            k=int(args.k),
            embed_dim=embed_dim,
            spec=spec,
            filter_selectivity=str(args.filter_selectivity),
            corpus_dir=str(args.corpus_dir),
            label=str(args.label),
            out_dir=str(args.out_dir),
            opensearch_uri=args.opensearch_uri,
        )

    if args.command == "bench-recover":
        embed_dim = _resolve_embed_dim(args.embed_dim, settings)
        return cmd_bench_recover(
            settings,
            idle_minutes=int(args.idle_minutes),
            doc_count=int(args.doc_count),
            embed_dim=embed_dim,
            corpus_dir=str(args.corpus_dir),
            label=str(args.label),
            out_dir=str(args.out_dir),
        )

    if args.command == "bench-plan-change":
        embed_dim = _resolve_embed_dim(args.embed_dim, settings)
        return cmd_bench_plan_change(
            settings,
            from_plan=str(args.from_plan),
            to_plan=str(args.to_plan),
            and_back=bool(args.and_back),
            pre_load_seconds=int(args.pre_load_seconds),
            post_settle_seconds=int(args.post_settle_seconds),
            doc_count=int(args.doc_count),
            embed_dim=embed_dim,
            corpus_dir=str(args.corpus_dir),
            label=str(args.label),
            out_dir=str(args.out_dir),
        )

    raise RuntimeError(f"Unhandled command: {args.command}")


def _main_console_script(command: str) -> int:
    """Entry point for setuptools `[project.scripts]` wrappers.

    Installed scripts invoke us with ``sys.argv == [script_path, <user args>]``;
    :func:`argparse` expects the subcommand name as ``argv[1]``.
    """
    sys.argv.insert(1, command)
    return main()


def main_bench_build_corpus() -> int:
    return _main_console_script("bench-build-corpus")


def main_bench_build_groundtruth() -> int:
    return _main_console_script("bench-build-groundtruth")


def main_bench_index() -> int:
    return _main_console_script("bench-index")


def main_bench_search() -> int:
    return _main_console_script("bench-search")


def main_bench_recall() -> int:
    return _main_console_script("bench-recall")


def main_bench_hybrid() -> int:
    return _main_console_script("bench-hybrid")


def main_bench_recover() -> int:
    return _main_console_script("bench-recover")


def main_bench_plan_change() -> int:
    return _main_console_script("bench-plan-change")
