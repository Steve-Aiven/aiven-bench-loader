# Corpus scripts

Scripts for building the benchmark corpora on the Mac host (never inside
Docker — the Docker image does not include the embedding stack).

All scripts must be run from the repository root:

```bash
cd /path/to/aiven-semantic-search-bench
bash scripts/build-corpus-smoke.sh
```

---

## Which corpus do I need?

This is the most important choice in the benchmark setup. The corpus size
determines whether the results are **meaningful** or just measuring HTTP overhead.

| Corpus | Docs | Queries | HNSW meaningful? | ef_search curve? | byte vs float Δrecall? | Colima RAM | Build time (MPS) |
|--------|------|---------|-----------------|-----------------|----------------------|-----------|-----------------|
| corpus-2k | 2k | 200 | No — exhaustive | No | No | 4 GiB | ~1 min |
| **corpus-smoke** | **5k** | **500** | **No** | **No** | **No** | **4 GiB** | **~4 min** |
| corpus-20k | 20k | 2k | Slightly | Slightly | ~0.5% | 4 GiB | ~12 min |
| **corpus-100k** | **100k** | **10k** | **Yes** | **Yes** | **~3–5%** | **8 GiB** | **~60 min** |

**The short version:**

- **corpus-smoke** is only suitable for pipeline validation — confirming the
  benchmark tooling works end-to-end. With 5k docs, HNSW returns near-perfect
  recall for every configuration because the graph is small enough that
  traversal visits most nodes. You cannot compare configurations meaningfully.

- **corpus-100k** is the minimum for production-grade benchmarking. At this
  scale, `ef_search`, quantization type, and index engine produce clearly
  different recall@K scores, and latency differences between `in_memory` and
  `on_disk` become measurable.

- **corpus-20k** is a useful intermediate step: fits in the current 4 GiB
  Colima VM and starts to show some differentiation between configurations.
  Recommended if you want to validate the matrix before investing in 100k.

### Why does corpus size matter so much?

At small corpus sizes, the HNSW approximate search degenerates into something
close to exhaustive search — the graph covers nearly all nodes at `ef_search=256`,
so every configuration returns the same results. The key indicators you cannot
measure at 5k docs:

1. **`ef_search` tradeoff** — the recall/latency curve that justifies tuning this parameter
2. **Quantization recall cost** — byte int8 vs float32 shows ~0–5% Δrecall only at scale
3. **`on_disk` latency** — memory-mapped I/O penalty only shows when the graph doesn't fit in CPU cache
4. **IVF training quality** — requires at least 4× `nlist` training points; at 100k, `nlist≈316` works well
5. **BM25 hybrid realism** — tiny indexes have trivially short posting lists

### Colima resize for 100k

```bash
colima stop
colima start --cpu 4 --memory 8 --disk 100
# Then bump OpenSearch JVM heap to 2g:
# Edit docker-compose.opensearch.yml → OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g
docker compose -f docker-compose.opensearch.yml down
docker compose -f docker-compose.opensearch.yml up -d
```

---

## Prerequisites

Install the `[build]` optional dependencies once:

```bash
pip install -e '.[build]'
```

This adds `sentence-transformers`, `datasets`, and `einops` — the libraries
needed to download the HuggingFace datasets and run the embedding model.
These are **not** present in the lean Docker image.

---

## Available scripts

| Script | Corpus dir | Docs | Queries | Purpose |
|--------|-----------|------|---------|---------|
| `build-corpus-2k.sh` | `corpus-2k-nomic/` | 2 000 | 200 | Sanity check / very quick iteration |
| `build-corpus-smoke.sh` | `corpus-smoke/` | 5 000 | 500 | Pipeline validation only |
| `build-corpus-20k.sh` | `corpus-20k/` | 20 000 | 2 000 | Development benchmark (some differentiation) |
| `build-corpus-100k.sh` | `corpus-100k/` | 100 000 | 10 000 | **Standard benchmark** (meaningful results) |
| `validate-corpus.sh` | any | — | — | Inspect and verify an existing corpus |

---

## Running bench-plan.sh with a different corpus

```bash
# Development (current 4 GiB Colima)
CORPUS_DIR=./corpus-20k DOC_COUNT=20000 QUERY_COUNT=2000 bash bench-plan.sh

# Standard benchmark (requires 8 GiB Colima + 2g JVM heap)
CORPUS_DIR=./corpus-100k DOC_COUNT=100000 QUERY_COUNT=10000 bash bench-plan.sh

# 100k samples per search cell (--rounds 200 in bench-plan.sh, or run manually)
CORPUS_DIR=./corpus-100k DOC_COUNT=100000 QUERY_COUNT=500 bash bench-plan.sh
# With 500 queries: bench-search --rounds 200 gives 100k latency samples
```

---

## Environment variables

All scripts respect the following variables (default values shown):

```bash
HF_EMBED_DEVICE=mps           # Embedding device: mps (Apple GPU), cuda, or cpu
HF_EMBED_MODEL=nomic-ai/nomic-embed-text-v1.5   # Embedding model
HF_EMBED_MAX_DIM=768          # Maximum output dimension to store
EMBED_BATCH_SIZE=64           # Documents per embedding batch
SEED=42                       # Sampling seed (deterministic)
BENCH_HF_DATASETS_STREAMING=1 # 1=stream from Hub (low disk); 0=cache locally
```

To override, set them before calling the script:

```bash
HF_EMBED_DEVICE=cpu EMBED_BATCH_SIZE=32 bash scripts/build-corpus-smoke.sh
```

To disable streaming and cache the HuggingFace datasets locally after the
first download (faster subsequent builds):

```bash
BENCH_HF_DATASETS_STREAMING=0 bash scripts/build-corpus-smoke.sh
```

---

## How corpus building works

1. **Sample text** — `bench-build-corpus` downloads text rows from the
   BeIR-hosted HuggingFace datasets (`msmarco`, `quora`, `fiqa`, `scifact`)
   and writes `docs.parquet` and `queries.parquet`.

2. **Embed** — Each text is embedded once using the chosen model and device.
   The full-dim vectors are written to `docs_embeddings.npy` and
   `queries_embeddings.npy`. Checkpoint saves happen every 5 000 rows, so an
   interrupted build can be resumed without re-embedding from the start.

3. **Groundtruth** — `bench-build-groundtruth` computes the exact top-100
   nearest neighbours for every query against every document using brute-force
   cosine similarity (chunked NumPy, no GPU required). For 10k queries × 100k
   docs this takes ~8–15 min on CPU.  Result is written to `qrels.npy`.

The same float32 corpus serves all data types in the benchmark matrix:

| Benchmark cell | Encoding applied |
|---------------|-----------------|
| L01–L07 (float/768-dim) | Vectors sent as-is |
| L02 (512-dim) | Prefix slice + L2 renorm at load time |
| L03 (256-dim) | Prefix slice + L2 renorm at load time |
| L08 (byte) | Scale `× 127` → int8 at index/query time |
| L09 (binary, planned) | Sign binarise + `np.packbits` at send time |

No corpus rebuild is required when switching data type or dimension.

---

## Model notes

The default model `nomic-ai/nomic-embed-text-v1.5` is:
- Apache 2.0 licensed, free to use commercially
- 768-dimensional with Matryoshka Representation Learning (MRL)
- ~270 MB weight download, cached in `~/.cache/huggingface` after first run
- Produces L2-normalised vectors (normalisation applied at load time)

CPU throughput is roughly 300–600 docs/sec. On Apple Silicon with MPS, expect
1 000–3 000 docs/sec depending on model batch size and memory bandwidth.

| Corpus | Embedding time (MPS) | Groundtruth time (CPU) |
|--------|---------------------|----------------------|
| 5k docs | ~30 s | ~5 s |
| 20k docs | ~2–5 min | ~30 s |
| 100k docs | ~30–45 min | ~8–15 min |
