# Corpus scripts

Scripts for building the benchmark corpora on the Mac host (never inside
Docker — the Docker image does not include the embedding stack).

All scripts must be run from the repository root:

```bash
cd /path/to/aiven-semantic-search-bench
bash scripts/build-corpus-smoke.sh
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

| Script | Corpus dir | Docs | Queries | Use case |
|--------|-----------|------|---------|----------|
| `build-corpus-smoke.sh` | `corpus-smoke/` | 5 000 | 500 | Default benchmark matrix (`bench-plan.sh`) |
| `build-corpus-2k.sh` | `corpus-2k-nomic/` | 2 000 | 200 | Quick sanity checks |
| `build-corpus-20k.sh` | `corpus-20k/` | 20 000 | 2 000 | Scale / production-size tests |
| `validate-corpus.sh` | any | — | — | Inspect and verify an existing corpus |

---

## Environment variables

All scripts respect the following variables (default values shown):

```bash
HF_EMBED_DEVICE=mps          # Embedding device: mps (Apple GPU), cuda, or cpu
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
   cosine similarity (chunked NumPy, no GPU required). Result is written to
   `qrels.npy` and used by `bench-recall` to measure recall@K accuracy.

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
- Produces L2-normalised vectors (or raw — normalisation is applied at load time)

CPU throughput is roughly 300–600 docs/sec. On Apple Silicon with MPS, expect
1 000–3 000 docs/sec depending on model batch size and memory bandwidth.

For the 5 000-doc smoke corpus, build time is typically:
- ~30 s with MPS on M-series Mac
- ~60–120 s on CPU

For a 20 000-doc corpus:
- ~2–5 min with MPS
- ~5–15 min on CPU
