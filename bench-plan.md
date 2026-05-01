# Local Benchmark Test Plan

OpenSearch k-NN benchmark matrix running against a local single-node OpenSearch 2.19
via Colima. The bench container is the lean runner image (no NiceGUI, no embedding stack).
Corpus is pre-built on the Mac with Apple GPU (MPS) and mounted read-only.

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Colima | ≥0.6 | Docker runtime on macOS |
| Docker CLI | ≥24 | Build + run containers |
| `curl` | any | OpenSearch health check |
| Python 3.10+ | any | Native corpus builds only (not needed to run benchmarks) |
| `pip install -e '.[build]'` | — | Required for `bench-build-corpus` / `bench-build-groundtruth` (installs `sentence-transformers`, `datasets`). Skip if using the pre-built corpora. |

Install Colima and Docker CLI if needed:
```bash
brew install colima docker
```

---

## Phase 0 — Start infrastructure

### 0.1 Start Colima

The OpenSearch JVM heap is set to 1 GiB (`-Xms1g -Xmx1g` in
`docker-compose.opensearch.yml`). Colima needs **at least 4 GiB total** to
leave room for the OS, the bench containers, and the OpenSearch off-heap
(direct memory, OS page cache). 2 CPUs is sufficient; 4 is comfortable.

```bash
colima start --cpu 2 --memory 4 --disk 60
```

If Colima is already running, check its allocation and resize if needed:
```bash
colima list
# If MEMORY is less than 4GiB:
colima stop && colima start --cpu 2 --memory 4 --disk 60
```

### 0.2 Start local OpenSearch

```bash
docker compose -f docker-compose.opensearch.yml up -d
```

Wait for green (up to 60 s on first pull):
```bash
until curl -sf http://localhost:9200/_cluster/health | grep -v '"status":"red"'; do
  echo "waiting for OpenSearch..."; sleep 5
done
echo "OpenSearch is ready"
```

### 0.3 Build the lean bench image

```bash
docker build -t aiven-semantic-search-bench:lean .
```

Expected build time: ~2 minutes (no PyTorch wheel).

---

## Phase 1 — Corpus

Both corpora are pre-built and fully populated:

| Directory | Docs | Queries | Dims | Ground truth | Metadata |
|-----------|------|---------|------|--------------|----------|
| `corpus-smoke/` | 5 000 | 500 | 256, 512, 768 | `qrels.npy` present | **required** — rebuild if `has_metadata: false` |
| `corpus-2k-nomic/` | 2 000 | 200 | 256, 512, 768 | `qrels.npy` present | optional (L07 not in 2k plan) |

The default benchmark script uses `corpus-smoke` (5 000 docs). That is sufficient to
observe meaningful differences across engine, mode, and quantization choices.

**L07 requires `has_metadata: true` in `corpus-smoke/manifest.json`.** Check with:
```bash
python3 -c "import json; m=json.load(open('corpus-smoke/manifest.json')); print('has_metadata:', m.get('has_metadata'))"
```
If `False`, rebuild:
```bash
bash scripts/build-corpus-smoke.sh
```

### Optional: build a larger corpus on Mac (MPS)

Run this on the host, never inside Docker (Docker on Mac has no GPU access):

```bash
# From the main branch (bench-build-corpus is not in lean-runner):
git stash   # or switch to main branch in another terminal
pip install -e '.[build]'

HF_EMBED_DEVICE=mps python3 -m aiven_semantic_search_bench bench-build-corpus \
    --dataset mixed --doc-count 20000 --query-count 2000 \
    --corpus-dir ./corpus-20k-nomic

python3 -m aiven_semantic_search_bench bench-build-groundtruth \
    --corpus-dir ./corpus-20k-nomic
```

Then re-run the benchmark script with `CORPUS_DIR=./corpus-20k-nomic`.

---

## Phase 2 — Benchmark matrix

The matrix covers the main axes from the README recommended matrix, constrained to
configurations supported by OpenSearch 2.19. Each cell runs three passes in sequence:

1. **bench-index** — bulk-index `DOC_COUNT` documents; reports docs/sec at each batch size.
2. **bench-search** — 3 rounds × `QUERY_COUNT` k-NN queries; reports p50/p90/p99 latency.
3. **bench-recall** — `QUERY_COUNT` queries compared against brute-force ground truth;
   reports recall@1/5/10/50/100 alongside latency.

Cell L07 (lucene) also runs **bench-hybrid** (BM25 + k-NN combined score).

| ID | Engine | Mode | Compression | Data type | Dim | Bench types | Notes |
|----|--------|------|-------------|-----------|-----|-------------|-------|
| L01 | faiss | in_memory | none | float | 768 | index, search, recall | baseline |
| L02 | faiss | in_memory | none | float | 512 | index, search, recall | dim reduction |
| L03 | faiss | in_memory | none | float | 256 | index, search, recall | dim reduction |
| L04 | faiss | on_disk | none | float | 768 | index, search, recall | disk offload |
| L05 | faiss | on_disk | 32x | float | 768 | index, search, recall | disk + compression |
| L06 | lucene | in_memory | none | float | 768 | index, search, recall | lucene baseline |
| L07 | lucene | in_memory | none | float | 768 | index, search, recall, hybrid×3 | lucene + hybrid (filter: none / low / high) |
| L08 | faiss | in_memory | none | byte | 768 | index, search, recall | int8 quantization |

> **Corpus:** All cells use the standard float32 nomic-embed corpus. L08 byte vectors are
> produced at bench time by scaling float32 `[-1, 1]` → int8 `[-127, 127]` — no separate
> corpus is needed.

### Planned cells (not yet runnable)

| ID | Config | Blocker |
|----|--------|---------|
| L09 | faiss/hnsw/in_memory/**binary**/768-bit | Needs `space_type=hamming` + bit-packing in `bench_index.py`; see below |
| L10 | faiss/**ivf**/in_memory/float/768 | Needs model training workflow; see below |

### What to look for

**L01 vs L02 vs L03 (dimension sweep)**
Latency and recall both decrease as `embed_dim` drops. The question is whether the recall
loss at 256 dims is acceptable. On this corpus recall@10 stays near 1.0 even at 256 dims,
but p50 latency drops from ~18 ms to ~2 ms — significant for high-QPS workloads.

**L01 vs L04 (in_memory vs on_disk, no compression)**
`on_disk` pages vectors from disk on access. On a fast local SSD the latency difference
may be small; on a loaded Aiven node with many tenants it will widen. This is the
baseline disk-offload comparison before adding compression.

**L04 vs L05 (on_disk without vs with 32x Faiss scalar compression)**
Adding 32× compression shrinks the on-disk footprint dramatically but reduces recall.
Check whether recall@10 stays above your target. If latency is similar to L04 but recall
drops, the compression ratio may need tuning.

**L01 vs L06 (faiss vs lucene, recall only)**
Faiss HNSW and Lucene HNSW share the same graph algorithm but different implementations.
Compare recall@K and p50/p99 latency to see which engine is better suited to your workload.

**L07 (lucene + hybrid with category/tenant filter)**
The hybrid pass blends BM25 lexical score with k-NN vector score using the same Lucene
index as L06. Three passes are run sequentially:

| Pass | `--filter-selectivity` | Filter | Matching docs |
|------|------------------------|--------|---------------|
| 1 | `none` | none — pure hybrid | 100% |
| 2 | `low` | `metadata.category = "infrastructure"` | ~25% |
| 3 | `high` | `metadata.category = "incident"` | ~6% |

This shows how pre-filter selectivity affects both latency and recall. A narrow filter
(`high`) forces OpenSearch to post-filter k-NN candidates, which can hurt recall if
too few candidates survive the filter. The index is built with `--with-text --with-metadata`
so both `content` (BM25) and `metadata.category` / `metadata.tenant_id` (keyword
filters) are present.

**Corpus requirement:** `corpus-smoke` must be built with `--with-metadata`
(`has_metadata: true` in `manifest.json`). Rebuild if needed:
```bash
bash scripts/build-corpus-smoke.sh
```

### L09 — binary data type

Binary vectors pack each dimension into a single bit (sign of the float). 768-dim float vectors become 96-byte vectors. Memory footprint is ~32× smaller than float32.

**What needs to change (no corpus rebuild required):**
1. `bench_index.py` — add `_binary_encode(vector)`: `np.packbits((vector > 0).astype(np.uint8)).tolist()`
2. `opensearch_client.py` — add `hamming` to allowed `space_type` values and route it correctly in `build_index_mapping`; binary uses `space_type=hamming`, not cosine/innerproduct
3. `bench_search.py` / `bench_recall.py` — encode query vectors the same way before sending

The corpus vectors are sign-binarized at send time — same float32 corpus, different encoding.

### L10 — Faiss IVF

IVF (Inverted File Index) clusters vectors into `nlist` partitions and probes only the nearest `nprobes` at query time. Faster than HNSW for high-throughput workloads, but requires a training step before the index can be created.

**Training workflow (two-step, not yet wired into `bench_index`):**

```
Step 1 — Build a training index with enough docs (≥ 39 × nlist):
  bench-index --doc-count <training_size> --label train --engine faiss ...
  (creates a plain float index to train from)

Step 2 — Train the IVF model:
  POST /_plugins/_knn/models/_train
  {
    "training_index": "bench",
    "training_field": "description_vector",
    "dimension": 768,
    "method": {
      "name": "ivf", "engine": "faiss",
      "space_type": "innerproduct",
      "parameters": { "nlist": 71, "nprobes": 8 }
    }
  }
  → returns { "model_id": "..." }

Step 3 — Poll until ready:
  GET /_plugins/_knn/models/{model_id}
  (wait for "state": "created")

Step 4 — Create the bench index using the trained model:
  field mapping uses "model_id": "{model_id}" instead of inline "method"
```

**nlist sizing:** use `sqrt(doc_count)` as a rule of thumb. For 5k docs: `nlist ≈ 71`. The current mapping builder uses `nlist=256` which requires >10k training points — it will be adjusted when IVF is implemented.

---

## Phase 3 — Interpreting results

Results are written to `./results/` as paired `.md` + `.json` files per run, named by
bench type and timestamp (e.g. `bench-recall-20260501-120000.md`).

The `.md` file is human-readable with a results table. The `.json` file is
machine-readable and suitable for ingestion by the orchestrator or ClickHouse.

To compare cells, sort the result files by label prefix `local/smoke/L0*`:
```bash
ls results/bench-recall-*.md | sort
```

---

## Phase 4 — Optional: stress test

After completing the matrix, run a sustained load test against the best-performing
configuration to observe behaviour under saturation:

```bash
docker run --rm --network host \
  -v "$(pwd)/corpus-smoke:/data/corpus:ro" \
  -v "$(pwd)/results:/app/results" \
  aiven-semantic-search-bench:lean \
  bench-stress \
    --embed-dim 768 --engine faiss --mode in_memory \
    --duration 300 --index-clients 4 --search-clients 8 \
    --label "local/smoke/stress" \
    --opensearch-uri http://localhost:9200
```

---

## Phase 5 — Tear down

```bash
docker compose -f docker-compose.opensearch.yml down
# Remove persistent volume (resets all index data):
docker compose -f docker-compose.opensearch.yml down -v
# Stop Colima:
colima stop
```

---

## Running via the loader API (orchestrator integration)

The lean container defaults to the FastAPI loader API on port 8080. The orchestrator
can submit the same matrix as JSON job specs:

```bash
docker run -d -p 8080:8080 \
  -e LOADER_API_KEY=dev-key \
  -v "$(pwd)/corpus-smoke:/data/corpus:ro" \
  -v "$(pwd)/results:/data/results" \
  aiven-semantic-search-bench:lean

curl -H "Authorization: Bearer dev-key" http://localhost:8080/healthz
```

The orchestrator reads `bench-plan.sh` (or an equivalent JSON matrix) and POSTs
individual job specs to `/run` with the OpenSearch URI inline.
