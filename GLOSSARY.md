# Glossary

Terms used in this benchmark suite, the test plan (`bench-plan.md`), and the
source code. Entries are grouped by concept area; forward references between
entries are linked inline.

---

## 1. Vector search fundamentals

### Embedding
A dense numerical representation of a piece of text. A sentence embedding
model (e.g. `nomic-embed-text-v1`) reads a string and produces a fixed-length
array of floating-point numbers. Two texts that are semantically similar will
produce embeddings that are close to each other in the vector space.

The length of the array is called the **embedding dimension** (see
[embed_dim](#embed_dim)).

### embed_dim
The number of dimensions in a vector. Nomic-embed produces 768-dimensional
vectors natively. The benchmark corpus also stores downsampled projections at
512 and 256 dimensions. Higher dimensions carry more semantic information but
cost more memory and compute. See also [dimension sweep](#dimension-sweep).

### L2 normalisation
Scaling a vector so that its Euclidean length equals 1.0. All vectors in this
corpus are L2-normalised before storage. A key consequence: **cosine
similarity = inner product** for L2-normalised vectors. This is why the Faiss
engine (which does not support `cosinesimil` natively) uses `innerproduct`
instead and still produces correct cosine-ranked results.

### k-NN (k-Nearest-Neighbours) search
Given a query vector, find the `k` stored document vectors that are most
similar. OpenSearch exposes this as the `knn` query type. The parameter `k` is
the number of results requested. See also [recall@K](#recallk).

### Ground truth / brute-force neighbours
The exact, mathematically correct top-K neighbours for a query, computed by
comparing the query against every document in the corpus (also called
exhaustive search). Approximate indexes (HNSW, IVF) trade a small amount of
accuracy for much faster retrieval. The ground truth is pre-computed once and
stored as `qrels.npy` inside the corpus directory. The **recall** benchmark
uses it as the reference to measure how accurate the approximate index is.

### Approximate Nearest Neighbour (ANN)
Any algorithm that finds *close-enough* neighbours without scanning every
document. All practical production vector indexes (HNSW, IVF) are ANN
algorithms. The degree of approximation is controlled by build-time and
query-time parameters and is measured by [recall@K](#recallk).

---

## 2. k-NN index parameters (the KnnSpec axes)

These are the configurable axes of the benchmark matrix. Every combination of
these values describes exactly one **cell** in the matrix and maps to a
`KnnSpec` object in code.

### engine
Which k-NN library OpenSearch delegates to.

| Value | Description |
|-------|-------------|
| `faiss` | Meta's Faiss library. Supports all data types, HNSW and IVF methods, `on_disk` mode, and compression. Generally faster for high-dimensional vectors. |
| `lucene` | Apache Lucene's built-in k-NN. Java-native, supports HNSW only, float32 only, always in-memory. Integrates tightly with Lucene's query engine, making it the only engine that supports [hybrid search](#hybrid-search). |

### method
The indexing algorithm used to organise vectors for fast retrieval.

| Value | Description |
|-------|-------------|
| `hnsw` | Hierarchical Navigable Small World. A graph-based index where each node points to its approximate nearest neighbours at multiple levels of a hierarchy. Build time is `O(n log n)`. Query time is sub-linear. Works for both Faiss and Lucene. Default and most commonly used method. |
| `ivf` | Inverted File Index. Clusters vectors into `nlist` groups (Voronoi cells) at build time using k-means. At query time, only the `nprobes` closest clusters are searched. Potentially faster than HNSW for very high-throughput workloads, but **requires a separate training step** before the index can be created. Faiss only. |

**Relationship:** HNSW builds and queries are self-contained. IVF requires a
pre-trained cluster model (a `model_id`) which is produced by a separate API
call and stored in OpenSearch's model registry.

### mode
Where HNSW graph vectors are stored at query time.

| Value | Description |
|-------|-------------|
| `in_memory` | Graph and vectors are memory-mapped into JVM heap + OS page cache. Lowest latency, highest RAM cost. |
| `on_disk` | Vectors are paged from disk on demand. Lower RAM footprint at the cost of higher latency for cache-cold queries. Faiss only. Available in OpenSearch 2.17+. |

**Relationship:** `on_disk` enables [compression](#compression). It cannot be
used with the Lucene engine.

### compression
Scalar quantization applied to stored vectors to reduce disk and memory
footprint. Only valid with `mode=on_disk` (Faiss only).

| Value | Meaning | Bits per dimension |
|-------|---------|-------------------|
| `none` | Full float32 | 32 bits |
| `1x` | SQ8 (8-bit per dim) | 8 bits, ~4× smaller |
| `2x` | SQ6 | 6 bits |
| `4x` | SQ4 | 4 bits |
| `8x` | SQ2 | 2 bits |
| `16x` | SQ1 | 1 bit |
| `32x` | Maximum compression | Smallest footprint, most recall loss |

**Relationship:** Higher compression ratios reduce disk and memory use but
also reduce [recall@K](#recallk). The L04 vs L05 cell comparison in the matrix
measures this trade-off.

### data_type
The numeric type used to store and transmit vector values. Determines memory
cost per dimension and requires specific vector encoding at index and query
time.

| Value | Storage | Range | Encoding |
|-------|---------|-------|----------|
| `float` | float32 (4 bytes/dim) | `[-1, 1]` (L2-normalised) | Send as-is |
| `fp16` | float16 (2 bytes/dim) | Same | Send float32; OpenSearch downcasts |
| `byte` | int8 (1 byte/dim) | `[-127, 127]` | Scale: `round(v × 127)` |
| `binary` | 1 bit/dim, packed | 0 or 1 | Sign-binarise + `np.packbits` (not yet implemented; requires `space_type=hamming`) |

All data types other than `float` require `engine=faiss`.

**Relationship:** `encode_vector()` in `opensearch_client.py` handles the
conversion from the float32 corpus vectors to the target data type at
index/query time. No corpus rebuild is required for `byte`, `fp16`, or
`binary` — the transformation happens in memory before the `_bulk` or `search`
API call.

### space_type
The distance metric used to rank vector similarity. Determines what it means
for two vectors to be "close."

| Value | Description | When used |
|-------|-------------|-----------|
| `cosinesimil` | Cosine similarity (angle between vectors). Produces scores in `[-1, 1]`. | Default for Lucene and specified in KnnSpec; Faiss translates this to `innerproduct`. |
| `innerproduct` | Dot product. Equivalent to cosine similarity when vectors are L2-normalised. Used internally by Faiss instead of `cosinesimil`. | Automatically applied by `build_index_mapping` for Faiss. |
| `l2` | Euclidean distance. Not used in the current corpus (which is L2-normalised, making cosine the natural metric). | Available but not part of the current matrix. |
| `hamming` | Counts differing bits between two binary vectors. Only meaningful for `data_type=binary`. | Required for L09 (planned). |

### m
An HNSW graph parameter. The number of bidirectional links each node maintains
in the graph. Higher `m` produces a denser graph: better recall and faster
queries, but more memory and longer build time. Default: 16.

### ef_construction
An HNSW build-time parameter. The size of the candidate list used while
inserting each node into the graph. Higher values produce a better graph
(higher recall) at the cost of longer index build time. Default: 128.
Does not affect query latency.

### ef_search
An HNSW query-time parameter. The number of candidate nodes explored during
a nearest-neighbour search. Higher values increase recall at the cost of
latency. Default: 256. This is the primary recall-vs-latency dial at query
time without rebuilding the index.

**Relationship between HNSW parameters:** `m` and `ef_construction` are set
once at index creation and stored in the index mapping. `ef_search` can be
changed per-query without rebuilding the index. All three are fields of
`KnnSpec`.

### nlist (IVF)
The number of Voronoi clusters (inverted lists) built during IVF training.
Controls the coarseness of the partitioning. Rule of thumb:
`nlist ≈ sqrt(doc_count)`. For 5 000 documents, `nlist ≈ 71`. Too large a
value requires more training documents (minimum ~39 × nlist) and a longer
training step; too small means each cluster is large and queries are slow.

### nprobes (IVF)
The number of IVF clusters probed at query time. Higher values increase recall
at the cost of latency. Analogous to `ef_search` for HNSW.

---

## 3. Corpus

### Corpus
A pre-built dataset stored on disk containing:

| File | Contents |
|------|----------|
| `docs.parquet` | Text documents with `doc_id`, `text`, and `source` columns |
| `vectors_768.npy` | Float32 embedding matrix, shape `(N, 768)` |
| `vectors_512.npy` | Downsampled to 512 dims |
| `vectors_256.npy` | Downsampled to 256 dims |
| `queries.parquet` | Query texts with `query_id` and `text` |
| `query_vectors_768.npy` | Query embeddings |
| `query_vectors_512.npy` | Query embeddings at 512 dims |
| `query_vectors_256.npy` | Query embeddings at 256 dims |
| `qrels.npy` | Ground-truth matrix, shape `(Q, 100)`, top-100 doc row indices per query |
| `manifest.json` | Metadata: model name, doc count, sources, build date |

**Relationship:** The bench commands (`bench-index`, `bench-search`,
`bench-recall`) all read from the same corpus. The corpus is mounted read-only
into the Docker bench container at `/data/corpus`.

### corpus-smoke
The default corpus used by the benchmark script. 5 000 documents, 500 queries,
dimensions 256/512/768, sourced equally from `fiqa`, `msmarco`, `quora`, and
`scifact` datasets. Sufficient to observe meaningful differences across the
matrix without large memory or time requirements.

### corpus-2k-nomic
A smaller corpus with 2 000 documents and 200 queries. Useful for quick
sanity checks but too small for reliable throughput measurements.

### qrels.npy
The "query relevance" file. A numpy array of shape `(Q, 100)` where each row
contains the top-100 document row indices (by cosine similarity) for one
query, computed by brute-force exhaustive search. Used exclusively by
`bench-recall` to measure how well the approximate index matches the exact
ranking. Built by `bench-build-groundtruth`.

### nomic-embed-text-v1
The embedding model used to build the corpus. Produces 768-dimensional
L2-normalised vectors. Trained by Nomic AI and available on Hugging Face. The
model is only needed to build corpora; it is not present in the lean Docker
image.

### MPS (Metal Performance Shaders)
Apple's GPU compute framework, used by PyTorch on Apple Silicon. When
`HF_EMBED_DEVICE=mps` is set, `bench-build-corpus` runs embedding on the Mac
GPU instead of the CPU, significantly reducing corpus build time. Not available
inside Docker containers.

---

## 4. Benchmark commands

Each command is a separate CLI entry point. In the Docker container they are
run as `--entrypoint <command>` arguments. They all read from the same corpus
directory and write results to `--out-dir`.

### bench-index
Measures **indexing throughput**. Resets the OpenSearch index to a clean state
and bulk-indexes `--doc-count` documents, repeating for each batch size in
`--batch-sizes`. Reports docs/second, p50/p95/p99 bulk request latency, and
total wall-clock time per batch size.

**What to watch:** Peak docs/sec. Diminishing returns at large batch sizes
indicate network or parsing overhead.

### bench-search
Measures **k-NN query latency**. Sends `--query-count` queries per round for
`--rounds` rounds, optionally across multiple concurrent clients. Includes a
warmup phase that discards results until p95 latency stabilises (OpenSearch
Benchmarks inspired). Reports p50/p90/p95/p99/p99.9 latency per round and
overall.

**What to watch:** p50 for typical latency, p99 for tail latency under steady
load.

### bench-recall
Measures **recall accuracy**. Sends `--query-count` queries and compares
OpenSearch's top-K results against the pre-computed ground truth in `qrels.npy`.
Reports recall@1, recall@5, recall@10, recall@50, recall@100 and query latency.

**What to watch:** recall@10 is the primary quality metric. Values below 0.9
indicate significant accuracy loss from the chosen compression or quantization.

### bench-hybrid
Measures **hybrid search** (BM25 + k-NN combined score). Only available with
the Lucene engine because Lucene handles both text and vector fields natively.
Requires the index to have been built with `--with-text` and `--with-metadata`.

**What to watch:** Whether hybrid recall exceeds pure k-NN recall (L06 vs L07).
A higher hybrid recall means lexical matching adds useful signal for the query
mix.

### bench-stress
Sustained mixed load test. Runs index and search operations concurrently for a
fixed duration (`--duration` seconds) at a target throughput, measuring how the
system behaves under saturation. Useful after the matrix to find the performance
cliff.

### bench-recover
Measures **cold-start latency** after a service pause (e.g. Aiven free-tier
auto-pause). Sends the first k-NN query immediately after the service wakes and
measures how long it takes to reach normal query latency. Not applicable to the
local benchmark.

### bench-plan-change
Measures the impact of an **Aiven plan change** (upgrade or downgrade) while
the service is live. Not applicable to the local benchmark.

### bench-build-corpus
One-time command to sample text from HuggingFace datasets, embed it with
`nomic-embed-text-v1`, and write the corpus files to disk. Requires the
`[build]` optional dependency group (`sentence-transformers`, `datasets`). Run
natively on the Mac, never inside Docker.

### bench-build-groundtruth
One-time command to compute brute-force top-100 neighbours for every query in
the corpus and write `qrels.npy`. Run after `bench-build-corpus`. Also requires
the `[build]` dependencies.

---

## 5. Metrics

### recall@K
The fraction of the true top-K neighbours (from ground truth) that appear in
the approximate top-K results. Ranges from 0.0 to 1.0; higher is better.

```
recall@K = |{approx top-K} ∩ {true top-K}| / K
```

For a well-tuned HNSW index, recall@10 is typically 0.95–1.0. Compression,
low `ef_search`, or data-type quantization reduce it.

**Relationship:** The brute-force ground truth in `qrels.npy` is computed at
the same `embed_dim` as the index being tested. Recall is therefore
dimension-specific — recall@10 at dim=256 is compared to ground truth at
dim=256, not to the dim=768 ideal.

### p50 / p90 / p95 / p99 / p99.9
Percentile latency values. `pN` means N% of requests completed within that
time. For example, `p95=8ms` means 95% of queries took ≤8ms; 5% took longer.

| Percentile | Meaning |
|-----------|---------|
| p50 | Median. Typical request experience. |
| p90 | 90th percentile. Better representation of the "long tail." |
| p95 | Standard SLA baseline in most systems. |
| p99 | Outlier latency. One in 100 requests. |
| p99.9 | Extreme outlier. One in 1 000 requests. Dominated by GC pauses. |

### docs/sec (indexing throughput)
The number of documents successfully indexed per second. Measured for each
batch size in `bench-index`. Reflects bulk API efficiency, network overhead,
and OpenSearch's HNSW graph construction rate.

### batch_size
The number of documents sent in a single `_bulk` API request. Each `_bulk`
call has fixed HTTP + JSON overhead, so larger batches amortise that cost but
stress the node's memory. The optimal batch size is workload-specific. The
`bench-index` sweep covers 1, 5, 10, 20, and 50.

---

## 6. Infrastructure

### OpenSearch
An open-source, Apache 2.0–licensed search and analytics engine forked from
Elasticsearch 7.10. All benchmarks in this project target the k-NN plugin,
which provides approximate nearest-neighbour search over dense vector fields.
Version used: 2.19.

### k-NN plugin
The OpenSearch component that adds the `knn_vector` field type and the `knn`
query clause. It bridges OpenSearch's document storage to underlying ANN
libraries (Faiss, Lucene). Configuration lives in the index mapping under the
`method` block of a `knn_vector` field.

### knn_vector
The OpenSearch field type used to store dense vectors. Each document that has
one of these fields stores one vector. The field mapping specifies `dimension`,
`engine`, `method`, `space_type`, `mode`, `compression_level`, and `data_type`.

### Colima
A lightweight macOS runtime for Docker and containerd using QEMU or Apple's
Virtualization.Framework. Used here as the local Docker host because Docker
Desktop is not required. Colima must be sized with enough RAM for the
OpenSearch JVM heap (≥4 GiB total recommended: 1–2 GiB for OpenSearch JVM
+ OS + bench containers).

### Lean runner image
The Docker image built from this repository's `Dockerfile`. It contains only:
- The benchmark CLI tools
- The FastAPI loader API
- The OpenSearch Python client
- ClickHouse metrics sink

It does **not** include PyTorch, sentence-transformers, NiceGUI, or any
embedding infrastructure. This keeps the image small (~163 MB content) and the
build fast (~2 minutes). Corpus building runs natively on the Mac instead.

### FastAPI loader API
An HTTP service (port 8080) exposed by the lean runner container. The
`aiven-bench-orchestrator` calls this API to submit benchmark job specs as JSON
and receive SSE-streamed progress. When running the benchmark matrix locally
from the shell, the API is not used — the CLI entry points are called directly.

### aiven-bench-orchestrator
An external service that submits benchmark jobs to the loader API, polls for
results, and stores them. Not part of this repository. In local testing, its
role is played manually by `bench-plan.sh`.

---

## 7. Result files

### .md result file
A human-readable Markdown table written to `./results/` after each benchmark
run. Named by bench type and timestamp
(e.g. `bench-recall-20260501-120000.md`). Contains the preflight ping summary,
parameters, and a formatted results table.

### .json result file
A machine-readable JSON file paired with every `.md` result. Contains the same
data in structured form, suitable for ingestion by the orchestrator or direct
import into ClickHouse.

### preflight
A brief latency probe run before each benchmark. The bench tool sends a few
HTTP GET requests to the OpenSearch cluster root before starting the actual
workload, and records min/p50/p95/mean latency. This measures network + HTTP
stack overhead independently of the benchmark itself, providing context for
interpreting benchmark numbers (especially useful on remote Aiven services
where network RTT is a non-trivial component of query latency).

### label
A dot-separated string attached to every result, used to group and compare
runs. Format used in `bench-plan.sh`: `local/smoke/{cell_id}`, e.g.
`local/smoke/L01`. The orchestrator uses similar labels to identify
which plan entry a result belongs to.

---

## 8. Benchmark matrix concepts

### Cell
One row of the benchmark matrix. A cell is a specific combination of
(`engine`, `method`, `mode`, `compression`, `data_type`, `embed_dim`) plus
the standard bench sequence: index → search → recall (and optionally hybrid).
Cells are labelled L01–L08 (runnable) and L09–L10 (planned).

### KnnSpec
The Python dataclass in `opensearch_client.py` that represents a cell's
complete k-NN configuration. One `KnnSpec` fully describes an index: it is
passed to `build_index_mapping` to create the index, and to `bench-index` /
`bench-search` / `bench-recall` to drive the workload. It serialises to JSON
for storage in result files.

### Dimension sweep (L01, L02, L03)
Running the same workload at different `embed_dim` values (768, 512, 256) to
observe the latency-recall trade-off from dimension reduction. Lower dimensions
are faster and cheaper but may lose semantic precision.

### Hybrid search (L07)
A query mode where OpenSearch blends a BM25 full-text score with a k-NN vector
score into a single ranking. Requires a Lucene index with both a `text` field
(for BM25) and a `knn_vector` field. The intuition is that lexical matching
(exact keyword hits) can boost recall for queries where semantic similarity
alone misses relevant documents.

### BM25
Best Match 25 — the probabilistic text ranking function used by OpenSearch
(and Elasticsearch) for keyword search. It scores documents based on term
frequency and inverse document frequency. In a hybrid query, the BM25 score
is combined with the k-NN score using a linear interpolation or rank-fusion
method.

### Scalar quantization (SQ)
A compression technique that maps float32 values to a smaller numeric type
(e.g. 8-bit integer) by dividing the value range into equally-sized buckets.
The `compression` parameter (`1x` through `32x`) controls how aggressively the
vectors are quantized on disk. Scalar quantization is distinct from
[data_type](#data_type) quantization: the former applies to on-disk storage
while the latter changes the data type transmitted in API requests and held in
memory.

### int8 quantization (byte data_type)
Storing and transmitting vector values as 8-bit signed integers. Each
dimension is scaled from `[-1, 1]` to `[-127, 127]` before being sent to
OpenSearch. On the Faiss side, this reduces the HNSW graph's memory footprint
by 4× compared to float32. Because the vectors are L2-normalised and
`innerproduct` is used, cosine ordering is perfectly preserved — hence
recall@10 = 1.0 on the smoke corpus.

### Sign binarisation (binary data_type, L09)
The most aggressive quantization: each float dimension is reduced to a single
bit based on its sign (positive → 1, negative → 0). 768 float32 dimensions
(3 072 bytes) become 96 bytes (96× smaller). Distance is measured with Hamming
distance (count of differing bits). Recall typically degrades more than byte
quantization but the memory savings are extreme.
