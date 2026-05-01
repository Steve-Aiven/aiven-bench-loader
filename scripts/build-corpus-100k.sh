#!/usr/bin/env bash
# Build corpus-100k: 100 000 docs × 10 000 queries (mixed preset).
#
# This is the minimum corpus size where HNSW configuration parameters
# (M, ef_construction, ef_search) and vector quantization (byte, fp16)
# produce meaningfully different recall@K scores.  At this scale:
#
#   - ef_search tradeoff curves are clearly visible (50 → 256 → 1024).
#   - byte quantization shows ~3–5% recall loss vs float32.
#   - on_disk (memory-mapped) vs in_memory latency gap is measurable.
#   - IVF training is reliable: nlist ≈ 316, requires ~12 000 training pts.
#   - BM25 hybrid query is realistic (non-trivial inverted index).
#
# This corpus is the recommended standard for comparing OpenSearch service
# plans and Aiven product tiers.
#
# PREREQUISITES
# ─────────────
#   pip install -e '.[build]'
#   colima start --cpu 4 --memory 8 --disk 100     # 8 GB for OpenSearch headroom
#
# Build time estimates (768-dim nomic, MPS):
#   Embedding:    ~45–75 min  (100k docs at batch_size=16; nomic-bert-2048 needs small batches)
#   Groundtruth:  ~8–15 min   (10k queries × 100k docs, chunked NumPy)
#   Total:        ~45–60 min
#
# Storage:
#   docs_embeddings.npy:    ~230 MB  (100k × 768 × float32)
#   queries_embeddings.npy: ~23 MB   (10k × 768 × float32)
#   docs.parquet + queries: ~80 MB
#   qrels.npy:              ~4 MB    (10k × 100 × int32)
#   Total:                  ~340 MB
#
# Run from the repository root:
#
#   bash scripts/build-corpus-100k.sh
#
# Override defaults via environment:
#   DOC_COUNT=50000 QUERY_COUNT=5000 bash scripts/build-corpus-100k.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/corpus-100k}"
DOC_COUNT="${DOC_COUNT:-100000}"
QUERY_COUNT="${QUERY_COUNT:-10000}"
SEED="${SEED:-42}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-16}"
DATASET="${DATASET:-mixed}"

export HF_EMBED_DEVICE="${HF_EMBED_DEVICE:-mps}"
export BENCH_HF_DATASETS_STREAMING="${BENCH_HF_DATASETS_STREAMING:-1}"

# ── Dependency check ─────────────────────────────────────────────────────────

if ! python3 -c "import sentence_transformers" 2>/dev/null; then
    echo "[corpus] ERROR: sentence_transformers not found."
    echo "         Run: pip install -e '.[build]'"
    exit 1
fi

# ── Memory check ─────────────────────────────────────────────────────────────
# 100k docs at 768-dim requires ~2 GB for the HNSW graph in OpenSearch.
# Warn if Colima VM appears to be under-provisioned.

COLIMA_MEM=$(colima list 2>/dev/null | awk 'NR>1 && /Running/{print $5}' | head -1)
if [[ -n "${COLIMA_MEM}" && "${COLIMA_MEM}" != *"GiB"* ]] || [[ "${COLIMA_MEM}" < "6" ]]; then
    echo "[corpus] WARNING: Colima VM may need ≥ 8 GiB for indexing 100k docs."
    echo "         If OpenSearch crashes during bench-index, run:"
    echo "           colima stop && colima start --cpu 4 --memory 8 --disk 100"
    echo ""
fi

# ── Banner ───────────────────────────────────────────────────────────────────

echo "============================================================"
echo " Building corpus-100k"
echo "  out_dir        = ${OUT_DIR}"
echo "  dataset        = ${DATASET}"
echo "  doc_count      = ${DOC_COUNT}"
echo "  query_count    = ${QUERY_COUNT}"
echo "  embed_device   = ${HF_EMBED_DEVICE}"
echo "  embed_batch    = ${EMBED_BATCH_SIZE}"
echo "  seed           = ${SEED}"
echo ""
echo "  Estimated build time (768-dim nomic on MPS):"
echo "    Embedding:    ~30–45 min"
echo "    Groundtruth:  ~8–15 min"
echo "    Total:        ~45–60 min"
echo ""
echo "  Use Ctrl-C to abort — progress is checkpointed."
echo "  Resume by re-running: already-embedded shards are skipped."
echo "============================================================"

# ── Step 1: embed ─────────────────────────────────────────────────────────────

echo ""
echo "[1/2] bench-build-corpus ..."
bench-build-corpus \
    --dataset       "${DATASET}" \
    --doc-count     "${DOC_COUNT}" \
    --query-count   "${QUERY_COUNT}" \
    --embed-batch-size "${EMBED_BATCH_SIZE}" \
    --seed          "${SEED}" \
    --out-dir       "${OUT_DIR}" \
    --with-metadata

# ── Step 2: groundtruth ───────────────────────────────────────────────────────
# For 10k queries × 100k docs at dim=768, this processes in chunks of 1000
# queries to keep peak memory under ~4 GB.  Expect 8–15 min on CPU.

echo ""
echo "[2/2] bench-build-groundtruth (CPU, ~8–15 min for 10k×100k) ..."
bench-build-groundtruth \
    --corpus-dir    "${OUT_DIR}" \
    --k             100

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo " corpus-100k complete: ${OUT_DIR}"
echo ""
echo " Use with bench-plan.sh:"
echo "   CORPUS_DIR=${OUT_DIR} DOC_COUNT=100000 QUERY_COUNT=10000 bash bench-plan.sh"
echo ""
echo " OpenSearch note: increase JVM heap before indexing 100k docs:"
echo "   Edit docker-compose.opensearch.yml → OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g"
echo "   Then: docker compose -f docker-compose.opensearch.yml up -d"
echo ""
echo " IVF note: with 100k docs, use nlist ≈ 316 (sqrt(100000))."
echo "   Requires ≥ 12 324 training points — 100k docs is sufficient."
echo "============================================================"
