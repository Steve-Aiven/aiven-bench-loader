#!/usr/bin/env bash
# Build corpus-20k: 20 000 docs × 2 000 queries (mixed preset).
#
# A production-representative size. At this scale:
#   - Indexing throughput differences between engines become clearly visible.
#   - HNSW graph construction time is non-trivial (~60–120s at batch_size=50).
#   - IVF training becomes feasible: nlist ≈ sqrt(20000) ≈ 141, requires
#     ~141 × 39 = 5499 training points (well within 20k).
#
# Build time on Apple MPS: ~5–10 min.
# Build time on CPU:        ~15–30 min.
#
# Run from the repository root:
#
#   bash scripts/build-corpus-20k.sh
#
# Override defaults via environment:
#   HF_EMBED_DEVICE=cpu EMBED_BATCH_SIZE=8 bash scripts/build-corpus-20k.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/corpus-20k}"
DOC_COUNT="${DOC_COUNT:-20000}"
QUERY_COUNT="${QUERY_COUNT:-2000}"
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

# ── Banner ───────────────────────────────────────────────────────────────────

echo "============================================================"
echo " Building corpus-20k"
echo "  out_dir        = ${OUT_DIR}"
echo "  dataset        = ${DATASET}"
echo "  doc_count      = ${DOC_COUNT}"
echo "  query_count    = ${QUERY_COUNT}"
echo "  embed_device   = ${HF_EMBED_DEVICE}"
echo "  embed_batch    = ${EMBED_BATCH_SIZE}"
echo "  seed           = ${SEED}"
echo "  streaming      = ${BENCH_HF_DATASETS_STREAMING}"
echo ""
echo "  Estimated build time:"
echo "    MPS (Apple Silicon): ~5–10 min"
echo "    CPU:                 ~15–30 min"
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
# The groundtruth computation is pure NumPy (chunked dot products) — no GPU.
# For 2 000 queries × 20 000 docs at dim=768 it typically takes 2–4 minutes.

echo ""
echo "[2/2] bench-build-groundtruth (CPU, ~2–4 min) ..."
bench-build-groundtruth \
    --corpus-dir    "${OUT_DIR}" \
    --k             100

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo " corpus-20k complete: ${OUT_DIR}"
echo " Use with bench-plan.sh:"
echo "   CORPUS_DIR=${OUT_DIR} DOC_COUNT=20000 QUERY_COUNT=2000 bash bench-plan.sh"
echo ""
echo " IVF note: with 20k docs, use nlist ≈ 141 (sqrt(20000))."
echo "   Requires ≥ 5 499 training points before the model is ready."
echo "============================================================"
