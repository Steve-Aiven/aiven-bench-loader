#!/usr/bin/env bash
# Build corpus-2k-nomic: 2 000 docs × 200 queries (mixed preset).
#
# Useful for quick sanity checks — small enough to build in under a minute
# on MPS, and to run the full bench-plan.sh matrix in ~5 minutes.
#
# Run from the repository root:
#
#   bash scripts/build-corpus-2k.sh
#
# Override defaults via environment:
#   HF_EMBED_DEVICE=cpu bash scripts/build-corpus-2k.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

OUT_DIR="${OUT_DIR:-${REPO_ROOT}/corpus-2k-nomic}"
DOC_COUNT="${DOC_COUNT:-2000}"
QUERY_COUNT="${QUERY_COUNT:-200}"
SEED="${SEED:-42}"
EMBED_BATCH_SIZE="${EMBED_BATCH_SIZE:-64}"
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
echo " Building corpus-2k-nomic"
echo "  out_dir        = ${OUT_DIR}"
echo "  dataset        = ${DATASET}"
echo "  doc_count      = ${DOC_COUNT}"
echo "  query_count    = ${QUERY_COUNT}"
echo "  embed_device   = ${HF_EMBED_DEVICE}"
echo "  embed_batch    = ${EMBED_BATCH_SIZE}"
echo "  seed           = ${SEED}"
echo "  streaming      = ${BENCH_HF_DATASETS_STREAMING}"
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

echo ""
echo "[2/2] bench-build-groundtruth ..."
bench-build-groundtruth \
    --corpus-dir    "${OUT_DIR}" \
    --k             100

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo " corpus-2k-nomic complete: ${OUT_DIR}"
echo " Use with bench-plan.sh:"
echo "   CORPUS_DIR=${OUT_DIR} DOC_COUNT=2000 QUERY_COUNT=200 bash bench-plan.sh"
echo "============================================================"
