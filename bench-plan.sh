#!/usr/bin/env bash
# bench-plan.sh — scriptable local benchmark matrix
#
# Runs all 7 cells defined in bench-plan.md against a local OpenSearch instance.
# Each cell executes bench-index → bench-search → bench-recall (and bench-hybrid
# for lucene cells) in sequence using the lean runner Docker image.
#
# Usage:
#   ./bench-plan.sh                        # all cells, defaults
#   CORPUS_DIR=./corpus-2k-nomic ./bench-plan.sh
#   CELLS="L01 L04 L07" ./bench-plan.sh   # subset
#
# Environment variables (all optional — defaults shown):
#   CORPUS_DIR       ./corpus-smoke
#   OPENSEARCH_URI   http://localhost:9200
#   DOC_COUNT        5000
#   QUERY_COUNT      500
#   IMAGE            aiven-semantic-search-bench:lean
#   RESULTS_DIR      ./results
#   CELLS            L01 L02 L03 L04 L05 L06 L07   (space-separated subset)
#   DRY_RUN          0   (set to 1 to print commands without running them)

set -uo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

CORPUS_DIR="${CORPUS_DIR:-./corpus-smoke}"
OPENSEARCH_URI="${OPENSEARCH_URI:-http://localhost:9200}"
DOC_COUNT="${DOC_COUNT:-5000}"
QUERY_COUNT="${QUERY_COUNT:-500}"
IMAGE="${IMAGE:-aiven-semantic-search-bench:lean}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
CELLS="${CELLS:-L01 L02 L03 L04 L05 L06 L07 L08}"
DRY_RUN="${DRY_RUN:-0}"

# ── Helpers ───────────────────────────────────────────────────────────────────

log() { echo "[bench-plan] $*"; }

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY-RUN: $*"
  else
    "$@"
  fi
}

# _docker_bench <bench-command> [extra flags...]
# Runs a single bench command inside the lean container.
# ClickHouse telemetry and raw-sample env vars are forwarded when set.
_docker_bench() {
  local cmd="$1"; shift
  local ch_flags=()
  [[ -n "${CLICKHOUSE_URL:-}" ]]      && ch_flags+=(-e "CLICKHOUSE_URL=$CLICKHOUSE_URL")
  [[ -n "${CLICKHOUSE_USER:-}" ]]     && ch_flags+=(-e "CLICKHOUSE_USER=$CLICKHOUSE_USER")
  [[ -n "${CLICKHOUSE_PASSWORD:-}" ]] && ch_flags+=(-e "CLICKHOUSE_PASSWORD=$CLICKHOUSE_PASSWORD")
  [[ -n "${CLICKHOUSE_DATABASE:-}" ]] && ch_flags+=(-e "CLICKHOUSE_DATABASE=$CLICKHOUSE_DATABASE")
  [[ -n "${BENCH_SAVE_RAW_SAMPLES:-}" ]] && ch_flags+=(-e "BENCH_SAVE_RAW_SAMPLES=$BENCH_SAVE_RAW_SAMPLES")
  run docker run --rm \
    --network host \
    "${ch_flags[@]}" \
    -v "$(realpath "$CORPUS_DIR"):/data/corpus:ro" \
    -v "$(realpath "$RESULTS_DIR"):/app/results" \
    --entrypoint "$cmd" \
    "$IMAGE" \
    --opensearch-uri "$OPENSEARCH_URI" \
    --corpus-dir /data/corpus \
    --out-dir /app/results \
    "$@"
}

# cell <id> [knn flags...]
# Runs bench-index → bench-search → bench-recall for the given k-NN config.
cell() {
  local id="$1"; shift
  local label="local/smoke/${id}"
  log "=== Cell ${id}: index ==="
  _docker_bench bench-index \
    --doc-count "$DOC_COUNT" \
    --label "$label" \
    "$@"

  # Brief pause: lets the JVM GC after bulk indexing before firing search queries.
  sleep 5

  log "=== Cell ${id}: search ==="
  _docker_bench bench-search \
    --query-count "$QUERY_COUNT" \
    --rounds 3 \
    --label "$label" \
    "$@"

  # Brief pause between search and recall to let OpenSearch recover.
  sleep 5

  log "=== Cell ${id}: recall ==="
  _docker_bench bench-recall \
    --query-count "$QUERY_COUNT" \
    --label "$label" \
    "$@"
}

# hybrid_cell <id> [knn flags...]
# Runs bench-index → bench-search → bench-recall → bench-hybrid × 3 filter levels.
#
# bench-hybrid is run three times to cover all filter-selectivity modes:
#   none — pure hybrid, no metadata filter (baseline latency/recall)
#   low  — filter on a broad category (~25% of docs, "infrastructure")
#   high — filter on a narrow category (~6% of docs, "incident")
#
# Requires the corpus to have been built with --with-metadata so that
# docs.parquet contains the metadata.category and metadata.tenant_id columns.
# Rebuild with: bash scripts/build-corpus-smoke.sh
hybrid_cell() {
  local id="$1"; shift
  local label="local/smoke/${id}"
  log "=== Cell ${id}: index ==="
  _docker_bench bench-index \
    --doc-count "$DOC_COUNT" \
    --label "$label" \
    --with-text --with-metadata \
    "$@"

  sleep 5

  log "=== Cell ${id}: search ==="
  _docker_bench bench-search \
    --query-count "$QUERY_COUNT" \
    --rounds 3 \
    --label "$label" \
    --with-text --with-metadata \
    "$@"

  sleep 5

  log "=== Cell ${id}: recall ==="
  _docker_bench bench-recall \
    --query-count "$QUERY_COUNT" \
    --label "$label" \
    --with-text --with-metadata \
    "$@"

  sleep 5

  for sel in none low high; do
    log "=== Cell ${id}: hybrid (filter-selectivity=${sel}) ==="
    _docker_bench bench-hybrid \
      --query-count "$QUERY_COUNT" \
      --label "$label" \
      --filter-selectivity "$sel" \
      "$@"
    sleep 3
  done
}

# ── Preflight ─────────────────────────────────────────────────────────────────

preflight() {
  log "Checking OpenSearch at ${OPENSEARCH_URI} ..."
  if ! curl -sf "${OPENSEARCH_URI}/_cluster/health" | grep -qv '"status":"red"'; then
    echo "ERROR: OpenSearch at ${OPENSEARCH_URI} is not reachable or status is red."
    echo "  Start it with: docker compose -f docker-compose.opensearch.yml up -d"
    exit 1
  fi
  log "OpenSearch OK"

  if [[ ! -f "${CORPUS_DIR}/manifest.json" ]]; then
    echo "ERROR: No corpus manifest at ${CORPUS_DIR}/manifest.json"
    echo "  Unpack a corpus tarball or build one natively on Mac."
    exit 1
  fi
  log "Corpus OK (${CORPUS_DIR})"

  if [[ "$DRY_RUN" != "1" ]] && ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "ERROR: Docker image '${IMAGE}' not found."
    echo "  Build it with: docker build -t ${IMAGE} ."
    exit 1
  fi
  log "Image OK (${IMAGE})"

  mkdir -p "$RESULTS_DIR"
}

# ── Matrix cells ──────────────────────────────────────────────────────────────
#
# Each function below matches one row in bench-plan.md.
# Individual cells can be run by setting CELLS="L01 L04" etc.

run_L01() {
  # faiss / hnsw / in_memory / float / 768-dim
  cell L01 \
    --engine faiss --method hnsw --mode in_memory --data-type float --embed-dim 768
}

run_L02() {
  # faiss / hnsw / in_memory / float / 512-dim  (Matryoshka slice)
  cell L02 \
    --engine faiss --method hnsw --mode in_memory --data-type float --embed-dim 512
}

run_L03() {
  # faiss / hnsw / in_memory / float / 256-dim  (Matryoshka slice)
  cell L03 \
    --engine faiss --method hnsw --mode in_memory --data-type float --embed-dim 256
}

run_L04() {
  # faiss / hnsw / on_disk / no compression / float / 768-dim
  # Exercises the disk-offload path without quantization. Vectors are stored on
  # disk and paged in on access; expect higher p99 latency vs in_memory.
  cell L04 \
    --engine faiss --method hnsw --mode on_disk --compression none --data-type float --embed-dim 768
}

run_L05() {
  # faiss / hnsw / on_disk / 32x Faiss scalar compression / float / 768-dim
  # Combines disk offload with Faiss SQ compression: smaller footprint, lower
  # recall. data_type must be float — on_disk+compression does not support byte/binary.
  cell L05 \
    --engine faiss --method hnsw --mode on_disk --compression 32x --data-type float --embed-dim 768
}

run_L06() {
  # lucene / hnsw / in_memory / float / 768-dim  (recall + search only, no hybrid)
  # Pure lucene baseline — compare recall and latency vs faiss L01 with same dim.
  cell L06 \
    --engine lucene --method hnsw --mode in_memory --data-type float --embed-dim 768
}

run_L07() {
  # lucene / hnsw / in_memory / float / 768-dim + hybrid BM25+kNN
  hybrid_cell L07 \
    --engine lucene --method hnsw --mode in_memory --data-type float --embed-dim 768
}

run_L08() {
  # faiss / hnsw / in_memory / byte / 768-dim
  # Float32 vectors are scaled to int8 [-127, 127] at send time by bench_index.
  # No corpus rebuild required. ~4× smaller memory footprint than float32.
  # Requires OpenSearch >= 2.17 with Faiss engine.
  cell L08 \
    --engine faiss --method hnsw --mode in_memory --data-type byte --embed-dim 768
}

# ── L09 / L10: planned, not yet runnable ─────────────────────────────────────
#
# L09 — binary data type (faiss/hnsw/in_memory/hamming/768-bit)
#   Requires: space_type=hamming (not cosinesimil/innerproduct), vectors sent as
#   96 packed unsigned bytes (np.packbits(sign(float_vec))).
#   bench_index.py needs a _binary_encode() path + mapping builder update for
#   hamming space type.  No new corpus needed — same sign-binarization at send time.
#
# L10 — faiss IVF (faiss/ivf/in_memory/float/768)
#   Requires a two-step workflow not yet wired into bench_index:
#     1. Index a training set (~39 × nlist docs) into a temp index.
#     2. POST /_plugins/_knn/models/_train → get model_id.
#     3. Poll GET /_plugins/_knn/models/{model_id} until state="created".
#     4. Create bench index with "model_id": model_id instead of inline "method".
#   nlist should be sqrt(doc_count) ≈ 71 for 5k docs (current nlist=256 is too
#   large — requires >10k training points).

# ── Main ──────────────────────────────────────────────────────────────────────

main() {
  log "Starting local benchmark run"
  log "  CORPUS_DIR     = ${CORPUS_DIR}"
  log "  OPENSEARCH_URI = ${OPENSEARCH_URI}"
  log "  DOC_COUNT      = ${DOC_COUNT}"
  log "  QUERY_COUNT    = ${QUERY_COUNT}"
  log "  IMAGE          = ${IMAGE}"
  log "  RESULTS_DIR    = ${RESULTS_DIR}"
  log "  CELLS          = ${CELLS}"
  log "  DRY_RUN        = ${DRY_RUN}"
  echo ""

  preflight

  local start
  start=$(date +%s)

  local failed=()
  for cell_id in $CELLS; do
    if ! "run_${cell_id}"; then
      log "WARNING: Cell ${cell_id} failed — continuing with remaining cells"
      failed+=("$cell_id")
    fi
  done

  if [[ ${#failed[@]} -gt 0 ]]; then
    log "Completed with failures in cells: ${failed[*]}"
  fi

  local elapsed=$(( $(date +%s) - start ))
  log "All cells complete in ${elapsed}s. Results in ${RESULTS_DIR}/"
}

main "$@"
