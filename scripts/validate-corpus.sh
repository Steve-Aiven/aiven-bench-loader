#!/usr/bin/env bash
# Validate and inspect a corpus directory.
#
# Checks that all required files exist, reports their sizes and shapes,
# and verifies the manifest is consistent.
#
# Usage:
#   bash scripts/validate-corpus.sh corpus-smoke
#   bash scripts/validate-corpus.sh corpus-20k
#   bash scripts/validate-corpus.sh /abs/path/to/corpus
#
# Returns exit code 0 if the corpus is complete and ready to use,
# non-zero if any required file is missing or the manifest is inconsistent.
set -uo pipefail

CORPUS_DIR="${1:-corpus}"

# ── Helpers ───────────────────────────────────────────────────────────────────

ok()   { echo "  [OK]  $*"; }
warn() { echo "  [!!]  $*"; }
fail() { echo "  [XX]  $*"; FAILED=1; }

human_bytes() {
    local bytes="$1"
    if   (( bytes >= 1073741824 )); then printf "%.1f GiB" "$(echo "scale=1; $bytes/1073741824" | bc)"
    elif (( bytes >= 1048576    )); then printf "%.1f MiB" "$(echo "scale=1; $bytes/1048576"    | bc)"
    elif (( bytes >= 1024       )); then printf "%.1f KiB" "$(echo "scale=1; $bytes/1024"       | bc)"
    else                                 echo "${bytes} B"
    fi
}

file_size() {
    stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null || echo "0"
}

# ── Main ──────────────────────────────────────────────────────────────────────

FAILED=0

echo "============================================================"
echo " Validating corpus: ${CORPUS_DIR}"
echo "============================================================"
echo ""

if [[ ! -d "${CORPUS_DIR}" ]]; then
    echo "  [XX] Directory does not exist: ${CORPUS_DIR}"
    exit 1
fi

# ── manifest.json ─────────────────────────────────────────────────────────────

MANIFEST="${CORPUS_DIR}/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
    fail "manifest.json missing — run bench-build-corpus first"
    FAILED=1
else
    ok "manifest.json ($(human_bytes "$(file_size "${MANIFEST}")"))"

    # Parse manifest with python (always available in repo venv)
    python3 - "${MANIFEST}" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
print(f"       preset         : {m.get('preset', '?')}")
print(f"       docs           : {m.get('actual_docs', m.get('target_docs', '?'))}")
print(f"       queries        : {m.get('actual_queries', m.get('target_queries', '?'))}")
print(f"       source_dim     : {m.get('source_dim', '?')}")
print(f"       supported_dims : {m.get('supported_dims', '?')}")
print(f"       embed_model    : {m.get('embed_model', '?')}")
print(f"       embed_backend  : {m.get('embed_backend', '?')}")
print(f"       has_metadata   : {m.get('has_metadata', False)}")
print(f"       seed           : {m.get('seed', '?')}")
print(f"       created_at_utc : {m.get('created_at_utc', '?')}")
if m.get('groundtruth_k'):
    print(f"       groundtruth_k  : {m.get('groundtruth_k')}")
    print(f"       groundtruth_at : {m.get('groundtruth_built_at_utc', '?')}")
else:
    print("       groundtruth_k  : (not yet built)")
EOF
fi

echo ""

# ── Parquet files ─────────────────────────────────────────────────────────────

echo "  Parquet files:"
for f in docs.parquet queries.parquet; do
    path="${CORPUS_DIR}/${f}"
    if [[ ! -f "${path}" ]]; then
        fail "${f} missing"
    else
        sz=$(human_bytes "$(file_size "${path}")")
        # Count rows using python
        rows=$(python3 -c "import pandas as pd; df=pd.read_parquet('${path}'); print(len(df))" 2>/dev/null || echo "(install pandas to see row count)")
        cols=$(python3 -c "import pandas as pd; df=pd.read_parquet('${path}'); print(','.join(df.columns))" 2>/dev/null || echo "")
        ok "${f}  ${rows}  ${cols:+[${cols}]}  (${sz})"
    fi
done

echo ""

# ── Embedding arrays ─────────────────────────────────────────────────────────

echo "  Embedding arrays:"
for f in docs_embeddings.npy queries_embeddings.npy; do
    path="${CORPUS_DIR}/${f}"
    if [[ ! -f "${path}" ]]; then
        fail "${f} missing"
    else
        sz=$(human_bytes "$(file_size "${path}")")
        shape=$(python3 -c "
import numpy as np
arr = np.load('${path}', mmap_mode='r')
print(f'shape={arr.shape}  dtype={arr.dtype}')
norms = np.linalg.norm(arr[:10], axis=1)
print(f'  first-10 norms: min={norms.min():.4f}  max={norms.max():.4f}  (should be ~1.0 if L2-normalised)')
" 2>/dev/null || echo "(install numpy to see shape/norms)")
        ok "${f}  ${sz}"
        echo "        ${shape}"
    fi
done

echo ""

# ── Ground truth ─────────────────────────────────────────────────────────────

echo "  Ground truth:"
QRELS="${CORPUS_DIR}/qrels.npy"
if [[ ! -f "${QRELS}" ]]; then
    warn "qrels.npy missing — run bench-build-groundtruth to enable bench-recall"
else
    sz=$(human_bytes "$(file_size "${QRELS}")")
    shape=$(python3 -c "
import numpy as np
arr = np.load('${QRELS}', mmap_mode='r')
print(f'shape={arr.shape}  dtype={arr.dtype}  ({arr.shape[0]} queries × top-{arr.shape[1]})')
" 2>/dev/null || echo "(install numpy to see shape)")
    ok "qrels.npy  ${sz}"
    echo "        ${shape}"
fi

echo ""

# ── Compatibility with bench matrix ──────────────────────────────────────────

echo "  Benchmark matrix compatibility:"
python3 - "${CORPUS_DIR}" "${MANIFEST}" <<'EOF'
import json, sys
from pathlib import Path
corpus_dir = Path(sys.argv[1])
try:
    m = json.load(open(sys.argv[2]))
except Exception:
    sys.exit(0)

supported = m.get("supported_dims", [768])
cells = {
    "L01": (768, "faiss/in_memory/float"),
    "L02": (512, "faiss/in_memory/float"),
    "L03": (256, "faiss/in_memory/float"),
    "L04": (768, "faiss/on_disk/float"),
    "L05": (768, "faiss/on_disk/32x/float"),
    "L06": (768, "lucene/in_memory/float"),
    "L07": (768, "lucene/in_memory/float + hybrid"),
    "L08": (768, "faiss/in_memory/byte"),
}
for cell_id, (dim, desc) in cells.items():
    if dim in supported:
        print(f"  [OK]  {cell_id} ({desc})")
    else:
        print(f"  [!!]  {cell_id} ({desc}) — dim={dim} not in supported_dims={supported}")
has_meta = m.get("has_metadata", False)
if not has_meta:
    print("  [!!]  L07 hybrid — docs.parquet lacks metadata columns (rebuild with --with-metadata)")
EOF

echo ""

# ── Summary ───────────────────────────────────────────────────────────────────

if [[ "${FAILED}" -ne 0 ]]; then
    echo "============================================================"
    echo " CORPUS INCOMPLETE — see failures above"
    echo " Re-run the appropriate build script:"
    echo "   bash scripts/build-corpus-smoke.sh"
    echo "============================================================"
    exit 1
else
    echo "============================================================"
    echo " Corpus is valid and ready to use"
    echo "   CORPUS_DIR=${CORPUS_DIR} bash bench-plan.sh"
    echo "============================================================"
    exit 0
fi
