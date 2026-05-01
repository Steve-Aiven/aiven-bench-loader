# syntax=docker/dockerfile:1.7
#
# Lean benchmark runner image.
#
# This image contains only the FastAPI loader API and the measurement commands
# (bench-index, bench-search, bench-recall, bench-hybrid, bench-stress,
# bench-recover, bench-plan-change).  The NiceGUI dashboard and the embedding /
# corpus-build stack have been removed; the heavy PyTorch wheel is no longer
# installed, which cuts image size from ~1.5 GB to ~450 MB and build time from
# ~8 min to ~2 min.
#
# Corpus is always supplied as a pre-built directory mounted at /data/corpus.
# Build it once on the host (Mac with MPS) using the main branch:
#
#   HF_EMBED_DEVICE=mps python3 -m aiven_semantic_search_bench bench-build-corpus \
#       --dataset mixed --doc-count 20000 --query-count 2000 \
#       --corpus-dir ./corpus-20k-nomic
#   python3 -m aiven_semantic_search_bench bench-build-groundtruth \
#       --corpus-dir ./corpus-20k-nomic
#
# Run modes:
#   * Default (loader API): uvicorn dashboard.api:app on :8080
#       docker run -p 8080:8080 -e LOADER_API_KEY=... aiven-semantic-search-bench:lean
#   * CLI benchmark (override entrypoint):
#       docker run --entrypoint bench-index aiven-semantic-search-bench:lean \
#           --doc-count 5000 --embed-dim 768 --label "local/smoke/L01" \
#           --opensearch-uri http://host.docker.internal:9200

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install dependencies from pyproject.toml (no torch, no sentence-transformers,
# no datasets, no nicegui — just the loader API + bench runner stack).
WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install .

# Non-root bench user with predictable uid/gid for volume permission parity.
RUN groupadd --system --gid 1000 bench \
 && useradd  --system --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin bench

WORKDIR /app
RUN mkdir -p /app/results /data/corpus /data/results \
 && chown -R bench:bench /app /data

# Loader API source (api.py + generated api_models.py).
# app.py and experiments.py have been removed from this branch.
COPY --chown=bench:bench dashboard/api.py dashboard/api_models.py /app/dashboard/

# Package source — imported via PYTHONPATH (no editable install in runtime).
COPY --chown=bench:bench src/ /app/src/

USER bench

ENV PYTHONPATH="/app/src:${PYTHONPATH}"

EXPOSE 8080

# The loader API is the only HTTP mode; LOADER_MODE switch is gone.
# Override ENTRYPOINT to run a CLI bench command directly.
ENTRYPOINT ["uvicorn", "dashboard.api:app", "--host", "0.0.0.0", "--port", "8080"]
CMD []
