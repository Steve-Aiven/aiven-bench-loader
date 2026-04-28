# syntax=docker/dockerfile:1.7

# ---------- Builder stage ----------
# Installs Python dependencies into an isolated virtualenv that gets copied
# into the runtime stage.  Keeping pip and build artifacts out of the final
# image keeps it small and reduces the attack surface.
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

RUN python3 -m venv /opt/venv \
    # Install CPU-only PyTorch first to avoid pulling ~3 GB of CUDA/cuDNN libs.
    # sentence-transformers will find torch already installed and skip the CUDA wheel.
 && /opt/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && /opt/venv/bin/pip install -r requirements.txt \
 && /opt/venv/bin/pip install .

# ---------- Runtime stage ----------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Run as a non-root user.  The uid/gid are fixed so that volume permissions
# are predictable on hosts that mount results/ or corpus/ from outside the
# container.
RUN groupadd --system --gid 1000 bench \
 && useradd  --system --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin bench

WORKDIR /app
RUN mkdir -p /app/results /data/corpus /data/results \
 && chown -R bench:bench /app /data

COPY --from=builder /opt/venv /opt/venv
COPY --chown=bench:bench dashboard/ /app/dashboard/
COPY --chown=bench:bench src/ /app/src/

USER bench

# Make the src package importable without pip install in runtime stage.
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

EXPOSE 8080

# Two run modes share the image:
#   * Default: standalone NiceGUI dashboard (./dashboard/app.py on :8080).
#   * LOADER_MODE=1 (with LOADER_API_KEY): FastAPI loader shim served by
#     uvicorn — what aiven/aiven-bench-orchestrator deploys as an
#     Aiven Application.
# The CLI commands (bench-build-corpus, bench-index, bench-search, ...) stay
# available either way; just override CMD when invoking the container.
ENTRYPOINT ["python3", "-c", "\
import os, subprocess, sys; \
cmd = ['uvicorn', 'dashboard.api:app', '--host', '0.0.0.0', '--port', '8080'] \
    if os.environ.get('LOADER_MODE') == '1' \
    else ['python3', '/app/dashboard/app.py']; \
sys.exit(subprocess.call(cmd))"]
CMD []
