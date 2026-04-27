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
 && /opt/venv/bin/pip install -r requirements.txt \
 && /opt/venv/bin/pip install .

# ---------- Runtime stage ----------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Streamlit does not need a browser inside the container.
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run as a non-root user.  The uid/gid are fixed so that volume permissions
# are predictable on hosts that mount results/ or corpus/ from outside the
# container.
RUN groupadd --system --gid 1000 bench \
 && useradd  --system --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin bench

WORKDIR /app
RUN mkdir -p /app/results /app/corpus \
 && chown -R bench:bench /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=bench:bench dashboard/ /app/dashboard/

USER bench

EXPOSE 8501

# Default: open the Streamlit UI (starts the background job runner too).
# Override to use the CLI instead:
#   docker compose run --rm bench bench-build-corpus --help
ENTRYPOINT ["python3", "-m", "streamlit", "run", "/app/dashboard/app.py"]
CMD []
