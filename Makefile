# Aiven OpenSearch k-NN Benchmark — developer shortcuts
#
# Local dev uses a .venv with the native torch (MPS on Apple Silicon).
# Docker uses a CPU-only torch to keep the image lean and avoid CUDA blowup.
#
# Usage:
#   make dev        — create .venv and install all dependencies (first time)
#   make run        — start the NiceGUI UI at http://localhost:8080
#   make docker     — build + start the Docker container (production path)
#   make docker-build — rebuild the Docker image without starting it
#   make corpus     — build the corpus natively (MPS on M-series, fastest)
#   make clean      — remove .venv and Python cache files

PYTHON   ?= python3
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYEXEC   := $(VENV)/bin/python

.PHONY: dev run corpus docker docker-build clean help

## ── Local development ────────────────────────────────────────────────────────

$(VENV)/bin/activate: requirements.txt pyproject.toml
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip --quiet
	# Install native torch first (MPS on Apple Silicon, CUDA on Linux with GPU).
	# This gives the fastest local corpus builds without needing Docker.
	$(PIP) install torch --quiet
	$(PIP) install -r requirements.txt --quiet
	$(PIP) install -e . --quiet
	@echo ""
	@echo "✓ Venv ready. Run 'make run' to start the UI."

dev: $(VENV)/bin/activate   ## Create .venv and install all dependencies

run: dev   ## Start the NiceGUI UI locally (http://localhost:8080)
	@echo "──────────────────────────────────────────────────────────────────"
	@echo "  NiceGUI UI → http://localhost:8080   (Ctrl-C to stop)"
	@echo "──────────────────────────────────────────────────────────────────"
	PYTHONUNBUFFERED=1 $(PYEXEC) -u dashboard/app.py

corpus: dev   ## Build the embedding corpus natively (fastest on Apple Silicon)
	$(PYEXEC) -m aiven_semantic_search_bench bench-build-corpus \
		--dataset mixed --doc-count 10000 --query-count 1000

## ── Docker (production) ──────────────────────────────────────────────────────

docker-build:   ## Rebuild the Docker image (CPU-only torch, production config)
	docker compose build bench

docker: docker-build   ## Rebuild and start the Docker container (http://localhost:8080)
	docker compose up bench

## ── Housekeeping ─────────────────────────────────────────────────────────────

clean:   ## Remove .venv and Python bytecode caches
	rm -rf $(VENV)
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -not -path './.venv/*' -delete 2>/dev/null || true

help:   ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
