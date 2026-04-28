.PHONY: generate generate-check test-contract install-dev

# GitHub raw URL for the canonical OpenAPI spec in the orchestrator repo.
# Update SPEC_SHA after merging changes to loader-api/openapi.yaml there.
ORCH_REPO    := aiven/aiven-bench-orchestrator
SPEC_SHA     := $(shell cat ../aiven-bench-orchestrator/loader-api/SPEC_SHA 2>/dev/null || cat .spec-sha 2>/dev/null || echo "main")
SPEC_URL     := https://raw.githubusercontent.com/$(ORCH_REPO)/$(SPEC_SHA)/loader-api/openapi.yaml
# When developing locally, use the spec directly from the sibling folder
LOCAL_SPEC   := ../aiven-bench-orchestrator/loader-api/openapi.yaml

# Regenerate dashboard/api_models.py from the OpenAPI spec.
# Prefers the local sibling path; falls back to GitHub raw URL.
generate:
	@echo "==> Generating dashboard/api_models.py from OpenAPI spec..."
	@if [ -f "$(LOCAL_SPEC)" ]; then \
		echo "    (using local spec at $(LOCAL_SPEC))"; \
		.venv/bin/datamodel-codegen \
			--input "$(LOCAL_SPEC)" \
			--input-file-type openapi \
			--output dashboard/api_models.py \
			--output-model-type pydantic_v2.BaseModel \
			--use-annotated \
			--target-python-version 3.12; \
	else \
		echo "    (fetching spec from $(SPEC_URL))"; \
		curl -fsSL "$(SPEC_URL)" -o /tmp/loader-openapi.yaml; \
		.venv/bin/datamodel-codegen \
			--input /tmp/loader-openapi.yaml \
			--input-file-type openapi \
			--output dashboard/api_models.py \
			--output-model-type pydantic_v2.BaseModel \
			--use-annotated \
			--target-python-version 3.12; \
	fi
	@echo "==> Done. dashboard/api_models.py updated."

# CI check: fail if generated models are stale.
generate-check:
	@echo "==> Checking for stale generated models..."
	@cp dashboard/api_models.py /tmp/api_models_before.py 2>/dev/null || true
	$(MAKE) generate
	@if ! diff -q /tmp/api_models_before.py dashboard/api_models.py > /dev/null 2>&1; then \
		echo "ERROR: dashboard/api_models.py is stale. Run 'make generate' and commit."; \
		exit 1; \
	fi
	@echo "==> Generated models are up-to-date."

# Run schemathesis contract tests against the live FastAPI app.
# Requires the app to be running: uvicorn dashboard.api:app --port 8081
test-contract:
	@echo "==> Running schemathesis contract tests against http://localhost:8081..."
	schemathesis run \
		--base-url http://localhost:8081 \
		--auth-type bearer \
		--auth "$(LOADER_API_KEY)" \
		$(LOCAL_SPEC)

# Install development dependencies into the project venv.
install-dev:
	python3 -m venv .venv
	.venv/bin/pip install datamodel-code-generator schemathesis "fastapi[standard]" uvicorn sse-starlette prometheus-client
