"""
Aiven REST API client.

Two usage patterns:

1. ``AivenDiscovery`` — token-only, no project/service pre-binding.
   Used by the Streamlit UI to list projects and services after the user
   pastes their personal API token, and to resolve the OpenSearch URI for
   a selected service.

2. ``AivenClient`` — bound to a specific project + service.
   Used by ``bench-plan-change`` to poll service state and trigger plan
   changes via the Aiven REST API.

API reference: https://api.aiven.io/doc/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

_BASE_URL = "https://api.aiven.io/v1"


# ── Discovery (UI login flow) ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ServiceInfo:
    """Minimal view of an Aiven service returned by the discovery helpers."""

    name: str
    service_type: str
    plan: str
    cloud_name: str
    state: str
    opensearch_version: str | None
    service_uri: str | None


@dataclass(frozen=True)
class AivenDiscovery:
    """
    Token-scoped Aiven API client.

    No project or service name is bound at construction time — the caller
    passes them per-method. This matches the UI flow where the user selects
    a project from a list and then selects services from a table.

    The token is never persisted to disk; it should live only in
    ``st.session_state`` on the Streamlit side.
    """

    api_token: str
    timeout_s: float = 30.0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"aivenv1 {self.api_token}",
            "Content-Type": "application/json",
        }

    # ── Project discovery ────────────────────────────────────────────────────

    def list_projects(self) -> list[dict[str, Any]]:
        """
        Return a list of project dicts the token can see.

        Each dict has at least: ``project_name``, ``account_id``.
        """
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.get(f"{_BASE_URL}/project", headers=self._headers())
            resp.raise_for_status()
            return resp.json().get("projects", [])

    def project_names(self) -> list[str]:
        """Convenience wrapper — return sorted project name strings."""
        return sorted(p["project_name"] for p in self.list_projects())

    # ── Service discovery ────────────────────────────────────────────────────

    def list_services(self, project: str) -> list[ServiceInfo]:
        """
        Return OpenSearch services in ``project``.

        Calls ``GET /v1/project/{project}/service`` and filters to
        ``service_type == "opensearch"``. Returns a list of ``ServiceInfo``
        objects with enough metadata to drive the UI services table.
        """
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.get(
                f"{_BASE_URL}/project/{project}/service",
                headers=self._headers(),
            )
            resp.raise_for_status()
            services = resp.json().get("services", [])

        out: list[ServiceInfo] = []
        for svc in services:
            if svc.get("service_type") != "opensearch":
                continue
            uc = svc.get("user_config", {})
            version = (
                uc.get("opensearch_version")
                or svc.get("metadata", {}).get("opensearch_version")
            )
            uri = svc.get("service_uri")
            out.append(
                ServiceInfo(
                    name=svc["service_name"],
                    service_type=svc["service_type"],
                    plan=svc.get("plan", ""),
                    cloud_name=svc.get("cloud_name", ""),
                    state=svc.get("state", "UNKNOWN"),
                    opensearch_version=version,
                    service_uri=uri,
                )
            )
        return out

    def get_service(self, project: str, service_name: str) -> ServiceInfo:
        """
        Return a ``ServiceInfo`` for a single service.

        Raises ``httpx.HTTPStatusError`` for 4xx/5xx responses.
        """
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.get(
                f"{_BASE_URL}/project/{project}/service/{service_name}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            svc = resp.json().get("service", resp.json())

        uc = svc.get("user_config", {})
        version = (
            uc.get("opensearch_version")
            or svc.get("metadata", {}).get("opensearch_version")
        )
        return ServiceInfo(
            name=svc["service_name"],
            service_type=svc.get("service_type", "opensearch"),
            plan=svc.get("plan", ""),
            cloud_name=svc.get("cloud_name", ""),
            state=svc.get("state", "UNKNOWN"),
            opensearch_version=version,
            service_uri=svc.get("service_uri"),
        )

    def verify_token(self) -> bool:
        """
        Return True if the token is valid by calling ``GET /v1/project``.

        Does not raise — returns False on auth failure so the UI can display
        a friendly message instead of an unhandled exception.
        """
        try:
            self.list_projects()
            return True
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                return False
            raise


# ── Per-service plan management (bench-plan-change) ───────────────────────────


@dataclass(frozen=True)
class AivenClient:
    """
    Bound Aiven API client for a specific project + service.

    Used by ``bench-plan-change`` to poll service state and trigger plan
    changes. The ``AivenDiscovery`` class above should be used for the UI
    login and service-selection flow instead.
    """

    project: str
    service_name: str
    api_token: str
    timeout_s: float = 30.0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"aivenv1 {self.api_token}",
            "Content-Type": "application/json",
        }

    def _service_url(self) -> str:
        return f"{_BASE_URL}/project/{self.project}/service/{self.service_name}"

    def get_service(self) -> dict[str, Any]:
        """Return the ``service`` object — includes ``state`` and ``plan``."""
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.get(self._service_url(), headers=self._headers())
            resp.raise_for_status()
            return resp.json().get("service", resp.json())

    def get_state_and_plan(self) -> tuple[str, str]:
        """Convenience wrapper — returns ``(state, plan)``."""
        svc = self.get_service()
        return svc.get("state", "UNKNOWN"), svc.get("plan", "UNKNOWN")

    def update_plan(self, new_plan: str) -> dict[str, Any]:
        """
        Trigger a plan change. The API returns immediately; the service
        transitions through ``REBALANCING`` before settling back to
        ``RUNNING`` on the new plan. Use ``wait_for_running`` to block.
        """
        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.put(
                self._service_url(),
                headers=self._headers(),
                json={"plan": new_plan},
            )
            resp.raise_for_status()
            return resp.json()
