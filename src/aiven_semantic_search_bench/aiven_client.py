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

    def list_available_plans(self, project: str, service_name: str) -> list[str]:
        """
        Return sorted plan names available for the given service's cloud region.

        Calls ``GET /v1/project/{project}/service_types`` to get all OpenSearch
        plans, then filters to those whose ``regions`` map contains the
        service's ``cloud_name``.  Falls back to the full unfiltered plan list
        if the cloud filter matches nothing (e.g. cloud name not found in the
        plan data).  Returns an empty list only on hard API errors.
        """
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                # Fetch the service to get its current cloud and plan.
                svc_resp = client.get(
                    f"{_BASE_URL}/project/{project}/service/{service_name}",
                    headers=self._headers(),
                )
                svc_resp.raise_for_status()
                svc_data = svc_resp.json().get("service", svc_resp.json())
                cloud = svc_data.get("cloud_name", "")

                # Fetch all available plan definitions for this project.
                types_resp = client.get(
                    f"{_BASE_URL}/project/{project}/service_types",
                    headers=self._headers(),
                )
                types_resp.raise_for_status()
                service_plans = (
                    types_resp.json()
                    .get("service_types", {})
                    .get("opensearch", {})
                    .get("service_plans", [])
                )

            def _is_standard(plan_name: str) -> bool:
                # custom-* plans require special account entitlement and will
                # be rejected with "Invalid plan or cloud setting" if not enabled.
                return not plan_name.startswith("custom-")

            all_names = sorted(
                p["service_plan"]
                for p in service_plans
                if "service_plan" in p and _is_standard(p["service_plan"])
            )
            if cloud:
                cloud_names = sorted(
                    p["service_plan"]
                    for p in service_plans
                    if "service_plan" in p
                    and _is_standard(p["service_plan"])
                    and cloud in p.get("regions", {})
                )
                return cloud_names if cloud_names else all_names
            return all_names
        except Exception:
            return []

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
            if not resp.is_success:
                # Extract Aiven's own error message from the JSON body when available.
                try:
                    body = resp.json()
                    aiven_msg = body.get("message") or str(body)
                except Exception:
                    aiven_msg = resp.text
                raise httpx.HTTPStatusError(
                    f"Aiven API {resp.status_code}: {aiven_msg}",
                    request=resp.request,
                    response=resp,
                )
            return resp.json()

    def wait_for_running(self, *, timeout_s: int = 600, poll_interval_s: int = 15) -> str:
        """
        Block until the service state is ``RUNNING``.

        Returns the final plan name.  Raises ``TimeoutError`` if the service
        does not reach RUNNING within ``timeout_s`` seconds.
        """
        import time as _time
        deadline = _time.monotonic() + timeout_s
        while _time.monotonic() < deadline:
            state, plan = self.get_state_and_plan()
            if state == "RUNNING":
                return plan
            _time.sleep(poll_interval_s)
        raise TimeoutError(
            f"Service {self.service_name!r} did not reach RUNNING "
            f"within {timeout_s}s (last state: {state!r})"
        )


# ── Thanos provisioning helpers ───────────────────────────────────────────────

_THANOS_SERVICE_TYPE = "thanos"
_THANOS_DEFAULT_PLAN = "startup-4"
_BENCH_THANOS_NAME   = "bench-metrics"


def _thanos_query_uri(client: httpx.Client, base: str, headers: dict, project: str, svc_name: str) -> str:
    """Return the Prometheus query-frontend URI for a Thanos service, or '' if unavailable."""
    r = client.get(f"{base}/project/{project}/service/{svc_name}", headers=headers)
    if r.status_code != 200:
        return ""
    svc = r.json().get("service", r.json())
    ci = svc.get("connection_info", {})
    return (
        ci.get("query_frontend_uri")
        or ci.get("query_uri")
        or svc.get("service_uri", "")
    )


def detect_thanos_integrations(
    *,
    api_token: str,
    project: str,
    opensearch_service_name: str,
) -> list[dict]:
    """
    Return all ``metrics`` integrations for the given OpenSearch service,
    enriched with the Thanos service state and query URI.

    Each dict has keys:
    - ``thanos_service``: Thanos service name
    - ``thanos_project``: project it lives in
    - ``thanos_state``: RUNNING / POWEROFF / REBUILDING / ...
    - ``query_uri``: Prometheus-compatible query URI ('' if not accessible)
    - ``active``: bool — whether Aiven considers the integration active
    - ``integration_id``: str or None
    - ``same_project``: bool — whether Thanos is in the same project as caller
    """
    headers = {"Authorization": f"aivenv1 {api_token}", "Content-Type": "application/json"}
    base = _BASE_URL

    with httpx.Client(timeout=30) as client:
        r = client.get(
            f"{base}/project/{project}/service/{opensearch_service_name}/integration",
            headers=headers,
        )
        r.raise_for_status()
        integrations = r.json().get("service_integrations", [])

        results = []
        for integ in integrations:
            if integ.get("integration_type") != "metrics":
                continue
            dest_svc  = integ.get("dest_service", "")
            dest_proj = integ.get("dest_project", project)

            # Fetch Thanos state (may be a cross-project reference)
            state_r = client.get(
                f"{base}/project/{dest_proj}/service/{dest_svc}",
                headers=headers,
            )
            thanos_state = "unknown"
            query_uri    = ""
            if state_r.status_code == 200:
                thanos_svc = state_r.json().get("service", state_r.json())
                thanos_state = thanos_svc.get("state", "unknown")
                ci = thanos_svc.get("connection_info", {})
                query_uri = (
                    ci.get("query_frontend_uri")
                    or ci.get("query_uri")
                    or thanos_svc.get("service_uri", "")
                )

            results.append({
                "thanos_service":  dest_svc,
                "thanos_project":  dest_proj,
                "thanos_state":    thanos_state,
                "query_uri":       query_uri,
                "active":          integ.get("active", False),
                "integration_id":  integ.get("service_integration_id"),
                "same_project":    dest_proj == project,
            })

        # Best-first ordering: same project → RUNNING → active
        results.sort(key=lambda x: (
            0 if x["same_project"] else 1,
            0 if x["thanos_state"] == "RUNNING" else 1,
            0 if x["active"] else 1,
        ))
        return results


def setup_thanos_for_opensearch(
    *,
    api_token: str,
    project: str,
    opensearch_service_name: str,
    cloud_name: str = "",  # kept for API compatibility, not used (Aiven inherits project default)
    thanos_service_name: str = _BENCH_THANOS_NAME,
    thanos_plan: str = _THANOS_DEFAULT_PLAN,
    wait_timeout_s: int = 300,
) -> tuple[str, str]:
    """
    Ensure a Thanos service in ``project`` is receiving metrics from
    ``opensearch_service_name``, creating one if needed.

    Decision logic (in priority order):
    1. If there's already a RUNNING Thanos in **the same project**, return its
       query URI immediately — no new service is created.
    2. Otherwise create ``thanos_service_name`` (startup-4, same project),
       wait for it to reach RUNNING, then return its URI.

    Returns ``(query_uri, thanos_service_name)`` where ``query_uri`` is the
    Prometheus-compatible Thanos query-frontend URL.
    Raises on any unrecoverable API error.
    """
    import time as _time

    headers = {"Authorization": f"aivenv1 {api_token}", "Content-Type": "application/json"}
    base = _BASE_URL

    # ── 1. Check existing integrations ────────────────────────────────────
    existing = detect_thanos_integrations(
        api_token=api_token,
        project=project,
        opensearch_service_name=opensearch_service_name,
    )
    same_project_running = [
        e for e in existing
        if e["same_project"] and e["thanos_state"] == "RUNNING" and e["query_uri"]
    ]
    if same_project_running:
        best = same_project_running[0]
        print(
            f"[thanos-setup] Using existing RUNNING Thanos "
            f"'{best['thanos_service']}' in project '{best['thanos_project']}' "
            f"(active={best['active']})."
        )
        return best["query_uri"], best["thanos_service"]

    with httpx.Client(timeout=60) as client:
        # ── 2. Create Thanos service if it doesn't exist ──────────────────
        check = client.get(
            f"{base}/project/{project}/service/{thanos_service_name}",
            headers=headers,
        )
        if check.status_code == 404:
            print(f"[thanos-setup] Creating Thanos service '{thanos_service_name}'…")
            # Thanos create does not accept cloud_name; the project's default cloud is used.
            resp = client.post(
                f"{base}/project/{project}/service",
                headers=headers,
                json={
                    "plan":         thanos_plan,
                    "service_name": thanos_service_name,
                    "service_type": _THANOS_SERVICE_TYPE,
                },
            )
            resp.raise_for_status()
        elif check.status_code == 200:
            print(f"[thanos-setup] Thanos service '{thanos_service_name}' already exists.")
        else:
            check.raise_for_status()

        # ── 3. Wait for Thanos to reach RUNNING ───────────────────────────
        print(f"[thanos-setup] Waiting for '{thanos_service_name}' to reach RUNNING…")
        deadline = _time.monotonic() + wait_timeout_s
        state = "UNKNOWN"
        while _time.monotonic() < deadline:
            r = client.get(
                f"{base}/project/{project}/service/{thanos_service_name}",
                headers=headers,
            )
            r.raise_for_status()
            svc = r.json().get("service", r.json())
            state = svc.get("state", "UNKNOWN")
            if state == "RUNNING":
                break
            print(f"[thanos-setup]   state={state!r} — waiting…")
            _time.sleep(15)
        else:
            raise TimeoutError(
                f"Thanos service '{thanos_service_name}' did not reach RUNNING "
                f"within {wait_timeout_s}s (last state: {state!r})"
            )
        print(f"[thanos-setup] '{thanos_service_name}' is RUNNING.")

        # ── 4. Confirm metrics integration (auto-created by Aiven) ────────
        integ_r = client.get(
            f"{base}/project/{project}/service/{thanos_service_name}/integration",
            headers=headers,
        )
        integrations = integ_r.json().get("service_integrations", [])
        os_integ = next(
            (i for i in integrations
             if i.get("source_service") == opensearch_service_name
             and i.get("integration_type") == "metrics"),
            None,
        )
        if os_integ:
            print(
                f"[thanos-setup] Metrics integration confirmed: "
                f"'{opensearch_service_name}' → '{thanos_service_name}' "
                f"active={os_integ.get('active')}."
            )
        else:
            print(
                "[thanos-setup] Metrics integration not yet visible — "
                "Aiven will provision it automatically."
            )

        # ── 5. Return the Thanos query-frontend URI ────────────────────────
        query_uri = _thanos_query_uri(client, base, headers, project, thanos_service_name)
        print(f"[thanos-setup] Thanos query URI ready.")
        return query_uri, thanos_service_name
