"""
Deployment metadata + OpenSearch preflight for benchmark reports.

Collects (when possible):

- **Loader** — hostname and optional Aiven plan/cloud via API or env overrides.
- **Target OpenSearch** — host/port from the benchmark URI and optional Aiven
  plan/cloud via API or env overrides.
- **Preflight** — timed HTTPS ``GET /`` to the OpenSearch base URL (same path as
  normal client traffic: TLS + HTTP + auth), repeated ``BENCH_PREFLIGHT_ROUNDS``
  times. ICMP ping is often blocked from containers; this measures realistic RTT.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any
from urllib.parse import unquote, urlparse

import httpx

from .config import Settings
from .stats import percentiles_ms


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _opensearch_base_url_and_auth(uri: str) -> tuple[str | None, tuple[str, str] | None]:
    if not uri.strip():
        return None, None
    p = urlparse(uri)
    if not p.scheme or not p.hostname:
        return None, None
    port = f":{p.port}" if p.port else ""
    base = f"{p.scheme}://{p.hostname}{port}/"
    if p.username is not None or p.password is not None:
        auth = (unquote(p.username or ""), unquote(p.password or ""))
    else:
        auth = None
    return base, auth


def preflight_opensearch(uri: str) -> dict[str, Any]:
    """
    Measure round-trip latency to the cluster root over HTTPS using the same
    credentials as ``uri``. Returns a dict suitable for report ``preflight``.
    """
    rounds = int(_env("BENCH_PREFLIGHT_ROUNDS") or "5")
    rounds = max(1, min(rounds, 30))
    timeout_s = float(_env("BENCH_PREFLIGHT_TIMEOUT_S") or "15")

    base, auth = _opensearch_base_url_and_auth(uri)
    if not base:
        return {
            "ok": False,
            "error": "missing or invalid OPENSEARCH_URI",
            "method": "HTTPS GET /",
            "rounds": 0,
        }

    samples: list[float] = []
    last_err = ""
    try:
        with httpx.Client(timeout=timeout_s, verify=True) as client:
            for _ in range(rounds):
                start = time.perf_counter()
                resp = client.get(base, auth=auth)
                elapsed_ms = (time.perf_counter() - start) * 1000
                samples.append(elapsed_ms)
                # Any HTTP response means TCP+TLS+HTTP completed.
                if resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}"
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "method": "HTTPS GET /",
            "rounds": len(samples),
            "target_base_url": base.split("@")[-1],
        }

    if not samples:
        return {
            "ok": False,
            "error": last_err or "no samples collected",
            "method": "HTTPS GET /",
            "rounds": 0,
            "target_base_url": base.split("@")[-1],
        }

    lat = percentiles_ms(samples)
    return {
        "ok": True,
        "method": "HTTPS GET /",
        "rounds": len(samples),
        "target_base_url": base.split("@")[-1],
        "min_ms": round(min(samples), 3),
        "p50_ms": round(lat["p50_ms"], 3),
        "p95_ms": round(lat["p95_ms"], 3),
        "p99_ms": round(lat["p99_ms"], 3),
        "max_ms": round(lat["max_ms"], 3),
        "mean_ms": round(lat["mean_ms"], 3),
    }


def collect_deployment_context(
    settings: Settings,
    opensearch_uri: str,
    *,
    aiven_api_token: str | None = None,
    aiven_project: str | None = None,
) -> dict[str, Any]:
    """
    Build deployment section: loader host + optional Aiven plan/cloud for loader
    and target OpenSearch.

    Resolution order: explicit ``BENCH_*`` env vars, then Aiven API when
    ``AIVEN_API_TOKEN``, ``AIVEN_PROJECT``, and the relevant service name env
    are set.
    """
    token = (
        (aiven_api_token if aiven_api_token is not None else settings.aiven_api_token)
        or _env("AIVEN_API_TOKEN")
    ).strip()
    project = (
        (aiven_project if aiven_project is not None else settings.aiven_project)
        or _env("AIVEN_PROJECT")
    ).strip()

    loader = {
        "hostname": _env("BENCH_LOADER_HOSTNAME")
        or _env("HOSTNAME")
        or socket.gethostname(),
        "service_name": _env("BENCH_LOADER_SERVICE_NAME"),
        "plan": _env("BENCH_LOADER_PLAN"),
        "cloud_name": _env("BENCH_LOADER_CLOUD"),
        "metadata_source": "unknown",
    }
    tgt_svc = _env("BENCH_TARGET_OPENSEARCH_SERVICE_NAME")
    if not tgt_svc:
        tgt_svc = (settings.aiven_service_name or "").strip() or None
    target = {
        "host": "",
        "port": None,
        "service_name": tgt_svc,
        "plan": _env("BENCH_TARGET_OPENSEARCH_PLAN"),
        "cloud_name": _env("BENCH_TARGET_OPENSEARCH_CLOUD"),
        "metadata_source": "unknown",
    }

    p = urlparse(opensearch_uri)
    if p.hostname:
        target["host"] = p.hostname
        target["port"] = p.port

    loader_api_name = _env("BENCH_LOADER_AIVEN_SERVICE_NAME")
    if loader["plan"] or loader["cloud_name"] or loader["service_name"]:
        loader["metadata_source"] = "env"
    if target["plan"] or target["cloud_name"] or target["service_name"]:
        target["metadata_source"] = "env"

    if token and project:
        from .aiven_client import AivenDiscovery

        disc = AivenDiscovery(api_token=token)
        if loader_api_name:
            try:
                info = disc.get_service(project, loader_api_name)
                if not loader["plan"]:
                    loader["plan"] = info.plan
                if not loader["cloud_name"]:
                    loader["cloud_name"] = info.cloud_name
                if not loader["service_name"]:
                    loader["service_name"] = info.name
                loader["metadata_source"] = "aiven_api"
            except Exception:
                if loader["metadata_source"] == "unknown":
                    loader["metadata_source"] = "aiven_api_error"

        tgt_name = (target.get("service_name") or "").strip()
        if tgt_name:
            try:
                info = disc.get_service(project, tgt_name)
                if not target["plan"]:
                    target["plan"] = info.plan
                if not target["cloud_name"]:
                    target["cloud_name"] = info.cloud_name
                target["metadata_source"] = "aiven_api"
            except Exception:
                if target["metadata_source"] == "unknown":
                    target["metadata_source"] = "aiven_api_error"

    # Normalize empty strings for JSON cleanliness
    for block in (loader, target):
        for k, v in list(block.items()):
            if v == "":
                block[k] = None

    return {"loader": loader, "target_opensearch": target}


def benchmark_report_extras(
    settings: Settings,
    opensearch_uri: str,
    *,
    aiven_api_token: str | None = None,
    aiven_project: str | None = None,
    print_preflight: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Run preflight (always when URI non-empty) and collect deployment metadata.

    Prints a one-line preflight summary when ``print_preflight`` is True.
    """
    deployment = collect_deployment_context(
        settings,
        opensearch_uri,
        aiven_api_token=aiven_api_token,
        aiven_project=aiven_project,
    )
    preflight = preflight_opensearch(opensearch_uri)

    if print_preflight:
        if preflight.get("ok"):
            print(
                f"[preflight] OpenSearch {preflight.get('method')} "
                f"({preflight.get('rounds')} rounds, {preflight.get('target_base_url')}): "
                f"min={preflight['min_ms']:.1f}ms "
                f"p50={preflight['p50_ms']:.1f}ms "
                f"p95={preflight['p95_ms']:.1f}ms "
                f"mean={preflight['mean_ms']:.1f}ms"
            )
        else:
            print(
                f"[preflight] WARNING: {preflight.get('error', 'failed')} "
                f"(target={preflight.get('target_base_url', opensearch_uri[:48])})"
            )

    return deployment, preflight
