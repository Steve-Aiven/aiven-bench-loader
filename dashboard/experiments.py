"""
Experiment management helpers for the Aiven k-NN Benchmark dashboard.

An *experiment* is a named subdirectory under ``results/experiments/`` that
contains all the benchmark-report JSON/MD files produced by one logical test
campaign.  Each experiment directory also holds a tiny ``experiment.json``
metadata file:

    {
        "name":        "v2.17 baseline",
        "slug":        "v2_17_baseline",
        "description": "Optional free-text notes.",
        "created_at":  "2026-04-27T18:00:00"
    }

The module also provides:
- ``migrate_clean_slate`` — deletes top-level ``results/*.json/.md`` files on
  first launch (user-confirmed one-time clean-up).
- ``derive_tags`` — auto-derives a list of human-readable tag strings from a
  report dict.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# ── Constants ──────────────────────────────────────────────────────────────

EXPERIMENTS_SUBDIR = "experiments"
META_FILE = "experiment.json"


# ── Slug helpers ───────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """Convert a human name to a safe filesystem slug."""
    return re.sub(r"[^\w\-]", "_", name.strip().lower())


# ── Directory helpers ──────────────────────────────────────────────────────

def experiments_root(results_dir: str | Path) -> Path:
    p = Path(results_dir) / EXPERIMENTS_SUBDIR
    p.mkdir(parents=True, exist_ok=True)
    return p


def experiment_dir(results_dir: str | Path, slug: str) -> Path:
    return experiments_root(results_dir) / slug


# ── One-time migration ─────────────────────────────────────────────────────

def migrate_clean_slate(results_dir: str | Path) -> int:
    """
    Delete any top-level ``*.json`` and ``*.md`` files in *results_dir* that
    were written by the old flat layout.

    Skips files that start with ``.`` (e.g. ``.bench_session.json``).

    Returns the number of files deleted.
    """
    p = Path(results_dir)
    removed = 0
    for ext in ("*.json", "*.md"):
        for f in sorted(p.glob(ext)):
            if f.name.startswith("."):
                continue
            f.unlink(missing_ok=True)
            removed += 1
    return removed


# ── Experiment CRUD ────────────────────────────────────────────────────────

def list_experiments(results_dir: str | Path) -> list[dict[str, str]]:
    """
    Return a list of experiment metadata dicts, ordered newest-first by
    ``created_at``.  Each dict is guaranteed to have ``name``, ``slug``, and
    ``created_at`` keys.
    """
    root = experiments_root(results_dir)
    exps: list[dict[str, str]] = []
    for meta_path in root.glob(f"*/{META_FILE}"):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
        slug = meta_path.parent.name
        exps.append({
            "name":        meta.get("name", slug),
            "slug":        slug,
            "description": meta.get("description", ""),
            "created_at":  meta.get("created_at", ""),
        })
    exps.sort(key=lambda x: x["created_at"], reverse=True)
    return exps


def create_experiment(
    results_dir: str | Path,
    name: str,
    description: str = "",
) -> dict[str, str]:
    """
    Create a new experiment directory and metadata file.

    If a directory with the same slug already exists the existing metadata is
    returned unchanged (idempotent).

    Returns the metadata dict.
    """
    slug = slugify(name)
    d = experiment_dir(results_dir, slug)
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / META_FILE
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            pass
    meta: dict[str, str] = {
        "name":        name,
        "slug":        slug,
        "description": description,
        "created_at":  time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def rename_experiment(
    results_dir: str | Path,
    slug: str,
    new_name: str,
    new_description: str | None = None,
) -> dict[str, str]:
    """
    Update the ``name`` (and optionally ``description``) in an experiment's
    metadata file.  The directory itself is *not* renamed — the slug is stable.

    Returns the updated metadata dict.
    """
    meta_path = experiment_dir(results_dir, slug) / META_FILE
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        meta = {"slug": slug, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    meta["name"] = new_name
    if new_description is not None:
        meta["description"] = new_description
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def delete_experiment(results_dir: str | Path, slug: str) -> int:
    """
    Delete an experiment directory and all its contents.

    Returns the number of files deleted.
    """
    import shutil
    d = experiment_dir(results_dir, slug)
    if not d.exists():
        return 0
    count = sum(1 for _ in d.rglob("*") if _.is_file())
    shutil.rmtree(d)
    return count


# ── Report helpers ─────────────────────────────────────────────────────────

def load_reports(results_dir: str | Path, slug: str) -> dict[str, list[dict[str, Any]]]:
    """
    Load all benchmark report JSON files from the given experiment and return
    them grouped by benchmark name (e.g. ``"bench-index"``).

    Each report dict has an extra ``_filename`` and ``_path`` key injected.
    """
    d = experiment_dir(results_dir, slug)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not d.exists():
        return {}
    for path in sorted(d.glob("*.json")):
        if path.name == META_FILE or path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data["_filename"] = path.name
        data["_path"] = str(path)
        grouped[data.get("name", "unknown")].append(data)
    return dict(grouped)


def list_reports(results_dir: str | Path, slug: str) -> list[dict[str, Any]]:
    """
    Return a flat list of all reports in the experiment, each decorated with
    ``_filename``, ``_path``, and ``_tags`` keys (plus the full report body).
    Ordered by filename (timestamp-prefixed, so chronological).
    """
    d = experiment_dir(results_dir, slug)
    reports: list[dict[str, Any]] = []
    if not d.exists():
        return []
    for path in sorted(d.glob("*.json")):
        if path.name == META_FILE or path.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        data["_filename"] = path.name
        data["_path"] = str(path)
        data["_tags"] = derive_tags(data)
        data["_size_kb"] = round(path.stat().st_size / 1024, 1)
        reports.append(data)
    return reports


def rename_report_label(path: str | Path, new_label: str) -> None:
    """
    Edit ``params.plan_label`` in the JSON file at *path* in place.
    Also patches the corresponding ``.md`` file's label line if it exists.
    """
    p = Path(path)
    data = json.loads(p.read_text())
    old_label = data.get("params", {}).get("plan_label", "")
    data.setdefault("params", {})["plan_label"] = new_label
    p.write_text(json.dumps(data, indent=2))

    md_path = p.with_suffix(".md")
    if md_path.exists():
        text = md_path.read_text()
        if old_label:
            text = text.replace(f"`plan_label`: {old_label}", f"`plan_label`: {new_label}", 1)
        md_path.write_text(text)


def delete_report(path: str | Path) -> None:
    """Remove both the .json and matching .md file for a report."""
    p = Path(path)
    p.unlink(missing_ok=True)
    md = p.with_suffix(".md")
    md.unlink(missing_ok=True)


def export_report_csv(report: dict[str, Any]) -> str:
    """
    Flatten ``report["results"]`` to CSV text.  Adds a ``plan_label`` column
    derived from ``report["params"]["plan_label"]``.
    """
    rows = report.get("results", [])
    if not rows:
        return ""
    label = report.get("params", {}).get("plan_label", "")
    headers = ["plan_label"] + list(rows[0].keys())
    lines = [",".join(headers)]
    for row in rows:
        values = [label] + [str(row.get(h, "")) for h in headers[1:]]
        lines.append(",".join(values))
    return "\n".join(lines)


# ── Auto-derived tags ──────────────────────────────────────────────────────

def derive_tags(rep: dict[str, Any]) -> list[str]:
    """
    Deterministically derive a list of human-readable tag strings from a
    report dict.  No schema changes are required — the tags are read from the
    existing ``params`` and ``params.knn_spec`` fields.
    """
    p = rep.get("params", {})
    s = p.get("knn_spec", {})
    candidates = [
        rep.get("name"),
        s.get("engine"),
        s.get("method"),
        s.get("mode"),
        s.get("compression") if s.get("compression") not in (None, "none") else None,
        s.get("data_type") if s.get("data_type") not in (None, "float") else None,
        (f"d={p['doc_count']}" if "doc_count" in p else
         f"d={p['documents']}" if "documents" in p else None),
        p.get("embed_model"),
        (f"os={p['opensearch_version']}" if "opensearch_version" in p else None),
    ]
    return [t for t in candidates if t]


# ── Experiment summary ─────────────────────────────────────────────────────

def experiment_summary(results_dir: str | Path, slug: str) -> dict[str, Any]:
    """
    Return high-level summary statistics for the experiment used in the
    Results tab header chips.
    """
    reports = list_reports(results_dir, slug)
    engines: set[str] = set()
    bench_types: set[str] = set()
    dates: list[str] = []
    for r in reports:
        s = r.get("params", {}).get("knn_spec", {})
        if s.get("engine"):
            engines.add(s["engine"])
        if r.get("name"):
            bench_types.add(r["name"])
        if r.get("generated_at"):
            dates.append(r["generated_at"][:10])
    dates.sort()
    return {
        "run_count":   len(reports),
        "bench_types": sorted(bench_types),
        "engines":     sorted(engines),
        "date_from":   dates[0] if dates else "",
        "date_to":     dates[-1] if dates else "",
    }
