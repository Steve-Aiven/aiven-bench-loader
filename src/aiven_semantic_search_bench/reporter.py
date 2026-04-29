"""
Benchmark report writer.

Every benchmark produces two artifacts:

1. A JSON file - the raw, machine-readable result. Useful for diffing across
   runs, plotting, or feeding into another tool.
2. A Markdown file - a human-readable summary you can paste into a blog post
   or share with a teammate. Includes a configuration block, a results table,
   and a notes section.

Both files share a timestamped filename so re-runs do not overwrite previous
results. They live under `results/` by default (gitignored).
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any


def _format_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 100:
            return f"{v:.1f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        return f"{v:.4f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={_format_value(val)}" for k, val in v.items())
    return str(v)


def _render_markdown(name: str, params: dict[str, Any], results: list[dict[str, Any]], notes: list[str]) -> str:
    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}_")
    lines.append("")

    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Python: `{platform.python_version()}`")
    lines.append(f"- Platform: `{platform.platform()}`")
    lines.append("")

    lines.append("## Parameters")
    lines.append("")
    for k, v in params.items():
        lines.append(f"- `{k}`: {_format_value(v)}")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    if not results:
        lines.append("_No results recorded._")
    else:
        headers = list(results[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for row in results:
            lines.append("| " + " | ".join(_format_value(row.get(h, "")) for h in headers) + " |")
    lines.append("")

    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines)


def write_report(
    name: str,
    *,
    params: dict[str, Any],
    results: list[dict[str, Any]],
    notes: list[str] | None = None,
    out_dir: str | Path = "results",
) -> tuple[Path, Path]:
    """
    Write JSON + Markdown reports under `out_dir/<name>-<timestamp>.{json,md}`.

    Returns the (json_path, md_path) tuple so the CLI can print where each
    artifact ended up.
    """
    notes = notes or []
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = p / f"{name}-{ts}.json"
    md_path = p / f"{name}-{ts}.md"

    payload = {
        "name": name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "params": params,
        "results": results,
        "notes": notes,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(_render_markdown(name, params, results, notes))

    # Notify the (optional) ClickHouse sink so the loader API can surface the
    # actual report path on the SSE `result` event and so the orchestrator can
    # find the report referenced from `bench_runs.summary`. No-op when CH is
    # not configured.
    try:
        from .clickhouse_sink import get_sink

        get_sink().report_written(json_path, md_path)
    except Exception:
        pass

    return json_path, md_path
