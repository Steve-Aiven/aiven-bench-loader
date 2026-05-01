"""
Timing and percentile helpers used across all benchmark scripts.

Latency distributions in benchmarks are almost never symmetric - a small
number of slow requests dominate the tail. Reporting only the mean hides
those tail effects, so every benchmark in this tool reports p50, p90, p95,
p99, p99.9, and the maximum observation. This matches the percentile set
reported by OpenSearch Benchmark (OSB) and is what you would look at when
deciding whether a service is "fast enough" for a real workload.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np


@contextmanager
def stopwatch() -> Iterator[dict[str, float]]:
    """
    Context manager that records elapsed wall-clock time in seconds.

    Usage:
        with stopwatch() as sw:
            do_work()
        print(sw["elapsed_s"])
    """
    record: dict[str, float] = {"elapsed_s": 0.0}
    t0 = time.perf_counter()
    try:
        yield record
    finally:
        record["elapsed_s"] = time.perf_counter() - t0


def percentiles_ms(values_ms: list[float]) -> dict[str, float]:
    """
    Return p50, p90, p95, p99, p99.9, max, and mean from a list of
    millisecond samples.

    p90 and p99.9 are added to match the percentile set reported by
    OpenSearch Benchmark (OSB), giving finer visibility into the tail
    without requiring a full histogram.

    Empty inputs return zeros so report templates never have to special-case
    a benchmark that produced no observations.
    """
    if not values_ms:
        return {
            "p50_ms": 0.0, "p90_ms": 0.0, "p95_ms": 0.0,
            "p99_ms": 0.0, "p999_ms": 0.0, "max_ms": 0.0,
            "mean_ms": 0.0, "count": 0,
        }

    arr = np.asarray(values_ms, dtype=float)
    return {
        "p50_ms":  float(np.percentile(arr, 50)),
        "p90_ms":  float(np.percentile(arr, 90)),
        "p95_ms":  float(np.percentile(arr, 95)),
        "p99_ms":  float(np.percentile(arr, 99)),
        "p999_ms": float(np.percentile(arr, 99.9)),
        "max_ms":  float(arr.max()),
        "mean_ms": float(arr.mean()),
        "count":   int(arr.size),
    }


def percentile_confidence(n: int) -> dict[str, str]:
    """
    Classify statistical confidence for each standard percentile level given
    ``n`` samples. Uses the rule-of-thumb minimum: at least ``100 / (1-p)``
    samples for reliable estimation at percentile ``p``.

    Returns a dict mapping percentile name → ``"ok"``, ``"low"``, or
    ``"insufficient"``.  Attach this to report notes so the reader knows
    which tail numbers to trust.

    Examples:
        n=500  → p99 is "low"   (need ~1,000; have 500)
        n=1500 → p99 is "ok"    (3 rounds × 500 queries)
        n=500  → p99.9 is "insufficient" (need ~10,000)
    """
    thresholds = {
        "p50":   20,
        "p90":   100,
        "p95":   200,
        "p99":   1_000,
        "p99.9": 10_000,
    }
    result: dict[str, str] = {}
    for pct, min_ok in thresholds.items():
        if n >= min_ok:
            result[pct] = "ok"
        elif n >= min_ok // 5:
            result[pct] = "low"
        else:
            result[pct] = "insufficient"
    return result


def confidence_note(n: int) -> str:
    """
    Human-readable one-liner for report notes.  E.g.:
      "500 samples: p50/p90/p95 reliable, p99 low (need ~1,000), p99.9 insufficient (need ~10,000)."
    """
    conf = percentile_confidence(n)
    thresholds = {"p50": 20, "p90": 100, "p95": 200, "p99": 1_000, "p99.9": 10_000}
    ok = [p for p, s in conf.items() if s == "ok"]
    low = [p for p, s in conf.items() if s == "low"]
    bad = [p for p, s in conf.items() if s == "insufficient"]
    parts: list[str] = []
    if ok:
        parts.append(f"{'/'.join(ok)} reliable")
    if low:
        parts.append(", ".join(
            f"{p} low (need ~{thresholds[p]:,})" for p in low
        ))
    if bad:
        parts.append(", ".join(
            f"{p} insufficient (need ~{thresholds[p]:,})" for p in bad
        ))
    return f"{n:,} samples: {'; '.join(parts)}."


def chunked(items: list, batch_size: int) -> list[list]:
    """Split `items` into chunks of at most `batch_size` items."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
