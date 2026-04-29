"""
ClickHouse sink for structured benchmark telemetry.

Provides a process-wide singleton that benchmark modules call to emit
structured logs (``sink.log``) and time-series metrics (``sink.metric``)
into the orchestrator-provisioned Aiven for ClickHouse service. The
orchestrator polls the same ClickHouse for an active job's metrics and
writes control directives (cancel / throttle / pause / resume) into
``bench_control``; the loader polls those via :py:meth:`ClickHouseSink.poll_directives`.

Activation
----------
Requires four env vars (set by the orchestrator when this loader is
deployed as an Aiven Application):

- ``CLICKHOUSE_URL``      e.g. ``https://clickhouse-bench-aiven.aivencloud.com:14833``
- ``CLICKHOUSE_USER``
- ``CLICKHOUSE_PASSWORD``
- ``CLICKHOUSE_DATABASE`` (defaults to ``bench``)

When any are unset (or ``clickhouse-connect`` is not installed) every
sink method becomes a no-op so the same code path runs in CLI /
standalone NiceGUI mode without ClickHouse.

Schema ownership
----------------
The orchestrator owns the table DDL (``bench_runs``, ``bench_logs``,
``bench_metrics``, ``bench_control``) and runs migrations. The loader
just inserts rows and reads ``bench_control``. If the schema is
missing or ClickHouse is unreachable, the sink logs one warning to
stderr and continues degrading gracefully.

Threading model
---------------
- Background daemon thread flushes batched ``bench_logs`` and
  ``bench_metrics`` rows every 500 ms.
- Public methods are thread-safe (queues are ``queue.Queue``).
- Single ``bind_job`` / ``unbind_job`` pair per benchmark run; the
  loader serialises jobs via ``_job_lock`` in ``dashboard/api.py`` so
  multiple concurrent ``bind_job`` calls are not expected.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import clickhouse_connect

    _HAVE_CH = True
except ImportError:
    _HAVE_CH = False


_FLUSH_INTERVAL_S = 0.5
_BATCH_LIMIT = 1000


# ── Public dataclasses ────────────────────────────────────────────────────────


@dataclass
class Directive:
    """A control directive read from ``bench_control`` for the active job."""

    job_id: str
    seq: int
    directive: str
    payload: dict[str, Any]
    ts: datetime


@dataclass
class ThrottleState:
    """
    Live-tunable worker-pool sizing shared between the directive applier
    (``bench_stress``'s directive-pump thread) and the worker threads.

    Workers consult ``target_index`` / ``target_search`` and the two
    pause events on every loop iteration. Throttle can only scale DOWN
    from the initial count; scale-up beyond the initial allocation is
    not supported (the pool size is fixed at start-of-run).
    """

    initial_index: int
    initial_search: int
    target_index: int = field(init=False)
    target_search: int = field(init=False)
    pause_index: threading.Event = field(default_factory=threading.Event)
    pause_search: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.target_index = self.initial_index
        self.target_search = self.initial_search

    def apply(self, directive: Directive) -> str:
        """Mutate state from a directive. Returns a human-readable summary."""
        d = directive.directive
        p = directive.payload or {}
        if d == "throttle":
            pool = p.get("pool", "")
            try:
                n = int(p.get("clients", 0))
            except (TypeError, ValueError):
                n = 0
            if pool == "search" and n > 0:
                clamped = min(n, self.initial_search)
                self.target_search = clamped
                return f"throttle search → {clamped} (requested {n}, cap {self.initial_search})"
            if pool == "index" and n > 0:
                clamped = min(n, self.initial_index)
                self.target_index = clamped
                return f"throttle index → {clamped} (requested {n}, cap {self.initial_index})"
            return f"throttle ignored (pool={pool!r}, clients={n})"
        if d == "pause":
            pool = p.get("pool", "")
            if pool == "search":
                self.pause_search.set()
                return "pause search pool"
            if pool == "index":
                self.pause_index.set()
                return "pause index pool"
            return f"pause ignored (pool={pool!r})"
        if d == "resume":
            pool = p.get("pool", "")
            if pool == "search":
                self.pause_search.clear()
                return "resume search pool"
            if pool == "index":
                self.pause_index.clear()
                return "resume index pool"
            return f"resume ignored (pool={pool!r})"
        return f"unhandled directive: {d}"


# ── Sink ──────────────────────────────────────────────────────────────────────


class ClickHouseSink:
    """Singleton-ish ClickHouse telemetry sink. See module docstring."""

    def __init__(self) -> None:
        self._enabled = False
        self._client: Any = None
        self._db = "bench"

        self._job_id: str | None = None
        self._started_at: datetime | None = None
        self._spec_dump: dict[str, Any] | None = None
        self._bench_type: str | None = None
        self._label: str | None = None

        self._log_q: queue.Queue[tuple] = queue.Queue()
        self._metric_q: queue.Queue[tuple] = queue.Queue()

        self._stop = threading.Event()
        self._flush_thread: threading.Thread | None = None

        self._last_seen_seq = 0
        self._lock = threading.Lock()
        self._last_report_path: Path | None = None
        self._warned_unreachable = False

        url = os.environ.get("CLICKHOUSE_URL", "").strip()
        user = os.environ.get("CLICKHOUSE_USER", "").strip()
        pwd = os.environ.get("CLICKHOUSE_PASSWORD", "").strip()
        db = (os.environ.get("CLICKHOUSE_DATABASE", "") or "bench").strip()

        if not url or not user or not pwd or not _HAVE_CH:
            return

        try:
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or (8443 if parsed.scheme == "https" else 8123)
            scheme = parsed.scheme or "https"
            if not host:
                raise ValueError(f"CLICKHOUSE_URL has no host: {url!r}")
            self._client = clickhouse_connect.get_client(
                host=host,
                port=port,
                username=user,
                password=pwd,
                database=db,
                interface=scheme,
                secure=(scheme == "https"),
                # Reasonable defaults for a long-running service
                send_receive_timeout=30,
                connect_timeout=10,
            )
            self._db = db
            self._enabled = True
            self._flush_thread = threading.Thread(
                target=self._flush_loop, name="ch-sink-flush", daemon=True
            )
            self._flush_thread.start()
        except Exception as exc:
            sys.stderr.write(
                f"[ch-sink] WARNING: failed to connect to ClickHouse ({url}): {exc}\n"
            )
            self._enabled = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def last_report_path(self) -> Path | None:
        with self._lock:
            return self._last_report_path

    @property
    def current_job_id(self) -> str | None:
        return self._job_id

    # ── Job lifecycle ─────────────────────────────────────────────────────────

    def bind_job(
        self,
        job_id: str,
        *,
        bench_type: str = "",
        label: str = "",
        spec: dict[str, Any] | None = None,
    ) -> None:
        """Mark the start of a new benchmark job. INSERTs a row into ``bench_runs``."""
        if not self._enabled:
            # Track job_id and reset last_report_path even in no-op mode so the
            # loader API can still surface the actual report path on the SSE
            # `result` event when CH is unconfigured (CLI / standalone runs).
            with self._lock:
                self._job_id = job_id
                self._last_report_path = None
            return
        with self._lock:
            self._job_id = job_id
            self._started_at = datetime.now(timezone.utc)
            self._bench_type = bench_type
            self._label = label
            self._spec_dump = spec or {}
            self._last_seen_seq = 0
            self._last_report_path = None
        try:
            self._client.insert(
                f"{self._db}.bench_runs",
                [
                    [
                        job_id,
                        self._started_at,
                        None,
                        "running",
                        bench_type,
                        label,
                        json.dumps(spec or {}, default=str),
                        json.dumps({}),
                    ]
                ],
                column_names=[
                    "job_id",
                    "started_at",
                    "finished_at",
                    "status",
                    "bench_type",
                    "label",
                    "spec",
                    "summary",
                ],
            )
        except Exception as exc:
            self._maybe_warn(exc)

    def unbind_job(
        self, *, status: str = "ok", summary: dict[str, Any] | None = None
    ) -> None:
        """Mark the end of the bound job. Upserts the ``bench_runs`` row."""
        if self._job_id is None:
            return
        finished = datetime.now(timezone.utc)
        if self._enabled:
            try:
                self._client.insert(
                    f"{self._db}.bench_runs",
                    [
                        [
                            self._job_id,
                            self._started_at or finished,
                            finished,
                            status,
                            self._bench_type or "",
                            self._label or "",
                            json.dumps(self._spec_dump or {}, default=str),
                            json.dumps(summary or {}, default=str),
                        ]
                    ],
                    column_names=[
                        "job_id",
                        "started_at",
                        "finished_at",
                        "status",
                        "bench_type",
                        "label",
                        "spec",
                        "summary",
                    ],
                )
            except Exception as exc:
                self._maybe_warn(exc)
        with self._lock:
            self._job_id = None
            self._started_at = None
            self._bench_type = None
            self._label = None
            self._spec_dump = None
            self._last_seen_seq = 0

    # ── Telemetry ─────────────────────────────────────────────────────────────

    def log(self, level: str, phase: str, msg: str) -> None:
        """Enqueue a structured log row. No-op outside a bound job or if disabled."""
        if not self._enabled or self._job_id is None:
            return
        self._log_q.put(
            (self._job_id, datetime.now(timezone.utc), level, phase, msg)
        )

    def metric(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Enqueue a metric sample. No-op outside a bound job or if disabled."""
        if not self._enabled or self._job_id is None:
            return
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        self._metric_q.put(
            (self._job_id, datetime.now(timezone.utc), name, v, labels or {})
        )

    def report_written(
        self, json_path: Path | str, md_path: Path | str | None = None
    ) -> None:
        """Called by ``write_report``; remembers the JSON path for SSE result events."""
        with self._lock:
            self._last_report_path = Path(json_path)
        self.log("info", "report", f"Wrote report: {json_path}")
        if md_path:
            self.log("info", "report", f"Wrote markdown: {md_path}")

    # ── Control plane ─────────────────────────────────────────────────────────

    def poll_directives(self) -> list[Directive]:
        """
        Return un-applied directives for the bound job, in ``seq`` order, then
        mark them applied (INSERT a tombstone row that ReplacingMergeTree
        collapses against the original via ``ORDER BY (job_id, seq)``).
        """
        if not self._enabled or self._job_id is None:
            return []
        try:
            rows = self._client.query(
                f"SELECT job_id, ts, seq, directive, payload "
                f"FROM {self._db}.bench_control FINAL "
                f"WHERE job_id = %(job_id)s AND seq > %(seq)s "
                f"AND applied_at IS NULL "
                f"ORDER BY seq ASC",
                parameters={"job_id": self._job_id, "seq": self._last_seen_seq},
            ).result_rows
        except Exception as exc:
            self._maybe_warn(exc)
            return []

        out: list[Directive] = []
        for job_id, ts, seq, directive, payload_raw in rows:
            payload = self._parse_payload(payload_raw)
            seq_i = int(seq)
            out.append(
                Directive(
                    job_id=str(job_id),
                    seq=seq_i,
                    directive=str(directive),
                    payload=payload,
                    ts=ts,
                )
            )
            if seq_i > self._last_seen_seq:
                self._last_seen_seq = seq_i

        if out:
            applied = datetime.now(timezone.utc)
            try:
                self._client.insert(
                    f"{self._db}.bench_control",
                    [
                        [
                            d.job_id,
                            d.ts,
                            d.seq,
                            d.directive,
                            json.dumps(d.payload, default=str),
                            applied,
                        ]
                        for d in out
                    ],
                    column_names=[
                        "job_id",
                        "ts",
                        "seq",
                        "directive",
                        "payload",
                        "applied_at",
                    ],
                )
            except Exception as exc:
                self._maybe_warn(exc)

        return out

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Stop the background flush thread and drain remaining rows."""
        self._stop.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=2)
        self._flush_pending(final=True)

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_payload(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8", errors="replace")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=_FLUSH_INTERVAL_S)
            self._flush_pending()
        self._flush_pending(final=True)

    def _flush_pending(self, *, final: bool = False) -> None:
        log_rows: list[list[Any]] = []
        while len(log_rows) < _BATCH_LIMIT:
            try:
                jid, ts, level, phase, msg = self._log_q.get_nowait()
            except queue.Empty:
                break
            log_rows.append([jid, ts, level, phase, msg])
        if log_rows:
            try:
                self._client.insert(
                    f"{self._db}.bench_logs",
                    log_rows,
                    column_names=["job_id", "ts", "level", "phase", "msg"],
                )
            except Exception as exc:
                self._maybe_warn(exc)

        m_rows: list[list[Any]] = []
        while len(m_rows) < _BATCH_LIMIT:
            try:
                jid, ts, name, val, labels = self._metric_q.get_nowait()
            except queue.Empty:
                break
            m_rows.append([jid, ts, name, val, labels])
        if m_rows:
            try:
                self._client.insert(
                    f"{self._db}.bench_metrics",
                    m_rows,
                    column_names=["job_id", "ts", "name", "value", "labels"],
                )
            except Exception as exc:
                self._maybe_warn(exc)

        if final and (not self._log_q.empty() or not self._metric_q.empty()):
            sys.stderr.write(
                f"[ch-sink] WARNING: flush_pending(final=True) "
                f"left {self._log_q.qsize()} log(s) and "
                f"{self._metric_q.qsize()} metric(s) un-flushed\n"
            )

    def _maybe_warn(self, exc: Exception) -> None:
        if not self._warned_unreachable:
            self._warned_unreachable = True
            sys.stderr.write(
                f"[ch-sink] WARNING: ClickHouse operation failed "
                f"(further failures suppressed): {exc}\n"
            )


# ── Module-level singleton ────────────────────────────────────────────────────


_SINK: ClickHouseSink | None = None
_SINK_LOCK = threading.Lock()


def get_sink() -> ClickHouseSink:
    """
    Return the process-wide :class:`ClickHouseSink` (lazy-init).

    Safe to call from any thread / module. Returns a no-op sink when
    ClickHouse is not configured.
    """
    global _SINK
    if _SINK is None:
        with _SINK_LOCK:
            if _SINK is None:
                _SINK = ClickHouseSink()
    return _SINK


def reset_sink_for_testing() -> None:
    """Test-only: stop the current sink and force re-init on next ``get_sink()``."""
    global _SINK
    with _SINK_LOCK:
        if _SINK is not None:
            try:
                _SINK.stop()
            except Exception:
                pass
        _SINK = None
