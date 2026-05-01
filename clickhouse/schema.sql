-- Aiven Bench ClickHouse Schema
-- ============================================================
-- Owned by the orchestrator.  The loader only INSERTs rows and
-- reads bench_control.  Run this DDL once against the Aiven for
-- ClickHouse service before deploying the loader.
--
-- Best-practice notes applied:
--   schema-pk-cardinality-order  — ORDER BY uses low→high cardinality.
--   schema-types-lowcardinality  — LowCardinality for status/bench_type/level/phase.
--   schema-types-avoid-nullable  — Nullable only where NULL has semantic meaning
--                                   (finished_at = NULL → still running).
--   schema-partition-low-cardinality — PARTITION BY toStartOfMonth keeps
--                                   partition count bounded (~12/year).
--   insert-async-small-batches   — loader uses async_insert=1 (see end of file).
-- ============================================================

-- ── bench_runs ────────────────────────────────────────────────────────────────
--
-- One row per benchmark job (and one upserted row when it finishes).
-- ReplacingMergeTree deduplicates by (job_id) keeping the latest version
-- (MAX started_at used as tie-break; use FINAL for exact reads).
--
-- ORDER BY: bench_type (low cardinality, ~7 values) → started_at (date,
--           coarser than DateTime) → job_id (high cardinality, used for
--           point lookups).  Satisfies schema-pk-cardinality-order.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bench.bench_runs
(
    job_id        String,
    started_at    DateTime64(3, 'UTC'),
    finished_at   Nullable(DateTime64(3, 'UTC')),   -- NULL = still running (semantic)
    status        LowCardinality(String),            -- running | ok | failed | cancelled
    bench_type    LowCardinality(String),            -- index | search | recall | hybrid | stress | recover | plan_change | corpus_build
    label         String,                            -- service_label/bench_type e.g. "os-2.19-local/search"
    spec          String DEFAULT '{}',               -- KnnSpec JSON blob
    summary       String DEFAULT '{}'                -- result summary JSON (report_path, metrics snapshot)
)
ENGINE = ReplacingMergeTree(started_at)
PARTITION BY toStartOfMonth(started_at)
ORDER BY (bench_type, toDate(started_at), job_id)
SETTINGS index_granularity = 8192;


-- ── bench_logs ────────────────────────────────────────────────────────────────
--
-- Structured stdout/stderr lines captured from every job.
-- MergeTree (append-only); no deduplication needed.
--
-- ORDER BY: job_id first — the most common filter.  Then ts for time-ordered
--           reads of a single job's log.  Cardinality: job_id is high but
--           most queries filter to a single job_id so the primary index is
--           effective.  Per schema-pk-cardinality-order: level + phase
--           (both low-cardinality) are in the ORDER BY prefix because they
--           appear in WHERE filters in monitoring dashboards.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bench.bench_logs
(
    job_id   String,
    ts       DateTime64(3, 'UTC'),
    level    LowCardinality(String),   -- info | warn | error
    phase    LowCardinality(String),   -- stdout | corpus | report | ...
    msg      String
)
ENGINE = MergeTree()
PARTITION BY toStartOfMonth(ts)
ORDER BY (job_id, level, ts)
SETTINGS index_granularity = 8192;


-- ── bench_metrics ─────────────────────────────────────────────────────────────
--
-- Time-series metric samples emitted during a job (latency p50/p99, throughput,
-- recall@10, etc.).  MergeTree (append-only).
--
-- `labels` is a Map(String, String) storing auxiliary key=value pairs
-- (e.g. {"batch_size":"50","clients":"8"}).  Avoid JSON strings for
-- structured map data — schema-json-when-to-use.
--
-- ORDER BY: job_id + name (both frequently filtered together) + ts.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bench.bench_metrics
(
    job_id   String,
    ts       DateTime64(3, 'UTC'),
    name     LowCardinality(String),   -- latency_p50_ms | throughput_ops | recall_at_10 | ...
    value    Float64,
    labels   Map(LowCardinality(String), String) DEFAULT map()
)
ENGINE = MergeTree()
PARTITION BY toStartOfMonth(ts)
ORDER BY (job_id, name, ts)
SETTINGS index_granularity = 8192;


-- ── bench_control ─────────────────────────────────────────────────────────────
--
-- Control-plane directives written by the orchestrator, read and applied by
-- the loader.  The loader marks applied rows by inserting a duplicate row
-- with applied_at != epoch; ReplacingMergeTree(applied_at) keeps the latest.
--
-- Unapplied directives have applied_at = epoch (toDateTime64(0, 3, 'UTC')).
-- Applied directives have applied_at = the actual application timestamp.
-- Use: WHERE applied_at = toDateTime64(0, 3, 'UTC') to find pending work.
--
-- Note: Nullable is NOT valid as a ReplacingMergeTree version column in
-- ClickHouse >= 22; epoch sentinel is the standard alternative.
--
-- Supported directives (directive column):
--   cancel   — stop the running job immediately
--   throttle — {"pool":"search|index","clients":N}
--   pause    — {"pool":"search|index"}
--   resume   — {"pool":"search|index"}
--
-- ORDER BY: job_id + seq (natural deduplication key).
-- ─────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS bench.bench_control
(
    job_id     String,
    ts         DateTime64(3, 'UTC'),
    seq        UInt32,
    directive  LowCardinality(String),                                -- cancel | throttle | pause | resume
    payload    String DEFAULT '{}',                                   -- JSON payload for the directive
    applied_at DateTime64(3, 'UTC') DEFAULT toDateTime64(0, 3, 'UTC') -- epoch = not yet applied
)
ENGINE = ReplacingMergeTree(applied_at)
PARTITION BY toStartOfMonth(ts)
ORDER BY (job_id, seq)
SETTINGS index_granularity = 8192;


-- ── Async insert settings ─────────────────────────────────────────────────────
--
-- The loader flushes bench_logs and bench_metrics every 500 ms in small
-- batches (~1000 rows).  Enable async inserts for the loader's DB user so
-- ClickHouse buffers these server-side (insert-async-small-batches rule).
--
-- Run once against the target ClickHouse service:
--
--   ALTER USER avnadmin SETTINGS
--       async_insert = 1,
--       wait_for_async_insert = 1,
--       async_insert_max_data_size = 10000000,
--       async_insert_busy_timeout_ms = 1000;
--
-- bench_runs uses direct inserts (one row per job) and does NOT need async.
-- bench_control uses direct inserts for low-volume directive writes.
-- ─────────────────────────────────────────────────────────────────────────────


-- ── Convenience views ─────────────────────────────────────────────────────────

-- Latest status per job (FINAL collapses ReplacingMergeTree duplicates).
CREATE VIEW IF NOT EXISTS bench.v_latest_runs AS
SELECT *
FROM bench.bench_runs FINAL;

-- Rolling summary: last 7 days of completed runs with key metrics.
-- Uses a subquery for FINAL so the table alias works in ClickHouse syntax.
CREATE VIEW IF NOT EXISTS bench.v_recent_summary AS
SELECT
    r.job_id,
    r.bench_type,
    r.label,
    r.started_at,
    r.finished_at,
    r.status,
    JSONExtractString(r.summary, 'report_path') AS report_path,
    avgIf(m.value, m.name = 'latency_p50_ms')  AS p50_ms,
    avgIf(m.value, m.name = 'latency_p99_ms')  AS p99_ms,
    avgIf(m.value, m.name = 'recall_at_10')    AS recall_at_10,
    avgIf(m.value, m.name = 'throughput_ops')  AS throughput_ops
FROM (SELECT * FROM bench.bench_runs FINAL) AS r
LEFT JOIN bench.bench_metrics AS m ON r.job_id = m.job_id
WHERE r.started_at >= now() - INTERVAL 7 DAY
GROUP BY r.job_id, r.bench_type, r.label, r.started_at, r.finished_at, r.status, report_path
ORDER BY r.started_at DESC;
