-- Bootstrap: create the bench database so schema.sql can reference bench.* tables.
-- Executed by Docker entrypoint BEFORE schema.sql (01 < 02 sort order).
CREATE DATABASE IF NOT EXISTS bench;
