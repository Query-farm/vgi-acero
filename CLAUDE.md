# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository.

## What this is

`vgi-acero` is a **client SDK** — an adapter, not a new VGI protocol
implementation. It wraps vgi-python's existing pure-Python, Arrow-native
`vgi.client.Client` (+ `CatalogClientMixin`) — the same wire-protocol code the
DuckDB extension speaks, independent of DuckDB — and exposes it as
`pyarrow.acero.Declaration` source nodes, scalar UDFs, and join/split helpers,
so any Acero-based pipeline can consume a VGI worker directly.

Sibling repos in this family: `vgi-polars` (client, Polars — the closest
analog to this package, mirrored throughout), `vgi-datafusion` (client,
DataFusion/Rust), `vgi-sqlite` (client, SQLite), `vgi-spark` (client, Spark).
`vgi-python`/`vgi-rust`/`vgi-go`/`vgi-java`/`vgi-typescript`/`vgi-csharp` are
worker SDKs — the opposite role.

Package import name is `vgi_acero` (distribution `vgi-acero`) — `vgi-python`
already owns the top-level `vgi` package.

## Build / Test

Needs a sibling `vgi-python` checkout at `../vgi-python` (see
`[tool.uv.sources]` in `pyproject.toml`).

```bash
uv sync                                   # local dev: editable path to ../vgi-python
uv run pytest -v                          # needs vgi-fixture-worker findable — see below
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/
uvx pydoclint --config pyproject.toml src/   # isolated — see pyproject.toml's comment on why
```

Tests drive the real `vgi-fixture-worker` subprocess (no mocking). By
default `tests/conftest.py` points at `~/Development/vgi-python/.venv/bin/vgi-fixture-worker`;
override with `VGI_TEST_WORKER=<command>` or `VGI_PYTHON=<path to vgi-python checkout>`.

## Architecture

- `catalog.py` — `VgiAceroCatalog` + `attach()`. One shared `Client` for
  catalog-metadata RPCs (`schemas`, `table_get`, `schema_contents`,
  `table_scan_function_get`, `bind`, ...); one **per-thread** `Client` (via
  `exchange_client()`) for exchange-mode RPCs (`table_function`,
  `scalar_function`). See its module docstring's "Thread safety" section —
  `vgi.client.Client` is not safe for concurrent exchange-mode use on one
  shared instance (confirmed by direct stress test), and Acero's own default
  multi-threaded execution (plus this package's own `vgi_scan_splits`'
  concurrent `union` pulling, and a registered scalar UDF being invocable from
  multiple worker threads) makes this at least as relevant here as it was for
  vgi-polars.
- `table.py` — `VgiAceroTable`: schema/scan-function/branch resolution
  (all cheap, scan-free unary catalog RPCs), native scan-function delegation
  (`_native_scan.py`), multi-branch union scanning.
- `_scan.py` — `make_vgi_scan_declaration()` (the one core mechanism every
  scan path builds on: wraps a VGI batch generator as an
  `ac.Declaration("scan", ...)`), `vgi_scan()` (bare function), and
  `vgi_scan_splits()` (real concurrent split fan-out via Acero's `union` exec
  node — confirmed empirically to pull concurrently, unlike vgi-polars'
  necessarily-sequential split chaining, since Polars' `register_io_source`
  has no concurrent-pull hook).
- `_filter_translate.py` — `pyarrow.compute.Expression` → VGI's
  `pushdown_filters` wire bytes. See its own module docstring for the full
  filter-vocabulary coverage and the `repr()`-parsing approach's known
  fragility.
- `_join.py` — `vgi_semi_join_scan()`: build-side join-key pushdown via
  `Client.table_function(join_keys=...)` (added to vgi-python alongside this
  package — see "vgi-python version floor" below).
- `_topn.py` — `vgi_topn_scan()`: adaptive Top-N re-querying, **not** genuine
  intra-stream dynamic filtering — `pyarrow.acero`'s Python API has no hook
  for a downstream node to feed an evolving bound back to an upstream source
  mid-execution. See its module docstring for the honest accounting of what
  this does and doesn't achieve relative to DuckDB's own TopN pushdown.
- `_scalar.py` — a VGI scalar function registered as a
  `pyarrow.compute.register_scalar_function` UDF, usable inside any Acero
  expression. Each call registers a fresh, uniquely-named function (no
  partial-application concept in `pyarrow.compute`'s registry) — see its
  module docstring.
- `_native_scan.py` — `read_parquet`/`read_csv` delegation straight to
  `ds.dataset(...)`, bypassing VGI entirely when a worker names a passthrough
  format reader instead of a VGI-hosted function.
- `_arguments.py`, `errors.py` — small shared plumbing, near-verbatim ports
  of vgi-polars' equivalents (these problems have nothing to do with the
  target engine).

## A `Declaration`'s lifecycle is NOT lazy the way DataFusion's is

This is the one structural difference from `vgi-datafusion`'s model worth
internalizing before touching `_scan.py`: Acero's Python API has no "logical
plan, not yet executed" phase for a custom source. `make_vgi_scan_declaration()`
calls `next(gen)` to learn the schema — **that is real execution**, not a lazy
peek. A `Declaration` this package returns must be part of a plan that
actually gets executed to completion, or the whole `Client` it came from must
be abandoned via `Client.stop(force=True)` (subprocess transport only) — never
discard a returned `Declaration` or reuse the same `Client` afterward if you
do. `VgiAceroTable.scan()`'s catalog-mode path avoids this for *schema
discovery* (via `Client.table_get()`, genuinely zero-execution) but the
*data* generator itself still carries the caveat once the scan actually runs.

## vgi-python version floor

`pyproject.toml` requires `vgi-python>=0.31.0`, which added two things this
package depends on directly (both closing gaps the original filter-pushdown
spike surfaced, of the same shape: existing wire-protocol capability with no
public `Client` entry point before this):

- `Client.bind()` — schema discovery without running `init()`/`process()`,
  used for the bare-function path (`vgi_scan()`, `_join.py`, `_topn.py`)
  wherever no catalog attach is available to ask instead.
- `Client.table_function(join_keys=..., ...)` / `table_in_out_function(join_keys=...)`
  — semi-join pushdown, used by `_join.vgi_semi_join_scan()`.

## Scope / non-goals (v1)

- Filter translation covers `constant`/`is_null`/`is_not_null`/`in`/`and`/`or`/
  `struct`/a bounded single-column `expression` fallback — **not** arbitrary
  multi-column expressions, and the `expression` fallback is gated on the
  resolved `FunctionInfo.supported_expression_filters` being non-empty for
  catalog-mode scans (best-effort/ungated on the bare-function path, which has
  no `FunctionInfo` to check).
- `vgi_topn_scan()` is adaptive re-querying, not intra-stream dynamic
  filtering (see `_topn.py`'s module docstring — a real `pyarrow.acero` API
  limitation, not a shrug).
- Catalog-table branches (`branch.source_table is not None`) and format
  branches (`branch.format_name is not None`) in a multi-branch table are not
  yet supported (`scan_all_branches()` raises `VgiAceroError` for either) —
  no established resolution path in this package yet, matching vgi-polars'
  own documented scope limit.
- Native scan delegation covers `read_parquet`/`read_csv` only — no
  `iceberg_scan` equivalent (`pyarrow.dataset` has no built-in Iceberg reader
  without an additional `pyiceberg` dependency this package doesn't carry).
