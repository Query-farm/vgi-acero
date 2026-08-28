# vgi-acero

`pyarrow.acero` connector for [VGI](https://github.com/Query-farm/vgi-python) (Vector Gateway
Interface) — scan VGI catalog tables and call VGI functions from Acero, Apache Arrow's own
streaming execution engine, **with no DuckDB in the loop**.

```python
import pyarrow.acero as ac
import pyarrow.dataset as ds
import vgi_acero as va

with va.attach("vgi-fixture-worker", name="example") as catalog:
    table = catalog.table("main", "filter_echo_table")
    scan = table.scan(columns=["n", "s"], filter=ds.field("n") >= 8)

    plan = ac.Declaration.from_sequence(
        [
            scan,
            ac.Declaration("aggregate", ac.AggregateNodeOptions([("n", "count", None, "n_count")])),
        ]
    )
    print(plan.to_table())
```

## What this is

An **adapter**, not a new VGI protocol implementation. It wraps vgi-python's existing
pure-Python, Arrow-native `vgi.client.Client` — the same wire-protocol code the DuckDB
extension speaks, independent of DuckDB — and exposes it as `pyarrow.acero.Declaration`
sources, so any Acero-based pipeline can read from (and, for scalar functions, call into) a
VGI worker directly.

Sibling repos in this family: `vgi-polars` (client, Polars — the closest analog to this
package), `vgi-datafusion` (client, Apache DataFusion/Rust), `vgi-sqlite` (client, SQLite),
`vgi-spark` (client, Spark). `vgi-python`/`vgi-rust`/`vgi-go`/`vgi-java`/`vgi-typescript`/
`vgi-csharp` are worker SDKs — the opposite role.

## What maps, and what does not

| VGI capability | Acero seam | Status |
|---|---|:-:|
| Table function (bare or catalog table) | `ac.Declaration("scan", ac.ScanNodeOptions(...))` | ✅ |
| Projection pushdown | `columns=` on `.scan()` | ✅ |
| Filter pushdown | `filter=` on `.scan()`, translated from a `pc.Expression` | ✅ (bounded — see `_filter_translate.py`) |
| Struct/nested-field filter pushdown | same translator | ✅ |
| Semi-join key pushdown | `vgi_semi_join_scan()` + `ac.HashJoinNodeOptions` | ✅ |
| Split-planned parallel scan | `vgi_scan_splits()` + `ac.Declaration("union", ...)` | ✅ |
| Multi-branch tables | `VgiAceroTable.scan_all_branches()` + `union` | ✅ |
| Native scan-function delegation (parquet passthrough) | `ac.ScanNodeOptions(ds.dataset(...))`, bypassing VGI entirely | ✅ (parquet confirmed) |
| Scalar function | `pyarrow.compute.register_scalar_function` bridge | ✅ |
| Dynamic (intra-stream) filter pushdown | — | ❌ no hook in `pyarrow.acero`'s Python API for a downstream node to feed a bound back to an upstream source mid-execution; see `_topn.py`'s adaptive-requery emulation instead |
| Catalog/schema/function discovery | `VgiAceroCatalog.schemas()`/`.tables()`/... | ✅ |

## Build / test

Needs a sibling `vgi-python` checkout (`../vgi-python` relative to this repo) — see
`pyproject.toml`'s `[tool.uv.sources]`.

```bash
uv sync
uv run pytest
uv run ruff check --fix . && uv run ruff format .
uv run mypy src/
uvx pydoclint --config pyproject.toml src/
```
