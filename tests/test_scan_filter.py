# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""End-to-end filter/projection pushdown correctness, bare and catalog-mode paths.

Asserts on the worker's own `pushed_filters` echo text (`filter_echo`) to
prove pushdown is REAL server-side application, not client-side
post-filtering — the technique the original spike used and this package's
translator tests reuse throughout.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.dataset as ds
from vgi.arguments import Arguments

from vgi_acero import vgi_scan

MAIN = "main"


def test_bare_scan_filter_pushdown(client: Any) -> None:
    decl = vgi_scan(
        client,
        schema_name=MAIN,
        function_name="filter_echo",
        arguments=Arguments(positional=(pa.scalar(10),)),
        pushdown_filters=_translate(client, (ds.field("n") >= 3) & (ds.field("n") <= 6)),
    )
    table = decl.to_table()
    assert sorted(table.column("n").to_pylist()) == [3, 4, 5, 6]
    assert all("n >= 3" in v and "n <= 6" in v for v in table.column("pushed_filters").to_pylist())


def test_bare_scan_in_filter_pushdown(client: Any) -> None:
    decl = vgi_scan(
        client,
        schema_name=MAIN,
        function_name="filter_echo",
        arguments=Arguments(positional=(pa.scalar(10),)),
        pushdown_filters=_translate(client, ds.field("n").isin([2, 5, 9])),
    )
    table = decl.to_table()
    assert sorted(table.column("n").to_pylist()) == [2, 5, 9]


def test_bare_scan_projection_pushdown(client: Any) -> None:
    decl = vgi_scan(
        client,
        schema_name=MAIN,
        function_name="filter_echo",
        arguments=Arguments(positional=(pa.scalar(10),)),
        projection_ids=[0, 2],  # n, pushed_filters -- skip s
    )
    table = decl.to_table()
    assert "s" not in table.schema.names
    assert {"n", "pushed_filters"}.issubset(table.schema.names)


def test_catalog_table_scan_filter_pushdown(catalog: Any) -> None:
    # example.data.filter_echo_table is a fixed 100-row dataset (n in 0..99).
    table_handle = catalog.table("data", "filter_echo_table")
    decl = table_handle.scan(filter=ds.field("n") >= 95)
    table = decl.to_table()
    assert sorted(table.column("n").to_pylist()) == [95, 96, 97, 98, 99]
    assert all("n >= 95" in v for v in table.column("pushed_filters").to_pylist())


def test_catalog_table_scan_expression_fallback_pushdown(catalog: Any) -> None:
    """filter_echo_table declares supported_expression_filters=["prefix","starts_with"] -- exercise it."""
    import pyarrow.compute as pc

    table_handle = catalog.table("data", "filter_echo_table")
    decl = table_handle.scan(filter=pc.starts_with(ds.field("s"), "row_1"))
    table = decl.to_table()
    # row_1, row_10..row_19 -- 11 rows total.
    assert sorted(table.column("n").to_pylist()) == [1, *range(10, 20)]


def test_catalog_table_scan_composes_with_local_acero_filter(catalog: Any) -> None:
    """VgiAceroTable.scan(filter=...) also applies the predicate locally — proving composition, not just pushdown."""
    table_handle = catalog.table("data", "filter_echo_table")
    decl = table_handle.scan(filter=ds.field("n") >= 3)
    plan = ac.Declaration.from_sequence([decl, ac.Declaration("filter", ac.FilterNodeOptions(ds.field("n") <= 6))])
    table = plan.to_table()
    assert sorted(table.column("n").to_pylist()) == [3, 4, 5, 6]


def _translate(client: Any, expr: Any) -> bytes | None:
    """Resolve `filter_echo`'s schema via `Client.bind()` and translate `expr` against it."""
    from vgi_acero._filter_translate import translate_predicate

    schema = client.bind(
        function_name="filter_echo", schema_name=MAIN, arguments=Arguments(positional=(pa.scalar(10),))
    ).output_schema
    return translate_predicate(expr, schema)
