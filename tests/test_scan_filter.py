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


def test_required_filters_enforced_on_the_ordinary_vgi_path(catalog: Any) -> None:
    """Regression: required_filters was only enforced for natively-delegated scans, never the ordinary VGI path."""
    from vgi_acero.errors import VgiAceroError

    table_handle = catalog.table("data", "rff_simple")
    assert table_handle.required_filters() == [["a"]]

    try:
        table_handle.scan()
        raise AssertionError("expected VgiAceroError for an unfiltered required_filters scan")
    except VgiAceroError:
        pass

    # A filter (even one the table can't actually push down here) satisfies
    # the "did you even try" gate.
    table_handle.scan(filter=ds.field("a") == 1).to_table()

    # The explicit escape hatch always works too.
    table_handle.scan(acknowledge_required_filters=True).to_table()


def test_scan_rejects_an_unknown_column_name(catalog: Any) -> None:
    """Regression: pyarrow.Schema.get_field_index() returns -1 (not KeyError) for an unknown name.

    Before this was validated, an unrecognized column silently resolved via
    Python negative indexing to the LAST real column instead of raising --
    confirmed live: a typo'd column name returned a real, wrong-column,
    non-empty table with no exception at all.
    """
    from vgi_acero.errors import VgiAceroError

    table_handle = catalog.table("data", "filter_echo_table")
    try:
        table_handle.scan(columns=["this_column_does_not_exist"]).to_table()
        raise AssertionError("expected VgiAceroError for an unknown column name")
    except VgiAceroError as e:
        assert "this_column_does_not_exist" in str(e)


def test_scan_honors_columns_locally_even_without_projection_pushdown(catalog: Any) -> None:
    """columns= must always be honored (a local ProjectNodeOptions), same posture filter= already has."""
    import dataclasses

    table_handle = catalog.table("data", "filter_echo_table")
    function_info = table_handle._function_info_get()
    assert function_info is not None
    no_projection_pushdown = dataclasses.replace(function_info, projection_pushdown=False)
    table_handle._function_info_get = lambda: no_projection_pushdown  # type: ignore[method-assign]
    result = table_handle.scan(columns=["n"]).to_table()
    assert result.schema.names == ["n"]


def _translate(client: Any, expr: Any) -> bytes | None:
    """Resolve `filter_echo`'s schema via `Client.bind()` and translate `expr` against it."""
    from vgi_acero._filter_translate import translate_predicate

    schema = client.bind(
        function_name="filter_echo", schema_name=MAIN, arguments=Arguments(positional=(pa.scalar(10),))
    ).output_schema
    return translate_predicate(expr, schema)
