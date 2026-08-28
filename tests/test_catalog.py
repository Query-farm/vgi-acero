# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""`VgiAceroCatalog`/`VgiAceroTable` basics: attach/detach, discovery, cross-schema resolution."""

from __future__ import annotations

from typing import Any

MAIN = "main"


def test_attach_detach(worker_location: str) -> None:
    import vgi_acero as va

    catalog = va.attach(worker_location, name="example")
    assert catalog.name == "example"
    catalog.detach()
    catalog.detach()  # safe to call twice


def test_schemas_and_tables(catalog: Any) -> None:
    schemas = catalog.schemas()
    assert MAIN in schemas
    # filter_echo_table is registered under schema `data`, not `main` -- its
    # scan FUNCTION lives in `main` (see test_cross_schema_scan_function_resolution).
    assert "data" in schemas
    tables = catalog.tables("data")
    assert "filter_echo_table" in tables


def test_table_functions_and_scalar_functions_listing(catalog: Any) -> None:
    table_fns = catalog.table_functions(MAIN)
    assert "filter_echo" in table_fns
    scalar_fns = catalog.scalar_functions(MAIN)
    assert "multiply" in scalar_fns


def test_arrow_schema_is_zero_execution(catalog: Any) -> None:
    table = catalog.table("data", "filter_echo_table")
    schema = table.arrow_schema
    assert schema.names == ["n", "s", "pushed_filters"]


def test_cross_schema_scan_function_resolution(catalog: Any) -> None:
    """example.data.filter_echo_table resolves its scan function to schema `main`, not `data`."""
    table = catalog.table("data", "filter_echo_table")
    assert table.scan_function_schema() == MAIN
    result = table.scan().to_table()
    assert result.num_rows == 100
