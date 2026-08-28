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


def test_repr_is_repl_friendly(catalog: Any) -> None:
    r = repr(catalog)
    assert "example" in r
    assert MAIN in r  # schema list should be visible without calling .schemas() explicitly

    table = catalog.table("data", "filter_echo_table")
    table_repr = repr(table)
    assert "filter_echo_table" in table_repr
    assert "pushed_filters" in table_repr  # a column name, proving the schema was actually included


def test_metadata_client_and_exchange_client_are_public_and_distinct(catalog: Any) -> None:
    """Regression: the SAFE client (per-thread, for scans/UDF calls) must not read as the private/discouraged one."""
    assert not hasattr(catalog, "_exchange_client"), "exchange_client() should be the only name now"
    assert catalog.metadata_client is not None
    assert catalog.exchange_client() is not None
    # Same thread -> same cached exchange client.
    assert catalog.exchange_client() is catalog.exchange_client()


def test_methods_raise_after_detach(worker_location: str) -> None:
    """Regression: calling a catalog/table method after detach() must raise, not silently spawn a new connection."""
    import vgi_acero as va

    catalog = va.attach(worker_location, name="example")
    catalog.detach()

    try:
        catalog.schemas()
        raise AssertionError("expected VgiAceroError after detach()")
    except va.VgiAceroError:
        pass

    try:
        _ = catalog.table("data", "filter_echo_table").arrow_schema
        raise AssertionError("expected VgiAceroError after detach()")
    except va.VgiAceroError:
        pass


def test_error_message_has_no_doubled_exception_type_prefix(worker_location: str) -> None:
    """Regression: a real remote error's message was observed doubled, e.g. 'ValueError: ValueError: ...'."""
    import vgi_acero as va

    try:
        va.attach(worker_location, name="totally_bogus_catalog_name_xyz")
        raise AssertionError("expected VgiAceroError for an unknown catalog name")
    except va.VgiAceroError as e:
        message = str(e)
        assert "ValueError: ValueError:" not in message
