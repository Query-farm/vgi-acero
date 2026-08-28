# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""Native scan-function delegation — no worker involved.

`_native_scan`'s handlers take a `ScanFunctionResult` and return a
`Declaration` that reads a real local file directly via `pyarrow.dataset`,
bypassing VGI entirely — this is exercised as a pure unit test against a
hand-built `ScanFunctionResult` and a real local parquet file, not against
`vgi-fixture-worker` (which has no parquet-delegating fixture).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from vgi.catalog.catalog_interface import ScanFunctionResult

from vgi_acero._native_scan import NATIVE_SCAN_HANDLERS


def test_read_parquet_delegation(tmp_path: Path) -> None:
    table = pa.table({"n": pa.array([1, 2, 3], type=pa.int64())})
    path = tmp_path / "data.parquet"
    pq.write_table(table, path)

    scan_fn = ScanFunctionResult(
        function_name="read_parquet",
        positional_arguments=[pa.scalar(str(path))],
        named_arguments={},
    )
    handler = NATIVE_SCAN_HANDLERS["read_parquet"]
    decl = handler(scan_fn, schema_name="main", table_name="t")
    result = decl.to_table()
    assert sorted(result.column("n").to_pylist()) == [1, 2, 3]


def test_read_parquet_delegation_rejects_unknown_named_arg(tmp_path: Path) -> None:
    from vgi_acero.errors import VgiAceroError

    scan_fn = ScanFunctionResult(
        function_name="read_parquet",
        positional_arguments=[pa.scalar(str(tmp_path / "x.parquet"))],
        named_arguments={"totally_unknown_option": pa.scalar(True)},
    )
    handler = NATIVE_SCAN_HANDLERS["read_parquet"]
    try:
        handler(scan_fn, schema_name="main", table_name="t")
        raise AssertionError("expected VgiAceroError")
    except VgiAceroError:
        pass


def test_read_parquet_delegation_requires_a_path(tmp_path: Path) -> None:
    from vgi_acero.errors import VgiAceroError

    scan_fn = ScanFunctionResult(function_name="read_parquet", positional_arguments=[], named_arguments={})
    handler = NATIVE_SCAN_HANDLERS["read_parquet"]
    try:
        handler(scan_fn, schema_name="main", table_name="t")
        raise AssertionError("expected VgiAceroError")
    except VgiAceroError:
        pass
