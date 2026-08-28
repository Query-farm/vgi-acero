# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""Scalar-function -> `pyarrow.compute` UDF bridge, usable inside an Acero expression.

Against `main.multiply(value, factor)` (`value` is an array param, `factor`
is a const/bind-time param) — the same reference fixture vgi-polars' own
`_scalar.py` documents.
"""

from __future__ import annotations

import threading
from typing import Any

import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.dataset as ds

MAIN = "main"


def test_scalar_function_inside_project_node(catalog: Any) -> None:
    multiply = catalog.scalar_function(MAIN, "multiply")
    table = pa.table({"value": pa.array([1, 2, 3], type=pa.int64())})
    source = ac.Declaration("table_source", ac.TableSourceNodeOptions(table))
    plan = ac.Declaration(
        "project",
        ac.ProjectNodeOptions([multiply(ds.field("value"), 10)], names=["doubled"]),
        inputs=[source],
    )
    result = plan.to_table()
    assert result.column("doubled").to_pylist() == [10, 20, 30]


def test_scalar_function_inside_filter_node(catalog: Any) -> None:
    multiply = catalog.scalar_function(MAIN, "multiply")
    table = pa.table({"value": pa.array([1, 2, 3, 4], type=pa.int64())})
    source = ac.Declaration("table_source", ac.TableSourceNodeOptions(table))
    plan = ac.Declaration(
        "filter",
        ac.FilterNodeOptions(multiply(ds.field("value"), 2) > 4),
        inputs=[source],
    )
    result = plan.to_table()
    assert result.column("value").to_pylist() == [3, 4]


def test_scalar_function_concurrent_calls_are_safe(catalog: Any) -> None:
    """Each thread must route through its own per-thread Client -- see catalog.py's thread-safety note."""
    multiply = catalog.scalar_function(MAIN, "multiply")
    errors: list[BaseException] = []
    results: list[list[int]] = [[] for _ in range(8)]

    def worker(i: int) -> None:
        try:
            table = pa.table({"value": pa.array([i, i + 1], type=pa.int64())})
            source = ac.Declaration("table_source", ac.TableSourceNodeOptions(table))
            plan = ac.Declaration(
                "project", ac.ProjectNodeOptions([multiply(ds.field("value"), 3)], names=["r"]), inputs=[source]
            )
            results[i] = plan.to_table().column("r").to_pylist()
        except BaseException as e:  # noqa: BLE001 - captured for the assertion below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    for i in range(8):
        assert results[i] == [i * 3, (i + 1) * 3]
