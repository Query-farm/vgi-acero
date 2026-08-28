# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""`vgi_scan()` against a bare (non-catalog) VGI table function.

Mirrors vgi-python's own `tests/table/generator/test_filter_echo_function.py`
assertions, driven through an `ac.Declaration.to_table()` instead of a raw
generator.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.acero as ac
from vgi.arguments import Arguments

from vgi_acero import vgi_scan

MAIN = "main"


def test_scan_row_count_and_columns(client: Any) -> None:
    decl = vgi_scan(
        client, schema_name=MAIN, function_name="filter_echo", arguments=Arguments(positional=(pa.scalar(10),))
    )
    table = decl.to_table()
    assert table.num_rows == 10
    assert set(table.schema.names) >= {"n", "s", "pushed_filters"}
    assert sorted(table.column("n").to_pylist()) == list(range(10))


def test_scan_composes_with_further_acero_nodes(client: Any) -> None:
    """A vgi_scan() Declaration is a genuine Acero source node, not a one-shot wrapper."""
    decl = vgi_scan(
        client, schema_name=MAIN, function_name="filter_echo", arguments=Arguments(positional=(pa.scalar(20),))
    )
    plan = ac.Declaration.from_sequence(
        [
            decl,
            ac.Declaration("aggregate", ac.AggregateNodeOptions([("n", "count", None, "n_count")])),
        ]
    )
    table = plan.to_table()
    assert table.column("n_count")[0].as_py() == 20
