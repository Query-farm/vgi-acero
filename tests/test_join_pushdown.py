# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""`vgi_semi_join_scan()` — build-side join-key pushdown onto a probe-side VGI scan.

Uses vgi-python's `Client.bind()` (zero-execution) to resolve `filter_echo`'s
schema, then pushes a small build-side key set down as a `join_keys` filter,
asserting on both the final row set AND the worker's own `pushed_filters`
echo text to prove the predicate was applied server-side.
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pyarrow.acero as ac
from vgi.arguments import Arguments

from vgi_acero import vgi_semi_join_scan

MAIN = "main"


def test_semi_join_scan_filters_to_build_side_keys(client: Any) -> None:
    build_keys = pa.array([3, 7], type=pa.int64())
    decl = vgi_semi_join_scan(
        client,
        schema_name=MAIN,
        function_name="filter_echo",
        join_column="n",
        build_side_keys=build_keys,
        arguments=Arguments(positional=(pa.scalar(10),)),
    )
    table = decl.to_table()
    assert sorted(table.column("n").to_pylist()) == [3, 7]
    assert all("n IN" in v for v in table.column("pushed_filters").to_pylist())


def test_semi_join_scan_composes_with_hash_join(client: Any) -> None:
    """A real ac.HashJoinNodeOptions on top of the pushed-down probe scan."""
    build_table = pa.table({"n": pa.array([2, 5, 9], type=pa.int64()), "label": ["a", "b", "c"]})
    build_decl = ac.Declaration("table_source", ac.TableSourceNodeOptions(build_table))

    probe_decl = vgi_semi_join_scan(
        client,
        schema_name=MAIN,
        function_name="filter_echo",
        join_column="n",
        build_side_keys=build_table.column("n"),
        arguments=Arguments(positional=(pa.scalar(10),)),
    )

    plan = ac.Declaration(
        "hashjoin",
        ac.HashJoinNodeOptions("inner", left_keys=["n"], right_keys=["n"]),
        inputs=[build_decl, probe_decl],
    )
    table = plan.to_table()
    # Both sides carry a join column named "n" in the output by default --
    # disambiguate on the build side's own "label" column instead.
    assert table.num_rows == 3
    assert sorted(table.column("label").to_pylist()) == ["a", "b", "c"]
