# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""`vgi_topn_scan()` — adaptive Top-N re-querying.

See `_topn.py`'s module docstring for why this is adaptive re-querying, not
genuine intra-stream dynamic filtering (pyarrow.acero's Python API has no
hook for the latter).
"""

from __future__ import annotations

from typing import Any

import pyarrow as pa
from vgi.arguments import Arguments

from vgi_acero._topn import vgi_topn_scan

MAIN = "main"


def test_topn_scan_matches_full_scan_top_n(client: Any) -> None:
    full = list(
        client.table_function(
            function_name="filter_echo", schema_name=MAIN, arguments=Arguments(positional=(pa.scalar(1000),))
        )
    )
    full_table = pa.Table.from_batches(full)
    expected = sorted(full_table.column("n").to_pylist())[:10]

    result = vgi_topn_scan(
        client,
        schema_name=MAIN,
        function_name="filter_echo",
        order_by="n",
        limit=10,
        initial_bound=5,  # deliberately too low -- forces at least one widen round
        ascending=True,
        arguments=Arguments(positional=(pa.scalar(1000),)),
    )
    assert result.column("n").to_pylist() == expected


def test_topn_scan_descending(client: Any) -> None:
    result = vgi_topn_scan(
        client,
        schema_name=MAIN,
        function_name="filter_echo",
        order_by="n",
        limit=5,
        initial_bound=990,
        ascending=False,
        arguments=Arguments(positional=(pa.scalar(1000),)),
    )
    assert result.column("n").to_pylist() == [999, 998, 997, 996, 995]


def test_topn_scan_handles_source_smaller_than_limit(client: Any) -> None:
    result = vgi_topn_scan(
        client,
        schema_name=MAIN,
        function_name="filter_echo",
        order_by="n",
        limit=100,
        initial_bound=2,
        ascending=True,
        arguments=Arguments(positional=(pa.scalar(5),)),
    )
    assert result.num_rows == 5
