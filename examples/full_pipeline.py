# Copyright 2026 Query Farm LLC - https://query.farm

r"""End-to-end vgi-acero pipeline against a real vgi-fixture-worker — no DuckDB anywhere in this file.

Demonstrates, in one Acero plan:
  - a catalog-mode scan with filter + projection pushdown (`VgiAceroTable.scan`)
  - a split-planned, concurrently-pulled union scan (`vgi_scan_splits`)
  - a semi-join key pushdown into a real `ac.HashJoinNodeOptions` (`vgi_semi_join_scan`)
  - a VGI scalar function called inside an Acero `ProjectNodeOptions` expression
  - a final `ac.AggregateNodeOptions`

Needs the sibling vgi-python checkout's venv on PATH, or set VGI_TEST_WORKER
to any vgi-fixture-worker command. Run with:
    PATH="$HOME/Development/vgi-python/.venv/bin:$PATH" \
    uv run --project ~/Development/vgi-acero python examples/full_pipeline.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.dataset as ds
from vgi.arguments import Arguments
from vgi.client.client import Client

import vgi_acero as va
from vgi_acero._scan import vgi_scan_splits


def _default_worker() -> str:
    override = os.environ.get("VGI_TEST_WORKER")
    if override:
        return override
    return str(Path.home() / "Development" / "vgi-python" / ".venv" / "bin" / "vgi-fixture-worker")


WORKER = _default_worker()


def main() -> None:
    """Run the full demonstration pipeline described in the module docstring."""
    with va.attach(WORKER, name="example") as catalog:
        # 1. Catalog-mode scan with filter + projection pushdown.
        filter_table = catalog.table("data", "filter_echo_table")
        filtered = filter_table.scan(columns=["n", "s"], filter=ds.field("n") >= 90)
        print("catalog-mode filtered scan:")
        print(filtered.to_table())

        # 2. Split-planned, concurrently-pulled union scan.
        def client_factory() -> Client:
            c = Client(WORKER)
            c.start()
            return c

        split_result = vgi_scan_splits(
            client_factory,
            schema_name="main",
            function_name="split_sequence",
            arguments=Arguments(named={"n": pa.scalar(30), "splits": pa.scalar(4)}),
        )
        try:
            print("\nsplit-planned union scan:")
            print(ac.Declaration.from_sequence([split_result.declaration]).to_table())
        finally:
            split_result.close()  # stop every per-split Client vgi_scan_splits() opened

        # 3. Semi-join key pushdown, composed with a real HashJoinNodeOptions.
        build_table = pa.table({"n": pa.array([5, 10, 90], type=pa.int64())})
        build_decl = ac.Declaration("table_source", ac.TableSourceNodeOptions(build_table))
        probe_client = catalog._exchange_client()
        probe_decl = va.vgi_semi_join_scan(
            probe_client,
            schema_name="main",
            function_name="filter_echo",
            join_column="n",
            build_side_keys=build_table.column("n"),
            arguments=Arguments(positional=(pa.scalar(100),)),
        )
        joined = ac.Declaration(
            "hashjoin",
            ac.HashJoinNodeOptions("inner", left_keys=["n"], right_keys=["n"], right_output=["s"]),
            inputs=[build_decl, probe_decl],
        )
        print("\nsemi-join pushdown + real HashJoinNodeOptions:")
        print(joined.to_table())

        # 4. A VGI scalar function inside a ProjectNodeOptions expression, then aggregate.
        multiply = catalog.scalar_function("main", "multiply")
        small = pa.table({"value": pa.array([1, 2, 3, 4, 5], type=pa.int64())})
        source = ac.Declaration("table_source", ac.TableSourceNodeOptions(small))
        projected = ac.Declaration(
            "project", ac.ProjectNodeOptions([multiply(ds.field("value"), 10)], names=["scaled"]), inputs=[source]
        )
        aggregated = ac.Declaration(
            "aggregate", ac.AggregateNodeOptions([("scaled", "sum", None, "total")]), inputs=[projected]
        )
        print("\nscalar-function project + aggregate:")
        print(aggregated.to_table())


if __name__ == "__main__":
    main()
