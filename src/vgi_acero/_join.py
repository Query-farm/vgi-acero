# Copyright 2026 Query Farm LLC - https://query.farm

"""Semi-join key pushdown — `Client.table_function(join_keys=...)` for an Acero probe-side scan.

Given an Acero plan that joins a (typically small) build-side result against
a VGI-backed probe-side scan, this pushes the build side's join column values
down as a `join_keys` filter on the probe scan — a real semi-join
optimization mirroring what DuckDB's own join pushdown into VGI already does
server-side, and now reachable from `Client` (`Client.table_function(join_keys=...)`)
since it was previously carried on the wire protocol
(`InitRequest.join_keys`/`TableFunctionPlanRequest.join_keys`) with no public
way to set it — see vgi-python's `Client.bind()`/`join_keys` addition this
package's floor version requires.

The caller is still responsible for the actual `ac.HashJoinNodeOptions` node
on top — this only narrows what the probe side has to produce, exactly the
way pushing `pushdown_filters` narrows an ordinary scan; it never replaces
the join itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.acero as ac

from vgi_acero._filter_translate import build_join_keys_filter_bytes
from vgi_acero._scan import make_vgi_scan_declaration

if TYPE_CHECKING:
    from vgi.arguments import Arguments
    from vgi.client.client import Client

__all__ = ["vgi_semi_join_scan"]


def vgi_semi_join_scan(
    client: Client,
    *,
    schema_name: str,
    function_name: str,
    join_column: str,
    build_side_keys: pa.Array[Any] | pa.ChunkedArray[Any],
    arguments: Arguments | None = None,
    projection_ids: list[int] | None = None,
    settings: dict[str, Any] | None = None,
) -> ac.Declaration:
    """Scan `function_name`, pushing `build_side_keys` down as a `join_keys` filter on `join_column`.

    Two things travel to the worker together, neither sufficient alone: the
    actual key values (`Client.table_function`'s `join_keys=` argument) and a
    `pushdown_filters` spec of type `join_keys` telling the worker to look
    them up (`_filter_translate.build_join_keys_filter_bytes`) — resolving
    `join_column`'s `column_index` for that spec needs the function's output
    schema, fetched via `Client.bind()` (a zero-execution RPC; see
    vgi-python's `Client.bind()` addition) rather than the old peek-a-batch
    workaround `_scan.vgi_scan` still has to use for schema discovery.

    Args:
        client: A started `vgi.client.Client` — see `_scan.vgi_scan`'s
            thread-safety note; use a dedicated client per thread if this is
            called concurrently for multiple probe-side scans.
        schema_name: The catalog schema declaring `function_name`.
        function_name: The probe-side table function to scan.
        join_column: The probe-side column name the join is on. Must match a
            column name the worker's `PushdownFilters.get_join_keys_column`
            can resolve — i.e. the same name as `build_side_keys`' own column.
        build_side_keys: The distinct join-key values from the (already
            materialized) build side, e.g. `build_table.column("id")`.
        arguments: Positional/named arguments for the probe function's bind.
        projection_ids: Optional column indices to project.
        settings: Optional dictionary of settings/pragmas.

    Returns:
        An `ac.Declaration` — see `_scan.make_vgi_scan_declaration`'s "must be
        executed" caveat, which applies here too. The caller composes this
        with the build side via a real `ac.Declaration("hashjoin",
        ac.HashJoinNodeOptions(...), inputs=[build_decl, probe_decl])`.

    """
    bind_response = client.bind(function_name=function_name, schema_name=schema_name, arguments=arguments)
    column_index = bind_response.output_schema.get_field_index(join_column)
    pushdown_filters = build_join_keys_filter_bytes(
        column_name=join_column, column_index=column_index, keys_column=join_column
    )

    # A Table column comes back as a ChunkedArray (e.g. build_table.column(...)),
    # which pa.record_batch()'s single-array-per-field constructor rejects.
    flat_keys = build_side_keys.combine_chunks() if isinstance(build_side_keys, pa.ChunkedArray) else build_side_keys
    join_keys = [pa.record_batch({join_column: flat_keys})]
    gen = client.table_function(
        function_name=function_name,
        schema_name=schema_name,
        arguments=arguments,
        projection_ids=projection_ids,
        pushdown_filters=pushdown_filters,
        join_keys=join_keys,
        settings=settings,
    )
    return make_vgi_scan_declaration(gen)
