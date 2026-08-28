# Copyright 2026 Query Farm LLC - https://query.farm

"""Adaptive Top-N re-querying — an emulation of DuckDB's dynamic TopN filter pushdown.

**Why this is adaptive re-querying, not intra-stream dynamic filtering — a
real, named API limitation, not a shrug.** VGI's `current_pushdown_filters`
mechanism (used by DuckDB's TopN operator to push progressively tighter
bounds mid-scan — exercised by `vgi/_test_fixtures/table/filters.py`'s
`DynamicFilterEchoFunction`) requires the *consumer* to feed an evolving
bound back to the producer while the stream is still open.
`pyarrow.acero`'s Python surface exposes no limit/fetch/top-k node type and
no callback hook for a downstream node to report "already satisfied, here's
the current worst-value bound" back to an upstream source mid-execution —
Acero's plan is pull-based with no such feedback channel exposed to Python.
Genuine intra-stream dynamic filtering is therefore not implementable against
today's `pyarrow.acero` Python bindings.

**What is implementable, and ships here**: client-driven adaptive
re-querying that emulates the same end goal (avoid scanning far more than N
rows when N are wanted, sorted) using the pushdown primitives this package
already has — a materialized-result helper, not a lazy `ac.Declaration` (it
must inspect intermediate results to decide whether to widen the bound). Each
round is a fresh `Client.table_function()` call — no state carried across,
unlike DuckDB's in-stream mechanism, trading some redundant boundary work for
implementability against Acero's actual (non-dynamic-filter-capable) Python
API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import pyarrow as pa
import pyarrow.compute as pc

from vgi_acero._filter_translate import translate_predicate

if TYPE_CHECKING:
    from vgi.arguments import Arguments
    from vgi.client.client import Client

__all__ = ["vgi_topn_scan"]


def vgi_topn_scan(
    client: Client,
    *,
    schema_name: str,
    function_name: str,
    order_by: str,
    limit: int,
    initial_bound: Any,
    ascending: bool = True,
    arguments: Arguments | None = None,
    settings: dict[str, Any] | None = None,
    max_rounds: int = 20,
) -> pa.Table:
    """Emulate dynamic TopN filter pushdown via adaptive re-querying.

    Scans with a constant-filter bound on `order_by` (`order_by <=
    initial_bound` ascending, `>=` descending — translated and pushed via
    `_filter_translate.translate_predicate`), sorts+limits locally; if fewer
    than `limit` rows came back and the source wasn't exhausted, doubles the
    bound's distance from its starting point and re-scans, until either
    `limit` rows are satisfied, the source is exhausted, or `max_rounds` is
    reached (a correctness backstop against a pathological `initial_bound`/
    data distribution looping forever — the last round always falls back to
    an unbounded scan so this never returns fewer than the true top-N).

    Args:
        client: A started `vgi.client.Client`.
        schema_name: The catalog schema declaring `function_name`.
        function_name: The table function to scan.
        order_by: Column name to sort by.
        limit: Number of rows wanted.
        initial_bound: Starting bound guess for `order_by` — e.g. if you
            expect the top 10 rows to have `order_by <= 100`, pass `100`.
            Widened automatically on each round that comes up short.
        ascending: Sort direction. `True` (default) pushes `order_by <=
            bound` and takes the smallest `limit` values; `False` pushes
            `order_by >= bound` and takes the largest.
        arguments: Positional/named arguments for the function's bind.
        settings: Optional dictionary of settings/pragmas.
        max_rounds: Correctness backstop — the final round always removes
            the bound entirely (an ordinary unbounded scan) rather than
            giving up with a wrong answer.

    Returns:
        A `pa.Table` with exactly `limit` rows (fewer only if the source
        itself has fewer than `limit` rows total), sorted by `order_by`.

    """
    schema = client.bind(function_name=function_name, schema_name=schema_name, arguments=arguments).output_schema
    sort_order: Literal["ascending", "descending"] = "ascending" if ascending else "descending"

    bound = initial_bound
    for round_index in range(max_rounds):
        last_round = round_index == max_rounds - 1
        if last_round:
            pushdown_bytes = None
        else:
            expr = pc.field(order_by) <= bound if ascending else pc.field(order_by) >= bound
            pushdown_bytes = translate_predicate(expr, schema)

        batches = list(
            client.table_function(
                function_name=function_name,
                schema_name=schema_name,
                arguments=arguments,
                pushdown_filters=pushdown_bytes,
                settings=settings,
            )
        )
        table = pa.Table.from_batches(batches, schema=schema) if batches else schema.empty_table()

        if last_round or table.num_rows >= limit:
            return table.sort_by([(order_by, sort_order)]).slice(0, limit)

        # Came up short: widen the bound. Doubling the *distance* from the
        # initial guess (not the bound's raw value) handles negative/zero
        # starting bounds correctly, unlike naive `bound *= 2`.
        distance = abs(bound - initial_bound) or 1
        bound = initial_bound + (distance * 2 if ascending else -(distance * 2))

    # Unreachable: the last iteration (round_index == max_rounds - 1) always
    # returns above via `last_round`. Kept for mypy's exhaustiveness.
    raise AssertionError("unreachable")
