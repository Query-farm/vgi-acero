# Copyright 2026 Query Farm LLC - https://query.farm

"""Building `ac.Declaration` source nodes from a VGI `Client`.

`pyarrow.acero`'s Python API has no "logical plan, not yet executed" phase for
a custom source the way DataFusion's `TableProvider::scan()` does — a
`Declaration` wrapping a live generator is already "this will pull batches
from something" once it exists. Concretely: `make_vgi_scan_declaration`'s
`next(gen)` call (to learn the schema) is real execution — the worker's
`bind()`/`init()`/first `process()` have already run by the time a
`Declaration` comes back. If the caller's plan never executes that
`Declaration` to completion (discards it, or an earlier plan error short-
circuits before reaching it), the generator is abandoned mid-stream: VGI's
subprocess transport is lockstep RPC, and there is no client-side "cancel this
one stream" primitive (`Client.stop(force=True)` abandons the *whole*
`Client`, subprocess transport only — see its own docstring). So: **a
`Declaration` returned from this module must be part of a plan that is
actually executed to completion, or the whole `Client` it came from abandoned
via `Client.stop(force=True)` — never discard it or reuse that `Client`
afterward if you do.**

`VgiAceroTable.scan()` (`table.py`) avoids this caveat for schema discovery by
using `Client.table_get()`/catalog RPCs instead of peeking a batch — only the
data generator itself is ever "in flight," and only once Acero actually pulls
from it. Prefer that path whenever a catalog attach is available.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.dataset as ds
from vgi.protocol import ScanSplit

if TYPE_CHECKING:
    from vgi.arguments import Arguments
    from vgi.client.client import Client

__all__ = ["SplitScanResult", "make_vgi_scan_declaration", "vgi_scan", "vgi_scan_splits"]


@dataclass
class SplitScanResult:
    """Result of `vgi_scan_splits()`: a plan plus the `Client`s it opened.

    `vgi_scan_splits()` creates one `Client` per split internally (via the
    caller's `client_factory`) — unlike `vgi_scan()`, where the caller
    already holds the one `Client` involved, there is no way for the caller
    to reach these otherwise. `Client` has no `__del__`/`atexit` cleanup, so
    without this, every split's subprocess worker would be silently orphaned
    for the life of the process. Call `close()` (or stop each of `clients`
    yourself) once you're done running `declaration`'s plan to completion —
    never before, per `make_vgi_scan_declaration`'s "must be executed"
    caveat, which applies to every one of these clients' open generators.

    Also usable as a context manager: `with vgi_scan_splits(...) as split:
    table = split.declaration.to_table()` runs `close()` on exit, matching
    the `with attach(...) as catalog:` pattern this package already uses
    elsewhere — the plain `try/finally` shown above works too, this is just
    the more idiomatic PyArrow/Python-shaped spelling of the same thing.

    Attributes:
        declaration: The `ac.Declaration("union", ...)` (or, for a
            zero-split result, a `table_source` over an empty table).
        clients: Every per-split `Client` this call created and started.
            Empty for a zero-split result — that branch's own internal
            `Client` (used only for unary `bind()`/`table_function_plan()`
            calls, never an open stream) is stopped before `vgi_scan_splits`
            returns, so there is nothing left for the caller to close.
    """

    declaration: ac.Declaration
    clients: list[Client]

    def close(self) -> None:
        """Stop every `Client` this result holds. Safe to call once you're done with `declaration`."""
        for client in self.clients:
            client.stop()

    def __enter__(self) -> Self:
        """Support `with vgi_scan_splits(...) as split:` — returns `self`."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Support `with vgi_scan_splits(...) as split:` — calls `close()`."""
        self.close()


def make_vgi_scan_declaration(gen: Iterator[pa.RecordBatch]) -> ac.Declaration:
    """Wrap a VGI batch generator as an Acero `Declaration` source node.

    Pulls the first batch to learn the schema (see module docstring for why
    that's real execution, not a lazy peek), then wraps the whole generator
    (first batch re-yielded, rest pulled lazily) as a `RecordBatchReader` ->
    `ds.InMemoryDataset` -> `ac.ScanNodeOptions` -> `ac.Declaration("scan",
    ...)`. This is the one core mechanism every scan path in this package
    (`vgi_scan`, `vgi_scan_splits`, `VgiAceroTable.scan`, `_join.vgi_semi_join_scan`)
    builds on.

    Args:
        gen: A generator of `pa.RecordBatch` — typically
            `Client.table_function(...)`'s return value.

    Returns:
        An `ac.Declaration` ready to compose with further Acero nodes, or run
        directly via `.to_table()`/`.to_reader()`.

    Raises:
        StopIteration: If `gen` yields zero batches — a VGI table function
            always produces at least an empty-schema batch on bind, so this
            would indicate a genuinely empty function registry response, not
            an empty *table*.

    """
    first = next(gen)
    schema = first.schema

    def _batches() -> Iterator[pa.RecordBatch]:
        yield first
        yield from gen

    reader = pa.RecordBatchReader.from_batches(schema, _batches())
    # pyarrow-stubs doesn't model `ScanNodeOptions` as an attribute of the
    # `pyarrow.acero` module (it's dynamically re-exported from
    # `pyarrow._dataset` at runtime), nor `InMemoryDataset`'s
    # reader-plus-schema constructor overload — both confirmed working at
    # runtime against the installed pyarrow version.
    dataset = ds.InMemoryDataset(reader, schema=schema)  # type: ignore[call-arg]
    return ac.Declaration("scan", ac.ScanNodeOptions(dataset))  # type: ignore[attr-defined]


def vgi_scan(
    client: Client,
    *,
    schema_name: str,
    function_name: str,
    arguments: Arguments | None = None,
    projection_ids: list[int] | None = None,
    pushdown_filters: bytes | None = None,
    join_keys: list[pa.RecordBatch] | None = None,
    settings: dict[str, Any] | None = None,
) -> ac.Declaration:
    """Scan a bare (non-catalog) VGI table function as an Acero `Declaration`.

    For a table function with no catalog entry — the shape the original
    filter-pushdown spike used against `main.filter_echo`, and the only way to
    reach several diagnostic fixtures not wired into a catalog as tables.
    Prefer `VgiAceroCatalog.table(...).scan()` when a catalog attach is
    available — its schema discovery has no execution side effect at all,
    unlike this function (see module docstring).

    Args:
        client: A started `vgi.client.Client`. Not thread-shared — see
            `VgiAceroCatalog._exchange_client()` if calling from a thread pool
            (e.g. one thread per `vgi_scan_splits()` split).
        schema_name: The catalog schema declaring `function_name`.
        function_name: The table function to scan.
        arguments: Positional/named arguments for the function's bind.
        projection_ids: Column indices to project, or `None` for all columns.
        pushdown_filters: Serialized filter-pushdown IPC bytes — build via
            `vgi_acero._filter_translate.translate_predicate`.
        join_keys: Serialized join-key batches for semi-join pushdown — see
            `_join.vgi_semi_join_scan` for the common case of building these
            from a build-side join column.
        settings: Optional dictionary of settings/pragmas.

    Returns:
        An `ac.Declaration` — see module docstring for the "must be executed"
        caveat.

    """
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


def vgi_scan_splits(
    client_factory: Callable[[], Client],
    *,
    schema_name: str,
    function_name: str,
    arguments: Arguments | None = None,
    projection_ids: list[int] | None = None,
    pushdown_filters: bytes | None = None,
    join_keys: list[pa.RecordBatch] | None = None,
    settings: dict[str, Any] | None = None,
) -> SplitScanResult:
    """Scan a split-capable VGI table function as a unioned, concurrently-pulled Acero plan.

    Unlike vgi-polars' equivalent (sequential split chaining — Polars'
    `register_io_source` has no concurrent-pull hook), Acero's `union` exec
    node genuinely pulls from multiple input `Declaration`s concurrently via
    its own thread pool (confirmed empirically — see this package's test
    suite) — so this is real parallelism, not a single-threaded relabeling of
    the same work.

    Each split gets its **own** `Client` (from `client_factory`, e.g.
    `VgiAceroCatalog._exchange_client`), never a shared one — `Client` is not
    thread-safe for concurrent exchange-mode calls, and Acero may pull from
    several `union` inputs concurrently. Unlike `vgi_scan()` (where the
    caller already holds the one `Client` involved), this function creates
    all of these itself, so the returned `SplitScanResult` carries them —
    call `.close()` once you're done executing `.declaration`'s plan. A
    partial failure while building split declarations (a later split's bind
    fails, a subprocess spawn errors) stops every split `Client` already
    opened before re-raising, rather than leaving them orphaned.

    Args:
        client_factory: Returns a fresh, already-started `Client` each call.
            One is created per split (plus one short-lived one for the
            planning call itself, stopped before this function returns).
        schema_name: The catalog schema declaring `function_name`.
        function_name: The split-capable table function to scan (check
            `FunctionInfo.supports_splits` first — a worker that doesn't opt
            in still answers, typically with one degenerate split for the
            whole scan).
        arguments: Positional/named arguments for the function's bind — must
            match what was passed to `table_function_plan()`.
        projection_ids: Column indices to project, or `None` for all columns.
        pushdown_filters: Serialized filter-pushdown IPC bytes, same as
            `vgi_scan`'s.
        join_keys: Serialized join-key batches for semi-join pushdown, same
            as `vgi_scan`'s — threaded into both the planning call and every
            split's redemption.
        settings: Optional dictionary of settings/pragmas.

    Returns:
        A `SplitScanResult` — `.declaration` is an `ac.Declaration("union",
        ...)` over one scan `Declaration` per split (see
        `make_vgi_scan_declaration`'s "must be executed" caveat, which
        applies to every split scan this builds); `.clients` is every
        `Client` you must `.close()` once that plan has run to completion.

    """
    plan_client = client_factory()
    try:
        plan = plan_client.table_function_plan(
            function_name=function_name,
            schema_name=schema_name,
            arguments=arguments,
            projection_ids=projection_ids,
            pushdown_filters=pushdown_filters,
            join_keys=join_keys,
            settings=settings,
        )

        # Client.table_function_plan() always fully deserializes plan.splits to
        # ScanSplit objects before returning (see its own docstring/implementation)
        # — the `list[ScanSplit] | list[bytes]` field type only reflects the wire
        # shape before that deserialization, so this assert should never fail.
        splits: list[ScanSplit] = []
        for split in plan.splits:
            assert isinstance(split, ScanSplit), f"expected a deserialized ScanSplit, got {type(split)}"
            splits.append(split)

        if not splits:
            # No splits is legal (a fully-pruned scan, or a genuinely empty
            # table) and means an empty result — NOT "fall back to an
            # ordinary whole-scan": a split-only function (one that
            # implements on_plan()/on_split() with no primary/secondary path)
            # rejects a non-split init outright, confirmed live against
            # vgi-fixture-worker's `split_zero` fixture (`RuntimeError: ...
            # is split-only but was initialized with no split tokens`). Use
            # Client.bind() (zero execution, and this connection is about to
            # be stopped anyway) to get the schema for an empty
            # TableSourceNodeOptions instead.
            bind_response = plan_client.bind(function_name=function_name, schema_name=schema_name, arguments=arguments)
            output_schema = bind_response.output_schema
            if projection_ids is not None:
                output_schema = pa.schema([output_schema.field(i) for i in projection_ids])
            empty_decl = ac.Declaration("table_source", ac.TableSourceNodeOptions(output_schema.empty_table()))
            return SplitScanResult(declaration=empty_decl, clients=[])
    finally:
        # Only ever used for unary bind()/table_function_plan() calls above —
        # no open stream, so it's always safe to stop here regardless of
        # which branch returns.
        plan_client.stop()

    split_clients: list[Client] = []
    try:
        split_decls = []
        for split in splits:
            split_client = client_factory()
            split_clients.append(split_client)
            gen = split_client.table_function(
                function_name=function_name,
                schema_name=schema_name,
                arguments=arguments,
                projection_ids=projection_ids,
                pushdown_filters=pushdown_filters,
                join_keys=join_keys,
                settings=settings,
                split_tokens=[split.token],
                split_execution_id=plan.execution_id,
                split_init_opaque_data=plan.init_opaque_data,
            )
            split_decls.append(make_vgi_scan_declaration(gen))
    except BaseException:
        for split_client in split_clients:
            split_client.stop()
        raise

    union_decl = ac.Declaration("union", ac.ExecNodeOptions(), inputs=split_decls)
    return SplitScanResult(declaration=union_decl, clients=split_clients)
