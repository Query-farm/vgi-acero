# Copyright 2026 Query Farm LLC - https://query.farm

"""Scalar-function -> `pyarrow.compute` UDF bridge, callable inside an Acero expression.

A VGI scalar function call is an out-of-process RPC regardless of how it's
invoked, so this rides `pyarrow.compute.register_scalar_function` — a real,
public (if `EXPERIMENTAL`-labeled) pyarrow API — rather than pretending
otherwise. The registered function is then usable directly inside any Acero
expression an `ac.FilterNodeOptions`/`ac.ProjectNodeOptions` node accepts,
e.g. `pc.field("x") > vgi_fn(pc.field("y"), 2)`.

Two argument kinds, both declared on `FunctionInfo.arguments` (a `pa.Schema`,
one field per declared parameter, in order) — same two kinds vgi-polars'
`_scalar.py` documents, same underlying wire mechanism: a **constant**
parameter (`ConstParam` in vgi-python — its field carries
`metadata[b"vgi_const"] == b"true"`) is bound once per call, not exchanged
per row, so the caller passes a plain Python value for it; every other field
is a per-row **array** parameter.

**Per-call registration, not per-function.** Unlike vgi-polars' `map_batches`
bridge (a single Python closure works for every invocation, since Polars
threads bind-time consts through the same `pl.Expr` object), a
`pyarrow.compute` UDF has no notion of a partially-applied argument at
registration time — its declared `in_types` are exactly its per-call array
arguments, nothing more. So `call()` registers a **fresh, uniquely-named**
scalar function (closing over that call's specific constant values) every
time it's invoked, and returns `pc.call_function(unique_name, array_exprs)`.
This means each call this bridge makes accumulates one more globally-visible
entry in pyarrow's compute function registry for the life of the process —
fine for the ordinary case (a handful of expressions built once per query),
worth knowing for a long-lived process issuing a very large number of
distinct scalar-function calls.

**Scoped secrets.** `call()` takes an optional keyword-only `secrets:
dict[str, Any] | None`, threaded straight through to
`Client.scalar_function`'s own `secrets` parameter.

**Per-chunk input dedup** (`dedup=True`, the default — the client-side
mirror of the DuckDB C++ extension's `vgi_exchange_input_dedup` setting).
Before shipping a chunk's array-argument rows to the worker, `_apply`
deduplicates them to their distinct value tuples, then scatters the worker's
output back to every original row. Scoped to within one call only. Gated on
`FunctionInfo.stability != VOLATILE`. Falls back to shipping the whole batch
unmodified whenever a row's values aren't hashable or dedup wouldn't reduce
the row count.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.compute as pc
from vgi.arguments import Arguments
from vgi.catalog.catalog_interface import FunctionInfo, FunctionStability, SchemaObjectType

from vgi_acero._arguments import is_const_field, to_scalar
from vgi_acero.errors import VGI_CLIENT_ERRORS, VgiAceroError, wrap_error

if TYPE_CHECKING:
    from vgi_acero.catalog import VgiAceroCatalog

#: The callable `VgiAceroCatalog.scalar_function` returns — see `make_scalar_function`.
ScalarFunctionCall = Callable[..., pc.Expression]


def _dedup_positions(batch: pa.RecordBatch) -> tuple[list[int], list[int]] | None:
    """Return `(distinct_positions, inverse)` if deduping `batch`'s rows is possible and worthwhile.

    Deduping is possible if every row's values are hashable, and worthwhile if
    there are fewer distinct rows than total rows. Returns `None` otherwise,
    in which case the caller ships the batch unmodified.
    """
    if batch.num_rows == 0:
        return None
    try:
        rows = [tuple(row.values()) for row in batch.to_pylist()]
        seen: dict[tuple[Any, ...], int] = {}
        distinct_positions: list[int] = []
        inverse: list[int] = []
        for i, row in enumerate(rows):
            idx = seen.get(row)
            if idx is None:
                idx = len(distinct_positions)
                seen[row] = idx
                distinct_positions.append(i)
            inverse.append(idx)
    except TypeError:
        return None  # an unhashable cell (struct/list-typed argument) — skip dedup
    if len(distinct_positions) == len(rows):
        return None  # already all-distinct — dedup would only add overhead
    return distinct_positions, inverse


def make_scalar_function(catalog: VgiAceroCatalog, schema_name: str, name: str) -> ScalarFunctionCall:
    """Return a callable for the scalar function `schema_name.name`.

    Pass a `pc.Expression` for each array parameter and a plain Python value
    for each constant parameter, in the function's declared argument order.
    The `FunctionInfo` (argument/output schema) is resolved on first use and
    cached.
    """
    cache: dict[str, FunctionInfo] = {}

    def _function_info() -> FunctionInfo:
        if "info" not in cache:
            try:
                # Catalog-metadata call — the shared client, not a per-thread
                # exchange one; see catalog.py's "Thread safety" docstring.
                infos = catalog.metadata_client.schema_contents(
                    attach_opaque_data=catalog.attach_opaque_data,
                    name=schema_name,
                    type=SchemaObjectType.SCALAR_FUNCTION,
                )
            except VGI_CLIENT_ERRORS as e:
                raise wrap_error(e) from e
            info = next((i for i in infos if i.name == name), None)
            if info is None:
                raise VgiAceroError(f"scalar function not found: {schema_name}.{name}")
            cache["info"] = info
        return cache["info"]

    def call(*args: Any, secrets: dict[str, Any] | None = None, dedup: bool = True) -> pc.Expression:
        info = _function_info()
        arg_schema = pa.ipc.read_schema(pa.py_buffer(info.arguments))
        out_schema = pa.ipc.read_schema(pa.py_buffer(info.output_schema))
        if len(args) != len(arg_schema.names):
            raise VgiAceroError(f"{schema_name}.{name} expects {len(arg_schema.names)} argument(s), got {len(args)}")

        const_fields = [(i, f) for i, f in enumerate(arg_schema) if is_const_field(f)]
        array_fields = [(i, f) for i, f in enumerate(arg_schema) if not is_const_field(f)]

        for i, f in const_fields:
            if isinstance(args[i], pc.Expression):
                raise VgiAceroError(
                    f"{schema_name}.{name}: argument {i} ('{f.name}') is a constant parameter — "
                    "pass a plain Python value, not a pc.Expression"
                )
        if not array_fields:
            raise VgiAceroError(
                f"{schema_name}.{name} has no non-constant (per-row) arguments — not supported by "
                "vgi-acero's scalar-function bridge"
            )

        # Dense, const-only positional order — see module docstring.
        const_arguments = Arguments(positional=tuple(to_scalar(args[i], f.type) for i, f in const_fields))
        array_exprs = [args[i] if isinstance(args[i], pc.Expression) else pc.scalar(args[i]) for i, _ in array_fields]
        array_schema = pa.schema([f for _, f in array_fields])

        out_field = out_schema.field(0)
        dedup_safe = dedup and info.stability != FunctionStability.VOLATILE

        def _apply(ctx: Any, *arrays: pa.Array[Any]) -> pa.Array[Any]:
            casted = [a.cast(array_schema.field(i).type) for i, a in enumerate(arrays)]
            batch = pa.RecordBatch.from_arrays(casted, schema=array_schema)

            inverse: list[int] | None = None
            if dedup_safe:
                dedup_result = _dedup_positions(batch)
                if dedup_result is not None:
                    distinct_positions, inverse = dedup_result
                    batch = batch.take(pa.array(distinct_positions, type=pa.int64()))

            try:
                # Acero may invoke a registered UDF from multiple worker
                # threads concurrently during plan execution — must use a
                # per-thread client, never one shared across calls.
                out_batches = list(
                    catalog.exchange_client().scalar_function(
                        function_name=name,
                        schema_name=schema_name,
                        input=iter([batch]),
                        arguments=const_arguments,
                        secrets=secrets,
                    )
                )
            except VGI_CLIENT_ERRORS as e:
                raise wrap_error(e) from e
            if not out_batches:
                empty: pa.Array[Any] = pa.array([], type=out_field.type)
                return empty
            out_table = pa.Table.from_batches(out_batches)
            result: pa.Array[Any] = out_table.column(0).combine_chunks()
            if inverse is not None:
                result = result.take(pa.array(inverse, type=pa.int64()))
            return result

        unique_name = f"vgi_acero_{catalog.name}_{schema_name}_{name}_{uuid.uuid4().hex}"
        pc.register_scalar_function(
            _apply,
            function_name=unique_name,
            function_doc={"summary": f"VGI scalar function {schema_name}.{name}", "description": ""},
            in_types={f.name: f.type for _, f in array_fields},
            out_type=out_field.type,
        )
        # pc.call_function() EXECUTES eagerly (expects real arrays, not
        # Expressions -- confirmed live: "Got unexpected argument type
        # Expression for compute function"). Building a deferred Expression
        # node that references a registered function by name has no public
        # API; Expression._call() is the same private mechanism this
        # module's own test suite already relies on for forcing non-default
        # SetLookupOptions onto an is_in expression.
        expr: pc.Expression = pc.Expression._call(unique_name, array_exprs)  # type: ignore[attr-defined] # noqa: SLF001
        return expr

    return call
