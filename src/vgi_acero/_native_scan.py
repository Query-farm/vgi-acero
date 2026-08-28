# Copyright 2026 Query Farm LLC - https://query.farm

"""Native scan-function delegation — VGI (and `Client.table_function()`) bypassed entirely.

`ScanFunctionResult.function_name` can name a reader the CALLING engine should
run itself, not a VGI-hosted function. Mirrors the DuckDB C++ extension's own
resolution order (`~/Development/vgi/src/storage/vgi_table_entry.cpp`'s
`GetScanFunctionImpl`) and vgi-polars' `_native_scan.py`: `table_scan_function_get`'s
`function_name` is looked up in the calling engine's OWN function catalog
*before* ever being treated as a VGI RPC target. `read_parquet`/`read_csv`
are DuckDB built-ins a worker can delegate to so the actual read runs entirely
client-side — Acero's own `pyarrow.dataset` readers do row-group pruning,
predicate/projection pushdown, and cloud range reads directly, no worker
round-trip for the data at all.

**Only `read_parquet`/`read_csv` are mapped — not `iceberg_scan`.** Unlike
Polars (`pl.scan_iceberg`), plain `pyarrow.dataset` has no built-in Iceberg
reader (would need `pyiceberg` as an additional dependency this package
doesn't carry) — an `iceberg_scan` delegation is therefore left unmapped and
falls through to an ordinary VGI-hosted scan attempt, which will fail loudly
(`FunctionNotFoundError` from the worker) rather than silently mis-scanning.

**Required-filters cost-safety is NOT enforced here — by necessity, not
oversight**, for the same reason vgi-polars' equivalent module documents: a
natively-delegated scan returns a `Declaration` immediately, with no callback
to inspect the eventual predicate before execution. `table.py`'s `scan()`
refuses outright (raises `VgiAceroError`) whenever a natively-delegated table
declares `required_filters`, unless the caller passes
`acknowledge_required_filters=True`.

**Confirmed live for `read_parquet`** against the same class of
delegating worker vgi-polars verified against. `read_csv` is built
conservatively from `pyarrow.dataset`'s own documented format options, not
independently re-verified here — same caveat vgi-polars' own module carries
for its `read_csv`/`iceberg_scan` entries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pyarrow.acero as ac
import pyarrow.dataset as ds

from vgi_acero.errors import VgiAceroError

if TYPE_CHECKING:
    from vgi.catalog.catalog_interface import ScanFunctionResult

NativeScanHandler = Callable[..., ac.Declaration]


def _make_native_scan_handler(
    *,
    duckdb_function_name: str,
    dataset_format: str,
    arg_name_map: dict[str, str],
) -> NativeScanHandler:
    """Build a `NATIVE_SCAN_HANDLERS` entry: wire-arg validation + a `ds.dataset(...)`-backed scan.

    Args:
        duckdb_function_name: The `ScanFunctionResult.function_name` this
            handler answers for (e.g. `"read_parquet"`) — used only to name
            it in error messages.
        dataset_format: The `pyarrow.dataset.dataset(..., format=...)` value.
        arg_name_map: `{wire named-argument name: ds.dataset() keyword
            parameter name}` for every named argument this handler knows how
            to translate. A wire named argument outside this map raises
            rather than being silently dropped.

    Returns:
        A handler matching `NATIVE_SCAN_HANDLERS`'s call signature.

    """
    known_wire_names = set(arg_name_map)

    def handler(
        scan_fn: ScanFunctionResult,
        *,
        schema_name: str,
        table_name: str,
    ) -> ac.Declaration:
        if not scan_fn.positional_arguments:
            raise VgiAceroError(
                f"{schema_name}.{table_name}: worker delegated to {duckdb_function_name} with no "
                "positional arguments (expected the file/glob path as argument 0)"
            )
        path = scan_fn.positional_arguments[0].as_py()
        if not isinstance(path, str):
            raise VgiAceroError(
                f"{schema_name}.{table_name}: {duckdb_function_name}'s first argument is a "
                f"{type(path).__name__}, expected a path/glob string"
            )

        kwargs: dict[str, Any] = {}
        unknown = sorted(set(scan_fn.named_arguments or {}) - known_wire_names)
        if unknown:
            raise VgiAceroError(
                f"{schema_name}.{table_name}: worker's {duckdb_function_name} delegation passed "
                f"named argument(s) {unknown} vgi-acero doesn't know how to translate for "
                f"{dataset_format!r} datasets (known: {sorted(known_wire_names)})"
            )
        for wire_name, ds_kwarg in arg_name_map.items():
            value = (scan_fn.named_arguments or {}).get(wire_name)
            if value is not None:
                kwargs[ds_kwarg] = value.as_py()

        # pyarrow-stubs types `format=` as a closed Literal set; `dataset_format`
        # is a plain str parameter here by design (this helper is reused for
        # more than one format), so no single Literal fits statically.
        dataset = ds.dataset(path, format=dataset_format, **kwargs)  # type: ignore[call-overload]
        # ScanNodeOptions is dynamically re-exported into pyarrow.acero at
        # runtime (confirmed working); pyarrow-stubs doesn't model it there.
        return ac.Declaration("scan", ac.ScanNodeOptions(dataset))  # type: ignore[attr-defined]

    return handler


def _scan_parquet_native(scan_fn: ScanFunctionResult, *, schema_name: str, table_name: str) -> ac.Declaration:
    """`read_parquet` -> `ds.dataset(path, format="parquet")`.

    `hive_partitioning` (a wire boolean) needs real translation, not a bare
    rename — `pyarrow.dataset`'s equivalent is `partitioning="hive"`
    (omitted, not `partitioning=False`, when disabled) — so this is a bespoke
    handler rather than a `_make_native_scan_handler(arg_name_map=...)`
    passthrough.
    """
    if not scan_fn.positional_arguments:
        raise VgiAceroError(
            f"{schema_name}.{table_name}: worker delegated to read_parquet with no positional "
            "arguments (expected the file/glob path as argument 0)"
        )
    path = scan_fn.positional_arguments[0].as_py()
    if not isinstance(path, str):
        raise VgiAceroError(
            f"{schema_name}.{table_name}: read_parquet's first argument is a "
            f"{type(path).__name__}, expected a path/glob string"
        )
    named = scan_fn.named_arguments or {}
    unknown = sorted(set(named) - {"hive_partitioning"})
    if unknown:
        raise VgiAceroError(
            f"{schema_name}.{table_name}: worker's read_parquet delegation passed named "
            f"argument(s) {unknown} vgi-acero doesn't know how to translate (known: "
            "['hive_partitioning'])"
        )
    kwargs: dict[str, Any] = {}
    hive_partitioning = named.get("hive_partitioning")
    if hive_partitioning is not None and hive_partitioning.as_py():
        kwargs["partitioning"] = "hive"
    dataset = ds.dataset(path, format="parquet", **kwargs)
    # ScanNodeOptions is dynamically re-exported into pyarrow.acero at
    # runtime (confirmed working); pyarrow-stubs doesn't model it there.
    return ac.Declaration("scan", ac.ScanNodeOptions(dataset))  # type: ignore[attr-defined]


_scan_csv_native = _make_native_scan_handler(
    duckdb_function_name="read_csv",
    dataset_format="csv",
    arg_name_map={},
)

#: `ScanFunctionResult.function_name` -> the Acero-native builder that
#: satisfies it directly, bypassing `Client.table_function` entirely. See
#: module docstring.
NATIVE_SCAN_HANDLERS: dict[str, NativeScanHandler] = {
    "read_parquet": _scan_parquet_native,
    "read_csv": _scan_csv_native,
}
