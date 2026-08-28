# Copyright 2026 Query Farm LLC - https://query.farm

"""`VgiAceroTable` — a lazy handle to one catalog table.

Schema and scan-function resolution are cheap, scan-free unary catalog RPCs
(`table_get` / `table_scan_function_get`) — the whole point of the catalog-mode
scan path (over the bare-function `vgi_scan()` in `_scan.py`): a
`VgiAceroTable.scan()`'s returned `Declaration` isn't "in flight" until Acero
actually pulls from it, because nothing before that point ever peeked a batch.

Closely mirrors `vgi_polars.table.VgiTable` — the underlying problems (schema/
scan-function/branch resolution, native-scan delegation, required-filters
cost-safety) are identical; only the terminal "returns a `pl.LazyFrame`" step
differs, become "returns an `ac.Declaration`" here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.acero as ac
import pyarrow.compute as pc
from vgi.catalog.catalog_interface import (
    ColumnStatistics,
    FunctionInfo,
    ScanFunctionResult,
    SchemaObjectType,
    TableInfo,
)

from vgi_acero._filter_translate import translate_predicate
from vgi_acero._scan import make_vgi_scan_declaration
from vgi_acero.errors import VGI_CLIENT_ERRORS, VgiAceroError

if TYPE_CHECKING:
    from vgi.catalog.catalog_interface import ScanBranch

    from vgi_acero.catalog import VgiAceroCatalog

__all__ = ["VgiAceroTable"]

# Mirrors the DuckDB C++ extension's own minimal v1.0 branch_filter binder
# scope (and vgi-polars' `_multi_branch.py`): an AND-chain of "col OP const"
# comparisons. OR is out of scope for a *branch* filter (it's a mandatory
# partition boundary, not an optional predicate — see `_parse_branch_filter`'s
# docstring), matching the C++ extension's own binder restriction.
_COMPARISON_RE = re.compile(r"^(\w+)\s*(<=|>=|<>|!=|=|<|>)\s*(.+)$")
_AND_SPLIT_RE = re.compile(r"(?i)\s+AND\s+")
_OPS: dict[str, Any] = {
    "=": lambda c, v: c == v,
    "<>": lambda c, v: c != v,
    "!=": lambda c, v: c != v,
    "<": lambda c, v: c < v,
    "<=": lambda c, v: c <= v,
    ">": lambda c, v: c > v,
    ">=": lambda c, v: c >= v,
}


def _parse_literal(text: str) -> Any:
    text = text.strip()
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1]
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if text.upper() in ("TRUE", "FALSE"):
        return text.upper() == "TRUE"
    raise VgiAceroError(f"branch_filter: literal not understood: {text!r}")


def parse_branch_filter(sql: str) -> pc.Expression:
    """Parse a `ScanBranch.branch_filter` SQL string into a `pc.Expression`.

    `branch_filter` is not an optional pushdown hint — it's a worker-declared
    partition boundary that MUST be applied, or overlapping branches produce
    duplicate/wrong rows in the unioned result. There is no local
    "always re-verify" fallback for it the way there is for a caller's own
    predicate (see `_filter_translate.py`'s module docstring) — an
    unparseable `branch_filter` raises rather than silently scanning the
    branch unconstrained, which would double-count overlapping rows.
    """
    conjuncts = _AND_SPLIT_RE.split(sql.strip())
    expr: pc.Expression | None = None
    for conjunct in conjuncts:
        match = _COMPARISON_RE.match(conjunct.strip())
        if not match:
            raise VgiAceroError(
                f"branch_filter expression not understood: {conjunct!r} — only AND-chains of "
                "'col OP const' comparisons are supported"
            )
        col, op, literal_text = match.groups()
        clause = _OPS[op](pc.field(col), _parse_literal(literal_text))
        expr = clause if expr is None else expr & clause
    assert expr is not None  # sql is non-empty; _AND_SPLIT_RE always yields >= 1 conjunct
    return expr


class VgiAceroTable:
    """A lazy handle to one table in an attached VGI catalog.

    Construct via `VgiAceroCatalog.table(schema_name, name)`, not directly.
    """

    def __init__(
        self,
        *,
        catalog: VgiAceroCatalog,
        schema_name: str,
        name: str,
        at_unit: str | None = None,
        at_value: str | None = None,
    ) -> None:
        """Wrap a resolved `(schema_name, name)` in `catalog`. Use `VgiAceroCatalog.table(...)`, not this directly."""
        self._catalog = catalog
        self.schema_name = schema_name
        self.name = name
        # Time travel: immutable per-instance — a `VgiAceroTable` at one AT
        # clause and one at another are different views (possibly different
        # schemas) and must not share the memoized resolution below. Get a
        # table at a different version via `VgiAceroCatalog.table(...,
        # at_unit=..., at_value=...)` again, not by mutating this one.
        self.at_unit = at_unit
        self.at_value = at_value
        self._table_info: TableInfo | None = None
        self._scan_function: ScanFunctionResult | None = None
        self._function_info: FunctionInfo | None | Any = _UNRESOLVED
        self._scan_function_schema: str | None = None
        self._scan_branches: list[Any] | None = None

    def _table_get(self) -> TableInfo:
        if self._table_info is None:
            try:
                info = self._catalog.client.table_get(
                    attach_opaque_data=self._catalog.attach_opaque_data,
                    schema_name=self.schema_name,
                    name=self.name,
                    at_unit=self.at_unit,
                    at_value=self.at_value,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiAceroError(str(e)) from e
            if info is None:
                raise VgiAceroError(f"table not found: {self.schema_name}.{self.name}")
            self._table_info = info
        return self._table_info

    def _scan_function_get(self) -> ScanFunctionResult:
        if self._scan_function is None:
            try:
                self._scan_function = self._catalog.client.table_scan_function_get(
                    attach_opaque_data=self._catalog.attach_opaque_data,
                    schema_name=self.schema_name,
                    name=self.name,
                    at_unit=self.at_unit,
                    at_value=self.at_value,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiAceroError(str(e)) from e
        return self._scan_function

    def _lookup_table_function(self, schema_name: str, function_name: str) -> FunctionInfo | None:
        try:
            infos = self._catalog.client.schema_contents(
                attach_opaque_data=self._catalog.attach_opaque_data,
                name=schema_name,
                type=SchemaObjectType.TABLE_FUNCTION,
            )
        except VGI_CLIENT_ERRORS:
            return None
        return next((i for i in infos if i.name == function_name), None)

    def _resolve_scan_function(self) -> None:
        """Resolve the `FunctionInfo` and the schema the scan function actually lives in.

        Not necessarily this table's own schema — a worker registers function
        names per schema and may reuse a name across schemas (observed live
        against `vgi-fixture-worker`: `data.filter_echo_table` resolves to a
        scan function only registered in schema `main`). Mirrors the DuckDB
        C++ extension's own resolution order: try the table's own schema
        first, then the catalog's default schema.
        """
        if self._function_info is not _UNRESOLVED:
            return
        scan_fn = self._scan_function_get()

        info = self._lookup_table_function(self.schema_name, scan_fn.function_name)
        if info is not None:
            self._function_info = info
            self._scan_function_schema = self.schema_name
            return

        default_schema = self._catalog.default_schema
        if default_schema != self.schema_name:
            info = self._lookup_table_function(default_schema, scan_fn.function_name)
            if info is not None:
                self._function_info = info
                self._scan_function_schema = default_schema
                return

        self._function_info = None
        self._scan_function_schema = self.schema_name

    def _function_info_get(self) -> FunctionInfo | None:
        """The `FunctionInfo` for the resolved scan function, if discoverable.

        Used to check `projection_pushdown`/`filter_pushdown` opt-in flags.
        `None` means "couldn't find it, assume no pushdown support" — always
        safe, since `scan()` composes any pushed filter with the caller's own
        local `FilterNodeOptions` node regardless.
        """
        self._resolve_scan_function()
        return self._function_info

    def scan_function_schema(self) -> str:
        """The schema to call the resolved scan function in.

        See `_resolve_scan_function`'s docstring for why this can differ from
        `self.schema_name`.
        """
        self._resolve_scan_function()
        assert self._scan_function_schema is not None
        return self._scan_function_schema

    @property
    def arrow_schema(self) -> pa.Schema:
        """The table's schema as a `pyarrow.Schema` (no scan)."""
        return pa.ipc.read_schema(pa.py_buffer(self._table_get().columns))

    def _scan_branches_get(self) -> list[ScanBranch]:
        """The table's scan branches (memoized).

        One `ScanBranch` for an ordinary single-source table, more for a
        multi-branch one. `Client.table_scan_branches_get` transparently
        falls back to wrapping `table_scan_function_get`'s single result as
        one branch for a worker that predates the branches RPC — so this is
        always safe to call.
        """
        if self._scan_branches is None:
            try:
                result = self._catalog.client.table_scan_branches_get(
                    attach_opaque_data=self._catalog.attach_opaque_data,
                    schema_name=self.schema_name,
                    name=self.name,
                    at_unit=self.at_unit,
                    at_value=self.at_value,
                )
            except VGI_CLIENT_ERRORS as e:
                raise VgiAceroError(str(e)) from e
            self._scan_branches = list(result.branches)
        return self._scan_branches

    def _scan_one(
        self,
        *,
        function_name: str,
        function_schema: str,
        columns: list[str] | None,
        filter: pc.Expression | None,  # noqa: A002 - matches ds.dataset()'s own kwarg name
        function_info: FunctionInfo | None,
    ) -> ac.Declaration:
        """Build one scan `Declaration` for a resolved `(function_schema, function_name)`.

        Applies projection/filter pushdown per `function_info`'s advertised
        flags (never both at once — see module docstring's "Non-goals"),
        then unconditionally composes a local `ac.FilterNodeOptions(filter)`
        node on top when `filter` is given — cheap insurance regardless of
        what got pushed, since VGI pushdown is only ever an optimization.
        """
        full_schema = self.arrow_schema
        projection_ids = None
        pushdown_bytes = None
        can_project = function_info is not None and function_info.projection_pushdown
        can_filter = function_info is not None and function_info.filter_pushdown

        if columns is not None and can_project:
            projection_ids = [full_schema.get_field_index(c) for c in columns]
        if filter is not None and can_filter and projection_ids is None:
            # Gate the bounded `expression`-type fallback on the worker
            # having declared it can evaluate at least one expression-filter
            # class at all (vgi/table_filter_pushdown.py's ExpressionFilter
            # needs a DuckDB-compatible engine worker-side) -- coarser than
            # matching this specific predicate's exact class against
            # `supported_expression_filters`, but a real, non-trivial gate
            # rather than pushing blind. See translate_predicate's docstring.
            allow_expression_fallback = bool(function_info and function_info.supported_expression_filters)
            pushdown_bytes = translate_predicate(
                filter, full_schema, allow_expression_fallback=allow_expression_fallback
            )

        gen = self._catalog._exchange_client().table_function(
            function_name=function_name,
            schema_name=function_schema,
            projection_ids=projection_ids,
            pushdown_filters=pushdown_bytes,
        )
        decl = make_vgi_scan_declaration(gen)
        if filter is not None:
            decl = ac.Declaration("filter", ac.FilterNodeOptions(filter), inputs=[decl])
        return decl

    def scan(
        self,
        *,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,  # noqa: A002
        acknowledge_required_filters: bool = False,
    ) -> ac.Declaration:
        """A pushdown-aware scan of this table as an Acero `Declaration`.

        Transparently handles a multi-branch table (a unioned `Declaration`
        of one scan per branch, each branch's `branch_filter` applied — see
        `scan_all_branches`) — the common, single-branch case takes the
        unchanged single-scan path with no multi-branch overhead beyond one
        extra (cheap, memoized, unary) `table_scan_branches_get` catalog call.

        **Native scan-function delegation** (`_native_scan.py`): when the
        resolved scan function names something Acero can satisfy directly
        (currently `read_parquet`/`read_csv` -> `ds.dataset(...)`) rather than
        a VGI-hosted function, this returns that native `Declaration` straight
        away — no worker round-trip for the data at all. `columns`/`filter`
        are not yet threaded into the native path (`ac.ScanNodeOptions`
        itself accepts `columns=`/`filter=` scan-level pushdown — wiring
        those through is a natural follow-up, not done here; see
        `_native_scan.py`'s module docstring). If the table also
        declares `required_filters`, this raises `VgiAceroError` unless
        `acknowledge_required_filters=True` — there's no hook to inspect the
        eventual predicate before execution for a natively-delegated scan, so
        refusing by default beats silently dropping a real safety guard.
        """
        branches = self._scan_branches_get()
        if len(branches) != 1:
            return self.scan_all_branches()

        from vgi_acero._native_scan import NATIVE_SCAN_HANDLERS

        scan_fn = self._scan_function_get()
        native_handler = NATIVE_SCAN_HANDLERS.get(scan_fn.function_name)
        if native_handler is not None:
            required = self.required_filters()
            if required and not acknowledge_required_filters:
                raise VgiAceroError(
                    f"{self.schema_name}.{self.name}: natively delegates to "
                    f"{scan_fn.function_name!r} and declares required_filters {required} that "
                    "vgi-acero cannot enforce for a native scan (no hook to inspect the eventual "
                    "predicate before execution). Pass scan(acknowledge_required_filters=True) "
                    "once you've applied the equivalent filter(s) yourself, or you WILL trigger a "
                    "full, possibly enormous, unfiltered remote read."
                )
            return native_handler(scan_fn, schema_name=self.schema_name, table_name=self.name)

        return self._scan_one(
            function_name=scan_fn.function_name,
            function_schema=self.scan_function_schema(),
            columns=columns,
            filter=filter,
            function_info=self._function_info_get(),
        )

    def scan_all_branches(self) -> ac.Declaration:
        """Force a branch-unioned scan regardless of branch count.

        Zero branches is legal (a fully-pruned multi-branch scan prunes to
        nothing) and returns an empty-schema-matching `Declaration` via a
        zero-row `TableSourceNodeOptions`. Catalog-table branches
        (`branch.source_table is not None`) and format branches
        (`branch.format_name is not None`) are not yet supported — raises,
        matching vgi-polars' own documented scope limit (neither has an
        established resolution path in this package yet).
        """
        branches = self._scan_branches_get()
        if not branches:
            empty = self.arrow_schema.empty_table()
            return ac.Declaration("table_source", ac.TableSourceNodeOptions(empty))

        decls = []
        for branch in branches:
            if branch.source_table is not None:
                raise VgiAceroError(
                    f"{self.schema_name}.{self.name}: a catalog-table branch "
                    f"(source_table={branch.source_table!r}) is not yet supported by vgi-acero."
                )
            if branch.format_name is not None:
                raise VgiAceroError(
                    f"{self.schema_name}.{self.name}: a format branch "
                    f"(format_name={branch.format_name!r}) is not yet supported by vgi-acero."
                )
            function_info = self._lookup_table_function(self.schema_name, branch.function_name)
            decl = self._scan_one(
                function_name=branch.function_name,
                function_schema=self.schema_name,
                columns=None,
                filter=None,
                function_info=function_info,
            )
            if branch.branch_filter:
                decl = ac.Declaration(
                    "filter", ac.FilterNodeOptions(parse_branch_filter(branch.branch_filter)), inputs=[decl]
                )
            decls.append(decl)
        return ac.Declaration("union", ac.ExecNodeOptions(), inputs=decls)

    def read(
        self,
        *,
        columns: list[str] | None = None,
        filter: pc.Expression | None = None,  # noqa: A002
        acknowledge_required_filters: bool = False,
    ) -> pa.Table:
        """An eager, full scan of this table. See `scan()` for the parameters."""
        return self.scan(
            columns=columns, filter=filter, acknowledge_required_filters=acknowledge_required_filters
        ).to_table()

    def statistics(self) -> list[ColumnStatistics]:
        """Per-column statistics (min/max/null presence/distinct count/...), if the worker advertises them.

        A plain catalog-metadata RPC, no scan.
        """
        try:
            return self._catalog.client.table_column_statistics(
                attach_opaque_data=self._catalog.attach_opaque_data,
                schema_name=self.schema_name,
                name=self.name,
            )
        except VGI_CLIENT_ERRORS as e:
            raise VgiAceroError(str(e)) from e

    def required_filters(self) -> list[list[str]]:
        """AND-of-OR-groups of column names a scan predicate must reference at least one of, per group.

        Purely declarative on the wire (`TableInfo.required_filters`) —
        `scan()` enforces it (for natively-delegated tables only, see
        `_native_scan.py`) as a cost-safety guard before scanning.
        """
        return list(self._table_get().required_filters)


class _Unresolved:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<unresolved>"


_UNRESOLVED = _Unresolved()
