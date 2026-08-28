# Copyright 2026 Query Farm LLC - https://query.farm

"""Best-effort translation of a pyarrow filter `Expression` into VGI's filter-pushdown wire format.

This is purely an optimization: the caller was always going to add their own
`ac.Declaration("filter", ac.FilterNodeOptions(expr))` node on top of the scan
to do anything with the predicate in Acero, so that node is the correctness
backstop for free — unlike vgi-polars, which must manually re-apply the
complete predicate locally because Polars' `register_io_source` does not
re-verify. See `VgiAceroTable.scan()`'s docstring for where that local filter
node gets added automatically for the convenience API. This module is
therefore free to be conservative: an un-translatable predicate, or one of its
conjuncts, is simply not pushed, never an error.

**How this walks the Expression.** `pyarrow.compute.Expression` exposes no
public decomposition API in pure Python (no `.op`/`.args` accessors). The only
two ways to get structure out of one are (a) parse `repr()` text, or (b)
serialize via `Expression.to_substrait()` and decode the Substrait protobuf
(needs the external `substrait` package, not installed here, and confirmed
this doesn't help anyway: `pyarrow.substrait.deserialize_expressions()` just
hands back another opaque `Expression`). This module takes route (a): parse
`repr()` through Python's own `ast.parse()` after a small regex pass turns
pyarrow's non-Python pretty-printing (`is_in(...)`'s `{value_set=...}` block,
`FieldRef.Nested(...)`, lowercase `true`/`false`/`null`) into valid Python
syntax. Good enough to prove the wire format round-trips correctly (see this
package's test suite, which does so against a real worker) — but `repr()`
isn't a documented/stable contract, so a production hardening pass would want
to walk Substrait instead.

**Full VGI filter vocabulary coverage** (`vgi/table_filter_pushdown.py`):

- `constant` (eq/ne/gt/ge/lt/le), `is_null`, `is_not_null`, `and`, `or`.
- `in` — only when the expression's `null_matching_behavior` is left at
  pyarrow's own default (`MATCH`), matching worker-side `InFilter.evaluate`'s
  own bare `pc.is_in(col, vals)` call (no options) exactly. An explicit
  non-default override is declined, never mistranslated.
- `struct` (nested-field filtering, one level) — a comparison against
  `pc.field(("parent", "child"))` becomes a `struct` filter on `parent` with
  a nested `child_filter`.
- `expression` (bounded function-call fallback) — for predicates the
  structured types above can't express (arithmetic, string functions, a
  spatial operator like `&&`). This filter type is fundamentally
  **single-column**: worker-side `ColumnRefNode.to_sql()`
  (`vgi/table_filter_pushdown.py:664-689`) always renders the filter's one
  `column_name` for every column reference inside the tree, regardless of
  index — so a subtree referencing more than one distinct column is declined,
  not silently mistranslated. Function calls with no pyarrow compute
  *options* block (arithmetic, most binary operators) translate generically;
  a small allowlist (`_OPTION_ARG_EXTRACTORS`) additionally handles the common
  options-bearing string-matching functions by flattening their option values
  into extra constant children in the position DuckDB's equivalent SQL
  function expects. Anything else with an unrecognized options block is
  declined.
- `join_keys` — **not** produced by this module. It needs a materialized key
  set from a build-side join, not a static predicate; see `_join.py`'s
  `vgi_semi_join_scan()`, which builds that spec type directly.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["build_join_keys_filter_bytes", "translate_predicate"]

_FILTER_VERSION = "1"

_COMPARE_OP = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.Lt: "lt",
    ast.LtE: "le",
}

# Matches: is_in(col, {value_set=TYPE:[v1, v2, ...], null_matching_behavior=X})
_ISIN_RE = re.compile(
    r"is_in\((\w+),\s*\{value_set=(\w+):\[(.*?)\],\s*null_matching_behavior=(\w+)\}\)",
    re.DOTALL,
)
# Matches: is_null(col, {nan_is_null=false}) -> is_null(col)
_ISNULL_OPTS_RE = re.compile(r"is_null\((\w+),\s*\{[^}]*\}\)")
# Matches: FieldRef.Nested(FieldRef.Name(parent) FieldRef.Name(child)) -> a placeholder
_NESTED_FIELD_RE = re.compile(r"FieldRef\.Nested\(FieldRef\.Name\((\w+)\)\s+FieldRef\.Name\((\w+)\)\)")
# Catches any options-bearing call this module doesn't specifically handle
# (e.g. cast(s, {to_type=int64, ...})) — run LAST, after the specific
# is_in/is_null/starts_with-etc. rewrites above have already consumed their
# own matches. Anything still matching this becomes an opaque placeholder:
# valid-Python (so it doesn't break ast.parse for the REST of the expression,
# including sibling AND-conjuncts that have nothing to do with it), but
# immediately untranslatable wherever it's referenced. Without this, one
# conjunct containing a truly unmapped options-bearing function would raise
# SyntaxError for the entire top-level ast.parse, silently dropping pushdown
# for every OTHER, perfectly translatable, sibling conjunct too.
_OPAQUE_OPTIONS_CALL_RE = re.compile(r"\w+\([^{}()]*,\s*\{[^{}]*\}\)")

# pyarrow's default is_in() null-matching behavior — the only value worker-side
# InFilter.evaluate (bare pc.is_in(col, vals), no options) matches exactly.
_ISIN_DEFAULT_NULL_MATCHING = "MATCH"

# Options-bearing functions this module knows how to flatten into extra
# constant children, `{pyarrow_function_name: [option_key, ...]}` in the
# order DuckDB's equivalent SQL function expects them positionally after the
# existing (non-option) children. Anything else with an options block is
# declined — see module docstring.
_OPTION_ARG_EXTRACTORS: dict[str, list[str]] = {
    "starts_with": ["pattern"],
    "ends_with": ["pattern"],
    "match_substring": ["pattern"],
}
# DuckDB SQL function name for a pyarrow function name, when it differs.
_FUNCTION_NAME_MAP: dict[str, str] = {
    "match_substring": "contains",
}
_OPTIONS_BLOCK_RE = re.compile(r"\{([^{}]*)\}")
_OPTION_KV_RE = re.compile(r"(\w+)=((?:\"[^\"]*\")|(?:[^,}]+))")


class UntranslatableExpression(Exception):  # noqa: N818 - internal control-flow signal, not a public error
    """Raised internally when a (sub)expression can't be translated. Never escapes this module."""


@dataclass
class _ValueColumns:
    """Accumulates the `value_N` columns a filter/expression spec's `value_ref` points at."""

    arrays: list[pa.Array[Any]] = field(default_factory=list)

    def add_scalar(self, value: Any) -> int:
        """Register a single literal, returning its `value_ref` index."""
        self.arrays.append(pa.array([value]))
        return len(self.arrays) - 1

    def add_list(self, values: pa.Array[Any]) -> int:
        """Register an IN-list literal, returning its `value_ref` index."""
        self.arrays.append(pa.array([values.to_pylist()], type=pa.list_(values.type)))
        return len(self.arrays) - 1


def _extract_options(text: str, keys: list[str]) -> list[Any] | None:
    """Pull `keys`' values out of a `{k=v, ...}` options block, in order.

    Returns `None` if any key is missing (declined, not guessed).
    """
    m = _OPTIONS_BLOCK_RE.search(text)
    if m is None:
        return None
    opts = dict(_OPTION_KV_RE.findall(m.group(1)))
    if not all(k in opts for k in keys):
        return None
    values = []
    for k in keys:
        raw = opts[k]
        try:
            values.append(ast.literal_eval(raw))
        except (ValueError, SyntaxError):
            return None
    return values


def _preprocess(
    text: str,
) -> tuple[str, dict[str, tuple[str, str, list[Any], str]], dict[str, tuple[str, str]], set[str]]:
    """Rewrite pyarrow's non-Python pretty-printing into something `ast.parse` can read.

    Returns `(rewritten_text, isin_placeholders, nested_field_placeholders, opaque_placeholders)`:
      - `isin_placeholders`: {name: (column, arrow_type_name, python_values, null_matching_behavior)}
      - `nested_field_placeholders`: {name: (parent_column, child_field)}
      - `opaque_placeholders`: names standing in for an unrecognized
        options-bearing call — immediately untranslatable wherever referenced
        (see `_OPAQUE_OPTIONS_CALL_RE`'s comment).
    """
    isin_placeholders: dict[str, tuple[str, str, list[Any], str]] = {}
    nested_placeholders: dict[str, tuple[str, str]] = {}
    opaque_placeholders: set[str] = set()
    counter = 0

    def isin_repl(m: re.Match[str]) -> str:
        nonlocal counter
        col, type_name, items_text, null_matching = m.group(1), m.group(2), m.group(3), m.group(4)
        items = [ast.literal_eval(tok.strip()) for tok in items_text.split(",") if tok.strip()]
        name = f"__isin_{counter}__"
        counter += 1
        isin_placeholders[name] = (col, type_name, items, null_matching)
        return name

    def nested_repl(m: re.Match[str]) -> str:
        nonlocal counter
        parent, child = m.group(1), m.group(2)
        name = f"__nested_{counter}__"
        counter += 1
        nested_placeholders[name] = (parent, child)
        return name

    def opaque_repl(m: re.Match[str]) -> str:
        nonlocal counter
        name = f"__opaque_{counter}__"
        counter += 1
        opaque_placeholders.add(name)
        return name

    text = _ISIN_RE.sub(isin_repl, text)
    text = _NESTED_FIELD_RE.sub(nested_repl, text)
    text = _ISNULL_OPTS_RE.sub(r"is_null(\1)", text)
    text = _OPAQUE_OPTIONS_CALL_RE.sub(opaque_repl, text)
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\bnull\b", "None", text)
    return text, isin_placeholders, nested_placeholders, opaque_placeholders


# ---------------------------------------------------------------------------
# Structured translation (constant / is_null / is_not_null / in / and / or / struct)
# ---------------------------------------------------------------------------


def _walk(
    node: ast.AST,
    schema: pa.Schema,
    isin_ph: dict[str, tuple[str, str, list[Any], str]],
    nested_ph: dict[str, tuple[str, str]],
    opaque_ph: set[str],
    values: _ValueColumns,
) -> dict[str, Any]:
    """Convert one ast node (from the preprocessed repr) to a structured filter spec dict.

    Raises `UntranslatableExpression` for anything this structured path can't
    express — the caller falls back to `_walk_as_expression_node` for a
    bounded, single-column `expression`-type translation instead.
    """
    if isinstance(node, ast.BoolOp):
        children = [_walk(v, schema, isin_ph, nested_ph, opaque_ph, values) for v in node.values]
        anchor = children[0]["column_name"]
        kind = "and" if isinstance(node.op, ast.And) else "or"
        return {
            "type": kind,
            "column_name": anchor,
            "column_index": schema.get_field_index(anchor),
            "children": children,
        }

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise UntranslatableExpression("chained comparison")
        left, right = node.left, node.comparators[0]
        op_type = type(node.ops[0])
        if op_type not in _COMPARE_OP:
            raise UntranslatableExpression(f"unsupported comparison operator {op_type}")

        if isinstance(left, ast.Name) and left.id in opaque_ph:
            raise UntranslatableExpression("unrecognized options-bearing call as comparison LHS")

        # struct: nested-field placeholder on either side of the comparison.
        if isinstance(left, ast.Name) and left.id in nested_ph:
            parent, child = nested_ph[left.id]
            if not isinstance(right, ast.Constant):
                raise UntranslatableExpression("struct comparison RHS must be a literal")
            ref = values.add_scalar(right.value)
            child_filter = {
                "type": "constant",
                "column_name": child,
                "column_index": 0,
                "op": _COMPARE_OP[op_type],
                "value_ref": ref,
            }
            struct_type = schema.field(parent).type
            return {
                "type": "struct",
                "column_name": parent,
                "column_index": schema.get_field_index(parent),
                "child_index": struct_type.get_field_index(child),
                "child_name": child,
                "child_filter": child_filter,
            }

        if not isinstance(left, ast.Name):
            raise UntranslatableExpression(f"comparison LHS must be a bare column: {ast.dump(left)}")
        if not isinstance(right, ast.Constant):
            raise UntranslatableExpression(f"comparison RHS must be a literal: {ast.dump(right)}")
        ref = values.add_scalar(right.value)
        return {
            "type": "constant",
            "column_name": left.id,
            "column_index": schema.get_field_index(left.id),
            "op": _COMPARE_OP[op_type],
            "value_ref": ref,
        }

    if isinstance(node, ast.Call):
        fname = node.func.id  # type: ignore[attr-defined]
        if fname == "is_valid":
            col = node.args[0].id  # type: ignore[attr-defined]
            return {"type": "is_not_null", "column_name": col, "column_index": schema.get_field_index(col)}
        if fname == "is_null":
            col = node.args[0].id  # type: ignore[attr-defined]
            return {"type": "is_null", "column_name": col, "column_index": schema.get_field_index(col)}
        if fname == "invert":
            inner = _walk(node.args[0], schema, isin_ph, nested_ph, opaque_ph, values)
            if inner["type"] != "is_null":
                raise UntranslatableExpression("invert() is only supported wrapping is_null()")
            inner["type"] = "is_not_null"
            return inner
        raise UntranslatableExpression(f"unsupported function in structured position: {fname}")

    if isinstance(node, ast.Name) and node.id in isin_ph:
        col, type_name, items, null_matching = isin_ph[node.id]
        if null_matching != _ISIN_DEFAULT_NULL_MATCHING:
            raise UntranslatableExpression(f"is_in with non-default null_matching_behavior={null_matching!r}")
        # pyarrow-stubs only models `type_for_alias` against a closed set of
        # Literal names — `type_name` here is genuinely dynamic (decoded from
        # the expression's own repr at runtime), so no static Literal fits.
        arrow_type = pa.type_for_alias(type_name)  # type: ignore[call-overload]
        ref = values.add_list(pa.array(items, type=arrow_type))
        return {"type": "in", "column_name": col, "column_index": schema.get_field_index(col), "value_ref": ref}

    raise UntranslatableExpression(f"unsupported expression node: {ast.dump(node)}")


# ---------------------------------------------------------------------------
# `expression`-type fallback (bounded function-call translation, single column)
# ---------------------------------------------------------------------------


def _walk_as_expression_node(
    node: ast.AST,
    isin_ph: dict[str, Any],
    nested_ph: dict[str, Any],
    opaque_ph: set[str],
    values: _ValueColumns,
) -> tuple[dict[str, Any], set[str]]:
    """Convert one ast node into an `expr_type` node for the `expression` filter type.

    Returns `(node_json, referenced_columns)` — the caller enforces the
    single-column constraint (`ColumnRefNode.to_sql` always renders the
    filter's one `column_name`, see module docstring) once at the top level.
    Raises `UntranslatableExpression` for anything not expressible this way.
    """
    if isinstance(node, ast.Name):
        if node.id in isin_ph or node.id in nested_ph:
            raise UntranslatableExpression("is_in/struct placeholders unsupported inside expression fallback")
        if node.id in opaque_ph:
            raise UntranslatableExpression("unrecognized options-bearing call inside expression fallback")
        return {"expr_type": "column_ref", "index": 0}, {node.id}

    if isinstance(node, ast.Constant):
        ref = values.add_scalar(node.value)
        return {"expr_type": "constant", "value_ref": ref}, set()

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1:
            raise UntranslatableExpression("chained comparison")
        op_type = type(node.ops[0])
        if op_type not in _COMPARE_OP:
            raise UntranslatableExpression(f"unsupported comparison operator {op_type}")
        left_node, left_cols = _walk_as_expression_node(node.left, isin_ph, nested_ph, opaque_ph, values)
        right_node, right_cols = _walk_as_expression_node(node.comparators[0], isin_ph, nested_ph, opaque_ph, values)
        return (
            {"expr_type": "comparison", "op": _COMPARE_OP[op_type], "left": left_node, "right": right_node},
            left_cols | right_cols,
        )

    if isinstance(node, ast.BoolOp):
        children_cols: set[str] = set()
        children = []
        for v in node.values:
            child_node, child_cols = _walk_as_expression_node(v, isin_ph, nested_ph, opaque_ph, values)
            children.append(child_node)
            children_cols |= child_cols
        kind = "and" if isinstance(node.op, ast.And) else "or"
        return {"expr_type": "conjunction", "conjunction_type": kind, "children": children}, children_cols

    if isinstance(node, ast.Call):
        fname = node.func.id  # type: ignore[attr-defined]
        sql_name = _FUNCTION_NAME_MAP.get(fname, fname)
        # Options-bearing functions (e.g. starts_with's `pattern=`) already had
        # their option values flattened into extra plain-Constant positional
        # args by `_rewrite_option_calls`, run once over the whole expression
        # before `ast.parse` — nothing left to do with options here.
        children_cols = set()
        children = []
        for a in node.args:
            child_node, child_cols = _walk_as_expression_node(a, isin_ph, nested_ph, opaque_ph, values)
            children.append(child_node)
            children_cols |= child_cols
        return {"expr_type": "function", "function_name": sql_name, "children": children}, children_cols

    raise UntranslatableExpression(f"unsupported expression-fallback node: {ast.dump(node)}")


def _rewrite_option_calls(text: str) -> str:
    """Flatten known options-bearing function calls into plain positional args.

    `starts_with(s, {pattern="row_", ignore_case=false})` ->
    `starts_with(s, "row_")` — only for functions in `_OPTION_ARG_EXTRACTORS`;
    anything else keeps its options block, which then fails `ast.parse` and
    is caught as untranslatable (declined), never guessed.
    """

    def repl(m: re.Match[str]) -> str:
        fname = m.group("fname")
        keys = _OPTION_ARG_EXTRACTORS.get(fname)
        if keys is None:
            return m.group(0)
        values = _extract_options(m.group("opts"), keys)
        if values is None:
            return m.group(0)
        extra = ", ".join(repr(v) for v in values)
        return f"{fname}({m.group('args')}, {extra})"

    pattern = re.compile(r"(?P<fname>\w+)\((?P<args>\w+),\s*(?P<opts>\{[^}]*\})\)")
    return pattern.sub(repl, text)


def _try_structured(
    node: ast.AST,
    schema: pa.Schema,
    isin_ph: dict[str, tuple[str, str, list[Any], str]],
    nested_ph: dict[str, tuple[str, str]],
    opaque_ph: set[str],
    values: _ValueColumns,
) -> dict[str, Any] | None:
    """Attempt the structured (constant/is_null/in/and/or/struct) translation for one conjunct."""
    try:
        return _walk(node, schema, isin_ph, nested_ph, opaque_ph, values)
    except UntranslatableExpression:
        return None


def _try_expression_fallback(
    node: ast.AST,
    schema: pa.Schema,
    isin_ph: dict[str, tuple[str, str, list[Any], str]],
    nested_ph: dict[str, tuple[str, str]],
    opaque_ph: set[str],
    values: _ValueColumns,
) -> dict[str, Any] | None:
    """Attempt the bounded, single-column `expression`-type fallback for one conjunct."""
    try:
        expr_node, columns = _walk_as_expression_node(node, isin_ph, nested_ph, opaque_ph, values)
    except UntranslatableExpression:
        return None
    if len(columns) != 1:
        return None
    (col,) = columns
    return {
        "type": "expression",
        "column_name": col,
        "column_index": schema.get_field_index(col),
        "expr": expr_node,
    }


def _flatten_and(node: ast.AST) -> list[ast.AST]:
    """Flatten a top-level chain of `and` into its conjuncts; anything else is one conjunct."""
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        out: list[ast.AST] = []
        for v in node.values:
            out.extend(_flatten_and(v))
        return out
    return [node]


def translate_predicate(
    expr: pc.Expression, schema: pa.Schema, *, allow_expression_fallback: bool = True
) -> bytes | None:
    """Best-effort translate `expr` into VGI `pushdown_filters` IPC bytes.

    Args:
        expr: A pyarrow filter `Expression`, e.g. `pc.field("n") >= 8` — the
            same object type Acero's `ScanNodeOptions`/`FilterNodeOptions`
            already accept as their own `filter=` argument.
        schema: The (unprojected) output schema of the VGI function being
            scanned — used to resolve `column_index` for referenced columns.
        allow_expression_fallback: Whether to attempt the bounded
            `expression`-type fallback (module docstring) for a conjunct the
            structured types can't express. `VgiAceroTable.scan()` passes
            `False` unless the resolved `FunctionInfo.supported_expression_filters`
            is non-empty — a worker that declared no expression-filter classes
            at all almost certainly can't evaluate one (see
            `vgi/table_filter_pushdown.py`'s `ExpressionFilter`, which needs a
            DuckDB-compatible engine worker-side). Callers with no
            `FunctionInfo` to check (the bare-function path, `vgi_scan()`)
            default to `True` — best effort, no capability signal available.

    Returns:
        Arrow-IPC-serialized bytes suitable for `Client.table_function(...,
        pushdown_filters=...)`, or `None` if nothing in `expr` could be
        translated (never raises — an unsupported shape is just a pushdown
        miss, always safe given the caller's own local filter node).

    """
    repr_text = repr(expr).removeprefix("<pyarrow.compute.Expression ").removesuffix(">")
    # Single preprocessing pass over the WHOLE expression: placeholder names
    # (for is_in / nested-field blocks) are unique across the expression, so
    # every conjunct's ast subtree can share one isin_ph/nested_ph mapping —
    # no need to re-preprocess (or unparse+reparse) per conjunct.
    rewritten = _rewrite_option_calls(repr_text)
    text, isin_ph, nested_ph, opaque_ph = _preprocess(rewritten)
    try:
        top_tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None

    values = _ValueColumns()
    specs: list[dict[str, Any]] = []

    for conjunct_node in _flatten_and(top_tree):
        spec = _try_structured(conjunct_node, schema, isin_ph, nested_ph, opaque_ph, values)
        if spec is None and allow_expression_fallback:
            spec = _try_expression_fallback(conjunct_node, schema, isin_ph, nested_ph, opaque_ph, values)
        if spec is not None:
            specs.append(spec)
        # else: untranslatable conjunct — silently skipped, see module docstring.

    if not specs:
        return None

    spec_field = pa.field("filter_spec", pa.string(), metadata={b"vgi_filter_version": _FILTER_VERSION.encode()})
    fields = [spec_field, *(pa.field(f"value_{i}", arr.type) for i, arr in enumerate(values.arrays))]
    batch = pa.RecordBatch.from_arrays(
        [pa.array([json.dumps(specs)]), *values.arrays],
        schema=pa.schema(fields),
    )

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def build_join_keys_filter_bytes(*, column_name: str, column_index: int, keys_column: str) -> bytes:
    """Build `pushdown_filters` IPC bytes containing one `join_keys` filter spec.

    Unlike every other filter type this module produces, a `join_keys` spec
    carries no `value_ref` — the actual key values travel separately as
    `Client.table_function`'s own `join_keys=` argument (see `_join.py`),
    resolved worker-side by matching `keys_column` against each join-key
    batch's column name (`PushdownFilters.get_join_keys_column`). This
    function builds only the spec that tells the worker to look them up —
    call it alongside passing the actual key batches, not instead of.

    Args:
        column_name: The probe-side column name this filter applies to.
        column_index: That column's index in the probe function's output schema.
        keys_column: The column name to look up within the `join_keys` batches.

    Returns:
        Arrow-IPC-serialized `pushdown_filters` bytes for `Client.table_function`.

    """
    spec = {
        "column_name": column_name,
        "column_index": column_index,
        "type": "join_keys",
        "keys_column": keys_column,
    }
    spec_field = pa.field("filter_spec", pa.string(), metadata={b"vgi_filter_version": _FILTER_VERSION.encode()})
    batch = pa.record_batch({"filter_spec": [json.dumps([spec])]}, schema=pa.schema([spec_field]))

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue().to_pybytes()
