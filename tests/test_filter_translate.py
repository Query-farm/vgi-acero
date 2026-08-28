# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""Pure unit tests for `_filter_translate.translate_predicate` — no worker involved.

Each test decodes the produced IPC bytes back to the JSON filter-spec list
and asserts on its shape directly, matching what `vgi/table_filter_pushdown.py`'s
`deserialize_filters`/`_parse_filter` actually expect (confirmed against the
real worker in `test_scan_filter.py`; these tests isolate translation
correctness from a subprocess round trip).
"""

from __future__ import annotations

import json
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds

from vgi_acero._filter_translate import build_join_keys_filter_bytes, translate_predicate

SCHEMA = pa.schema({"n": pa.int64(), "s": pa.utf8(), "flag": pa.bool_(), "addr": pa.struct({"city": pa.utf8()})})


def _decode(pushdown_bytes: bytes | None) -> list[dict[str, Any]]:
    assert pushdown_bytes is not None
    reader = pa.ipc.open_stream(pa.BufferReader(pushdown_bytes))
    batch = reader.read_next_batch()
    metadata = batch.schema.field(0).metadata
    assert metadata[b"vgi_filter_version"] == b"1"
    specs: list[dict[str, Any]] = json.loads(batch.column(0)[0].as_py())
    return specs


class TestConstant:
    def test_each_comparison_operator(self) -> None:
        cases = [
            (ds.field("n") == 5, "eq"),
            (ds.field("n") != 5, "ne"),
            (ds.field("n") > 5, "gt"),
            (ds.field("n") >= 5, "ge"),
            (ds.field("n") < 5, "lt"),
            (ds.field("n") <= 5, "le"),
        ]
        for expr, op in cases:
            specs = _decode(translate_predicate(expr, SCHEMA))
            assert len(specs) == 1
            assert specs[0] == {"column_name": "n", "column_index": 0, "type": "constant", "op": op, "value_ref": 0}

    def test_string_literal(self) -> None:
        specs = _decode(translate_predicate(ds.field("s") == "hello", SCHEMA))
        assert specs[0]["type"] == "constant"
        assert specs[0]["column_name"] == "s"

    def test_boolean_literal(self) -> None:
        specs = _decode(translate_predicate(ds.field("flag") == True, SCHEMA))  # noqa: E712
        assert specs[0]["type"] == "constant"
        assert specs[0]["op"] == "eq"


class TestNullChecks:
    def test_is_null(self) -> None:
        specs = _decode(translate_predicate(ds.field("s").is_null(), SCHEMA))
        assert specs == [{"column_name": "s", "column_index": 1, "type": "is_null"}]

    def test_is_valid_maps_to_is_not_null(self) -> None:
        specs = _decode(translate_predicate(ds.field("s").is_valid(), SCHEMA))
        assert specs == [{"column_name": "s", "column_index": 1, "type": "is_not_null"}]

    def test_invert_is_null_maps_to_is_not_null(self) -> None:
        specs = _decode(translate_predicate(~ds.field("s").is_null(), SCHEMA))
        assert specs == [{"column_name": "s", "column_index": 1, "type": "is_not_null"}]


class TestIn:
    def test_default_null_matching_translates(self) -> None:
        specs = _decode(translate_predicate(ds.field("n").isin([1, 2, 3]), SCHEMA))
        assert len(specs) == 1
        assert specs[0]["type"] == "in"
        assert specs[0]["column_name"] == "n"

    def test_string_value_containing_a_literal_comma(self) -> None:
        """Regression: a naive split(",") on the value list tears a comma-containing string apart."""
        specs = _decode(translate_predicate(ds.field("s").isin(["foo,bar", "baz"]), SCHEMA))
        assert len(specs) == 1
        assert specs[0]["type"] == "in"

    def test_large_list_beyond_pyarrows_pretty_print_truncation_declines_gracefully(self) -> None:
        """Regression: pyarrow truncates a long is_in() list with a bare literal "..." line.

        Confirmed live (pyarrow 25.0.1): ast.literal_eval chokes on it. This
        must decline the conjunct, never raise out of translate_predicate.
        """
        big_list = list(range(50))
        assert translate_predicate(ds.field("n").isin(big_list), SCHEMA) is None
        # Same, but as one conjunct of an AND chain -- the translatable sibling must still push.
        expr = (ds.field("n") >= 3) & ds.field("n").isin(big_list)
        specs = _decode(translate_predicate(expr, SCHEMA))
        assert len(specs) == 1
        assert specs[0]["op"] == "ge"

    def test_non_default_null_matching_behavior_is_declined(self) -> None:
        # `null_matching_behavior=MATCH` is the only value this module
        # translates — this test is the regression guard for that decision
        # (see _filter_translate.py's module docstring): confirm pyarrow's
        # own default repr is indeed MATCH, and a non-default request via
        # SetLookupOptions is left untranslated rather than mistranslated.
        import pyarrow.compute as pc

        default_expr = ds.field("n").isin([1, 2, 3])
        assert "null_matching_behavior=MATCH" in repr(default_expr)

        non_default = pc.Expression._call(  # noqa: SLF001 - only way to force a non-default option from Python
            "is_in", [ds.field("n")], options=pc.SetLookupOptions(value_set=pa.array([1, 2, 3]), skip_nulls=True)
        )
        assert translate_predicate(non_default, SCHEMA) is None


class TestConjunction:
    def test_and(self) -> None:
        expr = (ds.field("n") >= 3) & (ds.field("n") <= 6)
        specs = _decode(translate_predicate(expr, SCHEMA))
        # Top-level AND is flattened into two top-level specs (both ANDed
        # implicitly by PushdownFilters), not one nested AndFilter — matches
        # translate_predicate's _flatten_and behavior.
        assert len(specs) == 2
        assert {s["op"] for s in specs} == {"ge", "le"}

    def test_or(self) -> None:
        expr = (ds.field("n") == 1) | (ds.field("n") == 2)
        specs = _decode(translate_predicate(expr, SCHEMA))
        assert len(specs) == 1
        assert specs[0]["type"] == "or"
        assert len(specs[0]["children"]) == 2
        assert {c["op"] for c in specs[0]["children"]} == {"eq"}


class TestStruct:
    def test_nested_field_comparison(self) -> None:
        import pyarrow.compute as pc

        expr = pc.field(("addr", "city")) == "Seattle"
        specs = _decode(translate_predicate(expr, SCHEMA))
        assert len(specs) == 1
        assert specs[0]["type"] == "struct"
        assert specs[0]["column_name"] == "addr"
        assert specs[0]["child_name"] == "city"
        assert specs[0]["child_filter"]["type"] == "constant"
        assert specs[0]["child_filter"]["op"] == "eq"


class TestExpressionFallback:
    def test_arithmetic_function_falls_back_to_expression_type(self) -> None:
        import pyarrow.compute as pc

        expr = (pc.field("n") + 1) > 5
        specs = _decode(translate_predicate(expr, SCHEMA))
        assert len(specs) == 1
        assert specs[0]["type"] == "expression"
        assert specs[0]["column_name"] == "n"
        assert specs[0]["expr"]["expr_type"] == "comparison"

    def test_multi_column_expression_is_declined(self) -> None:
        import pyarrow.compute as pc

        expr = pc.field("n") > pc.field("flag").cast(pa.int64())
        assert translate_predicate(expr, SCHEMA) is None


class TestUnsupported:
    def test_untranslatable_predicate_returns_none(self) -> None:
        import pyarrow.compute as pc

        # A function this module's allowlist doesn't recognize at all, with
        # no bare-column single-reference shape either.
        expr = pc.field("s").cast(pa.int64()) > pc.field("n")
        assert translate_predicate(expr, SCHEMA) is None

    def test_mixed_translatable_and_untranslatable_and_conjuncts(self) -> None:
        """One translatable conjunct in an AND chain still gets pushed, even if a sibling can't be."""
        import pyarrow.compute as pc

        expr = (ds.field("n") >= 3) & (pc.field("s").cast(pa.int64()) > pc.field("n"))
        specs = _decode(translate_predicate(expr, SCHEMA))
        assert len(specs) == 1
        assert specs[0]["op"] == "ge"

    def test_options_bearing_call_wrapping_another_calls_result_does_not_poison_siblings(self) -> None:
        """Regression: an options-bearing function applied to a NESTED call's result.

        `match_substring(binary_join_element_wise(a, b, "-"), "foo")` has a
        parenthesized function call inside its own argument list, ahead of
        its `{...}` options block -- _OPAQUE_OPTIONS_CALL_RE must still
        recognize and shield it (one level of nesting), or its unrewritten
        `{...}` block breaks ast.parse for the whole expression, silently
        dropping the perfectly-translatable `n >= 3` sibling too.
        """
        import pyarrow.compute as pc

        nested = pc.match_substring(pc.binary_join_element_wise(ds.field("s"), ds.field("s"), "-"), "foo")
        expr = (ds.field("n") >= 3) & nested
        specs = _decode(translate_predicate(expr, SCHEMA))
        assert len(specs) == 1
        assert specs[0]["op"] == "ge"


class TestExpressionFallbackGating:
    def test_disabled_fallback_drops_the_conjunct(self) -> None:
        import pyarrow.compute as pc

        expr = (pc.field("n") + 1) > 5
        assert translate_predicate(expr, SCHEMA, allow_expression_fallback=False) is None

    def test_disabled_fallback_still_pushes_translatable_siblings(self) -> None:
        import pyarrow.compute as pc

        expr = (ds.field("n") >= 3) & ((pc.field("n") + 1) > 5)
        specs = _decode(translate_predicate(expr, SCHEMA, allow_expression_fallback=False))
        assert len(specs) == 1
        assert specs[0]["op"] == "ge"


class TestJoinKeysBytes:
    def test_build_join_keys_filter_bytes_shape(self) -> None:
        specs = _decode(build_join_keys_filter_bytes(column_name="n", column_index=0, keys_column="n"))
        assert specs == [{"column_name": "n", "column_index": 0, "type": "join_keys", "keys_column": "n"}]
