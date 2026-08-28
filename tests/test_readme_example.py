# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""Runs the README's own top-level example verbatim, against a real worker.

Regression guard for README/code drift — confirmed live during a DX review
that README.md's example previously named the wrong schema
(`catalog.table("main", "filter_echo_table")` instead of `"data"`) and would
have failed for the first person who pasted it. Mirrors the spirit of
vgi-python's own `tests/test_documentation_examples.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def test_readme_first_example_runs(worker_location: str) -> None:
    text = README.read_text()
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    assert match is not None, "README.md has no python code fence to test"
    code = match.group(1)
    # The README shows the friendly literal a reader would type if
    # `vgi-fixture-worker` were on PATH; substitute the resolved absolute
    # path so this test doesn't depend on PATH setup.
    code = code.replace('"vgi-fixture-worker"', repr(worker_location))
    exec(compile(code, str(README), "exec"), {"__name__": "__readme_example__"})  # noqa: S102
