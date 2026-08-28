# Copyright 2026 Query Farm LLC - https://query.farm

"""Shared fixtures for vgi-acero's test suite.

Drives `vgi-fixture-worker` (vgi-python's reference test/example worker,
catalog `example`) over the subprocess transport. Mirrors
`vgi-polars/tests/conftest.py`'s `worker_location` fixture, including its
deliberate deviation from every other repo in this family's `uv run --project
...` convention — see `worker_location`'s docstring.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from vgi.client.client import Client

import vgi_acero as va


def _vgi_python_venv() -> Path:
    override = os.environ.get("VGI_PYTHON")
    root = Path(override) if override else Path.home() / "Development" / "vgi-python"
    return root / ".venv" / "bin"


@pytest.fixture(scope="session")
def worker_location() -> str:
    """The `example` catalog's worker command.

    Deliberately NOT `uv run --project <vgi-python> vgi-fixture-worker` — `uv
    run` re-resolves/verifies the project environment on every invocation,
    which is slow under concurrent `uv` activity elsewhere on the machine.
    Points directly at vgi-python's own venv console script instead.
    """
    override = os.environ.get("VGI_TEST_WORKER")
    if override:
        return override
    return str(_vgi_python_venv() / "vgi-fixture-worker")


@pytest.fixture
def catalog(worker_location: str):
    """A fresh `VgiAceroCatalog` attached to the `example` catalog over subprocess."""
    with va.attach(worker_location, name="example") as cat:
        yield cat


@pytest.fixture
def client(worker_location: str):
    """A fresh, started, bare `vgi.client.Client` (no catalog attach) against `vgi-fixture-worker`."""
    with Client(worker_location) as c:
        yield c
