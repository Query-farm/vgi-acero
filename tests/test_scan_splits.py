# Copyright 2026 Query Farm LLC - https://query.farm

# ruff: noqa: S101, D101, D102, D103
"""`vgi_scan_splits()` — split-planned, concurrently-pulled Acero union scan.

Against `split_sequence` (vgi-python's own split-capable twin of `sequence`,
`vgi/_test_fixtures/table/splits.py`), row-for-row identical to `sequence(n)`.
"""

from __future__ import annotations

import pyarrow as pa
from vgi.arguments import Arguments
from vgi.client.client import Client

from vgi_acero import vgi_scan_splits

MAIN = "main"


def test_split_scan_reproduces_whole_scan(worker_location: str) -> None:
    args = Arguments(named={"n": pa.scalar(37), "splits": pa.scalar(5)})

    def client_factory() -> Client:
        c = Client(worker_location)
        c.start()
        return c

    result = vgi_scan_splits(client_factory, schema_name=MAIN, function_name="split_sequence", arguments=args)
    try:
        table = result.declaration.to_table()
        assert sorted(table.column("n").to_pylist()) == list(range(37))
    finally:
        result.close()


def test_zero_splits_yields_empty_result(worker_location: str) -> None:
    args = Arguments(named={"n": pa.scalar(10), "splits": pa.scalar(4)})

    def client_factory() -> Client:
        c = Client(worker_location)
        c.start()
        return c

    result = vgi_scan_splits(client_factory, schema_name=MAIN, function_name="split_zero", arguments=args)
    try:
        table = result.declaration.to_table()
        assert table.num_rows == 0
        # split_zero produces zero splits -- nothing for the caller to close.
        assert result.clients == []
    finally:
        result.close()


def test_split_scan_closes_all_clients_it_opened(worker_location: str) -> None:
    """Every per-split Client SplitScanResult reports must actually stop cleanly."""
    args = Arguments(named={"n": pa.scalar(20), "splits": pa.scalar(4)})

    def client_factory() -> Client:
        c = Client(worker_location)
        c.start()
        return c

    result = vgi_scan_splits(client_factory, schema_name=MAIN, function_name="split_sequence", arguments=args)
    result.declaration.to_table()
    assert len(result.clients) == 4
    result.close()  # must not raise
