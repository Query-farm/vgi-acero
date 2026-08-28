# Copyright 2026 Query Farm LLC - https://query.farm

"""vgi-acero — a `pyarrow.acero` connector for VGI, no DuckDB required.

An **adapter** over vgi-python's existing `vgi.client.Client`, not a new VGI
protocol implementation — see the package README's "What this is" section.

Public entry points:
    attach(...)              -- attach to a VGI catalog, returns a VgiAceroCatalog
    VgiAceroCatalog           -- an attached catalog (schemas/tables/scalar functions)
    VgiAceroTable             -- a lazy handle to one catalog table
    vgi_scan(...)             -- scan a bare (non-catalog) table function
    vgi_scan_splits(...)      -- split-planned, concurrently-pulled parallel scan
    vgi_semi_join_scan(...)   -- push build-side join keys down to a probe-side scan
    vgi_topn_scan(...)        -- adaptive Top-N re-query (see _topn.py for why)
    make_vgi_scan_declaration(...) -- wrap a raw VGI batch generator as a Declaration
    VgiAceroError             -- this package's public exception type
"""

from __future__ import annotations

from vgi_acero._join import vgi_semi_join_scan
from vgi_acero._scan import make_vgi_scan_declaration, vgi_scan, vgi_scan_splits
from vgi_acero._topn import vgi_topn_scan
from vgi_acero.catalog import VgiAceroCatalog, attach
from vgi_acero.errors import VgiAceroError
from vgi_acero.table import VgiAceroTable

__all__ = [
    "VgiAceroCatalog",
    "VgiAceroError",
    "VgiAceroTable",
    "attach",
    "make_vgi_scan_declaration",
    "vgi_scan",
    "vgi_scan_splits",
    "vgi_semi_join_scan",
    "vgi_topn_scan",
]
