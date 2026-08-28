# Copyright 2026 Query Farm LLC - https://query.farm

"""Public error type for vgi-acero.

`vgi.client.ClientError` (from vgi-python) is an implementation detail of how
vgi-acero talks to a worker, not part of this package's public surface — every
call site that can raise it catches it and re-raises `VgiAceroError` instead,
preserving the original message.
"""

from __future__ import annotations

from vgi.client.catalog_mixin import CatalogClientError
from vgi.client.client import ClientError

__all__ = ["VGI_CLIENT_ERRORS", "VgiAceroError"]


class VgiAceroError(Exception):
    """Raised for any VGI catalog/scan/function-call failure surfaced to a caller."""


#: vgi-python raises two unrelated exception types depending on call path:
#: `ClientError` (table/scalar function exchange RPCs, and `Client.bind()`) and
#: `CatalogClientError` (everything routed through
#: `CatalogClientMixin._catalog_connect()` — attach, schemas, table_get,
#: table_scan_function_get, schema_contents, ...). Neither is a subclass of the
#: other, so every vgi-acero call site catches both and re-raises `VgiAceroError`.
VGI_CLIENT_ERRORS = (ClientError, CatalogClientError)
