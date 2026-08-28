# Copyright 2026 Query Farm LLC - https://query.farm

"""Public error type for vgi-acero.

`vgi.client.ClientError` (from vgi-python) is an implementation detail of how
vgi-acero talks to a worker, not part of this package's public surface — every
call site that can raise it catches it and re-raises `VgiAceroError` instead,
preserving the original message.
"""

from __future__ import annotations

import re

from vgi.client.catalog_mixin import CatalogClientError
from vgi.client.client import ClientError

__all__ = ["VGI_CLIENT_ERRORS", "VgiAceroError", "wrap_error"]


class VgiAceroError(Exception):
    """Raised for any VGI catalog/scan/function-call failure surfaced to a caller."""


# Confirmed live against vgi-fixture-worker: an upstream vgi-rpc/vgi-python
# formatting quirk can double-prefix a remote error's type name -- e.g.
# str(e) == "ValueError: ValueError: No worker handles catalog 'x'" instead of
# a single "ValueError: ...". Root cause is in vgi-rpc's worker-side
# exception-to-log-message serialization (outside both vgi-python and this
# package), but a doubled prefix is what a vgi-acero caller actually sees, so
# it's worth collapsing defensively here rather than only upstream.
_DOUBLED_PREFIX_RE = re.compile(r"^(\w+): \1: ")


def wrap_error(e: Exception) -> VgiAceroError:
    """Re-raise a caught `VGI_CLIENT_ERRORS` exception as a `VgiAceroError`.

    Use as `raise wrap_error(e) from e` at every call site that catches
    `VGI_CLIENT_ERRORS` — collapses the doubled-prefix quirk described above
    when present, otherwise preserves the original message unchanged.
    """
    message = _DOUBLED_PREFIX_RE.sub(r"\1: ", str(e))
    return VgiAceroError(message)


#: vgi-python raises two unrelated exception types depending on call path:
#: `ClientError` (table/scalar function exchange RPCs, and `Client.bind()`) and
#: `CatalogClientError` (everything routed through
#: `CatalogClientMixin._catalog_connect()` — attach, schemas, table_get,
#: table_scan_function_get, schema_contents, ...). Neither is a subclass of the
#: other, so every vgi-acero call site catches both and re-raises `VgiAceroError`.
VGI_CLIENT_ERRORS = (ClientError, CatalogClientError)
