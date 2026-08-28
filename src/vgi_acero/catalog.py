# Copyright 2026 Query Farm LLC - https://query.farm

"""`VgiAceroCatalog` — an attached VGI catalog, and the `attach()` entry point.

Wraps `vgi.client.Client` (vgi-python's pure-Python, Arrow-native reference
client — the same wire-protocol implementation the DuckDB extension speaks,
just without DuckDB). See this package's README for the architectural
rationale: vgi-acero is an adapter over that existing client, not a new
protocol implementation. Closely mirrors `vgi_polars.catalog.VgiCatalog` —
the underlying problems (schema/catalog discovery, thread safety) are
identical; only the terminal "returns a `pl.LazyFrame`" step differs, become
"returns an `ac.Declaration`" here.

**Thread safety.** `vgi.client.Client`'s exchange-mode methods (`table_function`,
`scalar_function`, and friends) drive shared mutable state (`self._primary` /
`self._additional_workers`) with no locking — confirmed unsafe for concurrent
use on one shared instance by vgi-polars' own direct stress test (20 threads
calling `scalar_function` concurrently on one `Client`: 18 corrupted/errored,
0 correct). This matters here too: Acero's default execution is
multi-threaded, a registered scalar UDF (`_scalar.py`) can be invoked from
several worker threads concurrently, and `vgi_scan_splits()` deliberately
pulls from several split scans concurrently via Acero's `union` node. So every
exchange-mode call site uses `VgiAceroCatalog.exchange_client()` (one
lazily-created `Client` per calling thread, all sharing the one
`attach_opaque_data` from the catalog's single `catalog_attach`), never the
catalog's single shared `Client` directly. Catalog-metadata methods (`schemas`,
`table_get`, `schema_contents`, `table_scan_function_get`,
`table_column_statistics`, `bind`) keep using the shared `Client` — `CatalogClientMixin`
opens a short-lived connection per call rather than reusing `self._primary`,
so they don't have this hazard.
"""

from __future__ import annotations

import contextlib
import threading
from typing import TYPE_CHECKING, Any, Literal, Self

from vgi.catalog.catalog_interface import CatalogAttachResult, SchemaObjectType
from vgi.client.client import Client

from vgi_acero.errors import VGI_CLIENT_ERRORS, VgiAceroError, wrap_error

if TYPE_CHECKING:
    from collections.abc import Callable

    from vgi.catalog.catalog_interface import AttachOpaqueData

    from vgi_acero._scalar import ScalarFunctionCall
    from vgi_acero.table import VgiAceroTable

__all__ = ["VgiAceroCatalog", "attach"]

Transport = Literal["subprocess", "http", "tcp"]


class VgiAceroCatalog:
    """An attached VGI catalog. Construct via `attach()`, not directly."""

    def __init__(
        self, *, client: Client, client_factory: Callable[[], Client], name: str, attach_result: CatalogAttachResult
    ) -> None:
        """Wrap an already-attached `client`/`attach_result` pair. Use `attach()`, not this directly."""
        self._client = client
        self._client_factory = client_factory
        self._name = name
        self._attach_result = attach_result
        self._detached = False
        # One `Client` per calling thread for exchange-mode RPCs (table_function/
        # scalar_function/...) — see the module docstring's "Thread safety" section.
        self._thread_local = threading.local()
        self._exchange_clients_lock = threading.Lock()
        self._exchange_clients: list[Client] = []

    @property
    def name(self) -> str:
        """The attach alias this catalog was attached under."""
        return self._name

    @property
    def attach_opaque_data(self) -> AttachOpaqueData:
        """The opaque attachment id every catalog-scoped RPC threads through."""
        return self._attach_result.attach_opaque_data

    @property
    def default_schema(self) -> str:
        """The catalog's default schema.

        The second place (after a table's own schema) `VgiAceroTable` looks
        for its resolved scan function, mirroring the DuckDB C++ extension's
        own resolution order.
        """
        return self._attach_result.default_schema

    @property
    def metadata_client(self) -> Client:
        """The underlying vgi-python `Client` used for **catalog-metadata** RPCs only.

        Covers `schemas`, `table_get`, `schema_contents`,
        `table_scan_function_get`, `table_column_statistics`, `bind`. Exchange-mode
        calls (table/scalar function invocation) must use `exchange_client()`
        instead — see the module docstring's "Thread safety" section. Named
        `metadata_client`, not the bare `client`, specifically so it doesn't
        read as "the client, use this for anything" next to `exchange_client()`
        at a REPL's tab-completion list.

        Raises after `detach()` — this is also how `VgiAceroTable`'s own RPC
        methods (which reach the shared client only via this property, never
        `self._client` directly) inherit the same guard without each needing
        its own check: confirmed live, before this guard existed, that
        calling a table method after `detach()` silently spawned a fresh
        connection and returned real-looking data instead of failing.
        """
        self._check_not_detached()
        return self._client

    def exchange_client(self) -> Client:
        """A `Client` safe for the calling thread's exclusive use, for exchange-mode RPCs.

        Use this (never `metadata_client`) for anything that drives
        `Client.table_function`/`scalar_function`/`table_in_out_function` —
        e.g. building a `vgi_scan()`/`vgi_semi_join_scan()` Declaration
        against a function reached through this catalog's attach, or inside
        a `pc.register_scalar_function` callback Acero may invoke from
        multiple threads. Lazily creates and starts one per calling thread
        via the internal client factory, reusing it across that thread's
        subsequent calls; never shared across threads. See the module
        docstring's "Thread safety" section.
        """
        self._check_not_detached()
        existing: Client | None = getattr(self._thread_local, "client", None)
        if existing is not None:
            return existing
        new_client = self._client_factory()
        new_client.start()
        self._thread_local.client = new_client
        with self._exchange_clients_lock:
            self._exchange_clients.append(new_client)
        return new_client

    def _check_not_detached(self) -> None:
        if self._detached:
            raise VgiAceroError(f"catalog {self._name!r} has been detached and can no longer be used")

    def schemas(self) -> list[str]:
        """List schema names in this catalog."""
        self._check_not_detached()
        try:
            infos = self._client.schemas(attach_opaque_data=self.attach_opaque_data)
        except VGI_CLIENT_ERRORS as e:
            raise wrap_error(e) from e
        return [s.name for s in infos]

    def tables(self, schema_name: str) -> list[str]:
        """List table names in `schema_name`."""
        self._check_not_detached()
        try:
            infos = self._client.schema_contents(
                attach_opaque_data=self.attach_opaque_data,
                name=schema_name,
                type=SchemaObjectType.TABLE,
            )
        except VGI_CLIENT_ERRORS as e:
            raise wrap_error(e) from e
        return [t.name for t in infos]

    def table_functions(self, schema_name: str) -> list[str]:
        """List table function names registered in `schema_name`."""
        self._check_not_detached()
        try:
            infos = self._client.schema_contents(
                attach_opaque_data=self.attach_opaque_data,
                name=schema_name,
                type=SchemaObjectType.TABLE_FUNCTION,
            )
        except VGI_CLIENT_ERRORS as e:
            raise wrap_error(e) from e
        return [f.name for f in infos]

    def scalar_functions(self, schema_name: str) -> list[str]:
        """List scalar function names registered in `schema_name`."""
        self._check_not_detached()
        try:
            infos = self._client.schema_contents(
                attach_opaque_data=self.attach_opaque_data,
                name=schema_name,
                type=SchemaObjectType.SCALAR_FUNCTION,
            )
        except VGI_CLIENT_ERRORS as e:
            raise wrap_error(e) from e
        return [f.name for f in infos]

    def table(
        self, schema_name: str, name: str, *, at_unit: str | None = None, at_value: str | None = None
    ) -> VgiAceroTable:
        """A lazy handle to a catalog table.

        No RPC happens until `.arrow_schema`/`.scan()` is used.

        `at_unit`/`at_value` request a time-travel view — a worker that
        doesn't support it on this table rejects the request at bind, the
        same as any other unsupported bind option.
        """
        self._check_not_detached()
        from vgi_acero.table import VgiAceroTable

        return VgiAceroTable(catalog=self, schema_name=schema_name, name=name, at_unit=at_unit, at_value=at_value)

    def scalar_function(self, schema_name: str, name: str) -> ScalarFunctionCall:
        """A callable registered as a `pyarrow.compute` scalar UDF (see `_scalar.py`)."""
        self._check_not_detached()
        from vgi_acero._scalar import make_scalar_function

        return make_scalar_function(self, schema_name, name)

    def detach(self) -> None:
        """Detach from the catalog and close the underlying client(s).

        Closes every per-thread exchange client `exchange_client()` created.
        Safe to call more than once.
        """
        if self._detached:
            return
        self._detached = True
        try:
            try:
                self._client.catalog_detach(attach_opaque_data=self.attach_opaque_data)
            except VGI_CLIENT_ERRORS as e:
                raise wrap_error(e) from e
        finally:
            self._client.stop()
            with self._exchange_clients_lock:
                exchange_clients, self._exchange_clients = self._exchange_clients, []
            for exchange_client in exchange_clients:
                # Best-effort cleanup: one client's failure to stop must never
                # block stopping the rest.
                with contextlib.suppress(Exception):
                    exchange_client.stop()

    def __repr__(self) -> str:
        """A REPL-friendly summary — `name` plus its schema list (best-effort; never raises)."""
        if self._detached:
            return f"VgiAceroCatalog(name={self._name!r}, detached=True)"
        try:
            schema_names = self.schemas()
        except Exception:  # noqa: BLE001 - __repr__ must never raise
            return f"VgiAceroCatalog(name={self._name!r})"
        return f"VgiAceroCatalog(name={self._name!r}, schemas={schema_names!r})"

    def __enter__(self) -> Self:
        """Support `with attach(...) as catalog:` — returns `self`."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Support `with attach(...) as catalog:` — calls `detach()`."""
        self.detach()


def _detect_transport(location: str) -> Transport:
    """Auto-detect transport from `location`'s scheme.

    Mirrors the DuckDB extension's LOCATION scheme table (`http://`/`https://`
    -> HTTP, `tcp://` -> TCP, anything else -> subprocess/shlex argv).
    """
    if location.startswith(("http://", "https://")):
        return "http"
    if location.startswith("tcp://"):
        return "tcp"
    return "subprocess"


def attach(
    location: str,
    *,
    name: str,
    transport: Transport | None = None,
    options: dict[str, Any] | None = None,
    data_version_spec: str | None = None,
    implementation_version: str | None = None,
    bearer_token: str | None = None,
    worker_limit: int | None = None,
    **client_kwargs: Any,
) -> VgiAceroCatalog:
    """Attach to a VGI catalog and return a `VgiAceroCatalog`.

    Args:
        location: For subprocess transport (the default for anything that
            isn't a recognized URL scheme), the worker command (shlex-split,
            no shell — matches `vgi.client.Client`'s own semantics, e.g.
            `"uv run --project ~/Development/vgi-python vgi-fixture-worker"`).
            For HTTP, `"http://..."`/`"https://..."`. For TCP, `"tcp://host:port"`.
        name: The catalog name to attach to (a worker can serve more than one).
        transport: `"subprocess"`, `"http"`, or `"tcp"`. Defaults to `None`,
            which auto-detects from `location`'s scheme.
        options: Catalog-specific ATTACH options.
        data_version_spec: Semver constraint for the catalog's data version.
        implementation_version: Semver constraint for the worker's implementation.
        bearer_token: Static bearer token, HTTP transport only.
        worker_limit: Max concurrent workers, subprocess transport only.
        **client_kwargs: Passed through to `Client(...)` / `Client.from_http(...)`
            / `Client.from_tcp(...)`.

    Returns:
        The attached `VgiAceroCatalog`.

    """
    resolved_transport = transport if transport is not None else _detect_transport(location)

    def client_factory() -> Client:
        """Build one fresh, unstarted `Client` connected the same way every time.

        Used both for the initial attach-time client and, via
        `VgiAceroCatalog.exchange_client()`, once per thread thereafter. No
        `catalog_attach` here — exchange-mode RPCs don't take
        `attach_opaque_data` at all, so a fresh connection is immediately
        usable with no re-attach step.
        """
        if resolved_transport == "subprocess":
            return Client(location, worker_limit=worker_limit, **client_kwargs)
        if resolved_transport == "http":
            return Client.from_http(location, bearer_token=bearer_token, **client_kwargs)
        if resolved_transport == "tcp":
            host_port = location.removeprefix("tcp://")
            host, _, port = host_port.partition(":")
            if not port:
                raise ValueError(f"tcp transport expects 'tcp://host:port' or 'host:port', got {location!r}")
            return Client.from_tcp(host, int(port), **client_kwargs)
        raise ValueError(f"unknown transport: {resolved_transport!r}")

    client = client_factory()

    def _cleanup() -> None:
        with contextlib.suppress(Exception):
            client.stop()

    try:
        client.start()
        result = client.catalog_attach(
            name=name,
            options=options,
            data_version_spec=data_version_spec,
            implementation_version=implementation_version,
        )
    except (*VGI_CLIENT_ERRORS, OSError) as e:
        _cleanup()
        raise wrap_error(e) from e
    except BaseException:
        _cleanup()
        raise

    return VgiAceroCatalog(client=client, client_factory=client_factory, name=name, attach_result=result)
