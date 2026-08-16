"""Server entrypoint helpers."""
from __future__ import annotations

import ipaddress
from pathlib import Path


def serve(
    config: str | None = None,
    index: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    enable_indexing: bool = False,
    allowed_index_root: str | Path | None = None,
    api_key: str | None = None,
    require_auth: bool | None = None,
    allow_prompt_exposure: bool = False,
    max_top_k: int = 50,
    max_request_body_bytes: int = 1_048_576,
    max_concurrent_operations: int = 16,
    max_server_connections: int = 128,
    request_timeout_seconds: float = 60.0,
    index_timeout_seconds: float = 900.0,
    stream_max_duration_seconds: float = 300.0,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Serving requires uvicorn. Install with: pip install cheragh[fastapi]") from exc

    from .app import create_app

    # Anonymous local development remains compatible. A non-loopback bind is
    # fail-closed unless the caller explicitly made another trust decision.
    effective_require_auth = True if require_auth is None and not _is_loopback_host(host) else require_auth
    app = create_app(
        config_path=config,
        index_path=index,
        enable_indexing=enable_indexing,
        allowed_index_root=allowed_index_root,
        api_key=api_key,
        require_auth=effective_require_auth,
        allow_prompt_exposure=allow_prompt_exposure,
        max_top_k=max_top_k,
        max_request_body_bytes=max_request_body_bytes,
        max_concurrent_operations=max_concurrent_operations,
        request_timeout_seconds=request_timeout_seconds,
        index_timeout_seconds=index_timeout_seconds,
        stream_max_duration_seconds=stream_max_duration_seconds,
    )
    if (
        isinstance(max_server_connections, bool)
        or not isinstance(max_server_connections, int)
        or max_server_connections <= max_concurrent_operations
    ):
        raise ValueError("max_server_connections must be an integer greater than max_concurrent_operations")
    uvicorn.run(
        app,
        host=host,
        port=port,
        # Keep probe/cheap-route headroom above the expensive-operation
        # semaphore. Reusing the same limit can starve /health and /ready while
        # otherwise healthy streams occupy every application permit.
        limit_concurrency=max_server_connections,
        server_header=False,
        timeout_keep_alive=5,
    )


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        # Hostnames are not resolved here: DNS can change between the check and
        # bind. Treat unknown names as externally reachable.
        return False
