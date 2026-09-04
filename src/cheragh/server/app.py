"""FastAPI application factory for serving a RAGEngine."""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import logging
import math
import os
import secrets
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .. import __version__
from ..base import EmbeddingModel, _validate_top_k
from ..engine import RAGEngine
from ..vectorstores.memory import MemoryVectorStore

logger = logging.getLogger(__name__)
T = TypeVar("T")


class AskRequest(BaseModel):
    """Strict request model shared by the generated OpenAPI schema and handlers."""

    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=8_000)
    top_k: int | None = Field(default=None, ge=1)
    include_prompt: bool = False


class IndexRequest(BaseModel):
    """Strict payload accepted by the optional indexing endpoint."""

    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    path: str = Field(..., min_length=1, max_length=4_096)
    output: str = Field(default=".cheragh_index", min_length=1, max_length=4_096)
    incremental: bool = True
    chunk_size: int = Field(default=800, ge=1, le=100_000)
    chunk_overlap: int = Field(default=120, ge=0, le=100_000)

    @model_validator(mode="after")
    def validate_chunk_window(self) -> IndexRequest:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class _ServerBusyError(RuntimeError):
    pass


class _OperationTimeoutError(RuntimeError):
    pass


class _OperationLease:
    """Idempotent permit release for streaming response cleanup paths."""

    def __init__(self, release: Callable[[], None]):
        self._release = release
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
        self._release()


class _LeasedStream(Iterator[str]):
    """Close abandoned streams without releasing a still-running worker's permit."""

    def __init__(self, iterator: Iterator[str], lease: _OperationLease):
        self._iterator = iterator
        self._lease = lease
        self._context = contextvars.copy_context()
        self._lock = threading.Lock()
        self._running = False
        self._closing = False
        self._closed = False

    def __iter__(self) -> _LeasedStream:
        return self

    def __next__(self) -> str:
        with self._lock:
            if self._closing or self._closed:
                raise StopIteration
            if self._running:
                raise RuntimeError("stream is already being advanced")
            self._running = True
        try:
            # Preserve one request context across next()/close(), even when
            # Starlette schedules successive chunks on different workers.
            return self._context.run(next, self._iterator)
        except BaseException:
            with self._lock:
                self._closing = True
            raise
        finally:
            with self._lock:
                self._running = False
                close_now = self._closing and not self._closed
                if close_now:
                    self._closed = True
            if close_now:
                self._close_iterator()

    def close(self) -> None:
        with self._lock:
            self._closing = True
            if self._running or self._closed:
                # The advancing worker owns cleanup until next() really exits.
                return
            self._closed = True
        self._close_iterator()

    def _close_iterator(self) -> None:
        try:
            close = getattr(self._iterator, "close", None)
            if callable(close):
                self._context.run(close)
        except Exception as exc:
            logger.error("stream_close_failed", extra={"error_type": type(exc).__name__})
        finally:
            # Closing a generator before its first next() does not execute its
            # finally block, so the lease belongs to this wrapper instead.
            self._lease.release()


class _CloseStreamOperation:
    def __init__(self, stream: _LeasedStream):
        self._stream = stream

    async def __call__(self) -> None:
        # Provider close() can block. Schedule it before awaiting so request
        # cancellation cannot prevent cleanup from being submitted.
        future = asyncio.get_running_loop().run_in_executor(
            None, contextvars.copy_context().run, self._stream.close
        )
        future.add_done_callback(_consume_future_exception)
        await asyncio.shield(future)


class _OperationLimiter:
    """Keep timed-out synchronous work counted until its worker really exits."""

    def __init__(self, capacity: int):
        self._semaphore = threading.BoundedSemaphore(capacity)

    def try_acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()

    def reserve(self) -> _OperationLease | None:
        if not self.try_acquire():
            return None
        return _OperationLease(self.release)

    async def run(self, operation: Callable[[], T], *, timeout_seconds: float) -> T:
        if not self.try_acquire():
            raise _ServerBusyError("operation capacity exhausted")

        def guarded_operation() -> T:
            try:
                return operation()
            finally:
                self.release()

        try:
            future = asyncio.get_running_loop().run_in_executor(
                None, contextvars.copy_context().run, guarded_operation
            )
            # If the request times out, consume a later worker exception so
            # asyncio never prints provider messages to stderr as an
            # "exception was never retrieved" warning.
            future.add_done_callback(_consume_future_exception)
        except BaseException:
            self.release()
            raise

        try:
            # Shielding lets the worker finish and release its permit after a
            # timeout or client disconnect. Synchronous work cannot be safely
            # force-cancelled.
            return await asyncio.wait_for(asyncio.shield(future), timeout=timeout_seconds)
        except asyncio.TimeoutError as exc:
            raise _OperationTimeoutError("operation timed out") from exc


class _PayloadTooLargeError(RuntimeError):
    pass


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    if future.cancelled():
        return
    try:
        future.exception()
    except BaseException:
        # Retrieving is the only goal; request handling owns any synchronous
        # propagation while timed-out work must remain silent.
        pass


class _ProductionBoundaryMiddleware:
    """Bound request bodies, redact failures, and attach safe request IDs."""

    def __init__(self, app: Any, *, max_request_body_bytes: int):
        self.app = app
        self.max_request_body_bytes = max_request_body_bytes

    async def __call__(self, scope: dict[str, Any], receive: Callable[..., Any], send: Callable[..., Any]) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = _request_id_from_scope(scope)
        request_state = scope.setdefault("state", {})
        if isinstance(request_state, dict):
            request_state["request_id"] = request_id
        started_at = time.monotonic()
        response_started = False
        response_complete = False
        status_code = 500
        received_bytes = 0
        payload_too_large = False
        response_replaced = False

        try:
            declared_length = _declared_content_length(scope)
        except ValueError:
            await _send_json_response(send, 400, "Invalid Content-Length", request_id)
            _log_request(scope, request_id, 400, started_at)
            return
        if declared_length is not None and declared_length > self.max_request_body_bytes:
            await _send_json_response(send, 413, "Request body too large", request_id)
            _log_request(scope, request_id, 413, started_at)
            return

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes, payload_too_large
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_request_body_bytes:
                    payload_too_large = True
                    raise _PayloadTooLargeError("request body limit exceeded")
            return message

        async def send_with_boundaries(message: dict[str, Any]) -> None:
            nonlocal response_complete, response_started, response_replaced, status_code
            if response_replaced:
                return
            if payload_too_large and not response_started:
                # Framework body parsers may translate receive() failures into
                # a generic 400. The boundary still owns the authoritative 413.
                response_replaced = response_started = response_complete = True
                status_code = 413
                await _send_json_response(send, 413, "Request body too large", request_id)
                return
            if message.get("type") == "http.response.start":
                response_started = True
                status_code = int(message.get("status", 500))
                headers = list(message.get("headers", []))
                _replace_header(headers, b"x-request-id", request_id.encode("ascii"))
                _replace_header(headers, b"x-content-type-options", b"nosniff")
                _replace_header(headers, b"cache-control", b"no-store")
                message["headers"] = headers
            elif message.get("type") == "http.response.body" and not message.get("more_body", False):
                response_complete = True
            await send(message)

        try:
            await self.app(scope, limited_receive, send_with_boundaries)
        except _PayloadTooLargeError:
            if response_started:
                logger.warning("request_body_limit_after_response_start", extra={"request_id": request_id})
                if not response_complete:
                    await _finish_started_response_safely(send)
                return
            status_code = 413
            await _send_json_response(send_with_boundaries, 413, "Request body too large", request_id)
        except Exception as exc:
            # Provider exceptions can echo prompts, paths, or credentials.
            # Record only their type at this external trust boundary.
            logger.error(
                "unhandled_http_error",
                extra={"request_id": request_id, "error_type": type(exc).__name__},
            )
            if response_started:
                # Headers already left the process; a second response would
                # corrupt the ASGI stream. Finish the body when possible and
                # swallow the provider exception so its message/repr cannot be
                # emitted by the ASGI server after response start.
                if not response_complete:
                    await _finish_started_response_safely(send)
                return
            status_code = 500
            await _send_json_response(send_with_boundaries, 500, "Internal server error", request_id)
        finally:
            _log_request(scope, request_id, status_code, started_at)


def create_app(
    engine: RAGEngine | None = None,
    *,
    config_path: str | None = None,
    index_path: str | None = None,
    index_embedding_model: EmbeddingModel | None = None,
    enable_indexing: bool | None = None,
    allowed_index_root: str | Path | None = None,
    api_key: str | None = None,
    require_auth: bool | None = None,
    allow_prompt_exposure: bool = False,
    readiness_check: Callable[[], bool] | None = None,
    max_top_k: int = 50,
    max_request_body_bytes: int = 1_048_576,
    max_concurrent_operations: int = 16,
    request_timeout_seconds: float = 60.0,
    index_timeout_seconds: float = 900.0,
    stream_max_duration_seconds: float = 300.0,
):
    """Create a production-bounded FastAPI app around a RAG engine.

    ``fastapi`` is optional: ``pip install cheragh[fastapi]``.

    ``/ask``, ``/stream``, ``/stats`` and ``/index`` require ``X-API-Key``
    when ``api_key``/``CHERAGH_API_KEY`` is configured. Set ``require_auth``
    (or ``CHERAGH_REQUIRE_AUTH=true``) to fail startup when no key is set.

    ``/index`` is disabled by default. Enabling it additionally requires both
    an API key and an explicit allowed index root. Synchronous engine work is
    bounded by a shared concurrency limiter; timed-out work keeps its permit
    until its worker actually exits.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("The server requires FastAPI. Install with: pip install cheragh[fastapi]") from exc

    if readiness_check is not None and not callable(readiness_check):
        raise TypeError("readiness_check must be callable")

    indexing_enabled = (
        _as_bool(os.getenv("CHERAGH_ENABLE_INDEXING"), default=False)
        if enable_indexing is None
        else _require_bool(enable_indexing, "enable_indexing")
    )
    auth_required = (
        _as_bool(os.getenv("CHERAGH_REQUIRE_AUTH"), default=False)
        if require_auth is None
        else _require_bool(require_auth, "require_auth")
    )
    configured_api_key = api_key if api_key is not None else os.getenv("CHERAGH_API_KEY")
    if configured_api_key is not None and not isinstance(configured_api_key, str):
        raise TypeError("api_key must be a string")
    required_api_key = configured_api_key if configured_api_key and configured_api_key.strip() else None
    configured_root = allowed_index_root if allowed_index_root is not None else os.getenv("CHERAGH_INDEX_ROOT")
    if isinstance(configured_root, str) and not configured_root.strip():
        configured_root = None

    if auth_required and required_api_key is None:
        raise ValueError("API authentication is required but no API key is configured")
    if indexing_enabled and required_api_key is None:
        raise ValueError("enabling /index requires an API key")
    if indexing_enabled and configured_root is None:
        raise ValueError("enabling /index requires an explicit allowed_index_root")

    root = Path(configured_root).resolve() if configured_root is not None else None
    if root is not None and (not root.exists() or not root.is_dir()):
        raise ValueError("allowed_index_root must be an existing directory")

    max_top_k = _validate_top_k(max_top_k, name="max_top_k")
    max_request_body_bytes = _positive_int(max_request_body_bytes, "max_request_body_bytes")
    max_concurrent_operations = _positive_int(max_concurrent_operations, "max_concurrent_operations")
    request_timeout_seconds = _positive_float(request_timeout_seconds, "request_timeout_seconds")
    index_timeout_seconds = _positive_float(index_timeout_seconds, "index_timeout_seconds")
    stream_max_duration_seconds = _positive_float(stream_max_duration_seconds, "stream_max_duration_seconds")
    allow_prompt_exposure = _require_bool(allow_prompt_exposure, "allow_prompt_exposure")

    # Fail invalid security and resource settings before opening indexes or
    # initializing provider clients from a configuration file.
    if engine is None:
        if config_path:
            engine = RAGEngine.from_config(config_path)
        elif index_path:
            store = MemoryVectorStore.load(index_path, index_embedding_model)
            engine = RAGEngine(store.as_retriever())
        else:
            raise ValueError("create_app requires engine, config_path or index_path")
    if not callable(getattr(engine, "ask", None)) or not callable(getattr(engine, "stream", None)):
        raise TypeError("engine must provide callable ask() and stream() methods")
    default_top_k = min(_validate_top_k(getattr(engine, "top_k", max_top_k)), max_top_k)

    operation_limiter = _OperationLimiter(max_concurrent_operations)
    app = FastAPI(title="cheragh", version=__version__)
    app.add_middleware(_ProductionBoundaryMiddleware, max_request_body_bytes=max_request_body_bytes)

    async def require_api_key(x_api_key: list[str] | None = Header(default=None)) -> None:
        provided_api_key = x_api_key[0] if x_api_key is not None and len(x_api_key) == 1 else None
        if required_api_key is not None and not _constant_time_secret_matches(provided_api_key, required_api_key):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

    AuthDependency = Depends(require_api_key)

    class ManagedStreamingResponse(StreamingResponse):
        async def __call__(self, scope, receive, send) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                # StreamingResponse may skip its background callback when a
                # send fails or its task is cancelled after client disconnect.
                if self.background is not None:
                    await self.background()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/ready")
    async def ready():
        try:
            is_ready = readiness_check() if readiness_check is not None else engine is not None
        except Exception as exc:
            logger.error("readiness_check_failed", extra={"error_type": type(exc).__name__})
            is_ready = False
        if not is_ready:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready", "version": __version__}

    @app.post("/ask", dependencies=[AuthDependency])
    async def ask(request: AskRequest) -> dict[str, Any]:
        if request.top_k is not None and request.top_k > max_top_k:
            raise HTTPException(status_code=422, detail=f"top_k must be <= {max_top_k}")
        if request.include_prompt and not allow_prompt_exposure:
            raise HTTPException(status_code=403, detail="Prompt exposure is disabled")
        try:
            response = await operation_limiter.run(
                lambda: engine.ask(request.query, top_k=request.top_k or default_top_k),
                timeout_seconds=request_timeout_seconds,
            )
        except _ServerBusyError as exc:
            raise HTTPException(status_code=503, detail="Server is busy", headers={"Retry-After": "1"}) from exc
        except _OperationTimeoutError as exc:
            raise HTTPException(status_code=504, detail="Request timed out") from exc
        return response.to_dict(include_prompt=request.include_prompt)

    @app.post("/stream", dependencies=[AuthDependency])
    async def stream(request: AskRequest):
        if request.top_k is not None and request.top_k > max_top_k:
            raise HTTPException(status_code=422, detail=f"top_k must be <= {max_top_k}")
        if request.include_prompt:
            raise HTTPException(status_code=422, detail="include_prompt is not supported for streaming")
        stream_lease = operation_limiter.reserve()
        if stream_lease is None:
            raise HTTPException(status_code=503, detail="Server is busy", headers={"Retry-After": "1"})

        def guarded_stream() -> Iterator[str]:
            iterator: Iterator[str] | None = None
            deadline = time.monotonic() + stream_max_duration_seconds
            try:
                iterator = iter(engine.stream(request.query, top_k=request.top_k or default_top_k))
                while time.monotonic() < deadline:
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        return
                    if time.monotonic() >= deadline:
                        logger.warning("stream_duration_limit_reached")
                        return
                    yield chunk
            finally:
                if iterator is not None:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception as exc:
                            logger.error("stream_close_failed", extra={"error_type": type(exc).__name__})

        leased_stream = _LeasedStream(guarded_stream(), stream_lease)
        try:
            return ManagedStreamingResponse(
                leased_stream,
                media_type="text/plain",
                headers={"X-Accel-Buffering": "no"},
                background=_CloseStreamOperation(leased_stream),
            )
        except BaseException:
            leased_stream.close()
            raise

    @app.post("/index", dependencies=[AuthDependency])
    async def index(request: IndexRequest) -> dict[str, Any]:
        if not indexing_enabled:
            raise HTTPException(status_code=403, detail="Indexing endpoint is disabled")
        if root is None:  # Defensive: startup validation above should make this unreachable.
            raise HTTPException(status_code=503, detail="Indexing endpoint is not configured")
        from ..indexing import index_path as build_index

        try:
            input_path = _resolve_under_root(request.path, root)
            output_path = _resolve_under_root(request.output, root)
        except (OSError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Invalid index path") from exc

        try:
            return await operation_limiter.run(
                lambda: build_index(
                    input_path,
                    output_path,
                    chunk_size=request.chunk_size,
                    chunk_overlap=request.chunk_overlap,
                    incremental=request.incremental,
                ),
                timeout_seconds=index_timeout_seconds,
            )
        except _ServerBusyError as exc:
            raise HTTPException(status_code=503, detail="Server is busy", headers={"Retry-After": "1"}) from exc
        except _OperationTimeoutError as exc:
            raise HTTPException(status_code=504, detail="Indexing timed out") from exc
        except (FileNotFoundError, NotADirectoryError, PermissionError, ValueError) as exc:
            logger.warning("index_request_rejected", extra={"error_type": type(exc).__name__})
            raise HTTPException(status_code=400, detail="Indexing request could not be processed") from exc
        except Exception as exc:
            logger.error("index_operation_failed", extra={"error_type": type(exc).__name__})
            raise HTTPException(status_code=500, detail="Indexing failed")

    @app.get("/stats", dependencies=[AuthDependency])
    def stats() -> dict[str, Any]:
        retriever = getattr(engine, "retriever", None)
        store = getattr(retriever, "store", None)
        docs = getattr(store, "documents", None)
        cache_backend = getattr(engine, "cache_backend", None)
        return {
            "document_count": len(docs) if docs is not None else None,
            "top_k": getattr(engine, "top_k", None),
            "cache": cache_backend.stats().to_dict() if cache_backend is not None else None,
        }

    return app


def _resolve_under_root(path: str | Path, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes allowed index root") from exc
    return candidate


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError("expected a boolean value")


def _require_bool(value: bool, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _constant_time_secret_matches(provided: str | None, expected: str) -> bool:
    provided_bytes = b"" if provided is None else provided.encode("utf-8")
    if len(provided_bytes) > 4_096:
        provided_bytes = b""
    provided_digest = hashlib.sha256(provided_bytes).digest()
    expected_digest = hashlib.sha256(expected.encode("utf-8")).digest()
    return secrets.compare_digest(provided_digest, expected_digest)


def _request_id_from_scope(scope: dict[str, Any]) -> str:
    candidates = [value for name, value in scope.get("headers", []) if name.lower() == b"x-request-id"]
    if len(candidates) == 1:
        try:
            value = candidates[0].decode("ascii")
        except (UnicodeDecodeError, AttributeError):
            value = ""
        if 1 <= len(value) <= 128 and all(character.isalnum() or character in "._-" for character in value):
            return value
    return uuid.uuid4().hex


def _declared_content_length(scope: dict[str, Any]) -> int | None:
    values = [value for name, value in scope.get("headers", []) if name.lower() == b"content-length"]
    transfer_encoding = [value for name, value in scope.get("headers", []) if name.lower() == b"transfer-encoding"]
    if values and transfer_encoding:
        raise ValueError("Content-Length and Transfer-Encoding cannot be combined")
    if not values:
        return None
    if len(values) != 1:
        raise ValueError("ambiguous Content-Length")
    try:
        raw_value = values[0].decode("ascii")
        if not raw_value.isdigit():
            raise ValueError("invalid Content-Length")
        value = int(raw_value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("invalid Content-Length") from exc
    if value < 0:
        raise ValueError("invalid Content-Length")
    return value


def _replace_header(headers: list[tuple[bytes, bytes]], name: bytes, value: bytes) -> None:
    headers[:] = [(header_name, header_value) for header_name, header_value in headers if header_name.lower() != name]
    headers.append((name, value))


async def _send_json_response(send: Callable[..., Any], status: int, detail: str, request_id: str) -> None:
    body = json.dumps({"detail": detail, "request_id": request_id}, separators=(",", ":")).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"x-request-id", request_id.encode("ascii")),
                (b"x-content-type-options", b"nosniff"),
                (b"cache-control", b"no-store"),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _finish_started_response_safely(send: Callable[..., Any]) -> None:
    """Terminate an already-started ASGI response without exposing an error."""

    try:
        await send({"type": "http.response.body", "body": b"", "more_body": False})
    except Exception:
        # The client may already be gone or the server may have closed the
        # stream. There is no safe second response at this point.
        return


def _log_request(scope: dict[str, Any], request_id: str, status_code: int, started_at: float) -> None:
    logger.info(
        "http_request_complete",
        extra={
            "request_id": request_id,
            "http_method": str(scope.get("method", "")),
            "http_path": str(scope.get("path", "")),
            "http_status": status_code,
            "duration_ms": round((time.monotonic() - started_at) * 1_000, 3),
        },
    )


def app_from_env():  # pragma: no cover - runtime helper
    config_path = os.getenv("CHERAGH_CONFIG")
    index_path = os.getenv("CHERAGH_INDEX")
    return create_app(
        config_path=config_path,
        index_path=index_path,
        enable_indexing=_as_bool(os.getenv("CHERAGH_ENABLE_INDEXING"), default=False),
        allowed_index_root=os.getenv("CHERAGH_INDEX_ROOT"),
        api_key=os.getenv("CHERAGH_API_KEY"),
        require_auth=_as_bool(os.getenv("CHERAGH_REQUIRE_AUTH"), default=False),
    )


app = None  # uvicorn can use create_app through CLI; this sentinel avoids implicit heavy loading.
