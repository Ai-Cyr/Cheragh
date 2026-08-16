import asyncio
import contextlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import ValidationError

from cheragh.server.app import (
    AskRequest,
    IndexRequest,
    _as_bool,
    _OperationLimiter,
    _OperationTimeoutError,
    _ProductionBoundaryMiddleware,
    _positive_float,
    _resolve_under_root,
    _ServerBusyError,
    create_app,
)
from cheragh.server.main import _is_loopback_host, serve


def _headers(messages):
    start = next(message for message in messages if message["type"] == "http.response.start")
    return {name.decode("ascii").lower(): value.decode("ascii") for name, value in start.get("headers", [])}


async def _call_asgi(app, *, headers=None, chunks=None):
    messages = []
    request_chunks = list(chunks or [b""])

    async def receive():
        body = request_chunks.pop(0) if request_chunks else b""
        return {"type": "http.request", "body": body, "more_body": bool(request_chunks)}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "method": "POST",
            "path": "/ask",
            "headers": headers or [],
        },
        receive,
        send,
    )
    return messages


class _FakeHTTPException(Exception):
    def __init__(self, status_code, detail, headers=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.headers = headers or {}


class _FakeDependency:
    def __init__(self, dependency):
        self.dependency = dependency


class _FakeRoute:
    def __init__(self, endpoint, options):
        self.endpoint = endpoint
        self.options = options


class _FakeFastAPI:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.routes = {}
        self.middleware = []

    def add_middleware(self, middleware, **kwargs):
        self.middleware.append((middleware, kwargs))

    def _route(self, method, path, **options):
        def decorator(endpoint):
            self.routes[(method, path)] = _FakeRoute(endpoint, options)
            return endpoint

        return decorator

    def get(self, path, **options):
        return self._route("GET", path, **options)

    def post(self, path, **options):
        return self._route("POST", path, **options)


class _FakeJSONResponse:
    def __init__(self, *, status_code, content):
        self.status_code = status_code
        self.content = content


class _FakeStreamingResponse:
    def __init__(self, content, *, media_type, headers=None, background=None):
        self.content = content
        self.media_type = media_type
        self.headers = headers or {}
        self.background = background


@contextlib.contextmanager
def _fake_fastapi_modules():
    fastapi = types.ModuleType("fastapi")
    fastapi.__path__ = []
    fastapi.Depends = _FakeDependency
    fastapi.FastAPI = _FakeFastAPI
    fastapi.Header = lambda default=None: default
    fastapi.HTTPException = _FakeHTTPException
    responses = types.ModuleType("fastapi.responses")
    responses.JSONResponse = _FakeJSONResponse
    responses.StreamingResponse = _FakeStreamingResponse
    with patch.dict(sys.modules, {"fastapi": fastapi, "fastapi.responses": responses}):
        yield


class ServerRequestModelTests(unittest.TestCase):
    def test_only_literal_loopback_bindings_are_treated_as_local(self):
        for host in ("127.0.0.1", "::1", "[::1]", "localhost"):
            with self.subTest(host=host):
                self.assertTrue(_is_loopback_host(host))
        for host in ("0.0.0.0", "::", "api.internal", "localhost.example"):
            with self.subTest(host=host):
                self.assertFalse(_is_loopback_host(host))

    def test_serve_requires_auth_for_public_bind_and_bounds_uvicorn(self):
        uvicorn = types.ModuleType("uvicorn")
        uvicorn.run = Mock()
        with patch.dict(sys.modules, {"uvicorn": uvicorn}), patch("cheragh.server.app.create_app") as factory:
            factory.return_value = object()
            serve(
                config="rag.yaml",
                host="0.0.0.0",
                allow_prompt_exposure=True,
                max_top_k=23,
                max_request_body_bytes=456,
                max_concurrent_operations=7,
                max_server_connections=70,
                request_timeout_seconds=8,
                index_timeout_seconds=9,
                stream_max_duration_seconds=10,
            )

        self.assertTrue(factory.call_args.kwargs["require_auth"])
        self.assertTrue(factory.call_args.kwargs["allow_prompt_exposure"])
        self.assertEqual(factory.call_args.kwargs["max_top_k"], 23)
        self.assertEqual(factory.call_args.kwargs["max_request_body_bytes"], 456)
        self.assertEqual(factory.call_args.kwargs["request_timeout_seconds"], 8)
        self.assertEqual(factory.call_args.kwargs["index_timeout_seconds"], 9)
        self.assertEqual(factory.call_args.kwargs["stream_max_duration_seconds"], 10)
        self.assertEqual(uvicorn.run.call_args.kwargs["limit_concurrency"], 70)
        self.assertFalse(uvicorn.run.call_args.kwargs["server_header"])

        uvicorn.run.reset_mock()
        with patch.dict(sys.modules, {"uvicorn": uvicorn}), patch("cheragh.server.app.create_app") as factory:
            factory.return_value = object()
            serve(config="rag.yaml", host="127.0.0.1")
        self.assertIsNone(factory.call_args.kwargs["require_auth"])

        with patch.dict(sys.modules, {"uvicorn": uvicorn}), patch("cheragh.server.app.create_app") as factory:
            factory.return_value = object()
            with self.assertRaisesRegex(ValueError, "greater than max_concurrent_operations"):
                serve(config="rag.yaml", max_concurrent_operations=8, max_server_connections=8)

    def test_ask_request_is_strict_and_rejects_blank_or_extra_input(self):
        with self.assertRaises(ValidationError):
            AskRequest.model_validate({"query": "   "})
        with self.assertRaises(ValidationError):
            AskRequest.model_validate({"query": "hello", "include_prompt": 1})
        with self.assertRaises(ValidationError):
            AskRequest.model_validate({"query": "hello", "unexpected": "field"})

        request = AskRequest.model_validate({"query": "  hello  ", "top_k": 2})
        self.assertEqual(request.query, "hello")

    def test_index_request_rejects_invalid_window_and_coercion(self):
        with self.assertRaises(ValidationError):
            IndexRequest.model_validate({"path": "docs", "chunk_size": 100, "chunk_overlap": 100})
        with self.assertRaises(ValidationError):
            IndexRequest.model_validate({"path": "docs", "incremental": "true"})
        with self.assertRaises(ValidationError):
            IndexRequest.model_validate({"path": "docs", "chunk_size": 800, "extra": True})

    def test_security_environment_booleans_reject_typos(self):
        self.assertTrue(_as_bool("TRUE"))
        self.assertFalse(_as_bool(" off "))
        with self.assertRaises(ValueError):
            _as_bool("flase")
        for invalid in (float("nan"), float("inf"), 0, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _positive_float(invalid, "timeout")

    def test_resolve_under_root_rejects_parent_and_symlink_escapes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "allowed"
            outside = base / "outside"
            root.mkdir()
            outside.mkdir()

            self.assertEqual(_resolve_under_root("docs", root), root / "docs")
            with self.assertRaises(ValueError):
                _resolve_under_root("../outside", root)

            link = root / "escape"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                return
            with self.assertRaises(ValueError):
                _resolve_under_root(link / "document.txt", root)


class ProductionBoundaryMiddlewareTests(unittest.TestCase):
    def test_declared_oversize_body_is_rejected_before_downstream(self):
        downstream_called = False

        async def downstream(scope, receive, send):
            nonlocal downstream_called
            downstream_called = True

        middleware = _ProductionBoundaryMiddleware(downstream, max_request_body_bytes=4)
        messages = asyncio.run(
            _call_asgi(
                middleware,
                headers=[(b"content-length", b"5"), (b"x-request-id", b"safe-id")],
                chunks=[b"12345"],
            )
        )

        self.assertFalse(downstream_called)
        self.assertEqual(messages[0]["status"], 413)
        self.assertEqual(_headers(messages)["x-request-id"], "safe-id")
        self.assertEqual(_headers(messages)["cache-control"], "no-store")

    def test_chunked_oversize_body_is_rejected(self):
        async def downstream(scope, receive, send):
            while True:
                message = await receive()
                if not message.get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        middleware = _ProductionBoundaryMiddleware(downstream, max_request_body_bytes=4)
        messages = asyncio.run(_call_asgi(middleware, chunks=[b"12", b"345"]))

        self.assertEqual(messages[0]["status"], 413)
        self.assertEqual(len(_headers(messages)["x-request-id"]), 32)

    def test_ambiguous_body_framing_is_rejected(self):
        async def downstream(scope, receive, send):
            self.fail("ambiguous request reached downstream")

        middleware = _ProductionBoundaryMiddleware(downstream, max_request_body_bytes=100)
        messages = asyncio.run(
            _call_asgi(
                middleware,
                headers=[(b"content-length", b"2"), (b"transfer-encoding", b"chunked")],
                chunks=[b"{}"],
            )
        )
        self.assertEqual(messages[0]["status"], 400)

    def test_unhandled_exception_is_redacted_and_correlated(self):
        async def downstream(scope, receive, send):
            raise RuntimeError("SECRET filesystem path /srv/private")

        middleware = _ProductionBoundaryMiddleware(downstream, max_request_body_bytes=100)
        with self.assertLogs("cheragh.server.app", level="ERROR"):
            messages = asyncio.run(_call_asgi(middleware, headers=[(b"x-request-id", b"trace-123")]))

        body = b"".join(message.get("body", b"") for message in messages).decode("utf-8")
        self.assertEqual(messages[0]["status"], 500)
        self.assertNotIn("SECRET", body)
        self.assertEqual(json.loads(body)["request_id"], "trace-123")

    def test_response_headers_override_downstream_and_unsafe_request_id(self):
        downstream_request_id = None

        async def downstream(scope, receive, send):
            nonlocal downstream_request_id
            downstream_request_id = scope["state"]["request_id"]
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"x-request-id", b"downstream"), (b"cache-control", b"public")],
                }
            )
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = _ProductionBoundaryMiddleware(downstream, max_request_body_bytes=100)
        messages = asyncio.run(
            _call_asgi(middleware, headers=[(b"x-request-id", b"bad\r\nlog-injection")])
        )
        headers = _headers(messages)

        self.assertEqual(len(headers["x-request-id"]), 32)
        self.assertEqual(downstream_request_id, headers["x-request-id"])
        self.assertEqual(headers["cache-control"], "no-store")
        self.assertEqual(headers["x-content-type-options"], "nosniff")

    def test_exception_after_response_start_is_redacted_and_terminated(self):
        async def downstream(scope, receive, send):
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"partial", "more_body": True})
            raise RuntimeError("SECRET provider credential")

        middleware = _ProductionBoundaryMiddleware(downstream, max_request_body_bytes=100)
        with self.assertLogs("cheragh.server.app", level="ERROR") as logs:
            messages = asyncio.run(_call_asgi(middleware))

        self.assertEqual(messages[-1], {"type": "http.response.body", "body": b"", "more_body": False})
        self.assertNotIn("SECRET", "\n".join(logs.output))


class OperationLimiterTests(unittest.TestCase):
    def test_worker_exception_is_propagated_and_capacity_is_reusable(self):
        async def scenario():
            limiter = _OperationLimiter(1)

            def fail():
                raise RuntimeError("provider failed")

            with self.assertRaisesRegex(RuntimeError, "provider failed"):
                await limiter.run(fail, timeout_seconds=0.1)
            return await limiter.run(lambda: "reused", timeout_seconds=0.1)

        self.assertEqual(asyncio.run(scenario()), "reused")

    def test_timeout_keeps_capacity_reserved_until_worker_finishes(self):
        release_worker = threading.Event()
        worker_started = threading.Event()

        def blocking_operation():
            worker_started.set()
            release_worker.wait(timeout=2)
            return "done"

        async def scenario():
            limiter = _OperationLimiter(1)
            with self.assertRaises(_OperationTimeoutError):
                await limiter.run(blocking_operation, timeout_seconds=0.01)
            self.assertTrue(worker_started.is_set())
            with self.assertRaises(_ServerBusyError):
                await limiter.run(lambda: "second", timeout_seconds=0.1)

            release_worker.set()
            for _ in range(100):
                try:
                    return await limiter.run(lambda: "available", timeout_seconds=0.1)
                except _ServerBusyError:
                    await asyncio.sleep(0.005)
            self.fail("worker did not release its capacity")

        self.assertEqual(asyncio.run(scenario()), "available")


class FakeFastAPIFactoryTests(unittest.TestCase):
    class _Response:
        def to_dict(self, *, include_prompt=False):
            payload = {"query": "q", "answer": "safe", "sources": []}
            if include_prompt:
                payload["prompt"] = "system prompt"
            return payload

    class _Engine:
        retriever = None
        cache_backend = None
        top_k = 5

        def ask(self, query, top_k=None):
            return FakeFastAPIFactoryTests._Response()

        def stream(self, query, top_k=None):
            yield "safe"

    def _app(self, **kwargs):
        with _fake_fastapi_modules(), patch.dict(os.environ, {}, clear=True):
            return create_app(self._Engine(), **kwargs)

    def test_factory_registers_probes_and_production_boundary(self):
        app = self._app(max_request_body_bytes=321, readiness_check=lambda: False)

        self.assertIn(("GET", "/health"), app.routes)
        self.assertIn(("GET", "/ready"), app.routes)
        self.assertEqual(app.middleware[0][1]["max_request_body_bytes"], 321)
        readiness = asyncio.run(app.routes[("GET", "/ready")].endpoint())
        self.assertEqual(readiness.status_code, 503)
        self.assertEqual(readiness.content, {"status": "not_ready"})

    def test_auth_dependency_is_constant_time_safe_and_fail_closed(self):
        app = self._app(api_key="sëcret", require_auth=True)
        dependency = app.routes[("POST", "/ask")].options["dependencies"][0].dependency

        for invalid in (None, [], ["wrong"], ["sëcret", "duplicate"], ["é" * 5_000]):
            with self.subTest(invalid=invalid), self.assertRaises(_FakeHTTPException) as raised:
                asyncio.run(dependency(invalid))
            self.assertEqual(raised.exception.status_code, 401)
            self.assertEqual(raised.exception.headers["WWW-Authenticate"], "ApiKey")
        asyncio.run(dependency(["sëcret"]))

    def test_factory_rejects_insecure_indexing_and_missing_required_auth(self):
        with tempfile.TemporaryDirectory() as tmp, _fake_fastapi_modules(), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "API key"):
                create_app(self._Engine(), enable_indexing=True, allowed_index_root=tmp)
            with self.assertRaisesRegex(ValueError, "allowed_index_root"):
                create_app(self._Engine(), enable_indexing=True, api_key="secret")
            with self.assertRaisesRegex(ValueError, "allowed_index_root"):
                create_app(self._Engine(), enable_indexing=True, api_key="secret", allowed_index_root="   ")
            with self.assertRaisesRegex(ValueError, "authentication"):
                create_app(self._Engine(), require_auth=True)
            with self.assertRaisesRegex(ValueError, "authentication"):
                create_app(self._Engine(), require_auth=True, api_key="   ")
            with self.assertRaisesRegex(TypeError, "api_key"):
                create_app(self._Engine(), api_key=123)
            with self.assertRaisesRegex(TypeError, "readiness_check"):
                create_app(self._Engine(), readiness_check=True)
            with self.assertRaisesRegex(TypeError, "ask.*stream"):
                create_app(object())

    def test_ask_route_bounds_top_k_and_prompt_exposure(self):
        app = self._app(api_key="secret", max_top_k=2)
        ask = app.routes[("POST", "/ask")].endpoint

        with self.assertRaises(_FakeHTTPException) as top_k_error:
            asyncio.run(ask(AskRequest(query="hello", top_k=3)))
        self.assertEqual(top_k_error.exception.status_code, 422)
        with self.assertRaises(_FakeHTTPException) as prompt_error:
            asyncio.run(ask(AskRequest(query="hello", include_prompt=True)))
        self.assertEqual(prompt_error.exception.status_code, 403)
        self.assertNotIn("prompt", asyncio.run(ask(AskRequest(query="hello"))))

    def test_stream_permit_is_released_even_before_iteration_starts(self):
        app = self._app(max_concurrent_operations=1)
        stream = app.routes[("POST", "/stream")].endpoint

        first_response = asyncio.run(stream(AskRequest(query="first")))
        with self.assertRaises(_FakeHTTPException) as busy:
            asyncio.run(stream(AskRequest(query="second")))
        self.assertEqual(busy.exception.status_code, 503)

        asyncio.run(first_response.background())
        second_response = asyncio.run(stream(AskRequest(query="second")))
        asyncio.run(second_response.background())

    def test_stream_close_failure_does_not_leak_capacity_or_details(self):
        class BrokenCloseIterator:
            def __init__(self):
                self.done = False

            def __iter__(self):
                return self

            def __next__(self):
                if self.done:
                    raise StopIteration
                self.done = True
                return "safe"

            def close(self):
                raise RuntimeError("SECRET close detail")

        class BrokenCloseEngine(self._Engine):
            def stream(self, query, top_k=None):
                return BrokenCloseIterator()

        with _fake_fastapi_modules(), patch.dict(os.environ, {}, clear=True):
            app = create_app(BrokenCloseEngine(), max_concurrent_operations=1)
        stream = app.routes[("POST", "/stream")].endpoint

        first_response = asyncio.run(stream(AskRequest(query="first")))
        with self.assertLogs("cheragh.server.app", level="ERROR") as logs:
            self.assertEqual(list(first_response.content), ["safe"])
        self.assertNotIn("SECRET", "\n".join(logs.output))

        second_response = asyncio.run(stream(AskRequest(query="second")))
        asyncio.run(second_response.background())

    def test_index_route_redacts_runtime_failures_and_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = self._app(api_key="secret", enable_indexing=True, allowed_index_root=tmp)
            index = app.routes[("POST", "/index")].endpoint

            with self.assertRaises(_FakeHTTPException) as path_error:
                asyncio.run(index(IndexRequest(path="../outside", output="index")))
            self.assertEqual(path_error.exception.status_code, 400)
            self.assertEqual(path_error.exception.detail, "Invalid index path")

            with patch("cheragh.indexing.index_path", side_effect=RuntimeError("SECRET /private/index")):
                with self.assertLogs("cheragh.server.app", level="ERROR"):
                    with self.assertRaises(_FakeHTTPException) as runtime_error:
                        asyncio.run(index(IndexRequest(path=".", output="index")))
            self.assertEqual(runtime_error.exception.status_code, 500)
            self.assertEqual(runtime_error.exception.detail, "Indexing failed")
            self.assertNotIn("SECRET", runtime_error.exception.detail)


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI optional dependency is not installed")
class FastAPIServerIntegrationTests(unittest.TestCase):
    class _Response:
        def to_dict(self, *, include_prompt=False):
            payload = {"query": "q", "answer": "safe", "sources": []}
            if include_prompt:
                payload["prompt"] = "system prompt"
            return payload

    class _Engine:
        retriever = None
        cache_backend = None
        top_k = 5

        def ask(self, query, top_k=None):
            return FastAPIServerIntegrationTests._Response()

        def stream(self, query, top_k=None):
            yield "safe"

    def test_auth_validation_health_readiness_and_redaction(self):
        from fastapi.testclient import TestClient

        from cheragh.server.app import create_app

        with patch.dict(os.environ, {}, clear=True):
            app = create_app(
                self._Engine(),
                api_key="secret",
                require_auth=True,
                max_request_body_bytes=100,
                readiness_check=lambda: True,
            )
        client = TestClient(app, raise_server_exceptions=False)

        self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(client.get("/ready").json()["status"], "ready")
        unauthorized = client.post("/ask", json={"query": "hello"})
        self.assertEqual(unauthorized.status_code, 401)
        self.assertEqual(unauthorized.headers["www-authenticate"], "ApiKey")

        headers = {"X-API-Key": "secret", "X-Request-ID": "client-42"}
        self.assertEqual(client.post("/ask", headers=headers, json={"query": "   "}).status_code, 422)
        self.assertEqual(
            client.post("/ask", headers=headers, json={"query": "hello", "extra": True}).status_code,
            422,
        )
        prompt = client.post("/ask", headers=headers, json={"query": "hello", "include_prompt": True})
        self.assertEqual(prompt.status_code, 403)
        response = client.post("/ask", headers=headers, json={"query": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], "client-42")
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_indexing_configuration_fails_closed(self):
        from cheragh.server.app import create_app

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "API key"):
                create_app(self._Engine(), enable_indexing=True, allowed_index_root=tmp)
            with self.assertRaisesRegex(ValueError, "allowed_index_root"):
                create_app(self._Engine(), enable_indexing=True, api_key="secret")
            with self.assertRaisesRegex(ValueError, "authentication"):
                create_app(self._Engine(), require_auth=True)

    def test_engine_and_index_errors_do_not_leak_details(self):
        from fastapi.testclient import TestClient

        from cheragh.server.app import create_app

        class ExplodingEngine(self._Engine):
            def ask(self, query, top_k=None):
                raise RuntimeError("SECRET provider credential")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            ask_client = TestClient(
                create_app(ExplodingEngine(), api_key="secret"),
                raise_server_exceptions=False,
            )
            ask_response = ask_client.post("/ask", headers={"X-API-Key": "secret"}, json={"query": "hello"})
            self.assertEqual(ask_response.status_code, 500)
            self.assertNotIn("SECRET", ask_response.text)

            index_client = TestClient(
                create_app(
                    self._Engine(),
                    api_key="secret",
                    enable_indexing=True,
                    allowed_index_root=tmp,
                ),
                raise_server_exceptions=False,
            )
            with patch("cheragh.indexing.index_path", side_effect=RuntimeError("SECRET /private/index")):
                index_response = index_client.post(
                    "/index",
                    headers={"X-API-Key": "secret"},
                    json={"path": ".", "output": "index"},
                )
            self.assertEqual(index_response.status_code, 500)
            self.assertNotIn("SECRET", index_response.text)


if __name__ == "__main__":
    unittest.main()
