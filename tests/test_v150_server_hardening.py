"""Regression checks for request context, effective limits, and disconnect cleanup."""
from __future__ import annotations

import asyncio
import contextvars
import importlib.util
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from cheragh.server.app import AskRequest, _LeasedStream, _OperationLimiter, create_app


class OperationContextTests(unittest.TestCase):
    def test_concurrent_workers_inherit_isolated_request_context(self):
        tenant = contextvars.ContextVar("request_tenant", default="unscoped")
        barrier = threading.Barrier(2)

        async def scenario():
            limiter = _OperationLimiter(2)

            async def request(identity):
                token = tenant.set(identity)
                try:
                    def operation():
                        barrier.wait(timeout=2)
                        current_identity = tenant.get()
                        tenant.set("worker-local")
                        return current_identity

                    result = await limiter.run(operation, timeout_seconds=3)
                    self.assertEqual(tenant.get(), identity)
                    return result
                finally:
                    tenant.reset(token)

            return await asyncio.gather(request("tenant-a"), request("tenant-b"))

        self.assertEqual(asyncio.run(scenario()), ["tenant-a", "tenant-b"])
        self.assertEqual(tenant.get(), "unscoped")


class StreamLeaseTests(unittest.TestCase):
    def test_stream_context_survives_chunk_and_close_worker_changes(self):
        tenant = contextvars.ContextVar("stream_tenant", default="unscoped")
        cleanup_identity = []

        def provider():
            self.assertEqual(tenant.get(), "tenant-a")
            token = tenant.set("provider-scope")
            try:
                yield "first"
                yield tenant.get()
            finally:
                tenant.reset(token)
                cleanup_identity.append(tenant.get())

        limiter = _OperationLimiter(1)
        token = tenant.set("tenant-a")
        stream = _LeasedStream(provider(), limiter.reserve())
        tenant.reset(token)
        with ThreadPoolExecutor(max_workers=1) as first_worker, ThreadPoolExecutor(max_workers=1) as second_worker:
            self.assertEqual(first_worker.submit(next, stream).result(timeout=2), "first")
            self.assertEqual(second_worker.submit(next, stream).result(timeout=2), "provider-scope")
            second_worker.submit(stream.close).result(timeout=2)
            self.assertEqual(first_worker.submit(tenant.get).result(timeout=2), "unscoped")
        self.assertEqual(cleanup_identity, ["tenant-a"])
        self.assertEqual(tenant.get(), "unscoped")

    def test_cleanup_waits_for_inflight_next_and_closes_provider_once(self):
        started = threading.Event()
        unblock = threading.Event()
        closed = threading.Event()
        close_calls = []

        def provider():
            try:
                started.set()
                unblock.wait(timeout=3)
                yield "chunk"
            finally:
                close_calls.append(1)
                closed.set()

        limiter = _OperationLimiter(1)
        stream = _LeasedStream(provider(), limiter.reserve())
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(next, stream)
            try:
                self.assertTrue(started.wait(timeout=2))
                stream.close()
                stream.close()
                self.assertFalse(closed.is_set())
                self.assertIsNone(limiter.reserve(), "in-flight provider work lost its capacity permit")
            finally:
                unblock.set()
            self.assertEqual(future.result(timeout=2), "chunk")

        self.assertTrue(closed.is_set())
        self.assertEqual(close_calls, [1])
        replacement = limiter.reserve()
        self.assertIsNotNone(replacement)
        replacement.release()
        with self.assertRaises(StopIteration):
            next(stream)


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "FastAPI optional dependency is not installed")
class ServerHardeningIntegrationTests(unittest.TestCase):
    class Response:
        def to_dict(self, *, include_prompt=False):
            return {"answer": "safe", "sources": []}

    class Engine:
        top_k = 100

        def __init__(self):
            self.calls = []
            self.stream_closed = False

        def ask(self, query, top_k=None):
            self.calls.append(("ask", top_k))
            return ServerHardeningIntegrationTests.Response()

        def stream(self, query, top_k=None):
            self.calls.append(("stream", top_k))
            try:
                yield "first"
                yield "second"
            finally:
                self.stream_closed = True

    def app(self, engine=None, **kwargs):
        with patch.dict(os.environ, {}, clear=True):
            return create_app(engine or self.Engine(), **kwargs)

    def test_omitted_top_k_cannot_bypass_server_limit(self):
        from fastapi.testclient import TestClient

        engine = self.Engine()
        with TestClient(self.app(engine, max_top_k=3)) as client:
            self.assertEqual(client.post("/ask", json={"query": "q"}).status_code, 200)
            self.assertEqual(client.post("/stream", json={"query": "q"}).text, "firstsecond")
            self.assertEqual(client.post("/ask", json={"query": "q", "top_k": 2}).status_code, 200)
        self.assertEqual(engine.calls, [("ask", 3), ("stream", 3), ("ask", 2)])

    def test_security_validation_precedes_provider_and_index_loading(self):
        with patch.dict(os.environ, {}, clear=True), patch(
            "cheragh.server.app.RAGEngine.from_config"
        ) as load_config, patch("cheragh.server.app.MemoryVectorStore.load") as load_index:
            with self.assertRaisesRegex(ValueError, "authentication"):
                create_app(config_path="provider-config.yaml", require_auth=True)
            with self.assertRaisesRegex(ValueError, "max_concurrent_operations"):
                create_app(index_path="private-index", max_concurrent_operations=0)
        load_config.assert_not_called()
        load_index.assert_not_called()

    def test_chunked_oversize_body_retains_413_through_fastapi_parser(self):
        from fastapi.testclient import TestClient

        engine = self.Engine()
        with TestClient(self.app(engine, max_request_body_bytes=32)) as client:
            response = client.post(
                "/ask",
                content=iter([b'{"query":"', b"x" * 40, b'"}']),
                headers={"Content-Type": "application/json", "X-Request-ID": "body-limit"},
            )
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "Request body too large")
        self.assertEqual(response.headers["x-request-id"], "body-limit")
        self.assertEqual(engine.calls, [])

    def test_send_failure_closes_suspended_provider_and_releases_capacity(self):
        engine = self.Engine()
        app = self.app(engine, max_concurrent_operations=1)
        endpoint = next(route.endpoint for route in app.routes if route.path == "/stream")

        async def scenario():
            response = await endpoint(AskRequest(query="q"))

            async def receive():
                await asyncio.Future()

            async def send(message):
                if message["type"] == "http.response.body":
                    raise OSError("client disconnected")

            # Starlette versions differ in whether transport errors are wrapped
            # in ClientDisconnect or an exception group; cleanup must be stable.
            with self.assertRaises(Exception):
                await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            self.assertTrue(engine.stream_closed)
            replacement = await endpoint(AskRequest(query="next"))
            await replacement.background()

        asyncio.run(scenario())

    def test_cancelled_response_before_first_chunk_releases_capacity(self):
        app = self.app(max_concurrent_operations=1)
        endpoint = next(route.endpoint for route in app.routes if route.path == "/stream")

        async def scenario():
            response = await endpoint(AskRequest(query="q"))
            started = asyncio.Event()

            async def receive():
                await asyncio.Future()

            async def send(message):
                started.set()
                await asyncio.Future()

            task = asyncio.create_task(
                response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
            )
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            replacement = await endpoint(AskRequest(query="next"))
            await replacement.background()

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
