import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from cheragh import Document, RAGEngine
from cheragh.base import BaseRetriever, LLMClient


class Retriever(BaseRetriever):
    def retrieve(self, query, top_k=5):
        return [Document("production context", doc_id="doc")]


def test_async_stream_keeps_event_loop_responsive():
    released = threading.Event()
    loop_thread = threading.get_ident()

    class Provider(LLMClient):
        def generate(self, prompt, **kwargs):
            raise AssertionError("expected streaming")

        def stream(self, prompt, **kwargs):
            assert threading.get_ident() != loop_thread
            assert released.wait(2), "provider blocked the event loop"
            yield "answer"

    async def run():
        async def release():
            await asyncio.sleep(0.01)
            released.set()

        task = asyncio.create_task(release())
        try:
            engine = RAGEngine(Retriever(), Provider())
            assert [chunk async for chunk in engine.astream("question")] == ["answer"]
        finally:
            released.set()
            await task

    asyncio.run(run())


@pytest.mark.parametrize("fail_after_cancel", [False, True])
def test_cancelled_async_stream_closes_after_inflight_next(fail_after_cancel):
    started = threading.Event()
    released = threading.Event()
    closed = threading.Event()
    request_context = contextvars.ContextVar("request_context", default="outside")
    failures = []

    class Provider(LLMClient):
        def generate(self, prompt, **kwargs):
            raise AssertionError("expected streaming")

        def stream(self, prompt, **kwargs):
            token = request_context.set("provider")
            try:
                yield "first"
                started.set()
                assert released.wait(2)
                if fail_after_cancel:
                    raise RuntimeError("private provider failure")
                yield "second"
            finally:
                try:
                    request_context.reset(token)
                except ValueError as exc:
                    failures.append(exc)
                closed.set()

    async def run():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda _, context: failures.append(context))
        stream = RAGEngine(Retriever(), Provider()).astream("question")
        assert await anext(stream) == "first"
        pending = asyncio.create_task(anext(stream))
        try:
            assert await asyncio.to_thread(started.wait, 1)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert not closed.is_set()
            assert request_context.get() == "outside"
        finally:
            released.set()
            await stream.aclose()
            assert await asyncio.to_thread(closed.wait, 1)
        await asyncio.sleep(0)

    asyncio.run(run())
    assert failures == []


def test_async_stream_early_close_releases_provider():
    closed = threading.Event()

    class Provider(LLMClient):
        def generate(self, prompt, **kwargs):
            raise AssertionError("expected streaming")

        def stream(self, prompt, **kwargs):
            try:
                yield "first"
                yield "second"
            finally:
                closed.set()

    async def run():
        stream = RAGEngine(Retriever(), Provider()).astream("question")
        assert await anext(stream) == "first"
        await stream.aclose()
        assert closed.is_set()

    asyncio.run(run())


def test_async_stream_propagates_provider_error():
    class Provider(LLMClient):
        def generate(self, prompt, **kwargs):
            raise ValueError("provider error")

    async def run():
        stream = RAGEngine(Retriever(), Provider()).astream("question")
        with pytest.raises(ValueError, match="provider error"):
            await anext(stream)

    asyncio.run(run())


def test_cancellation_does_not_wait_for_saturated_executor():
    started = threading.Event()
    released = threading.Event()
    closed = threading.Event()

    class Provider(LLMClient):
        def generate(self, prompt, **kwargs):
            raise AssertionError("expected streaming")

        def stream(self, prompt, **kwargs):
            try:
                started.set()
                assert released.wait(2)
                yield "answer"
            finally:
                closed.set()

    async def run():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        stream = RAGEngine(Retriever(), Provider()).astream("question")
        pending = asyncio.create_task(anext(stream))
        try:
            for _ in range(100):
                if started.is_set():
                    break
                await asyncio.sleep(0.001)
            assert started.is_set()
            pending.cancel()
            done, _ = await asyncio.wait({pending}, timeout=0.1)
            assert pending in done, "cancellation waited for the blocked provider"
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert not closed.is_set()
        finally:
            released.set()
            await stream.aclose()
        assert await asyncio.to_thread(closed.wait, 1)

    asyncio.run(run())
