"""Bridge synchronous provider iterators to asyncio without blocking its loop."""
from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

from .base import _close_stream


class _StreamWorker:
    def __init__(self, iterator: Iterator[str]):
        self.iterator = iterator
        self.context = contextvars.copy_context()
        self.lock = threading.Lock()
        self.running = False
        self.closing = False
        self.closed = False

    def advance(self) -> tuple[bool, str]:
        with self.lock:
            if self.closing or self.closed:
                return False, ""
            self.running = True
        try:
            return True, self.context.run(next, self.iterator)
        except StopIteration:
            with self.lock:
                self.closing = True
            return False, ""
        except BaseException:
            with self.lock:
                self.closing = True
            raise
        finally:
            with self.lock:
                self.running = False
                close_now = self.closing and not self.closed
                if close_now:
                    self.closed = True
            if close_now:
                self.context.run(_close_stream, self.iterator)

    def request_close(self) -> bool:
        """Claim idle cleanup, or leave it to the active worker without waiting."""
        with self.lock:
            self.closing = True
            if self.running or self.closed:
                # next() cannot be interrupted safely; its worker owns cleanup.
                return False
            self.closed = True
            return True

    def close(self) -> None:
        self.context.run(_close_stream, self.iterator)


def _consume_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


async def _iterate_in_worker(iterator: Iterator[str]) -> AsyncIterator[str]:
    worker = _StreamWorker(iterator)
    loop = asyncio.get_running_loop()
    try:
        while True:
            future = loop.run_in_executor(None, worker.advance)
            # A cancelled consumer must not leave a later provider failure
            # unobserved (including potentially sensitive exception messages).
            future.add_done_callback(_consume_exception)
            available, chunk = await asyncio.shield(future)
            if not available:
                break
            yield chunk
    finally:
        # Do not queue behind a blocked next() in a saturated executor just to
        # mark it cancelled. Its worker closes when that call actually exits.
        if worker.request_close():
            closing = loop.run_in_executor(None, worker.close)
            closing.add_done_callback(_consume_exception)
            await asyncio.shield(closing)
