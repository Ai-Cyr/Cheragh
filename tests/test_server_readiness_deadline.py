"""Readiness extensions share bounded work without blocking liveness probes."""
import asyncio
import importlib.util
import threading

import pytest

from cheragh.server.app import create_app


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None, reason="FastAPI optional dependency is not installed",
)


def test_blocked_readiness_is_bounded_and_does_not_block_liveness(monkeypatch):
    import httpx

    for name in ("CHERAGH_API_KEY", "CHERAGH_REQUIRE_AUTH", "CHERAGH_ENABLE_INDEXING", "CHERAGH_INDEX_ROOT"):
        monkeypatch.delenv(name, raising=False)
    started = threading.Event()
    release = threading.Event()

    def readiness():
        started.set()
        release.wait(timeout=2)
        return True

    class Engine:
        def ask(self, *args, **kwargs):
            raise AssertionError("capacity is reserved by readiness")

        def stream(self, *args, **kwargs):
            yield "unused"

    app = create_app(
        Engine(), readiness_check=readiness, max_concurrent_operations=1, request_timeout_seconds=0.03,
    )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            try:
                response = await asyncio.wait_for(client.get("/ready"), timeout=0.5)
                assert started.is_set()
                assert response.status_code == 503
                assert response.json() == {"status": "not_ready"}
                assert (await asyncio.wait_for(client.get("/health"), timeout=0.5)).status_code == 200
                assert (await client.get("/ready")).status_code == 503
                assert (await client.post("/ask", json={"query": "q"})).status_code == 503
            finally:
                release.set()

    asyncio.run(scenario())
