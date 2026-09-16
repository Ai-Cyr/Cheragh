"""HTTP regressions for readiness results and bounded response conversion."""
from __future__ import annotations

import asyncio
import importlib.util
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cheragh.server.app import create_app


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None, reason="FastAPI optional dependency is not installed",
)


class _Engine:
    top_k = 5

    def ask(self, query, top_k=None):
        return SimpleNamespace(to_dict=lambda **kwargs: {"answer": "safe", "sources": []})

    def stream(self, query, top_k=None):
        yield "safe"


@pytest.fixture(autouse=True)
def clean_server_environment(monkeypatch):
    for name in ("CHERAGH_API_KEY", "CHERAGH_REQUIRE_AUTH", "CHERAGH_ENABLE_INDEXING", "CHERAGH_INDEX_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_response_conversion_is_bounded_and_does_not_block_probes():
    import httpx

    started = threading.Event()
    release = threading.Event()

    class Response:
        def to_dict(self, **kwargs):
            started.set()
            release.wait(timeout=2)
            return {"answer": "safe", "sources": []}

    engine = _Engine()
    engine.ask = lambda *args, **kwargs: Response()
    app = create_app(engine, max_concurrent_operations=1, request_timeout_seconds=0.03)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            try:
                response = await asyncio.wait_for(client.post("/ask", json={"query": "q"}), timeout=0.5)
                assert started.is_set()
                assert response.status_code == 504
                assert (await asyncio.wait_for(client.get("/health"), timeout=0.5)).status_code == 200
                assert (await client.post("/ask", json={"query": "next"})).status_code == 503
                assert (await client.post("/stream", json={"query": "next"})).status_code == 503
            finally:
                release.set()

            async def retry():
                while True:
                    response = await client.post("/ask", json={"query": "done"})
                    if response.status_code != 503:
                        return response
                    await asyncio.sleep(0.005)

            assert (await asyncio.wait_for(retry(), timeout=1)).status_code == 200

    asyncio.run(scenario())


async def _async_readiness():
    return False


class _AsyncReadiness:
    async def __call__(self):
        return False


def _generator_readiness():
    yield False


@pytest.mark.parametrize("callback", [_async_readiness, _AsyncReadiness(), _generator_readiness])
def test_invalid_readiness_callbacks_fail_before_provider_loading(callback):
    with patch("cheragh.server.app.RAGEngine.from_config", return_value=_Engine()) as load:
        with pytest.raises(TypeError, match="readiness_check"):
            create_app(config_path="provider.yaml", readiness_check=callback)
    load.assert_not_called()


@pytest.mark.parametrize("result", ["false", {"ready": False}, 1, object()])
def test_readiness_requires_a_boolean_result(result):
    from fastapi.testclient import TestClient

    with TestClient(create_app(_Engine(), readiness_check=lambda: result)) as client:
        response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_disguised_async_readiness_does_not_report_ready_or_leak_coroutines():
    from fastapi.testclient import TestClient

    pending = _async_readiness()
    try:
        with TestClient(create_app(_Engine(), readiness_check=lambda: pending)) as client:
            response = client.get("/ready")
        assert response.status_code == 503
        assert pending.cr_frame is None, "an invalid callback's coroutine must be closed"
    finally:
        pending.close()
