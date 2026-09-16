"""Regression coverage for monitored workloads and server startup configuration."""
from __future__ import annotations

import asyncio
import importlib.util
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from cheragh.cli.main import main
from cheragh.server.app import create_app
from cheragh.server.main import serve


class _Engine:
    top_k = 5
    retriever = None
    cache_backend = None

    def ask(self, query, top_k=None):
        return SimpleNamespace(to_dict=lambda **_: {"answer": "safe", "sources": []})

    def stream(self, query, top_k=None):
        yield "safe"


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"max_concurrent_operations": 8, "max_server_connections": 8}, "max_server_connections"),
        ({"max_server_connections": True}, "max_server_connections"),
        ({"max_concurrent_operations": 0}, "max_concurrent_operations"),
        ({"port": 0}, "port"),
        ({"port": 65_536}, "port"),
        ({"port": True}, "port"),
        ({"port": 8000.5}, "port"),
        ({"host": "  "}, "host"),
        ({"host": None}, "host"),
    ],
)
def test_invalid_listener_configuration_precedes_provider_initialization(options, expected):
    uvicorn = SimpleNamespace(run=Mock())
    with patch.dict("sys.modules", {"uvicorn": uvicorn}), patch("cheragh.server.app.create_app") as factory:
        with pytest.raises(ValueError, match=expected):
            serve(config="provider-config.yaml", **options)
    factory.assert_not_called()
    uvicorn.run.assert_not_called()


@pytest.mark.parametrize("options, expected", [([], None), (["--enable-indexing"], True), (["--no-enable-indexing"], False)])
def test_cli_preserves_environment_default_and_explicit_indexing_overrides(options, expected):
    with patch("cheragh.server.main.serve") as run:
        assert main(["serve", "--config", "rag.yaml", *options]) == 0
    assert run.call_args.kwargs["enable_indexing"] is expected


def test_serve_preserves_environment_indexing_default():
    uvicorn = SimpleNamespace(run=Mock())
    with patch.dict("sys.modules", {"uvicorn": uvicorn}), patch("cheragh.server.app.create_app") as factory:
        serve(config="rag.yaml")
    assert factory.call_args.kwargs["enable_indexing"] is None


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="FastAPI optional dependency is not installed")
def test_stats_timeout_retains_shared_capacity_and_keeps_probes_responsive(monkeypatch):
    import httpx

    for variable in ("CHERAGH_ENABLE_INDEXING", "CHERAGH_REQUIRE_AUTH", "CHERAGH_API_KEY", "CHERAGH_INDEX_ROOT"):
        monkeypatch.delenv(variable, raising=False)
    started = threading.Event()
    release = threading.Event()

    def slow_stats():
        started.set()
        assert release.wait(timeout=5), "test did not release the cache worker"
        return SimpleNamespace(to_dict=lambda: {"hits": 3})

    engine = _Engine()
    engine.cache_backend = SimpleNamespace(stats=slow_stats)
    app = create_app(engine, max_concurrent_operations=1, request_timeout_seconds=0.05)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            try:
                response = await client.get("/stats")
                assert started.is_set()
                assert response.status_code == 504
                assert response.json()["detail"] == "Request timed out"

                # The timed-out cache backend still owns the same permit as
                # asks and streams, preventing unbounded abandoned work.
                for route in ("/ask", "/stream"):
                    busy = await client.post(route, json={"query": "q"})
                    assert busy.status_code == 503
                    assert busy.headers["retry-after"] == "1"
                assert (await client.get("/stats")).status_code == 503
                assert (await client.get("/health")).status_code == 200
                assert (await client.get("/ready")).status_code == 200
            finally:
                release.set()

            async def capacity_is_reusable():
                while True:
                    response = await client.post("/ask", json={"query": "q"})
                    if response.status_code != 503:
                        return response
                    await asyncio.sleep(0.005)

            assert (await asyncio.wait_for(capacity_is_reusable(), timeout=2)).status_code == 200
            response = await client.get("/stats")
            assert response.status_code == 200
            assert response.json() == {"document_count": None, "top_k": 5, "cache": {"hits": 3}}

    asyncio.run(scenario())


@pytest.mark.skipif(importlib.util.find_spec("fastapi") is None, reason="FastAPI optional dependency is not installed")
def test_indexing_environment_and_cli_override_reach_the_http_endpoint(monkeypatch, tmp_path):
    import httpx

    monkeypatch.setenv("CHERAGH_ENABLE_INDEXING", "true")
    monkeypatch.setenv("CHERAGH_REQUIRE_AUTH", "true")
    monkeypatch.setenv("CHERAGH_API_KEY", "secret")
    monkeypatch.setenv("CHERAGH_INDEX_ROOT", str(tmp_path))

    async def scenario():
        for flag, expected in ((None, 200), (False, 403)):
            app = create_app(_Engine(), enable_indexing=flag)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
                with patch("cheragh.indexing.index_path", return_value={"indexed": 1}) as index:
                    response = await client.post("/index", headers={"X-API-Key": "secret"}, json={"path": "."})
                assert response.status_code == expected
                assert index.called is (flag is None)

    asyncio.run(scenario())
