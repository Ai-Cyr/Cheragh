"""Bundled UI and authenticated local-index activation contracts."""
from __future__ import annotations

import asyncio
import importlib.util
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from cheragh import Document, HashingEmbedding, MemoryVectorStore, RAGEngine
from cheragh.base import EmbeddingModel
from cheragh.server.app import create_app


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("fastapi") is None, reason="FastAPI optional dependency is not installed",
)
AUTH = {"X-API-Key": "test-session-secret"}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ("CHERAGH_API_KEY", "CHERAGH_REQUIRE_AUTH", "CHERAGH_ENABLE_INDEXING", "CHERAGH_INDEX_ROOT"):
        monkeypatch.delenv(name, raising=False)


def local_index(tmp_path: Path) -> Path:
    store = MemoryVectorStore(HashingEmbedding(dimension=64))
    store.add_documents([Document("The deployment identifier is old-lighthouse.", doc_id="old")])
    index = tmp_path / "index"
    store.save(index)
    (tmp_path / "corpus.txt").write_text("The deployment identifier is new-orchid.", encoding="utf-8")
    return index


def local_app(tmp_path, **options):
    index = local_index(tmp_path)
    return create_app(
        index_path=str(index), api_key=AUTH["X-API-Key"],
        enable_indexing=True, allowed_index_root=tmp_path, **options,
    )


def index_body(**options):
    return {"path": "corpus.txt", "output": "index", "activate": True, **options}


def test_browser_shell_and_known_assets_are_public_but_contain_no_session_secrets():
    from fastapi.testclient import TestClient

    engine = RAGEngine(MemoryVectorStore(HashingEmbedding()).as_retriever())
    with TestClient(create_app(engine, api_key=AUTH["X-API-Key"])) as client:
        for path, content_type in (("/ui/", "text/html"), ("/ui/app.css", "text/css"), ("/ui/app.js", "javascript")):
            response = client.get(path)
            assert response.status_code == 200
            assert content_type in response.headers["content-type"]
            assert AUTH["X-API-Key"] not in response.text
            policy = response.headers["content-security-policy"]
            assert "script-src 'self'" in policy
            assert "frame-ancestors 'none'" in policy
            assert "unsafe-inline" not in policy
            assert response.headers["cache-control"] == "no-store"
            assert response.headers["referrer-policy"] == "no-referrer"
        for path in ("/ui/pyproject.toml", "/ui/missing.js", "/ui/%2e%2e/app.py"):
            assert client.get(path).status_code == 404


def test_relative_redirects_and_assets_preserve_proxy_prefix():
    from fastapi.testclient import TestClient

    engine = RAGEngine(MemoryVectorStore(HashingEmbedding()).as_retriever())
    with TestClient(create_app(engine), root_path="/cheragh", base_url="http://testserver") as client:
        assert client.get("/cheragh/", follow_redirects=False).headers["location"] == "ui/"
        assert client.get("/cheragh/ui", follow_redirects=False).headers["location"] == "ui/"
        response = client.get("/cheragh/")
        assert response.status_code == 200
        assert response.url.path == "/cheragh/ui/"
        assert client.get("/cheragh/ui/app.js").status_code == 200


def test_ui_settings_and_catalogue_share_api_authentication_and_hide_absolute_paths(tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(local_app(tmp_path, max_top_k=3)) as client:
        for path in ("/ui/api/settings", "/ui/api/techniques"):
            assert client.get(path).status_code == 401
            assert client.get(path, headers={"X-API-Key": "wrong"}).status_code == 401
        settings = client.get("/ui/api/settings", headers=AUTH)
        assert str(tmp_path) not in settings.text
        assert AUTH["X-API-Key"] not in settings.text
        assert settings.json()["index_output"] == "index"
        assert settings.json()["can_activate_index"] is True
        assert settings.json()["default_top_k"] == 3
        assert settings.json()["auth_required"] is True
        catalogue = client.get("/ui/api/techniques", headers=AUTH).json()
        assert len(catalogue) == 46
        assert {item["status"] for item in catalogue} >= {"stable", "beta", "experimental"}


def test_disabled_indexing_stays_disabled_in_ui_and_api(tmp_path):
    from fastapi.testclient import TestClient

    index = local_index(tmp_path)
    with TestClient(create_app(index_path=str(index), api_key=AUTH["X-API-Key"])) as client:
        settings = client.get("/ui/api/settings", headers=AUTH).json()
        assert settings["indexing_enabled"] is False
        assert settings["can_activate_index"] is False
        assert client.post("/index", headers=AUTH, json=index_body()).status_code == 403


def test_successful_index_activation_updates_answers_without_restart(tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(local_app(tmp_path)) as client:
        assert client.post("/index", json=index_body()).status_code == 401
        before = client.post("/ask", headers=AUTH, json={"query": "deployment identifier"}).json()
        assert "old-lighthouse" in before["answer"]
        result = client.post("/index", headers=AUTH, json=index_body())
        assert result.status_code == 200, result.text
        assert result.json()["activated"] is True
        after = client.post("/ask", headers=AUTH, json={"query": "deployment identifier"}).json()
        assert "new-orchid" in after["answer"]
        assert "old-lighthouse" not in after["answer"]
        assert client.get("/stats", headers=AUTH).json()["document_count"] == 1
    assert MemoryVectorStore.load(tmp_path / "index").embedding_model.dimension == 64


def test_index_without_activation_does_not_replace_serving_engine(tmp_path):
    from fastapi.testclient import TestClient

    with TestClient(local_app(tmp_path)) as client:
        result = client.post("/index", headers=AUTH, json=index_body(activate=False, output="other"))
        assert result.status_code == 200
        assert result.json()["activated"] is False
        answer = client.post("/ask", headers=AUTH, json={"query": "deployment identifier"}).json()["answer"]
        assert "old-lighthouse" in answer


@pytest.mark.parametrize("output", ["other", "../outside"])
def test_activation_rejects_other_indexes_and_path_escape_before_writes(tmp_path, output):
    from fastapi.testclient import TestClient

    with TestClient(local_app(tmp_path)) as client, patch("cheragh.indexing.index_path") as build:
        result = client.post("/index", headers=AUTH, json=index_body(output=output))
        assert result.status_code == 400
        build.assert_not_called()


@pytest.mark.parametrize("mode", ["engine", "config"])
def test_activation_cannot_discard_custom_engine_or_configured_provider(tmp_path, mode):
    from fastapi.testclient import TestClient

    local_index(tmp_path)
    engine = RAGEngine(MemoryVectorStore(HashingEmbedding()).as_retriever())
    with patch("cheragh.server.app.RAGEngine.from_config", return_value=engine):
        source = {"engine": engine} if mode == "engine" else {"config_path": "provider.yaml"}
        app = create_app(**source, api_key=AUTH["X-API-Key"], enable_indexing=True, allowed_index_root=tmp_path)
    with TestClient(app) as client, patch("cheragh.indexing.index_path") as build:
        assert client.get("/ui/api/settings", headers=AUTH).json()["can_activate_index"] is False
        assert client.post("/index", headers=AUTH, json=index_body()).status_code == 400
        build.assert_not_called()


def test_live_activation_does_not_share_custom_embedding_transports(tmp_path):
    from fastapi.testclient import TestClient

    class CustomEmbedding(EmbeddingModel):
        def embed_documents(self, texts):
            return HashingEmbedding(dimension=64).embed_documents(texts)

        def embed_query(self, text):
            return HashingEmbedding(dimension=64).embed_query(text)

    embedding = CustomEmbedding()
    store = MemoryVectorStore(embedding)
    store.add_documents([Document("Existing corpus", doc_id="old")])
    store.save(tmp_path / "index")
    app = create_app(
        index_path=str(tmp_path / "index"), index_embedding_model=embedding,
        api_key=AUTH["X-API-Key"], enable_indexing=True, allowed_index_root=tmp_path,
    )
    with TestClient(app) as client, patch("cheragh.indexing.index_path") as build:
        assert client.get("/ui/api/settings", headers=AUTH).json()["can_activate_index"] is False
        assert client.post("/index", headers=AUTH, json=index_body()).status_code == 400
        build.assert_not_called()


def test_timed_out_index_worker_never_activates_and_retains_write_lease(tmp_path):
    import httpx
    from cheragh.indexing import index_path as build_index

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    app = local_app(tmp_path, index_timeout_seconds=0.03, max_concurrent_operations=2)

    def slow_build(*args, **kwargs):
        started.set()
        release.wait(timeout=3)
        try:
            return build_index(*args, **kwargs)
        finally:
            finished.set()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            try:
                with patch("cheragh.indexing.index_path", side_effect=slow_build):
                    response = await client.post("/index", headers=AUTH, json=index_body())
                    assert response.status_code == 504
                    assert started.is_set()
                    blocked = await client.post("/index", headers=AUTH, json=index_body())
                    assert blocked.status_code == 503
                    assert blocked.headers["retry-after"] == "1"
                    assert (await client.get("/health")).status_code == 200
                    release.set()
                    assert await asyncio.to_thread(finished.wait, 2)
                # Wait for the worker's candidate load/lease cleanup as well.
                for _ in range(100):
                    result = await client.get("/stats", headers=AUTH)
                    if result.status_code == 200:
                        break
                    await asyncio.sleep(0.005)
                answer = (await client.post("/ask", headers=AUTH, json={"query": "deployment identifier"})).json()
                assert "old-lighthouse" in answer["answer"]
                assert "new-orchid" in MemoryVectorStore.load(tmp_path / "index").documents[0].content
            finally:
                release.set()

    asyncio.run(scenario())


def test_inflight_queries_keep_their_original_engine_during_activation(tmp_path):
    import httpx

    app = local_app(tmp_path)
    started = threading.Event()
    release = threading.Event()
    original_ask = RAGEngine.ask

    def slow_ask(self, query, **kwargs):
        if query == "old query":
            started.set()
            release.wait(timeout=3)
        return original_ask(self, "deployment identifier", **kwargs)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            old_request = asyncio.create_task(client.post("/ask", headers=AUTH, json={"query": "old query"}))
            try:
                assert await asyncio.to_thread(started.wait, 2)
                result = await client.post("/index", headers=AUTH, json=index_body())
                assert result.status_code == 200
                current = await client.post("/ask", headers=AUTH, json={"query": "current"})
                assert "new-orchid" in current.json()["answer"]
            finally:
                release.set()
            assert "old-lighthouse" in (await old_request).json()["answer"]

    with patch.object(RAGEngine, "ask", slow_ask):
        asyncio.run(scenario())


def test_failed_candidate_load_preserves_the_serving_engine(tmp_path):
    from fastapi.testclient import TestClient

    app = local_app(tmp_path)
    with TestClient(app) as client, patch("cheragh.indexing.index_path", return_value={"indexed_documents": 1}):
        with patch("cheragh.server.app.MemoryVectorStore.load", side_effect=ValueError("bad snapshot")):
            assert client.post("/index", headers=AUTH, json=index_body()).status_code == 400
        answer = client.post("/ask", headers=AUTH, json={"query": "deployment identifier"}).json()["answer"]
        assert "old-lighthouse" in answer
        # The failed load must also release the writer admission lease.
        assert client.post("/index", headers=AUTH, json=index_body()).status_code == 200


def test_queued_index_worker_does_not_start_writes_after_timeout(tmp_path):
    import httpx

    app = local_app(tmp_path, index_timeout_seconds=0.03)
    blocker_started = threading.Event()
    release = threading.Event()

    def occupy_executor():
        blocker_started.set()
        release.wait(timeout=3)

    async def scenario():
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
        blocker = loop.run_in_executor(None, occupy_executor)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            try:
                for _ in range(100):
                    if blocker_started.is_set():
                        break
                    await asyncio.sleep(0.005)
                assert blocker_started.is_set()
                with patch("cheragh.indexing.index_path") as build:
                    response = await client.post("/index", headers=AUTH, json=index_body())
                    assert response.status_code == 504
                    release.set()
                    await blocker
                    # FIFO marker runs after the abandoned queued worker.
                    await loop.run_in_executor(None, lambda: None)
                    build.assert_not_called()
                answer = (await client.post("/ask", headers=AUTH, json={"query": "deployment identifier"})).json()
                assert "old-lighthouse" in answer["answer"]
                with patch("cheragh.indexing.index_path", return_value={"indexed_documents": 0}):
                    result = await client.post("/index", headers=AUTH, json=index_body(activate=False))
                assert result.status_code == 200
            finally:
                release.set()
                await blocker

    asyncio.run(scenario())


def test_cancelled_index_request_does_not_activate_and_keeps_worker_accounted(tmp_path):
    import httpx
    from cheragh.indexing import index_path as build_index

    app = local_app(tmp_path, max_concurrent_operations=2)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def slow_build(*args, **kwargs):
        started.set()
        release.wait(timeout=3)
        try:
            return build_index(*args, **kwargs)
        finally:
            finished.set()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client:
            with patch("cheragh.indexing.index_path", side_effect=slow_build):
                pending = asyncio.create_task(client.post("/index", headers=AUTH, json=index_body()))
                try:
                    assert await asyncio.to_thread(started.wait, 2)
                    pending.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    assert (await client.post("/index", headers=AUTH, json=index_body())).status_code == 503
                    release.set()
                    assert await asyncio.to_thread(finished.wait, 2)
                    answer = (await client.post("/ask", headers=AUTH, json={"query": "deployment identifier"})).json()
                    assert "old-lighthouse" in answer["answer"]
                finally:
                    release.set()
                    if not pending.done():
                        await pending

    asyncio.run(scenario())
