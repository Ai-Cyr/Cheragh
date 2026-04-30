"""FastAPI application factory for serving a RAGEngine."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .. import __version__
from ..base import HashingEmbedding
from ..engine import RAGEngine
from ..vectorstores.memory import MemoryVectorStore


def create_app(
    engine: RAGEngine | None = None,
    *,
    config_path: str | None = None,
    index_path: str | None = None,
    enable_indexing: bool | None = None,
    allowed_index_root: str | Path | None = None,
    api_key: str | None = None,
    max_top_k: int = 50,
):
    """Create a FastAPI app around a RAG engine.

    ``fastapi`` is an optional dependency: ``pip install cheragh[fastapi]``.

    Runtime hardening defaults:
    - ``/index`` is disabled unless ``enable_indexing=True`` or
      ``CHERAGH_ENABLE_INDEXING=true``.
    - When enabled, ``/index`` can only read/write below ``allowed_index_root``
      or ``CHERAGH_INDEX_ROOT``.
    - Set ``api_key`` or ``CHERAGH_API_KEY`` to require ``X-API-Key`` on
      API endpoints.
    """
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("The server requires FastAPI. Install with: pip install cheragh[fastapi]") from exc

    if engine is None:
        if config_path:
            engine = RAGEngine.from_config(config_path)
        elif index_path:
            store = MemoryVectorStore.load(index_path, HashingEmbedding())
            engine = RAGEngine(store.as_retriever())
        else:
            raise ValueError("create_app requires engine, config_path or index_path")

    env_enable_indexing = _as_bool(os.getenv("CHERAGH_ENABLE_INDEXING"), default=False)
    indexing_enabled = env_enable_indexing if enable_indexing is None else enable_indexing
    root = Path(allowed_index_root or os.getenv("CHERAGH_INDEX_ROOT") or os.getcwd()).resolve()
    required_api_key = api_key or os.getenv("CHERAGH_API_KEY")
    max_top_k = max(1, int(max_top_k))

    app = FastAPI(title="cheragh", version=__version__)

    async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        if required_api_key and x_api_key != required_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    AuthDependency = Depends(require_api_key)

    class AskRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=8_000)
        top_k: int | None = Field(default=None, ge=1, le=max_top_k)
        include_prompt: bool = False

    class IndexRequest(BaseModel):
        path: str = Field(..., min_length=1)
        output: str = Field(default=".cheragh_index", min_length=1)
        incremental: bool = True
        chunk_size: int = Field(default=800, ge=1, le=100_000)
        chunk_overlap: int = Field(default=120, ge=0, le=100_000)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.post("/ask", dependencies=[AuthDependency])
    def ask(request: AskRequest) -> dict[str, Any]:
        response = engine.ask(request.query, top_k=request.top_k)
        data = response.to_dict()
        if request.include_prompt and response.trace is not None:
            data["trace"] = response.trace.to_dict(include_prompt=True)
        return data

    @app.post("/stream", dependencies=[AuthDependency])
    def stream(request: AskRequest):
        return StreamingResponse(engine.stream(request.query, top_k=request.top_k), media_type="text/plain")

    @app.post("/index", dependencies=[AuthDependency])
    def index(request: IndexRequest) -> dict[str, Any]:
        if not indexing_enabled:
            raise HTTPException(status_code=403, detail="Indexing endpoint is disabled")
        from ..indexing import index_path as build_index

        try:
            input_path = _resolve_under_root(request.path, root)
            output_path = _resolve_under_root(request.output, root)
            if request.chunk_overlap >= request.chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size")
            return build_index(
                input_path,
                output_path,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                incremental=request.incremental,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - server integration
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/stats", dependencies=[AuthDependency])
    def stats() -> dict[str, Any]:
        retriever = getattr(engine, "retriever", None)
        store = getattr(retriever, "store", None)
        docs = getattr(store, "documents", None)
        return {
            "document_count": len(docs) if docs is not None else None,
            "top_k": getattr(engine, "top_k", None),
            "cache": engine.cache_backend.stats().to_dict() if getattr(engine, "cache_backend", None) else None,
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
        raise ValueError(f"path must stay under allowed index root: {root}") from exc
    return candidate


def _as_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def app_from_env():  # pragma: no cover - runtime helper
    config_path = os.getenv("CHERAGH_CONFIG")
    index_path = os.getenv("CHERAGH_INDEX")
    return create_app(
        config_path=config_path,
        index_path=index_path,
        enable_indexing=_as_bool(os.getenv("CHERAGH_ENABLE_INDEXING"), default=False),
        allowed_index_root=os.getenv("CHERAGH_INDEX_ROOT"),
        api_key=os.getenv("CHERAGH_API_KEY"),
    )


app = None  # uvicorn can use create_app through CLI; this sentinel avoids implicit heavy loading.
