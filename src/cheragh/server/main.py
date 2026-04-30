"""Server entrypoint helpers."""
from __future__ import annotations

from pathlib import Path


def serve(
    config: str | None = None,
    index: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    enable_indexing: bool = False,
    allowed_index_root: str | Path | None = None,
    api_key: str | None = None,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Serving requires uvicorn. Install with: pip install cheragh[fastapi]") from exc

    from .app import create_app

    app = create_app(
        config_path=config,
        index_path=index,
        enable_indexing=enable_indexing,
        allowed_index_root=allowed_index_root,
        api_key=api_key,
    )
    uvicorn.run(app, host=host, port=port)
