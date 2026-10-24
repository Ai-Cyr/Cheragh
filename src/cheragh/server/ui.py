"""Serve the bundled browser interface without exposing server configuration."""
from __future__ import annotations

from importlib.resources import files
from typing import Any, Callable


_BROWSER_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; "
        "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
        "base-uri 'none'; form-action 'self'; frame-ancestors 'none'; object-src 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}


def register_ui_routes(app: Any, auth: Any, settings: Callable[[], dict[str, Any]]) -> None:
    """Register only the known assets; private data stays behind the API key."""

    def asset(name: str, media_type: str):
        from fastapi.responses import Response

        content = files("cheragh.server").joinpath("static", name).read_bytes()
        return Response(content=content, media_type=media_type, headers=_BROWSER_HEADERS)

    @app.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import RedirectResponse

        # Relative locations preserve a proxy's root_path and ignore Host input.
        return RedirectResponse("ui/", headers=_BROWSER_HEADERS)

    @app.get("/ui", include_in_schema=False)
    async def ui_redirect():
        from fastapi.responses import RedirectResponse

        return RedirectResponse("ui/", headers=_BROWSER_HEADERS)

    @app.get("/ui/", include_in_schema=False)
    async def ui():
        return asset("index.html", "text/html")

    @app.get("/ui/app.css", include_in_schema=False)
    async def stylesheet():
        return asset("app.css", "text/css")

    @app.get("/ui/app.js", include_in_schema=False)
    async def javascript():
        return asset("app.js", "text/javascript")

    @app.get("/ui/api/settings", dependencies=[auth])
    async def ui_settings() -> dict[str, Any]:
        return settings()

    @app.get("/ui/api/techniques", dependencies=[auth])
    async def techniques() -> list[dict[str, object]]:
        from ..catalog import TECHNIQUES

        return [technique.to_dict() for technique in TECHNIQUES]
