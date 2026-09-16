"""Probe a running production image using only the Python standard library."""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")
    key = os.environ["CHERAGH_API_KEY"]

    def request(path: str, *, body: dict | None = None, authenticated: bool = False) -> tuple[int, bytes]:
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["X-API-Key"] = key
        payload = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base_url + path, data=payload, headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            assert response.headers.get("X-Request-ID"), "Missing request correlation ID"
            assert response.headers.get("X-Content-Type-Options") == "nosniff"
            assert response.headers.get("Cache-Control") == "no-store"
            return response.status, response.read()

    deadline = time.monotonic() + 45
    while True:
        try:
            status, _ = request("/ready")
            if status == 200:
                break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Server did not become ready within 45 seconds")
        time.sleep(0.25)

    assert request("/health")[0] == 200
    assert request("/stats")[0] == 401, "Unauthenticated access was permitted"
    assert request("/stats", authenticated=True)[0] == 200
    question = {"query": "What is the production release identifier?"}
    assert request("/ask", body=question)[0] == 401
    status, body = request("/ask", body=question, authenticated=True)
    assert status == 200, (status, body)
    result = json.loads(body)
    assert "amber-lighthouse" in result["answer"], result
    assert result["sources"], "Query returned no sources"
    assert "prompt" not in result, "Default HTTP output exposed a prompt"
    status, _ = request("/index", body={"path": "/data"}, authenticated=True)
    assert status == 403, f"Indexing should be disabled, got HTTP {status}"
    print("Server readiness, authentication, retrieval and indexing restrictions passed")


if __name__ == "__main__":
    main()
