"""Exercise the packaged GUI in Chromium against a real, authenticated server.

Install the package with its fastapi/config extras and Playwright separately,
then run ``python -I scripts/smoke_ui.py``. No provider or model download is used.
The application is mounted under a URL prefix to verify relative asset/API URLs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
import secrets
import socket
import tempfile
import threading
import time
from urllib.parse import urlsplit


@contextmanager
def running_server(root: Path, key: str | None):
    from fastapi import FastAPI
    import uvicorn

    from cheragh.indexing import index_path
    from cheragh.server import create_app

    corpus = root / "corpus.txt"
    corpus.write_text(
        'The deployment code is amber-lighthouse. '
        'Literal source markup: <img src="x" onerror="window.cheraghXss=true">.',
        encoding="utf-8",
    )
    index = root / "index"
    index_path(corpus, index)
    application = create_app(
        index_path=str(index), api_key=key or "", require_auth=key is not None,
        enable_indexing=key is not None, allowed_index_root=root, max_top_k=7,
    )
    mounted = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    mounted.mount("/cheragh", application)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        mounted, host="127.0.0.1", port=port, log_level="warning", access_log=False,
    ))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("GUI smoke server did not start")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}/cheragh", corpus
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        if thread.is_alive():
            raise RuntimeError("GUI smoke server did not stop cleanly")


def exercise(base_url: str, corpus: Path, key: str, *, browser_executable: str | None,
             screenshot_dir: Path | None) -> None:
    from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright

    requested_urls: list[str] = []
    page_errors: list[str] = []
    console_errors: list[tuple[str, str]] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable, headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1000}, reduced_motion="reduce")
        page = context.new_page()
        page.set_default_timeout(15_000)
        page.on("request", lambda request: requested_urls.append(request.url))
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("console", lambda message: console_errors.append((message.text, message.location.get("url", "")))
                if message.type == "error" else None)
        try:
            response = page.goto(base_url + "/", wait_until="networkidle")
            assert response is not None and response.ok, "GUI could not be loaded"
            expect(page).to_have_url(base_url + "/ui/")
            expect(page.get_by_label("Clé API", exact=True)).to_be_visible()
            if screenshot_dir is not None:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_dir / "welcome.png"), full_page=True)

            # Authentication is enforced by the real API, not just by the form.
            page.get_by_label("Clé API", exact=True).fill("incorrect-key")
            with page.expect_response(lambda response: response.status == 401):
                page.get_by_role("button", name="Se connecter", exact=True).click()
            expect(page.get_by_role("button", name="Se connecter", exact=True)).to_be_enabled()
            page.get_by_label("Clé API", exact=True).fill(key)
            page.get_by_role("button", name="Se connecter", exact=True).click()
            expect(page.get_by_role("button", name="Se déconnecter", exact=True)).to_be_visible()

            top_k = page.get_by_label("Nombre de sources", exact=True)
            expect(top_k).to_have_attribute("max", "7")
            top_k.fill("2")
            question = page.get_by_label("Votre question", exact=True)
            question.fill("What is the deployment code?")
            with page.expect_request(lambda request: request.url.endswith("/ask")) as asked:
                page.get_by_role("button", name="Envoyer la question", exact=True).click()
            assert asked.value.post_data_json["top_k"] == 2
            expect(page.locator(".answer-text").last).to_contain_text("amber-lighthouse")
            expect(page.locator("#source-list")).to_contain_text("corpus.txt")
            source = page.locator("#source-list details").first
            if source.get_attribute("open") is None:
                source.locator("summary").click()
            expect(source).to_contain_text('<img src="x" onerror="window.cheraghXss=true">')
            assert page.locator('img[src="x"]').count() == 0, "Source markup became executable HTML"
            assert page.evaluate("window.cheraghXss === undefined"), "Source markup executed"
            assert page.evaluate("localStorage.length + sessionStorage.length") == 0, "Session was persisted"

            if screenshot_dir is not None:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_dir / "desktop.png"), full_page=True)

            page.get_by_role("button", name="Architectures", exact=True).click()
            search = page.get_by_label("Rechercher une architecture", exact=True)
            search.fill("RAPTOR")
            expect(page.get_by_text("RAPTOR", exact=False).last).to_be_visible()
            search.fill("architecture-that-does-not-exist")
            expect(page.get_by_text("Aucune architecture", exact=False)).to_be_visible()
            search.fill("")
            page.get_by_label("Maturité", exact=True).select_option("stable")
            expect(page.locator("#technique-grid")).not_to_contain_text("Expérimentale")
            page.get_by_label("Maturité", exact=True).select_option("all")

            # Rebuild and activate the served index without restarting the server.
            corpus.write_text("The deployment code is coral-garden.", encoding="utf-8")
            page.get_by_role("button", name="Documents", exact=True).click()
            page.get_by_label("Chemin des documents", exact=True).fill("corpus.txt")
            expect(page.get_by_label("Dossier de l’index", exact=True)).to_have_value("index")
            page.get_by_label("Activer cet index après l’indexation", exact=True).check()
            with page.expect_response(lambda response: response.url.endswith("/index")) as indexed:
                page.get_by_role("button", name="Indexer les documents").click()
            assert indexed.value.ok, "Document indexing failed"
            assert indexed.value.json().get("activated") is True, "New index was not activated"
            page.get_by_role("button", name="Conversation", exact=True).click()
            question.fill("What is the deployment code?")
            page.get_by_role("button", name="Envoyer la question", exact=True).click()
            expect(page.locator(".answer-text").last).to_contain_text("coral-garden")

            # Mobile viewport must retain usable navigation and avoid overflow.
            page.set_viewport_size({"width": 390, "height": 844})
            page.get_by_role("button", name="Documents", exact=True).click()
            expect(page.get_by_label("Chemin des documents", exact=True)).to_be_visible()
            page.get_by_role("button", name="Conversation", exact=True).click()
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth + 1"
            ), "GUI overflows the mobile viewport"
            if screenshot_dir is not None:
                page.screenshot(path=str(screenshot_dir / "mobile.png"), full_page=True)

            # Pause a genuine HTTP response after the server has answered, then
            # disconnect before releasing it. Late results must not restore data.
            cdp = context.new_cdp_session(page)
            paused_responses: list[dict] = []
            cdp.on("Fetch.requestPaused", lambda event: paused_responses.append(event))
            cdp.send("Fetch.enable", {"patterns": [{
                "urlPattern": "*/ask", "requestStage": "Response",
            }]})
            question.fill("What is the deployment code after reindexing?")
            page.get_by_role("button", name="Envoyer la question", exact=True).click()
            deadline = time.monotonic() + 10
            while not paused_responses:
                if time.monotonic() >= deadline:
                    raise RuntimeError("The live query response was not intercepted")
                page.wait_for_timeout(50)
            assert paused_responses[0].get("responseStatusCode") == 200
            page.get_by_role("button", name="Se déconnecter", exact=True).click()
            try:
                cdp.send("Fetch.continueRequest", {"requestId": paused_responses[0]["requestId"]})
            except PlaywrightError as exc:
                # Aborting fetch on disconnect can invalidate an already-paused
                # interception. That is also a correct, stronger cancellation.
                if "Invalid InterceptionId" not in str(exc) and "Invalid interceptionId" not in str(exc):
                    raise
            cdp.send("Fetch.disable")
            page.wait_for_timeout(100)
            expect(page.get_by_role("button", name="Se connecter", exact=True)).to_be_visible()
            expect(page.get_by_label("Clé API", exact=True)).to_have_value("")
            expect(page.locator("#messages")).not_to_contain_text("amber-lighthouse")
            expect(page.locator("#messages")).not_to_contain_text("coral-garden")
            assert page.evaluate("localStorage.length + sessionStorage.length") == 0
            assert not page_errors, f"Browser JavaScript errors: {page_errors}"
            unexpected_console_errors = [
                message for message, location in console_errors
                if not ("status of 401" in message and location.endswith("/ui/api/settings"))
            ]
            assert not unexpected_console_errors, f"Browser console errors: {unexpected_console_errors}"
            origin = urlsplit(base_url)
            assert all(
                (urlsplit(url).scheme, urlsplit(url).netloc) == (origin.scheme, origin.netloc)
                for url in requested_urls
            ), "GUI attempted a third-party request"
            assert all(key not in url for url in requested_urls), "API key appeared in a request URL"
        finally:
            context.close()
            browser.close()


def exercise_anonymous(base_url: str, *, browser_executable: str | None) -> None:
    """Check the documented local quickstart with indexing explicitly disabled."""
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=browser_executable, headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(15_000)
        page_errors: list[str] = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        try:
            response = page.goto(base_url + "/ui/", wait_until="networkidle")
            assert response is not None and response.ok
            expect(page.get_by_label("Clé API", exact=True)).to_have_value("")
            page.get_by_role("button", name="Se connecter", exact=True).click()
            expect(page.get_by_role("button", name="Se déconnecter", exact=True)).to_be_visible()
            page.get_by_label("Votre question", exact=True).fill("What is the deployment code?")
            page.get_by_role("button", name="Envoyer la question", exact=True).click()
            expect(page.locator(".answer-text").last).to_contain_text("amber-lighthouse")
            page.get_by_role("button", name="Documents", exact=True).click()
            expect(page.locator("#index-fields")).to_have_attribute("disabled", "")
            expect(page.get_by_label("Chemin des documents", exact=True)).to_be_disabled()
            expect(page.get_by_role("button", name="Indexer les documents")).to_be_disabled()
            expect(page.locator("#index-notice")).to_be_visible()
            expect(page.locator("#index-notice")).to_contain_text("indexation est désactivée")
            assert not page_errors, f"Anonymous browser JavaScript errors: {page_errors}"
        finally:
            context.close()
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-executable", help="Optional local Chromium executable")
    parser.add_argument("--screenshot-dir", type=Path, help="Write desktop and mobile validation screenshots")
    arguments = parser.parse_args()
    key = secrets.token_urlsafe(32)
    with tempfile.TemporaryDirectory(prefix="cheragh-gui-smoke-") as directory:
        with running_server(Path(directory), key) as (base_url, corpus):
            exercise(base_url, corpus, key, browser_executable=arguments.browser_executable,
                     screenshot_dir=arguments.screenshot_dir)
    with tempfile.TemporaryDirectory(prefix="cheragh-gui-anonymous-") as directory:
        with running_server(Path(directory), None) as (base_url, _):
            exercise_anonymous(base_url, browser_executable=arguments.browser_executable)
    print("GUI authentication, anonymous access, retrieval, sources, live indexing and mobile browser checks passed")


if __name__ == "__main__":
    main()
