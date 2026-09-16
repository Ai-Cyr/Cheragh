"""Exercise a genuinely installed distribution, without models or network calls."""
from __future__ import annotations

import importlib.metadata
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile

import cheragh


def main() -> None:
    package = Path(cheragh.__file__).resolve().parent
    site_packages = Path(sysconfig.get_path("purelib")).resolve()
    assert package.is_relative_to(site_packages), f"Not an installed artifact: {package}"
    assert (package / "py.typed").is_file(), "Missing typing marker"
    assert cheragh.__version__ == importlib.metadata.version("cheragh")
    # Importing public adapters must not require their optional model runtimes.
    for name in cheragh.__all__:
        getattr(cheragh, name)
    for technique in cheragh.TECHNIQUES:
        if technique.implementation:
            module, name = technique.implementation.rsplit(".", 1)
            assert getattr(import_module(module), name) is not None
    cli = str(Path(sysconfig.get_path("scripts")) / "cheragh")
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}

    with tempfile.TemporaryDirectory(prefix="cheragh-smoke-") as directory:
        workdir = Path(directory)
        (workdir / "corpus.txt").write_text(
            "The production release identifier is amber-lighthouse. "
            "The release identifier must be checked before deploying the package.",
            encoding="utf-8",
        )

        def run(*arguments: str) -> dict:
            result = subprocess.run(
                [cli, *arguments], cwd=workdir, env=environment,
                check=True, capture_output=True, text=True, timeout=30,
            )
            return json.loads(result.stdout)

        run("index", "corpus.txt", "--output", "index", "--dimension", "64")
        run("inspect-index", "--index", "index")
        result = run("ask", "What is the production release identifier?", "--index", "index", "--json")
        assert "amber-lighthouse" in result["answer"], result
        assert result["sources"], "The persisted index returned no sources"
        assert "prompt" not in result, "Default CLI output exposed a prompt"
    print(f"Installed Cheragh {cheragh.__version__}: index, reload and retrieval passed")


if __name__ == "__main__":
    main()
