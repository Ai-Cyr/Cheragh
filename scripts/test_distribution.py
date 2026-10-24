"""Run the shipped test suite against a non-editable, installed distribution.

Usage: python -I scripts/test_distribution.py SOURCE_ROOT [pytest options...]
SOURCE_ROOT may be a checkout or an extracted source distribution. Tests and
their documentation/configuration fixtures stay there; package imports must
come exclusively from this interpreter's site-packages directory.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
from pathlib import Path
import sys
import sysconfig
import tempfile

import cheragh


def installed_package() -> Path:
    package = Path(cheragh.__file__).resolve().parent
    library_paths = {
        Path(sysconfig.get_path(name)).resolve() for name in ("purelib", "platlib")
    }
    if not any(package.is_relative_to(path) for path in library_paths):
        raise RuntimeError(f"Tests require an installed artifact, not a source checkout: {package}")
    distribution = importlib.metadata.distribution("cheragh")
    if distribution.version != cheragh.__version__:
        raise RuntimeError("Installed package and distribution metadata versions disagree")
    if not (package / "py.typed").is_file():
        raise RuntimeError("Installed distribution is missing its typing marker")
    return package


def check_import_locations(package: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "cheragh" and not name.startswith("cheragh."):
            continue
        location = getattr(module, "__file__", None)
        if location is not None and not Path(location).resolve().is_relative_to(package):
            raise RuntimeError(f"Test imported {name} outside the installed package: {location}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error("run this script with python -I to isolate the installed package")
    root = args.source_root.resolve()
    if not (root / "tests").is_dir() or not (root / "pyproject.toml").is_file():
        parser.error("SOURCE_ROOT must contain tests/ and pyproject.toml")

    package = installed_package()
    print(f"Testing installed Cheragh {cheragh.__version__} at {package}", flush=True)
    # Isolated mode protects this interpreter. Also prevent inherited source
    # paths or pytest options from selecting a different package in subprocesses.
    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTEST_ADDOPTS", None)
    import pytest

    arguments = args.pytest_args
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    original_directory = Path.cwd()
    try:
        with tempfile.TemporaryDirectory(prefix="cheragh-installed-tests-") as directory:
            os.chdir(directory)
            status = pytest.main([
                "--import-mode=importlib", "-o", "pythonpath=", "-o", "addopts=",
                str(root / "tests"), *arguments,
            ])
            check_import_locations(package)
            return int(status)
    finally:
        os.chdir(original_directory)


if __name__ == "__main__":
    raise SystemExit(main())
