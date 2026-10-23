"""Load and validate RAG configuration from YAML or JSON."""
from __future__ import annotations

from pathlib import Path
import json
import os
import re
from typing import Any


_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_MAX_CONFIG_DEPTH = 64
_MAX_CONFIG_NODES = 100_000


def load_raw_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML/JSON config file without schema validation."""

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("YAML config requires PyYAML. Install with: pip install cheragh[config]") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            # Parser errors can quote the input line, including literal API
            # keys. Keep the location useful without logging configuration.
            mark = getattr(exc, "problem_mark", None)
            location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
            raise ValueError(f"Invalid YAML config{location}") from None
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def load_config(path: str | Path, *, validate: bool = True) -> dict[str, Any]:
    """Load a config file and return a normalized dict.

    Validation is enabled by default in v1.0.0 and is implemented with Pydantic.
    Use ``validate=False`` only for migration tooling that needs raw config data.
    """

    data = _resolve_environment_references(load_raw_config(path))
    if not validate:
        return data
    from .schema import validate_config

    return validate_config(data).to_legacy_dict()


def _resolve_environment_references(value: Any, *, location: str = "config") -> Any:
    """Resolve exact ``${NAME}`` strings without exposing secret values.

    Interpolation is deliberately limited to whole scalar strings.  This keeps
    configuration validation strict, avoids surprising substitutions inside
    prompts or paths, and lets deployments reference provider credentials from
    their secret manager through the process environment.
    """

    active: set[int] = set()
    remaining = _MAX_CONFIG_NODES

    def resolve(item: Any, path: str, depth: int) -> Any:
        nonlocal remaining
        remaining -= 1
        if remaining < 0:
            raise ValueError(f"Config exceeds {_MAX_CONFIG_NODES} expanded values")
        if depth > _MAX_CONFIG_DEPTH:
            raise ValueError(f"Config exceeds {_MAX_CONFIG_DEPTH} nesting levels")
        if isinstance(item, (dict, list)):
            identity = id(item)
            if identity in active:
                raise ValueError(f"Config contains a cyclic reference at {path}")
            active.add(identity)
            try:
                if isinstance(item, dict):
                    return {
                        key: resolve(child, f"{path}.{key}", depth + 1)
                        for key, child in item.items()
                    }
                return [resolve(child, f"{path}[{index}]", depth + 1) for index, child in enumerate(item)]
            finally:
                active.remove(identity)
        if not isinstance(item, str):
            return item
        match = _ENV_REFERENCE.fullmatch(item)
        if match is None:
            return item
        variable = match.group(1)
        resolved = os.environ.get(variable)
        if resolved is None or not resolved.strip():
            raise ValueError(
                f"Environment variable {variable!r} referenced by {path} is not set or is empty"
            )
        return resolved

    return resolve(value, location, 0)
