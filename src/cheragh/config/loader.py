"""Load and validate RAG configuration from YAML or JSON."""
from __future__ import annotations

from pathlib import Path
import json
import os
import re
from typing import Any


_ENV_REFERENCE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


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
        data = yaml.safe_load(text) or {}
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

    if isinstance(value, dict):
        return {
            key: _resolve_environment_references(item, location=f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_environment_references(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if not isinstance(value, str):
        return value
    match = _ENV_REFERENCE.fullmatch(value)
    if match is None:
        return value
    variable = match.group(1)
    resolved = os.environ.get(variable)
    if resolved is None or not resolved.strip():
        raise ValueError(
            f"Environment variable {variable!r} referenced by {location} is not set or is empty"
        )
    return resolved
