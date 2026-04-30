"""Load and validate RAG configuration from YAML or JSON."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any


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

    data = load_raw_config(path)
    if not validate:
        return data
    from .schema import validate_config

    return validate_config(data).to_legacy_dict()
