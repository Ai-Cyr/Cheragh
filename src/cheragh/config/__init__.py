"""Configuration helpers."""
from __future__ import annotations

from importlib import import_module

from .loader import load_config, load_raw_config

_LAZY_EXPORTS = {
    "RAGConfig": (".schema", "RAGConfig"),
    "ObservabilityConfig": (".schema", "ObservabilityConfig"),
    "IndexingConfig": (".schema", "IndexingConfig"),
    "validate_config": (".schema", "validate_config"),
    "load_and_validate_config": (".schema", "load_and_validate_config"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


__all__ = ["load_config", "load_raw_config", *_LAZY_EXPORTS.keys()]
