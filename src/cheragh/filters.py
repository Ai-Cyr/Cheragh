"""Metadata filter helpers shared by retrievers."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_OPERATOR_KEYS = {"$eq", "$ne", "$in", "$nin", "$exists", "$contains", "$gt", "$gte", "$lt", "$lte"}


def metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any] | None) -> bool:
    """Return whether ``metadata`` satisfies simple Mongo-style filters.

    Supported forms:

    - ``{"tenant": "acme"}`` equality
    - ``{"tenant": ["acme", "beta"]}`` membership shortcut
    - ``{"score": {"$gte": 0.8}}`` numeric comparison
    - ``{"tags": {"$contains": "legal"}}`` collection/string contains
    - ``{"archived": {"$ne": True}}`` inequality
    - ``{"field": {"$exists": True}}`` presence/absence
    """

    if not filters:
        return True
    for key, expected in filters.items():
        actual = metadata.get(key)
        exists = key in metadata
        if isinstance(expected, Mapping) and any(op in expected for op in _OPERATOR_KEYS):
            if not _matches_operators(actual, exists, expected):
                return False
        elif isinstance(expected, (list, tuple, set, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def _matches_operators(actual: Any, exists: bool, operators: Mapping[str, Any]) -> bool:
    for op, expected in operators.items():
        if op == "$exists":
            if bool(expected) != exists:
                return False
            continue
        if not exists:
            return False
        if op == "$eq" and actual != expected:
            return False
        if op == "$ne" and actual == expected:
            return False
        if op == "$in" and actual not in _as_membership_values(expected):
            return False
        if op == "$nin" and actual in _as_membership_values(expected):
            return False
        if op == "$contains" and not _contains(actual, expected):
            return False
        if op in {"$gt", "$gte", "$lt", "$lte"} and not _compare(actual, expected, op):
            return False
        if op not in _OPERATOR_KEYS:
            return False
    return True


def _as_membership_values(value: Any) -> set[Any]:
    if isinstance(value, set):
        return value
    if isinstance(value, (list, tuple, frozenset)):
        return set(value)
    return {value}


def _contains(actual: Any, expected: Any) -> bool:
    if isinstance(actual, str):
        return str(expected) in actual
    if isinstance(actual, Sequence) and not isinstance(actual, (bytes, bytearray)):
        return expected in actual
    if isinstance(actual, (set, frozenset)):
        return expected in actual
    return False


def _compare(actual: Any, expected: Any, op: str) -> bool:
    try:
        if op == "$gt":
            return actual > expected
        if op == "$gte":
            return actual >= expected
        if op == "$lt":
            return actual < expected
        if op == "$lte":
            return actual <= expected
    except TypeError:
        return False
    return False
