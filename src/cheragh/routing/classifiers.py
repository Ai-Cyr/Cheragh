"""Query classification utilities for application-level routing."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RouteDecision:
    """Classifier decision returned before executing a route."""

    route: str
    confidence: float
    reason: str
    query_type: str = "qa"

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "confidence": self.confidence,
            "reason": self.reason,
            "query_type": self.query_type,
        }


class QueryClassifier(ABC):
    """Interface for query classifiers used by :class:`QueryRouter`."""

    @abstractmethod
    def classify(
        self,
        query: str,
        routes: Mapping[str, object],
        route_descriptions: Mapping[str, str] | None = None,
        default_route: str | None = None,
    ) -> RouteDecision:
        """Return the route that should handle ``query``."""


class RuleBasedQueryClassifier(QueryClassifier):
    """Fast dependency-free classifier based on multilingual keyword rules.

    It is intentionally transparent and deterministic. For production systems
    that need trained routing, replace this object with any implementation of
    :class:`QueryClassifier`.
    """

    def __init__(self, rules: Sequence["RouteRule"] | None = None):
        from .rules import default_rules

        self.rules = list(rules) if rules is not None else default_rules()

    def classify(
        self,
        query: str,
        routes: Mapping[str, object],
        route_descriptions: Mapping[str, str] | None = None,
        default_route: str | None = None,
    ) -> RouteDecision:
        if not routes:
            raise ValueError("At least one route is required")
        available = set(routes)
        default = default_route if default_route in available else _first_available(routes)
        scored: list[RouteDecision] = []
        for rule in self.rules:
            if rule.route not in available:
                continue
            score = rule.score(query, route_descriptions or {})
            if score > 0:
                scored.append(
                    RouteDecision(
                        route=rule.route,
                        confidence=min(1.0, score),
                        reason=rule.reason,
                        query_type=rule.query_type,
                    )
                )
        if scored:
            return max(scored, key=lambda decision: decision.confidence)

        # Soft fallback: route names/descriptions can still match the query.
        description_match = _score_descriptions(query, route_descriptions or {}, available)
        if description_match is not None:
            return description_match

        return RouteDecision(route=default, confidence=0.35, reason="default route", query_type="qa")


class KeywordIntentClassifier(RuleBasedQueryClassifier):
    """Backward-readable alias for the default rule-based classifier."""


def _first_available(routes: Mapping[str, object]) -> str:
    return next(iter(routes.keys()))


def _score_descriptions(query: str, descriptions: Mapping[str, str], available: set[str]) -> RouteDecision | None:
    terms = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
    if not terms:
        return None
    best_route = None
    best_score = 0.0
    for route, description in descriptions.items():
        if route not in available:
            continue
        route_terms = set(re.findall(r"\w+", f"{route} {description}".lower(), flags=re.UNICODE))
        if not route_terms:
            continue
        overlap = len(terms & route_terms)
        score = overlap / max(4, len(terms))
        if score > best_score:
            best_score = score
            best_route = route
    if best_route is None or best_score <= 0:
        return None
    return RouteDecision(route=best_route, confidence=min(0.75, 0.4 + best_score), reason="route description keyword match", query_type=best_route)
