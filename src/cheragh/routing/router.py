"""Application-level query router."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from .classifiers import QueryClassifier, RouteDecision, RuleBasedQueryClassifier


@dataclass
class RoutedResponse:
    """Generic response used when a route does not return a structured object."""

    query: str
    route: str
    result: Any
    decision: RouteDecision
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def answer(self) -> str:
        if isinstance(self.result, str):
            return self.result
        if isinstance(self.result, dict) and "answer" in self.result:
            return str(self.result["answer"])
        return str(self.result)

    def to_dict(self) -> dict[str, Any]:
        result = self.result
        if hasattr(result, "to_dict") and callable(result.to_dict):
            result = result.to_dict()
        return {
            "query": self.query,
            "route": self.route,
            "decision": self.decision.to_dict(),
            "result": result,
            "metadata": self.metadata,
        }


class QueryRouter:
    """Route user queries to specialized engines or callables.

    Routes may be :class:`RAGEngine` instances, pipelines exposing ``ask`` or
    ``run``, retrievers exposing ``retrieve``, or plain callables.
    """

    def __init__(
        self,
        routes: Mapping[str, Any],
        route_descriptions: Mapping[str, str] | None = None,
        classifier: QueryClassifier | None = None,
        default_route: str | None = None,
        fallback_route: str | None = None,
        include_routing_metadata: bool = True,
    ):
        if not routes:
            raise ValueError("At least one route is required")
        self.routes = dict(routes)
        self.route_descriptions = dict(route_descriptions or {})
        self.classifier = classifier or RuleBasedQueryClassifier()
        self.default_route = default_route if default_route in self.routes else next(iter(self.routes))
        self.fallback_route = fallback_route if fallback_route in self.routes else None
        self.include_routing_metadata = include_routing_metadata
        self.last_decision: RouteDecision | None = None

    def route(self, query: str) -> RouteDecision:
        decision = self.classifier.classify(
            query=query,
            routes=self.routes,
            route_descriptions=self.route_descriptions,
            default_route=self.default_route,
        )
        if decision.route not in self.routes:
            decision = RouteDecision(
                route=self.fallback_route or self.default_route,
                confidence=0.25,
                reason=f"classifier selected unavailable route: {decision.route}",
                query_type=decision.query_type,
            )
        self.last_decision = decision
        return decision

    def ask(self, query: str, **kwargs: Any) -> Any:
        decision = self.route(query)
        route = self.routes[decision.route]
        result = self._execute(route, query, **kwargs)
        return self._attach_metadata(result, query, decision)

    def run(self, query: str, **kwargs: Any) -> Any:
        """Alias for compatibility with pipeline-like APIs."""
        return self.ask(query, **kwargs)

    def stream(self, query: str, **kwargs: Any):
        decision = self.route(query)
        route = self.routes[decision.route]
        if hasattr(route, "stream") and callable(route.stream):
            yield from route.stream(query, **kwargs)
            return
        result = self._execute(route, query, **kwargs)
        if isinstance(result, str):
            yield result
        elif hasattr(result, "answer"):
            yield str(result.answer)
        else:
            yield str(result)

    def _execute(self, route: Any, query: str, **kwargs: Any) -> Any:
        if hasattr(route, "ask") and callable(route.ask):
            return route.ask(query, **kwargs)
        if hasattr(route, "run") and callable(route.run):
            return route.run(query, **kwargs)
        if hasattr(route, "retrieve") and callable(route.retrieve):
            top_k = int(kwargs.pop("top_k", 5))
            return route.retrieve(query, top_k=top_k)
        if callable(route):
            return route(query, **kwargs)
        raise TypeError(f"Route object of type {type(route).__name__!r} is not executable")

    def _attach_metadata(self, result: Any, query: str, decision: RouteDecision) -> Any:
        if not self.include_routing_metadata:
            return result
        routing_payload = {
            "route": decision.route,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "query_type": decision.query_type,
        }
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            result.metadata.setdefault("routing", routing_payload)
            return result
        if isinstance(result, dict):
            result.setdefault("routing", routing_payload)
            return result
        return RoutedResponse(query=query, route=decision.route, result=result, decision=decision, metadata={"routing": routing_payload})
