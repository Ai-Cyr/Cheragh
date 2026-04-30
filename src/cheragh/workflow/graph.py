"""Minimal directed workflow executor for RAG pipelines."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class WorkflowResult:
    """Result returned by :class:`RAGWorkflow.run`."""

    state: dict[str, Any]
    executed_nodes: list[str]

    @property
    def answer(self) -> str | None:
        value = self.state.get("answer") or self.state.get("generation")
        if hasattr(value, "answer"):
            return str(value.answer)
        if value is None:
            return None
        return str(value)

    def to_dict(self) -> dict[str, Any]:
        def serialize(value: Any) -> Any:
            if hasattr(value, "to_dict") and callable(value.to_dict):
                return value.to_dict()
            if isinstance(value, list):
                return [serialize(item) for item in value]
            if isinstance(value, dict):
                return {key: serialize(item) for key, item in value.items()}
            return value

        return {"state": serialize(self.state), "executed_nodes": self.executed_nodes}


@dataclass
class _WorkflowNode:
    name: str
    component: Any


class RAGWorkflow:
    """A tiny DAG executor for composing RAG steps.

    Nodes receive and return a mutable ``state`` dictionary. Plain functions may
    either accept ``state`` as their only argument or keyword arguments matching
    keys in the state. If a node returns a dictionary, it is merged into state;
    otherwise it is stored under the node name.
    """

    def __init__(self):
        self.nodes: dict[str, _WorkflowNode] = {}
        self.edges: dict[str, list[str]] = {}
        self._incoming: dict[str, set[str]] = {}

    def add_node(self, name: str, component: Any) -> "RAGWorkflow":
        if not name:
            raise ValueError("node name cannot be empty")
        if name in self.nodes:
            raise ValueError(f"node {name!r} already exists")
        self.nodes[name] = _WorkflowNode(name=name, component=component)
        self.edges.setdefault(name, [])
        self._incoming.setdefault(name, set())
        return self

    def connect(self, source: str, target: str) -> "RAGWorkflow":
        if source not in self.nodes:
            raise KeyError(f"Unknown source node: {source}")
        if target not in self.nodes:
            raise KeyError(f"Unknown target node: {target}")
        self.edges.setdefault(source, []).append(target)
        self._incoming.setdefault(target, set()).add(source)
        return self

    def run(self, query: str | None = None, initial_state: dict[str, Any] | None = None, start_at: str | None = None) -> WorkflowResult:
        state = dict(initial_state or {})
        if query is not None:
            state.setdefault("query", query)
        order = self._execution_order(start_at=start_at)
        executed: list[str] = []
        for node_name in order:
            node = self.nodes[node_name]
            result = self._execute_component(node.component, state)
            if isinstance(result, dict):
                state.update(result)
            elif result is not None:
                state[node_name] = result
            executed.append(node_name)
        return WorkflowResult(state=state, executed_nodes=executed)

    def ask(self, query: str, **kwargs: Any) -> WorkflowResult:
        return self.run(query=query, initial_state=kwargs or None)

    def _execution_order(self, start_at: str | None = None) -> list[str]:
        if not self.nodes:
            return []
        incoming = {name: set(values) for name, values in self._incoming.items()}
        if start_at:
            if start_at not in self.nodes:
                raise KeyError(f"Unknown start node: {start_at}")
            reachable = self._reachable_from(start_at)
            incoming = {name: {src for src in deps if src in reachable} for name, deps in incoming.items() if name in reachable}
        else:
            reachable = set(self.nodes)
        ready = [name for name in self.nodes if name in reachable and not incoming.get(name)]
        order: list[str] = []
        while ready:
            name = ready.pop(0)
            if name in order:
                continue
            order.append(name)
            for target in self.edges.get(name, []):
                if target not in reachable:
                    continue
                incoming.setdefault(target, set()).discard(name)
                if not incoming[target]:
                    ready.append(target)
        if len(order) != len(reachable):
            missing = sorted(reachable - set(order))
            raise ValueError(f"Workflow contains a cycle or disconnected dependency among: {missing}")
        return order

    def _reachable_from(self, start: str) -> set[str]:
        seen: set[str] = set()
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.edges.get(node, []))
        return seen

    def _execute_component(self, component: Any, state: dict[str, Any]) -> Any:
        if hasattr(component, "run") and callable(component.run):
            return component.run(state)
        if callable(component):
            try:
                return component(state)
            except TypeError:
                return component(**state)
        raise TypeError(f"Workflow component {component!r} is not executable")
