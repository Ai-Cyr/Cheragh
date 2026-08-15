"""Experimental Agentic RAG with a bounded, allow-listed tool loop.

The agent can only call tools explicitly registered in :class:`ToolRegistry`.
There is no Python evaluator, shell tool, dynamic import, URL fetcher, or fallback
function execution in this module.  Every run is bounded by ``max_steps``.

Maturity and limitations
------------------------
This API is experimental.  It is an orchestration primitive, not a security
sandbox: applications remain responsible for ensuring that each registered
handler is itself safe, authorized, tenant-aware, and appropriately rate-limited.
The built-in LLM planner expects strict JSON and does not provide a guarantee
that a model's plan is correct.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from typing import Any, Callable, Iterable, Protocol, Sequence

from ..base import BaseRetriever, Document, LLMClient, _validate_top_k


_TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class ToolSpec:
    """Planner-visible description of an allow-listed tool."""

    name: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}


@dataclass(frozen=True)
class AgentTool:
    """A named callable that must be explicitly registered before execution."""

    name: str
    description: str
    handler: Callable[[str], Any]

    def __post_init__(self) -> None:
        if not _TOOL_NAME_PATTERN.fullmatch(self.name):
            raise ValueError("tool name must match ^[a-z][a-z0-9_]{0,63}$")
        if not self.description.strip():
            raise ValueError("tool description must not be empty")
        if not callable(self.handler):
            raise TypeError("tool handler must be callable")

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description)


@dataclass(frozen=True)
class ToolObservation:
    """Structured result of a tool call, including denied and failed calls."""

    tool_name: str
    ok: bool
    output: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "output": _json_safe(self.output),
            "error": self.error,
        }


class ToolRegistry:
    """Explicit allow-list and sole execution boundary for agent tools."""

    def __init__(self, tools: Iterable[AgentTool] = (), *, max_input_chars: int = 8_000):
        max_input_chars = _validate_top_k(max_input_chars, name="max_input_chars")
        self.max_input_chars = max_input_chars
        self._tools: dict[str, AgentTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AgentTool) -> None:
        """Add one tool to the allow-list; duplicate names are rejected."""
        if not isinstance(tool, AgentTool):
            raise TypeError("tool must be an AgentTool")
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def execute(self, name: str, tool_input: str) -> ToolObservation:
        """Execute only an exact, registered name; failures become observations."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolObservation(name, False, error="tool_not_registered")
        if not isinstance(tool_input, str):
            return ToolObservation(name, False, error="tool_input_must_be_text")
        if len(tool_input) > self.max_input_chars:
            return ToolObservation(name, False, error="tool_input_too_large")
        try:
            return ToolObservation(name, True, output=tool.handler(tool_input))
        except Exception as exc:  # Tools are an application boundary; expose no traceback.
            return ToolObservation(name, False, error=f"tool_failed:{type(exc).__name__}")


@dataclass(frozen=True)
class AgentAction:
    """One planner action: call an allow-listed tool or return a final answer."""

    kind: str
    tool_name: str | None = None
    tool_input: str = ""
    answer: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.kind not in {"tool", "final"}:
            raise ValueError("action kind must be 'tool' or 'final'")
        if self.kind == "tool" and not self.tool_name:
            raise ValueError("tool actions require tool_name")
        if self.kind == "final" and self.tool_name is not None:
            raise ValueError("final actions cannot include tool_name")
        if not isinstance(self.tool_input, str) or not isinstance(self.answer, str):
            raise TypeError("tool_input and answer must be strings")

    @classmethod
    def call_tool(cls, name: str, tool_input: str, *, reason: str = "") -> "AgentAction":
        return cls("tool", tool_name=name, tool_input=tool_input, reason=reason)

    @classmethod
    def finish(cls, answer: str, *, reason: str = "") -> "AgentAction":
        return cls("final", answer=answer, reason=reason)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AgentAction":
        kind = value.get("kind")
        if kind == "tool":
            name = value.get("tool")
            tool_input = value.get("input", "")
            if not isinstance(name, str) or not isinstance(tool_input, str):
                raise ValueError("tool action requires string fields 'tool' and 'input'")
            return cls.call_tool(name, tool_input, reason=str(value.get("reason", "")))
        if kind == "final":
            answer = value.get("answer", "")
            if not isinstance(answer, str):
                raise ValueError("final action requires a string 'answer'")
            return cls.finish(answer, reason=str(value.get("reason", "")))
        raise ValueError("planner JSON field 'kind' must be 'tool' or 'final'")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "answer": self.answer,
            "reason": self.reason,
        }


@dataclass
class AgentStep:
    """One plan/action/observe step."""

    number: int
    action: AgentAction
    observation: ToolObservation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "action": self.action.to_dict(),
            "observation": self.observation.to_dict() if self.observation else None,
        }


class AgentPlanner(Protocol):
    """Planner interface for deterministic or model-backed policies."""

    def plan(
        self,
        query: str,
        steps: Sequence[AgentStep],
        tools: Sequence[ToolSpec],
    ) -> AgentAction:
        """Choose the next bounded action."""


class ScriptedPlanner:
    """Deterministic ordered planner for tests, demos, and fixed workflows."""

    def __init__(self, actions: Iterable[AgentAction], *, exhausted_answer: str = ""):
        self._actions = tuple(actions)
        if not self._actions:
            raise ValueError("actions must not be empty")
        self._index = 0
        self.exhausted_answer = exhausted_answer

    def plan(
        self,
        query: str,
        steps: Sequence[AgentStep],
        tools: Sequence[ToolSpec],
    ) -> AgentAction:
        del query, steps, tools
        if self._index >= len(self._actions):
            return AgentAction.finish(self.exhausted_answer, reason="script_exhausted")
        action = self._actions[self._index]
        self._index += 1
        return action


class LLMJSONPlanner:
    """LLM planner accepting only a strict JSON action object.

    Parsing JSON narrows the interface but does not make model output trusted.
    The engine still checks every requested tool against :class:`ToolRegistry`.
    """

    def __init__(self, llm_client: LLMClient, *, max_history_chars: int = 12_000):
        max_history_chars = _validate_top_k(max_history_chars, name="max_history_chars")
        self.llm_client = llm_client
        self.max_history_chars = max_history_chars

    def plan(
        self,
        query: str,
        steps: Sequence[AgentStep],
        tools: Sequence[ToolSpec],
    ) -> AgentAction:
        history = json.dumps([step.to_dict() for step in steps], ensure_ascii=False, default=str)
        if len(history) > self.max_history_chars:
            history = history[-self.max_history_chars :]
        prompt = (
            "Tu pilotes un agent RAG borné. Choisis exactement une action. "
            "N'utilise que les outils listés. Retourne uniquement un objet JSON, sans markdown: "
            '{"kind":"tool","tool":"nom","input":"texte","reason":"..."} ou '
            '{"kind":"final","answer":"réponse","reason":"..."}.\n\n'
            f"Question: {query}\n"
            f"Outils: {json.dumps([tool.to_dict() for tool in tools], ensure_ascii=False)}\n"
            f"Historique: {history}"
        )
        raw = self.llm_client.generate(prompt)
        if not isinstance(raw, str):
            raise TypeError("planner LLM must return text")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("planner returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("planner JSON must be an object")
        return AgentAction.from_dict(payload)


class RetrievalToolAdapter:
    """Expose a retriever as a bounded, serializable agent tool."""

    def __init__(self, retriever: BaseRetriever, *, top_k: int = 5, max_document_chars: int = 2_000):
        top_k = _validate_top_k(top_k)
        max_document_chars = _validate_top_k(max_document_chars, name="max_document_chars")
        self.retriever = retriever
        self.top_k = top_k
        self.max_document_chars = max_document_chars

    def __call__(self, query: str) -> dict[str, Any]:
        normalized_query = " ".join(query.split())
        if not normalized_query:
            raise ValueError("retrieval query must not be empty")
        documents = list(self.retriever.retrieve(normalized_query, top_k=self.top_k))
        if not all(isinstance(document, Document) for document in documents):
            raise TypeError("retriever.retrieve() must return Document objects")
        return {
            "query": normalized_query,
            "documents": [
                {
                    "doc_id": document.doc_id,
                    "score": document.score,
                    "content": document.content[: self.max_document_chars],
                    "metadata": _json_safe(document.metadata),
                }
                for document in documents
            ],
        }

    def as_tool(
        self,
        name: str = "retrieve",
        description: str = "Retrieve relevant evidence for a natural-language query.",
    ) -> AgentTool:
        return AgentTool(name, description, self)


@dataclass
class AgentTrace:
    """Structured record of every planner decision and tool observation."""

    query: str
    available_tools: tuple[ToolSpec, ...]
    max_steps: int
    steps: list[AgentStep] = field(default_factory=list)
    stop_reason: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "available_tools": [tool.to_dict() for tool in self.available_tools],
            "max_steps": self.max_steps,
            "steps": [step.to_dict() for step in self.steps],
            "stop_reason": self.stop_reason,
            "error": self.error,
        }


@dataclass
class AgenticRAGResult:
    """Final answer, status, and bounded agent trace."""

    query: str
    answer: str
    status: str
    trace: AgentTrace
    maturity: str = "experimental"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "status": self.status,
            "maturity": self.maturity,
            "trace": self.trace.to_dict(),
        }


class AgenticRAGEngine:
    """Run a bounded plan/action/observe loop over an explicit tool registry."""

    def __init__(
        self,
        planner: AgentPlanner,
        tools: ToolRegistry | Iterable[AgentTool],
        *,
        max_steps: int = 6,
        limit_answer: str = "Je ne sais pas : la limite d'étapes de l'agent a été atteinte.",
        planner_error_answer: str = "Je ne sais pas : le planificateur n'a pas produit d'action valide.",
    ):
        max_steps = _validate_top_k(max_steps, name="max_steps")
        self.planner = planner
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.max_steps = max_steps
        self.limit_answer = limit_answer
        self.planner_error_answer = planner_error_answer

    def ask(self, query: str) -> AgenticRAGResult:
        query = " ".join(query.split())
        if not query:
            raise ValueError("query must not be empty")
        trace = AgentTrace(query, self.registry.specs, self.max_steps)

        for number in range(1, self.max_steps + 1):
            try:
                action = self.planner.plan(query, tuple(trace.steps), trace.available_tools)
                if not isinstance(action, AgentAction):
                    raise TypeError("planner.plan() must return AgentAction")
            except Exception as exc:
                trace.stop_reason = "planner_error"
                trace.error = type(exc).__name__
                return AgenticRAGResult(query, self.planner_error_answer, "planner_error", trace)

            if action.kind == "final":
                trace.steps.append(AgentStep(number, action))
                trace.stop_reason = "final_answer"
                return AgenticRAGResult(query, action.answer, "completed", trace)

            observation = self.registry.execute(action.tool_name or "", action.tool_input)
            trace.steps.append(AgentStep(number, action, observation))

        trace.stop_reason = "max_steps_reached"
        return AgenticRAGResult(query, self.limit_answer, "max_steps_exceeded", trace)

    def run(self, query: str) -> AgenticRAGResult:
        """Alias for :meth:`ask`."""
        return self.ask(query)


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation without evaluating user data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
