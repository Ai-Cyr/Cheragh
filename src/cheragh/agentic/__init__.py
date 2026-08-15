"""Experimental, bounded Agentic RAG components.

Only explicitly registered tools can be called.  The registry is an allow-list,
not a sandbox for unsafe application-provided handlers.
"""

from .engine import (
    AgentAction,
    AgentPlanner,
    AgentStep,
    AgentTool,
    AgentTrace,
    AgenticRAGEngine,
    AgenticRAGResult,
    LLMJSONPlanner,
    RetrievalToolAdapter,
    ScriptedPlanner,
    ToolObservation,
    ToolRegistry,
    ToolSpec,
)

__all__ = [
    "AgentAction",
    "AgentPlanner",
    "AgentStep",
    "AgentTool",
    "AgentTrace",
    "AgenticRAGEngine",
    "AgenticRAGResult",
    "LLMJSONPlanner",
    "RetrievalToolAdapter",
    "ScriptedPlanner",
    "ToolObservation",
    "ToolRegistry",
    "ToolSpec",
]
