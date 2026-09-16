"""Conversational RAG engine.

The wrapper keeps per-session history, rewrites follow-up questions with recent
conversation context, and stores assistant answers for subsequent turns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from time import time
from typing import Any

from ..engine import RAGEngine, RAGResponse
from ..base import LLMClient, _validate_non_negative_int


@dataclass
class ConversationTurn:
    """One user/assistant exchange."""

    user: str
    assistant: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user": self.user,
            "assistant": self.assistant,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class InMemoryConversationStore:
    """Simple per-session conversation store."""

    def __init__(self):
        self._sessions: dict[str, list[ConversationTurn]] = {}

    def append(self, session_id: str, turn: ConversationTurn) -> None:
        self._sessions.setdefault(session_id, []).append(turn)

    def get(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        turns = list(self._sessions.get(session_id, []))
        if limit is not None:
            _validate_non_negative_int(limit, name="limit")
            return deepcopy(turns[-limit:]) if limit else []
        return deepcopy(turns)

    def clear(self, session_id: str | None = None) -> None:
        if session_id is None:
            self._sessions.clear()
        else:
            self._sessions.pop(session_id, None)

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        return {session: [turn.to_dict() for turn in turns] for session, turns in self._sessions.items()}


class ConversationalRAGEngine:
    """Add conversation memory and follow-up handling to a :class:`RAGEngine`."""

    def __init__(
        self,
        engine: RAGEngine,
        memory: InMemoryConversationStore | None = None,
        max_history_turns: int = 6,
        condense_followups: bool = True,
        include_history_in_query: bool = True,
        session_id: str = "default",
        *,
        query_rewriter: LLMClient | None = None,
    ):
        self.engine = engine
        self.memory = memory or InMemoryConversationStore()
        self.max_history_turns = _validate_non_negative_int(max_history_turns, name="max_history_turns")
        self.query_rewriter = query_rewriter
        self.condense_followups = condense_followups
        self.include_history_in_query = include_history_in_query
        self.default_session_id = session_id

    def ask(self, query: str, session_id: str | None = None, **kwargs: Any) -> RAGResponse:
        sid = session_id or self.default_session_id
        history = self.memory.get(sid, limit=self.max_history_turns)
        standalone_query = self._standalone_query(query, history)
        response = self.engine.ask(standalone_query, **kwargs)
        response.metadata.setdefault("conversation", {})
        response.metadata["conversation"].update(
            {
                "session_id": sid,
                "turn_index": len(self.memory.get(sid)) + 1,
                "original_query": query,
                "standalone_query": standalone_query,
                "history_turns_used": len(history),
            }
        )
        self.memory.append(
            sid,
            ConversationTurn(
                user=query,
                assistant=response.answer,
                metadata={"standalone_query": standalone_query, "source_count": len(response.sources)},
            ),
        )
        return response

    def run(self, query: str, **kwargs: Any) -> RAGResponse:
        return self.ask(query, **kwargs)

    def stream(self, query: str, session_id: str | None = None, **kwargs: Any):
        # Streaming cannot know the final answer until all chunks are consumed;
        # we buffer minimally so the turn can still be stored.
        sid = session_id or self.default_session_id
        history = self.memory.get(sid, limit=self.max_history_turns)
        standalone_query = self._standalone_query(query, history)
        chunks: list[str] = []
        for chunk in self.engine.stream(standalone_query, **kwargs):
            chunks.append(str(chunk))
            yield chunk
        self.memory.append(sid, ConversationTurn(user=query, assistant="".join(chunks), metadata={"standalone_query": standalone_query}))

    def history(self, session_id: str | None = None, limit: int | None = None) -> list[ConversationTurn]:
        return self.memory.get(session_id or self.default_session_id, limit=limit)

    def clear(self, session_id: str | None = None) -> None:
        self.memory.clear(session_id or self.default_session_id)

    def _standalone_query(self, query: str, history: list[ConversationTurn]) -> str:
        if not self.condense_followups or not history or not self.include_history_in_query:
            return query
        if self.query_rewriter is not None:
            prompt = (
                "Rewrite the latest user question as one self-contained retrieval question. "
                "Resolve pronouns and omitted entities using this conversation, preserve constraints and language, "
                "and do not answer the question or add facts from earlier assistant answers as assumptions. "
                "If the latest question already stands alone, return it unchanged. Return only the rewritten question.\n"
                + "\n".join(f"User: {turn.user}\nAssistant: {turn.assistant}" for turn in history)
                + f"\nLatest user question: {query}\nStandalone question:"
            )
            rewritten = self.query_rewriter.generate(prompt).strip()
            if not rewritten:
                raise ValueError("Conversational query rewriter returned an empty question")
            return rewritten
        if not _looks_like_followup(query):
            # Still include a compact context line for better continuity, but
            # keep the new user query dominant.
            last = history[-1]
            return f"Contexte conversationnel récent: utilisateur={last.user!r}; assistant={last.assistant[:300]!r}.\nQuestion: {query}"
        rendered = []
        for turn in history[-self.max_history_turns :]:
            rendered.append(f"Utilisateur: {turn.user}")
            rendered.append(f"Assistant: {turn.assistant[:500]}")
        return "\n".join(rendered + [f"Question de suivi: {query}"])


def _looks_like_followup(query: str) -> bool:
    q = query.strip().lower()
    followup_markers = (
        "et ", "mais ", "donc ", "cela", "celui", "celle", "ce ", "cet ", "cette ", "ces ",
        "il ", "elle ", "ils ", "elles ", "son ", "sa ", "ses ", "leur ", "leurs ",
        "them", "it", "that", "those", "these", "he ", "she ", "they ", "what about",
    )
    return len(q.split()) <= 8 or q.startswith(followup_markers) or any(marker in q for marker in [" ce document", " ce contrat", " cette clause", " above", " previous"])
