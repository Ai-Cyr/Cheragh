"""Query rewriting and expansion helpers."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Sequence

from ..base import LLMClient


class QueryTransformer(ABC):
    """Interface for query expansion/rewrite components."""

    @abstractmethod
    def transform(self, query: str) -> list[str]:
        """Return one or more query variants. The original query should usually be included."""


class IdentityQueryTransformer(QueryTransformer):
    """Return the original query only."""

    def transform(self, query: str) -> list[str]:
        return [query]


@dataclass
class MultiQueryTransformer(QueryTransformer):
    """Generate several query variants.

    If an LLM is provided, it is asked to rewrite the query. Otherwise a small
    deterministic fallback creates lexical variants that are safe for tests and
    offline execution.
    """

    llm_client: LLMClient | None = None
    num_queries: int = 4
    prompt_template: str = (
        "Réécris la question suivante en {num_queries} requêtes de recherche courtes, "
        "complémentaires et sans numérotation.\nQuestion: {query}\nRequêtes:"
    )

    def transform(self, query: str) -> list[str]:
        variants = [query]
        if self.llm_client is not None:
            prompt = self.prompt_template.format(query=query, num_queries=max(self.num_queries - 1, 1))
            raw = self.llm_client.generate(prompt)
            variants.extend(_parse_lines(raw))
        else:
            variants.extend(_fallback_variants(query))
        return _dedupe_keep_order(variants)[: max(self.num_queries, 1)]


@dataclass
class StepBackQueryTransformer(QueryTransformer):
    """Add a broader “step-back” version of the query."""

    llm_client: LLMClient | None = None

    def transform(self, query: str) -> list[str]:
        if self.llm_client is None:
            return _dedupe_keep_order([query, _fallback_step_back(query)])
        prompt = (
            "Formule une question plus générale qui aide à répondre à la question suivante. "
            "Réponds avec une seule question courte.\nQuestion: " + query
        )
        return _dedupe_keep_order([query, self.llm_client.generate(prompt).strip()])


def build_query_transformer(provider: str | QueryTransformer | None = None, **kwargs) -> QueryTransformer | None:
    """Build a query transformer from a provider string."""
    if provider is None or provider is False:
        return None
    if isinstance(provider, QueryTransformer):
        return provider
    p = str(provider).lower().replace("_", "-")
    if p in {"identity", "none"}:
        return IdentityQueryTransformer()
    if p in {"multi-query", "multiquery", "multi"}:
        return MultiQueryTransformer(**kwargs)
    if p in {"step-back", "stepback"}:
        return StepBackQueryTransformer(**kwargs)
    raise ValueError(f"Unsupported query transformer: {provider}")


def _parse_lines(raw: str) -> list[str]:
    lines = []
    for line in raw.splitlines():
        clean = re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        if clean:
            lines.append(clean)
    return lines


def _fallback_variants(query: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", query).strip()
    no_question_words = re.sub(
        r"\b(quel|quelle|quels|quelles|comment|pourquoi|est-ce que|what|how|why|which)\b",
        "",
        normalized,
        flags=re.I,
    ).strip()
    keywordish = " ".join(re.findall(r"\w+", normalized.lower())[:12])
    return [candidate for candidate in [no_question_words, keywordish] if candidate]


def _fallback_step_back(query: str) -> str:
    words = re.findall(r"\w+", query)
    core = " ".join(words[:10]) if words else query
    return f"contexte général {core}".strip()


def _dedupe_keep_order(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        clean = re.sub(r"\s+", " ", item).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return output
