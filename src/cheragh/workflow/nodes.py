"""Reusable workflow node adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..base import BaseRetriever, LLMClient
from ..compression import ContextCompressor
from ..pipeline import AdvancedRAGPipeline, DEFAULT_ANSWER_PROMPT_FR
from ..query import QueryTransformer


@dataclass
class FunctionNode:
    """Wrap a function as a workflow node."""

    fn: Callable[..., Any]
    output_key: str | None = None

    def run(self, state: dict[str, Any]) -> dict[str, Any] | Any:
        result = self.fn(state)
        if self.output_key is not None:
            return {self.output_key: result}
        return result


@dataclass
class TransformQueryNode:
    transformer: QueryTransformer
    query_key: str = "query"
    output_key: str = "query_variants"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        return {self.output_key: self.transformer.transform(str(state[self.query_key]))}


@dataclass
class RetrieveNode:
    retriever: BaseRetriever
    query_key: str = "query"
    output_key: str = "documents"
    top_k: int = 5

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = str(state[self.query_key])
        return {self.output_key: self.retriever.retrieve(query, top_k=int(state.get("top_k", self.top_k)))}


@dataclass
class CompressNode:
    compressor: ContextCompressor
    query_key: str = "query"
    documents_key: str = "documents"
    output_key: str = "documents"

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        return {self.output_key: self.compressor.compress(str(state[self.query_key]), list(state.get(self.documents_key, [])))}


@dataclass
class GenerateNode:
    llm_client: LLMClient
    query_key: str = "query"
    documents_key: str = "documents"
    output_key: str = "answer"
    prompt_template: str = DEFAULT_ANSWER_PROMPT_FR

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        context = AdvancedRAGPipeline._format_context(list(state.get(self.documents_key, [])))
        prompt = self.prompt_template.format(context=context, query=state[self.query_key])
        return {self.output_key: self.llm_client.generate(prompt), "prompt": prompt}
