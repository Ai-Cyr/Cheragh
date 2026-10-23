"""Reusable workflow node adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..base import BaseRetriever, LLMClient, _validate_top_k
from ..compression import ContextCompressor
from ..engine import _validate_answer_prompt, _validate_query, _validated_document_results, _validated_query_variants
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
        query = _validate_query(state[self.query_key])
        variants, _ = _validated_query_variants(self.transformer.transform(query))
        return {self.output_key: variants or [query]}


@dataclass
class RetrieveNode:
    retriever: BaseRetriever
    query_key: str = "query"
    output_key: str = "documents"
    top_k: int = 5

    def __post_init__(self) -> None:
        self.top_k = _validate_top_k(self.top_k)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = _validate_query(state[self.query_key])
        top_k = _validate_top_k(state.get("top_k", self.top_k))
        documents = _validated_document_results(
            self.retriever.retrieve(query, top_k=top_k), limit=top_k, name="retriever result"
        )
        return {self.output_key: documents}


@dataclass
class CompressNode:
    compressor: ContextCompressor
    query_key: str = "query"
    documents_key: str = "documents"
    output_key: str = "documents"
    top_k: int = 5

    def __post_init__(self) -> None:
        self.top_k = _validate_top_k(self.top_k)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = _validate_query(state[self.query_key])
        top_k = _validate_top_k(state.get("top_k", self.top_k))
        documents = _validated_document_results(
            state.get(self.documents_key, []), limit=top_k, name="documents"
        )
        result = _validated_document_results(
            self.compressor.compress(query, documents), limit=top_k, name="compressor result"
        )
        return {self.output_key: result}


@dataclass
class GenerateNode:
    llm_client: LLMClient
    query_key: str = "query"
    documents_key: str = "documents"
    output_key: str = "answer"
    prompt_template: str = DEFAULT_ANSWER_PROMPT_FR
    top_k: int = 5

    def __post_init__(self) -> None:
        self.top_k = _validate_top_k(self.top_k)
        _validate_answer_prompt(self.prompt_template)

    def run(self, state: dict[str, Any]) -> dict[str, Any]:
        query = _validate_query(state[self.query_key])
        top_k = _validate_top_k(state.get("top_k", self.top_k))
        documents = _validated_document_results(
            state.get(self.documents_key, []), limit=top_k, name="documents"
        )
        context = AdvancedRAGPipeline._format_context(documents)
        prompt = self.prompt_template.format(context=context, query=query)
        answer = self.llm_client.generate(prompt)
        if not isinstance(answer, str):
            raise TypeError("llm_client.generate() must return a str")
        return {self.output_key: answer, "prompt": prompt}
