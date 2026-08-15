"""Federated RAG over multiple engines, retrievers or callables."""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Mapping

from ..base import BaseRetriever, Document, ExtractiveLLMClient, LLMClient, _validate_top_k
from ..schema import RAGResponse, Source
from ..pipeline import AdvancedRAGPipeline, DEFAULT_ANSWER_PROMPT_FR
from ..citations import validate_citations
from ..tracing import RAGTrace


@dataclass
class FederatedSourceResult:
    """Evidence retrieved from one federated source."""

    source_name: str
    documents: list[Document] = field(default_factory=list)
    answer: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "answer": self.answer,
            "error": self.error,
            "documents": [
                {"doc_id": doc.doc_id, "score": doc.score, "preview": doc.content[:240], "metadata": doc.metadata}
                for doc in self.documents
            ],
        }


class FederatedRetriever(BaseRetriever):
    """Retriever that queries several source retrievers and merges evidence."""

    def __init__(
        self,
        sources: Mapping[str, Any],
        top_k_per_source: int = 5,
        continue_on_error: bool = True,
    ):
        self.sources = dict(sources)
        self.top_k_per_source = _validate_top_k(top_k_per_source, name="top_k_per_source")
        self.continue_on_error = continue_on_error
        self.last_results: list[FederatedSourceResult] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        self.last_results = []
        merged: dict[str, Document] = {}
        for source_name, source in self.sources.items():
            try:
                result = _query_source(source_name, source, query, self.top_k_per_source)
            except Exception as exc:  # pragma: no cover - defensive branch
                if not self.continue_on_error:
                    raise
                result = FederatedSourceResult(source_name=source_name, error=str(exc))
            self.last_results.append(result)
            qualified_documents: list[Document] = []
            for rank, doc in enumerate(result.documents):
                original_doc_id = doc.doc_id
                local_doc_id = str(original_doc_id) if original_doc_id is not None else f"document-{rank + 1}"
                qualified_doc_id = f"{source_name}::{local_doc_id}"
                score = doc.score if doc.score is not None else 1.0 / (rank + 1)
                metadata = {
                    **dict(doc.metadata or {}),
                    "original_doc_id": original_doc_id,
                    "source_name": source_name,
                    "federated_rank": rank + 1,
                }
                qualified = Document(doc.content, metadata=metadata, doc_id=qualified_doc_id, score=score)
                qualified_documents.append(qualified)
                merged[qualified_doc_id] = qualified
            result.documents = qualified_documents
            if result.answer and not result.documents:
                doc_id = f"{source_name}::answer"
                merged[doc_id] = Document(
                    result.answer,
                    metadata={
                        "source_name": source_name,
                        "source_type": "answer",
                        "original_doc_id": "answer",
                    },
                    doc_id=doc_id,
                    score=0.5,
                )
        ordered = sorted(merged.values(), key=lambda doc: (doc.score is not None, doc.score or 0.0), reverse=True)
        return ordered[:top_k]


def _query_source(source_name: str, source: Any, query: str, top_k: int) -> FederatedSourceResult:
    if isinstance(source, BaseRetriever) or hasattr(source, "retrieve"):
        docs = list(source.retrieve(query, top_k=top_k))
        return FederatedSourceResult(source_name=source_name, documents=docs)
    if hasattr(source, "retriever") and hasattr(source.retriever, "retrieve"):
        docs = list(source.retriever.retrieve(query, top_k=top_k))
        return FederatedSourceResult(source_name=source_name, documents=docs)
    if hasattr(source, "ask") and callable(source.ask):
        response = _call_source_ask(source.ask, query, top_k)
        return _coerce_callable_result(source_name, response)
    if callable(source):
        value = source(query)
        return _coerce_callable_result(source_name, value)
    raise TypeError(f"Unsupported federated source {source_name!r}: {source!r}")


def _coerce_callable_result(source_name: str, value: Any) -> FederatedSourceResult:
    if isinstance(value, FederatedSourceResult):
        return value
    if isinstance(value, Document):
        return FederatedSourceResult(source_name=source_name, documents=[value])
    if isinstance(value, str):
        return FederatedSourceResult(source_name=source_name, answer=value)

    wrapped_response = getattr(value, "response", None)
    if wrapped_response is not None and wrapped_response is not value:
        result = _coerce_callable_result(source_name, wrapped_response)
        wrapper_documents = getattr(value, "documents", None)
        if not result.documents and wrapper_documents is not None:
            result.documents = _coerce_documents(source_name, wrapper_documents)
        if result.answer is None:
            result.answer = getattr(value, "answer", None)
        return result

    if hasattr(value, "retrieved_documents") or hasattr(value, "documents") or hasattr(value, "answer"):
        documents = getattr(value, "retrieved_documents", None)
        if documents is None:
            documents = getattr(value, "documents", None)
        return FederatedSourceResult(
            source_name=source_name,
            documents=_coerce_documents(source_name, documents),
            answer=getattr(value, "answer", None),
        )
    if isinstance(value, list):
        return FederatedSourceResult(source_name=source_name, documents=_coerce_documents(source_name, value))
    if isinstance(value, dict):
        if value.get("response") is not None:
            result = _coerce_callable_result(source_name, value["response"])
            if not result.documents:
                result.documents = _coerce_documents(
                    source_name,
                    value.get("documents") or value.get("docs"),
                )
            if result.answer is None:
                result.answer = value.get("answer")
            return result
        docs = value.get("documents") or value.get("docs") or []
        answer = value.get("answer")
        return FederatedSourceResult(
            source_name=source_name,
            documents=_coerce_documents(source_name, docs),
            answer=answer,
        )
    return FederatedSourceResult(source_name=source_name, answer=str(value))


def _coerce_documents(source_name: str, value: Any) -> list[Document]:
    if value is None:
        return []
    if isinstance(value, Document):
        return [value]
    if isinstance(value, str):
        return [Document(value, metadata={"source_name": source_name})]
    return [
        item if isinstance(item, Document) else Document(str(item), metadata={"source_name": source_name})
        for item in value
    ]


def _call_source_ask(ask: Any, query: str, top_k: int) -> Any:
    """Call an ``ask`` method without assuming it accepts ``top_k``."""

    try:
        parameters = inspect.signature(ask).parameters
    except (TypeError, ValueError):
        return ask(query)
    top_k_parameter = parameters.get("top_k")
    if top_k_parameter is not None:
        if top_k_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            return ask(query, top_k)
        return ask(query, top_k=top_k)
    return ask(query)


@dataclass
class FederatedRAGResult:
    """Detailed response from :class:`FederatedRAGEngine`."""

    response: RAGResponse
    source_results: list[FederatedSourceResult] = field(default_factory=list)

    @property
    def answer(self) -> str:
        return self.response.answer

    @property
    def sources(self) -> list[Source]:
        return self.response.sources

    @property
    def metadata(self) -> dict[str, Any]:
        return self.response.metadata

    @property
    def query(self) -> str:
        return self.response.query

    @property
    def retrieved_documents(self) -> list[Document]:
        return self.response.retrieved_documents

    @property
    def warnings(self) -> list[str]:
        return self.response.warnings

    @property
    def trace(self) -> RAGTrace | None:
        return self.response.trace

    def to_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        data = self.response.to_dict(include_prompt=include_prompt)
        data["federated"] = {"sources": [result.to_dict() for result in self.source_results]}
        return data


class FederatedRAGEngine:
    """RAG engine that federates retrieval over several sources.

    Sources can be regular ``BaseRetriever`` instances, ``RAGEngine`` objects,
    objects exposing ``ask`` or ``retrieve``, or callables returning documents,
    answers, dictionaries, or ``RAGResponse`` objects.
    """

    def __init__(
        self,
        sources: Mapping[str, Any],
        llm_client: LLMClient | None = None,
        top_k_per_source: int = 5,
        top_k: int = 8,
        answer_prompt: str = DEFAULT_ANSWER_PROMPT_FR,
        continue_on_error: bool = True,
        trace_enabled: bool = True,
    ):
        if not sources:
            raise ValueError("FederatedRAGEngine requires at least one source")
        self.sources = dict(sources)
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.retriever = FederatedRetriever(self.sources, top_k_per_source=top_k_per_source, continue_on_error=continue_on_error)
        self.top_k = _validate_top_k(top_k)
        self.answer_prompt = answer_prompt
        self.trace_enabled = trace_enabled

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> FederatedRAGResult:
        trace = RAGTrace(query=query) if self.trace_enabled else None
        effective_top_k = self.top_k if top_k is None else _validate_top_k(top_k)
        step = trace.start_step("federated_retrieval", source_count=len(self.sources), top_k=effective_top_k) if trace else None
        docs = self.retriever.retrieve(query, top_k=effective_top_k)
        if trace:
            trace.query_variants = [f"{result.source_name}:{query}" for result in self.retriever.last_results]
            for result in self.retriever.last_results:
                trace.add_retrieval(f"{result.source_name}:{query}", result.documents)
        if step:
            step.finish(document_count=len(docs), source_count=len(self.retriever.last_results))

        context = AdvancedRAGPipeline._format_context(docs)
        source_answers = [result for result in self.retriever.last_results if result.answer]
        if source_answers:
            rendered_answers = "\n".join(f"[{item.source_name}] {item.answer}" for item in source_answers)
            context = f"{context}\n\nRéponses intermédiaires par source:\n{rendered_answers}" if context else rendered_answers
        prompt = self.answer_prompt.format(context=context, query=query)
        if trace:
            trace.prompt = prompt
        gen_step = trace.start_step("federated_generation", document_count=len(docs)) if trace else None
        answer = self.llm_client.generate(prompt, **generate_kwargs)
        if gen_step:
            gen_step.finish(answer_chars=len(answer))
        if trace:
            trace.record_generation(prompt=prompt, answer=answer, model=getattr(self.llm_client, "model", None))
        validation = validate_citations(answer, docs, require_citations=False)
        warnings = validation.warnings + [
            f"source_error:{result.source_name}"
            for result in self.retriever.last_results
            if result.error
        ]
        if trace:
            trace.warnings.extend(warnings)
            trace.finish(
                answer_chars=len(answer),
                prompt_chars=len(prompt),
                source_count=len(self.retriever.last_results),
            )

        response = RAGResponse(
            query=query,
            answer=answer,
            sources=[Source.from_document(doc) for doc in docs],
            retrieved_documents=docs,
            prompt=prompt,
            metadata={
                "architecture": "federated_rag",
                "source_count": len(self.sources),
                "sources_queried": [result.source_name for result in self.retriever.last_results],
                "source_errors": {result.source_name: result.error for result in self.retriever.last_results if result.error},
                "top_k": effective_top_k,
            },
            citations=validation.citations,
            warnings=warnings,
            grounded_score=validation.grounded_score,
            unsourced_claims=validation.unsourced_claims,
            citation_validation=validation,
            trace=trace,
        )
        return FederatedRAGResult(response=response, source_results=list(self.retriever.last_results))

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.retriever.retrieve(query, top_k=top_k)
