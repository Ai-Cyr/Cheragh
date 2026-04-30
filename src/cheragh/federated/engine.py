"""Federated RAG over multiple engines, retrievers or callables."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..base import BaseRetriever, Document, ExtractiveLLMClient, LLMClient
from ..engine import RAGEngine, RAGResponse, Source
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
        self.top_k_per_source = top_k_per_source
        self.continue_on_error = continue_on_error
        self.last_results: list[FederatedSourceResult] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
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
            for rank, doc in enumerate(result.documents):
                key = f"{source_name}:{doc.doc_id or doc.content}"
                score = doc.score if doc.score is not None else 1.0 / (rank + 1)
                metadata = {**doc.metadata, "source_name": source_name, "federated_rank": rank + 1}
                merged[key] = Document(doc.content, metadata=metadata, doc_id=doc.doc_id or key, score=score)
            if result.answer and not result.documents:
                doc_id = f"{source_name}::answer"
                merged[doc_id] = Document(result.answer, metadata={"source_name": source_name, "source_type": "answer"}, doc_id=doc_id, score=0.5)
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
        response = source.ask(query, top_k=top_k)
        docs = list(getattr(response, "retrieved_documents", []) or [])
        answer = getattr(response, "answer", None)
        return FederatedSourceResult(source_name=source_name, documents=docs, answer=answer)
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
    if hasattr(value, "retrieved_documents") or hasattr(value, "answer"):
        return FederatedSourceResult(
            source_name=source_name,
            documents=list(getattr(value, "retrieved_documents", []) or []),
            answer=getattr(value, "answer", None),
        )
    if isinstance(value, list):
        documents = [item if isinstance(item, Document) else Document(str(item), metadata={"source_name": source_name}) for item in value]
        return FederatedSourceResult(source_name=source_name, documents=documents)
    if isinstance(value, dict):
        docs = value.get("documents") or value.get("docs") or []
        documents = [doc if isinstance(doc, Document) else Document(str(doc), metadata={"source_name": source_name}) for doc in docs]
        answer = value.get("answer")
        return FederatedSourceResult(source_name=source_name, documents=documents, answer=answer)
    return FederatedSourceResult(source_name=source_name, answer=str(value))


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

    def to_dict(self) -> dict[str, Any]:
        data = self.response.to_dict()
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
        **engine_kwargs: Any,
    ):
        if not sources:
            raise ValueError("FederatedRAGEngine requires at least one source")
        self.sources = dict(sources)
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.retriever = FederatedRetriever(self.sources, top_k_per_source=top_k_per_source, continue_on_error=continue_on_error)
        self.top_k = top_k
        self.answer_prompt = answer_prompt
        self.trace_enabled = trace_enabled
        self.engine_kwargs = engine_kwargs

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> FederatedRAGResult:
        trace = RAGTrace() if self.trace_enabled else None
        effective_top_k = top_k or self.top_k
        step = trace.start_step("federated_retrieval", source_count=len(self.sources), top_k=effective_top_k) if trace else None
        docs = self.retriever.retrieve(query, top_k=effective_top_k)
        if trace:
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
        validation = validate_citations(answer, docs, require_citations=False)

        response = RAGResponse(
            query=query,
            answer=answer,
            sources=[Source(doc.doc_id, doc.score, doc.content[:240], dict(doc.metadata)) for doc in docs],
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
            warnings=validation.warnings + [f"source_error:{r.source_name}" for r in self.retriever.last_results if r.error],
            grounded_score=validation.grounded_score,
            unsourced_claims=validation.unsourced_claims,
            citation_validation=validation,
            trace=trace,
        )
        return FederatedRAGResult(response=response, source_results=list(self.retriever.last_results))

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.retriever.retrieve(query, top_k=top_k)
