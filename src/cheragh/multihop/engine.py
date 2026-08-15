"""Multi-hop RAG engine.

The engine decomposes complex questions into sub-queries, performs iterative
retrieval, carries forward a compact evidence chain, then synthesizes one final
answer from all gathered evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from ..base import BaseRetriever, Document, ExtractiveLLMClient, LLMClient, _validate_top_k
from ..schema import RAGResponse, Source
from ..pipeline import AdvancedRAGPipeline, DEFAULT_ANSWER_PROMPT_FR
from ..citations import validate_citations
from ..tracing import RAGTrace


class QueryDecomposer(Protocol):
    """Protocol for query decomposition strategies."""

    def decompose(self, query: str, max_steps: int = 4) -> list[str]:
        """Return ordered sub-queries for a complex question."""


@dataclass
class RuleBasedQueryDecomposer:
    """Dependency-free decomposer for analytical and comparative questions.

    The decomposer is intentionally conservative: it keeps the original query,
    splits obvious conjunctions, and adds focused comparison/causal sub-queries
    when relevant.
    """

    include_original: bool = True

    def decompose(self, query: str, max_steps: int = 4) -> list[str]:
        q = " ".join(query.split())
        candidates: list[str] = []
        if self.include_original:
            candidates.append(q)

        # Split around common multi-part separators while avoiding tiny shards.
        parts = re.split(r"\s+(?:et|puis|ensuite|ainsi que|versus|vs\.?|compared to|and|then)\s+", q, flags=re.I)
        for part in parts:
            part = part.strip(" ,;:.?\n\t")
            if len(part.split()) >= 3 and part.lower() != q.lower():
                candidates.append(part)

        lowered = q.lower()
        if any(word in lowered for word in ("compare", "comparer", "différence", "difference", "versus", " vs ")):
            candidates.append(f"Éléments de comparaison pour: {q}")
        if any(word in lowered for word in ("pourquoi", "cause", "causes", "raison", "explain", "why")):
            candidates.append(f"Causes et justifications documentées pour: {q}")
        if any(word in lowered for word in ("risque", "impact", "conséquence", "consequence", "impact")):
            candidates.append(f"Risques, impacts et conséquences liés à: {q}")

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            key = item.lower()
            if key not in seen:
                deduped.append(item)
                seen.add(key)
            if len(deduped) >= max_steps:
                break
        return deduped or [q]


@dataclass
class EvidenceHop:
    """One retrieval hop in a multi-hop chain."""

    step: int
    query: str
    documents: list[Document]
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "query": self.query,
            "rationale": self.rationale,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "score": doc.score,
                    "preview": doc.content[:240],
                    "metadata": doc.metadata,
                }
                for doc in self.documents
            ],
        }


@dataclass
class MultiHopRAGResult:
    """Detailed response from :class:`MultiHopRAGEngine`."""

    response: RAGResponse
    hops: list[EvidenceHop] = field(default_factory=list)
    decomposed_queries: list[str] = field(default_factory=list)

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
        data["multi_hop"] = {
            "decomposed_queries": self.decomposed_queries,
            "hops": [hop.to_dict() for hop in self.hops],
        }
        return data


class MultiHopRAGEngine:
    """Iterative multi-hop retrieval and synthesis.

    Parameters
    ----------
    retriever:
        Retriever used at each hop.
    llm_client:
        LLM used to synthesize the final answer. Defaults to an extractive local
        fallback.
    decomposer:
        Optional decomposer. Defaults to :class:`RuleBasedQueryDecomposer`.
    max_steps:
        Maximum number of retrieval hops.
    top_k_per_step:
        Number of documents retrieved at each hop.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        llm_client: LLMClient | None = None,
        decomposer: QueryDecomposer | None = None,
        max_steps: int = 4,
        top_k_per_step: int = 4,
        final_top_k: int = 8,
        answer_prompt: str = DEFAULT_ANSWER_PROMPT_FR,
        trace_enabled: bool = True,
    ):
        max_steps = _validate_top_k(max_steps, name="max_steps")
        self.retriever = retriever
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.decomposer = decomposer or RuleBasedQueryDecomposer()
        self.max_steps = max_steps
        self.top_k_per_step = _validate_top_k(top_k_per_step, name="top_k_per_step")
        self.final_top_k = _validate_top_k(final_top_k, name="final_top_k")
        self.answer_prompt = answer_prompt
        self.trace_enabled = trace_enabled

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> MultiHopRAGResult:
        trace = RAGTrace(query=query) if self.trace_enabled else None
        subqueries, hops, docs = self._collect_evidence(query, top_k=top_k, trace=trace)
        context = AdvancedRAGPipeline._format_context(docs)
        prompt = self.answer_prompt.format(context=context, query=query)
        if trace:
            trace.prompt = prompt
        generation_step = trace.start_step("multi_hop_generation", document_count=len(docs)) if trace else None
        answer = self.llm_client.generate(prompt, **generate_kwargs)
        if generation_step:
            generation_step.finish(answer_chars=len(answer))
        if trace:
            trace.record_generation(prompt=prompt, answer=answer, model=getattr(self.llm_client, "model", None))

        validation = validate_citations(answer, docs, require_citations=False)
        if trace:
            trace.warnings.extend(validation.warnings)
            trace.finish(answer_chars=len(answer), prompt_chars=len(prompt))
        effective_top_k = self.final_top_k if top_k is None else top_k
        response = RAGResponse(
            query=query,
            answer=answer,
            sources=[Source.from_document(doc) for doc in docs],
            retrieved_documents=docs,
            prompt=prompt,
            metadata={
                "architecture": "multi_hop",
                "decomposed_queries": subqueries,
                "hop_count": len(hops),
                "top_k": effective_top_k,
            },
            citations=validation.citations,
            warnings=validation.warnings,
            grounded_score=validation.grounded_score,
            unsourced_claims=validation.unsourced_claims,
            citation_validation=validation,
            trace=trace,
        )
        return MultiHopRAGResult(response=response, hops=hops, decomposed_queries=subqueries)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Expose multi-hop retrieval as a retriever-compatible method."""
        _, _, documents = self._collect_evidence(query, top_k=top_k, trace=None)
        return documents

    def _collect_evidence(
        self,
        query: str,
        *,
        top_k: int | None,
        trace: RAGTrace | None,
    ) -> tuple[list[str], list[EvidenceHop], list[Document]]:
        """Collect and merge evidence without invoking the generation client."""

        step = trace.start_step("multi_hop_decomposition", max_steps=self.max_steps) if trace else None
        subqueries = self.decomposer.decompose(query, max_steps=self.max_steps)
        if trace:
            trace.query_variants = list(subqueries)
        if step:
            step.finish(query_count=len(subqueries))

        hops: list[EvidenceHop] = []
        merged: dict[str, Document] = {}
        evidence_snippets: list[str] = []
        for idx, subquery in enumerate(subqueries, start=1):
            retrieval_query = self._augment_query(subquery, evidence_snippets)
            step = trace.start_step("multi_hop_retrieval", hop=idx, query=retrieval_query) if trace else None
            documents = self.retriever.retrieve(retrieval_query, top_k=self.top_k_per_step)
            if trace:
                trace.add_retrieval(retrieval_query, documents)
            if step:
                step.finish(document_count=len(documents))
            hops.append(
                EvidenceHop(
                    step=idx,
                    query=retrieval_query,
                    documents=documents,
                    rationale="retrieved_subquery_evidence",
                )
            )
            for document in documents:
                key = document.doc_id or document.content
                previous = merged.get(key)
                if previous is None or (document.score or 0.0) > (previous.score or 0.0):
                    merged[key] = document
            evidence_snippets.extend(document.content[:360] for document in documents[:2])

        ordered = sorted(
            merged.values(),
            key=lambda document: (document.score is not None, document.score or 0.0),
            reverse=True,
        )
        effective_top_k = self.final_top_k if top_k is None else _validate_top_k(top_k)
        return subqueries, hops, ordered[:effective_top_k]

    def _augment_query(self, query: str, evidence_snippets: list[str]) -> str:
        if not evidence_snippets:
            return query
        compact_evidence = " ".join(evidence_snippets[-3:])[:700]
        return f"{query}\nContexte découvert précédemment: {compact_evidence}"
