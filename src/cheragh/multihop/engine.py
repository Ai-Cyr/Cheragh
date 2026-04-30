"""Multi-hop RAG engine.

The engine decomposes complex questions into sub-queries, performs iterative
retrieval, carries forward a compact evidence chain, then synthesizes one final
answer from all gathered evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from ..base import BaseRetriever, Document, ExtractiveLLMClient, LLMClient
from ..engine import RAGResponse, Source
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

    def to_dict(self) -> dict[str, Any]:
        data = self.response.to_dict()
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
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        self.retriever = retriever
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.decomposer = decomposer or RuleBasedQueryDecomposer()
        self.max_steps = max_steps
        self.top_k_per_step = top_k_per_step
        self.final_top_k = final_top_k
        self.answer_prompt = answer_prompt
        self.trace_enabled = trace_enabled

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> MultiHopRAGResult:
        trace = RAGTrace() if self.trace_enabled else None
        step = trace.start_step("multi_hop_decomposition", max_steps=self.max_steps) if trace else None
        subqueries = self.decomposer.decompose(query, max_steps=self.max_steps)
        if step:
            step.finish(query_count=len(subqueries))

        hops: list[EvidenceHop] = []
        merged: dict[str, Document] = {}
        evidence_snippets: list[str] = []

        for idx, subquery in enumerate(subqueries, start=1):
            retrieval_query = self._augment_query(subquery, evidence_snippets)
            step = trace.start_step("multi_hop_retrieval", hop=idx, query=retrieval_query) if trace else None
            docs = self.retriever.retrieve(retrieval_query, top_k=self.top_k_per_step)
            if trace:
                trace.add_retrieval(retrieval_query, docs)
            if step:
                step.finish(document_count=len(docs))
            hops.append(EvidenceHop(step=idx, query=retrieval_query, documents=docs, rationale="retrieved_subquery_evidence"))

            for doc in docs:
                key = doc.doc_id or doc.content
                previous = merged.get(key)
                if previous is None or (doc.score or 0.0) > (previous.score or 0.0):
                    merged[key] = doc
            evidence_snippets.extend(doc.content[:360] for doc in docs[:2])

        ordered_docs = sorted(merged.values(), key=lambda doc: (doc.score is not None, doc.score or 0.0), reverse=True)
        docs = ordered_docs[: (top_k or self.final_top_k)]
        context = AdvancedRAGPipeline._format_context(docs)
        prompt = self.answer_prompt.format(context=context, query=query)
        if trace:
            trace.prompt = prompt
        generation_step = trace.start_step("multi_hop_generation", document_count=len(docs)) if trace else None
        answer = self.llm_client.generate(prompt, **generate_kwargs)
        if generation_step:
            generation_step.finish(answer_chars=len(answer))

        validation = validate_citations(answer, docs, require_citations=False)
        response = RAGResponse(
            query=query,
            answer=answer,
            sources=[Source(doc.doc_id, doc.score, doc.content[:240], dict(doc.metadata)) for doc in docs],
            retrieved_documents=docs,
            prompt=prompt,
            metadata={
                "architecture": "multi_hop",
                "decomposed_queries": subqueries,
                "hop_count": len(hops),
                "top_k": top_k or self.final_top_k,
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
        return self.ask(query, top_k=top_k).response.retrieved_documents

    def _augment_query(self, query: str, evidence_snippets: list[str]) -> str:
        if not evidence_snippets:
            return query
        compact_evidence = " ".join(evidence_snippets[-3:])[:700]
        return f"{query}\nContexte découvert précédemment: {compact_evidence}"
