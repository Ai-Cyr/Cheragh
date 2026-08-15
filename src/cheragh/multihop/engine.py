"""Static and observation-driven multi-hop RAG.

The dynamic path follows the planning -> retrieval -> observation shape used by
IRCoT/ReAct-style systems, but deliberately does not claim to reproduce their
prompts, training, or reasoning quality.  Planning is an injectable boundary;
the bundled rule-based planner is deterministic and the optional LLM adapter
uses a strict JSON contract with a deterministic fallback.

Only :meth:`MultiHopRAGEngine.ask` synthesizes a final answer.  The compatible
``retrieve()`` method may perform several retrieval/planning hops but never calls
the engine's answer-generation client.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import numbers
import re
from typing import Any, Mapping, Protocol, Sequence

from ..base import (
    BaseRetriever,
    Document,
    ExtractiveLLMClient,
    LLMClient,
    _snapshot_document,
    _validate_top_k,
)
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

    def __post_init__(self) -> None:
        if not isinstance(self.include_original, bool):
            raise TypeError("include_original must be a bool")

    def decompose(self, query: str, max_steps: int = 4) -> list[str]:
        q = _validate_query(query)
        max_steps = _validate_top_k(max_steps, name="max_steps")
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


class PlanningAction(str, Enum):
    """Explicit control decision returned by a multi-hop planner."""

    NEXT = "next"
    STOP = "stop"


@dataclass(frozen=True)
class PlanningDecision:
    """Request another retrieval query or stop the evidence loop."""

    action: PlanningAction
    query: str | None = None
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            action = self.action if isinstance(self.action, PlanningAction) else PlanningAction(self.action)
        except (TypeError, ValueError) as exc:
            raise ValueError("action must be PlanningAction.NEXT or PlanningAction.STOP") from exc
        if self.query is not None and not isinstance(self.query, str):
            raise TypeError("planning query must be a string or None")
        query = " ".join(self.query.split()) if self.query is not None else None
        if action is PlanningAction.NEXT and not query:
            raise ValueError("a NEXT planning decision requires a non-empty query")
        if action is PlanningAction.STOP and query:
            raise ValueError("a STOP planning decision cannot contain a query")
        if not isinstance(self.rationale, str):
            raise TypeError("planning rationale must be a string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("planning metadata must be a mapping")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "metadata", deepcopy(dict(self.metadata)))

    @classmethod
    def next(cls, query: str, *, rationale: str = "", **metadata: Any) -> "PlanningDecision":
        return cls(PlanningAction.NEXT, query=query, rationale=rationale, metadata=metadata)

    @classmethod
    def stop(cls, *, rationale: str = "", **metadata: Any) -> "PlanningDecision":
        return cls(PlanningAction.STOP, rationale=rationale, metadata=metadata)

    def snapshot(self) -> "PlanningDecision":
        return PlanningDecision(
            action=self.action,
            query=self.query,
            rationale=self.rationale,
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "query": self.query,
            "rationale": self.rationale,
            "metadata": deepcopy(self.metadata),
        }


@dataclass
class EvidenceHop:
    """One retrieval hop in a multi-hop chain."""

    step: int
    query: str
    documents: list[Document]
    rationale: str = ""
    planned_query: str | None = None
    observation: str = ""

    def __post_init__(self) -> None:
        self.step = _validate_top_k(self.step, name="step")
        self.query = _validate_query(self.query, name="hop query")
        if self.planned_query is not None:
            self.planned_query = _validate_query(self.planned_query, name="planned query")
        if not isinstance(self.rationale, str) or not isinstance(self.observation, str):
            raise TypeError("hop rationale and observation must be strings")
        self.documents = _validated_document_snapshots(self.documents)

    def snapshot(self) -> "EvidenceHop":
        return EvidenceHop(
            step=self.step,
            query=self.query,
            documents=self.documents,
            rationale=self.rationale,
            planned_query=self.planned_query,
            observation=self.observation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "query": self.query,
            "planned_query": self.planned_query,
            "rationale": self.rationale,
            "observation": self.observation,
            "documents": [
                {
                    "doc_id": doc.doc_id,
                    "score": doc.score,
                    "preview": doc.content[:240],
                    "metadata": deepcopy(doc.metadata or {}),
                }
                for doc in self.documents
            ],
        }


@dataclass(frozen=True)
class PlanningContext:
    """Read-only-style snapshot supplied before the next retrieval hop."""

    original_query: str
    next_step: int
    max_steps: int
    hops: tuple[EvidenceHop, ...] = ()
    evidence: tuple[Document, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "original_query", _validate_query(self.original_query))
        object.__setattr__(self, "next_step", _validate_top_k(self.next_step, name="next_step"))
        object.__setattr__(self, "max_steps", _validate_top_k(self.max_steps, name="max_steps"))
        if self.next_step > self.max_steps:
            raise ValueError("next_step cannot exceed max_steps")
        if any(not isinstance(hop, EvidenceHop) for hop in self.hops):
            raise TypeError("hops must contain only EvidenceHop values")
        if self.next_step != len(self.hops) + 1:
            raise ValueError("next_step must equal len(hops) + 1")
        object.__setattr__(self, "hops", tuple(hop.snapshot() for hop in self.hops))
        object.__setattr__(self, "evidence", tuple(_validated_document_snapshots(self.evidence)))

    @property
    def remaining_steps(self) -> int:
        return self.max_steps - self.next_step + 1


class MultiHopPlanner(Protocol):
    """Plan the next retrieval query from prior evidence observations."""

    def plan(self, context: PlanningContext) -> PlanningDecision:
        """Return an explicit NEXT or STOP decision."""


class RuleBasedMultiHopPlanner:
    """Deterministic evidence-aware planner for offline use.

    It starts from the original query, consumes rule-based decomposition
    candidates, and enriches later queries with salient terms discovered in the
    latest observation.  It stops early on an empty or duplicate-only hop.  The
    heuristics are transparent baselines, not a semantic reasoning model.
    """

    def __init__(self, decomposer: QueryDecomposer | None = None, *, bridge_terms: int = 3) -> None:
        if isinstance(bridge_terms, bool) or not isinstance(bridge_terms, int):
            raise TypeError("bridge_terms must be an int")
        if bridge_terms < 0:
            raise ValueError("bridge_terms must be >= 0")
        if decomposer is not None and not callable(getattr(decomposer, "decompose", None)):
            raise TypeError("decomposer must define decompose()")
        self.decomposer = decomposer if decomposer is not None else RuleBasedQueryDecomposer()
        self.bridge_terms = bridge_terms

    def plan(self, context: PlanningContext) -> PlanningDecision:
        if not isinstance(context, PlanningContext):
            raise TypeError("context must be a PlanningContext")
        if not context.hops:
            return PlanningDecision.next(
                context.original_query,
                rationale="deterministic_initial_query",
            )

        latest = context.hops[-1]
        if not latest.documents:
            return PlanningDecision.stop(rationale="no_evidence_observed")
        if len(context.hops) > 1 and not _new_evidence_ids(latest, context.hops[:-1]):
            return PlanningDecision.stop(rationale="no_new_evidence_observed")

        candidates = _validated_subqueries(
            self.decomposer.decompose(context.original_query, max_steps=context.max_steps),
            max_steps=context.max_steps,
        )
        executed = [(hop.planned_query or hop.query).casefold() for hop in context.hops]
        pending = next(
            (
                candidate
                for candidate in candidates
                if not any(
                    planned == candidate.casefold()
                    or planned.startswith(f"{candidate.casefold()} ")
                    for planned in executed
                )
            ),
            None,
        )
        if pending is None:
            return PlanningDecision.stop(rationale="decomposition_exhausted")

        bridge = _salient_bridge_terms(latest.documents, pending, limit=self.bridge_terms)
        query = f"{pending} {' '.join(bridge)}" if bridge else pending
        return PlanningDecision.next(
            query,
            rationale="evidence_conditioned_follow_up" if bridge else "decomposition_follow_up",
            bridge_terms=bridge,
            observed_doc_ids=[document.doc_id for document in latest.documents if document.doc_id],
        )


MULTI_HOP_PLANNER_PROMPT = """Planifie la prochaine étape de recherche documentaire.

Réponds uniquement avec un objet JSON sur une ligne:
- {{"action":"next","query":"requête précise","rationale":"raison courte"}}
- {{"action":"stop","rationale":"raison courte"}}

Arrête si les preuves suffisent, si la dernière recherche est vide, ou si aucune
nouvelle requête utile n'est possible. N'invente pas de preuve.

Question initiale: {query}
Étape suivante: {next_step}/{max_steps}
Observations:
{observations}

Décision JSON:"""


class LLMMultiHopPlanner:
    """Optional JSON-only planning adapter with a deterministic fallback."""

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        fallback: MultiHopPlanner | None = None,
        prompt: str = MULTI_HOP_PLANNER_PROMPT,
    ) -> None:
        if not callable(getattr(llm_client, "generate", None)):
            raise TypeError("llm_client must define generate()")
        if fallback is not None and not callable(getattr(fallback, "plan", None)):
            raise TypeError("fallback must define plan()")
        if not isinstance(prompt, str) or not all(
            placeholder in prompt
            for placeholder in ("{query}", "{next_step}", "{max_steps}", "{observations}")
        ):
            raise ValueError("planner prompt must contain query, step, budget and observations placeholders")
        self.llm_client = llm_client
        self.fallback = fallback if fallback is not None else RuleBasedMultiHopPlanner()
        self.prompt = prompt

    def plan(self, context: PlanningContext) -> PlanningDecision:
        if not isinstance(context, PlanningContext):
            raise TypeError("context must be a PlanningContext")
        prompt = self.prompt.format(
            query=context.original_query,
            next_step=context.next_step,
            max_steps=context.max_steps,
            observations=_format_planning_observations(context.hops),
        )
        raw = self.llm_client.generate(prompt)
        try:
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise TypeError("planner output must be a JSON object")
            action = PlanningAction(str(payload.get("action", "")).strip().casefold())
            rationale = payload.get("rationale", "llm_planner")
            if not isinstance(rationale, str):
                raise TypeError("planner rationale must be a string")
            if action is PlanningAction.STOP:
                query = payload.get("query")
                if query is not None and (not isinstance(query, str) or query.strip()):
                    raise ValueError("STOP planner output cannot contain a query")
                if not context.hops:
                    raise ValueError("planner cannot claim evidence is sufficient before retrieval")
                return PlanningDecision.stop(rationale=rationale, planner="llm")
            query = payload.get("query")
            if not isinstance(query, str):
                raise TypeError("NEXT planner output requires a string query")
            return PlanningDecision.next(query, rationale=rationale, planner="llm")
        except (json.JSONDecodeError, TypeError, ValueError):
            fallback = self.fallback.plan(context)
            if not isinstance(fallback, PlanningDecision):
                raise TypeError("fallback.plan() must return a PlanningDecision")
            fallback = fallback.snapshot()
            return PlanningDecision(
                action=fallback.action,
                query=fallback.query,
                rationale=f"llm_output_invalid:{fallback.rationale}",
                metadata={**fallback.metadata, "planner": "deterministic_fallback"},
            )


@dataclass
class MultiHopRAGResult:
    """Detailed response from :class:`MultiHopRAGEngine`."""

    response: RAGResponse
    hops: list[EvidenceHop] = field(default_factory=list)
    decomposed_queries: list[str] = field(default_factory=list)
    planning_decisions: list[PlanningDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.response, RAGResponse):
            raise TypeError("response must be a RAGResponse")
        if isinstance(self.decomposed_queries, (str, bytes)):
            raise TypeError("decomposed_queries must be an iterable of strings")
        if any(not isinstance(hop, EvidenceHop) for hop in self.hops):
            raise TypeError("hops must contain only EvidenceHop values")
        if any(not isinstance(decision, PlanningDecision) for decision in self.planning_decisions):
            raise TypeError("planning_decisions must contain only PlanningDecision values")
        self.hops = [hop.snapshot() for hop in self.hops]
        self.decomposed_queries = [
            _validate_query(query, name="decomposed query") for query in self.decomposed_queries
        ]
        self.planning_decisions = [decision.snapshot() for decision in self.planning_decisions]

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
            "decomposed_queries": list(self.decomposed_queries),
            "hops": [hop.to_dict() for hop in self.hops],
            "planning_decisions": [decision.to_dict() for decision in self.planning_decisions],
        }
        return data


@dataclass
class _EvidenceCollection:
    queries: list[str]
    hops: list[EvidenceHop]
    documents: list[Document]
    decisions: list[PlanningDecision]
    mode: str


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
        Optional static decomposer. Defaults to
        :class:`RuleBasedQueryDecomposer` when no planner is supplied.
    planner:
        Optional observation-driven planner. Supplying it activates the dynamic
        loop and is mutually exclusive with an explicit ``decomposer``.
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
        planner: MultiHopPlanner | None = None,
    ):
        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must define retrieve()")
        if llm_client is not None and not callable(getattr(llm_client, "generate", None)):
            raise TypeError("llm_client must define generate()")
        if decomposer is not None and not callable(getattr(decomposer, "decompose", None)):
            raise TypeError("decomposer must define decompose()")
        if planner is not None and not callable(getattr(planner, "plan", None)):
            raise TypeError("planner must define plan()")
        if planner is not None and decomposer is not None:
            raise ValueError("planner and decomposer are mutually exclusive execution strategies")
        if not isinstance(answer_prompt, str) or "{context}" not in answer_prompt or "{query}" not in answer_prompt:
            raise ValueError("answer_prompt must contain {context} and {query}")
        if not isinstance(trace_enabled, bool):
            raise TypeError("trace_enabled must be a boolean")
        self.retriever = retriever
        self.llm_client = llm_client if llm_client is not None else ExtractiveLLMClient()
        self.decomposer = decomposer if decomposer is not None else RuleBasedQueryDecomposer()
        self.planner = planner
        self.max_steps = _validate_top_k(max_steps, name="max_steps")
        self.top_k_per_step = _validate_top_k(top_k_per_step, name="top_k_per_step")
        self.final_top_k = _validate_top_k(final_top_k, name="final_top_k")
        self.answer_prompt = answer_prompt
        self.trace_enabled = trace_enabled

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> MultiHopRAGResult:
        normalized_query = _validate_query(query)
        trace = RAGTrace(query=normalized_query) if self.trace_enabled else None
        collection = self._collect_evidence(normalized_query, top_k=top_k, trace=trace)
        subqueries, hops, docs = collection.queries, collection.hops, collection.documents
        context = AdvancedRAGPipeline._format_context(docs)
        prompt = self.answer_prompt.format(context=context, query=normalized_query)
        if trace:
            trace.prompt = prompt
        generation_step = trace.start_step("multi_hop_generation", document_count=len(docs)) if trace else None
        answer = self.llm_client.generate(prompt, **generate_kwargs)
        if not isinstance(answer, str):
            if generation_step:
                generation_step.finish(error="invalid_generation_type")
            if trace:
                trace.warnings.append("invalid_generation_type")
                trace.finish(prompt_chars=len(prompt))
            raise TypeError("llm_client.generate() must return a string")
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
            query=normalized_query,
            answer=answer,
            sources=[Source.from_document(doc) for doc in docs],
            retrieved_documents=docs,
            prompt=prompt,
            metadata={
                "architecture": "multi_hop",
                "decomposed_queries": subqueries,
                "hop_count": len(hops),
                "top_k": effective_top_k,
                "planning_mode": collection.mode,
                "planning_decisions": [decision.to_dict() for decision in collection.decisions],
                "stop_reason": collection.decisions[-1].rationale if collection.decisions else "",
            },
            citations=validation.citations,
            warnings=validation.warnings,
            grounded_score=validation.grounded_score,
            unsourced_claims=validation.unsourced_claims,
            citation_validation=validation,
            trace=trace,
        )
        return MultiHopRAGResult(
            response=response,
            hops=hops,
            decomposed_queries=subqueries,
            planning_decisions=collection.decisions,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Collect evidence without invoking the final-answer generation client."""

        collection = self._collect_evidence(_validate_query(query), top_k=top_k, trace=None)
        return [_snapshot_document(document) for document in collection.documents]

    def _collect_evidence(
        self,
        query: str,
        *,
        top_k: int | None,
        trace: RAGTrace | None,
    ) -> _EvidenceCollection:
        """Collect and merge evidence without invoking the generation client."""

        effective_top_k = self.final_top_k if top_k is None else _validate_top_k(top_k)
        if self.planner is not None:
            return self._collect_dynamic_evidence(
                query,
                top_k=effective_top_k,
                trace=trace,
            )
        return self._collect_static_evidence(query, top_k=effective_top_k, trace=trace)

    def _collect_static_evidence(
        self,
        query: str,
        *,
        top_k: int,
        trace: RAGTrace | None,
    ) -> _EvidenceCollection:
        """Preserve the original decomposition-first execution path."""

        step = trace.start_step("multi_hop_decomposition", max_steps=self.max_steps) if trace else None
        raw_subqueries = self.decomposer.decompose(query, max_steps=self.max_steps)
        subqueries = _validated_subqueries(raw_subqueries, max_steps=self.max_steps)
        if trace:
            trace.query_variants = list(subqueries)
        if step:
            step.finish(query_count=len(subqueries))

        hops: list[EvidenceHop] = []
        merged: dict[str, Document] = {}
        evidence_snippets: list[str] = []
        decisions: list[PlanningDecision] = []
        for idx, subquery in enumerate(subqueries, start=1):
            retrieval_query = self._augment_query(subquery, evidence_snippets)
            decision = PlanningDecision.next(subquery, rationale="static_decomposition")
            decisions.append(decision)
            hop = self._retrieve_hop(
                step_number=idx,
                planned_query=subquery,
                retrieval_query=retrieval_query,
                rationale=decision.rationale,
                trace=trace,
            )
            hops.append(hop)
            _merge_hop_documents(merged, hop)
            evidence_snippets.extend(document.content[:360] for document in hop.documents[:2])

        decisions.append(
            PlanningDecision.stop(
                rationale="max_steps_reached" if len(subqueries) == self.max_steps else "static_plan_complete"
            )
        )
        ordered = _ordered_evidence(merged, top_k=top_k)
        return _EvidenceCollection(
            queries=list(subqueries),
            hops=hops,
            documents=ordered,
            decisions=decisions,
            mode="static",
        )

    def _collect_dynamic_evidence(
        self,
        query: str,
        *,
        top_k: int,
        trace: RAGTrace | None,
    ) -> _EvidenceCollection:
        """Run a bounded plan -> retrieve -> observe loop."""

        planner = self.planner
        if planner is None:  # pragma: no cover - guarded by _collect_evidence
            raise RuntimeError("dynamic evidence collection requires a planner")
        hops: list[EvidenceHop] = []
        merged: dict[str, Document] = {}
        queries: list[str] = []
        decisions: list[PlanningDecision] = []

        while len(hops) < self.max_steps:
            context = PlanningContext(
                original_query=query,
                next_step=len(hops) + 1,
                max_steps=self.max_steps,
                hops=tuple(hops),
                evidence=tuple(merged.values()),
            )
            planning_step = (
                trace.start_step("multi_hop_planning", hop=context.next_step)
                if trace
                else None
            )
            decision = planner.plan(context)
            if not isinstance(decision, PlanningDecision):
                raise TypeError("planner.plan() must return a PlanningDecision")
            decision = decision.snapshot()
            decisions.append(decision)
            if planning_step:
                planning_step.finish(action=decision.action.value, rationale=decision.rationale)
            if decision.action is PlanningAction.STOP:
                break

            planned_query = decision.query
            if planned_query is None:  # pragma: no cover - PlanningDecision validates this
                raise RuntimeError("NEXT decision unexpectedly lacks a query")
            if planned_query.casefold() in {executed.casefold() for executed in queries}:
                decisions.append(
                    PlanningDecision.stop(
                        rationale="duplicate_planned_query",
                        duplicate_query=planned_query,
                    )
                )
                break
            queries.append(planned_query)
            hop = self._retrieve_hop(
                step_number=len(hops) + 1,
                planned_query=planned_query,
                retrieval_query=planned_query,
                rationale=decision.rationale,
                trace=trace,
            )
            hops.append(hop)
            _merge_hop_documents(merged, hop)

        if not decisions or decisions[-1].action is PlanningAction.NEXT:
            decisions.append(PlanningDecision.stop(rationale="max_steps_reached"))
        if trace:
            trace.query_variants = list(queries)
        return _EvidenceCollection(
            queries=queries,
            hops=hops,
            documents=_ordered_evidence(merged, top_k=top_k),
            decisions=decisions,
            mode="dynamic",
        )

    def _retrieve_hop(
        self,
        *,
        step_number: int,
        planned_query: str,
        retrieval_query: str,
        rationale: str,
        trace: RAGTrace | None,
    ) -> EvidenceHop:
        step = (
            trace.start_step("multi_hop_retrieval", hop=step_number, query=retrieval_query)
            if trace
            else None
        )
        raw_documents = self.retriever.retrieve(retrieval_query, top_k=self.top_k_per_step)
        documents = _validated_document_snapshots(raw_documents, limit=self.top_k_per_step)
        annotated = [
            _annotate_hop_document(
                document,
                step=step_number,
                retrieval_query=retrieval_query,
                planned_query=planned_query,
                rationale=rationale,
            )
            for document in documents
        ]
        observation = _observation_summary(annotated)
        if trace:
            trace.add_retrieval(retrieval_query, annotated)
        if step:
            step.finish(document_count=len(annotated), observation=observation)
        return EvidenceHop(
            step=step_number,
            query=retrieval_query,
            documents=annotated,
            rationale=rationale,
            planned_query=planned_query,
            observation=observation,
        )

    def _augment_query(self, query: str, evidence_snippets: list[str]) -> str:
        if not evidence_snippets:
            return query
        compact_evidence = " ".join(evidence_snippets[-3:])[:700]
        return f"{query}\nContexte découvert précédemment: {compact_evidence}"


def _validate_query(query: Any, *, name: str = "query") -> str:
    if not isinstance(query, str):
        raise TypeError(f"{name} must be a string")
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _validated_subqueries(values: Any, *, max_steps: int) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("decomposer.decompose() must return an iterable of strings")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise TypeError("decomposer.decompose() must return an iterable of strings") from exc
    queries: list[str] = []
    seen: set[str] = set()
    for value in iterator:
        query = _validate_query(value, name="decomposed query")
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= max_steps:
            break
    if not queries:
        raise ValueError("decomposer.decompose() must return at least one non-empty query")
    return queries


def _validated_document_snapshots(documents: Any, *, limit: int | None = None) -> list[Document]:
    if isinstance(documents, (str, bytes)):
        raise TypeError("retriever must return an iterable of Document instances")
    if limit is not None:
        limit = _validate_top_k(limit, name="limit")
    try:
        iterator = iter(documents)
    except TypeError as exc:
        raise TypeError("retriever must return an iterable of Document instances") from exc
    snapshots: list[Document] = []
    index = 0
    while limit is None or len(snapshots) < limit:
        try:
            document = next(iterator)
        except StopIteration:
            break
        if not isinstance(document, Document):
            raise TypeError(f"retriever result {index} must be a Document")
        if not isinstance(document.content, str):
            raise TypeError(f"retriever result {index}.content must be a string")
        if not document.content.strip():
            raise ValueError(f"retriever result {index}.content must be non-empty")
        if not isinstance(document.metadata, dict):
            raise TypeError(f"retriever result {index}.metadata must be a dict")
        if document.doc_id is not None and not isinstance(document.doc_id, str):
            raise TypeError(f"retriever result {index}.doc_id must be a string or None")
        if document.score is not None:
            if isinstance(document.score, bool) or not isinstance(document.score, numbers.Real):
                raise TypeError(f"retriever result {index}.score must be a real number or None")
            if not math.isfinite(float(document.score)):
                raise ValueError(f"retriever result {index}.score must be finite")
        snapshots.append(_snapshot_document(document))
        index += 1
    return snapshots


def _annotate_hop_document(
    document: Document,
    *,
    step: int,
    retrieval_query: str,
    planned_query: str,
    rationale: str,
) -> Document:
    snapshot = _snapshot_document(document)
    key = "multi_hop_provenance"
    upstream = deepcopy(snapshot.metadata.get(key)) if key in snapshot.metadata else None
    original_doc_id = snapshot.doc_id
    if not snapshot.doc_id or not snapshot.doc_id.strip():
        snapshot.doc_id = _anonymous_document_id(snapshot)
    provenance: dict[str, Any] = {
        "first_seen_step": step,
        "last_seen_step": step,
        "seen_steps": [step],
        "retrieval_queries": [retrieval_query],
        "planned_queries": [planned_query],
        "rationales": [rationale],
        "occurrences": [
            {
                "step": step,
                "retrieval_query": retrieval_query,
                "planned_query": planned_query,
                "score": snapshot.score,
            }
        ],
    }
    if original_doc_id != snapshot.doc_id:
        provenance["synthetic_doc_id"] = True
        provenance["original_doc_id"] = original_doc_id
    if upstream is not None:
        provenance["upstream"] = upstream
    snapshot.metadata[key] = provenance
    return snapshot


def _merge_hop_documents(merged: dict[str, Document], hop: EvidenceHop) -> None:
    for document in hop.documents:
        key = _document_key(document)
        previous = merged.get(key)
        if previous is None:
            merged[key] = _snapshot_document(document)
            continue

        selected = document if _ranking_score(document) > _ranking_score(previous) else previous
        combined = _snapshot_document(selected)
        previous_provenance = previous.metadata.get("multi_hop_provenance", {})
        current_provenance = document.metadata.get("multi_hop_provenance", {})
        if not isinstance(previous_provenance, Mapping) or not isinstance(current_provenance, Mapping):
            raise TypeError("multi_hop_provenance metadata must be a mapping")
        occurrences = [
            *deepcopy(list(previous_provenance.get("occurrences", []))),
            *deepcopy(list(current_provenance.get("occurrences", []))),
        ]
        seen_steps = _unique_values(
            [*previous_provenance.get("seen_steps", []), *current_provenance.get("seen_steps", [])]
        )
        combined.metadata["multi_hop_provenance"] = {
            "first_seen_step": min(seen_steps),
            "last_seen_step": max(seen_steps),
            "seen_steps": seen_steps,
            "retrieval_queries": _unique_values(
                [
                    *previous_provenance.get("retrieval_queries", []),
                    *current_provenance.get("retrieval_queries", []),
                ]
            ),
            "planned_queries": _unique_values(
                [
                    *previous_provenance.get("planned_queries", []),
                    *current_provenance.get("planned_queries", []),
                ]
            ),
            "rationales": _unique_values(
                [*previous_provenance.get("rationales", []), *current_provenance.get("rationales", [])]
            ),
            "occurrences": occurrences,
        }
        upstream = previous_provenance.get("upstream", current_provenance.get("upstream"))
        if upstream is not None:
            combined.metadata["multi_hop_provenance"]["upstream"] = deepcopy(upstream)
        if previous_provenance.get("synthetic_doc_id") or current_provenance.get("synthetic_doc_id"):
            combined.metadata["multi_hop_provenance"]["synthetic_doc_id"] = True
            combined.metadata["multi_hop_provenance"]["original_doc_id"] = deepcopy(
                previous_provenance.get("original_doc_id", current_provenance.get("original_doc_id"))
            )
        merged[key] = combined


def _ordered_evidence(merged: Mapping[str, Document], *, top_k: int) -> list[Document]:
    ordered = sorted(
        merged.values(),
        key=lambda document: (document.score is not None, _ranking_score(document)),
        reverse=True,
    )
    return [_snapshot_document(document) for document in ordered[:top_k]]


def _ranking_score(document: Document) -> float:
    if document.score is None:
        return -math.inf
    try:
        score = float(document.score)
    except (TypeError, ValueError):
        return -math.inf
    return score if math.isfinite(score) else -math.inf


def _document_key(document: Document) -> str:
    if document.doc_id and document.doc_id.strip():
        return f"id:{document.doc_id}"
    return f"content:{_content_fingerprint(document.content)}"


def _unique_values(values: Sequence[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if value not in unique:
            unique.append(deepcopy(value))
    return unique


def _observation_summary(documents: Sequence[Document]) -> str:
    if not documents:
        return "no_documents"
    identifiers = [str(document.doc_id or _anonymous_document_id(document)) for document in documents]
    return f"documents={len(documents)}; evidence_ids={','.join(identifiers)}"


def _content_fingerprint(content: str) -> str:
    normalized = " ".join(content.split()).casefold()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=10).hexdigest()


def _anonymous_document_id(document: Document) -> str:
    return f"multi-hop-anonymous-{_content_fingerprint(document.content)}"


def _new_evidence_ids(latest: EvidenceHop, previous_hops: Sequence[EvidenceHop]) -> list[str]:
    previous = {
        _document_key(document)
        for hop in previous_hops
        for document in hop.documents
    }
    return [_document_key(document) for document in latest.documents if _document_key(document) not in previous]


def _salient_bridge_terms(documents: Sequence[Document], query: str, *, limit: int) -> list[str]:
    if limit == 0:
        return []
    blocked = {
        *re.findall(r"[\w-]{3,}", query.casefold(), flags=re.UNICODE),
        "avec",
        "dans",
        "des",
        "pour",
        "que",
        "qui",
        "sur",
        "the",
        "and",
        "for",
        "from",
        "this",
        "with",
    }
    counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for document in documents:
        for token in re.findall(r"[\w-]{3,}", document.content.casefold(), flags=re.UNICODE):
            if token in blocked:
                continue
            first_seen.setdefault(token, len(first_seen))
            counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts, key=lambda token: (-counts[token], first_seen[token], token))
    return ranked[:limit]


def _format_planning_observations(hops: Sequence[EvidenceHop]) -> str:
    if not hops:
        return "Aucune preuve observée."
    lines: list[str] = []
    for hop in hops[-4:]:
        lines.append(f"Étape {hop.step}; requête={hop.query}; observation={hop.observation}")
        for document in hop.documents[:2]:
            label = document.doc_id or "anonymous"
            preview = " ".join(document.content.split())[:240]
            lines.append(f"- [{label}] {preview}")
    return "\n".join(lines)
