"""Semantic entity-seeded local GraphRAG with mixed, budgeted evidence.

Follows Microsoft's local-search dataflow: entity-description vector search,
graph relationships, associated text units and community reports. It does not
implement DRIFT's iterative question expansion or claim/covariate extraction.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, TYPE_CHECKING

from ..base import Document, EmbeddingModel, _snapshot_document, _validate_non_negative_int, _validate_top_k
from ..citations import validate_citations
from ..pipeline import AdvancedRAGPipeline
from ..schema import RAGResponse, Source
from ..security.access_control import AccessPolicy, Principal
from ..tracing import RAGTrace
from ..vectorstores import MemoryVectorStore
from .extraction import SemanticKnowledgeGraph
from .engine import _norm_entity, _triple_sort_key

if TYPE_CHECKING:
    from .engine import CommunityGraphRAGEngine


@dataclass(frozen=True)
class LocalGraphSearchConfig:
    max_entities: int = 8
    max_graph_entities: int = 32
    max_relationships: int = 16
    max_reports: int = 4
    graph_hops: int = 1
    max_input_tokens: int = 8000
    max_output_tokens: int = 1500
    min_entity_similarity: float = 0.0
    token_counter: Callable[[str], int] | None = None

    def __post_init__(self) -> None:
        for name in ("max_entities", "max_graph_entities", "max_relationships", "max_reports", "max_input_tokens", "max_output_tokens"):
            _validate_top_k(getattr(self, name), name=name)
        _validate_non_negative_int(self.graph_hops, name="graph_hops")
        if self.max_graph_entities < self.max_entities:
            raise ValueError("max_graph_entities must be at least max_entities")
        if (isinstance(self.min_entity_similarity, bool) or not isinstance(self.min_entity_similarity, (int, float))
                or not math.isfinite(self.min_entity_similarity) or not -1 <= self.min_entity_similarity <= 1):
            raise ValueError("min_entity_similarity must be finite and between -1 and 1")
        if self.token_counter is not None and not callable(self.token_counter):
            raise TypeError("token_counter must be callable")


class SemanticLocalSearch:
    def __init__(self, engine: CommunityGraphRAGEngine, embedding_model: EmbeddingModel,
                 config: LocalGraphSearchConfig | None = None):
        if config is not None and not isinstance(config, LocalGraphSearchConfig):
            raise TypeError("local_search_config must be LocalGraphSearchConfig")
        self.engine = engine
        self.config = config or LocalGraphSearchConfig()
        self.counter = self.config.token_counter or (lambda text: len(text.encode("utf-8")))
        graph = engine._graph
        self.entities: dict[str, Document] = {}
        for entity in graph.entities():
            key = _norm_entity(entity)
            if isinstance(graph, SemanticKnowledgeGraph) and key in graph.entity_records:
                record = graph.entity_records[key]
                content = f"{record.name} ({record.entity_type}): {record.description}"
                provenance = set(record.source_doc_ids)
            else:
                relations = [triple for triple in graph.triples
                             if key in {_norm_entity(triple.subject), _norm_entity(triple.object)}]
                content = entity + "\n" + "\n".join(
                    f"{triple.subject} {triple.relation} {triple.object}" for triple in relations
                )
                provenance = set(graph.entity_to_doc_ids.get(key, ()))
                provenance.update(triple.doc_id for triple in relations if triple.doc_id)
                provenance.update(source for triple in relations for source in triple.metadata.get("source_doc_ids", []))
            provenance.update(graph.entity_to_doc_ids.get(key, ()))
            self.entities[key] = Document(content, doc_id=key, metadata={"source_doc_ids": sorted(provenance)})
        self.store = MemoryVectorStore(embedding_model)
        self.store.add_documents(list(self.entities.values()))

    def _authorized(self, allowed_doc_ids: Iterable[str] | None,
                    principal: Principal | Mapping[str, Any] | None, access_policy: AccessPolicy | None) -> set[str]:
        allowed = set(self.engine._documents_by_id)
        if allowed_doc_ids is not None:
            if isinstance(allowed_doc_ids, (str, bytes)):
                raise TypeError("allowed_doc_ids must be an iterable of source IDs")
            requested = list(allowed_doc_ids)
            if any(not isinstance(item, str) or not item for item in requested):
                raise ValueError("allowed_doc_ids must contain non-empty strings")
            allowed.intersection_update(requested)
        if principal is not None or access_policy is not None:
            policy = access_policy if access_policy is not None else AccessPolicy()
            allowed.intersection_update(
                document.doc_id for document in self.engine.documents if policy.authorize(document, principal).allowed
            )
        return allowed

    def _evidence(self, query: str, top_k: int, allowed: set[str]) -> tuple[list[Document], list[list[Document]]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        top_k = _validate_top_k(top_k)
        graph = self.engine._graph
        eligible = {key for key, doc in self.entities.items()
                    if doc.metadata["source_doc_ids"] and set(doc.metadata["source_doc_ids"]) <= allowed}
        matches = self.store.as_retriever().retrieve(query, top_k=max(1, len(self.entities)))
        seeds = [doc for doc in matches if doc.doc_id in eligible and
                 doc.score is not None and doc.score > self.config.min_entity_similarity][:self.config.max_entities]
        scores = {doc.doc_id: float(doc.score) for doc in seeds}
        reached = set(scores)
        frontier = set(reached)
        valid_triples = [triple for triple in graph.triples
                         if triple.doc_id in allowed
                         and set(triple.metadata.get("source_doc_ids", [triple.doc_id])) <= allowed
                         and _norm_entity(triple.subject) in eligible and _norm_entity(triple.object) in eligible]
        for _ in range(self.config.graph_hops):
            candidates: dict[str, float] = defaultdict(float)
            for triple in valid_triples:
                source, target = _norm_entity(triple.subject), _norm_entity(triple.object)
                if source in frontier:
                    candidates[target] += float(triple.metadata.get("weight", 1))
                if target in frontier:
                    candidates[source] += float(triple.metadata.get("weight", 1))
            frontier = set(sorted((item for item in candidates if item not in reached),
                                  key=lambda item: (-candidates[item], item))[
                :max(0, self.config.max_graph_entities - len(reached))
            ])
            reached.update(frontier)
        relationships = [triple for triple in valid_triples
                         if _norm_entity(triple.subject) in reached and _norm_entity(triple.object) in reached]
        relationships.sort(key=lambda triple: (
            -int(_norm_entity(triple.subject) in scores) - int(_norm_entity(triple.object) in scores),
            -float(triple.metadata.get("weight", 1)), _triple_sort_key(triple),
        ))
        relationships = relationships[:self.config.max_relationships]
        doc_scores: dict[str, float] = defaultdict(float)
        for entity in reached:
            for doc_id in graph.entity_to_doc_ids.get(entity, ()):
                if doc_id in allowed:
                    doc_scores[doc_id] += scores.get(entity, 0.25)
        for triple in relationships:
            if triple.doc_id:
                doc_scores[triple.doc_id] += 0.1 * float(triple.metadata.get("weight", 1))
        sources = []
        for doc_id in sorted(doc_scores, key=lambda item: (-doc_scores[item], item))[:top_k]:
            document = _snapshot_document(self.engine._documents_by_id[doc_id])
            document.score = doc_scores[doc_id]
            document.metadata.update({"retrieval_method": "community_graph_semantic_local",
                                      "matched_entities": sorted(scores), "source_doc_ids": [doc_id]})
            sources.append(document)
        entity_documents = [self._record("entity", key, self.entities[key].content,
                                         self.entities[key].metadata["source_doc_ids"])
                            for key in sorted(reached, key=lambda key: (-scores.get(key, 0), key))]
        relationship_documents = [self._record(
            "relationship", f"{triple.subject}|{triple.object}|{triple.doc_id}|{triple.relation}",
            f"{triple.subject} -> {triple.object}: {triple.relation}",
            triple.metadata.get("source_doc_ids", [triple.doc_id]),
        ) for triple in relationships]
        reports = [report for report in self.engine._reports
                   if report.metadata.get("provenance_complete") is True and report.doc_ids
                   and set(report.doc_ids) <= allowed
                   and any(_norm_entity(entity) in reached for entity in report.entities)]
        reports.sort(key=lambda report: (
            -sum(_norm_entity(entity) in scores for entity in report.entities),
            -report.metadata.get("level", 0), report.community_id,
        ))
        report_documents = [self.engine._report_document(report, 0.0, ())
                            for report in reports[:self.config.max_reports]]
        return sources, [sources, entity_documents, relationship_documents, report_documents]

    @staticmethod
    def _record(kind: str, key: str, content: str, source_ids: Iterable[str]) -> Document:
        identity = hashlib.sha256(key.encode()).hexdigest()[:24]
        return Document(content, doc_id=f"graph-{kind}:{identity}", metadata={
            "retrieval_method": f"community_graph_{kind}", "source_doc_ids": sorted(set(source_ids)),
        })

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self._evidence(query, top_k, set(self.engine._documents_by_id))[0]

    def ask(self, query: str, *, top_k: int = 5, allowed_doc_ids: Iterable[str] | None = None,
            principal: Principal | Mapping[str, Any] | None = None, access_policy: AccessPolicy | None = None,
            **generate_kwargs: Any) -> RAGResponse:
        allowed = self._authorized(allowed_doc_ids, principal, access_policy)
        sources, groups = self._evidence(query, top_k, allowed)

        def prompt(documents: list[Document]) -> str:
            return self.engine.answer_prompt.format(context=AdvancedRAGPipeline._format_context(documents), query=query)

        def count(text: str) -> int:
            result = _validate_non_negative_int(self.counter(text), name="token_counter result")
            if text and result == 0:
                raise ValueError("token_counter must be positive for non-empty text")
            return result

        if count(prompt([])) >= self.config.max_input_tokens:
            raise ValueError("local max_input_tokens cannot fit the question and instructions")
        # Round-robin preserves mixed evidence instead of spending the entire
        # window on entity descriptions before reaching text units/relations.
        documents: list[Document] = []
        excluded = 0
        for index in range(max(map(len, groups), default=0)):
            for group in groups:
                if index >= len(group):
                    continue
                document = group[index]
                if count(prompt([*documents, document])) <= self.config.max_input_tokens:
                    documents.append(document)
                else:
                    excluded += 1
        warnings = ["local_context_budget_excluded_records"] if excluded else []
        answer = "Je ne sais pas : les sources autorisées ne fournissent pas de preuves suffisantes."
        context_prompt = prompt(documents)
        if documents:
            if any(key in generate_kwargs for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")):
                raise ValueError("set the output budget in LocalGraphSearchConfig")
            generated = self.engine.llm_client.generate(
                context_prompt, max_tokens=self.config.max_output_tokens, **generate_kwargs,
            )
            if not isinstance(generated, str) or not generated.strip():
                raise ValueError("local GraphRAG LLM must return a non-empty string")
            if count(generated) > self.config.max_output_tokens:
                raise ValueError("local GraphRAG response exceeds max_output_tokens")
            validation = validate_citations(generated, documents, require_citations=True)
            if validation.ok:
                answer = generated
            else:
                warnings.append("invalid_local_citations_answer_withheld")
                documents = []
        else:
            warnings.append("no_authorized_local_evidence")
        validation = validate_citations(answer, documents, require_citations=bool(documents))
        trace = RAGTrace(query=query) if self.engine.trace_enabled else None
        if trace:
            trace.add_retrieval(query, documents)
            trace.record_generation(prompt=context_prompt, answer=answer,
                                    model=getattr(self.engine.llm_client, "model", None))
            trace.warnings.extend(warnings)
            trace.finish(architecture="community_graph_rag_semantic_local", context_record_count=len(documents))
        return RAGResponse(
            query=query, answer=answer, sources=[Source.from_document(doc) for doc in documents],
            retrieved_documents=documents, prompt=context_prompt, citations=validation.citations,
            warnings=warnings, grounded_score=validation.grounded_score, citation_validation=validation,
            trace=trace,
            metadata={"architecture": "community_graph_rag_semantic_local", "mode": "local",
                      "candidate_source_document_count": len(sources),
                      "source_document_count": sum(doc.doc_id in self.engine._documents_by_id for doc in documents),
                      "context_record_count": len(documents),
                      "context_tokens": count(context_prompt), "excluded_context_records": excluded,
                      "access_control_enabled": allowed_doc_ids is not None or principal is not None or access_policy is not None,
                      "limitations": ["citation_ids_do_not_prove_entailment", "no_claim_covariates_or_drift"]},
        )
