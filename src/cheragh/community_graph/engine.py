"""Community GraphRAG with a dependency-light baseline and optional paper methods.

The design follows the high-level workflow in *From Local to Global: A Graph
RAG Approach to Query-Focused Summarization* (Edge et al., arXiv:2404.16130):
partition an entity graph, pre-generate a report for every community, and use
those reports for global questions.  Local search instead starts from entities
named in the query and expands to their communities and source documents.

The default remains a deterministic single-level baseline. Inject
``LeidenCommunityDetector`` and ``LLMCommunitySummarizer`` for hierarchical
indexing, and call ``ask_global_map_reduce`` for query-focused map/reduce over
all reports at a selected hierarchy frontier. Graph extraction is still the
rule-based default unless a domain-appropriate graph is supplied. This is not
a reproduction of the paper's evaluation or the Microsoft GraphRAG product.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from ..base import (
    Document,
    ExtractiveLLMClient,
    LLMClient,
    _snapshot_document,
    _snapshot_documents,
    _validate_top_k,
)
from ..citations import validate_citations
from ..graph.engine import KnowledgeGraph, KnowledgeTriple, build_knowledge_graph
from ..pipeline import AdvancedRAGPipeline, DEFAULT_ANSWER_PROMPT_FR
from ..schema import RAGResponse, Source
from ..tracing import RAGTrace


_WORD_RE = re.compile(r"[\wÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)
_STOPWORDS = {
    "a",
    "au",
    "aux",
    "and",
    "are",
    "avec",
    "de",
    "des",
    "du",
    "en",
    "est",
    "et",
    "for",
    "in",
    "is",
    "la",
    "le",
    "les",
    "of",
    "on",
    "ou",
    "pour",
    "que",
    "qui",
    "sur",
    "the",
    "un",
    "une",
}


@dataclass(frozen=True)
class Community:
    """One mutually exclusive group of graph entities.

    ``entities``, ``triples`` and ``doc_ids`` are tuples so membership cannot
    be changed accidentally.  Accessors on :class:`CommunityGraphRAGEngine`
    still return fresh snapshots because ``KnowledgeTriple.metadata`` is a
    mutable mapping for backward compatibility.
    """

    community_id: int
    entities: tuple[str, ...]
    triples: tuple[KnowledgeTriple, ...]
    doc_ids: tuple[str, ...]
    level: int = 0
    parent_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_id": self.community_id,
            "entities": list(self.entities),
            "triples": [_snapshot_triple(triple).to_dict() for triple in self.triples],
            "doc_ids": list(self.doc_ids),
            "level": self.level,
            "parent_id": self.parent_id,
        }


@dataclass(frozen=True)
class CommunityReport:
    """Pre-generated summary and provenance for one community."""

    community_id: int
    title: str
    summary: str
    entities: tuple[str, ...]
    doc_ids: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def report_id(self) -> str:
        return f"community:{self.community_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "community_id": self.community_id,
            "title": self.title,
            "summary": self.summary,
            "entities": list(self.entities),
            "doc_ids": list(self.doc_ids),
            "metadata": deepcopy(self.metadata),
        }


@runtime_checkable
class CommunitySummarizer(Protocol):
    """Protocol for an injectable community report summarizer."""

    def summarize(self, community: Community, documents: Sequence[Document]) -> str:
        """Return report text for ``community`` using source snapshots."""


class DeterministicCommunitySummarizer:
    """Dependency-free fallback that renders entities and graph relations."""

    def summarize(self, community: Community, documents: Sequence[Document]) -> str:
        entities = ", ".join(community.entities) or "aucune entité nommée"
        relations = []
        for triple in sorted(community.triples, key=_triple_sort_key)[:12]:
            relations.append(f"{triple.subject} — {triple.relation} → {triple.object}")
        relation_text = "; ".join(relations) or "aucune relation explicite"
        excerpts = [re.sub(r"\s+", " ", document.content).strip()[:240] for document in documents[:4]]
        excerpt_text = " | ".join(excerpt for excerpt in excerpts if excerpt) or "aucun extrait disponible"
        source_label = "document source" if len(documents) == 1 else "documents source"
        return (
            f"La communauté {community.community_id} regroupe {entities}. "
            f"Relations principales : {relation_text}. "
            f"Extraits représentatifs : {excerpt_text}. "
            f"Provenance : {len(documents)} {source_label}."
        )


SummarizerCallable = Callable[[Community, Sequence[Document]], str]


def detect_communities(
    graph: KnowledgeGraph,
    *,
    resolution: float = 1.0,
    max_iterations: int = 50,
) -> list[Community]:
    """Partition ``graph`` with deterministic modularity local moving.

    Repeated triples increase an undirected edge's weight, matching the paper's
    aggregation of duplicate relations into weighted edges.  Node order and all
    tie breaks are lexical, so the same graph yields the same partition across
    runs and regardless of triple insertion order.  This is a single-level
    Louvain-like baseline; it does not claim Leiden's connectivity guarantees or
    build a hierarchy.
    """

    if isinstance(resolution, bool) or not isinstance(resolution, (int, float)):
        raise TypeError("resolution must be a number")
    if not math.isfinite(float(resolution)) or resolution <= 0:
        raise ValueError("resolution must be > 0 and finite")
    max_iterations = _validate_top_k(max_iterations, name="max_iterations")

    graph_snapshot = _snapshot_graph(graph)
    nodes, weights = _weighted_adjacency(graph_snapshot)
    if not nodes:
        return []

    node_to_group = _local_moving_partition(
        nodes,
        weights,
        resolution=float(resolution),
        max_iterations=max_iterations,
    )
    grouped: dict[int, list[str]] = defaultdict(list)
    for node in nodes:
        grouped[node_to_group[node]].append(node)

    labels = _entity_labels(graph_snapshot)
    ordered_groups = sorted((tuple(sorted(members)) for members in grouped.values()), key=lambda members: members)
    communities: list[Community] = []
    for community_id, members in enumerate(ordered_groups):
        member_set = set(members)
        triples = tuple(
            _snapshot_triple(triple)
            for triple in sorted(graph_snapshot.triples, key=_triple_sort_key)
            if _norm_entity(triple.subject) in member_set or _norm_entity(triple.object) in member_set
        )
        doc_ids: set[str] = set()
        for entity in members:
            doc_ids.update(graph_snapshot.entity_to_doc_ids.get(entity, set()))
        doc_ids.update(triple.doc_id for triple in triples if triple.doc_id)
        communities.append(
            Community(
                community_id=community_id,
                entities=tuple(labels.get(entity, entity) for entity in members),
                triples=triples,
                doc_ids=tuple(sorted(doc_ids)),
            )
        )
    return communities


class CommunityGraphRAGEngine:
    """Single-level Community GraphRAG with global and local search.

    Parameters
    ----------
    documents:
        Source documents.  They are defensively copied and missing identifiers
        are assigned inside the index without mutating caller-owned objects.
    graph:
        Optional prebuilt ``KnowledgeGraph``.  Supplying a richer extractor is
        recommended; the default graph builder is only a rule-based baseline.
    summarizer:
        Object implementing ``summarize(community, documents)`` or a callable
        with the same arguments.  The deterministic fallback performs no model
        call.
    llm_client:
        Generator for final answers.  It defaults to Cheragh's deterministic
        extractive fallback.
    top_k:
        Strict maximum number of report documents (global) or source documents
        (local) included in an answer.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        *,
        graph: KnowledgeGraph | None = None,
        summarizer: CommunitySummarizer | SummarizerCallable | None = None,
        llm_client: LLMClient | None = None,
        top_k: int = 5,
        resolution: float = 1.0,
        max_iterations: int = 50,
        answer_prompt: str = DEFAULT_ANSWER_PROMPT_FR,
        require_citations: bool = False,
        trace_enabled: bool = True,
        community_detector: Callable[[KnowledgeGraph], Sequence[Community]] | None = None,
    ):
        validated_top_k = _validate_top_k(top_k)
        snapshots = _snapshot_documents(documents)
        explicit_ids: list[str] = []
        for document in snapshots:
            if document.doc_id is None:
                continue
            if not isinstance(document.doc_id, str):
                raise TypeError("document ids must be strings or None")
            if document.doc_id.strip():
                explicit_ids.append(document.doc_id)
        if len(explicit_ids) != len(set(explicit_ids)):
            duplicate = sorted(doc_id for doc_id, count in Counter(explicit_ids).items() if count > 1)[0]
            raise ValueError(f"duplicate document id: {duplicate}")
        used_ids = set(explicit_ids)
        self._documents_by_id: dict[str, Document] = {}
        for index, document in enumerate(snapshots):
            doc_id = document.doc_id
            if doc_id is None or not doc_id.strip():
                candidate = f"doc-{index}"
                suffix = 1
                while candidate in used_ids:
                    candidate = f"doc-{index}-{suffix}"
                    suffix += 1
                doc_id = candidate
                used_ids.add(doc_id)
            document.doc_id = doc_id
            self._documents_by_id[doc_id] = document

        source_graph = graph if graph is not None else build_knowledge_graph(_snapshot_documents(snapshots))
        self._baseline_partition = community_detector is None
        self._rule_based_graph = graph is None
        self._architecture = "community_graph_rag_baseline" if self._baseline_partition else "community_graph_rag"
        self._graph = _snapshot_graph(source_graph)
        self._communities = tuple(
            community_detector(_snapshot_graph(self._graph)) if community_detector is not None else
            detect_communities(
                self._graph,
                resolution=resolution,
                max_iterations=max_iterations,
            )
        )
        from .paper import validate_hierarchy

        validate_hierarchy(self._communities, self._graph.entities())
        self._communities = tuple(_snapshot_community(community) for community in self._communities)
        self._communities_by_id = {community.community_id: community for community in self._communities}
        self._summarizer = summarizer or DeterministicCommunitySummarizer()
        self._reports_by_id: dict[int, CommunityReport] = {}
        for community in sorted(self._communities, key=lambda item: (-item.level, item.community_id)):
            self._reports_by_id[community.community_id] = self._build_report(community)
        self._reports = tuple(self._reports_by_id[community.community_id] for community in self._communities)
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.top_k = validated_top_k
        self.answer_prompt = answer_prompt
        self.require_citations = bool(require_citations)
        self.trace_enabled = bool(trace_enabled)

        self._entity_to_community: dict[str, int] = {}
        for community in sorted(self._communities, key=lambda item: (item.level, item.community_id)):
            for entity in community.entities:
                self._entity_to_community[_norm_entity(entity)] = community.community_id

    @classmethod
    def from_documents(cls, documents: Iterable[Document], **kwargs: Any) -> "CommunityGraphRAGEngine":
        return cls(documents, **kwargs)

    @property
    def documents(self) -> list[Document]:
        """Return source snapshots; caller mutation cannot alter the index."""

        return _snapshot_documents(self._documents_by_id.values())

    @property
    def graph(self) -> KnowledgeGraph:
        """Return a defensive graph snapshot."""

        return _snapshot_graph(self._graph)

    @property
    def communities(self) -> list[Community]:
        """Return defensive community snapshots."""

        return [_snapshot_community(community) for community in self._communities]

    @property
    def reports(self) -> list[CommunityReport]:
        """Return defensive community-report snapshots."""

        return [_snapshot_report(report) for report in self._reports]

    def global_search(self, query: str, top_k: int = 5) -> list[Document]:
        """Rank community reports lexically and return at most ``top_k``.

        Returned document ids use ``community:<id>``.  Their metadata contains a
        complete ``source_doc_ids`` provenance chain to the original corpus.
        """

        top_k = _validate_top_k(top_k)
        ranked = self._rank_reports(query)
        return [self._report_document(report, score, matched_terms) for report, score, matched_terms in ranked[:top_k]]

    def local_search(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve source documents through matched entities and communities."""

        top_k = _validate_top_k(top_k)
        entity_scores = self._match_entities(query)
        report_scores = {report.community_id: score for report, score, _ in self._rank_reports(query)}
        doc_scores: dict[str, float] = defaultdict(float)
        doc_entities: dict[str, set[str]] = defaultdict(set)
        doc_communities: dict[str, set[int]] = defaultdict(set)

        if entity_scores:
            for entity, entity_score in entity_scores.items():
                community_id = self._entity_to_community[entity]
                community = self._communities_by_id[community_id]
                direct_ids = self._graph.entity_to_doc_ids.get(entity, set())
                for doc_id in community.doc_ids:
                    if doc_id not in self._documents_by_id:
                        continue
                    direct_boost = 1.0 if doc_id in direct_ids else 0.35
                    doc_scores[doc_id] += entity_score * direct_boost
                    doc_scores[doc_id] += report_scores.get(community_id, 0.0) * 0.2
                    doc_entities[doc_id].add(_display_entity(self._graph, entity))
                    doc_communities[doc_id].add(community_id)
        else:
            # Entity-free questions still get deterministic community evidence,
            # using the same report scores as global search.
            for report, report_score, _ in self._rank_reports(query):
                base_score = report_score if report_score > 0 else 1.0 / (1 + report.community_id)
                for doc_id in report.doc_ids:
                    if doc_id not in self._documents_by_id:
                        continue
                    doc_scores[doc_id] = max(doc_scores[doc_id], base_score * 0.5)
                    doc_communities[doc_id].add(report.community_id)

        ordered_ids = sorted(doc_scores, key=lambda doc_id: (-doc_scores[doc_id], doc_id))[:top_k]
        results: list[Document] = []
        for doc_id in ordered_ids:
            document = _snapshot_document(self._documents_by_id[doc_id])
            document.score = float(doc_scores[doc_id])
            document.metadata.update(
                {
                    "retrieval_method": "community_graph_local",
                    "community_ids": sorted(doc_communities[doc_id]),
                    "matched_entities": sorted(doc_entities[doc_id], key=str.casefold),
                    "source_doc_id": doc_id,
                }
            )
            results.append(document)
        return results

    def search(self, query: str, top_k: int = 5, *, mode: str = "global") -> list[Document]:
        """Search report documents globally or source documents locally."""

        normalized_mode = _validate_mode(mode)
        if normalized_mode == "global":
            return self.global_search(query, top_k=top_k)
        return self.local_search(query, top_k=top_k)

    def retrieve(self, query: str, top_k: int = 5, *, mode: str = "global") -> list[Document]:
        """Retriever-compatible alias for :meth:`search`."""

        return self.search(query, top_k=top_k, mode=mode)

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        *,
        mode: str = "global",
        **generate_kwargs: Any,
    ) -> RAGResponse:
        """Answer from ranked reports (global) or entity evidence (local)."""

        normalized_mode = _validate_mode(mode)
        effective_top_k = self.top_k if top_k is None else _validate_top_k(top_k)
        documents = self.search(query, top_k=effective_top_k, mode=normalized_mode)
        trace = RAGTrace(query=query) if self.trace_enabled else None
        if trace:
            retrieval_step = trace.start_step(
                "community_graph_retrieval",
                mode=normalized_mode,
                top_k=effective_top_k,
            )
            trace.add_retrieval(query, documents)
            retrieval_step.finish(document_count=len(documents))

        context = AdvancedRAGPipeline._format_context(documents)
        prompt = self.answer_prompt.format(context=context, query=query)
        if trace:
            trace.prompt = prompt
            generation_step = trace.start_step("community_graph_generation", document_count=len(documents))
        else:
            generation_step = None
        answer = self.llm_client.generate(prompt, **generate_kwargs)
        if generation_step:
            generation_step.finish(answer_chars=len(answer))
        if trace:
            trace.record_generation(prompt=prompt, answer=answer, model=getattr(self.llm_client, "model", None))

        validation = validate_citations(answer, documents, require_citations=self.require_citations)
        if trace:
            trace.warnings.extend(validation.warnings)
            trace.finish(
                answer_chars=len(answer),
                prompt_chars=len(prompt),
                architecture=self._architecture,
                mode=normalized_mode,
            )
        selected_communities = sorted(
            {
                community_id
                for document in documents
                for community_id in _document_community_ids(document)
            }
        )
        response_documents = _snapshot_documents(documents)
        return RAGResponse(
            query=query,
            answer=answer,
            sources=[Source.from_document(document) for document in response_documents],
            retrieved_documents=response_documents,
            prompt=prompt,
            metadata={
                "architecture": self._architecture,
                "mode": normalized_mode,
                "top_k": effective_top_k,
                "community_count": len(self._communities),
                "selected_communities": selected_communities,
                "limitations": [
                    *(["single_level_deterministic_partition_not_hierarchical_leiden"]
                      if self._baseline_partition else []),
                    "lexical_report_scoring_not_llm_map_reduce",
                    *(["rule_based_graph_when_no_graph_is_injected"] if self._rule_based_graph else []),
                ],
            },
            citations=validation.citations,
            warnings=list(validation.warnings),
            grounded_score=validation.grounded_score,
            unsourced_claims=list(validation.unsourced_claims),
            citation_validation=validation,
            trace=trace,
        )

    def ask_global(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        """Convenience wrapper for global report search and generation."""

        return self.ask(query, top_k=top_k, mode="global", **generate_kwargs)

    def ask_local(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        """Convenience wrapper for local entity/community search."""

        return self.ask(query, top_k=top_k, mode="local", **generate_kwargs)

    def ask_global_map_reduce(self, query: str, **kwargs: Any) -> RAGResponse:
        """Answer from all authorized reports using bounded, scored map/reduce.

        See :func:`cheragh.community_graph.global_map_reduce` for budgets,
        hierarchy selection and fail-closed source authorization. Existing
        ``ask_global`` keeps its lexical top-k behavior for compatibility.
        """
        from .paper import global_map_reduce

        return global_map_reduce(self, query, **kwargs)

    def _build_report(self, community: Community) -> CommunityReport:
        children = [
            _snapshot_report(self._reports_by_id[child.community_id])
            for child in self._communities if child.parent_id == community.community_id
        ]
        # Recover source dependencies from the graph and descendants as well
        # as the detector's declaration. A detector omitting a source ID must
        # never turn a mixed report into apparently public evidence.
        source_ids = set(community.doc_ids)
        source_ids.update(triple.doc_id for triple in community.triples if triple.doc_id)
        source_ids.update(doc_id for entity in community.entities
                          for doc_id in self._graph.entity_to_doc_ids.get(_norm_entity(entity), ()))
        source_ids.update(doc_id for child in children for doc_id in child.doc_ids)
        source_documents = [
            _snapshot_document(self._documents_by_id[doc_id])
            for doc_id in sorted(source_ids)
            if doc_id in self._documents_by_id
        ]
        public_community = _snapshot_community(community)
        public_documents = _snapshot_documents(source_documents)
        summarizer = self._summarizer
        hierarchical_summarize = getattr(summarizer, "summarize_hierarchy", None)
        if callable(hierarchical_summarize):
            summary = hierarchical_summarize(public_community, public_documents, children)
        elif isinstance(summarizer, CommunitySummarizer):
            summary = summarizer.summarize(public_community, public_documents)
        elif callable(summarizer):
            summary = summarizer(public_community, public_documents)
        else:  # defensive error for dynamically typed callers
            raise TypeError("summarizer must be callable or implement summarize(community, documents)")
        if not isinstance(summary, str):
            raise TypeError("community summarizer must return str")
        summary = summary.strip()
        if not summary:
            raise ValueError("community summarizer returned an empty report")

        title_entities = community.entities[:3]
        title = " / ".join(title_entities) if title_entities else f"Community {community.community_id}"
        provenance = [
            {
                "doc_id": document.doc_id,
                "metadata": deepcopy(document.metadata),
                "preview": document.content[:240],
            }
            for document in source_documents
        ]
        return CommunityReport(
            community_id=community.community_id,
            title=title,
            summary=summary,
            entities=tuple(community.entities),
            doc_ids=tuple(sorted(source_ids)),
            metadata={
                "source_doc_ids": sorted(source_ids),
                "provenance": provenance,
                "triple_count": len(community.triples),
                "baseline": self._baseline_partition and isinstance(self._summarizer, DeterministicCommunitySummarizer),
                "level": community.level,
                "parent_id": community.parent_id,
                "provenance_complete": bool(source_ids)
                and all(doc_id in self._documents_by_id for doc_id in source_ids)
                and all(triple.doc_id in self._documents_by_id for triple in community.triples)
                and all(self._graph.entity_to_doc_ids.get(_norm_entity(entity)) for entity in community.entities)
                and all(child.metadata.get("provenance_complete") is True for child in children),
            },
        )

    def _rank_reports(self, query: str) -> list[tuple[CommunityReport, float, tuple[str, ...]]]:
        query_terms = _tokens(query)
        report_terms = {
            report.community_id: _tokens(f"{report.title} {report.summary} {' '.join(report.entities)}")
            for report in self._reports
        }
        document_frequency = Counter(
            term
            for terms in report_terms.values()
            for term in set(terms)
        )
        report_count = max(1, len(self._reports))
        ranked: list[tuple[CommunityReport, float, tuple[str, ...]]] = []
        for report in self._reports:
            counts = Counter(report_terms[report.community_id])
            matched = tuple(sorted(set(query_terms) & set(counts)))
            if query_terms:
                weighted_overlap = 0.0
                total_query_weight = 0.0
                for term in set(query_terms):
                    inverse_frequency = math.log((report_count + 1) / (document_frequency.get(term, 0) + 1)) + 1.0
                    total_query_weight += inverse_frequency
                    if counts.get(term):
                        weighted_overlap += inverse_frequency * counts[term] / (counts[term] + 0.5)
                lexical_score = weighted_overlap / total_query_weight if total_query_weight else 0.0
            else:
                lexical_score = 0.0
            entity_bonus = max(
                (
                    0.35
                    for entity in report.entities
                    if _contains_phrase(query, entity)
                ),
                default=0.0,
            )
            score = lexical_score + entity_bonus
            ranked.append((_snapshot_report(report), float(score), matched))
        return sorted(ranked, key=lambda item: (-item[1], item[0].community_id))

    def _report_document(
        self,
        report: CommunityReport,
        score: float,
        matched_terms: tuple[str, ...],
    ) -> Document:
        metadata = deepcopy(report.metadata)
        metadata.update(
            {
                "architecture": self._architecture,
                "retrieval_method": "community_report",
                "community_id": report.community_id,
                "community_ids": [report.community_id],
                "entities": list(report.entities),
                "source_doc_ids": list(report.doc_ids),
                "matched_terms": list(matched_terms),
                "report_title": report.title,
            }
        )
        return Document(
            content=f"{report.title}\n{report.summary}",
            metadata=metadata,
            doc_id=report.report_id,
            score=score,
        )

    def _match_entities(self, query: str) -> dict[str, float]:
        query_terms = set(_tokens(query))
        matched: dict[str, float] = {}
        for entity in sorted(self._entity_to_community):
            entity_terms = set(_tokens(entity))
            exact = _contains_phrase(query, entity)
            overlap = len(query_terms & entity_terms) / len(entity_terms) if entity_terms else 0.0
            if exact or overlap > 0:
                matched[entity] = (1.0 if exact else 0.0) + overlap
        return matched


def _weighted_adjacency(graph: KnowledgeGraph) -> tuple[list[str], dict[str, dict[str, float]]]:
    nodes = set(graph.entities())
    weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for triple in sorted(graph.triples, key=_triple_sort_key):
        subject = _norm_entity(triple.subject)
        obj = _norm_entity(triple.object)
        if not subject or not obj or subject == obj:
            continue
        nodes.update((subject, obj))
        weights[subject][obj] += 1.0
        weights[obj][subject] += 1.0
    # Preserve caller-supplied graph edges that have no corresponding triple.
    for subject in sorted(graph.adjacency):
        normalized_subject = _norm_entity(subject)
        nodes.add(normalized_subject)
        for obj in sorted(graph.adjacency[subject]):
            normalized_obj = _norm_entity(obj)
            if not normalized_obj or normalized_obj == normalized_subject:
                continue
            nodes.add(normalized_obj)
            if normalized_obj not in weights[normalized_subject]:
                weights[normalized_subject][normalized_obj] = 1.0
                weights[normalized_obj][normalized_subject] = 1.0
    ordered_nodes = sorted(node for node in nodes if node)
    return ordered_nodes, {node: dict(weights.get(node, {})) for node in ordered_nodes}


def _local_moving_partition(
    nodes: list[str],
    weights: dict[str, dict[str, float]],
    *,
    resolution: float,
    max_iterations: int,
) -> dict[str, int]:
    membership = {node: index for index, node in enumerate(nodes)}
    degree = {node: sum(weights[node].values()) for node in nodes}
    total_weight_twice = sum(degree.values())
    if total_weight_twice <= 0:
        return membership
    community_degree = {membership[node]: degree[node] for node in nodes}

    for _ in range(max_iterations):
        moved = False
        for node in nodes:
            node_degree = degree[node]
            if node_degree <= 0:
                continue
            current = membership[node]
            weights_by_community: dict[int, float] = defaultdict(float)
            for neighbor, edge_weight in weights[node].items():
                weights_by_community[membership[neighbor]] += edge_weight

            community_degree[current] -= node_degree
            candidates = set(weights_by_community)
            candidates.add(current)
            best = current
            best_gain = weights_by_community.get(current, 0.0) - (
                resolution * node_degree * community_degree.get(current, 0.0) / total_weight_twice
            )
            for candidate in sorted(candidates):
                gain = weights_by_community.get(candidate, 0.0) - (
                    resolution * node_degree * community_degree.get(candidate, 0.0) / total_weight_twice
                )
                if gain > best_gain + 1e-12 or (abs(gain - best_gain) <= 1e-12 and candidate < best):
                    best = candidate
                    best_gain = gain
            membership[node] = best
            community_degree[best] = community_degree.get(best, 0.0) + node_degree
            if best != current:
                moved = True
        if not moved:
            break
    return membership


def _snapshot_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    snapshot = KnowledgeGraph()
    for triple in sorted(graph.triples, key=_triple_sort_key):
        snapshot.add_triple(_snapshot_triple(triple))
    for entity, doc_ids in graph.entity_to_doc_ids.items():
        snapshot.entity_to_doc_ids[_norm_entity(entity)].update(str(doc_id) for doc_id in doc_ids)
    for entity, neighbors in graph.adjacency.items():
        normalized_entity = _norm_entity(entity)
        snapshot.adjacency[normalized_entity].update(_norm_entity(neighbor) for neighbor in neighbors if neighbor)
    return snapshot


def _snapshot_triple(triple: KnowledgeTriple) -> KnowledgeTriple:
    return KnowledgeTriple(
        subject=triple.subject,
        relation=triple.relation,
        object=triple.object,
        doc_id=triple.doc_id,
        metadata=deepcopy(triple.metadata or {}),
    )


def _snapshot_community(community: Community) -> Community:
    return Community(
        community_id=community.community_id,
        entities=tuple(community.entities),
        triples=tuple(_snapshot_triple(triple) for triple in community.triples),
        doc_ids=tuple(community.doc_ids),
        level=community.level,
        parent_id=community.parent_id,
    )


def _snapshot_report(report: CommunityReport) -> CommunityReport:
    return CommunityReport(
        community_id=report.community_id,
        title=report.title,
        summary=report.summary,
        entities=tuple(report.entities),
        doc_ids=tuple(report.doc_ids),
        metadata=deepcopy(report.metadata),
    )


def _entity_labels(graph: KnowledgeGraph) -> dict[str, str]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for triple in graph.triples:
        candidates[_norm_entity(triple.subject)].add(triple.subject.strip())
        candidates[_norm_entity(triple.object)].add(triple.object.strip())
    for entity in graph.entities():
        candidates[_norm_entity(entity)].add(entity.strip())
    return {
        normalized: sorted(labels, key=lambda label: (label.casefold(), label))[0]
        for normalized, labels in candidates.items()
        if labels
    }


def _display_entity(graph: KnowledgeGraph, normalized_entity: str) -> str:
    return _entity_labels(graph).get(normalized_entity, normalized_entity)


def _triple_sort_key(triple: KnowledgeTriple) -> tuple[str, str, str, str, str, str, str]:
    return (
        _norm_entity(triple.subject),
        triple.relation.casefold(),
        triple.relation,
        _norm_entity(triple.object),
        triple.doc_id or "",
        triple.subject,
        triple.object,
    )


def _tokens(text: str) -> list[str]:
    return [token for token in (_normalize_text(match) for match in _WORD_RE.findall(text)) if token not in _STOPWORDS]


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_phrase = _normalize_text(phrase)
    if not normalized_phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text) is not None


def _norm_entity(value: str) -> str:
    return _normalize_text(value)


def _validate_mode(mode: str) -> str:
    if not isinstance(mode, str):
        raise TypeError("mode must be a string")
    normalized = mode.strip().lower()
    if normalized not in {"global", "local"}:
        raise ValueError("mode must be 'global' or 'local'")
    return normalized


def _document_community_ids(document: Document) -> list[int]:
    value = document.metadata.get("community_ids", [])
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if isinstance(item, int) and not isinstance(item, bool)]
    return []
