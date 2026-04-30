"""Lightweight Graph RAG.

This module builds a small in-memory knowledge graph from documents using a
rule-based extractor, then combines graph-neighbour evidence with lexical/vector
retrieval for generation. It is deliberately dependency-free and can be replaced
with an LLM/entity extractor later without changing the public engine API.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import re
from typing import Any, Iterable

from ..base import BaseRetriever, Document, EmbeddingModel, ExtractiveLLMClient, HashingEmbedding, LLMClient
from ..engine import RAGEngine, RAGResponse
from ..vectorstores import MemoryVectorStore


@dataclass(frozen=True)
class KnowledgeTriple:
    """A relation extracted from source documents."""

    subject: str
    relation: str
    object: str
    doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "doc_id": self.doc_id,
            "metadata": self.metadata,
        }


@dataclass
class KnowledgeGraph:
    """Small in-memory graph store."""

    triples: list[KnowledgeTriple] = field(default_factory=list)
    entity_to_doc_ids: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    adjacency: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    def add_triple(self, triple: KnowledgeTriple) -> None:
        if triple not in self.triples:
            self.triples.append(triple)
        s = _norm_entity(triple.subject)
        o = _norm_entity(triple.object)
        self.adjacency[s].add(o)
        self.adjacency[o].add(s)
        if triple.doc_id:
            self.entity_to_doc_ids[s].add(triple.doc_id)
            self.entity_to_doc_ids[o].add(triple.doc_id)

    def entities(self) -> list[str]:
        return sorted(set(self.adjacency) | set(self.entity_to_doc_ids))

    def triples_for_entities(self, entities: Iterable[str], depth: int = 1) -> list[KnowledgeTriple]:
        seeds = {_norm_entity(entity) for entity in entities if entity}
        if not seeds:
            return []
        reachable = set(seeds)
        queue = deque((seed, 0) for seed in seeds)
        while queue:
            entity, level = queue.popleft()
            if level >= depth:
                continue
            for neighbor in self.adjacency.get(entity, set()):
                if neighbor not in reachable:
                    reachable.add(neighbor)
                    queue.append((neighbor, level + 1))
        return [triple for triple in self.triples if _norm_entity(triple.subject) in reachable or _norm_entity(triple.object) in reachable]

    def doc_ids_for_entities(self, entities: Iterable[str], depth: int = 1) -> set[str]:
        triples = self.triples_for_entities(entities, depth=depth)
        doc_ids = {triple.doc_id for triple in triples if triple.doc_id}
        for entity in entities:
            doc_ids.update(self.entity_to_doc_ids.get(_norm_entity(entity), set()))
        return doc_ids

    def to_dict(self) -> dict[str, Any]:
        return {"triples": [triple.to_dict() for triple in self.triples], "entities": self.entities()}


class GraphRAGRetriever(BaseRetriever):
    """Retriever that blends graph-neighbour evidence and a fallback retriever."""

    def __init__(
        self,
        documents: Iterable[Document],
        graph: KnowledgeGraph,
        fallback_retriever: BaseRetriever | None = None,
        graph_depth: int = 1,
        graph_boost: float = 0.15,
    ):
        self.documents = {doc.doc_id or f"doc-{idx}": doc for idx, doc in enumerate(documents)}
        for doc_id, doc in list(self.documents.items()):
            doc.doc_id = doc.doc_id or doc_id
        self.graph = graph
        self.fallback_retriever = fallback_retriever
        self.graph_depth = graph_depth
        self.graph_boost = graph_boost

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        entities = _extract_entities(query)
        graph_doc_ids = self.graph.doc_ids_for_entities(entities, depth=self.graph_depth)
        merged: dict[str, Document] = {}

        for doc_id in graph_doc_ids:
            source = self.documents.get(doc_id)
            if source is None:
                continue
            merged[doc_id] = Document(
                source.content,
                metadata={**source.metadata, "retrieval_method": "graph", "matched_entities": entities},
                doc_id=source.doc_id,
                score=(source.score or 0.0) + self.graph_boost,
            )

        if self.fallback_retriever is not None:
            for doc in self.fallback_retriever.retrieve(query, top_k=max(top_k, top_k * 2)):
                key = doc.doc_id or doc.content
                previous = merged.get(key)
                if previous is None or (doc.score or 0.0) > (previous.score or 0.0):
                    metadata = {**doc.metadata, "retrieval_method": metadata_method(doc.metadata, "fallback")}
                    merged[key] = Document(doc.content, metadata=metadata, doc_id=doc.doc_id, score=doc.score)

        ordered = sorted(merged.values(), key=lambda doc: (doc.score is not None, doc.score or 0.0), reverse=True)
        return ordered[:top_k]


def metadata_method(metadata: dict[str, Any], fallback: str) -> str:
    value = metadata.get("retrieval_method")
    if value:
        return f"graph+{value}" if value != "graph" else "graph"
    return f"graph+{fallback}"


class GraphRAGEngine:
    """Graph-enhanced RAG engine.

    The engine extracts lightweight triples from documents, retrieves graph
    neighbours for entities mentioned in the query, and uses a regular
    :class:`RAGEngine` for grounded generation.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        embedding_model: EmbeddingModel | None = None,
        llm_client: LLMClient | None = None,
        graph: KnowledgeGraph | None = None,
        retriever: BaseRetriever | None = None,
        graph_depth: int = 1,
        top_k: int = 5,
        **engine_kwargs: Any,
    ):
        self.documents = [Document(doc.content, metadata=dict(doc.metadata), doc_id=doc.doc_id, score=doc.score) for doc in documents]
        self.embedding_model = embedding_model or HashingEmbedding()
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.graph = graph or build_knowledge_graph(self.documents)
        
        if retriever is None:
            store = MemoryVectorStore(self.embedding_model)
            store.add_documents(self.documents)
            fallback = store.as_retriever()
        else:
            fallback = retriever
        self.retriever = GraphRAGRetriever(self.documents, self.graph, fallback_retriever=fallback, graph_depth=graph_depth)
        self.engine = RAGEngine(self.retriever, llm_client=self.llm_client, top_k=top_k, **engine_kwargs)

    @classmethod
    def from_documents(cls, documents: Iterable[Document], **kwargs: Any) -> "GraphRAGEngine":
        return cls(documents, **kwargs)

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        response = self.engine.ask(query, top_k=top_k, **generate_kwargs)
        entities = _extract_entities(query)
        response.metadata.update(
            {
                "architecture": "graph_rag",
                "query_entities": entities,
                "graph_triples": [triple.to_dict() for triple in self.graph.triples_for_entities(entities, depth=1)[:20]],
            }
        )
        return response

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.retriever.retrieve(query, top_k=top_k)


def build_knowledge_graph(documents: Iterable[Document]) -> KnowledgeGraph:
    graph = KnowledgeGraph()
    for idx, doc in enumerate(documents):
        doc_id = doc.doc_id or f"doc-{idx}"
        doc.doc_id = doc_id
        for triple in extract_triples(doc.content, doc_id=doc_id, metadata=doc.metadata):
            graph.add_triple(triple)
        # Ensure entity-to-document mapping exists even without explicit triples.
        for entity in _extract_entities(doc.content):
            graph.entity_to_doc_ids[_norm_entity(entity)].add(doc_id)
    return graph


def extract_triples(text: str, doc_id: str | None = None, metadata: dict[str, Any] | None = None) -> list[KnowledgeTriple]:
    triples: list[KnowledgeTriple] = []
    meta = dict(metadata or {})
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    relation_patterns = [
        (r"(?P<s>[A-ZÀ-ÖØ-Þ][\wÀ-ÿ&' -]{1,60})\s+(?:est|sont|is|are)\s+(?P<rel>[^.]{2,40}?)(?:\s+(?:de|du|des|pour|for|of)\s+)(?P<o>[A-ZÀ-ÖØ-Þ][\wÀ-ÿ&' -]{1,60})", "est lié à"),
        (r"(?P<s>[A-ZÀ-ÖØ-Þ][\wÀ-ÿ&' -]{1,60})\s+(?:travaille avec|collabore avec|works with|partners with)\s+(?P<o>[A-ZÀ-ÖØ-Þ][\wÀ-ÿ&' -]{1,60})", "collabore avec"),
        (r"(?P<s>[A-ZÀ-ÖØ-Þ][\wÀ-ÿ&' -]{1,60})\s+(?:contient|inclut|includes|contains)\s+(?P<o>[A-ZÀ-ÖØ-Þ][\wÀ-ÿ&' -]{1,60})", "contient"),
    ]
    for sentence in sentences:
        for pattern, default_relation in relation_patterns:
            for match in re.finditer(pattern, sentence):
                subject = _clean_entity(match.group("s"))
                obj = _clean_entity(match.group("o"))
                rel = _clean_relation(match.groupdict().get("rel") or default_relation)
                if subject and obj and _norm_entity(subject) != _norm_entity(obj):
                    triples.append(KnowledgeTriple(subject=subject, relation=rel, object=obj, doc_id=doc_id, metadata=meta))
        entities = _extract_entities(sentence)
        if len(entities) >= 2:
            head = entities[0]
            for tail in entities[1:4]:
                if _norm_entity(head) != _norm_entity(tail):
                    triples.append(KnowledgeTriple(subject=head, relation="co-mention", object=tail, doc_id=doc_id, metadata=meta))
    # De-duplicate preserving order.
    deduped: list[KnowledgeTriple] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for triple in triples:
        key = (_norm_entity(triple.subject), triple.relation.lower(), _norm_entity(triple.object), triple.doc_id)
        if key not in seen:
            deduped.append(triple)
            seen.add(key)
    return deduped


def _extract_entities(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-ZÀ-ÖØ-Þ][\wÀ-ÿ&']*(?:[ -][A-ZÀ-ÖØ-Þ0-9][\wÀ-ÿ&']*){0,4}\b", text)
    stop = {"Le", "La", "Les", "Un", "Une", "Des", "The", "This", "That", "Question", "Contexte", "Résumé", "Resume"}
    entities: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        entity = _clean_entity(candidate)
        key = _norm_entity(entity)
        if not entity or entity in stop or len(entity) < 2 or key in seen:
            continue
        entities.append(entity)
        seen.add(key)
    return entities[:20]


def _clean_entity(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" ,;:.!?()[]{}\"'"))
    # Trim trailing predicate fragments from greedy regexes.
    value = re.sub(r"\s+(est|sont|is|are|contient|inclut|includes|contains)\b.*$", "", value, flags=re.I)
    return value[:80]


def _clean_relation(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip(" ,;:.!?()[]{}\"'"))
    return value[:80] or "lié à"


def _norm_entity(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
