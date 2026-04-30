"""RAPTOR-style hierarchical summarization RAG.

This implementation is intentionally lightweight and deterministic: documents
are grouped by embedding similarity when possible and by stable batches as a
fallback, summarized level by level, then all leaves and summaries are indexed
for retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from ..base import BaseRetriever, Document, EmbeddingModel, ExtractiveLLMClient, HashingEmbedding, LLMClient, cosine_similarity
from ..engine import RAGEngine, RAGResponse
from ..vectorstores import MemoryVectorStore


SUMMARY_PROMPT = """Résume les extraits suivants en conservant les faits vérifiables, entités, dates et relations.
Ce résumé servira de nœud hiérarchique pour un système RAG.

Extraits:
{context}

Résumé:"""


@dataclass
class RAPTORNode:
    """Node in a RAPTOR tree."""

    document: Document
    level: int
    child_ids: list[str] = field(default_factory=list)
    cluster_id: str | None = None

    def to_document(self) -> Document:
        metadata = dict(self.document.metadata)
        metadata.update({"raptor_level": self.level, "raptor_child_ids": self.child_ids})
        if self.cluster_id is not None:
            metadata["raptor_cluster_id"] = self.cluster_id
        return Document(self.document.content, metadata=metadata, doc_id=self.document.doc_id, score=self.document.score)


@dataclass
class RAPTORIndex:
    """In-memory RAPTOR tree index."""

    nodes: list[RAPTORNode] = field(default_factory=list)

    def documents(self) -> list[Document]:
        return [node.to_document() for node in self.nodes]

    def levels(self) -> dict[int, list[RAPTORNode]]:
        result: dict[int, list[RAPTORNode]] = {}
        for node in self.nodes:
            result.setdefault(node.level, []).append(node)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": {str(level): len(nodes) for level, nodes in self.levels().items()},
            "node_count": len(self.nodes),
        }


class RAPTORRetrieverV2(BaseRetriever):
    """Retriever over all RAPTOR leaves and summary nodes."""

    def __init__(self, index: RAPTORIndex, embedding_model: EmbeddingModel | None = None):
        self.index = index
        self.embedding_model = embedding_model or HashingEmbedding()
        self.store = MemoryVectorStore(self.embedding_model)
        self.store.add_documents(index.documents())
        self.retriever = self.store.as_retriever()

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        docs = self.retriever.retrieve(query, top_k=top_k)
        for doc in docs:
            doc.metadata = {**doc.metadata, "retrieval_method": "raptor"}
        return docs


class RAPTOREngine:
    """Hierarchical summarization RAG engine.

    Unlike the legacy ``RAPTORRetriever`` class, this engine avoids mandatory
    clustering dependencies and exposes an end-to-end ``ask`` API.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        embedding_model: EmbeddingModel | None = None,
        llm_client: LLMClient | None = None,
        levels: int = 2,
        branching_factor: int = 4,
        min_cluster_size: int = 2,
        top_k: int = 6,
        **engine_kwargs: Any,
    ):
        if levels < 0:
            raise ValueError("levels must be >= 0")
        if branching_factor <= 1:
            raise ValueError("branching_factor must be > 1")
        self.embedding_model = embedding_model or HashingEmbedding()
        self.llm_client = llm_client or ExtractiveLLMClient()
        self.levels = levels
        self.branching_factor = branching_factor
        self.min_cluster_size = min_cluster_size
        self.index = self.build_index(list(documents))
        self.retriever = RAPTORRetrieverV2(self.index, embedding_model=self.embedding_model)
        self.engine = RAGEngine(self.retriever, llm_client=self.llm_client, top_k=top_k, **engine_kwargs)

    @classmethod
    def from_documents(cls, documents: Iterable[Document], **kwargs: Any) -> "RAPTOREngine":
        return cls(documents, **kwargs)

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        response = self.engine.ask(query, top_k=top_k, **generate_kwargs)
        response.metadata.update({"architecture": "raptor", "raptor_index": self.index.to_dict()})
        return response

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.retriever.retrieve(query, top_k=top_k)

    def build_index(self, documents: list[Document]) -> RAPTORIndex:
        index = RAPTORIndex()
        current = []
        for idx, doc in enumerate(documents):
            doc_id = doc.doc_id or f"raptor-leaf-{idx}"
            leaf = Document(doc.content, metadata={**doc.metadata, "raptor_level": 0, "node_type": "leaf"}, doc_id=doc_id, score=doc.score)
            node = RAPTORNode(leaf, level=0, child_ids=[], cluster_id=f"L0-{idx}")
            index.nodes.append(node)
            current.append(node)

        for level in range(1, self.levels + 1):
            if len(current) < self.min_cluster_size:
                break
            groups = self._group_nodes(current)
            next_level: list[RAPTORNode] = []
            for cluster_idx, group in enumerate(groups):
                if len(group) < self.min_cluster_size:
                    continue
                summary = self._summarize_group(group)
                child_ids = [node.document.doc_id for node in group if node.document.doc_id]
                doc = Document(
                    summary,
                    metadata={"raptor_level": level, "node_type": "summary", "raptor_child_ids": child_ids},
                    doc_id=f"raptor::L{level}::C{cluster_idx}",
                )
                summary_node = RAPTORNode(doc, level=level, child_ids=child_ids, cluster_id=f"L{level}-{cluster_idx}")
                index.nodes.append(summary_node)
                next_level.append(summary_node)
            if not next_level:
                break
            current = next_level
        return index

    def _group_nodes(self, nodes: list[RAPTORNode]) -> list[list[RAPTORNode]]:
        if len(nodes) <= self.branching_factor:
            return [nodes]
        # Greedy similarity grouping. This avoids sklearn while still grouping
        # semantically close nodes when embeddings are meaningful.
        texts = [node.document.content for node in nodes]
        try:
            matrix = self.embedding_model.embed_documents(texts)
        except Exception:  # pragma: no cover - defensive fallback for custom embedders
            matrix = None
        unused = set(range(len(nodes)))
        groups: list[list[RAPTORNode]] = []
        while unused:
            seed = min(unused)
            unused.remove(seed)
            group_indices = [seed]
            if matrix is not None and unused:
                scores = cosine_similarity(matrix[seed], matrix[list(unused)])
                ranked_unused = [idx for _, idx in sorted(zip(scores, list(unused)), reverse=True)]
            else:
                ranked_unused = sorted(unused)
            for idx in ranked_unused[: self.branching_factor - 1]:
                if idx in unused:
                    unused.remove(idx)
                    group_indices.append(idx)
            groups.append([nodes[idx] for idx in group_indices])
        return groups

    def _summarize_group(self, group: list[RAPTORNode]) -> str:
        context = "\n\n---\n\n".join(
            f"[{node.document.doc_id or i}]\n{node.document.content}" for i, node in enumerate(group, start=1)
        )
        if len(context) > 9000:
            context = context[:9000] + "\n..."
        prompt = SUMMARY_PROMPT.format(context=context)
        return self.llm_client.generate(prompt).strip() or "\n".join(node.document.content[:400] for node in group)
