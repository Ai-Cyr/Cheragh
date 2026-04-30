"""Parent-child retrieval architecture.

The retriever indexes small child chunks, retrieves the most relevant children,
and returns the larger parent sections/documents to the generator. This keeps
retrieval precise while giving the LLM enough context to answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Any

from ..base import BaseRetriever, Document, EmbeddingModel, HashingEmbedding
from ..ingestion import RecursiveTextChunker
from ..vectorstores import MemoryVectorStore


@dataclass
class ParentChildIndex:
    """In-memory parent/child index used by :class:`ParentChildRetriever`."""

    parents: dict[str, Document] = field(default_factory=dict)
    children: list[Document] = field(default_factory=list)

    def add_parent(self, parent: Document) -> str:
        parent_id = parent.doc_id or f"parent-{len(self.parents)}"
        parent.doc_id = parent_id
        self.parents[parent_id] = parent
        return parent_id

    def add_child(self, child: Document, parent_id: str) -> None:
        child.metadata = {**child.metadata, "parent_doc_id": parent_id}
        if child.doc_id is None:
            child.doc_id = f"{parent_id}#child-{len(self.children)}"
        self.children.append(child)

    def parent_for_child(self, child: Document) -> Document | None:
        parent_id = _parent_id(child)
        return self.parents.get(parent_id) if parent_id else None


class ParentChildRetriever(BaseRetriever):
    """Retrieve small child chunks and return their larger parent documents.

    Parameters
    ----------
    parent_documents:
        Documents returned to the caller after child retrieval.
    child_documents:
        Small chunks indexed for retrieval. Each child should have
        ``metadata['parent_doc_id']`` or ``metadata['parent_section_id']``. If
        omitted, child chunks are created from the parent documents.
    child_retriever:
        Optional existing retriever over child chunks. When omitted, a
        :class:`MemoryVectorStore` is built from ``child_documents``.
    """

    def __init__(
        self,
        parent_documents: Iterable[Document],
        child_documents: Iterable[Document] | None = None,
        embedding_model: EmbeddingModel | None = None,
        child_retriever: BaseRetriever | None = None,
        top_k_children: int = 12,
        top_k_parents: int = 4,
        child_chunk_size: int = 350,
        child_chunk_overlap: int = 60,
        include_child_matches: bool = True,
    ):
        if top_k_children <= 0:
            raise ValueError("top_k_children must be > 0")
        if top_k_parents <= 0:
            raise ValueError("top_k_parents must be > 0")
        self.top_k_children = top_k_children
        self.top_k_parents = top_k_parents
        self.include_child_matches = include_child_matches
        self.embedding_model = embedding_model or HashingEmbedding()

        self.index = ParentChildIndex()
        for parent in parent_documents:
            parent_id = parent.doc_id or f"parent-{len(self.index.parents)}"
            self.index.add_parent(Document(parent.content, metadata=dict(parent.metadata), doc_id=parent_id, score=parent.score))

        if child_documents is None:
            chunker = RecursiveTextChunker(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap, min_chunk_size=20)
            generated_children: list[Document] = []
            for parent_id, parent in self.index.parents.items():
                for child in chunker.split_documents([parent]):
                    child.metadata.update({"parent_doc_id": parent_id, "chunk_role": "child_chunk"})
                    generated_children.append(child)
            child_documents = generated_children

        for child in child_documents:
            child_copy = Document(child.content, metadata=dict(child.metadata), doc_id=child.doc_id, score=child.score)
            parent_id = _parent_id(child_copy)
            if parent_id is None or parent_id not in self.index.parents:
                # Treat orphan children as their own parent to avoid silently
                # dropping evidence.
                parent_id = child_copy.doc_id or f"orphan-parent-{len(self.index.parents)}"
                self.index.add_parent(Document(child_copy.content, metadata=dict(child_copy.metadata), doc_id=parent_id))
            self.index.add_child(child_copy, parent_id)

        if child_retriever is None:
            store = MemoryVectorStore(self.embedding_model)
            store.add_documents(self.index.children)
            self.child_retriever = store.as_retriever()
        else:
            self.child_retriever = child_retriever

    @classmethod
    def from_hierarchical_chunks(
        cls,
        chunks: Iterable[Document],
        embedding_model: EmbeddingModel | None = None,
        **kwargs: Any,
    ) -> "ParentChildRetriever":
        """Build from output produced by ``HierarchicalChunker``.

        Parent sections are detected with ``metadata['chunk_role'] ==
        'parent_section'`` and children with ``'child_chunk'``.
        """
        parents: dict[str, Document] = {}
        children: list[Document] = []
        fallback_parents: dict[str, list[str]] = {}
        for chunk in chunks:
            role = chunk.metadata.get("chunk_role")
            if role == "parent_section":
                parents[chunk.doc_id or f"parent-{len(parents)}"] = chunk
            elif role == "child_chunk":
                children.append(chunk)
                parent_id = _parent_id(chunk)
                if parent_id:
                    fallback_parents.setdefault(parent_id, []).append(chunk.content)
        for parent_id, texts in fallback_parents.items():
            if parent_id not in parents:
                parents[parent_id] = Document("\n\n".join(texts), metadata={"parent_doc_id": parent_id}, doc_id=parent_id)
        if not parents:
            parents = {chunk.doc_id or f"parent-{idx}": chunk for idx, chunk in enumerate(chunks)}
        return cls(parents.values(), child_documents=children or None, embedding_model=embedding_model, **kwargs)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        child_k = max(self.top_k_children, top_k)
        child_hits = self.child_retriever.retrieve(query, top_k=child_k)
        parent_scores: dict[str, float] = {}
        child_matches: dict[str, list[dict[str, Any]]] = {}

        for child in child_hits:
            parent_id = _parent_id(child)
            if parent_id is None:
                continue
            score = float(child.score or 0.0)
            parent_scores[parent_id] = max(parent_scores.get(parent_id, float("-inf")), score)
            child_matches.setdefault(parent_id, []).append(
                {
                    "child_doc_id": child.doc_id,
                    "score": child.score,
                    "preview": child.content[:240],
                    "metadata": dict(child.metadata),
                }
            )

        ordered_parent_ids = sorted(parent_scores, key=lambda pid: parent_scores[pid], reverse=True)
        limit = min(top_k, self.top_k_parents)
        results: list[Document] = []
        for parent_id in ordered_parent_ids[:limit]:
            parent = self.index.parents.get(parent_id)
            if parent is None:
                continue
            metadata = dict(parent.metadata)
            metadata.update(
                {
                    "retrieval_method": "parent_child",
                    "matched_child_count": len(child_matches.get(parent_id, [])),
                }
            )
            if self.include_child_matches:
                metadata["child_matches"] = child_matches.get(parent_id, [])
            results.append(Document(parent.content, metadata=metadata, doc_id=parent.doc_id, score=parent_scores[parent_id]))
        return results


def _parent_id(doc: Document) -> str | None:
    for key in ("parent_section_id", "parent_doc_id", "parent_id"):
        value = doc.metadata.get(key)
        if value:
            return str(value)
    if doc.doc_id and "#child-" in doc.doc_id:
        return doc.doc_id.split("#child-", 1)[0]
    return None
