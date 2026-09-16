"""Parent-child retrieval architecture.

The retriever indexes small child chunks, retrieves the most relevant children,
and returns the larger parent sections/documents to the generator. This keeps
retrieval precise while giving the LLM enough context to answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Iterable, Any

from ..base import BaseRetriever, Document, EmbeddingModel, HashingEmbedding, _snapshot_document, _validate_top_k
from ..ingestion import RecursiveTextChunker
from ..vectorstores import MemoryVectorStore
from ..security.access_control import AccessPolicy, Principal


@dataclass
class ParentChildIndex:
    """In-memory parent/child index used by :class:`ParentChildRetriever`."""

    parents: dict[str, Document] = field(default_factory=dict)
    children: list[Document] = field(default_factory=list)

    def add_parent(self, parent: Document) -> str:
        parent = _snapshot_document(parent)
        parent_id = parent.doc_id or f"parent-{len(self.parents)}"
        if parent_id in self.parents:
            raise ValueError(f"duplicate parent document id: {parent_id}")
        parent.doc_id = parent_id
        self.parents[parent_id] = parent
        return parent_id

    def add_child(self, child: Document, parent_id: str) -> None:
        child = _snapshot_document(child)
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

    Parent expansion rechecks policies exposed by child-retriever wrappers.
    Applications may also supply ``principal``/``policy`` here. Missing parent
    sections are reconstructed only when the children's source metadata agree.
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
        max_candidate_children: int = 10_000,
        *,
        principal: Principal | Mapping[str, Any] | None = None,
        policy: AccessPolicy | None = None,
    ):
        self.top_k_children = _validate_top_k(top_k_children, name="top_k_children")
        self.top_k_parents = _validate_top_k(top_k_parents, name="top_k_parents")
        self.max_candidate_children = _validate_top_k(max_candidate_children, name="max_candidate_children")
        self.include_child_matches = include_child_matches
        self.embedding_model = embedding_model or HashingEmbedding()
        self.principal = principal
        self.policy = policy if policy is not None else (AccessPolicy() if principal is not None else None)

        self.index = ParentChildIndex()
        for parent in parent_documents:
            parent_id = parent.doc_id or f"parent-{len(self.index.parents)}"
            self.index.add_parent(Document(parent.content, metadata=deepcopy(parent.metadata), doc_id=parent_id, score=parent.score))

        if child_documents is None:
            chunker = RecursiveTextChunker(chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap, min_chunk_size=20)
            generated_children: list[Document] = []
            for parent_id, parent in self.index.parents.items():
                for child in chunker.split_documents([parent]):
                    child.metadata.update({"parent_doc_id": parent_id, "chunk_role": "child_chunk"})
                    generated_children.append(child)
            child_documents = generated_children

        for child in child_documents:
            child_copy = _snapshot_document(child)
            child_parent_id = _parent_id(child_copy)
            if child_parent_id is None or child_parent_id not in self.index.parents:
                # Treat orphan children as their own parent to avoid silently
                # dropping evidence.
                child_parent_id = child_copy.doc_id or f"orphan-parent-{len(self.index.parents)}"
                self.index.add_parent(Document(child_copy.content, metadata=dict(child_copy.metadata), doc_id=child_parent_id))
            self.index.add_child(child_copy, child_parent_id)

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
        chunks = list(chunks)
        parents: dict[str, Document] = {}
        children: list[Document] = []
        fallback_parents: dict[str, list[Document]] = {}
        for chunk in chunks:
            role = chunk.metadata.get("chunk_role")
            if role == "parent_section":
                parent_key = chunk.doc_id or f"parent-{len(parents)}"
                if parent_key in parents:
                    raise ValueError(f"duplicate parent section id: {parent_key}")
                parents[parent_key] = chunk
            elif role == "child_chunk":
                children.append(chunk)
                parent_id = _parent_id(chunk)
                if parent_id:
                    fallback_parents.setdefault(parent_id, []).append(chunk)
        for parent_id, siblings in fallback_parents.items():
            if parent_id not in parents:
                metadata = _reconstructed_parent_metadata(siblings)
                metadata.update({"parent_doc_id": parent_id, "chunk_role": "parent_section",
                                 "reconstructed_from_children": [child.doc_id for child in siblings]})
                parents[parent_id] = Document("\n\n".join(child.content for child in siblings), metadata=metadata, doc_id=parent_id)
        if not parents:
            parents = {chunk.doc_id or f"parent-{idx}": chunk for idx, chunk in enumerate(chunks)}
        return cls(parents.values(), child_documents=children or None, embedding_model=embedding_model, **kwargs)

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        top_k = _validate_top_k(top_k)
        parents = [_snapshot_document(parent) for parent in self.index.parents.values()]
        if self.policy is not None:
            parents = self.policy.filter_documents(parents, self.principal)
        for policy, principal in _child_access_policies(self.child_retriever):
            parents = policy.filter_documents(parents, principal)
        allowed_parents = {parent.doc_id: parent for parent in parents if parent.doc_id in self.index.parents}
        if not allowed_parents:
            return []
        limit = min(top_k, self.top_k_parents)
        child_k = min(max(self.top_k_children, top_k), self.max_candidate_children)
        while True:
            child_hits = list(self.child_retriever.retrieve(query, top_k=child_k))[:child_k]
            eligible_parents = {_parent_id(child) for child in child_hits if _parent_id(child) in allowed_parents}
            if len(eligible_parents) >= limit or len(child_hits) < child_k or child_k >= self.max_candidate_children:
                break
            child_k = min(child_k * 2, self.max_candidate_children)
        parent_scores: dict[str, float] = {}
        child_matches: dict[str, list[dict[str, Any]]] = {}

        for child in child_hits:
            parent_id = _parent_id(child)
            if parent_id is None or parent_id not in allowed_parents:
                continue
            score = float(child.score or 0.0)
            parent_scores[parent_id] = max(parent_scores.get(parent_id, float("-inf")), score)
            child_matches.setdefault(parent_id, []).append(
                {
                    "child_doc_id": child.doc_id,
                    "score": child.score,
                    "preview": child.content[:240],
                    "metadata": deepcopy(child.metadata),
                }
            )

        ordered_parent_ids = sorted(parent_scores, key=lambda pid: parent_scores[pid], reverse=True)
        results: list[Document] = []
        for parent_id in ordered_parent_ids[:limit]:
            parent = allowed_parents.get(parent_id)
            if parent is None:
                continue
            metadata = deepcopy(parent.metadata)
            metadata.update(
                {
                    "retrieval_method": "parent_child",
                    "matched_child_count": len(child_matches.get(parent_id, [])),
                    "parent_candidate_limit_reached": child_k >= self.max_candidate_children and len(eligible_parents) < limit,
                }
            )
            if self.include_child_matches:
                metadata["child_matches"] = child_matches.get(parent_id, [])
            results.append(Document(parent.content, metadata=metadata, doc_id=parent.doc_id, score=parent_scores[parent_id]))
        return results


def _reconstructed_parent_metadata(children: list[Document]) -> dict[str, Any]:
    structural = {"chunk_index", "chunk_role", "source_char_start", "source_char_end",
                  "normalized_char_start", "normalized_char_end", "token_start", "token_end"}
    metadata = {key: deepcopy(value) for key, value in children[0].metadata.items() if key not in structural}
    for child in children[1:]:
        other = {key: value for key, value in child.metadata.items() if key not in structural}
        if other != metadata:
            raise ValueError("cannot reconstruct a parent from children with conflicting source metadata; supply an explicit parent")
    return metadata


def _child_access_policies(retriever: Any) -> Iterable[tuple[AccessPolicy, Any]]:
    """Reapply every exposed access policy after crossing the child/parent boundary."""
    pending = [retriever]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        policy = getattr(current, "policy", None)
        if policy is not None and callable(getattr(policy, "filter_documents", None)):
            yield policy, getattr(current, "principal", None)
        for name in ("retriever", "base_retriever", "child_retriever"):
            nested = getattr(current, name, None)
            if nested is not None:
                pending.append(nested)
        nested_many = getattr(current, "retrievers", ())
        if isinstance(nested_many, Mapping):
            pending.extend(nested_many.values())
        elif isinstance(nested_many, Sequence) and not isinstance(nested_many, (str, bytes)):
            pending.extend(nested_many)


def _parent_id(doc: Document) -> str | None:
    for key in ("parent_section_id", "parent_doc_id", "parent_id"):
        value = doc.metadata.get(key)
        if value:
            return str(value)
    if doc.doc_id and "#child-" in doc.doc_id:
        return doc.doc_id.split("#child-", 1)[0]
    return None
