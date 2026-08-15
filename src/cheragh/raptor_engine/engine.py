"""RAPTOR-style hierarchical summarization and retrieval.

The index construction remains intentionally lightweight and dependency-free.
Retrieval supports both strategies evaluated in the RAPTOR paper: a collapsed
(``flat``) tree search for backwards compatibility and a budgeted, top-down
beam traversal that follows relevant summaries towards their leaves.

This module is a practical baseline, not a reproduction of RAPTOR's learned
Gaussian-mixture clustering or model-specific summarization pipeline.
"""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Iterable

from ..base import (
    BaseRetriever,
    Document,
    EmbeddingModel,
    ExtractiveLLMClient,
    HashingEmbedding,
    LLMClient,
    _snapshot_document,
    _validate_non_negative_int,
    _validate_top_k,
    cosine_similarity,
)
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

    def __post_init__(self) -> None:
        _validate_document(self.document, name="RAPTOR node document")
        self.level = _validate_non_negative_int(self.level, name="RAPTOR node level")
        if isinstance(self.child_ids, (str, bytes)) or not isinstance(
            self.child_ids,
            Sequence,
        ):
            raise TypeError("RAPTOR child_ids must be a sequence of strings")
        children: list[str] = []
        for child_id in self.child_ids:
            if not isinstance(child_id, str):
                raise TypeError("RAPTOR child_ids must contain only strings")
            if not child_id.strip():
                raise ValueError("RAPTOR child_ids must contain non-empty strings")
            if child_id in children:
                raise ValueError(f"RAPTOR child_ids must be unique: {child_id!r}")
            children.append(child_id)
        if self.cluster_id is not None:
            if not isinstance(self.cluster_id, str):
                raise TypeError("RAPTOR cluster_id must be a string or None")
            if not self.cluster_id.strip():
                raise ValueError("RAPTOR cluster_id must be non-empty when provided")
        self.child_ids = children

    def to_document(self) -> Document:
        """Return a detached document snapshot suitable for indexing/output."""

        metadata = deepcopy(self.document.metadata)
        metadata.update({"raptor_level": self.level, "raptor_child_ids": list(self.child_ids)})
        if self.cluster_id is not None:
            metadata["raptor_cluster_id"] = self.cluster_id
        return Document(
            self.document.content,
            metadata=metadata,
            doc_id=self.document.doc_id,
            score=self.document.score,
        )

    def snapshot(self) -> "RAPTORNode":
        """Return a defensive copy of this node and its mutable fields."""

        return RAPTORNode(
            document=_snapshot_document(self.document),
            level=self.level,
            child_ids=list(self.child_ids),
            cluster_id=self.cluster_id,
        )


@dataclass
class RAPTORIndex:
    """In-memory RAPTOR tree index."""

    nodes: list[RAPTORNode] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.nodes, (str, bytes)) or not isinstance(self.nodes, Sequence):
            raise TypeError("RAPTOR index nodes must be a sequence of RAPTORNode values")
        if any(not isinstance(node, RAPTORNode) for node in self.nodes):
            raise TypeError("RAPTOR index nodes must contain only RAPTORNode values")
        self.nodes = list(self.nodes)

    def documents(self) -> list[Document]:
        return [node.to_document() for node in self.nodes]

    def snapshot(self) -> "RAPTORIndex":
        """Return an index snapshot isolated from caller-owned nodes."""

        return RAPTORIndex(nodes=[node.snapshot() for node in self.nodes])

    def levels(self) -> dict[int, list[RAPTORNode]]:
        result: dict[int, list[RAPTORNode]] = {}
        for node in self.nodes:
            result.setdefault(node.level, []).append(node)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "levels": {
                str(level): len(nodes)
                for level, nodes in sorted(self.levels().items())
            },
            "node_count": len(self.nodes),
        }


class RAPTORRetrieverV2(BaseRetriever):
    """Retrieve from a RAPTOR tree using collapsed or top-down search.

    Args:
        index: Tree/forest to snapshot and search.
        embedding_model: Shared encoder for summaries and leaf documents.
        retrieval_mode: ``"flat"`` (the legacy collapsed-tree behaviour) or
            ``"tree"`` for top-down beam traversal. Paper terminology aliases
            such as ``"collapsed_tree"`` and ``"tree_traversal"`` are accepted.
        beam_width: Maximum number of nodes selected at each traversal depth.
        traversal_budget: Maximum number of unique candidate nodes scored per
            query. Selected/visited nodes are necessarily bounded by this cap.

    The tree traversal ranks candidates by the mean similarity along their
    root-to-node path. This deterministic path score keeps ancestor relevance
    in the decision instead of treating every leaf as an unrelated flat item.
    """

    _MODE_ALIASES = {
        "flat": "flat",
        "collapsed": "flat",
        "collapsed-tree": "flat",
        "collapsed_tree": "flat",
        "tree": "tree",
        "beam": "tree",
        "traversal": "tree",
        "tree-traversal": "tree",
        "tree_traversal": "tree",
    }

    def __init__(
        self,
        index: RAPTORIndex,
        embedding_model: EmbeddingModel | None = None,
        retrieval_mode: str = "flat",
        beam_width: int = 4,
        traversal_budget: int = 64,
    ):
        if not isinstance(index, RAPTORIndex):
            raise TypeError("index must be a RAPTORIndex")
        self.index = index.snapshot()
        self.embedding_model = embedding_model or HashingEmbedding()
        if not callable(getattr(self.embedding_model, "embed_documents", None)) or not callable(
            getattr(self.embedding_model, "embed_query", None)
        ):
            raise TypeError("embedding_model must define embed_documents() and embed_query()")
        self.retrieval_mode = self._normalize_mode(retrieval_mode)
        self.beam_width = _validate_top_k(beam_width, name="beam_width")
        self.traversal_budget = _validate_top_k(
            traversal_budget,
            name="traversal_budget",
        )
        self._nodes_by_id: dict[str, RAPTORNode] = {}
        self._node_positions: dict[str, int] = {}
        for position, node in enumerate(self.index.nodes):
            node_id = node.document.doc_id
            if not node_id:
                node_id = f"raptor-node-{position}"
                node.document.doc_id = node_id
            if node_id in self._nodes_by_id:
                raise ValueError(f"RAPTOR node IDs must be unique: {node_id!r}")
            self._nodes_by_id[node_id] = node
            self._node_positions[node_id] = position

        self._children: dict[str, tuple[str, ...]] = {}
        parents: dict[str, list[str]] = {node_id: [] for node_id in self._nodes_by_id}
        for node_id, node in self._nodes_by_id.items():
            children = tuple(node.child_ids)
            for child_id in children:
                if child_id == node_id:
                    raise ValueError(f"RAPTOR node {node_id!r} cannot be its own child")
                if child_id not in self._nodes_by_id:
                    raise ValueError(
                        f"RAPTOR node {node_id!r} references unknown child {child_id!r}"
                    )
                child = self._nodes_by_id[child_id]
                if child.level >= node.level:
                    raise ValueError(
                        "RAPTOR child levels must be strictly lower than their parent "
                        f"({node_id!r} level {node.level}, {child_id!r} level {child.level})"
                    )
            self._children[node_id] = children
            for child_id in children:
                parents[child_id].append(node_id)
        self._parents = {
            node_id: tuple(sorted(parent_ids, key=self._root_sort_key))
            for node_id, parent_ids in parents.items()
        }
        roots = [node_id for node_id, parent_ids in self._parents.items() if not parent_ids]
        self._root_ids = tuple(sorted(roots, key=self._root_sort_key))

        self.store = MemoryVectorStore(self.embedding_model)
        self.store.add_documents(self.index.documents())
        self.retriever = self.store.as_retriever()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        retrieval_mode: str | None = None,
        beam_width: int | None = None,
        traversal_budget: int | None = None,
    ) -> list[Document]:
        query = _validate_query(query)
        top_k = _validate_top_k(top_k)
        mode = self.retrieval_mode if retrieval_mode is None else self._normalize_mode(retrieval_mode)
        effective_beam_width = (
            self.beam_width
            if beam_width is None
            else _validate_top_k(beam_width, name="beam_width")
        )
        effective_budget = (
            self.traversal_budget
            if traversal_budget is None
            else _validate_top_k(traversal_budget, name="traversal_budget")
        )
        if not self._nodes_by_id:
            return []
        if mode == "flat":
            return self._retrieve_flat(query, top_k=top_k)
        return self._retrieve_tree(
            query,
            top_k=top_k,
            beam_width=effective_beam_width,
            traversal_budget=effective_budget,
        )

    def _retrieve_flat(self, query: str, *, top_k: int) -> list[Document]:
        # Ask the vector store for every node before applying our stable
        # tie-break. NumPy's reverse argsort otherwise makes equal-score order
        # depend on insertion position.
        docs = self.retriever.retrieve(query, top_k=len(self._nodes_by_id))
        docs.sort(key=lambda doc: (-self._finite_score(doc.score), doc.doc_id or ""))
        results: list[Document] = []
        for doc in docs[:top_k]:
            node_id = doc.doc_id or ""
            path = self._canonical_path(node_id)
            output = _snapshot_document(doc)
            output.metadata.update(
                {
                    "retrieval_method": "raptor",
                    "raptor_retrieval_mode": "flat",
                    "raptor_path": list(path),
                    "raptor_path_levels": [
                        self._nodes_by_id[path_id].level for path_id in path
                    ],
                }
            )
            results.append(output)
        return results

    def _retrieve_tree(
        self,
        query: str,
        *,
        top_k: int,
        beam_width: int,
        traversal_budget: int,
    ) -> list[Document]:
        query_vector = self.embedding_model.embed_query(query)
        embeddings = self.store.embeddings
        if embeddings is None:
            return []
        scores: dict[str, float] = {}

        # Candidate = (node id, ancestor ids, ancestor levels, ancestor scores).
        frontier: list[tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]]] = [
            (node_id, (), (), ()) for node_id in self._root_ids
        ]
        terminal: list[tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]]] = []
        deepest: list[tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]]] = []
        visited: set[str] = set()

        while frontier:
            # Score only roots or children reached by the traversal. This is a
            # genuine hierarchical search rather than a flat all-node search
            # followed by hierarchical filtering.
            unscored_ids = sorted(
                {candidate[0] for candidate in frontier if candidate[0] not in scores}
            )
            unscored_ids = unscored_ids[: traversal_budget - len(scores)]
            if unscored_ids:
                positions = [self._node_positions[node_id] for node_id in unscored_ids]
                candidate_scores = cosine_similarity(query_vector, embeddings[positions])
                scores.update(
                    {
                        node_id: self._finite_score(candidate_scores[position])
                        for position, node_id in enumerate(unscored_ids)
                    }
                )
            candidates = self._materialize_candidates(
                [candidate for candidate in frontier if candidate[0] in scores],
                scores=scores,
            )
            candidates = self._deduplicate_candidates(candidates)
            candidates.sort(key=self._candidate_sort_key)
            selected = [
                candidate
                for candidate in candidates
                if candidate[0] not in visited
            ][:beam_width]
            if not selected:
                break
            deepest = selected
            next_frontier: list[
                tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]]
            ] = []
            for node_id, path, levels, path_scores in selected:
                visited.add(node_id)
                children = [
                    child_id
                    for child_id in self._children.get(node_id, ())
                    if child_id not in visited and child_id not in path
                ]
                if not children:
                    terminal.append((node_id, path, levels, path_scores))
                    continue
                for child_id in children:
                    next_frontier.append((child_id, path, levels, path_scores))
            frontier = next_frontier

        # Keep completed leaves and the deepest scored frontier. The latter is
        # the deterministic fallback when the score cap stops another branch.
        result_candidates = terminal + deepest
        result_candidates = self._deduplicate_candidates(result_candidates)
        result_candidates.sort(key=self._candidate_sort_key)
        terminal_ids = {candidate[0] for candidate in terminal}
        results: list[Document] = []
        for node_id, path, levels, path_scores in result_candidates[:top_k]:
            node = self._nodes_by_id[node_id]
            output = node.to_document()
            output.score = self._path_score(path_scores)
            output.metadata.update(
                {
                    "retrieval_method": "raptor",
                    "raptor_retrieval_mode": "tree",
                    "raptor_path": list(path),
                    "raptor_path_levels": list(levels),
                    "raptor_path_scores": list(path_scores),
                    "raptor_node_score": scores[node_id],
                    "raptor_terminal": node_id in terminal_ids,
                    "raptor_visited_nodes": len(visited),
                    "raptor_scored_nodes": len(scores),
                    "raptor_beam_width": beam_width,
                    "raptor_traversal_budget": traversal_budget,
                }
            )
            results.append(output)
        return results

    def _materialize_candidates(
        self,
        frontier: list[tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]]],
        *,
        scores: dict[str, float],
    ) -> list[tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]]]:
        materialized = []
        for node_id, parent_path, parent_levels, parent_scores in frontier:
            node = self._nodes_by_id[node_id]
            materialized.append(
                (
                    node_id,
                    parent_path + (node_id,),
                    parent_levels + (node.level,),
                    parent_scores + (scores[node_id],),
                )
            )
        return materialized

    def _deduplicate_candidates(
        self,
        candidates: list[tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]]],
    ) -> list[tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]]]:
        by_id: dict[
            str,
            tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]],
        ] = {}
        for candidate in candidates:
            current = by_id.get(candidate[0])
            if current is None or self._candidate_sort_key(candidate) < self._candidate_sort_key(current):
                by_id[candidate[0]] = candidate
        return list(by_id.values())

    def _canonical_path(self, node_id: str) -> tuple[str, ...]:
        if node_id not in self._nodes_by_id:
            return ()
        reverse_path = [node_id]
        seen = {node_id}
        current = node_id
        while self._parents.get(current):
            parent = self._parents[current][0]
            if parent in seen:
                break
            reverse_path.append(parent)
            seen.add(parent)
            current = parent
        return tuple(reversed(reverse_path))

    def _root_sort_key(self, node_id: str) -> tuple[int, str]:
        node = self._nodes_by_id.get(node_id)
        return (-(node.level if node is not None else -1), node_id)

    @classmethod
    def _normalize_mode(cls, retrieval_mode: str) -> str:
        if not isinstance(retrieval_mode, str):
            raise TypeError("retrieval_mode must be a string")
        normalized = retrieval_mode.strip().lower().replace(" ", "-")
        try:
            return cls._MODE_ALIASES[normalized]
        except KeyError as exc:
            accepted = ", ".join(sorted(cls._MODE_ALIASES))
            raise ValueError(
                f"Unknown RAPTOR retrieval_mode {retrieval_mode!r}; expected one of: {accepted}"
            ) from exc

    @staticmethod
    def _finite_score(score: Any) -> float:
        if isinstance(score, bool) or not isinstance(score, Real):
            raise TypeError("RAPTOR similarity scores must be real numbers")
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("RAPTOR similarity scores must be finite")
        return value

    @classmethod
    def _candidate_sort_key(
        cls,
        candidate: tuple[str, tuple[str, ...], tuple[int, ...], tuple[float, ...]],
    ) -> tuple[float, int, str, tuple[str, ...]]:
        node_id, path, _levels, scores = candidate
        return (-cls._path_score(scores), -len(path), node_id, path)

    @staticmethod
    def _path_score(scores: tuple[float, ...]) -> float:
        if not scores:
            raise ValueError("RAPTOR paths must contain at least one score")
        if any(not math.isfinite(score) for score in scores):
            raise ValueError("RAPTOR path scores must be finite")
        return sum(scores) / len(scores)


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
        retrieval_mode: str = "flat",
        beam_width: int = 4,
        traversal_budget: int = 64,
        **engine_kwargs: Any,
    ):
        self.levels = _validate_non_negative_int(levels, name="levels")
        branching_factor = _validate_top_k(branching_factor, name="branching_factor")
        if branching_factor == 1:
            raise ValueError("branching_factor must be > 1")
        min_cluster_size = _validate_top_k(min_cluster_size, name="min_cluster_size")
        if min_cluster_size == 1:
            raise ValueError("min_cluster_size must be > 1")
        self.embedding_model = embedding_model or HashingEmbedding()
        self.llm_client = llm_client or ExtractiveLLMClient()
        if not callable(getattr(self.embedding_model, "embed_documents", None)) or not callable(
            getattr(self.embedding_model, "embed_query", None)
        ):
            raise TypeError("embedding_model must define embed_documents() and embed_query()")
        if not callable(getattr(self.llm_client, "generate", None)):
            raise TypeError("llm_client must define generate()")
        self.branching_factor = branching_factor
        self.min_cluster_size = min_cluster_size
        self.index = self.build_index(list(documents))
        self.retriever = RAPTORRetrieverV2(
            self.index,
            embedding_model=self.embedding_model,
            retrieval_mode=retrieval_mode,
            beam_width=beam_width,
            traversal_budget=traversal_budget,
        )
        self.engine = RAGEngine(self.retriever, llm_client=self.llm_client, top_k=top_k, **engine_kwargs)

    @classmethod
    def from_documents(cls, documents: Iterable[Document], **kwargs: Any) -> "RAPTOREngine":
        return cls(documents, **kwargs)

    def ask(self, query: str, top_k: int | None = None, **generate_kwargs: Any) -> RAGResponse:
        query = _validate_query(query)
        if top_k is not None:
            top_k = _validate_top_k(top_k)
        response = self.engine.ask(query, top_k=top_k, **generate_kwargs)
        response.metadata.update({"architecture": "raptor", "raptor_index": self.index.to_dict()})
        return response

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        retrieval_mode: str | None = None,
        beam_width: int | None = None,
        traversal_budget: int | None = None,
    ) -> list[Document]:
        top_k = _validate_top_k(top_k)
        return self.retriever.retrieve(
            query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            beam_width=beam_width,
            traversal_budget=traversal_budget,
        )

    def build_index(self, documents: list[Document]) -> RAPTORIndex:
        index = RAPTORIndex()
        current = []
        used_ids: set[str] = set()
        for idx, doc in enumerate(documents):
            _validate_document(doc, name=f"documents[{idx}]")
            doc_id = doc.doc_id or f"raptor-leaf-{idx}"
            if doc_id in used_ids:
                raise ValueError(f"RAPTOR document IDs must be unique: {doc_id!r}")
            used_ids.add(doc_id)
            snapshot = _snapshot_document(doc)
            leaf = Document(
                snapshot.content,
                metadata={**snapshot.metadata, "raptor_level": 0, "node_type": "leaf"},
                doc_id=doc_id,
                score=snapshot.score,
            )
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
                summary_id = f"raptor::L{level}::C{cluster_idx}"
                if summary_id in used_ids:
                    raise ValueError(f"RAPTOR node IDs must be unique: {summary_id!r}")
                used_ids.add(summary_id)
                doc = Document(
                    summary,
                    metadata={"raptor_level": level, "node_type": "summary", "raptor_child_ids": child_ids},
                    doc_id=summary_id,
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
                candidate_indices = sorted(unused)
                scores = cosine_similarity(matrix[seed], matrix[candidate_indices])
                ranked_unused = [
                    idx
                    for _, idx in sorted(
                        zip(scores, candidate_indices),
                        key=lambda item: (-float(item[0]), item[1]),
                    )
                ]
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
        generated = self.llm_client.generate(prompt)
        if not isinstance(generated, str):
            raise TypeError("llm_client.generate() must return a string")
        return generated.strip() or "\n".join(node.document.content[:400] for node in group)


def _validate_query(query: Any) -> str:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    normalized = " ".join(query.split())
    if not normalized:
        raise ValueError("query must not be empty")
    return normalized


def _validate_document(document: Any, *, name: str) -> None:
    if not isinstance(document, Document):
        raise TypeError(f"{name} must be a Document")
    if not isinstance(document.content, str):
        raise TypeError(f"{name}.content must be a string")
    if not isinstance(document.metadata, dict):
        raise TypeError(f"{name}.metadata must be a dict")
    if document.doc_id is not None:
        if not isinstance(document.doc_id, str):
            raise TypeError(f"{name}.doc_id must be a string or None")
        if not document.doc_id.strip():
            raise ValueError(f"{name}.doc_id must be non-empty when provided")
    if document.score is not None:
        if isinstance(document.score, bool) or not isinstance(document.score, Real):
            raise TypeError(f"{name}.score must be a real number or None")
        if not math.isfinite(float(document.score)):
            raise ValueError(f"{name}.score must be finite")
