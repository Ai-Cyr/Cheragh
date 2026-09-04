"""Structured technique catalogue.

The catalogue makes coverage explicit: an exported class is not automatically
called stable, and paper-inspired baselines document their limitations.  It is
also the source used by the ``cheragh techniques`` CLI.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class TechniqueStatus(str, Enum):
    STABLE = "stable"
    BETA = "beta"
    EXPERIMENTAL = "experimental"
    PLANNED = "planned"


class TechniqueFamily(str, Enum):
    INDEXING = "indexing"
    RETRIEVAL = "retrieval"
    QUERY = "query"
    AUGMENTATION = "augmentation"
    ORCHESTRATION = "orchestration"
    STRUCTURED = "structured"
    MULTIMODAL = "multimodal"
    EVALUATION = "evaluation"
    GOVERNANCE = "governance"


@dataclass(frozen=True)
class TechniqueSpec:
    """One documented RAG capability and its maturity contract."""

    id: str
    name: str
    family: TechniqueFamily
    status: TechniqueStatus
    implementation: str | None
    summary: str
    references: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.implementation is not None and self.status != TechniqueStatus.PLANNED

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["family"] = self.family.value
        payload["status"] = self.status.value
        payload["available"] = self.available
        return payload


def _spec(
    id: str,
    name: str,
    family: TechniqueFamily,
    status: TechniqueStatus,
    implementation: str | None,
    summary: str,
    *,
    references: tuple[str, ...] = (),
    limitations: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
) -> TechniqueSpec:
    return TechniqueSpec(id, name, family, status, implementation, summary, references, limitations, aliases)


TECHNIQUES: tuple[TechniqueSpec, ...] = (
    _spec("naive-rag", "Naive RAG", TechniqueFamily.ORCHESTRATION, TechniqueStatus.STABLE, "cheragh.RAGEngine", "Retrieve, augment and generate with citations."),
    _spec("recursive-chunking", "Recursive chunking", TechniqueFamily.INDEXING, TechniqueStatus.STABLE, "cheragh.RecursiveTextChunker", "Recursive character-aware document splitting."),
    _spec("semantic-chunking", "Semantic chunking", TechniqueFamily.INDEXING, TechniqueStatus.BETA, "cheragh.SemanticChunker", "Embedding-guided topic boundary detection."),
    _spec("hierarchical-chunking", "Hierarchical chunking", TechniqueFamily.INDEXING, TechniqueStatus.BETA, "cheragh.HierarchicalChunker", "Parent/child chunks with section provenance."),
    _spec("sentence-window", "Sentence-window retrieval", TechniqueFamily.INDEXING, TechniqueStatus.EXPERIMENTAL, "cheragh.SentenceWindowRetriever", "Retrieve a sentence and expand its surrounding context."),
    _spec(
        "propositional",
        "Propositional retrieval",
        TechniqueFamily.INDEXING,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.PropositionalRetriever",
        "Index generated atomic propositions while returning source context.",
        limitations=("Proposition quality depends on the injected generator; the fallback is rule based.",),
    ),
    _spec("bm25", "BM25", TechniqueFamily.RETRIEVAL, TechniqueStatus.STABLE, "cheragh.BM25Retriever", "Standalone sparse lexical first-stage retrieval without an embedding model."),
    _spec("dense", "Dense retrieval", TechniqueFamily.RETRIEVAL, TechniqueStatus.STABLE, "cheragh.MemoryVectorStore", "Single-vector semantic retrieval with pluggable embeddings."),
    _spec("hybrid", "Hybrid sparse+dense", TechniqueFamily.RETRIEVAL, TechniqueStatus.STABLE, "cheragh.HybridSearchRetriever", "Score fusion between BM25 and dense retrieval."),
    _spec("reranking", "Cross-encoder reranking", TechniqueFamily.RETRIEVAL, TechniqueStatus.BETA, "cheragh.CrossEncoderReranker", "Second-stage candidate reranking."),
    _spec("rrf", "Reciprocal Rank Fusion", TechniqueFamily.RETRIEVAL, TechniqueStatus.BETA, "cheragh.ReciprocalRankFusionRetriever", "Canonical rank fusion over multiple retrievers."),
    _spec("mmr", "Maximal Marginal Relevance", TechniqueFamily.RETRIEVAL, TechniqueStatus.EXPERIMENTAL, "cheragh.MMRRetriever", "Balance relevance and result diversity."),
    _spec(
        "splade",
        "Learned sparse retrieval (SPLADE)",
        TechniqueFamily.RETRIEVAL,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.retrieval.SPLADERetriever",
        "Learned sparse exact retrieval through an injectable or optional SPLADE encoder.",
        references=("https://arxiv.org/abs/2107.05720",),
        limitations=("Exact in-memory scoring; large corpora require an external inverted index.",),
    ),
    _spec(
        "colbert",
        "Late-interaction retrieval (ColBERT)",
        TechniqueFamily.RETRIEVAL,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.retrieval.ColBERTRetriever",
        "Token-level MaxSim late interaction with injectable encoders.",
        references=("https://arxiv.org/abs/2004.12832",),
        limitations=("Exact in-memory MaxSim; no compressed ANN index is bundled and the default encoder is not a trained ColBERT model.",),
    ),
    _spec("hyde", "HyDE", TechniqueFamily.QUERY, TechniqueStatus.EXPERIMENTAL, "cheragh.HyDERetriever", "Retrieve from an LLM-generated hypothetical answer.", references=("https://arxiv.org/abs/2212.10496",)),
    _spec("hyqe", "HyQE", TechniqueFamily.QUERY, TechniqueStatus.EXPERIMENTAL, "cheragh.HyQERetriever", "Index hypothetical questions associated with source documents."),
    _spec("rag-fusion", "RAG-Fusion", TechniqueFamily.QUERY, TechniqueStatus.EXPERIMENTAL, "cheragh.RAGFusionRetriever", "Generate multiple queries and fuse their result ranks."),
    _spec(
        "self-query",
        "Self-query retrieval",
        TechniqueFamily.QUERY,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.SelfQueryRetriever",
        "Generate a semantic query plus metadata filters.",
        limitations=("The bundled parser supports a bounded metadata-filter grammar; custom parsers remain injectable.",),
    ),
    _spec("step-back", "Step-back prompting", TechniqueFamily.QUERY, TechniqueStatus.EXPERIMENTAL, "cheragh.StepBackRetriever", "Retrieve using a more abstract companion query."),
    _spec("query-decomposition", "Query decomposition", TechniqueFamily.QUERY, TechniqueStatus.EXPERIMENTAL, "cheragh.QueryDecompositionRetriever", "Split complex questions into retrievable sub-questions."),
    _spec("context-compression", "Contextual compression", TechniqueFamily.AUGMENTATION, TechniqueStatus.BETA, "cheragh.ContextualCompressionRetriever", "Retrieve, then remove irrelevant and redundant text through an injectable compression pipeline."),
    _spec(
        "long-context-packing",
        "Long-context packing",
        TechniqueFamily.AUGMENTATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.LongContextPacker",
        "Pack scored evidence under a strict token budget with source quotas and boundary-aware ordering.",
        references=("https://arxiv.org/abs/2406.15319",),
        limitations=(
            "Context packing only; LongRAG's grouped long units and long-reader workflow are separate. Exact limits require the target tokenizer.",
        ),
    ),
    _spec("chain-of-note", "Chain-of-Note", TechniqueFamily.AUGMENTATION, TechniqueStatus.EXPERIMENTAL, "cheragh.ChainOfNoteRetriever", "Generate evidence notes before final synthesis."),
    _spec(
        "crag",
        "Corrective RAG",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.CorrectiveRAGEngine",
        "Grade evidence into correct/ambiguous/incorrect actions, refine it and optionally retrieve externally.",
        references=("https://arxiv.org/abs/2401.15884",),
        limitations=(
            "External/web search and semantic graders are injectable; the bundled decompose/recompose refiner is lexical.",
        ),
    ),
    _spec(
        "self-rag",
        "Inference-time Self-RAG",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.self_rag.SelfRAGEngine",
        "Gate retrieval, critique evidence and refine a grounded answer at inference time.",
        references=("https://openreview.net/forum?id=hSyW5go0v8",),
        limitations=("Modular inference approximation; it does not train reflection-token models.",),
    ),
    _spec(
        "flare",
        "FLARE active retrieval",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.FLAREPipeline",
        "Interleave look-ahead drafting, uncertainty-triggered retrieval and grounded regeneration.",
        references=("https://arxiv.org/abs/2305.06983",),
        limitations=(
            "Token-confidence adapters are injectable; text-only LLM clients use the documented draft-length fallback.",
        ),
    ),
    _spec(
        "adaptive-rag",
        "Adaptive RAG",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.AdaptiveRAGEngine",
        "Route query complexity across no retrieval, single-step RAG and iterative RAG engines.",
        references=("https://arxiv.org/abs/2403.14403",),
        limitations=(
            "Classifier and iterative engine are injectable; the bundled heuristic is deterministic rather than learned.",
        ),
    ),
    _spec("parent-child", "Parent-child retrieval", TechniqueFamily.ORCHESTRATION, TechniqueStatus.BETA, "cheragh.ParentChildRetriever", "Search fine chunks and return larger parent context."),
    _spec(
        "multi-hop",
        "Multi-hop RAG",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.BETA,
        "cheragh.MultiHopRAGEngine",
        "Interleave bounded planning, retrieval and evidence observations before final synthesis.",
        references=(
            "https://arxiv.org/abs/2212.10509",
            "https://arxiv.org/abs/2210.03629",
        ),
        limitations=(
            "Planner quality is application-provided; bundled rule-based/JSON adapters do not reproduce trained reasoning policies.",
        ),
    ),
    _spec(
        "raptor",
        "RAPTOR",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.RAPTOREngine",
        "Build summary trees and retrieve through collapsed or budgeted top-down beam traversal.",
        references=("https://arxiv.org/abs/2401.18059",),
        limitations=(
            "Optional UMAP/GMM soft clustering and paper_tree traversal; semantic embeddings, an abstractive summarizer and benchmark validation are caller responsibilities.",
        ),
    ),
    _spec(
        "graph-rag",
        "Graph-enhanced RAG",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.GraphRAGEngine",
        "Blend entity/relation neighbourhoods with vector retrieval.",
        references=("https://arxiv.org/abs/2404.16130",),
        limitations=("Graph-lite baseline; no community detection or global community summaries.",),
    ),
    _spec(
        "agentic-rag",
        "Agentic RAG",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.agentic.AgenticRAGEngine",
        "Bounded plan/action/observation loop over explicitly registered tools.",
        references=("https://arxiv.org/abs/2210.03629",),
        limitations=("Inference orchestration only; tools must be registered by the application.",),
    ),
    _spec("federated", "Federated RAG", TechniqueFamily.ORCHESTRATION, TechniqueStatus.BETA, "cheragh.FederatedRAGEngine", "Merge evidence from multiple retrievers or domains."),
    _spec(
        "conversational",
        "Conversational RAG",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.BETA,
        "cheragh.ConversationalRAGEngine",
        "Retrieve with a bounded query-context window.",
        limitations=("The in-memory store keeps all turns; applications must provide retention for long-lived sessions.",),
    ),
    _spec("sql-rag", "SQL RAG", TechniqueFamily.STRUCTURED, TechniqueStatus.BETA, "cheragh.SQLRAGEngine", "Generate and execute guarded read-only SQLite queries."),
    _spec(
        "multimodal-rag",
        "Multimodal RAG",
        TechniqueFamily.MULTIMODAL,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.MultimodalRAGEngine",
        "Cross-modal text/image retrieval with media provenance and grounded generation.",
        references=("https://arxiv.org/abs/2103.00020",),
        limitations=("Bundled CLIP adapter covers text/local images; audio/video need transcripts or a custom encoder.",),
    ),
    _spec("retrieval-evaluation", "Retrieval evaluation", TechniqueFamily.EVALUATION, TechniqueStatus.STABLE, "cheragh.evaluate_retrieval", "Hit rate, MRR, precision, recall, nDCG and context precision."),
    _spec("generation-evaluation", "Generation evaluation", TechniqueFamily.EVALUATION, TechniqueStatus.BETA, "cheragh.evaluate_generation", "Deterministic citation and lexical grounding diagnostics."),
    _spec(
        "claim-evaluation",
        "Claim-level faithfulness evaluation",
        TechniqueFamily.EVALUATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.ClaimEvaluator",
        "Separate claim support, contradiction and citation-to-evidence alignment with injectable judges.",
        references=(
            "https://arxiv.org/abs/2309.15217",
            "https://arxiv.org/abs/2408.08067",
        ),
        limitations=(
            "The dependency-free lexical fallback cannot detect semantic paraphrases or contradictions; inject an NLI/LLM judge.",
        ),
    ),
    _spec("access-controlled-rag", "Access-controlled RAG", TechniqueFamily.GOVERNANCE, TechniqueStatus.BETA, "cheragh.AccessControlledRAGEngine", "Filter retrieved evidence using tenant, collection, role and classification policy."),
    _spec(
        "community-graphrag",
        "Community GraphRAG",
        TechniqueFamily.ORCHESTRATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.CommunityGraphRAGEngine",
        "Detect communities, build reports and support global or local graph-grounded search.",
        references=("https://arxiv.org/abs/2404.16130",),
        limitations=("Optional hierarchical Leiden and bounded global map-reduce; graph extraction, report quality and benchmark validation remain application responsibilities.",),
    ),
    _spec(
        "colpali",
        "Visual-document late interaction",
        TechniqueFamily.MULTIMODAL,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.ColPaliRetriever",
        "Retrieve page images with token-to-patch MaxSim and a ColPali-compatible encoder boundary.",
        references=("https://arxiv.org/abs/2407.01449",),
        limitations=("Exact in-memory scoring; the official model adapter is optional and has heavyweight dependencies.",),
    ),
    _spec(
        "temporal-rag",
        "Temporal RAG",
        TechniqueFamily.RETRIEVAL,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.TemporalRetriever",
        "Filter by validity windows, weight freshness and resolve document versions at retrieval time.",
        references=("https://aclanthology.org/2024.emnlp-main.394/",),
        limitations=("Requires reliable temporal metadata and comparable base scores; no temporal graph or query-time parser is bundled.",),
    ),
    _spec(
        "retrieval-training",
        "Retriever/RAG training",
        TechniqueFamily.EVALUATION,
        TechniqueStatus.EXPERIMENTAL,
        "cheragh.RetrievalTrainingPipeline",
        "Prepare hard negatives, distillation labels and RAFT-style records behind an injectable trainer boundary.",
        references=(
            "https://arxiv.org/abs/2104.08051",
            "https://arxiv.org/abs/2403.10131",
            "https://arxiv.org/abs/2407.02485",
        ),
        limitations=("Grounded RAFT data and optional Torch contrastive/KL training; callers supply encoders and optimizers. No pretrained weights, LLM SFT runner or distributed training.",),
    ),
)


_BY_ID = {spec.id: spec for spec in TECHNIQUES}
_BY_ALIAS = {alias: spec for spec in TECHNIQUES for alias in spec.aliases}


def get_technique(identifier: str) -> TechniqueSpec:
    """Return a technique by stable identifier or alias."""

    key = identifier.strip().lower()
    try:
        return _BY_ID.get(key) or _BY_ALIAS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown RAG technique: {identifier}") from exc


def list_techniques(
    *,
    status: TechniqueStatus | str | None = None,
    family: TechniqueFamily | str | None = None,
    available: bool | None = None,
) -> list[TechniqueSpec]:
    """List techniques using stable, composable filters."""

    status_value = TechniqueStatus(status) if status is not None else None
    family_value = TechniqueFamily(family) if family is not None else None
    return [
        spec
        for spec in TECHNIQUES
        if (status_value is None or spec.status == status_value)
        and (family_value is None or spec.family == family_value)
        and (available is None or spec.available is available)
    ]
