# Changelog

## 1.3.0

- Add strict long-context packing with source quotas, deduplication, optional truncation and lost-in-the-middle-aware ordering.
- Add top-down, beam-limited RAPTOR tree traversal while preserving collapsed-tree retrieval.
- Expand Adaptive RAG to route queries across no retrieval, single-step and iterative engines.
- Let FLARE trigger retrieval through injectable token-confidence signals, with a documented length-based fallback.
- Add claim-level faithfulness, contradiction and citation-alignment evaluation behind injectable judge interfaces.
- Deepen Corrective RAG and multi-hop orchestration with bounded, auditable correction and planning components.
- Grow the machine-readable catalogue to 44 available techniques and document the remaining paper-level gaps.

The new orchestration and evaluation components remain experimental. They expose replaceable model boundaries and deterministic fallbacks; they do not bundle the trained classifiers, readers or judges used by every referenced paper.

## 1.2.0

- Implement the four architectures previously marked as planned: Community GraphRAG, ColPali-compatible visual late interaction, Temporal RAG and retrieval-aware training adapters.
- Add a standalone BM25 retriever and make `retriever.type: bm25` a true sparse-only configuration.
- Add and expose canonical multi-retriever Reciprocal Rank Fusion.
- Add direct deterministic tests for every architecture in the 42-technique catalogue.
- Document the scope and limitations of paper-inspired baselines explicitly.

All 42 catalogue entries now have an available implementation or bounded baseline. The four new architectures remain experimental; this release does not claim exhaustive coverage of every RAG method in the literature.

## 1.1.0

- Fix tenant and collection authorization so requests cannot grant themselves access.
- Add strict ACL handling for missing tenant metadata and unknown classifications.
- Disable unsafe legacy pickle loading unless explicitly opted in.
- Prevent index output self-ingestion and reuse unchanged embeddings during incremental updates.
- Add inference-time Self-RAG and bounded Agentic RAG components.
- Add optional SPLADE-style learned sparse and ColBERT-style late-interaction retrievers.
- Add dependency-light multimodal retrieval with an optional CLIP adapter.
- Add a machine-readable technique catalogue and `cheragh techniques list/show`.
- Tighten configuration validation and preserve configured `top_k` in the CLI.

Self-RAG, Agentic RAG, learned retrieval and multimodal RAG are experimental in 1.1.
