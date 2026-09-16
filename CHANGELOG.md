# Changelog

## Unreleased

- Support Sentence Transformers 6 alongside 5, with CPU integration tests for local embeddings, reranking, token embeddings and CLIP.
- Encode CLIP text and image batches separately and restore document order, supporting versions that reject mixed-modality batches.

- Correct HyQE additive candidate scoring, HyDE document-space averaging, true cosine, BM25 consistency and rank-based query/federated fusion.
- Add trained SPLADE and ColBERTv2 encoders with checkpoint-specific tokenization/projection; correct ColPali processing and MaxSim masks.
- Add passage-conditioned segmented Self-RAG with a Transformers logits decoder and bounded beam search.
- Add actual strategy-labelled Adaptive-RAG training, causal/seq2seq RAFT SFT and joint RankRAG ranking/answer generation.
- Add LongRAG grouping/retrieval/reader, full Step-Back and sequential Chain-of-Note inference paths.
- Add semantic GraphRAG extraction/local search, calibrated CRAG grading/refinement, concrete NLI/LLM claim evaluation and TimeR4 dual retrieval/temporal supervision.
- Preserve chunk coverage and source offsets, validate extracted evidence, retain parent ACLs and authorize all source dependencies of derived summaries.
- Add real tiny-model learning tests, opt-in published-checkpoint checks, CPU research CI and an updated 44-technique fidelity audit.

- Make Python async streaming non-blocking, preserve provider context and close cancelled streams even when the executor is saturated.
- Intersect multi-tenant retrieval with the selected tenant and collection, including admin users, while preserving custom ACL policies and progressively searching past denied results.
- Isolate cached authorized retrieval by principal, policy and selected collection, including changes in permissions.
- Make the default extractive client answer from retrieved evidence instead of echoing system instructions and placeholder citations.
- Remove cache-key type/separator collisions, preserve reserved-key JSON metadata and prevent SQLite quarantine from deleting a concurrent replacement.
- Validate and copy cached embeddings before publication; keep vector-store documents and embeddings aligned on allocation failure.
- Rebuild untracked local indexes without retaining unrelated documents from a previous corpus.
- Apply HTTP capacity and timeout limits to stats, validate listener settings before loading providers and honor explicit indexing overrides.
- Exercise installed wheel/sdist through indexing and generation, retain validated CI artifacts with hashes, and audit the deployed server dependency profile.

- Add optional RAPTOR global/local UMAP–GMM soft clustering, bounded summary inputs and cosine-based `paper_tree` traversal.
- Add hierarchical Leiden community detection and budgeted GraphRAG global map-reduce with source authorization and citation checks.
- Connect FLARE to aligned generation-time token log probabilities and mask uncertain tokens from retrieval queries.
- Add grounded RAFT supervision with verified quote spans, reproducible document shuffling and fixed-size oracle dropout contexts.
- Add optional real PyTorch retrieval optimization and strict Self-RAG reflection probability scoring boundaries.
- Preserve request context and stream cleanup under cancellation; enforce omitted `top_k` and chunked request limits before provider work.
- Close OpenAI/Azure/LiteLLM transports explicitly and accept usage-only streaming events.
- Encode Redis prefix/namespace/key boundaries independently; the new v2 key format starts cold and leaves legacy keys untouched (see the production migration guide).
- Audit all 44 technique entries against their documented scope and retain explicit research and deployment qualification limits.

- Add a stable `response_id` to every RAG response and carry it into feedback evaluation metadata.
- Make feedback evaluation exports use the evaluator's canonical `query` field while continuing to accept legacy `question` datasets.
- Deepen access-controlled retrieval progressively, up to a configurable candidate limit, so authorized evidence is not lost behind a fixed over-fetch window.
- Preserve context packing, cache and tracing configuration when creating request-scoped access-controlled engines, including multi-tenant proxies.

## 1.4.0

- Harden the FastAPI boundary with fail-closed authentication on public binds, strict request models, payload and `top_k` limits, admission control, bounded operations, safe request IDs and redacted errors.
- Add configurable provider timeouts and retries, exact environment-secret references and secret-safe normalized config output.
- Validate and cap query transformations, retriever/compressor output and streaming chunks at every core engine trust boundary.
- Make local vector snapshots checksummed and recoverable across interrupted writes; detect corrupt documents, manifests and embeddings before serving them.
- Make incremental indexing detect source races, incomplete stores and configuration drift while using atomic manifests and process-safe locks.
- Add thread-safe cache backends, per-key single-flight, a bounded LRU memory cache and safer Redis/SQLite failure handling.
- Harden JSONL tracing for concurrent writers and monotonic latency measurement.
- Add a non-root, read-only-friendly multi-stage container, a production deployment guide, a security policy and supply-chain CI gates.

Cheragh remains a beta toolkit: production deployment still requires a TLS/API gateway, external secret management, rate limiting, backups, provider-specific evaluation and operational monitoring. The local memory store is intended for modest corpora, not horizontal multi-writer workloads.

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
