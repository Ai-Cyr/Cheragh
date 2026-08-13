# Changelog

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
