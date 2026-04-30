# cheragh v1.0.0 — production baseline

v1.0.0 stabilizes the public API and adds production-oriented operations around tracing and incremental indexing.

## Stable public API

Applications should import these names from `cheragh` or `cheragh.schema`:

- `Document`
- `Chunk`
- `Source`
- `RAGResponse`
- `RetrieverProtocol`
- `EmbeddingProtocol`
- `LLMProtocol`
- `RerankerProtocol`

The `RAGResponse.to_dict(include_prompt=False)` contract is stable for JSON APIs. Prompt export remains opt-in to avoid leaking sensitive context.

## Observability

Tracing is enabled by default in `RAGEngine`. Each response can include:

- per-step duration;
- retrieved document ids and scores;
- query variants;
- compression stats;
- estimated input/output tokens;
- estimated LLM cost when pricing is configured;
- warnings from grounding/citation validation.

### JSONL trace export

```python
from cheragh import Document, HashingEmbedding, RAGEngine, StaticLLMClient

engine = RAGEngine.from_documents(
    [Document("SQLite read-only protects data", doc_id="d1")],
    embedding_model=HashingEmbedding(),
    llm_client=StaticLLMClient("SQLite is protected [source: d1]"),
    trace_export_path=".cheragh/traces.jsonl",
    trace_pricing={"input_per_1k": 0.00015, "output_per_1k": 0.0006},
)
engine.ask("How is SQLite protected?")
```

Config equivalent:

```yaml
observability:
  enabled: true
  trace_export_path: .cheragh/traces.jsonl
  trace_include_prompt: false
  pricing:
    currency: USD
    input_per_1k: 0.00015
    output_per_1k: 0.0006
```

## Incremental indexing

`cheragh index` is incremental by default. The index stores `index_manifest.json` with `sha256`, `mtime`, file size and chunk ids per source file.

```bash
cheragh index ./docs --output .cheragh_index
cheragh index ./docs --output .cheragh_index --dry-run
cheragh index ./docs --output .cheragh_index --force
cheragh inspect-index --index .cheragh_index
```

Production defaults:

- unchanged files are kept;
- changed files are re-chunked;
- deleted files are removed from the persisted document set;
- large files can be skipped with `--max-file-size-mb`;
- generated folders are excluded by default;
- a `.index.lock` file prevents concurrent writers.

## Operational checks

```bash
cheragh doctor
cheragh doctor --json
cheragh validate-config rag.yaml
```

## Compatibility note

v1.0.0 keeps the v0.9 retrieval quality additions: Unicode tokenization, metadata filters, citation offsets and retrieval metrics. For new production deployments, prefer config validation plus incremental indexing and JSONL trace export from day one.
