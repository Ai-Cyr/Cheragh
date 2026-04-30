# Production notes

Recommended production stack for v1.0.0:

1. Use real embeddings: OpenAI, Azure OpenAI, Cohere, Voyage or SentenceTransformers.
2. Use a persistent vector database for large corpora; use hybrid search for smaller or mixed-language corpora.
3. Retrieve a large candidate set, then rerank.
4. Enable metadata filters for tenant, ACL, language, freshness and document type.
5. Enable context compression before generation.
6. Keep `strict_grounding=True` and `require_citations=True` for sensitive use cases.
7. Preserve citation offsets from ingestion (`source_char_start`, `source_char_end`) and surface them in answers/logs.
8. Evaluate with `recall@k`, `nDCG@k` and `context_precision@k`, not only hit-rate.
9. Enable tracing and export traces for debugging.
10. Prefer incremental indexing for file-system based corpora.

## Recommended YAML preset

Start from `examples/presets/production_v100.yaml` or `examples/presets/production_hybrid.yaml`.

```yaml
ingestion:
  path: ./docs
  chunk_size: 900
  chunk_overlap: 150
  max_file_size_mb: 50

embedding:
  provider: openai
  model: text-embedding-3-small

retriever:
  type: hybrid
  top_k: 6
  alpha: 0.55
  filters: {}
  tokenizer:
    strip_accents: true
    keep_hyphenated: true
    ngram_range: [1, 2]
    use_default_stopwords: true

reranker:
  enabled: true
  provider: keyword
  first_stage_top_k: 40

compression:
  enabled: true
  type: default

query:
  enabled: true
  type: multi-query

generation:
  provider: openai
  model: gpt-4o-mini

strict_grounding: true
require_citations: true
flag_unsourced_sentences: true
trace_enabled: true
min_score: 0.03

cache:
  enabled: true
  backend: sqlite
  path: .cheragh/cache.sqlite
  serializer: json
  ttl: 3600

observability:
  enabled: true
  trace_export_path: .cheragh/traces.jsonl
  trace_include_prompt: false

indexing:
  incremental: true
  use_lock: true
```

## Evaluation gate before release

Run a labeled retrieval set in JSONL/YAML and reject builds that regress key metrics:

- `recall@10` for coverage ;
- `ndcg@10` for ranking quality ;
- `context_precision@5` for context cleanliness ;
- citation coverage and citation accuracy for generation quality.

Example:

```python
from cheragh import RetrievalExample, evaluate_retrieval

examples = [
    RetrievalExample("préavis contrat alpha", {"contract-alpha"}),
    {"query": "politique sécurité SQLite", "expected_doc_ids": ["sqlite-hardening"]},
]

result = evaluate_retrieval(examples, engine.retriever, top_k=10)
assert result.metrics["recall@10"] >= 0.85
assert result.metrics["ndcg@10"] >= 0.75
```
