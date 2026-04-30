# cheragh quickstart

## Install

```bash
pip install cheragh
```

Optional integrations:

```bash
pip install "cheragh[openai,chroma,qdrant,rerank,fastapi,pdf,docx]"
```

## Index a folder

```bash
cheragh index ./docs --output .cheragh_index --incremental
```

## Ask a question

```bash
cheragh ask "Résume le corpus" --index .cheragh_index --json
```

## Use the Python API

```python
from cheragh import RAGEngine, HashingEmbedding

engine = RAGEngine.from_path(
    "./docs",
    embedding_model=HashingEmbedding(),
    retriever_type="memory",
    query_transformer="multi-query",
    compressor="default",
    strict_grounding=True,
)

response = engine.ask("Quelle est la décision principale ?")
print(response.answer)
print(response.trace.to_dict())
```

## Serve an API

```bash
cheragh serve --config rag.yaml --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /ask`
- `POST /stream`
- `POST /index`
- `GET /stats`
