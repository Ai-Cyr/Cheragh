# FastAPI example

```bash
pip install "cheragh[fastapi]"
export RAG_DOCS_PATH=./docs
uvicorn main:app --reload
```

Endpoints:

- `GET /health`
- `POST /ask` with body `{"query": "...", "top_k": 5}`

This example uses `HashingEmbedding` by default so it runs offline. For better quality, build an engine with `SentenceTransformerEmbedding`, `OpenAIEmbedding`, Chroma, Qdrant, or FAISS.
