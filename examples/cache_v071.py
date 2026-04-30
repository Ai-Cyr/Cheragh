from pathlib import Path
import tempfile

from cheragh import Document, HashingEmbedding, RAGEngine, StaticLLMClient
from cheragh.cache import SQLiteCache


def main():
    docs = [
        Document("Le contrat Alpha prévoit un préavis de 30 jours.", doc_id="alpha"),
        Document("Le contrat Beta prévoit un préavis de 60 jours.", doc_id="beta"),
    ]

    with tempfile.TemporaryDirectory() as tmp:
        cache = SQLiteCache(Path(tmp) / "rag-cache.sqlite", default_ttl=3600)
        engine = RAGEngine.from_documents(
            docs,
            embedding_model=HashingEmbedding(dimension=128),
            retriever_type="memory",
            llm_client=StaticLLMClient("Le préavis est de 30 jours. [source: alpha]"),
            cache_backend=cache,
        )

        first = engine.ask("Quel est le préavis du contrat Alpha ?")
        second = engine.ask("Quel est le préavis du contrat Alpha ?")

        print(first.answer)
        print(second.metadata["cache"])


if __name__ == "__main__":
    main()
