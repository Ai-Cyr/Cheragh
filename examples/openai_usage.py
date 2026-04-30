"""Example with OpenAI-compatible generation.

Install extras first:

    pip install -e '.[openai,local]'

Then set OPENAI_API_KEY and run this file.
"""

from cheragh import (
    AdvancedRAGPipeline,
    Document,
    HybridSearchRetriever,
    OpenAILLMClient,
    SentenceTransformerEmbedding,
)


docs = [
    Document("Le RAG récupère des passages pertinents avant de générer la réponse.", doc_id="rag"),
    Document("Les embeddings sémantiques aident à retrouver des passages proches du sens de la requête.", doc_id="embeddings"),
]

embedder = SentenceTransformerEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
retriever = HybridSearchRetriever(docs, embedder, alpha=0.6, cache_path=".cache/openai_demo.pkl")
llm = OpenAILLMClient(model="gpt-4o-mini")

pipeline = AdvancedRAGPipeline(retriever, llm)
print(pipeline.run("À quoi servent les embeddings dans un système RAG ?"))
