"""Minimal usage without Dataiku or remote APIs.

Run from the repository root:

    python examples/basic_usage.py
"""

from cheragh import (
    AdvancedRAGPipeline,
    Document,
    HashingEmbedding,
    HybridSearchRetriever,
    CallableLLMClient,
)


def tiny_llm(prompt: str, **_: object) -> str:
    # Replace this with OpenAILLMClient, an internal API client, or any callable.
    return "Réponse générée par le LLM applicatif.\n\n" + prompt[:400]


documents = [
    Document(
        "Le RAG combine une étape de recherche documentaire avec une génération contrôlée par les sources.",
        doc_id="rag_intro",
        metadata={"source": "demo"},
    ),
    Document(
        "Un package Python moderne contient généralement pyproject.toml, src/<package> et des tests.",
        doc_id="python_packaging",
        metadata={"source": "demo"},
    ),
]

embedder = HashingEmbedding(dimension=128)
retriever = HybridSearchRetriever(documents, embedder, alpha=0.4, cache_path=".cache/hybrid.pkl")
pipeline = AdvancedRAGPipeline(retriever, CallableLLMClient(tiny_llm), top_k=2)

result = pipeline.run("Comment structurer un package Python pour du RAG ?")
print(result["answer"])
print("\nSources:")
for source in result["sources"]:
    print(source)
