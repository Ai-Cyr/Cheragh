# Architectures RAG avancées — v0.6.0

Cette version ajoute quatre architectures au-dessus des briques existantes.

## MultiHopRAGEngine

Décompose une question complexe en sous-questions, récupère les preuves étape par étape, puis synthétise une réponse finale.

```python
from cheragh import RAGEngine, HashingEmbedding, StaticLLMClient, Document
from cheragh.multihop import MultiHopRAGEngine

base = RAGEngine.from_documents(docs, embedding_model=HashingEmbedding())
engine = MultiHopRAGEngine(base.retriever, llm_client=StaticLLMClient("réponse"), max_steps=4)
result = engine.ask("Compare les ventes Q1 et les risques fournisseurs")

print(result.answer)
print(result.decomposed_queries)
print(result.hops)
```

## GraphRAGEngine

Construit un graphe de connaissances léger à partir des documents, récupère les documents reliés aux entités de la question, puis génère une réponse.

```python
from cheragh.graph import GraphRAGEngine

engine = GraphRAGEngine.from_documents(docs, embedding_model=embedder, llm_client=llm)
response = engine.ask("Quels fournisseurs sont liés au Produit B ?")

print(response.metadata["query_entities"])
print(response.metadata["graph_triples"])
```

## RAPTOREngine

Crée des résumés hiérarchiques des documents, indexe les feuilles et les résumés, puis interroge toute la hiérarchie.

```python
from cheragh.raptor_engine import RAPTOREngine

engine = RAPTOREngine.from_documents(
    docs,
    embedding_model=embedder,
    llm_client=llm,
    levels=2,
    branching_factor=4,
)
response = engine.ask("Résume les risques transverses")
```

## FederatedRAGEngine

Interroge plusieurs sources : `RAGEngine`, retrievers, objets avec `.ask(...)` ou `.retrieve(...)`, ou callables.

```python
from cheragh.federated import FederatedRAGEngine

engine = FederatedRAGEngine(
    sources={
        "finance": finance_engine,
        "legal": legal_engine,
        "risk": risk_retriever,
    },
    llm_client=llm,
)
response = engine.ask("Quels impacts financiers ont les clauses à risque ?")

print(response.metadata["sources_queried"])
print(response.sources)
```

## Choix d'architecture

- Question simple factuelle : `RAGEngine`
- Question comparative ou analytique : `MultiHopRAGEngine`
- Corpus riche en entités et relations : `GraphRAGEngine`
- Corpus volumineux ou synthèses transverses : `RAPTOREngine`
- Plusieurs index ou domaines : `FederatedRAGEngine`
