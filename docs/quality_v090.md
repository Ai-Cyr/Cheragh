# v0.9.0 — Qualité RAG

La version 0.9.0 se concentre sur la qualité de recherche et la vérifiabilité des réponses.

## Tokenisation hybride améliorée

`HybridSearchRetriever` utilise désormais `RetrievalTokenizer` au lieu d'une simple découpe par espaces. Le tokenizer gère :

- Unicode et apostrophes typographiques ;
- suppression optionnelle des accents, utile en français ;
- termes composés avec traits d'union ;
- stopwords FR/EN configurables ;
- n-grams de mots, par défaut `(1, 2)`.

Exemple :

```python
from cheragh import Document, HashingEmbedding, HybridSearchRetriever, RetrievalTokenizer

docs = [Document("Le stockage read-only protège SQLite.", doc_id="sqlite")]
retriever = HybridSearchRetriever(
    docs,
    HashingEmbedding(),
    tokenizer=RetrievalTokenizer(ngram_range=(1, 2)),
)
```

## Filtres metadata enrichis

Les filtres metadata supportent maintenant des opérateurs simples :

```python
filters = {
    "tenant": "acme",
    "lang": {"$in": ["fr", "en"]},
    "archived": {"$ne": True},
    "quality": {"$gte": 0.8},
    "tags": {"$contains": "legal"},
}
```

Ils fonctionnent dans `HybridSearchRetriever` et dans le vector store mémoire. Pour le retriever hybride, tu peux définir des filtres au constructeur et aussi au moment de la requête :

```python
retriever.retrieve("clause de résiliation", top_k=5, filters={"tenant": "acme"})
```

## Métriques retrieval

`evaluate_retrieval` calcule maintenant :

- `hit_rate@k` ;
- `mrr` ;
- `precision@k` ;
- `recall@k` ;
- `ndcg@k` ;
- `context_precision@k`.

Les labels peuvent pointer vers un document parent : un chunk `contract#chunk-2` matche automatiquement `contract` via son `parent_doc_id` ou le préfixe avant `#`.

```python
from cheragh import RetrievalExample, evaluate_retrieval

result = evaluate_retrieval(
    [RetrievalExample("préavis contrat", {"contract"})],
    retriever,
    top_k=5,
)
print(result.metrics)
```

## Offsets de citations

Les chunkers `RecursiveTextChunker` et `TokenTextChunker` ajoutent des offsets dans `Document.metadata` :

- `source_char_start` ;
- `source_char_end` ;
- `parent_doc_id` ;
- `chunk_index` ;
- `chunker`.

Le contexte envoyé au LLM inclut une ligne `location` lorsque ces métadonnées existent. Les sources de `RAGResponse` exposent aussi `source.location`.

## Presets production

Deux niveaux sont disponibles :

1. `RAGEngine.from_config("examples/presets/production_hybrid.yaml")` pour une configuration YAML reproductible ;
2. `production_hybrid_rag(...)` pour créer rapidement un pipeline hybride strict depuis du code.

Le preset production recommandé combine :

- embeddings réels ;
- hybrid search avec n-grams ;
- metadata filters ;
- reranking ;
- compression ;
- query transformation ;
- strict grounding ;
- citations obligatoires ;
- cache JSON signé/sécurisé selon le backend ;
- tracing activé.
