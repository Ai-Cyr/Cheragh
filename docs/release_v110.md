# Cheragh 1.1 — sécurité et techniques RAG modernes

La version 1.1 ajoute des briques modernes sans présenter les baselines comme des reproductions complètes des publications. Consultez le catalogue avec `cheragh techniques list` ou `cheragh techniques show <id>`.

## Installation

```bash
pip install 'cheragh[learned-retrieval,multimodal]'
```

Les encodeurs restent injectables : les tests et les petits déploiements peuvent fonctionner sans charger de modèle externe.

## SPLADE et learned sparse

```python
from cheragh import Document, SPLADERetriever

retriever = SPLADERetriever(
    [Document("Paris est la capitale de la France", doc_id="paris")],
    model_name="naver/splade-cocondenser-ensembledistil",
)
results = retriever.retrieve("capitale française", top_k=3)
```

Le scoring est un produit scalaire sparse exact en mémoire. Pour un grand corpus, persistez les poids dans un moteur à index inversé.

## ColBERT / late interaction

```python
from cheragh import ColBERTRetriever, Document

retriever = ColBERTRetriever(
    [Document("Un passage à indexer", doc_id="passage")],
    token_encoder=my_colbert_encoder,
)
results = retriever.retrieve("ma requête", top_k=5)
```

Cheragh calcule le MaxSim canonique exact. L'ANN multi-vecteur compressé n'est pas inclus ; injectez un encodeur entraîné ColBERT pour une qualité fidèle au modèle.

## Self-RAG d'inférence

```python
from cheragh import LexicalEvidenceCritic, SelfRAGEngine

engine = SelfRAGEngine(
    retriever=retriever,
    llm_client=llm,
    evidence_critic=LexicalEvidenceCritic(),
    max_refinements=2,
)
result = engine.ask("Quelle politique s'applique ?")
print(result.answer, result.trace.stop_reason)
```

Cette API orchestre gate, retrieval, critique et raffinement. Elle n'entraîne pas un modèle à reflection tokens.

## Agentic RAG borné

```python
from cheragh import AgenticRAGEngine, LLMJSONPlanner, RetrievalToolAdapter, ToolRegistry

registry = ToolRegistry()
registry.register(RetrievalToolAdapter(retriever).as_tool())
engine = AgenticRAGEngine(
    planner=LLMJSONPlanner(llm),
    tools=registry,
    max_steps=4,
)
result = engine.ask("Compare les deux politiques")
```

Seuls les outils enregistrés sont appelables. La longueur des entrées et le nombre d'étapes sont limités ; les handlers applicatifs restent responsables de leur propre isolation.

## Multimodal texte/image

```python
from cheragh import CLIPMultimodalEmbedding, Modality, MultimodalDocument, MultimodalRetriever

documents = [
    MultimodalDocument(
        content="Schéma d'architecture",
        modality=Modality.IMAGE,
        uri="./architecture.png",
        doc_id="architecture",
    )
]
retriever = MultimodalRetriever(documents, CLIPMultimodalEmbedding())
results = retriever.retrieve("schéma du pipeline", top_k=3)
```

L'adaptateur fourni couvre le texte et les images locales. Pour audio/vidéo, fournissez un transcript ou un `MultimodalEmbeddingModel` personnalisé.

## Changements de sécurité

- Une requête tenant/collection doit déjà appartenir au `Principal`; un administrateur doit être explicite.
- Le mode ACL strict refuse les documents sans tenant/collection et les classifications inconnues.
- `load_cache()` ne désérialise plus de pickle sans `allow_unsafe_pickle=True` explicite.
- Un output d'index sous le corpus source est automatiquement exclu.

Ces changements peuvent révéler des métadonnées historiques incomplètes. Corrigez les documents concernés au lieu de désactiver le mode strict dans les environnements multi-tenant.
