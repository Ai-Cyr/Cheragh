# Architectures RAG v0.5.0

Cette version ajoute quatre architectures composables au-dessus des briques RAG existantes.

## Parent-child retrieval

`ParentChildRetriever` indexe des petits chunks enfants, puis remonte les sections ou documents parents plus larges au moment de la génération. Cela améliore la précision du retrieval sans priver le LLM du contexte nécessaire.

```python
from cheragh import Document, HashingEmbedding, ParentChildRetriever

parents = [Document("Contrat complet...", doc_id="contract-a")]
children = [Document("Préavis de 30 jours", metadata={"parent_doc_id": "contract-a"})]

retriever = ParentChildRetriever(
    parent_documents=parents,
    child_documents=children,
    embedding_model=HashingEmbedding(),
)
```

## Corrective RAG

`CorrectiveRAGEngine` vérifie la qualité du contexte récupéré. Si le contexte est trop faible, il reformule ou renvoie une réponse de fallback au lieu de générer une réponse mal sourcée.

```python
from cheragh import CorrectiveRAGEngine

corrective = CorrectiveRAGEngine(
    base_engine=engine,
    min_context_score=0.12,
    max_retries=2,
)
response = corrective.ask("Quelle clause s'applique ?")
```

## Conversational RAG

`ConversationalRAGEngine` ajoute une mémoire de session et transforme les questions de suivi en requêtes autonomes.

```python
from cheragh import ConversationalRAGEngine

chat = ConversationalRAGEngine(engine)
chat.ask("Résume le contrat A", session_id="u1")
chat.ask("Et quelles clauses sont risquées ?", session_id="u1")
```

## RAGWorkflow

`RAGWorkflow` est un exécuteur DAG minimal pour composer des pipelines personnalisés.

```python
from cheragh import RAGWorkflow, RetrieveNode, GenerateNode

workflow = RAGWorkflow()
workflow.add_node("retrieve", RetrieveNode(retriever, top_k=5))
workflow.add_node("generate", GenerateNode(llm))
workflow.connect("retrieve", "generate")

result = workflow.ask("Explique le RAG")
```
