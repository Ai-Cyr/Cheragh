# Architectures enterprise v0.7.0

La v0.7.0 ajoute cinq blocs : SQL RAG, Structured RAG, contrôle d'accès, multi-tenancy et feedback loop.

## SQLRAGEngine

```python
from cheragh import SQLRAGEngine

engine = SQLRAGEngine.from_records(
    "sales",
    [
        {"client": "Alpha", "revenue": 100, "quarter": "Q1"},
        {"client": "Beta", "revenue": 180, "quarter": "Q1"},
    ],
)

response = engine.ask("Quel est le revenu total ?")
print(response.answer)
print(response.metadata["sql"])
```

Le moteur valide les requêtes SQL en lecture seule et limite les résultats.

## StructuredRAG

```python
from cheragh import StructuredRAG

rag = StructuredRAG.from_tables({
    "customers": [
        {"name": "Alice", "score": 10},
        {"name": "Bob", "score": 5},
    ]
})

print(rag.ask("Qui a le score le plus grand ?").answer)
```

## Access control

```python
from cheragh import AccessControlledRAGEngine, AccessPolicy, Principal

guarded = AccessControlledRAGEngine(base_engine, policy=AccessPolicy(require_tenant_match=True))
response = guarded.ask(
    "Que dit le contrat ?",
    principal=Principal(user_id="u1", tenant_ids={"tenant-a"}, max_classification="internal"),
)
```

Les documents sont filtrés via leurs métadonnées : `tenant_id`, `collection_id`, `classification`, `allowed_users`, `allowed_roles`, `denied_users`, `denied_roles`.

## Multi-tenancy

```python
from cheragh import MultiTenantRAGEngine

mt = MultiTenantRAGEngine()
mt.add_collection("tenant-a", "contracts", engine_a, default=True)
mt.add_collection("tenant-b", "contracts", engine_b, default=True)

response = mt.ask("Résume le contrat", tenant_id="tenant-a", collection_id="contracts")
```

## Feedback loop

```python
from cheragh import FeedbackLoop

feedback = FeedbackLoop.from_jsonl("feedback.jsonl")
feedback.log_feedback(
    query="Quelle est la clause de résiliation ?",
    rating="bad",
    correct_answer="Préavis de 30 jours.",
    correct_source_ids=["contract-page-12"],
)

print(feedback.summary().to_dict())
feedback.export_evalset("evalset.jsonl", only_negative=True)
```
