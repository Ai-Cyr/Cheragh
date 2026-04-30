# v0.8.0 — Hardening

La version 0.8.0 renforce les chemins production les plus sensibles.

## Cache sécurisé ou signé

- `SQLiteCache` et `RedisCache` utilisent `serializer="json"` par défaut.
- `serializer="signed-pickle"` exige `secret_key` et `allow_pickle=true`.
- Le pickle persistant non signé est refusé sauf `allow_unsigned_pickle=true`, à réserver aux caches locaux entièrement fiables.

## SQLite read-only réel

`SQLRAGEngine` combine plusieurs barrières :

- ouverture SQLite `mode=ro` pour les bases fichier ;
- `PRAGMA query_only=ON` ;
- authorizer SQLite refusant écritures, `PRAGMA`, `ATTACH`, DDL et transactions ;
- progress handler pour interrompre les requêtes trop coûteuses.

Les moteurs créés avec `from_records()` ou `StructuredRAG.from_tables()` matérialisent les données en mode écriture temporaire puis passent automatiquement en lecture seule.

## Validation Pydantic

`load_config()` valide les fichiers YAML/JSON par défaut avec un schéma Pydantic v2. La CLI expose aussi :

```bash
cheragh validate-config rag.yaml
cheragh validate-config rag.yaml --json
```

La validation rejette notamment les clés inconnues, `chunk_overlap >= chunk_size`, les types de retriever non supportés et les caches pickle persistants non signés.

## Docker reproductible

L'image de démonstration pinne :

- l'image Python via `PYTHON_IMAGE=python:3.12.7-slim-bookworm` ;
- `pip==24.3.1` ;
- les dépendances top-level dans `docker/constraints.txt` ;
- Qdrant dans `docker-compose.yml`.
