# v0.7.1/v0.8.0 — Caching avancé et sécurisé

La v0.7.1 ajoute une vraie couche de cache configurable pour les embeddings, le retrieval, le reranking et les réponses LLM.

## Backends

```python
from cheragh.cache import MemoryCache, SQLiteCache, RedisCache
```

Backends disponibles :

- `MemoryCache` : cache local en mémoire, utile pour tests et serveurs courts.
- `SQLiteCache` : cache persistant local, sans service externe.
- `RedisCache` : cache partagé entre workers, optionnel, nécessite `redis`.

Tous les backends supportent :

- TTL par entrée
- namespaces
- invalidation par namespace
- `clear()` global
- nettoyage des entrées expirées
- statistiques hit/miss/set/delete

## Configuration YAML

```yaml
cache:
  enabled: true
  backend: sqlite
  path: .cheragh/cache.sqlite
  ttl: 3600
  namespace: default
  serializer: json  # défaut sécurisé v0.8.0
  cache_embeddings: true
  cache_retrieval: true
  cache_reranking: true
  cache_llm: true
```

Usage :

```python
from cheragh import RAGEngine

engine = RAGEngine.from_config("rag.yaml")
response = engine.ask("Que dit le document ?")
print(response.metadata["cache"])
```

## Usage direct

```python
from cheragh.cache import SQLiteCache

cache = SQLiteCache(".cheragh/cache.sqlite", default_ttl=3600, serializer="json")
cache.set("key", {"answer": "..."})
print(cache.get("key"))
print(cache.stats().to_dict())
```

## Wrappers de composants

```python
from cheragh.cache import (
    CachedEmbeddingModel,
    CachedRetriever,
    CachedReranker,
    CachedLLMClient,
)
```

Ces wrappers permettent de mettre en cache indépendamment chaque étape du pipeline RAG.

## Invalidation

```python
cache.invalidate_namespace("embeddings")
cache.invalidate_namespace("retrieval")
cache.clear()
```

## Redis

```yaml
cache:
  enabled: true
  backend: redis
  redis_url: redis://localhost:6379/0
  ttl: 3600
```

`RedisCache` est importé paresseusement : le package continue à fonctionner sans installer `redis` tant que ce backend n'est pas utilisé.


## Note sécurité v0.8.0

Les caches persistants SQLite et Redis utilisent désormais `serializer="json"` par défaut. Ce mode évite l'exécution de code lors de la lecture d'entrées de cache et prend en charge les types courants du projet, dont `Document` et les listes de vecteurs.

Pour conserver la compatibilité avec des objets Python arbitraires, utilisez explicitement un pickle signé :

```yaml
cache:
  enabled: true
  backend: sqlite
  path: .cheragh/cache.sqlite
  serializer: signed-pickle
  secret_key: ${CHERAGH_CACHE_SECRET}
  allow_pickle: true
```

Le pickle non signé est bloqué sur les backends persistants sauf opt-in explicite `allow_unsigned_pickle: true`, réservé aux caches locaux jetables et entièrement fiables.
