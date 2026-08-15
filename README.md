# Cheragh

<p align="center">
  <img width="1456" alt="Cheragh" src="https://github.com/user-attachments/assets/4cad2b57-04bb-46f7-a26f-21cf670ec14b" />
</p>

<p align="center">
  <a href="https://github.com/Ai-Cyr/Cheragh/actions/workflows/tests.yml"><img alt="CI" src="https://github.com/Ai-Cyr/Cheragh/actions/workflows/tests.yml/badge.svg" /></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" />
  <a href="LICENSE"><img alt="Licence MIT" src="https://img.shields.io/badge/Licence-MIT-green.svg" /></a>
</p>

Cheragh est une boîte à outils Python composable pour construire, évaluer et exploiter des pipelines RAG : ingestion, chunking, retrieval sparse/dense/hybride, reranking, citations, traces, configuration, CLI et API FastAPI. Il fonctionne localement avec des composants déterministes et accepte des fournisseurs externes pour les embeddings, les LLM et les vector stores.

> [!IMPORTANT]
> Le projet est globalement en **bêta**. Le noyau RAG, le retrieval classique et l'évaluation retrieval sont stables. Les architectures inspirées de publications sont des briques bêta ou expérimentales, pas des reproductions complètes des articles.

## Points forts

- noyau léger : Python 3.10+, NumPy et Pydantic ;
- ingestion locale de texte, Markdown, HTML, JSON, CSV, YAML et XML, avec PDF et DOCX en option ;
- BM25, recherche dense et hybride, filtres metadata, reranking et compression ;
- index local inspectable et incrémental, ou adaptateurs FAISS, Chroma et Qdrant ;
- réponses structurées avec sources, citations, avertissements et trace d'exécution ;
- configuration YAML/JSON validée, CLI, serveur FastAPI et évaluation ;
- ACL, isolation tenant/collection et cache sûr par défaut ;
- catalogue machine-readable de 42 techniques avec statut et limites.

## Installation

Installation depuis le dépôt :

```bash
git clone https://github.com/Ai-Cyr/Cheragh.git
cd Cheragh
python -m pip install .
```

Pour travailler sur le code :

```bash
python -m pip install -e ".[dev]"
```

Les intégrations restent optionnelles :

| Extra | Usage |
| --- | --- |
| `local`, `rerank` | embeddings Sentence Transformers et reranking local |
| `openai`, `cohere`, `voyage`, `anthropic`, `litellm` | fournisseurs d'embeddings ou de génération |
| `pdf`, `docx`, `config` | chargeurs documentaires et YAML |
| `faiss`, `chroma`, `qdrant`, `redis` | stockage vectoriel et cache |
| `learned-retrieval`, `multimodal`, `raptor`, `bm25` | techniques spécialisées |
| `fastapi` | serveur HTTP avec Uvicorn |
| `all` | toutes les intégrations d'exécution, hors outils de développement |

Exemple :

```bash
python -m pip install ".[local,openai,qdrant,pdf,docx]"
```

Les modèles locaux sont téléchargés à leur première utilisation. Les fournisseurs hébergés nécessitent leurs propres identifiants, par exemple `OPENAI_API_KEY`.

## Démarrage rapide en Python

Cet exemple est autonome, déterministe et ne nécessite ni clé API ni téléchargement de modèle :

```python
from cheragh import Document, HashingEmbedding, RAGEngine, StaticLLMClient

documents = [
    Document(
        "Le RAG combine recherche documentaire et génération.",
        doc_id="rag",
        metadata={"topic": "retrieval"},
    ),
    Document(
        "Cheragh expose une API Python, une CLI et un serveur FastAPI.",
        doc_id="cheragh",
        metadata={"topic": "project"},
    ),
]

engine = RAGEngine.from_documents(
    documents,
    embedding_model=HashingEmbedding(dimension=256),
    llm_client=StaticLLMClient(
        "Cheragh combine retrieval et génération dans une API composable. [source: rag]"
    ),
    retriever_type="hybrid",
    top_k=2,
)

response = engine.ask("Comment Cheragh utilise-t-il le RAG ?")
print(response.answer)
for source in response.sources:
    print(source.doc_id, source.score)
```

`StaticLLMClient` sert ici de double de test pour exécuter tout le parcours hors ligne ; le client par défaut est extractif. `HashingEmbedding` est un baseline lexical déterministe, pas un encodeur sémantique entraîné. Pour un usage réel, branchez un LLM et remplacez l'embedder par `SentenceTransformerEmbedding`, `OpenAIEmbedding`, `CohereEmbedding`, `VoyageEmbedding` ou votre propre implémentation des protocoles publics.

## Parcours CLI

Indexer un corpus puis l'interroger :

```bash
cheragh index ./docs --output .cheragh_index
cheragh ask "Résume le corpus" --index .cheragh_index --json
cheragh inspect-index --index .cheragh_index
```

L'indexation est incrémentale par défaut. Le mode local utilise `HashingEmbedding` ; l'embedder et sa dimension sont ensuite dérivés et vérifiés depuis le manifeste.

Commandes disponibles :

| Commande | Rôle |
| --- | --- |
| `cheragh init` | créer un fichier `rag.yaml` de départ |
| `cheragh validate-config` | valider et normaliser une configuration |
| `cheragh index` | indexer un chemin ou une configuration |
| `cheragh ask` | interroger une configuration ou un index local |
| `cheragh eval` | évaluer le retrieval sur un dataset JSONL |
| `cheragh inspect-index` | inspecter le manifeste d'un index |
| `cheragh doctor` | vérifier l'installation et les dépendances optionnelles |
| `cheragh techniques` | consulter le catalogue de techniques |
| `cheragh serve` | lancer l'API FastAPI |

`index` accepte exactement un chemin ou `--config`. `ask` accepte au plus un `--config` ou `--index` et utilise `.cheragh_index` par défaut. `serve` exige exactement une de ces deux sources. Utilisez `--help` sur chaque commande pour les options détaillées.

## Configuration validée

Configuration locale minimale :

```yaml
ingestion:
  path: ./docs
  chunk_size: 800
  chunk_overlap: 120

embedding:
  provider: hashing
  dimension: 384

retriever:
  type: memory
  top_k: 5

vectorstore:
  path: ./.cheragh_index

generation:
  provider: extractive

strict_grounding: true
require_citations: false

indexing:
  incremental: true
  use_lock: true
```

Les chemins relatifs sont résolus depuis le dossier du fichier de configuration :

```bash
cheragh validate-config rag.yaml
cheragh index --config rag.yaml
cheragh ask "Quelle est la décision principale ?" --index .cheragh_index --json
```

La même configuration peut être chargée depuis Python avec `RAGEngine.from_config("rag.yaml")`, ou indexée avec `index_from_config("rag.yaml")`.

## Contrat de l'API

La frontière publique stable s'appuie sur `Document`, `Chunk`, `Source`, `RAGResponse` et les protocoles `RetrieverProtocol`, `EmbeddingProtocol`, `LLMProtocol` et `RerankerProtocol`.

`RAGEngine` fournit trois constructeurs principaux :

- `RAGEngine.from_documents(...)` pour des documents déjà chargés ;
- `RAGEngine.from_path(...)` pour ingérer un fichier ou un dossier ;
- `RAGEngine.from_config(...)` pour une configuration validée.

Une réponse contient la requête, la réponse, les sources, les documents récupérés, les citations, les avertissements, le score de grounding, les affirmations non sourcées, la validation de citations, les metadata et la trace. Le prompt n'est exposé par `to_dict()` que si `include_prompt=True`.

Les appels disponibles sont `ask`, `aask`, `stream`, `astream` et `stream_with_response`. Pour récupérer le résultat structuré après un flux texte :

```python
stream = engine.stream_with_response("Explique le RAG")
print("".join(stream))
print(stream.response.sources)  # disponible après consommation du flux
```

Le paramètre `top_k` suit le même contrat partout : entier strictement positif, hors booléens.

## Indexation et évaluation

`MemoryVectorStore.save()` produit un snapshot local composé de `manifest.json`, `documents.jsonl` et `embeddings.npy`. L'indexation CLI ajoute `index_manifest.json` pour suivre les fichiers, les options de chunking et les mises à jour incrémentales.

L'évaluation retrieval inclut `hit_rate@k`, `mrr`, `precision@k`, `recall@k`, `ndcg@k` et `context_precision@k`. Les API principales sont `evaluate_retrieval(...)`, `RetrievalExample` et `evaluate_pipeline(...)`.

## Techniques et maturité

Le catalogue intégré est la source de vérité :

```bash
cheragh techniques list
cheragh techniques list --status experimental --available
cheragh techniques show self-rag
```

| Statut | Couverture actuelle |
| --- | --- |
| **Stable** · 6 | RAG naïf, chunking récursif, BM25, dense, hybride, évaluation retrieval |
| **Bêta** · 12 | chunking sémantique/hiérarchique, reranking/RRF, compression, parent-child, multi-hop, fédéré, conversationnel, SQL, ACL, évaluation génération |
| **Expérimental** · 20 | HyDE/HyQE/RAG-Fusion, CRAG, Self-RAG, Agentic RAG, RAPTOR, GraphRAG-lite, FLARE, SPLADE, ColBERT, multimodal et autres variantes |
| **Planifié** · 4 | Community GraphRAG, ColPali, Temporal RAG, entraînement retrieval-aware |

Quelques limites importantes :

- SPLADE et ColBERT utilisent des calculs exacts en mémoire, sans index distribué ou ANN multivecteur compressé ;
- Self-RAG couvre l'orchestration d'inférence, pas l'entraînement avec reflection tokens ;
- RAPTOR et GraphRAG-lite sont des baselines pédagogiques, pas des implémentations complètes des publications ;
- Agentic RAG exécute une boucle bornée avec des outils explicitement enregistrés ;
- le multimodal actuel couvre le texte et les images locales, avec CLIP en option.

Consultez [la note de version 1.1](docs/release_v110.md) pour les contrats détaillés.

## Sécurité et production

- ACL et isolation tenant/collection peuvent fonctionner en mode fail-closed ;
- le chargement du cache historique `pickle` est désactivé par défaut ;
- les prompts sont exclus des traces par défaut ;
- l'endpoint HTTP `POST /index` est désactivé par défaut et doit être borné avec `--index-root` s'il est activé ;
- l'indexation locale exclut son propre output et utilise un verrou d'écriture ;
- `strict_grounding` et la validation de citations sont des garde-fous déterministes, pas une garantie contre les hallucinations.

`MemoryVectorStore` convient aux prototypes et aux corpus modestes. Pour une charge importante, utilisez de vrais embeddings sémantiques, un vector store persistant, du reranking et des seuils d'évaluation adaptés. Voir le [guide de production](docs/production.md).

## Documentation

- [Démarrage rapide](docs/quickstart.md)
- [Guide de production](docs/production.md)
- [Architectures RAG](docs/architectures_v05.md) et [architectures avancées](docs/architectures_v06.md)
- [Structured et Enterprise RAG](docs/enterprise_v07.md)
- [Sécurité et RAG moderne — v1.1](docs/release_v110.md)
- [Historique des versions](CHANGELOG.md)

## Développement

```bash
python -m pip install -e ".[dev]"
ruff check .
mypy --no-incremental
pytest
```

Le package est typé (`py.typed`) et testé sur Python 3.10 à 3.13. Les intégrations lourdes restent optionnelles afin que le noyau puisse être testé sans service externe.

## Licence

Cheragh est distribué sous licence [MIT](LICENSE).
