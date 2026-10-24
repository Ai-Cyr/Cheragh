# Semantic, late et contextual chunking

Ces trois méthodes améliorent des étapes différentes de l'indexation. Le choix
se fait avec `ingestion.chunking_strategy`, `--chunking-strategy` ou le paramètre
Python `chunking_strategy`. L'interface présente les mêmes choix et indique
les capacités manquantes du moteur configuré.

| Méthode | Opération | Dépendances |
| --- | --- | --- |
| `semantic` | Compare les embeddings de phrases voisines et coupe aux ruptures de similarité. | Encodeur sémantique local ou fournisseur d'embeddings. |
| `late` | Encode le document entier une fois, puis moyenne les vecteurs de tokens de chaque passage. | Encodeur Transformers à tokenizer rapide, avec fenêtre suffisante. |
| `contextual` | Demande au LLM de situer chaque passage dans son document, puis embedde contexte + passage. | LLM génératif et modèle d'embedding. |

`recursive`, `sentence`, `paragraph` et `token` restent disponibles sans modèle.
`token` compte par défaut les mots séparés par des espaces ; ce ne sont pas les
tokens exacts d'un modèle. Le chunker Python accepte aussi un tokenizer rapide.
Pour les autres méthodes, taille et chevauchement s'expriment en caractères.

## Découpage sémantique

Installer `cheragh[local,config]` pour un encodeur Sentence Transformers local.
Exemple `rag.yaml`, avec les chemins relatifs au fichier de configuration :

```yaml
ingestion:
  path: ./corpus
  chunking_strategy: semantic
  chunk_size: 1000
  chunk_overlap: 100
  chunking_options:
    buffer_size: 1
    breakpoint_percentile: 90
    embedding_batch_size: 32
embedding:
  provider: sentence-transformers
  model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
retriever:
  type: memory
generation:
  provider: extractive
```

```bash
cheragh index --config rag.yaml --output ./index-semantic
```

`buffer_size` ajoute les phrases voisines au texte encodé pour décider d'une
frontière. Sans percentile, `breakpoint_threshold` (défaut `0.75`) est le seuil
de similarité cosinus sous lequel une rupture est proposée. Avec un percentile,
le seuil s'adapte aux distances adjacentes du document. Les contraintes
`min_sentences`, `min_chunk_size` et `chunk_size` bornent les groupes. Le
chevauchement s'applique au repli utilisé pour une phrase ou un groupe trop long,
pas à chaque frontière sémantique.

Les batchs, le nombre de phrases (`max_sentences`, défaut 10 000) et la longueur
du contexte encodé (`max_buffer_chars`, défaut 32 000) sont bornés. Configurer ces
limites selon celles du fournisseur ; une limite en caractères ne constitue
pas une garantie sur son nombre de tokens. La qualité des frontières dépend
de l'encodeur, de la langue et des seuils ; `HashingEmbedding` est refusé par
la sélection intégrée de cette méthode.

## Late chunking

Installer `cheragh[late-chunking,config]`. Dans la configuration précédente,
remplacer les blocs ingestion et embedding par :

```yaml
ingestion:
  path: ./corpus
  chunking_strategy: late
  chunk_size: 500
  chunk_overlap: 60
embedding:
  provider: transformers-token
  model: sentence-transformers/all-MiniLM-L6-v2
  device: cpu
  max_tokens: 512
```

Cet exemple convient à des documents courts. Choisir un encodeur à contexte
long pour de longs documents, fixer `max_tokens` à sa capacité réelle et épingler
`embedding.revision` à une révision de modèle pour un déploiement reproductible.
Le budget inclut les tokens spéciaux. Dépasser la capacité configurée, celle du
tokenizer ou celle du modèle provoque une erreur avant l'inférence : aucun
document n'est tronqué silencieusement.

L'encodeur doit exposer `last_hidden_state` et des offsets de caractères via
un tokenizer rapide. Aucun code distant de modèle n'est exécuté automatiquement.
Une architecture ou un checkpoint demandant un pooling, une projection ou des
instructions spécifiques doit être adapté et évalué avant utilisation.

```python
from cheragh import Document, LateChunker, TransformersTokenEmbedding
from cheragh.vectorstores import MemoryVectorStore

embedder = TransformersTokenEmbedding(max_tokens=512)
chunks, vectors = LateChunker(
    embedder, chunk_size=120, chunk_overlap=20,
).chunk_and_embed([Document("Berlin est une ville allemande. Elle est la capitale du pays.")])
store = MemoryVectorStore(embedder)
store.add_embeddings(chunks, vectors)
results = store.similarity_search("Quelle ville est la capitale ?", top_k=2)
```

Le pooling exclut les tokens spéciaux et les tokens de remplissage, puis
normalise la moyenne. La requête utilise le même encodeur et le même pooling.
Les extraits suivent les phrases sous une limite de caractères ; un token
intersectant deux extraits participe aux deux moyennes. Les offsets et les ACL
du document d'origine sont conservés.

## Contextual chunking

Utiliser `chunking_strategy: contextual` avec un modèle d'embedding et un vrai
LLM configuré. Exemple de blocs pour un fournisseur OpenAI, nécessitant
`cheragh[openai,config]` et `OPENAI_API_KEY` dans l'environnement :

```yaml
ingestion:
  path: ./corpus
  chunking_strategy: contextual
  chunk_size: 800
  chunk_overlap: 100
  chunking_options:
    max_document_chars: 20000
    max_context_chars: 1000
    max_chunks_per_document: 256
    embedding_batch_size: 32
embedding:
  provider: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
generation:
  provider: openai
  model: gpt-4.1-mini
  api_key: ${OPENAI_API_KEY}
  timeout_seconds: 60
retriever:
  type: memory
```

Le LLM reçoit le document complet et chaque passage, ce qui représente un appel
par passage. Les limites documentaires et le nombre de passages sont validés
avant les appels. Le contexte généré doit être non vide et respecter sa limite ;
une réponse excessive provoque une erreur. Régler aussi les budgets de tokens,
timeouts et quotas du fournisseur : les limites de caractères ne les remplacent pas.

Seul le texte envoyé à l'embedder est enrichi. `Document.content` reste l'extrait
original pour les citations ; le contexte est marqué comme généré dans
`metadata.contextual_context` et `contextual_context_generated`. Le document
complet doit appartenir au même périmètre d'autorisation que ses passages.
Le contexte peut contenir des erreurs du LLM : mesurer la fidélité et le rappel
sur le corpus. Cette méthode implémente l'enrichissement dense, pas toute la
recette de contextual retrieval combinant BM25 et reranking.

## Intégration et persistance

`RAGEngine.from_documents(..., chunk=True, retriever_type="memory", ...)`,
`RAGEngine.from_path(...)` et `index_path(...)` acceptent les trois stratégies.
Passer explicitement `embedding_model` et, pour `contextual`, `llm_client`.
`chunking_options` contient les paramètres propres à la stratégie. Les modes
late/contextual sont intégrés au store mémoire (`memory` ou `vector`) ; les
autres retrievers sont refusés avant le calcul. Pour intégrer un autre backend,
insérer directement les vecteurs retournés par `chunk_and_embed` dans son API.

Réencoder uniquement le contenu des passages supprimerait le résultat de
late/contextual chunking. C'est pourquoi leurs méthodes `split_documents` et
les parcours d'ingestion retournant uniquement du texte refusent cet usage.
`add_embeddings` valide le lot puis publie documents et vecteurs ensemble.
Les snapshots conservent ces vecteurs et se rechargent avec le même embedder.

L'indexation incrémentale réutilise les sources inchangées et reconstruit lorsque
la stratégie, ses options ou l'identité de l'encodeur changent. Pour contextual,
l'identité de génération est également prise en compte. La configuration
persiste seulement son empreinte, pas ses secrets. Avec un client Python opaque,
passer un `generation_fingerprint` explicite à `index_path` et le changer si le
modèle, le prompt applicatif ou ses réglages changent.

Dans l'interface, les modèles sont ceux du serveur ; la requête HTTP ne peut
pas installer ni sélectionner un fournisseur. Les choix indisponibles indiquent
leur prérequis. Les index construits avec des modèles configurés nécessitent
un raccordement par le déploiement ; l'activation immédiate reste réservée au
moteur local Hashing décrit dans le [guide de l'interface](interface.md).

Late et contextual chunking restent **expérimentaux**. Les tests incluent un
petit BERT local pour vérifier l'effet du contexte complet, le pooling et les
limites, ainsi que la conservation des vecteurs dans les index. Ils ne prouvent
pas les performances d'un checkpoint de production sur votre corpus.
