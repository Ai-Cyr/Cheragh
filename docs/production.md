# Exploiter Cheragh en production

Ce guide décrit une base d'exploitation pour Cheragh 1.4.0. Le projet reste en
bêta : validez chaque combinaison de fournisseur, modèle, vector store et corpus
avec vos propres jeux d'évaluation avant de traiter du trafic réel.

## Responsabilités de la plateforme

Cheragh fournit un moteur RAG, une CLI et une API FastAPI. Il ne remplace pas :

- un gestionnaire de secrets ou une PKI ;
- une passerelle d'API avec identité, quotas et protection anti-abus ;
- la sauvegarde du corpus, du vector store et des journaux ;
- la supervision des fournisseurs LLM/embeddings ;
- une politique de sécurité contre l'exfiltration et l'injection de prompt.

`strict_grounding`, les ACL et la validation des citations réduisent certains
risques, mais ne prouvent ni la véracité d'une réponse ni l'absence de fuite.

## Construire et vérifier le paquet

Les dépendances de build sont figées dans `pyproject.toml`. La CI construit la
wheel et la source distribution, exécute `twine check`, puis installe chacun des
deux artefacts dans un environnement vierge. Reproduisez ce contrôle avant une
publication :

```bash
python -m venv .venv-release
. .venv-release/bin/activate
python -m pip install "pip==26.2.1" "build==1.5.0" "twine==7.0.0"
python -m build
python -m twine check dist/*
python -m pip install dist/*.whl
python -m pip check
cheragh --help
sha256sum dist/* > dist/SHA256SUMS
```

Publiez exactement les artefacts validés par la CI ; ne reconstruisez pas la
wheel entre validation et publication. Conservez leurs sommes SHA-256 avec les
informations de version et de provenance du build.

### Verrouiller les dépendances d'un déploiement

Les bornes de versions de `pyproject.toml` protègent contre les changements de
version majeure, mais elles ne constituent pas un lockfile. Générez un verrou
par plateforme et par ensemble d'extras, puis révisez-le à chaque mise à jour :

```bash
python -m pip install "pip-tools>=7,<8"
python -m piptools compile \
  --extra fastapi --extra config --extra qdrant \
  --generate-hashes \
  --output-file requirements.lock \
  pyproject.toml
python -m pip install --require-hashes -r requirements.lock
python -m pip check
```

La CI exécute `pip-audit` sur les dépendances du noyau. Auditez également le
lockfile correspondant aux extras réellement déployés :

```bash
python -m pip install "pip-audit==2.10.1"
python -m pip_audit --strict --progress-spinner=off -r requirements.lock
```

Un audit de dépendances ne détecte pas les vulnérabilités de votre code, des
modèles ou du système de base. Ajoutez l'analyse de code, d'image et de secrets
imposée par votre organisation, avec une procédure documentée de traitement des
exceptions.

## Image Docker

Le `Dockerfile` construit d'abord une wheel, puis installe uniquement cette
wheel et les extras d'exécution dans une image séparée. Le processus final :

- utilise Python `3.12.14-slim-bookworm` par défaut ;
- s'exécute avec l'UID/GID non-root `10001:10001` ;
- ne contient ni dépôt Git, ni tests, ni exemples, ni outils de build ;
- expose un healthcheck HTTP sur `/ready` ;
- désactive l'indexation HTTP dans la commande par défaut.

Construction locale :

```bash
docker build --pull --tag cheragh:1.4.0 .
docker image inspect cheragh:1.4.0 --format '{{.Config.User}}'
```

Le tag de base fixe la version Python, mais un registre peut republier un tag.
Pour un déploiement immuable, résolvez le digest multi-architecture approuvé par
votre registre, puis transmettez la référence complète :

```bash
docker build \
  --build-arg 'PYTHON_IMAGE=python:3.12.14-slim-bookworm@sha256:DIGEST_APPROUVE' \
  --tag registry.example/cheragh:1.4.0 .
```

Le build accepte aussi `CHERAGH_EXTRAS`; la valeur par défaut est
`fastapi,config`. Gardez uniquement les intégrations nécessaires. Les versions
directes utilisées par l'image sont contraintes dans `docker/constraints.txt`.
Pour une reproductibilité complète, remplacez ce fichier dans votre pipeline
par le lockfile à hashes validé pour la plateforme cible.

Le `docker-compose.yml` démarre uniquement Cheragh : le serveur par défaut lit
un `MemoryVectorStore` local et ne consomme pas Qdrant. Pour utiliser Qdrant,
construisez explicitement l'extra `qdrant`, fournissez une configuration qui le
référence, puis gérez le service Qdrant séparément. Ne lancez pas un vector store
inutilisé en supposant obtenir une architecture haute disponibilité.

### Exécution durcie

Préparez l'index avant le démarrage et rendez `/data` accessible à l'UID 10001.
Injectez la clé depuis le gestionnaire de secrets de la plateforme, jamais dans
le Dockerfile, le dépôt, une ligne de commande en clair ou un fichier versionné.

```bash
export CHERAGH_API_KEY="$(commande-du-gestionnaire-de-secrets)"
docker run --detach \
  --name cheragh \
  --publish 127.0.0.1:8000:8000 \
  --env CHERAGH_API_KEY \
  --mount type=bind,src="$PWD/data",dst=/data,readonly \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --user 10001:10001 \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  --pids-limit 256 \
  --memory 2g \
  --cpus 2 \
  --restart unless-stopped \
  cheragh:1.4.0
```

Le profil Compose applique les mêmes principes, exige `CHERAGH_API_KEY` et ne
publie l'API que sur la boucle locale :

```bash
export CHERAGH_API_KEY="$(commande-du-gestionnaire-de-secrets)"
install -d -m 0750 data
sudo chown 10001:10001 data
docker compose up --build --detach
```

Avec Docker rootless ou un remappage d'UID, remplacez `10001:10001` par les
identifiants hôte correspondant au processus du conteneur. Vérifiez les droits
sur un index préparé avant le démarrage ; ne rendez pas le volume accessible à
tous pour contourner une erreur de permission.

Placez le reverse proxy TLS sur le même hôte ou adaptez explicitement le réseau
Compose. Ne remplacez pas le binding `127.0.0.1` par `0.0.0.0` sans passerelle.

Le montage `/data` doit contenir un index complet et cohérent. Une image qui
utilise `--config` doit recevoir la configuration en lecture seule et conserver
ses sorties dans un volume explicitement inscriptible.

Ne placez aucune clé dans un argument de build : les arguments et couches
peuvent rester visibles dans l'historique de l'image. Inspectez et signez
l'image, générez un SBOM, analysez-la, puis déployez-la par digest.

## Frontière HTTP, TLS et authentification

Uvicorn n'est pas la frontière Internet recommandée. Placez l'API derrière un
reverse proxy ou une passerelle qui :

- termine TLS avec une version et des suites cryptographiques approuvées ;
- authentifie les utilisateurs/services avec OIDC, mTLS ou l'identité cloud ;
- applique quotas, rate limiting, taille maximale de requête et filtrage réseau ;
- impose des délais de connexion, d'en-tête et d'inactivité adaptés au streaming ;
- limite CORS aux origines nécessaires et journalise les décisions d'accès ;
- masque `/docs`, `/openapi.json` et `/stats` s'ils ne sont pas requis.

`CHERAGH_REQUIRE_AUTH=true` fait échouer le démarrage lorsqu'aucune clé valable
n'est configurée ; l'image et le profil Compose l'activent par défaut.
`CHERAGH_API_KEY` active une clé partagée via l'en-tête `X-API-Key` pour
`/ask`, `/stream`, `/index` et `/stats`. C'est une protection minimale, pas une
identité utilisateur, un RBAC ni une rotation automatique. Utilisez une valeur
aléatoire longue, comparez et faites tourner la clé via la passerelle, et
réservez cette protection à un réseau de confiance.

`GET /health` reste volontairement public et ne vérifie que le processus.
`GET /ready` confirme que le moteur est initialisé et accepte une
`readiness_check` injectable avec `create_app()`. La vérification par défaut
n'effectue pas d'appel profond aux fournisseurs : injectez un contrôle borné des
dépendances critiques (LLM, embeddings, vector store) avant de router du trafic.
Ne mettez jamais de secret ou de contenu utilisateur dans une réponse de santé.

Les limites de corps, concurrence, timeout et durée de streaming sont bornées au
niveau du processus. Consultez le [guide du serveur HTTP](production_server.md)
pour leurs valeurs, le modèle de timeout et les en-têtes de sécurité.

L'endpoint `POST /index` est désactivé par défaut. S'il est indispensable :

1. exécutez l'indexation dans un worker séparé plutôt que dans le service public ;
2. utilisez explicitement `--enable-indexing --index-root /data` ;
3. restreignez l'accès à une identité d'administration et à un réseau privé ;
4. scannez les documents, bornez leur taille et refusez les formats inutiles ;
5. ne laissez qu'un writer modifier un index local.

## Secrets et données sensibles

Stockez les clés LLM, embeddings et vector store dans le gestionnaire de secrets
de l'orchestrateur. Accordez une identité et des permissions minimales à chaque
environnement. Séparez développement, préproduction et production, puis
révoquez rapidement toute clé exposée.

Par défaut, gardez :

```yaml
strict_grounding: true
require_citations: true
flag_unsourced_sentences: true
observability:
  enabled: true
  trace_include_prompt: false
```

Les questions, chunks, réponses, citations et métadonnées peuvent contenir des
données personnelles ou des secrets. Avant export de traces :

- définissez une liste de champs autorisés et une politique de masquage ;
- séparez les journaux d'audit des traces de débogage ;
- chiffrez en transit et au repos, avec rotation des clés ;
- fixez une durée de rétention et un processus de suppression ;
- contrôlez l'accès aux sauvegardes et aux outils d'observabilité.

Autorisez uniquement les sorties réseau nécessaires. Considérez les documents
et instructions récupérés comme non fiables : ils ne doivent jamais pouvoir
modifier les outils autorisés, les secrets, les ACL ou la configuration système.

## Configuration de référence

Commencez par `examples/presets/production_v100.yaml` ou
`examples/presets/production_hybrid.yaml`, puis versionnez une configuration
sans secret. Un profil raisonnable conserve retrieval large, reranking,
compression, citations et tracing sans prompts :

```yaml
ingestion:
  path: ./docs
  chunk_size: 900
  chunk_overlap: 150
  max_file_size_mb: 50

embedding:
  provider: openai
  model: text-embedding-3-small
  api_key: ${OPENAI_API_KEY}
  timeout_seconds: 60
  max_retries: 2

retriever:
  type: hybrid
  top_k: 6
  alpha: 0.55
  filters: {}
  tokenizer:
    strip_accents: true
    keep_hyphenated: true
    ngram_range: [1, 2]
    use_default_stopwords: true

reranker:
  enabled: true
  provider: keyword
  first_stage_top_k: 40

compression:
  enabled: true
  type: default

query:
  enabled: true
  type: multi-query

generation:
  provider: openai
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
  timeout_seconds: 60
  max_retries: 2

strict_grounding: true
require_citations: true
flag_unsourced_sentences: true
trace_enabled: true
min_score: 0.03

cache:
  enabled: true
  backend: sqlite
  path: .cheragh/cache.sqlite
  serializer: json
  ttl: 3600

observability:
  enabled: true
  trace_export_path: .cheragh/traces.jsonl
  trace_include_prompt: false

indexing:
  incremental: true
  use_lock: true
```

Validez et construisez l'index depuis la même configuration :

```bash
cheragh validate-config configs/production.yaml
cheragh index --config configs/production.yaml --dry-run
cheragh index --config configs/production.yaml
cheragh inspect-index --index /data/.cheragh_index
```

Les chemins relatifs sont résolus depuis le fichier de configuration. Un
`output=` explicite remplace `vectorstore.path`. L'indexation par configuration
locale cible le format `MemoryVectorStore`; pour FAISS, Chroma ou Qdrant,
utilisez le flux d'ingestion propre au fournisseur. Le cache mémoire applique
un LRU de 10 000 entrées par défaut ; ajustez `cache.max_entries` au volume et à
la taille réelle des valeurs, car cette borne compte les entrées, pas les octets.

Une valeur scalaire exactement égale à `${NOM_VARIABLE}` est résolue depuis
l'environnement avant validation. Les interpolations partielles sont refusées :
gardez donc les secrets séparés du texte et des URL. `cheragh validate-config
--json` masque les champs sensibles ; sa sortie reste une donnée opérationnelle
à protéger. Les timeouts de génération et d'embeddings, ainsi que les retries
pris en charge par chaque fournisseur, doivent rester bornés selon le budget de
latence et les quotas du fournisseur.

## Dimensionnement et montée en charge

Mesurez séparément ingestion, retrieval, reranking, packing et génération. Les
limites habituelles sont la mémoire du modèle/index local, la latence du LLM et
les quotas des fournisseurs.

- Un index local est chargé dans chaque processus. Évitez plusieurs workers
  Uvicorn lourds dans le même conteneur ; préférez un processus par conteneur et
  des réplicas horizontaux.
- Pour un grand corpus ou plusieurs réplicas, utilisez un vector store externe
  avec haute disponibilité, timeouts, retries bornés et limites de connexion.
- Rendez les clés de cache dépendantes du tenant, de la collection, des ACL, du
  modèle et de la version d'index. Ne partagez jamais une réponse entre tenants.
- Mettez en file les indexations et appliquez une admission control lorsque les
  quotas LLM approchent de leur limite.
- Bornez `top_k`, la taille des documents, le budget de contexte et le nombre
  d'étapes agentiques ou multi-hop.

Testez charge nominale, pic, panne partielle et reprise. Les objectifs doivent
couvrir au minimum p50/p95/p99, taux d'erreur, saturation, coût par requête et
fraîcheur de l'index.

## Sauvegardes et restauration

Le corpus original est la source de vérité ; l'index reste un artefact
reconstructible. Définissez néanmoins RPO et RTO pour les deux.

Pour un index local :

1. arrêtez ou verrouillez le writer ;
2. copiez atomiquement données, manifeste et métadonnées de version ;
3. chiffrez la sauvegarde et stockez-la hors du domaine de panne principal ;
4. vérifiez les hashes et la rétention ;
5. restaurez régulièrement dans un environnement isolé ;
6. exécutez `cheragh inspect-index` puis le jeu d'évaluation avant bascule.

Pour Qdrant, Chroma, FAISS ou un stockage managé, utilisez le mécanisme de
snapshot cohérent du fournisseur. Sauvegardez aussi les ACL, filtres, versions
de modèles, paramètres de chunking et configuration nécessaires à la
reconstruction. Ne copiez pas un index local pendant une écriture active.

Déployez les nouveaux index sous un identifiant immuable, testez-les, puis
basculez le trafic. Conservez l'index précédent pendant la fenêtre de rollback.

## Observabilité et alertes

Exportez les traces vers un stockage borné et contrôlé. Le JSONL local convient
au diagnostic d'une instance, pas à l'agrégation multi-réplicas sans collecteur.
Surveillez :

- disponibilité, latence et codes HTTP ;
- latence/erreurs par étape et fournisseur ;
- tokens et coût estimé ;
- documents récupérés, scores et réponses sans source, après anonymisation ;
- hit-rate et évictions du cache ;
- âge, version et volume de l'index ;
- saturation CPU, mémoire, disque, connexions et quotas.

Alertez sur une dégradation soutenue plutôt que sur un événement isolé. Testez
la coupure du LLM, du vector store et du réseau, ainsi que l'expiration d'une clé.

## Gate qualité avant déploiement

Rejetez un build qui régresse sur un jeu versionné et représentatif :

- `recall@10` pour la couverture ;
- `ndcg@10` pour le classement ;
- `context_precision@5` pour la propreté du contexte ;
- couverture et exactitude des citations ;
- support claim-level et contradictions pour la génération ;
- latence, consommation de tokens et coût.

```python
from cheragh import RetrievalExample, evaluate_retrieval

examples = [
    RetrievalExample("préavis contrat alpha", {"contract-alpha"}),
    {"query": "politique sécurité SQLite", "expected_doc_ids": ["sqlite-hardening"]},
]

result = evaluate_retrieval(examples, engine.retriever, top_k=10)
assert result.metrics["recall@10"] >= 0.85
assert result.metrics["ndcg@10"] >= 0.75
```

Conservez les résultats avec le SHA du code, la version d'index, les modèles et
la configuration. Déployez progressivement, observez, puis augmentez le trafic.

## Checklist de mise en production

- [ ] Wheel/sdist construites une seule fois, vérifiées et conservées avec hashes.
- [ ] Dépendances et extras verrouillés, audités et soumis à une politique de mise à jour.
- [ ] Image de base et image finale déployées par digest, SBOM et scan approuvés.
- [ ] Conteneur non-root, filesystem en lecture seule, capacités supprimées.
- [ ] TLS, identité, quotas et limites appliqués à la passerelle.
- [ ] Secrets injectés par la plateforme et rotation testée.
- [ ] Indexation HTTP désactivée ou isolée sur un worker administratif.
- [ ] Isolation tenant/ACL testée en mode fail-closed.
- [ ] Traces sans prompts par défaut, rétention et chiffrement configurés.
- [ ] Sauvegarde et restauration testées selon les RPO/RTO.
- [ ] Gate retrieval/génération/latence/coût validée sur le corpus cible.
- [ ] Déploiement progressif et rollback vers l'image et l'index précédents testés.
