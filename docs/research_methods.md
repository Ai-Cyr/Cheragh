# Utiliser les mécanismes issus des publications

Les modes ci-dessous complètent les baselines existantes. Ils restent expérimentaux : leurs contrats logiciels sont testés, mais les performances des articles ne sont pas reproduites. La [matrice des 44 techniques](research_fidelity.md) décrit les écarts restants. Les exemples supposent des documents, encodeurs et modèles configurés par l'application.

## RAPTOR : clustering souple et parcours multi-niveau

Installer `pip install '.[raptor]'`. Le mode historique reste `greedy`; le nouveau mode active UMAP global/local, sélection du nombre de composantes GMM par BIC et appartenances multiples. Les documents ne sont pas supprimés lorsqu'aucune probabilité ne dépasse le seuil : leur meilleure composante est conservée.

```python
from cheragh import RAPTOREngine, RAPTORClusteringConfig

engine = RAPTOREngine(
    documents,
    embedding_model=semantic_embedding,
    llm_client=llm,
    clustering_mode="umap_gmm",
    clustering_config=RAPTORClusteringConfig(random_state=224),
    retrieval_mode="paper_tree",
    beam_width=4,
    traversal_budget=64,
    summary_input_token_budget=3500,
    retrieval_token_budget=2000,
    token_estimator=lambda text: len(tokenizer.encode(text)),
)
answer = engine.ask("Quels thèmes relient ces documents ?", top_k=12)
```

`paper_tree` sélectionne les nœuds par leur cosine propre et conserve les niveaux sélectionnés. `beam_width` borne les choix par étage; `top_k` borne le total, extension opérationnelle par rapport au papier. Le mode `collapsed` demeure disponible. Les groupes trop longs sont reclusterisés puis divisés si nécessaire; un document indivisible trop long déclenche une erreur au lieu d'être tronqué silencieusement. Le budget de résumé inclut son prompt; celui de retrieval borne les textes récupérés et leurs séparateurs, pas tout le prompt de réponse. Sans tokenizer injecté, le comptage reste approximatif. Le summarizer et les embeddings sémantiques sont à fournir. [RAPTOR, §3](https://arxiv.org/html/2401.18059v1).

## GraphRAG : Leiden hiérarchique et global map-reduce

Installer `pip install '.[graphrag]'`. Fournir un graphe extrait du corpus avec les identifiants des documents sources.

```python
from cheragh import (
    CommunityGraphRAGEngine, GlobalMapReduceConfig,
    LeidenCommunityDetector, LLMCommunitySummarizer,
)

engine = CommunityGraphRAGEngine(
    documents, graph=knowledge_graph, llm_client=llm,
    community_detector=LeidenCommunityDetector(max_cluster_size=10, random_seed=42),
    summarizer=LLMCommunitySummarizer(llm),
)
answer = engine.ask_global_map_reduce(
    "Quels sont les principaux thèmes ?",
    level=1,
    allowed_doc_ids=authorized_document_ids,
    config=GlobalMapReduceConfig(max_map_calls=64, max_concurrency=1),
    token_counter=lambda text: len(tokenizer.encode(text)),
)
```

`ask_global()` conserve la recherche historique. `ask_global_map_reduce()` traite tous les rapports autorisés du niveau demandé, y compris les branches plus courtes. Il mélange les rapports avec une seed, découpe sous budget, produit des réponses intermédiaires et un score d'utilité 0–100, écarte les scores nuls et réduit les réponses les mieux scorées. Dépasser `max_map_calls` échoue avant tout appel modèle. Le compteur par défaut compte les octets UTF-8; injecter le tokenizer pour exploiter précisément la fenêtre du modèle. Les budgets d'entrée et de sortie sont séparés et doivent ensemble respecter sa fenêtre.

Les rapports pré-calculés mêlant des sources autorisées et interdites sont entièrement exclus. `allowed_doc_ids` doit être calculé par le serveur; on peut aussi passer `principal` et `access_policy`. La provenance connue est recomposée à partir du graphe et des descendants. Une citation valide identifie une source; elle ne prouve pas l'exactitude sémantique. L'extraction des entités et la qualité des résumés restent à évaluer. Un contexte trop petit peut limiter le contenu d'un résumé; la couverture des communautés ne signifie pas la conservation de chaque fait. [GraphRAG, §3.1](https://arxiv.org/html/2404.16130v2).

## FLARE : probabilités de génération et requêtes masquées

Utiliser un modèle supportant les log-probabilités Chat Completions :

```python
from cheragh import FLAREPipeline, OpenAIChatClient

llm = OpenAIChatClient(model=model_name, timeout=30, max_retries=2)
pipeline = FLAREPipeline(
    retriever, llm,
    draft_generator=lambda prompt: llm.generate_with_confidence(
        prompt, max_completion_tokens=128,
    ),
    confidence_threshold=0.5,
    masking_threshold=0.5,
    max_iterations=8,
)
answer = pipeline.ask("Explique les changements de politique.")
```

Les probabilités proviennent du même appel que le brouillon. Le texte doit correspondre exactement aux tokens, y compris les caractères Unicode formés de plusieurs tokens. Toute probabilité absente ou mal alignée provoque une erreur explicite. Aucun cache mutable du dernier brouillon n'est partagé entre requêtes. Les tokens incertains déclenchent la recherche puis sont masqués; si tout disparaît, la question originale est utilisée. Les seuils de déclenchement et de masquage sont indépendants. Le fallback historique par longueur reste disponible sans `draft_generator`.

Ce mode implémente la stratégie de masquage de FLARE; il ne reproduit pas son prompt expérimental, son corpus, son modèle ni tous ses modes de génération de questions. La ponctuation et les probabilités des modèles doivent être calibrées pour le cas d'usage. [FLARE, §3.2](https://arxiv.org/html/2305.06983v2), [contrat Chat Completions](https://developers.openai.com/api/reference/resources/chat).

## Self-RAG : scores de reflection tokens

`ReflectionTokenDistribution` et `ReflectionTokenScorer` calculent les scores
normalisés de pertinence, support et utilité à partir des probabilités d'un
modèle compatible. `ReflectionTokenRetrievalGate` peut remplacer la porte
initiale du moteur existant :

```python
from cheragh import ReflectionTokenDistribution, ReflectionTokenGroup, ReflectionTokenRetrievalGate

def reflection_provider(query):
    # Retourner les vraies log-probabilités des deux tokens spéciaux,
    # obtenues au même point de génération d'un checkpoint Self-RAG.
    scores = model_initial_reflection_logprobs(query)
    return ReflectionTokenDistribution.from_logprobs(
        ReflectionTokenGroup.INITIAL_RETRIEVAL,
        scores,
        model_id=checkpoint_revision,
    )

gate = ReflectionTokenRetrievalGate(reflection_provider, threshold=0.2)
decision = gate.decide("Quelle preuve soutient cette affirmation ?")
```

Les groupes incomplets, masses nulles et valeurs invalides sont refusés.
L'adaptateur ne produit aucune probabilité à partir d'un simple texte. Le moteur
historique reste une approximation : la génération par segments, la recherche
en faisceau et l'entraînement des reflection tokens ne sont pas implémentés par
ces seuls calculs. [Self-RAG](https://arxiv.org/html/2310.11511v1).

## RAFT : supervision ancrée et export SFT

```python
from cheragh import RAFTDatasetBuilder, RAFTGeneratedAnswer

def teacher(question, oracle_documents, verified_answer):
    rationale = generate_explanation_with_exact_quotes(question, oracle_documents)
    return RAFTGeneratedAnswer(answer=verified_answer, rationale=rationale)

records = RAFTDatasetBuilder(
    oracle_probability=0.8, seed=42,
    shuffle_documents=True, context_document_count=5,
    answer_generator=teacher,
).build(examples)
rows = [{"messages": record.to_messages()} for record in records]
```

Les explications doivent contenir des citations sous la forme `##begin_quote##extrait exact##end_quote##`. Chaque citation doit appartenir à un document oracle et la réponse finale doit préserver la réponse de référence. Le générateur reçoit des copies des documents. Les oracles retirés ne figurent pas dans le contexte d'entrée, mais la supervision conserve les explications ancrées comme dans RAFT. Avec cinq documents de contexte, prévoir au moins cinq distracteurs par exemple pour les cas sans oracle. La validation vérifie les citations, pas toute la logique de l'explication. Le fine-tuning du LLM et son évaluation restent distincts. [RAFT, §3](https://arxiv.org/html/2403.10131v1).

## Entraîner les encodeurs de retrieval

Installer une version de PyTorch adaptée au matériel puis `pip install '.[training]'`.

```python
from cheragh import TorchRetrievalTrainer

trainer = TorchRetrievalTrainer(
    query_encoder, document_encoder, optimizer,
    temperature=0.1, normalize_embeddings=True, max_grad_norm=1.0,
)
metrics = trainer.fit(training_examples, epochs=3, batch_size=8, seed=42)
```

Les encodeurs reçoivent une séquence de textes et renvoient un tenseur différentiable `(batch, dimension)`. L'optimiseur doit posséder leurs paramètres. Les exemples ordinaires utilisent une loss contrastive multi-positive; les exemples distillés utilisent `T² × KL(teacher || student)` avec leur température. Les candidats restent propres à chaque question pour éviter de transformer les positifs d'autres questions en faux négatifs. Les gradients non finis empêchent l'étape d'optimisation; les modes des modules sont restaurés après entraînement. Ce trainer ne réalise pas le fine-tuning RAFT d'un LLM, l'entraînement distribué, les checkpoints ou la préparation des modèles.

## Validation avant déploiement

La CI du noyau continue de tester Python 3.10 à 3.13. Un job CPU séparé installe UMAP, graspologic et PyTorch puis exécute les tests des nouveaux mécanismes sans téléchargement de poids. Pour reproduire localement :

```bash
python -m pip install -e '.[dev,fastapi,raptor,graphrag,training]'
python -m pip check
ruff check src tests
mypy --no-incremental
pytest
python -m build
python -m twine check dist/*
```

Une mise en production doit encore être qualifiée sur le corpus réel, avec les modèles choisis, les droits d'accès effectifs et la charge visée. Le [guide de production](production.md) décrit le déploiement, les sauvegardes et le retour arrière; le [serveur](production_server.md) conserve des limites explicites sur l'interruption des appels fournisseurs et la montée en charge.
