# Qualification de production des 46 techniques

Cette matrice relie chacune des 46 entrées du [catalogue](../src/cheragh/catalog/techniques.py)
à des tests concrets et aux validations propres au déploiement. Elle distingue
la robustesse du logiciel, vérifiable sur des doubles ou des backends locaux,
de la qualité d'un modèle sur un corpus et de la capacité à supporter une charge.
Un test d'algorithme réussi ne démontre ni un benchmark de publication ni un SLO.

Les statuts du catalogue sont : 6 techniques stables, 12 en bêta et
28 expérimentales. « Expérimental » reste une limite de qualification, même
quand les entrées, les budgets, les erreurs et la provenance sont contrôlés.
Les tests référencés ci-dessous existent dans le dépôt ; leur présence ne vaut
pas preuve de leur exécution avec les modèles, dépendances et données déployés.

## Exécuter les contrôles

Depuis un environnement Python isolé, pour le noyau et le serveur :

```bash
python -m pip install -e ".[dev,fastapi]"
python -m pip check
ruff check src tests scripts
mypy --no-incremental
pytest -q -ra
python -m build
python -m twine check dist/*.whl dist/*.tar.gz
```

`-ra` rend visibles les tests ignorés. Un backend optionnel absent produit des
tests ignorés et ne peut donc pas être déclaré validé par cette exécution. Pour
chaque extra réellement déployé, exécuter aussi le job correspondant de la
[CI](../.github/workflows/tests.yml) avec ses versions de Python et de dépendances.
Les jobs CPU couvrent les backends de recherche, les modèles locaux construits
pendant les tests et les intégrations de stockage ; ils ne remplacent pas un
essai avec un checkpoint distant ou un service hébergé.

Conserver pour chaque version livrée : révision Git, versions des dépendances,
identifiants et révisions des checkpoints, commande et rapport de tests avec les
raisons des skips, jeu d'évaluation versionné, résultats de charge et empreintes
des distributions. Le [guide de production](production.md) décrit la construction,
le serveur, les sauvegardes et la restauration.

## Matrice des techniques

La colonne « Vérifications dans le dépôt » indique des fichiers représentatifs,
pas une liste exhaustive. Le contrôle indiqué dans la dernière colonne s'ajoute
aux exigences communes de la section suivante.

| ID du catalogue | Statut | Vérifications dans le dépôt | Préconditions et contrôle avant trafic |
| --- | --- | --- | --- |
| `naive-rag` | Stable | [Contrats du moteur](../tests/test_api_consistency_core.py), [streaming](../tests/test_async_stream_production.py) : `top_k`, snapshots, absence de contexte, fermeture des flux. | Choisir les embeddings et le générateur ; mesurer réponses sourcées, abstention, latence et coût sur le corpus cible. Le générateur extractif et HashingEmbedding servent de baselines locales. |
| `recursive-chunking` | Stable | [Ingestion](../tests/test_v02_features.py), [frontières tokenizers](../tests/test_retrieval_cross_review.py) : métadonnées parent et préservation du texte. | Fixer taille/chevauchement et tokenizer ; vérifier Unicode, tableaux et limites de fichiers avec les documents réels. |
| `semantic-chunking` | Bêta | [Ruptures thématiques](../tests/test_v041_chunking_routing.py), [stabilité numérique](../tests/test_retrieval_cross_review.py). | Qualifier l'encodeur, les seuils de rupture et le coût d'indexation ; mesurer le rappel après découpage. |
| `late-chunking` | Expérimental | [Vrai encodeur BERT local et pooling](../tests/test_late_chunking.py), [vecteurs pré-calculés](../tests/test_precomputed_memory_vectors.py), [index et cache](../tests/test_advanced_chunking_pipeline.py). | Installer `late-chunking` et qualifier un encodeur à offsets fiables ; fixer sa capacité réelle, rejeter les documents trop longs, conserver les vecteurs calculés et le même encodeur de requête. Le store mémoire est le chemin intégré. |
| `contextual-chunking` | Expérimental | [Contexte, limites et provenance](../tests/test_contextual_chunking.py), [index incrémental](../tests/test_advanced_chunking_pipeline.py), [capacités HTTP](../tests/test_server_chunking.py). | Qualifier embeddings et LLM, borner longueur, passages et coût ; vérifier la fidélité du contexte généré. Les citations gardent le texte original. Le chemin dense ne reproduit pas la recette BM25/reranking complète. |
| `hierarchical-chunking` | Bêta | [Métadonnées hiérarchiques](../tests/test_v041_chunking_routing.py), [reconstruction parent](../tests/test_retrieval_cross_review.py). | Préserver tenant/ACL et identités parent-enfant ; contrôler les parents assemblés et les limites de contexte. |
| `sentence-window` | Expérimental | [Fenêtre et provenance](../tests/test_v120_architecture_coverage.py), [snapshots des index](../tests/test_api_consistency_core.py). | Évaluer la segmentation dans les langues du corpus, l'expansion de contexte et la consommation mémoire ; ne pas mélanger des domaines d'autorisation dans un document. |
| `propositional` | Expérimental | [Propositions vers parent](../tests/test_v120_architecture_coverage.py), [format du propositionizer](../tests/test_final_research_integration.py). | Fournir un LLM ou TransformersPropositionizer ; contrôler atomicité, fidélité, coût d'indexation et rappel. Un prompt générique ne reproduit pas le modèle Dense X. |
| `bm25` | Stable | [BM25 sans embeddings](../tests/test_v120_bm25_rrf.py) : filtres, paramètres, `top_k`, configuration. | Ajuster tokenizer, `k1` et `b` sur des requêtes représentatives ; mesurer mémoire et temps de recherche au volume cible. |
| `dense` | Stable | [Persistance et filtres](../tests/test_v02_features.py), [stockages optionnels](../tests/test_optional_storage_production.py). | Fixer modèle, dimension et normalisation ; valider migration, restauration, filtrage et charge sur le backend choisi. L'index mémoire est exact et limité par la RAM. |
| `hybrid` | Stable | [Retrieval hybride](../tests/test_core.py), [frontières algorithmiques](../tests/test_retrieval_algorithm_boundaries.py). | Calibrer la fusion sur les scores réels et les langues servies ; vérifier rappel et latence par rapport à BM25 et dense seuls. |
| `reranking` | Bêta | [Scores et snapshots](../tests/test_v120_architecture_coverage.py), [compatibilité Sentence Transformers](../tests/test_sentence_transformers_compatibility.py). | Qualifier le checkpoint cross-encoder, les longueurs maximales, le batching et le budget de candidats ; mesurer le gain de qualité et la latence. |
| `rrf` | Bêta | [RRF canonique](../tests/test_v120_bm25_rrf.py) : déduplication, identité par source et score de rang. | Définir les sources, le nombre de candidats et la constante de fusion ; vérifier que les ACL sont appliquées à chaque source. |
| `mmr` | Expérimental | [Pertinence/diversité](../tests/test_v120_architecture_coverage.py), [candidats bornés](../tests/test_architecture_runtime_contracts.py). | Calibrer `lambda_mult` et `fetch_k` ; mesurer le compromis diversité/rappel et le coût des embeddings des candidats. |
| `splade` | Expérimental | [Scoring sparse](../tests/test_v110_learned_retrieval.py), [pooling et checkpoint local](../tests/test_learned_model_fidelity.py). | Installer `learned-retrieval`, figer un checkpoint validé ; dimensionner l'index exact en mémoire. Un index inversé externe est nécessaire aux volumes qui dépassent ce profil. |
| `colbert` | Expérimental | [MaxSim](../tests/test_v110_learned_retrieval.py), [masques, projection et tokenizer](../tests/test_learned_model_fidelity.py). | Installer `learned-retrieval`, qualifier le checkpoint et ses marqueurs ; mesurer RAM/VRAM et latence. Pas d'index ANN compressé fourni. |
| `hyde` | Expérimental | [Moyenne des hypothèses et scores](../tests/test_query_paper_algorithms.py), [stabilité numérique](../tests/test_retrieval_cross_review.py). | Qualifier ensemble générateur/encodeur, nombre d'hypothèses et coût ; vérifier que des hypothèses erronées ne diminuent pas le rappel. |
| `hyqe` | Expérimental | [Formules de classement](../tests/test_query_paper_algorithms.py), [questions vers documents](../tests/test_v120_architecture_coverage.py). | Évaluer les questions générées, l'encodeur, le cache et le budget d'indexation/reranking ; conserver les preuves originales pour la réponse. |
| `rag-fusion` | Expérimental | [Votes uniques](../tests/test_query_paper_algorithms.py), [variantes et fusion](../tests/test_v120_architecture_coverage.py). | Borner variantes et candidats par requête ; mesurer coût, rappel et dérive sémantique des reformulations. |
| `self-query` | Expérimental | [Filtres avant classement](../tests/test_v120_architecture_coverage.py), [JSON et filtres invalides](../tests/test_retrieval_cross_review.py). | Déclarer le schéma métier et tester les comparaisons réelles ; conserver les ACL dans une couche d'autorisation indépendante des filtres proposés par le LLM. |
| `step-back` | Expérimental | [Étapes de lecture et synthèse](../tests/test_reading_note_stepback.py), [fusion des preuves](../tests/test_v120_architecture_coverage.py). | Évaluer les abstractions et principes produits ; mesurer la fidélité de la synthèse et le coût des appels supplémentaires. |
| `query-decomposition` | Expérimental | [Sous-questions et fusion](../tests/test_v120_architecture_coverage.py), [frontières des algorithmes](../tests/test_retrieval_algorithm_boundaries.py). | Fixer les budgets de sous-questions/candidats ; tester questions ambiguës, décompositions erronées et rappel de la réponse complète. |
| `context-compression` | Bêta | [Extraits et snapshots](../tests/test_v120_architecture_coverage.py), [préservation des négations](../tests/test_retrieval_cross_review.py), [appels bornés](../tests/test_architecture_runtime_contracts.py). | Garder la vérification d'extraction activée ; valider que les preuves nécessaires survivent à la compression, puis mesurer économie de tokens et coût du compresseur. |
| `long-context-packing` | Expérimental | [Budget exact](../tests/test_v130_context_packing.py), [intégration moteur](../tests/test_v130_context_packing_engine.py), [LongRAG](../tests/test_long_rag_paper.py). | Fournir le tokenizer cible et réserver la place du prompt/réponse ; qualifier regroupement, lecteur à une/deux étapes et rappel sous contrainte de contexte. |
| `chain-of-note` | Expérimental | [Notes et preuves citées](../tests/test_reading_note_stepback.py), [filtrage des notes](../tests/test_v120_architecture_coverage.py). | Qualifier les catégories et citations produites ; borner notes et longueur. Aucun checkpoint entraîné selon l'article n'est fourni. |
| `crag` | Expérimental | [Actions et snapshot évalué](../tests/test_v130_crag_advanced.py), [grader sémantique](../tests/test_crag_semantic.py). | Calibrer le grader sur des labels réservés, fournir le refiner et la recherche externe ; tester erreurs, provenance et autorisation des preuves externes. |
| `self-rag` | Expérimental | [Recherche segmentée](../tests/test_self_rag_segmented.py), [scores de réflexion](../tests/test_v150_self_rag_reflection.py), [décodeur](../tests/test_self_rag_transformers_decoder.py). | Fournir un checkpoint à tokens de réflexion compatible ; qualifier budgets de faisceau/segments et comportement Retrieve/No/Continue. Les modèles 7B/13B et benchmarks ne sont pas démontrés par les doubles de test. |
| `flare` | Expérimental | [Incertitude](../tests/test_v130_flare_uncertainty.py), [probabilités token](../tests/test_v150_flare_probabilities.py). | Utiliser des probabilités calibrées si disponibles, borner itérations et appels ; le fallback fondé sur la longueur d'un brouillon n'est pas une estimation probabiliste. |
| `adaptive-rag` | Expérimental | [Routage des stratégies](../tests/test_v130_adaptive_rag.py), [apprentissage](../tests/test_adaptive_learning.py). | Qualifier le classifieur sur les classes et coûts réels ; contrôler les réponses sans retrieval et les erreurs de routage. L'heuristique par défaut ne remplace pas cette évaluation. |
| `parent-child` | Bêta | [Contexte parent](../tests/test_v05_architectures.py), [autorisation des parents](../tests/test_retrieval_cross_review.py). | Maintenir les identités et ACL des parents ; vérifier mises à jour/suppressions, absence de parent orphelin et taille du contexte élargi. |
| `multi-hop` | Bêta | [Planification et observations](../tests/test_v130_multihop_planning.py), [intégration](../tests/test_v06_architectures.py). | Qualifier le planner et ses arrêts ; borner hops/candidats et mesurer qualité sur des questions exigeant plusieurs preuves. |
| `raptor` | Expérimental | [Traversée bornée](../tests/test_v130_raptor_traversal.py), [UMAP/GMM](../tests/test_v150_raptor_paper.py), [ACL des résumés](../tests/test_research_provenance_boundaries.py). | Installer `raptor` pour le clustering papier ; fournir embeddings et résumeur validés, contrôler fidélité des résumés et dépendances d'autorisation, mémoire et coût de reconstruction. |
| `graph-rag` | Expérimental | [Métadonnées graphe](../tests/test_v06_architectures.py), [autorisation et snapshots](../tests/test_architecture_authorization_and_snapshots.py). | Qualifier extraction et expansion ; ce moteur GraphRAG-lite n'effectue ni partition en communautés ni synthèse globale de communautés. |
| `agentic-rag` | Expérimental | [Registre, entrées et max_steps](../tests/test_agentic_rag.py). | N'enregistrer que les outils autorisés ; définir authentification, timeout et quotas de chaque outil, puis tester interruptions et échecs. L'orchestrateur ne peut annuler de force un appel synchrone arbitraire. |
| `federated` | Bêta | [Fusion multi-sources](../tests/test_v06_architectures.py), [concurrence et ACL](../tests/test_architecture_runtime_boundaries.py). | Authentifier et borner chaque source ; définir politique en cas d'échec partiel, normalisation des identités et objectifs de latence de la fédération. |
| `conversational` | Bêta | [Historique et requête autonome](../tests/test_v05_architectures.py), [rétention et fermeture des flux](../tests/test_architecture_runtime_boundaries.py). | Définir identité de session, isolation tenant, rétention, suppression et stockage partagé si plusieurs workers ; qualifier la reformulation et limiter le contexte historique. |
| `sql-rag` | Bêta | [SQL et tables](../tests/test_v07_enterprise.py), [allowlist et budget SQL](../tests/test_architecture_runtime_boundaries.py). | Configurer tables autorisées, lecture seule, plafond de lignes et budget d'exécution ; tester le schéma réel, les vues, les jointures et les refus d'accès. |
| `multimodal-rag` | Expérimental | [Recherche texte/image](../tests/test_v110_multimodal.py), [contrat média](../tests/test_api_consistency_multimodal.py), [CLIP](../tests/test_sentence_transformers_compatibility.py). | Installer `multimodal`, qualifier l'encodeur et les formats ; protéger les fichiers média et dimensionner RAM/VRAM. Audio/vidéo nécessitent transcription ou encodeur applicatif. |
| `retrieval-evaluation` | Stable | [Métriques retrieval](../tests/test_v02_features.py), [régressions fonctionnelles](../tests/test_functional_regressions.py). | Constituer des labels de pertinence représentatifs et sans fuite d'entraînement ; publier métriques par segment, tailles d'échantillons et seuils de non-régression. |
| `generation-evaluation` | Bêta | [Métriques de génération](../tests/test_v04_features.py), [évaluation sémantique](../tests/test_semantic_evaluation.py). | Calibrer les scores sur des annotations humaines ; évaluer les preuves réellement utilisées et distinguer citation, support, exactitude et abstention. |
| `claim-evaluation` | Expérimental | [Support, contradictions et citations](../tests/test_v130_claim_evaluation.py), [juges sémantiques](../tests/test_semantic_evaluation.py). | Installer `evaluation-semantic` pour NLI ou fournir un juge LLM ; calibrer segmentation et scores. Le fallback lexical est un diagnostic, pas une preuve de factualité. |
| `access-controlled-rag` | Bêta | [Fail-closed et tenant](../tests/test_v101_security.py), [collections](../tests/test_tenant_collection_boundaries.py), [preuves dérivées](../tests/test_research_provenance_boundaries.py). | Fournir le principal depuis l'authentification applicative ; tester isolation tenant/collection/rôles avant tout envoi au modèle, y compris cache, résumés et sources dérivées. |
| `community-graphrag` | Expérimental | [Communautés et rapports](../tests/test_v120_community_graph.py), [Leiden/map-reduce](../tests/test_v150_community_graph_paper.py), [sémantique et ACL](../tests/test_research_graph_semantics.py). | Installer `graphrag` sous Python 3.10–3.12 pour Leiden ; qualifier extraction, liens d'entités, rapports, budgets et recherche locale/globale. Vérifier chaque dépendance de provenance avant synthèse. |
| `colpali` | Expérimental | [MaxSim, filtres et transactions](../tests/test_v120_colpali.py), [masques et traitement modèle](../tests/test_learned_model_fidelity.py). | Installer `colpali`, qualifier le checkpoint officiel sur les pages cibles ; mesurer VRAM, batchs et index exact en mémoire. Les poids complets ne sont pas validés par les petits modèles de test. |
| `temporal-rag` | Expérimental | [Validité et conflits](../tests/test_v120_temporal_rag.py), [FKS/TKS et contraintes](../tests/test_time_r4.py). | Définir fuseau, dates de validité et règle de conflit ; TimeR4 requiert des faits fiables et un encodeur temporel entraîné. Tester que réécriture et filtres n'élargissent jamais les droits ni l'intervalle demandé. |
| `retrieval-training` | Expérimental | [Mining/distillation](../tests/test_v120_training.py), [optimisation PyTorch](../tests/test_v150_training_torch.py), [RAFT/RankRAG](../tests/test_generative_training.py). | Installer l'extra d'entraînement adapté au matériel ; séparer train/validation/test, contrôler provenance des exemples, convergence et restauration des checkpoints. L'entraînement distribué n'est pas implémenté. |

## Conditions communes avant activation

Pour chaque composition retenue, fixer des seuils mesurables de rappel, qualité
de réponse, abstention, p95/p99 de latence, taux d'erreur, mémoire et coût. Les
tester sur un corpus et une charge représentatifs, en incluant les refus
d'autorisation, résultats vides, réponses fournisseur invalides, timeouts et
retries. Conserver le résultat par architecture et par version de modèle : un
succès du RAG simple ne qualifie pas automatiquement une composition multi-hop.

Les limites de candidats et d'itérations bornent le travail demandé aux
collaborateurs. Elles ne remplacent pas un timeout transport, une limite de
tokens ni les quotas du fournisseur. Les handlers synchrones applicatifs
doivent coopérer pour interrompre réellement un calcul ou un appel bloqué.

Appliquer les ACL avant les prompts et les caches, en tenant compte des sources
dont dépendent les résumés, graphes et autres preuves dérivées. Vérifier les
parcours `ask`, async, streaming et arrêt anticipé de la composition réellement
exposée. Tester le comportement de plusieurs requêtes simultanées et, si un
index change en service, la cohérence des résultats pendant ces changements.

Qualifier sauvegarde/restauration et retour à la version précédente avec les
artefacts effectivement livrés. Activer ensuite la combinaison validée dans
un périmètre limité et surveillé, avec des seuils d'arrêt. Les intégrations
explicitement exclues du [guide de production](production.md), dont Chroma,
ne deviennent pas qualifiées par leur présence dans cette matrice.
