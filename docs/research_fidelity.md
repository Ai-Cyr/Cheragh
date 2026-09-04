# Fidélité aux publications et qualification pour la production

Audit du 4 septembre 2026. Les **44 techniques** du [catalogue](../src/cheragh/catalog/techniques.py) ont un point d'entrée disponible : 6 stables, 12 bêta et 26 expérimentales. Ces statuts décrivent les composants logiciels. Ils ne prouvent ni la reproduction d'un article ni l'aptitude d'une configuration à servir un corpus réel.

La matrice distingue ce que le code exécute des validations restant à effectuer. Une référence indique une méthode apparentée; elle ne transfère pas les performances de l'article à Cheragh. « — » signifie qu'aucun article précis n'est attribué ici à ce patron d'ingénierie. Les références HyQE, Dense X Retrieval, Step-back et Chain-of-Note complètent celles du catalogue pour clarifier les écarts.

## Matrice des 44 techniques

| Identifiant / technique | Statut | Portée du composant | Référence primaire | Écart ou qualification restante |
|---|---|---|---|---|
| `naive-rag` · Naive RAG | Stable | `RAGEngine` : retrieval, contexte, génération, citations et traces. | — | Mesurer la qualité du retriever et du générateur configurés; une réponse citée peut rester incorrecte. |
| `recursive-chunking` · Recursive chunking | Stable | Découpage récursif fondé sur des séparateurs et des tailles en caractères. | — | Valider conservation des faits, tableaux, offsets et budget du tokenizer cible. |
| `semantic-chunking` · Semantic chunking | Bêta | Détection de frontières thématiques à partir des embeddings. | — | Calibrer les seuils sur les langues et types de documents du corpus. |
| `hierarchical-chunking` · Hierarchical chunking | Bêta | Chunks parent/enfant et provenance de sections. | — | Tester la couverture et la reconstruction des sources après mise à jour du corpus. |
| `sentence-window` · Sentence-window retrieval | Expérimental | Recherche de phrases puis extension à leur fenêtre de contexte. | — | Choisir une segmentation linguistique et une fenêtre adaptées; mesurer le gain de rappel. |
| `propositional` · Propositional retrieval | Expérimental | Index de propositions générées et retour au contexte source. | [Dense X Retrieval](https://arxiv.org/abs/2312.06648) | La segmentation de secours ne garantit pas des propositions atomiques autonomes; évaluer extracteur et granularité. |
| `bm25` · BM25 | Stable | Retrieval lexical sparse autonome. | — | Fixer tokenisation et paramètres sur un jeu annoté; mesurer coût et mémoire à la taille visée. |
| `dense` · Dense retrieval | Stable | Recherche mono-vecteur avec embedder interchangeable et stockage local. | — | `HashingEmbedding` est une baseline lexicale; qualifier modèle sémantique, index et rappel ANN éventuel. |
| `hybrid` · Hybrid sparse+dense | Stable | Fusion pondérée des résultats sparse et dense. | — | Calibrer normalisation et poids; comparer à chaque branche seule. |
| `reranking` · Cross-encoder reranking | Bêta | Reclassement d'un ensemble de candidats avec un modèle optionnel. | — | Mesurer rappel du premier étage, troncature du modèle, latence et qualité hors domaine. |
| `rrf` · Reciprocal Rank Fusion | Bêta | Fusion des rangs, dédoublonnage et borne du nombre de résultats. | — | Choisir constante et pools; valider les identifiants communs et les résultats à égalité. |
| `mmr` · Maximal Marginal Relevance | Expérimental | Compromis similarité à la question et diversité des documents. | — | Calibrer le poids de diversité et vérifier que les preuves nécessaires restent présentes. |
| `splade` · Learned sparse retrieval | Expérimental | Encodeur SPLADE injectable/optionnel et scoring sparse exact. | [SPLADE](https://arxiv.org/abs/2107.05720) | Pas de réentraînement ni de grand index inversé fourni; qualifier poids, vocabulaire et sparsité. |
| `colbert` · Late-interaction retrieval | Expérimental | Calcul MaxSim token-à-token exact en mémoire. | [ColBERT](https://arxiv.org/abs/2004.12832) | Le défaut n'est pas un checkpoint ColBERT entraîné; valider encodeur, masques, mémoire et index à grande échelle. |
| `hyde` · HyDE | Expérimental | Génération d'une réponse hypothétique puis recherche dans les vraies sources. | [HyDE](https://arxiv.org/abs/2212.10496) | Qualifier générateur et encodeur; comparer aux requêtes directes et aux paramètres de l'article. |
| `hyqe` · HyQE | Expérimental | Questions hypothétiques indexées avec retour aux documents d'origine. | [HyQE](https://arxiv.org/abs/2410.15262) | Variante d'indexation; ne pas confondre avec la reproduction du classement de candidats et de ses scores dans l'article. |
| `rag-fusion` · RAG-Fusion | Expérimental | Variantes de requête, recherches multiples et fusion RRF. | [RAG-Fusion](https://arxiv.org/abs/2402.03367) | Évaluer dérive des variantes, gains de rappel et multiplication du coût. |
| `self-query` · Self-query retrieval | Expérimental | Séparation requête sémantique/filtres via un parseur borné. | — | Tester les opérateurs admis et les erreurs de parsing; les filtres générés ne remplacent pas les ACL. |
| `step-back` · Step-back prompting | Expérimental | Recherche complémentaire via une question plus abstraite. | [Step-back](https://arxiv.org/abs/2310.06117) | Qualifier abstraction et utilisation des preuves; les prompts et évaluations du papier ne sont pas reproduits. |
| `query-decomposition` · Query decomposition | Expérimental | Décomposition en sous-questions et fusion des résultats. | — | Évaluer dépendances entre questions, couverture et budget d'appels. |
| `context-compression` · Contextual compression | Bêta | Filtrage/compression du contexte après retrieval. | — | Mesurer perte de preuves, conservation des négations et exactitude des citations après compression. |
| `long-context-packing` · Long-context packing | Expérimental | Budget de rendu, quotas de source et placement des preuves. | [LongRAG](https://arxiv.org/html/2406.15319v1) | Un packer seul ne construit pas les groupes documentaires et le long retriever du papier; utiliser le tokenizer cible. LongRAG n'exige pas de nouvel entraînement. |
| `chain-of-note` · Chain-of-Note | Expérimental | Notes de lecture générées avant synthèse. | [Chain-of-Note](https://arxiv.org/abs/2311.09210) | L'article inclut un entraînement sur des notes; ici, valider prompts, refus et résistance aux documents trompeurs. |
| `crag` · Corrective RAG | Expérimental | Décisions correct/ambiguous/incorrect, source externe et raffinement injectables. | [CRAG](https://arxiv.org/abs/2401.15884) | Grader/refiner lexicaux par défaut; fournir un évaluateur calibré et une source externe pertinente. |
| `self-rag` · Inference-time Self-RAG | Expérimental | Révisions bornées; adaptateur de probabilités des reflection tokens, scores du papier et porte initiale optionnels. | [Self-RAG](https://arxiv.org/html/2310.11511v1) | Pas de modèle à reflection tokens ni de génération par segments et recherche en faisceau; voir décision ci-dessous. |
| `flare` · FLARE active retrieval | Expérimental | Brouillon avec probabilités token-level, adaptateur OpenAI, masquage des tokens incertains et régénération. | [FLARE](https://arxiv.org/abs/2305.06983) | Adaptateur et contrats testés; calibrer le vrai modèle. Le fallback texte seul fondé sur la longueur reste heuristique. |
| `adaptive-rag` · Adaptive RAG | Expérimental | Routes sans retrieval, simple et itérative; classifieur injectable. | [Adaptive-RAG](https://arxiv.org/abs/2403.14403) | Aucun classifieur appris sur les labels d'efficacité des stratégies; évaluer routage, coût et regret par question. |
| `parent-child` · Parent-child retrieval | Bêta | Recherche de petits chunks puis récupération du parent. | — | Mesurer rappel et coût du contexte étendu; maintenir les liens parent/enfant à l'indexation. |
| `multi-hop` · Multi-hop RAG | Bêta | Boucle de planning, retrieval et observations avec arrêt borné. | [IRCoT](https://arxiv.org/abs/2212.10509), [ReAct](https://arxiv.org/abs/2210.03629) | Planners déterministes/JSON injectables; aucun protocole expérimental ou ensemble de prompts original reproduit. |
| `raptor` · RAPTOR | Expérimental | Arbre de résumés; UMAP/GMM souple optionnel, budgets et mode `paper_tree` en plus des modes historiques. | [RAPTOR](https://arxiv.org/html/2401.18059v1) | Mécanismes testés avec UMAP/scikit-learn; `top_k` et budgets peuvent limiter le parcours. Qualifier embeddings et résumés; aucun résultat QA du papier reproduit. |
| `graph-rag` · Graph-enhanced RAG | Expérimental | Voisinage d'entités/relations combiné au retrieval vectoriel. | [GraphRAG](https://arxiv.org/html/2404.16130v2) | Cette classe graph-lite ne réalise pas le pipeline global du papier; utiliser et qualifier le composant Community GraphRAG. |
| `agentic-rag` · Agentic RAG | Expérimental | Boucle bornée sur des outils explicitement enregistrés. | [ReAct](https://arxiv.org/abs/2210.03629) | Qualifier politique, permissions des outils, erreurs et budgets; aucun agent entraîné fourni. |
| `federated` · Federated RAG | Bêta | Agrégation de plusieurs retrievers/domaines. | — | Tester indisponibilités partielles, identités documentaires, scores et droits d'accès par source. |
| `conversational` · Conversational RAG | Bêta | Fenêtre de contexte conversationnel bornée pour la requête. | — | Prévoir rétention des tours stockés, isolation des sessions et évaluation des références aux tours précédents. |
| `sql-rag` · SQL RAG | Bêta | Génération et exécution contrôlée de requêtes SQLite en lecture seule. | — | Valider sémantique SQL, limites d'exécution, schéma exposé et autorisations métier. |
| `multimodal-rag` · Multimodal RAG | Expérimental | Texte/images locales, provenance média et adaptateur CLIP optionnel. | [CLIP](https://arxiv.org/abs/2103.00020) | CLIP ne démontre pas la qualité du pipeline RAG; évaluer documents visuels, langues et formats; audio/vidéo demandent une adaptation. |
| `retrieval-evaluation` · Retrieval evaluation | Stable | Hit rate, MRR, précision, rappel, nDCG et précision du contexte. | — | Nécessite jugements pertinents et identifiants cohérents; rapporter les métriques par segment et l'incertitude d'échantillonnage. |
| `generation-evaluation` · Generation evaluation | Bêta | Diagnostics déterministes de citations et d'ancrage lexical. | — | Le recouvrement lexical n'établit pas la vérité ni l'absence de contradiction; compléter par annotations ou juge validé. |
| `claim-evaluation` · Claim-level faithfulness | Expérimental | Support, contradiction et alignement citation/preuve avec juge injectable. | [RAGAS](https://arxiv.org/abs/2309.15217), [RAGChecker](https://arxiv.org/abs/2408.08067) | Le fallback lexical ne détecte pas les contradictions; calibrer segmentation et juge NLI/LLM, mesurer l'accord humain. |
| `access-controlled-rag` · Access-controlled RAG | Bêta | Politique tenant, collection, rôle et classification sur les preuves. | — | Tester les refus sur index, cache, provenance et flux réels; dériver l'identité d'une authentification fiable. |
| `community-graphrag` · Community GraphRAG | Expérimental | Leiden hiérarchique optionnel, rapports LLM, recherche globale map-reduce et modes historiques. | [GraphRAG](https://arxiv.org/html/2404.16130v2) | Mécanismes testés avec graspologic; extraction sémantique, qualité des rapports et évaluation corpus restent à fournir. |
| `colpali` · Visual-document late interaction | Expérimental | MaxSim token-à-patch; adaptateur du modèle officiel optionnel. | [ColPali](https://arxiv.org/abs/2407.01449) | Pas d'index multivecteur distribué; qualifier rendu des pages, poids, mémoire GPU et retrieval visuel. |
| `temporal-rag` · Temporal RAG | Expérimental | Fenêtres de validité, fraîcheur et sélection de versions. | [TimeR4](https://aclanthology.org/2024.emnlp-main.394/) | Pas de graphe temporel, boucle retrieve–rewrite–retrieve–rerank ou entraînement contrastif temporel; exiger des dates fiables. |
| `retrieval-training` · Retriever/RAG training | Expérimental | Mining, distillation, données RAFT avec citations vérifiées et trainer retrieval PyTorch optionnel. | [Hard negatives](https://arxiv.org/abs/2104.08051), [RAFT](https://arxiv.org/html/2403.10131v1), [RankRAG](https://arxiv.org/abs/2407.02485) | Optimisation retrieval testée sur CPU; aucun checkpoint métier entraîné, trainer génératif RAFT ou reproduction de RankRAG fourni. |

## Approfondissements implémentés et validation logicielle

Les mécanismes suivants ont une implémentation et des tests ciblés. RAPTOR a été vérifié avec les bibliothèques UMAP/scikit-learn, GraphRAG avec graspologic et le trainer avec PyTorch sur CPU. Les tests de génération utilisent des collaborateurs contrôlés; ils ne démontrent pas la qualité d'un LLM réel. Aucune expérience de reproduction scientifique n'est revendiquée.

| Changement | Mécanisme visé | État de cet audit | Validation encore nécessaire |
|---|---|---|---|
| RAPTOR UMAP/GMM | Réduction globale/locale, choix du nombre de composantes par BIC et appartenances souples. | Implémenté; backend réel testé. | Qualité des résumés, embeddings du domaine, échelle et reproduction des benchmarks. |
| RAPTOR `paper_tree` | Classement par cosine à chaque profondeur et conservation multi-niveau sous les bornes configurées. | Implémenté; tests de parcours et budgets. | Configurer `top_k` et `beam_width`; le tokenizer du modèle est nécessaire pour une borne exacte. |
| GraphRAG Leiden | Partition hiérarchique avec provenance des communautés. | Implémenté; backend réel testé. | Extraction de graphe fiable, coût d'indexation et stabilité des communautés sur mises à jour. |
| GraphRAG global map-reduce | Rapports mélangés, blocs de contexte, réponses intermédiaires scorées, filtrage puis réduction. | Implémenté; tests de budget, accès et provenance. | Qualité du LLM, calibration des helpfulness scores et fidélité des synthèses au corpus. |
| RAFT données ancrées | Réponses pédagogiques liées à des citations vérifiées dans les documents oracle. | Implémenté; contrats et mélange des documents testés. | Vérification humaine des réponses et distracteurs; entraînement génératif et évaluation séparée à réaliser. |
| Trainer retrieval optionnel | Optimisation contrastive et distillation derrière un adaptateur PyTorch. | Implémenté; gradients et apprentissage testés sur CPU. | Modèle/données métier, sauvegarde/rechargement applicatifs, GPU/distribution et validation hors entraînement. |
| Self-RAG scores et porte | Groupes complets, normalisation stable, scores relevance/support/utility et décision initiale de retrieval. | Implémenté; 54 tests ajoutés, 5 historiques préservés. | Probabilités réelles d'un checkpoint compatible; génération par segments et recherche en faisceau absentes. |
| FLARE probabilités token-level | Brouillon structuré, adaptateur fournisseur et masque pour la requête de recherche. | Implémenté; contrats testés. | Disponibilité et calibration des log-probabilités du fournisseur choisi; qualité sur réponses longues. |

Pour RAPTOR, le papier associe embeddings, UMAP/GMM, résumés récursifs et retrieval multi-niveau. Le nouveau mode souple subdivise les clusters dépassant le budget de résumé et refuse un nœud isolé trop volumineux. Les modes historiques restent des variantes Cheragh. `paper_tree` conserve les sélections multi-niveau sous une borne totale `top_k`; son budget de retrieval compte les contenus concaténés, sans les wrappers du prompt final. [Méthode RAPTOR, §3](https://arxiv.org/html/2401.18059v1).

Pour GraphRAG, la recherche globale du papier traite les rapports d'un niveau de communautés, répartis en blocs après mélange. Chaque réponse intermédiaire reçoit une utilité entre 0 et 100; les réponses nulles sont supprimées, les autres classées avant réduction sous budget. Un simple top-k lexical de rapports ne réalise pas ce parcours. [Méthode GraphRAG, §3.1.6](https://arxiv.org/html/2404.16130v2).

## Décision d'implémentation Self-RAG

**L'adaptateur de scores est implémenté sans distribuer de poids; il ne transforme pas le moteur actuel en Self-RAG du papier.** Celui-ci apprend un vocabulaire de reflection tokens et un générateur à partir de données annotées par un critic; les passages récupérés sont masqués dans la loss. À l'inférence, la décision yes/no/continue dépend des segments précédents, les continuations sont générées par passage et une recherche en faisceau sélectionne les segments. Le moteur actuel critique et révise une réponse complète. [Self-RAG, §3](https://arxiv.org/html/2310.11511v1).

`ReflectionTokenDistribution` accepte les probabilités ou log-probabilités réelles de chaque groupe; `ReflectionTokenScorer` calcule les scores suivants, conformément au [code d'inférence des auteurs](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_short_form.py) :

| Score | Calcul après normalisation au sein du groupe |
|---|---|
| Pertinence | Probabilité de `Relevant`. |
| Support | Probabilité de `Fully supported` + 0,5 × probabilité de `Partially supported`. |
| Utilité | Espérance des poids −1; −0,5; 0; 0,5; 1 associés aux utilités 1 à 5. |
| Candidat | Somme pondérée pertinence/support/utilité; contribution optionnelle de `exp(moyenne des log-probabilités de séquence)`. |

Le contrat refuse groupes incomplets, valeurs non finies, probabilités négatives et masse totale nulle, normalise les log-probabilités de façon stable et expose l'identifiant du modèle lorsqu'il est fourni. Aucune probabilité manquante n'est remplacée par du recouvrement lexical. `ReflectionTokenRetrievalGate` prend un fournisseur explicite et applique le seuil strict sur P(Yes)/(P(Yes)+P(No)), pour la décision initiale. Il accepte un groupe initial complet à deux tokens ou le groupe complet incluant Continue; il ne simule pas la réutilisation de preuves et refuse une masse initiale nulle. Le branchement à un checkpoint compatible, la génération par segments, le beam search et la calibration restent distincts. Les tests synthétiques valident les calculs, pas la capacité du modèle à s'auto-évaluer.

## Conditions communes avant exploitation

Une qualification de production porte sur une configuration précise, un corpus versionné et une charge cible. Pour chaque technique retenue, conserver les éléments suivants avec le résultat de validation :

1. **Données et qualité** : jeu de questions représentatif séparé de l'entraînement, preuves annotées, questions sans réponse et comparaison à une baseline simple; rappel, précision des citations et qualité des réponses mesurés.
2. **Modèles et reproductibilité** : versions des poids, prompts, tokenizer, dépendances, seeds et paramètres. Un double déterministe teste les contrats mais ne remplace pas l'évaluation d'un modèle.
3. **Ressources et disponibilité** : latences p50/p95, débit, mémoire, coût/tokens, limites d'appels et comportement sous panne du fournisseur, du stockage ou d'une dépendance optionnelle.
4. **Accès et données** : isolation tenant/session, filtres réellement appliqués, cache, suppression et mise à jour des sources, protection des secrets et contenu des traces.
5. **Exploitation** : procédure de déploiement et retour arrière, sauvegarde/restauration vérifiées, seuils d'alerte et critères d'acceptation décidés pour l'application.

Les tests unitaires et d'intégration du dépôt contribuent à ces conditions. Aucun résultat présenté dans ce document n'établit que toutes les combinaisons des 44 techniques sont prêtes pour la production, ni que les performances publiées ont été reproduites. Pour les paramètres d'exploitation, consulter aussi le [guide de production](production.md) et les [limites du serveur](production_server.md).
