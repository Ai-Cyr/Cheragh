# Audit des 44 techniques — 16 septembre 2026

Les 44 entrées ont été relues, leurs mécanismes vérifiés et les écarts corrigés ou explicitement qualifiés. Cet audit porte sur le code et ses contrats. Il ne revendique ni les scores des publications ni une supériorité mesurée sur un corpus métier. Les statuts restent **6 stable, 12 bêta, 26 expérimental**.

Les nouveaux chemins complets coexistent avec les interfaces historiques. Les points d’entrée, dépendances, modèles requis et limites sont détaillés dans les [recettes](research_recipes.md) et la [matrice de fidélité](research_fidelity.md). Les tests cités ci-dessous se trouvent dans `tests/`.

| Technique | Conclusion de la vérification | Preuves exécutables |
|---|---|---|
| `naive-rag` | Boucle retrieval → contexte → génération, citations, abstention, validation des composants et streaming conservés. | `test_core.py`, `test_extractive_fallback.py`, `test_async_stream_production.py` |
| `recursive-chunking` | Frontières sur le texte source ; couverture des fins courtes, CRLF, Unicode, offsets réels et limites strictes. | `test_retrieval_algorithm_boundaries.py`, `test_v041_chunking_routing.py` |
| `semantic-chunking` | Buffers de phrases, distances cosinus, seuil absolu/percentile ; texte et offsets conservés, fallback borné. | `test_retrieval_algorithm_boundaries.py`, `test_v090_quality.py` |
| `hierarchical-chunking` | Niveaux parent/enfant et métadonnées de provenance relus ; expansion des parents et couverture vérifiées. | `test_v041_chunking_routing.py`, `test_v090_quality.py` |
| `sentence-window` | Toutes les phrases classées avant expansion ; fin de corpus couverte, métadonnées copiées et source conservée. | `test_retrieval_algorithm_boundaries.py`, `test_v120_architecture_coverage.py` |
| `propositional` | Extraction complète et retour aux sources sans plafond de candidats caché ; adaptateur du checkpoint Dense X avec format Title/Section/Content, JSON strict et refus de troncature. | `test_final_research_integration.py`, `test_v120_architecture_coverage.py` |
| `bm25` | BM25 à IDF positif de type Lucene ; formule analytique vérifiée et résultat indépendant de la présence de rank-bm25. | `test_retrieval_algorithm_boundaries.py`, `test_v120_bm25_rrf.py` |
| `dense` | Cosinus réel avec encodeurs non normalisés, zéros et grandes valeurs ; validation et snapshots du stockage. | `test_retrieval_algorithm_boundaries.py`, `test_storage_cache_boundaries.py` |
| `hybrid` | Normalisation des candidats autorisés et fusion pondérée ; égalités stables, sorties indépendantes des sources. | `test_api_consistency_backend_filters.py`, `test_v120_bm25_rrf.py` |
| `reranking` | Cross-encoder concret ; exactement un score fini par candidat, formats N et N×1 admis. | `test_retrieval_algorithm_boundaries.py`, `test_api_consistency_advanced.py` |
| `rrf` | Un vote par document par liste, rangs d’origine et identifiants stables. | `test_v120_bm25_rrf.py`, `test_query_paper_algorithms.py` |
| `mmr` | Pertinence et redondance cosinus, sélection incrémentale stable, pool au moins égal à top_k. | `test_v120_architecture_coverage.py`, `test_review_fixes.py` |
| `splade` | MLM entraîné, max/sum log1p(ReLU), masque et produit sparse ; checkpoint officiel exécuté. | `test_learned_model_fidelity.py` |
| `colbert` | Checkpoint ColBERTv2, projection 128D, marqueurs query/document, augmentation MASK, ponctuation, L2 et MaxSim exact. | `test_learned_model_fidelity.py` |
| `hyde` | Moyenne Eq. 7/8 dans l’espace document ; produit scalaire par défaut, cosinus explicite ; hypothèses validées. | `test_query_paper_algorithms.py` |
| `hyqe` | Reranker de candidats avec score additif contexte/questions Eq. 2/5, max/mean, cache et couverture des partitions. | `test_query_paper_algorithms.py` |
| `rag-fusion` | Variantes distinctes, dédoublonnage et RRF canonique. | `test_query_paper_algorithms.py`, `test_v120_architecture_coverage.py` |
| `self-query` | JSON, champs, types et opérateurs validés avant filtrage ; contraintes invalides refusées. | `test_retrieval_algorithm_boundaries.py` |
| `step-back` | Moteur complet : abstraction, recherche générale, principes, synthèse sur preuves générales et spécifiques. | `test_reading_note_stepback.py` |
| `query-decomposition` | Sous-questions distinctes ; fusion RRF par défaut pour éviter de comparer les scores de requêtes différentes. | `test_reading_note_stepback.py` |
| `context-compression` | Phrases extraites vérifiées contre les sources ; paraphrases inventées et suppressions de négations rejetées. | `test_retrieval_algorithm_boundaries.py`, `test_v120_architecture_coverage.py` |
| `long-context-packing` | LongRAG ajouté : regroupement par voisinages, maximum des produits scalaires des fragments, sources intégrales et lecteur à une/deux étapes. | `test_long_rag_paper.py`, `test_v130_context_packing.py` |
| `chain-of-note` | Notes séquentielles direct/contextual/unknown, extraits exacts, provenance, abstention et connaissances paramétriques explicites. | `test_reading_note_stepback.py` |
| `crag` | Cross-encoder concret calibré, agrégation maximum, actions correct/ambiguous/incorrect et décomposition/recomposition sémantique. | `test_crag_semantic.py`, `test_v130_crag_advanced.py` |
| `self-rag` | Décodeur Transformers, logits réels, segments par passage, Yes/No/Continue, faisceau et budgets globaux ; aucune probabilité fabriquée. | `test_self_rag_segmented.py`, `test_self_rag_transformers_decoder.py`, `test_v150_self_rag_reflection.py` |
| `flare` | Log-probabilités alignées, masquage des incertitudes et régénération ; arrêt sur sentinelle exacte, IDs de sources stables. | `test_v150_flare_probabilities.py`, `test_final_research_integration.py` |
| `adaptive-rag` | Labels A/B/C calculés sur les trois stratégies exécutées ; classifier T5/BERT réellement entraînable, sauvegarde et rechargement. | `test_adaptive_learning.py`, `test_v130_adaptive_rag.py` |
| `parent-child` | Sur-recherche progressive des parents distincts ; métadonnées et documents copiés ; contrôle des sources lors de l’expansion. | `test_retrieval_algorithm_boundaries.py`, `test_review_fixes.py` |
| `multi-hop` | Planning JSON, observations, retrieval itératif et arrêts bornés relus ; fallback déterministe identifié. | `test_v130_multihop_planning.py` |
| `raptor` | UMAP/GMM réel, singletons conservés, résumés bornés sans couper les groupes, niveaux paper_tree et provenance des feuilles. | `test_v150_raptor_paper.py`, `test_raptor_levels_summaries.py`, `test_v130_raptor_traversal.py` |
| `graph-rag` | Voisinage graph-lite relu ; l’entrée demeure une baseline distincte du pipeline communautaire complet. | `test_v05_architectures.py`, `test_v120_architecture_coverage.py` |
| `agentic-rag` | Boucle ReAct plan/action/observation sur outils enregistrés ; validation des entrées et budgets vérifiés. | `test_agentic_rag.py` |
| `federated` | Fusion RRF par défaut, identités qualifiées par source, scores initiaux conservés et erreurs partielles explicites. | `test_retrieval_algorithm_boundaries.py`, `test_v05_architectures.py` |
| `conversational` | Réécriture LLM autonome à partir de l’historique ; copies des tours et limite zéro corrigée. | `test_retrieval_algorithm_boundaries.py`, `test_v05_architectures.py` |
| `sql-rag` | Schéma → SQL → exécution SQLite en lecture seule → preuves ; bornes et refus des opérations non autorisées relus. | `test_v06_architectures.py`, `test_functional_regressions.py` |
| `multimodal-rag` | Adaptateur CLIP texte/image et provenance relus ; audio/vidéo restent des transcriptions/adaptateurs spécifiques. | `test_v110_multimodal.py`, `test_api_consistency_multimodal.py` |
| `retrieval-evaluation` | Pertinence graduée cohérente, gains finis/non négatifs, doublons, profondeur et dénominateur de précision du contexte corrigés. | `test_retrieval_algorithm_boundaries.py`, `test_v090_quality.py` |
| `generation-evaluation` | Diagnostics lexicaux et de citations conservés avec leur portée explicite ; juge sémantique disponible séparément. | `test_v130_claim_evaluation.py` |
| `claim-evaluation` | Décomposition LLM stricte, juges LLM/NLI concrets, couverture par fenêtres, support/contradiction indépendants et citations vérifiées. | `test_semantic_evaluation.py`, `test_v130_claim_evaluation.py` |
| `access-controlled-rag` | Identité, tenant/collection, caches et preuves dérivées ; isolation réévaluée après expansion et synthèse. | `test_tenant_collection_boundaries.py`, `test_v101_security.py`, `test_storage_cache_boundaries.py` |
| `community-graphrag` | Extraction LLM ancrée, entités/relations/descriptions, gleaning, Leiden pondéré, rapports, recherche locale sémantique et map-reduce global. | `test_research_graph_semantics.py`, `test_v150_community_graph_paper.py` |
| `colpali` | Prétraitement officiel Query/PAD, types de tokens, dtype et masques ; MaxSim exact et lots bornés. | `test_learned_model_fidelity.py`, `test_v120_colpali.py` |
| `temporal-rag` | TimeR4 : faits sourcés, FKS/TKS séparés, réécriture ancrée, contraintes strictes, second retrieval/reranking et exemples contrastifs temporels. | `test_time_r4.py`, `test_v120_temporal_rag.py` |
| `retrieval-training` | Trainer retrieval contrastif/distillation ; vrai SFT causal/seq2seq RAFT, mélange des tâches RankRAG, classement et réponse avec les mêmes poids. | `test_generative_training.py`, `test_v150_training_raft.py`, `test_v150_training_torch.py` |

## Étendue de la preuve

- Tests mathématiques indépendants : scores HyQE/HyDE, BM25, cosinus, MaxSim, pertes et masquage prompt/padding.
- Modèles réels minuscules sur CPU : inférence et cache Llama, T5/BERT de classification, GPT2/T5 de SFT, gradients et sauvegarde/rechargement. Leur apprentissage sur quelques exemples vérifie le trainer, pas sa généralisation.
- Poids publiés exécutés : SPLADE, ColBERTv2, NLI MiniLM et un cross-encoder TinyBERT. Résultats et commandes dans le [rapport de vérification](research_validation.md).
- Backends UMAP/GMM et Leiden réellement installés et testés. Aucun appel LLM payant n’est nécessaire à la suite locale.
- Revue croisée des nouveaux composants : provenance des résumés, ACL des parents, contraintes temporelles, candidats Self-RAG et budgets.

Les benchmarks complets, le modèle ColPali complet, les poids Self-RAG 7B/13B et le grand propositionizer n’ont pas été exécutés. Le checkpoint temporel entraîné doit être fourni. Chain-of-Note conserve un chemin d’inférence par prompts ; CRAG utilise un cross-encoder calibré et non les poids T5 des auteurs. Les déviations sont indiquées dans la matrice.
