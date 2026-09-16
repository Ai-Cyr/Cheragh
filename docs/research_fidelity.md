# Fidélité aux publications et portée des composants

État vérifié le **16 septembre 2026**. Les 44 techniques disposent d’une implémentation disponible ; les mécanismes de recherche avancés restent expérimentaux. Les références qualifient la méthode suivie, sans transférer ses performances à Cheragh.

L’[audit exhaustif](sota_implementation_audit.md) relie chaque entrée à ses tests. Les [recettes](research_recipes.md) montrent les nouveaux chemins entraînables et les [résultats de vérification](research_validation.md) distinguent tests logiciels, exécution de checkpoints et benchmarks.

| Technique | Statut | Point d’entrée | Portée / limites restantes | Références primaires |
|---|---|---|---|---|
| `naive-rag` | stable | `cheragh.RAGEngine` | Une citation prouve un lien documentaire, pas la vérité de la réponse. | — |
| `recursive-chunking` | stable | `cheragh.RecursiveTextChunker` | Découpage en caractères ; TokenTextChunker accepte un tokenizer pour un budget subword exact. | — |
| `semantic-chunking` | beta | `cheragh.SemanticChunker` | Calibrer seuil/percentile et segmentation linguistique sur le corpus. | — |
| `hierarchical-chunking` | beta | `cheragh.HierarchicalChunker` | Maintenir identifiants et parenté lors des mises à jour du corpus. | — |
| `sentence-window` | experimental | `cheragh.SentenceWindowRetriever` | La segmentation par défaut est une baseline ; adapter aux langues et documents. | — |
| `propositional` | experimental | `cheragh.PropositionalRetriever` | Use TransformersPropositionizer for the trained Dense X extractor or supply an LLM; validate atomicity on the target corpus. | [source 1](https://arxiv.org/abs/2312.06648) |
| `bm25` | stable | `cheragh.BM25Retriever` | IDF positif, k1=1.5/b=0.75 ; tokenisation et paramètres à mesurer sur un jeu annoté. | — |
| `dense` | stable | `cheragh.MemoryVectorStore` | HashingEmbedding est lexical ; choisir un modèle sémantique entraîné adapté au domaine. | — |
| `hybrid` | stable | `cheragh.HybridSearchRetriever` | Poids et normalisation à calibrer ; comparer aux deux branches séparées. | — |
| `reranking` | beta | `cheragh.CrossEncoderReranker` | Le rappel est limité par les candidats et la fenêtre du checkpoint choisi. | — |
| `rrf` | beta | `cheragh.ReciprocalRankFusionRetriever` | Rangs dédoublonnés ; régler constante et tailles des pools. | — |
| `mmr` | experimental | `cheragh.MMRRetriever` | Régler le poids pertinence/diversité et vérifier la conservation des preuves. | — |
| `splade` | experimental | `cheragh.retrieval.SPLADERetriever` | Exact in-memory scoring; large corpora require an external inverted index. | [source 1](https://arxiv.org/abs/2107.05720) |
| `colbert` | experimental | `cheragh.retrieval.ColBERTRetriever` | The default ColBERTv2 checkpoint includes its trained projection and tokenizer conventions; exact in-memory MaxSim has no compressed ANN index. | [source 1](https://arxiv.org/abs/2004.12832) |
| `hyde` | experimental | `cheragh.HyDERetriever` | Moyenne Eq. 7/8 et produit scalaire ; le cosinus est une variante explicite. | [source 1](https://arxiv.org/abs/2212.10496) |
| `hyqe` | experimental | `cheragh.HyQEReranker` | Score additif Eq. 2/5 ; partitionnement des longues sources et cache sont des extensions. | [source 1](https://arxiv.org/abs/2410.15262) |
| `rag-fusion` | experimental | `cheragh.RAGFusionRetriever` | Mesurer dérive des variantes, rappel et coût des recherches multiples. | — |
| `self-query` | experimental | `cheragh.SelfQueryRetriever` | The bundled parser supports a bounded metadata-filter grammar; custom parsers remain injectable. | — |
| `step-back` | experimental | `cheragh.StepBackRAGEngine` | Principes générés puis synthèse sur sources ; pas de prompts/benchmarks originaux reproduits. | [source 1](https://arxiv.org/abs/2310.06117) |
| `query-decomposition` | experimental | `cheragh.QueryDecompositionRetriever` | Fusion RRF ; utiliser MultiHopRAGEngine pour les dépendances entre sous-questions. | — |
| `context-compression` | beta | `cheragh.ContextualCompressionRetriever` | Extraction vérifiée par défaut ; valider aussi l’utilité des extraits conservés. | — |
| `long-context-packing` | experimental | `cheragh.LongContextPacker` | LongRAGRetriever and LongRAGEngine additionally implement document grouping, max-chunk dot scoring and a one/two-stage long reader. Exact limits require the target tokenizer. | [source 1](https://arxiv.org/abs/2406.15319) |
| `chain-of-note` | experimental | `cheragh.ChainOfNoteRAGEngine` | Prompted inference; no paper-trained Chain-of-Note checkpoint is bundled. | [source 1](https://arxiv.org/abs/2311.09210) |
| `crag` | experimental | `cheragh.CorrectiveRAGEngine` | Use CrossEncoderRetrievalGrader with held-out calibration and SemanticKnowledgeRefiner; the legacy defaults remain lexical. External search is application-provided. | [source 1](https://arxiv.org/abs/2401.15884) |
| `self-rag` | experimental | `cheragh.SegmentedSelfRAGEngine` | TransformersSelfRAGDecoder requires a trained reflection-token checkpoint. Critic/generator training and published benchmarks are not reproduced. | [source 1](https://openreview.net/forum?id=hSyW5go0v8) |
| `flare` | experimental | `cheragh.FLAREPipeline` | Token-confidence adapters are injectable; text-only LLM clients use the documented draft-length fallback. | [source 1](https://arxiv.org/abs/2305.06983) |
| `adaptive-rag` | experimental | `cheragh.AdaptiveRAGEngine` | AdaptiveSilverDatasetBuilder collects actual strategy outcomes; TransformersComplexityClassifier trains T5/BERT routes. Deploy a validated classifier instead of the legacy heuristic. | [source 1](https://arxiv.org/abs/2403.14403) |
| `parent-child` | beta | `cheragh.ParentChildRetriever` | Qualifier modèles, données et charge sur un corpus représentatif. | — |
| `multi-hop` | beta | `cheragh.MultiHopRAGEngine` | Planner quality is application-provided; bundled rule-based/JSON adapters do not reproduce trained reasoning policies. | [source 1](https://arxiv.org/abs/2212.10509), [source 2](https://arxiv.org/abs/2210.03629) |
| `raptor` | experimental | `cheragh.RAPTOREngine` | Optional UMAP/GMM soft clustering and paper_tree traversal; semantic embeddings, an abstractive summarizer and benchmark validation are caller responsibilities. | [source 1](https://arxiv.org/abs/2401.18059) |
| `graph-rag` | experimental | `cheragh.GraphRAGEngine` | Graph-lite baseline; no community detection or global community summaries. | [source 1](https://arxiv.org/abs/2404.16130) |
| `agentic-rag` | experimental | `cheragh.agentic.AgenticRAGEngine` | Inference orchestration only; tools must be registered by the application. | [source 1](https://arxiv.org/abs/2210.03629) |
| `federated` | beta | `cheragh.FederatedRAGEngine` | RRF par défaut ; chaque source demeure responsable de son authentification et de ses ACL. | — |
| `conversational` | beta | `cheragh.ConversationalRAGEngine` | The in-memory store keeps all turns; applications must provide retention for long-lived sessions. | — |
| `sql-rag` | beta | `cheragh.SQLRAGEngine` | SQLite en lecture seule ; les autorisations métier et la pertinence SQL restent à qualifier. | — |
| `multimodal-rag` | experimental | `cheragh.MultimodalRAGEngine` | Bundled CLIP adapter covers text/local images; audio/video need transcripts or a custom encoder. | [source 1](https://arxiv.org/abs/2103.00020) |
| `retrieval-evaluation` | stable | `cheragh.evaluate_retrieval` | Nécessite pertinence annotée, IDs cohérents et un jeu de validation séparé. | — |
| `generation-evaluation` | beta | `cheragh.evaluate_generation` | Diagnostic lexical ; compléter avec ClaimEvaluator et un juge sémantique calibré. | — |
| `claim-evaluation` | experimental | `cheragh.ClaimEvaluator` | Concrete NLIFaithfulnessJudge, LLMFaithfulnessJudge and LLMClaimSegmenter are available; the default lexical fallback is only a diagnostic. Calibrate against human labels. | [source 1](https://arxiv.org/abs/2309.15217), [source 2](https://arxiv.org/abs/2408.08067) |
| `access-controlled-rag` | beta | `cheragh.AccessControlledRAGEngine` | Dériver Principal d’une identité fiable ; vérifier les droits réels avant déploiement. | — |
| `community-graphrag` | experimental | `cheragh.CommunityGraphRAGEngine` | LLMGraphExtractor, entity embeddings, hierarchical Leiden, local context and global map-reduce are available. Qualify extraction/report quality and benchmark performance. | [source 1](https://arxiv.org/abs/2404.16130) |
| `colpali` | experimental | `cheragh.ColPaliRetriever` | Exact in-memory scoring; the official model adapter is optional and has heavyweight dependencies. | [source 1](https://arxiv.org/abs/2407.01449) |
| `temporal-rag` | experimental | `cheragh.TemporalRetriever` | TimeR4Retriever additionally implements FKS/TKS retrieval, grounded rewriting and temporal reranking. Reliable facts and a trained temporal encoder remain required. | [source 1](https://aclanthology.org/2024.emnlp-main.394/) |
| `retrieval-training` | experimental | `cheragh.RetrievalTrainingPipeline` | TorchRetrievalTrainer and TransformersGenerativeTrainer perform real optimization; RankRAGModel ranks and answers with shared weights. Supply domain data/checkpoints; distributed training is not implemented. | [source 1](https://arxiv.org/abs/2104.08051), [source 2](https://arxiv.org/abs/2403.10131), [source 3](https://arxiv.org/abs/2407.02485) |

## Écarts délibérés et modèles requis

- **Self-RAG** : `SegmentedSelfRAGEngine` et `TransformersSelfRAGDecoder` réalisent l’inférence par segments et faisceau avec de vrais logits. Le moteur historique `SelfRAGEngine` conserve sa boucle de critique/révision. Aucun entraînement du critic ni distribution de poids n’est fourni. Les scores positifs sont multipliés en log-space ; les scores non positifs sont écartés explicitement pour éviter des inversions de classement par produit signé. Voir le [contrat détaillé](../src/cheragh/self_rag/README.md).
- **Adaptive-RAG** : résultats des trois stratégies → labels A/B/C → optimisation du classifieur. Le coût peut être mesuré par l’application. Les tests prouvent l’apprentissage et le round-trip, pas le regret de routage sur des questions inédites. [Papier](https://arxiv.org/abs/2403.14403), [code des auteurs](https://github.com/starsuzi/Adaptive-RAG).
- **LongRAG** : groupes du graphe documentaire, score maximum des fragments et lecteur long à une/deux étapes. Le packer historique reste disponible séparément. Les compteurs exacts des modèles sont à injecter et l’encodeur ne doit pas tronquer les fragments. [Méthode](https://arxiv.org/html/2406.15319v1).
- **GraphRAG/RAPTOR** : extraction, clustering, résumés et parcours sont implémentés, avec provenance complète des preuves dérivées. Les LLM/embeddings, la qualité des extractions et le coût à l’échelle restent à évaluer. La classe GraphRAG-lite conserve sa portée distincte. [GraphRAG](https://arxiv.org/html/2404.16130v2), [RAPTOR](https://arxiv.org/html/2401.18059v1).
- **CRAG** : le cross-encoder concret remplace le T5 evaluator entraîné des auteurs. Une calibration sur paires annotées est requise ; la calibration miniature du test ne convient pas automatiquement à un corpus métier. Activer `preserve_correction_sources=True` pour conserver les deux pools d’une correction ambiguë. [Méthode](https://arxiv.org/abs/2401.15884).
- **TimeR4** : faits temporels sourcés, deux espaces FKS/TKS, réécriture ancrée et score sémantique/temporel. L’application fournit ou entraîne le checkpoint TKS ; les intervalles généraux et politiques ACL sont des extensions. Les corruptions temporelles s’intègrent au trainer contrastif, sans reproduire l’entraînement complet des auteurs. [Papier](https://aclanthology.org/2024.emnlp-main.394/), [code](https://github.com/qianxinying/TimeR4).
- **RAFT/RankRAG** : les trainers modifient réellement les poids causal/seq2seq ; RankRAG emploie le même modèle pour P(True) et la réponse. Les recettes ne livrent pas les corpus complets ni un checkpoint métier. Pas d’entraînement distribué ou de benchmark reproduit. [RAFT](https://arxiv.org/abs/2403.10131), [RankRAG](https://arxiv.org/abs/2407.02485).
- **Chain-of-Note** : trois modes de notes, preuves vérifiées et abstention par prompts. Le fine-tuning sur les notes du papier n’est pas reproduit. [Papier](https://arxiv.org/abs/2311.09210).

## Qualification d’une configuration

Choisir des poids et tokenizers versionnés, un corpus et des questions annotées hors entraînement. Mesurer rappel, qualité des réponses et citations, taux d’abstention, latence, mémoire et coût ; comparer à un RAG simple. Vérifier les ACL, mises à jour, sauvegardes et pannes dans cette configuration. Ces résultats sont distincts de la validation logicielle rapportée ici. Voir les guides [production](production.md) et [serveur](production_server.md).
