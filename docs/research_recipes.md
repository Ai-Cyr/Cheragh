# Recettes des composants de recherche

Ces exemples correspondent au code de ce dépôt. Ils montrent les chemins appris et les algorithmes ajoutés, sans revendiquer les résultats des publications. Chaque bloc définit des fonctions composables : leurs arguments sont les documents, modèles, clients et annotations fournis par l’application. Définir ces fonctions ne télécharge aucun modèle et ne lance aucun entraînement.

Les documents doivent porter des `doc_id` stables et uniques. Fournissez uniquement les preuves autorisées aux composants qui n’acceptent pas directement un principal. Un filtre posé après un appel LLM ne protège pas les données déjà envoyées à ce modèle. Les étapes d’indexation, d’extraction et d’entraînement doivent elles aussi disposer des droits nécessaires sur leurs entrées.

## Dépendances et modèles

Depuis ce checkout, installez seulement les extras utiles, par exemple :

```bash
python -m pip install -e ".[learned-retrieval,self-rag,training-generative,evaluation-semantic]"
```

| Chemin | Extra nécessaire |
|---|---|
| SPLADE et ColBERT avec checkpoints Transformers | `learned-retrieval` |
| Décodeur Self-RAG Transformers | `self-rag` |
| Classifieur Adaptive-RAG appris | `research` |
| CRAG avec cross-encoder Transformers | `research` ou `learned-retrieval` |
| Entraînement RAFT/RankRAG | `training-generative` |
| Évaluation des affirmations avec NLI | `evaluation-semantic` |
| Évaluation avec un client LLM fourni | Extra de ce client, si nécessaire |
| LongRAG et TimeR4 avec encodeurs fournis | Aucun extra algorithmique ; ceux des encodeurs/clients utilisés |
| Encodeurs `SentenceTransformerEmbedding` | `local` |
| RAPTOR avec UMAP/GMM | `raptor` |
| GraphRAG avec Leiden | `graphrag` |
| Entraînement contrastif des encodeurs | `training` |

Les imports des adaptateurs restent possibles sans torch, Transformers ou Sentence Transformers. Ces bibliothèques sont chargées lors de l’utilisation des backends correspondants. Les exemples de chargement emploient `local_files_only=True` : `checkpoint` désigne un dossier local ou un identifiant déjà présent dans le cache Hugging Face. Acquérez et versionnez explicitement les poids et tokenizers appropriés. Un checkpoint de chat ordinaire ne remplace pas des poids entraînés pour Self-RAG, ColBERT ou SPLADE.

Les limites de contexte et les compteurs doivent correspondre aux modèles réellement utilisés. Un compteur d’embedding peut différer du compteur du lecteur. Les budgets de contexte seuls ne comprennent pas nécessairement les instructions, la question et la sortie : réservez leur place dans la fenêtre du modèle.

## SPLADE et ColBERT

```python
from cheragh.base import Document
from cheragh.retrieval.learned import (
    ColBERTRetriever, ColBERTTokenEncoder, LearnedSparseRetriever, SPLADEEncoder,
)


def construire_splade(passages: list[Document], checkpoint: str):
    encoder = SPLADEEncoder(
        model_name=checkpoint,
        pooling="max",
        max_length=256,
        device="cpu",
        model_kwargs={"local_files_only": True},
    )
    return LearnedSparseRetriever(passages, encoder=encoder, batch_size=16)


def construire_colbert(passages: list[Document], checkpoint: str):
    encoder = ColBERTTokenEncoder(
        model_name=checkpoint,
        query_max_length=32,
        document_max_length=180,
        projection_dimension=128,
        device="cpu",
        model_kwargs={"local_files_only": True},
    )
    return ColBERTRetriever(
        passages, token_encoder=encoder, batch_size=16, score_batch_size=32,
    )
```

Les objets retournés exposent `retrieve(question, top_k=5)` et se composent avec `RAGEngine`. SPLADE applique le pooling `max(log1p(ReLU(logits)))` avec masque, puis un produit scalaire sparse. ColBERT utilise les marqueurs query/document, la projection apprise, l’augmentation MASK et le MaxSim.

Les formats de référence sont ceux de `naver/splade-cocondenser-ensembledistil` et de `colbert-ir/colbertv2.0`. Le second adaptateur exige les poids BERT **et** la projection ColBERT ; un simple encodeur de tokens Sentence Transformer n’est pas équivalent. Ces backends tronquent aux longueurs configurées : découpez vos passages avant indexation et vérifiez aussi les questions longues. Le calcul exact en mémoire n’est pas un index distribué destiné à des millions de passages.

## Self-RAG avec tokens de réflexion

```python
from cheragh.base import BaseRetriever
from cheragh.self_rag import SegmentedSelfRAGEngine, TransformersSelfRAGDecoder


def construire_selfrag(retriever_autorise: BaseRetriever, checkpoint: str):
    decoder = TransformersSelfRAGDecoder.from_pretrained(
        checkpoint,
        model_kwargs={"local_files_only": True},
        tokenizer_kwargs={"local_files_only": True},
    )
    return SegmentedSelfRAGEngine(
        retriever_autorise,
        decoder,
        top_k=5,
        beam_width=2,
        max_segments=6,
        max_new_tokens=256,
        max_model_calls=128,
        max_retrieval_calls=32,
        max_generated_tokens=8192,
        retrieval_threshold=0.2,
    )
```

Appelez `engine.ask(question)` ; le résultat contient `answer`, `segments`, `documents`, `log_score` et `trace`. Les distributions viennent des logits aux positions des tokens de réflexion. Les décisions Retrieve/No Retrieval/Continue conservent l’état de chaque branche ; les scores de segments positifs sont multipliés en espace logarithmique. Les arrêts sur budgets sont exposés dans le résultat.

Le checkpoint doit avoir appris les tokens de réflexion, comme les checkpoints des [auteurs de Self-RAG](https://github.com/AkariAsai/self-rag). L’adaptateur décode chaque branche de manière gloutonne ; le faisceau compare les continuations conditionnées sur les passages. Les citations indiquent les passages utilisés, sans prouver chaque inférence. Le chargement par défaut se fait selon les paramètres Transformers ; pour un modèle déjà placé sur votre matériel, utilisez `TransformersSelfRAGDecoder(model, tokenizer)`.

## Adaptive-RAG : labels issus des réponses et classifieur entraîné

```python
from cheragh.adaptive import AdaptiveRAGEngine, AdaptiveRAGRoute
from cheragh.adaptive_learning import (
    AdaptiveSilverDatasetBuilder, AdaptiveTrainingQuestion,
    TransformersComplexityClassifier,
)


def entrainer_routage(strategies, questions, model, tokenizer, destination):
    # strategies : dictionnaire comportant exactement les trois routes.
    # Chaque valeur est callable(query) ou expose ask(query).
    # questions : séquence d'AdaptiveTrainingQuestion avec réponses de référence.
    collector = AdaptiveSilverDatasetBuilder(strategies)
    exemples = [example for question in questions
                if (example := collector.collect(question)) is not None]
    classifier = TransformersComplexityClassifier(
        model, tokenizer, max_input_tokens=384,
    )
    report = classifier.fit(
        exemples, epochs=3, batch_size=8, learning_rate=3e-5,
    )
    classifier.save_pretrained(destination)
    return classifier, report


def construire_adaptatif(single_step_engine, iterative_engine, llm, classifier):
    return AdaptiveRAGEngine(
        single_step_engine,
        iterative_engine=iterative_engine,
        llm_client=llm,
        classifier=classifier,
        fallback_to_single_step=False,
    )


def question_annotee(question: str, reponses: tuple[str, ...]):
    return AdaptiveTrainingQuestion(question, reponses)


def strategies_routes(direct, single_step, iterative):
    return {
        AdaptiveRAGRoute.NO_RETRIEVAL: direct,
        AdaptiveRAGRoute.SINGLE_STEP: single_step,
        AdaptiveRAGRoute.ITERATIVE: iterative,
    }
```

La collecte **exécute les trois stratégies** pour chaque question. Par défaut, elle sélectionne la stratégie correcte la plus simple ; un évaluateur et une fonction de coût peuvent être fournis au constructeur. Si toutes les stratégies échouent, l’exemple est omis sauf prior de dataset explicite. L’égalité exacte normalisée par défaut doit être remplacée si elle n’est pas adaptée à votre tâche.

Le modèle fourni est soit un T5/seq2seq compatible avec les tokens A/B/C, soit un classifieur à trois logits. L’entraînement met à jour ses paramètres. Les questions sont tronquées à `max_input_tokens`, comme dans le chemin de référence. Séparez collecte, entraînement, validation et test ; une précision d’entraînement n’est pas une preuve de généralisation. Les probabilités A/B/C ne sont pas des probabilités calibrées de réussite de la réponse. Pour recharger, utilisez `TransformersComplexityClassifier.from_pretrained(destination)`.

## LongRAG : regroupement et lecture des documents complets

```python
from cheragh.long_rag import LongRAGEngine, LongRAGRetriever


def construire_longrag(documents, embeddings, llm, adjacency,
                       compter_embedding, compter_lecteur):
    retriever = LongRAGRetriever(
        documents,
        embeddings,
        adjacency=adjacency,
        max_group_size=5,
        chunk_tokens=512,
        token_counter=compter_embedding,
    )
    return LongRAGEngine(
        retriever,
        llm,
        token_counter=compter_lecteur,
        max_input_tokens=32000,
        max_output_tokens=1500,
        short_answer=True,
        short_answer_max_tokens=256,
        budget_policy="raise",
    )


def repondre_longrag(engine, question, principal, policy, tenant_id, collection_id):
    return engine.ask(
        question, top_k=4, principal=principal, access_policy=policy,
        tenant_id=tenant_id, collection_id=collection_id,
    )
```

`adjacency` est un dictionnaire `doc_id -> liste de doc_id liés` ; les références doivent appartenir au corpus. Les liens sont utilisés tels quels : ajoutez les deux directions pour représenter une relation non orientée. `max_group_size` compte les documents et `top_k` compte les groupes. Le regroupement suit l’[algorithme 1 de LongRAG](https://arxiv.org/html/2406.15319v1) ; le score est le maximum des produits scalaires sur **tous** les petits fragments du groupe, sans normalisation ajoutée par le retriever.

L’autorisation et la portée tenant/collection précèdent le regroupement et le classement. Les sources originales complètes arrivent au lecteur. `retrieve_groups()` expose les groupes et leurs sources ; `retrieve()` fournit des documents agrégés pour le protocole standard. Utilisez l’autorisation interne avant toute agrégation sensible. Le mode `skip_groups` peut remplacer `raise` si vous acceptez l’exclusion de groupes entiers, enregistrée dans les métadonnées. La seconde passe condense la première réponse citée ; elle dispose de son propre budget de sortie.

## CRAG : cross-encoder calibré et raffinement sémantique

```python
from cheragh.context_packing import LongContextPacker
from cheragh.corrective import CorrectiveRAGEngine
from cheragh.corrective.semantic import (
    CrossEncoderRetrievalGrader, SemanticKnowledgeRefiner,
)
from cheragh.engine import RAGEngine


def construire_crag(retriever_autorise, external_retriever_autorise, llm,
                    checkpoint, paires_validation, labels_validation,
                    compter_lecteur):
    # paires_validation : [(question, Document), ...]
    # labels_validation : [0 ou 1, ...], avec les deux classes représentées.
    grader = CrossEncoderRetrievalGrader(
        model_name=checkpoint,
        local_files_only=True,
        correct_threshold=0.7,
        incorrect_threshold=0.3,
        max_input_tokens=512,
        max_windows_per_document=64,
    )
    calibration = grader.calibrate(paires_validation, labels_validation)
    refiner = SemanticKnowledgeRefiner(
        grader,
        sentences_per_strip=2,
        min_relevance=0.5,
        max_refined_tokens=4000,
        token_counter=compter_lecteur,
    )
    base = RAGEngine(
        retriever_autorise,
        llm_client=llm,
        strict_grounding=True,
        context_packer=LongContextPacker(
            token_budget=6000, token_estimator=compter_lecteur,
            truncate_oversized=False,
        ),
    )
    engine = CorrectiveRAGEngine(
        base_engine=base,
        retrieval_grader=grader,
        knowledge_refiner=refiner,
        external_retriever=external_retriever_autorise,
        preserve_correction_sources=True,
        max_retries=0,
    )
    return engine, calibration
```

Appelez `engine.ask(question, top_k=5, max_tokens=1000)`. Une calibration logistique ajustée est obligatoire avant les scores de confiance ; ses paramètres peuvent aussi être fournis via `LogisticCalibration(scale, bias)`. Les seuils ci-dessus sont illustratifs et doivent être validés sur vos données. La calibration ne doit pas utiliser le jeu de test.

Le maximum des probabilités documentaires déclenche Correct, Ambiguous ou Incorrect. Le même modèle filtre les passages, recomposés dans leur ordre d’origine avec leurs offsets. `preserve_correction_sources=True` garde les preuves internes et externes en cas ambigu ; `top_k` limite alors chaque récupération et le packer limite le contexte combiné. Le moteur ne configure aucun fournisseur de recherche web : injectez le retriever externe approprié.

Ce chemin est une **substitution cross-encoder** à l’évaluateur T5 de [CRAG](https://arxiv.org/html/2401.15884v3). Le petit modèle `cross-encoder/ms-marco-TinyBERT-L2-v2` permet un smoke test économique mais montre des faux négatifs sur certaines paraphrases. Pour les documents longs, toutes les fenêtres sont évaluées et leur maximum est utilisé ; calibrez aussi ce comportement sur la distribution de longueurs réelle.

## Affirmations : décomposition, support et contradiction

```python
from cheragh.evaluation.claims import ClaimEvaluator
from cheragh.evaluation.semantic import (
    LLMClaimSegmenter, LLMFaithfulnessJudge, NLIFaithfulnessJudge,
)


def evaluer_avec_llm(answer, sources_autorisees, llm):
    evaluator = ClaimEvaluator(
        segmenter=LLMClaimSegmenter(
            llm, generation_kwargs={"max_tokens": 2000},
        ),
        scorer=LLMFaithfulnessJudge(
            llm, max_evidence_chars=8000, overlap_chars=512,
            generation_kwargs={"max_tokens": 1500},
        ),
    )
    return evaluator.evaluate(answer, sources_autorisees)


def evaluer_avec_nli(answer, sources_autorisees, llm_decomposition,
                     checkpoint, label_mapping=None):
    judge = NLIFaithfulnessJudge(
        model_name=checkpoint,
        label_mapping=label_mapping,
        max_length=512,
        overlap_tokens=64,
        batch_size=8,
        device="cpu",
        model_kwargs={"local_files_only": True},
    )
    evaluator = ClaimEvaluator(
        segmenter=LLMClaimSegmenter(
            llm_decomposition, generation_kwargs={"max_tokens": 2000},
        ),
        scorer=judge,
        support_threshold=0.7,
        contradiction_threshold=0.7,
    )
    return evaluator.evaluate(answer, sources_autorisees)
```

`label_mapping` doit refléter le checkpoint : clés `entailment`, `contradiction`, `neutral`, avec leurs indices distincts 0/1/2. Il peut être omis si `config.id2label` les nomme sans ambiguïté. Ne copiez pas l’ordre d’un autre modèle.

La décomposition LLM exige des extraits exacts de la réponse et les citations associées. Le juge LLM exige des extraits de preuve pour ses décisions positives ; le juge NLI couvre le texte par fenêtres. Support et contradiction sont conservés indépendamment, ce qui permet de signaler des preuves contradictoires. Les seuils NLI demandent une validation métier. Ces diagnostics ne reproduisent pas les suites complètes RAGAS/RAGChecker et ne constituent pas une preuve de vérité. Le découpage par fenêtres peut manquer une inférence répartie entre plusieurs fenêtres. La réponse complète et chaque affirmation doivent tenir dans les budgets des modèles concernés.

## TimeR4 : réécriture temporelle ancrée et double retrieval

```python
from cheragh.temporal.time_r4 import (
    TemporalConstraint, TemporalFact, TemporalInterval, TimeR4Retriever,
)


def fait_source(fact_id, subject, predicate, object_name, start, end, source):
    return TemporalFact(
        fact_id, subject, predicate, object_name,
        TemporalInterval(start, end), source,
    )


def rechercher_temporel(facts, fact_encoder, temporal_encoder, llm,
                       principal, policy, question, periode):
    retriever = TimeR4Retriever(
        facts,
        fact_embedding_model=fact_encoder,
        temporal_embedding_model=temporal_encoder,
        llm=llm,
        principal=principal,
        policy=policy,
        anchor_top_k=5,
        candidate_top_k=50,
        semantic_weight=0.8,
        max_rewrite_chars=24000,
        max_rewrite_tokens=2048,
        rewrite_failure="raise",
    )
    return retriever.retrieve_with_trace(
        question, top_k=5,
        constraints=[TemporalConstraint("during", TemporalInterval.at(periode))],
    )
```

`source` est le document original avec son ID et ses métadonnées d’accès. Un `TemporalFact` représente une assertion du graphe ; sa construction ne vérifie pas automatiquement que la source implique cette assertion. Fournissez des faits validés.

Les encodeurs FKS et TKS ont des rôles distincts. Fournir deux encodeurs génériques n’équivaut pas à entraîner TKS au temps. `from_pretrained(..., temporal_model_name=...)` charge des checkpoints Sentence Transformer avec l’extra `local`. `build_temporal_training_example` peut construire les corruptions de temps, contenu et contenu+temps pour `TorchRetrievalTrainer`, à partir d’un positif et de négatifs connus.

Le principal et la policy s’appliquent aux sources **avant** l’envoi des faits au LLM. Les contraintes applicatives sont figées et intersectées avec celles de la réécriture ; utilisez une liste ou un tuple. Les années/mois/jours ISO désignent leur période UTC entière ; les timestamps doivent être zonés. Les intervalles sont fermés, avec bornes ouvertes explicites. Une réécriture non ancrée échoue. Le plafond de candidats peut limiter le rappel et apparaît dans la trace. Ce composant fournit le retrieval de [TimeR4](https://aclanthology.org/2024.emnlp-main.394/), sans les poids temporels entraînés de la publication ni une réponse finale automatique.

## RAFT : construction de cibles vérifiées puis SFT

```python
from cheragh.training import RAFTDatasetBuilder, TransformersGenerativeTrainer


def entrainer_raft(examples, teacher, model, tokenizer, destination):
    # examples : RetrievalTrainingExample avec réponse, oracle(s) et distracteurs.
    # teacher(question, oracle_documents, verified_answer) -> RAFTGeneratedAnswer.
    records = RAFTDatasetBuilder(
        oracle_probability=0.8,
        seed=42,
        shuffle_documents=True,
        context_document_count=5,
        answer_generator=teacher,
    ).build(examples)
    trainer = TransformersGenerativeTrainer(
        model, tokenizer, max_input_tokens=4096, max_target_tokens=512,
    )
    report = trainer.fit(
        records, epochs=1, batch_size=2, learning_rate=2e-5,
    )
    trainer.save_pretrained(destination)
    return trainer, report
```

Le teacher reçoit les oracles, même quand ils sont retirés du prompt d’entraînement. Il doit retourner `RAFTGeneratedAnswer(answer, rationale)` avec des extraits exacts entre `##begin_quote##` et `##end_quote##`. Le builder vérifie ces extraits et l’accord de la réponse avec l’annotation. Cela ne valide pas toutes les inférences de la justification.

Pour cette configuration, fournissez cinq distracteurs disponibles par exemple afin que le cas sans oracle puisse toujours remplir le contexte ; le nombre d’oracles inclus doit laisser une place aux distracteurs. Les cibles restent inchangées lorsque l’oracle est absent du prompt. Le trainer refuse par défaut les données sans justification citée ou sans distracteur. Les ablations demandent `GenerativeTrainingExample.from_raft(..., require_rationale=False, require_distractors=False)` explicitement.

Le SFT entraîne réellement les poids causaux ou seq2seq, masque prompt/padding et conserve la supervision EOS. Il refuse les séquences hors budget. Le matériel, le modèle, le tokenizer, les séparations train/test et le format de chat relèvent de l’application. Il n’implémente ni entraînement distribué, ni LoRA automatique, ni reprise de l’état de l’optimiseur. [Publication RAFT](https://arxiv.org/abs/2403.10131).

## RankRAG : un même modèle pour classer et répondre

```python
from cheragh.training import (
    RankRAGDatasetBuilder, RankRAGEngine, RankRAGModel,
    TransformersGenerativeTrainer,
)


def entrainer_rankrag(retrieval_examples, instruction_examples, sample_count,
                     model, tokenizer, retriever_autorise):
    supervised = [
        row
        for example in retrieval_examples
        for row in RankRAGDatasetBuilder.from_retrieval_example(example, seed=42)
    ]
    # instruction_examples : GenerativeTrainingExample déjà validés.
    mixture = RankRAGDatasetBuilder.blend(
        {"instruction": instruction_examples, "rag": supervised},
        {"instruction": 0.3, "rag": 0.7},
        sample_count=sample_count,
        seed=42,
    )
    trainer = TransformersGenerativeTrainer(
        model, tokenizer, max_input_tokens=4096, max_target_tokens=512,
    )
    report = trainer.fit(mixture, epochs=1, batch_size=2, learning_rate=2e-5)
    engine = RankRAGEngine(
        retriever_autorise,
        RankRAGModel(trainer),
        candidate_top_k=100,
        top_k=5,
        max_new_tokens=256,
    )
    return engine, trainer, report
```

Les exemples de retrieval contiennent des positifs et négatifs annotés. Ils produisent des tâches de classement binaire, de sélection d’indices de passages et, si la réponse est disponible, des tâches QA. Les poids de mélange ci-dessus sont illustratifs : ils ne reproduisent pas les datasets et proportions de la [publication RankRAG](https://arxiv.org/abs/2407.02485).

`engine.ask(question)` classe les candidats par la probabilité de `True` sur le vocabulaire complet, puis génère avec **les mêmes poids**. Le tokenizer doit représenter `True` par un token connu. Les sources retournées sont les passages conditionnant la génération ; le chemin ne garantit pas des citations inline ou l’implication de chaque affirmation. Le modèle causal est le chemin de référence ; seq2seq est une extension explicite. Sauvegardez le trainer avec `save_pretrained` et rechargez-le avec `TransformersGenerativeTrainer.from_pretrained`.

## RAPTOR et GraphRAG communautaire

```python
from cheragh.raptor_engine import RAPTOREngine
from cheragh.community_graph import (
    CommunityGraphRAGEngine, LeidenCommunityDetector, LLMCommunitySummarizer,
    LLMGraphExtractor, LocalGraphSearchConfig,
)


def construire_raptor(documents, embeddings, llm, compter_tokens):
    return RAPTOREngine(
        documents,
        embedding_model=embeddings,
        llm_client=llm,
        clustering_mode="umap_gmm",
        retrieval_mode="paper_tree",
        levels=3,
        summary_input_token_budget=3500,
        summary_max_tokens=100,
        retrieval_token_budget=6000,
        token_estimator=compter_tokens,
    )


def construire_graphrag(documents, embeddings, llm, compter_tokens):
    return CommunityGraphRAGEngine(
        documents,
        llm_client=llm,
        embedding_model=embeddings,
        graph_extractor=LLMGraphExtractor(
            llm, max_gleanings=1, token_counter=compter_tokens,
        ),
        community_detector=LeidenCommunityDetector(max_cluster_size=10),
        summarizer=LLMCommunitySummarizer(llm, token_counter=compter_tokens),
        local_search_config=LocalGraphSearchConfig(
            max_input_tokens=8000, max_output_tokens=1500,
            token_counter=compter_tokens,
        ),
        require_citations=True,
    )
```

La construction effectue l’indexation et appelle le LLM pour les résumés/extractions. RAPTOR conserve les groupes singletons et expose `start_level`/`num_levels` uniquement avec `paper_tree` ; utilisez des niveaux réellement présents dans l’index. Son mode `collapsed` effectue une recherche sur tous les niveaux. Les backends UMAP/GMM demandent l’extra `raptor`.

GraphRAG extrait des entités et relations appuyées sur des citations exactes, puis construit la hiérarchie Leiden et les rapports. `engine.ask_local(question, principal=..., access_policy=...)` utilise la recherche sémantique locale. `engine.ask_global_map_reduce(question, ...)` sélectionne le chemin global map/reduce ; `ask_global` désigne la voie historique sur rapports classés. Les options d’autorisation sont à passer au chemin concerné avant génération. Les dépendances de provenance mixtes sont filtrées de façon conservatrice, ce qui peut réduire la couverture d’un sous-corpus autorisé.

Leiden requiert `graphrag` ; encodeurs et clients LLM restent fournis par l’application. La validation des citations/extraits vérifie la provenance syntaxique, sans garantir une résolution parfaite des entités ou la vérité sémantique. Ces chemins ne couvrent pas DRIFT ni l’extraction de covariates. Voir les publications [RAPTOR](https://arxiv.org/abs/2401.18059) et [GraphRAG](https://arxiv.org/abs/2404.16130).

## Vérification avant utilisation sur votre corpus

Conservez un jeu de validation distinct pour les seuils, budgets et calibrations, puis un jeu de test intact pour mesurer rappel, qualité des réponses, contradictions, coûts et latence. Les tests minuscules de ce dépôt prouvent les mécanismes, les gradients et les contrats d’API ; ils ne prouvent pas les gains sur vos données. Les limites mesurées et les modèles effectivement exécutés sont recensés dans la [matrice de fidélité](research_fidelity.md) et le [rapport de validation](research_validation.md).
