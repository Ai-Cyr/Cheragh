# Architectures RAG — v1.3.0

Cheragh v1.3 approfondit plusieurs composants déjà présents sans transformer chaque baseline en reproduction de son article d'origine. Le catalogue contient **44 techniques disponibles** : 6 stables, 12 bêta et 26 expérimentales. « Disponible » signifie que le point d'entrée existe et respecte les contrats du package ; cela ne garantit ni équivalence scientifique avec un papier, ni qualité de production sur un corpus donné.

## Matrice complète du catalogue

L'état ci-dessous reprend directement les familles, noms, maturités et points d'entrée du catalogue. Il n'existe plus d'entrée `planned` dans cette version.

| Famille | Technique | Maturité | Point d'entrée | Portée réelle |
|---|---|---:|---|---|
| Indexation | Recursive chunking | stable | `cheragh.RecursiveTextChunker` | Découpage récursif sensible aux caractères |
| Indexation | Semantic chunking | bêta | `cheragh.SemanticChunker` | Frontières thématiques guidées par embeddings |
| Indexation | Hierarchical chunking | bêta | `cheragh.HierarchicalChunker` | Chunks parent/enfant avec provenance de section |
| Indexation | Sentence-window retrieval | expérimental | `cheragh.SentenceWindowRetriever` | Recherche d'une phrase puis expansion de sa fenêtre |
| Indexation | Propositional retrieval | expérimental | `cheragh.PropositionalRetriever` | Index de propositions générées, retour au contexte source |
| Retrieval | BM25 | stable | `cheragh.BM25Retriever` | Recherche lexicale sparse seule |
| Retrieval | Dense retrieval | stable | `cheragh.MemoryVectorStore` | Recherche sémantique mono-vecteur en mémoire |
| Retrieval | Hybrid sparse+dense | stable | `cheragh.HybridSearchRetriever` | Fusion pondérée BM25 et dense |
| Retrieval | Cross-encoder reranking | bêta | `cheragh.CrossEncoderReranker` | Reclassement d'un pool de candidats |
| Retrieval | Reciprocal Rank Fusion | bêta | `cheragh.ReciprocalRankFusionRetriever` | RRF canonique sur plusieurs retrievers |
| Retrieval | Maximal Marginal Relevance | expérimental | `cheragh.MMRRetriever` | Compromis pertinence/diversité |
| Retrieval | Learned sparse retrieval (SPLADE) | expérimental | `cheragh.retrieval.SPLADERetriever` | Scoring sparse appris exact, encodeur injectable |
| Retrieval | Late-interaction retrieval (ColBERT) | expérimental | `cheragh.retrieval.ColBERTRetriever` | MaxSim token-à-token exact en mémoire |
| Retrieval | Temporal RAG | expérimental | `cheragh.TemporalRetriever` | Fenêtres de validité, fraîcheur et versions |
| Requête | HyDE | expérimental | `cheragh.HyDERetriever` | Recherche depuis une réponse hypothétique générée |
| Requête | HyQE | expérimental | `cheragh.HyQERetriever` | Index de questions hypothétiques liées aux sources |
| Requête | RAG-Fusion | expérimental | `cheragh.RAGFusionRetriever` | Variantes de requête puis fusion des rangs |
| Requête | Self-query retrieval | expérimental | `cheragh.SelfQueryRetriever` | Requête sémantique et filtres de métadonnées bornés |
| Requête | Step-back prompting | expérimental | `cheragh.StepBackRetriever` | Recherche complémentaire à un niveau plus abstrait |
| Requête | Query decomposition | expérimental | `cheragh.QueryDecompositionRetriever` | Décomposition en sous-questions récupérables |
| Augmentation | Contextual compression | bêta | `cheragh.ContextualCompressionRetriever` | Filtrage de texte non pertinent ou redondant |
| Augmentation | Long-context packing | expérimental | `cheragh.LongContextPacker` | Budget strict, quotas de source et placement aux bords |
| Augmentation | Chain-of-Note | expérimental | `cheragh.ChainOfNoteRetriever` | Notes de preuve avant synthèse |
| Orchestration | Naive RAG | stable | `cheragh.RAGEngine` | Retrieve, augment, generate et citations |
| Orchestration | Corrective RAG | expérimental | `cheragh.CorrectiveRAGEngine` | Évaluation tri-état et correction bornée du contexte |
| Orchestration | Inference-time Self-RAG | expérimental | `cheragh.self_rag.SelfRAGEngine` | Porte, critique et révision à l'inférence |
| Orchestration | FLARE active retrieval | expérimental | `cheragh.FLAREPipeline` | Brouillon look-ahead, incertitude et retrieval actif |
| Orchestration | Adaptive RAG | expérimental | `cheragh.AdaptiveRAGEngine` | Routage sans retrieval, simple ou itératif |
| Orchestration | Parent-child retrieval | bêta | `cheragh.ParentChildRetriever` | Recherche de petits chunks, retour du parent |
| Orchestration | Multi-hop RAG | bêta | `cheragh.MultiHopRAGEngine` | Planning, retrieval et observations bornés |
| Orchestration | RAPTOR | expérimental | `cheragh.RAPTOREngine` | Arbre de résumés, recherche plate ou top-down |
| Orchestration | Graph-enhanced RAG | expérimental | `cheragh.GraphRAGEngine` | Voisinage entités/relations et recherche vectorielle |
| Orchestration | Agentic RAG | expérimental | `cheragh.agentic.AgenticRAGEngine` | Boucle bornée sur outils explicitement enregistrés |
| Orchestration | Federated RAG | bêta | `cheragh.FederatedRAGEngine` | Fusion de plusieurs retrievers ou domaines |
| Orchestration | Conversational RAG | bêta | `cheragh.ConversationalRAGEngine` | Fenêtre bornée de contexte conversationnel |
| Orchestration | Community GraphRAG | expérimental | `cheragh.CommunityGraphRAGEngine` | Communautés, rapports et recherche globale/locale |
| Structuré | SQL RAG | bêta | `cheragh.SQLRAGEngine` | Génération et exécution de requêtes SQLite en lecture seule |
| Multimodal | Multimodal RAG | expérimental | `cheragh.MultimodalRAGEngine` | Retrieval texte/image avec provenance média |
| Multimodal | Visual-document late interaction | expérimental | `cheragh.ColPaliRetriever` | MaxSim token-à-patch sur pages-images |
| Évaluation | Retrieval evaluation | stable | `cheragh.evaluate_retrieval` | Hit rate, MRR, précision, rappel, nDCG, context precision |
| Évaluation | Generation evaluation | bêta | `cheragh.evaluate_generation` | Citations et diagnostics lexicaux déterministes |
| Évaluation | Claim-level faithfulness evaluation | expérimental | `cheragh.ClaimEvaluator` | Support, contradiction et alignement citation-preuve |
| Évaluation | Retriever/RAG training | expérimental | `cheragh.RetrievalTrainingPipeline` | Données, négatifs difficiles, distillation et frontière trainer |
| Gouvernance | Access-controlled RAG | bêta | `cheragh.AccessControlledRAGEngine` | Politiques tenant, collection, rôle et classification |

Totaux par famille : indexation 5, retrieval 9, requête 6, augmentation 3, orchestration 13, structuré 1, multimodal 2, évaluation 4 et gouvernance 1.

## 1. LongContextPacker intégré à RAGEngine

`LongContextPacker` se place désormais à la frontière de génération de `RAGEngine`. L'ordre réel est : transformation de requête, retrieval, seuil de score, compression éventuelle, **packing**, génération. `ask()` et `stream_with_response()` appliquent le même packing.

Le packer :

- déduplique les preuves et les sélectionne par score, avec ordre stable en cas d'égalité ;
- compte le texte réellement envoyé, y compris séparateurs et en-têtes `[source: id]` ;
- impose un budget global et, facultativement, un budget par source ;
- propose les ordres `relevance`, `input` et `lost_in_the_middle` ;
- peut tronquer un document trop grand et ajuster les offsets de provenance usuels ;
- renvoie les documents retenus, le contexte rendu et des diagnostics sérialisables dans `response.metadata["context_packing"]` et la trace.

```python
from cheragh import (
    Document,
    HashingEmbedding,
    LongContextPacker,
    RAGEngine,
    StaticLLMClient,
)

documents = [
    Document("Alpha décrit la règle principale.", metadata={"source": "A"}, doc_id="a"),
    Document("Beta ajoute une exception vérifiée.", metadata={"source": "B"}, doc_id="b"),
]
packer = LongContextPacker(
    token_budget=40,
    per_source_token_budget=24,
    ordering="lost_in_the_middle",
)
engine = RAGEngine.from_documents(
    documents,
    embedding_model=HashingEmbedding(64),
    retriever_type="memory",
    llm_client=StaticLLMClient("Réponse [source: a]"),
    context_packer=packer,
)

response = engine.ask("Quelle est la règle ?", top_k=2)
print([doc.doc_id for doc in response.retrieved_documents])
print(response.metadata["context_packing"]["tokens_used"])
```

Si le packing élimine tout le contexte et que `strict_grounding=True`, le moteur n'appelle pas le LLM et retourne le fallback structuré avec `context_packing_empty`.

**Écart avec [LongRAG](https://arxiv.org/abs/2406.15319).** LongRAG regroupe le corpus en unités longues et associe un long retriever à un long reader. Cheragh sélectionne et ordonne des preuves **déjà récupérées** ; il ne construit pas les unités 4K du papier, ne fournit pas son retriever/reader et ne reproduit pas ses résultats. Le budget n'est exact que par rapport à `token_estimator` : injecter le tokenizer du modèle cible est nécessaire pour une limite modèle-exacte.

## 2. RAPTOR : recherche plate ou parcours d'arbre borné

`RAPTOREngine` construit récursivement des feuilles et des nœuds de résumé. Deux modes de retrieval sont exposés :

- `flat` reste le comportement compatible : tous les nœuds sont candidats à la recherche vectorielle ;
- `tree` effectue une vraie traversée top-down, sélectionne au plus `beam_width` nœuds par profondeur et ne visite jamais plus de `traversal_budget` nœuds.

Le score d'un candidat en mode arbre est la moyenne des similarités sur son chemin racine-nœud. Si le budget s'épuise avant une feuille, le front le plus profond est retourné. Les métadonnées indiquent le chemin, les niveaux, les scores du chemin, le nombre de nœuds visités/scorés et si le résultat est terminal.

```python
from cheragh import Document, RAPTOREngine

engine = RAPTOREngine.from_documents(
    [
        Document("Les pommes sont des fruits.", doc_id="fruit-1"),
        Document("Les poires sont des fruits.", doc_id="fruit-2"),
        Document("Les photons relèvent de la physique.", doc_id="physique-1"),
    ],
    levels=1,
    branching_factor=2,
    min_cluster_size=2,
    retrieval_mode="tree",
    beam_width=2,
    traversal_budget=6,
)

hits = engine.retrieve("Quel texte parle de pommes ?", top_k=2)
for hit in hits:
    print(hit.doc_id, hit.metadata["raptor_path"])
```

`beam_width` et `traversal_budget` peuvent aussi être surchargés par appel à `retrieve()`. L'index et les résultats sont des snapshots défensifs.

**Écart avec [RAPTOR](https://arxiv.org/abs/2401.18059).** Le papier construit un arbre par embeddings, clustering et résumés abstraits récursifs, notamment avec UMAP/GMM et appartenance souple. Cheragh emploie un groupement glouton déterministe et des résumeurs injectables ; le fallback n'est pas un modèle RAPTOR entraîné. Le beam search et le budget sont des contrôles d'inférence Cheragh, pas une reproduction des configurations expérimentales du papier.

## 3. AdaptiveRAGEngine : trois routes explicites

`AdaptiveRAGEngine` choisit une stratégie avant exécution :

| Route | Exécution |
|---|---|
| `NO_RETRIEVAL` | Appel direct au LLM, sans source, avec le warning `adaptive_rag_no_retrieval` |
| `SINGLE_STEP` | Délégation au moteur RAG simple fourni |
| `ITERATIVE` | Délégation au moteur itératif fourni, typiquement `MultiHopRAGEngine` |

La décision contient route, confiance facultative et justification. La réponse enregistre route demandée et route exécutée. Si le moteur itératif manque, le comportement par défaut retombe explicitement sur le moteur simple ; `fallback_to_single_step=False` transforme ce cas en erreur.

```python
from cheragh import Document, HashingEmbedding, RAGEngine, StaticLLMClient
from cheragh.adaptive import AdaptiveRAGEngine, HeuristicComplexityClassifier
from cheragh.multihop import MultiHopRAGEngine, RuleBasedMultiHopPlanner

llm = StaticLLMClient("Réponse documentée [source: a]")
single = RAGEngine.from_documents(
    [Document("Alpha et Beta appliquent des règles différentes.", doc_id="a")],
    embedding_model=HashingEmbedding(64),
    retriever_type="memory",
    llm_client=llm,
)
iterative = MultiHopRAGEngine(
    single.retriever,
    llm_client=llm,
    planner=RuleBasedMultiHopPlanner(),
    max_steps=2,
)
adaptive = AdaptiveRAGEngine(
    single,
    iterative_engine=iterative,
    llm_client=llm,
    classifier=HeuristicComplexityClassifier(),
)

response = adaptive.ask("Compare Alpha et Beta et explique leur impact")
print(response.metadata["adaptive_rag"])
```

`LLMComplexityClassifier` fournit une autre frontière, avec repli heuristique si son label n'est pas reconnu. Un classifieur métier peut être injecté directement.

**Écart avec [Adaptive-RAG](https://arxiv.org/abs/2403.14403).** L'article apprend un classifieur de complexité à partir de labels automatiquement collectés selon les résultats de stratégies candidates. Cheragh fournit les trois routes et leurs contrats, mais pas ce jeu de labels ni ce classifieur entraîné. `HeuristicComplexityClassifier` est une baseline déterministe ; la qualité de la route itérative dépend entièrement du moteur configuré.

## 4. FLARE : incertitude injectable et réponse structurée

`FLAREPipeline` génère un brouillon de la prochaine phrase, évalue son incertitude, récupère des preuves si nécessaire, puis régénère la phrase avec contexte. La boucle est limitée par `max_iterations`.

`TokenConfidenceUncertaintyEstimator` adapte les log-probabilités ou scores de confiance d'un fournisseur en `TokenConfidence`. La recherche se concentre sur les spans sous le seuil. Un `DraftUncertaintyEstimator` métier peut décider directement. Sans signal de confiance, `LengthBasedDraftUncertainty` conserve une heuristique textuelle explicite.

```python
from cheragh import BM25Retriever, Document, LLMClient
from cheragh.flare import FLAREPipeline, TokenConfidence, TokenConfidenceUncertaintyEstimator

class QueueLLM(LLMClient):
    def __init__(self):
        self.outputs = iter([
            "Le délai est peut-être de 30 jours.",
            "Le délai vérifié est de 14 jours. [source: contrat]",
            "[DONE]",
        ])

    def generate(self, prompt, **kwargs):
        return next(self.outputs)

retriever = BM25Retriever([
    Document("Le contrat fixe le délai à 14 jours.", doc_id="contrat")
])
uncertainty = TokenConfidenceUncertaintyEstimator(
    lambda draft: [TokenConfidence("peut-être de 30 jours", 0.12)],
    threshold=0.5,
)
pipeline = FLAREPipeline(retriever, QueueLLM(), uncertainty_estimator=uncertainty)

response = pipeline.ask("Quel est le délai ?", top_k=1)
print(response.answer)
print(response.metadata["iterations"][0]["low_confidence_spans"])
```

`run()` reste compatible et retourne un dictionnaire ; `ask()` retourne un `RAGResponse` avec sources, citations et diagnostics par itération. Comme plusieurs prompts produisent la réponse, `response.prompt` est volontairement vide et `multi_prompt_generation=True` le signale.

**Écart avec [FLARE](https://arxiv.org/abs/2305.06983).** L'article détecte les tokens de faible probabilité dans une phrase anticipée et régénère avec retrieval actif. Cheragh expose ce point de contrôle, mais un `LLMClient` texte seul ne donne pas de log-probabilités : l'adaptateur fournisseur doit les fournir. Le fallback fondé sur la longueur n'est pas une mesure d'incertitude et ne doit pas être présenté comme FLARE complet.

## 5. Multi-hop dynamique : planifier, récupérer, observer

Le chemin historique reste disponible avec `decomposer=` ou avec le décomposeur statique par défaut. Fournir `planner=` active une boucle dynamique :

1. le planner reçoit un `PlanningContext` contenant la question, le prochain numéro d'étape, le budget, les hops et les preuves observées sous forme de snapshots ;
2. il retourne un `PlanningDecision.next(query, ...)` ou `PlanningDecision.stop(...)` ;
3. le moteur récupère les documents, construit une observation, fusionne les doublons et rappelle le planner ;
4. `max_steps` est une limite stricte. Si elle est atteinte, un STOP `max_steps_reached` est ajouté sans appel de planning supplémentaire.

```python
from cheragh import BM25Retriever, Document, StaticLLMClient
from cheragh.multihop import MultiHopRAGEngine, RuleBasedMultiHopPlanner

retriever = BM25Retriever([
    Document("Alpha utilise le produit Orion.", doc_id="alpha"),
    Document("Orion dépend du fournisseur Delta.", doc_id="orion"),
])
engine = MultiHopRAGEngine(
    retriever,
    llm_client=StaticLLMClient("Delta est le fournisseur. [source: orion]"),
    planner=RuleBasedMultiHopPlanner(),
    max_steps=3,
    top_k_per_step=2,
)

result = engine.ask("Quel fournisseur dépend du produit utilisé par Alpha ?")
print([decision.action.value for decision in result.planning_decisions])
print(result.response.metadata["stop_reason"])
print(result.retrieved_documents[0].metadata["multi_hop_provenance"])
```

`LLMMultiHopPlanner` accepte une décision JSON stricte et retombe sur le planner déterministe si la sortie est invalide. Les preuves fusionnées consignent étapes, requêtes, scores et occurrences. `retrieve()` peut effectuer le planning et les recherches, mais n'appelle jamais le LLM de **synthèse finale**.

**Écart avec [IRCoT](https://arxiv.org/abs/2212.10509) et [ReAct](https://arxiv.org/abs/2210.03629).** Cheragh reprend la forme observable raison/action/observation et l'entrelacement retrieval-raisonnement. Il ne reproduit ni leurs prompts, ni leurs traces de chain-of-thought, ni une politique entraînée, ni leurs protocoles d'évaluation. Le planner déterministe extrait seulement des suivis et termes-ponts transparents ; la qualité sémantique avancée doit être injectée.

## 6. ClaimEvaluator : faithfulness et citations séparées

`ClaimEvaluator` segmente la réponse, score chaque claim contre les preuves, puis évalue séparément si les citations attachées pointent vers des preuves qui soutiennent réellement le claim.

```python
from cheragh import Document
from cheragh.evaluation import evaluate_claims

result = evaluate_claims(
    "Paris est la capitale de la France [source: france]. "
    "La Lune est faite de fromage [source: inconnu].",
    [Document("Paris est la capitale de la France.", doc_id="france")],
)

print(result.metrics)
print(result.supported_claims)
print(result.unsupported_claims)
print(result.unknown_citations)
```

Les frontières sont injectables :

- `ClaimSegmenter` peut remplacer le découpage déterministe par phrases par une décomposition en faits atomiques ;
- `EvidenceEntailmentScorer` peut fournir support et contradiction via NLI, LLM ou règles métier ;
- les diagnostics conservent le claim, son statut, les citations connues/inconnues, l'alignement par citation et les meilleures preuves ;
- les documents, métadonnées et sérialisations sont copiés aux frontières.

Le fallback `LexicalEntailmentScorer` mesure seulement le rappel des tokens informatifs du claim dans la preuve. Il retourne volontairement zéro pour la contradiction : un recouvrement lexical ne sait pas interpréter une négation, une paraphrase ou une incohérence numérique.

**Écart avec [RAGAS](https://arxiv.org/abs/2309.15217) et [RAGChecker](https://arxiv.org/abs/2408.08067).** Ces travaux proposent des cadres d'évaluation automatisée et fine-grained plus larges, souvent appuyés par des juges modèles et des protocoles définis. Cheragh n'en reproduit ni les prompts de juge, ni les métriques expérimentales complètes. Son API est une frontière locale et auditable ; pour parler de faithfulness sémantique, il faut injecter et valider un juge approprié.

## 7. CorrectiveRAGEngine : CRAG avancé

Le chemin correctif distingue maintenant trois actions de retrieval :

| Action | Contexte utilisé |
|---|---|
| `correct` | Les preuves primaires sont conservées |
| `ambiguous` | Les preuves d'un `external_retriever` injecté sont entrelacées avant les preuves primaires |
| `incorrect` | Le contexte primaire est écarté ; seules les preuves externes corrigées sont retenues |

Après correction, le contexte est réévalué. Un `KnowledgeRefiner` peut décomposer/recomposer les documents avant génération ; `knowledge_refiner="lexical"` active la baseline par phrases. Les variantes de requête restent bornées par `max_retries`. Avec `min_grounded_score`, le moteur peut essayer une autre variante utilisable si la première réponse est insuffisamment ancrée.

```python
from cheragh import BM25Retriever, CorrectiveRAGEngine, Document, StaticLLMClient

primary = BM25Retriever([
    Document("Le football se joue demain.", doc_id="hors-sujet")
])
external = BM25Retriever([
    Document("Paris est la capitale de la France.", doc_id="reference")
])
engine = CorrectiveRAGEngine(
    retriever=primary,
    external_retriever=external,
    llm_client=StaticLLMClient("Paris. [source: reference]"),
    knowledge_refiner="lexical",
    min_context_score=0.75,
    max_retries=0,
)

response = engine.ask("Quelle est la capitale de la France ?", top_k=1)
print(response.metadata["retrieval_action"])
print(response.metadata["external_document_ids"])
print(response.retrieved_documents[0].metadata["corrective_provenance"])
```

Un grader injecté peut retourner `RetrievalGrade`, une table avec `action`, ou un score normalisé. Les collaborateurs reçoivent des snapshots. La génération s'effectue via un retriever interne figé sur les documents effectivement évalués : un retriever primaire stateful ne peut donc pas substituer un autre contexte entre correction et génération.

**Écart avec [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884).** Cheragh expose les actions correct/ambiguous/incorrect, une source externe et la décomposition/recomposition, mais ne fournit ni moteur de recherche web, ni évaluateur entraîné du papier, ni son pipeline exact de filtrage des connaissances. `LexicalRetrievalGrader` et `LexicalKnowledgeRefiner` sont des baselines offline ; un déploiement sérieux doit injecter, calibrer et évaluer ses propres composants.

## Garanties communes et critères de choix

- Les limites `top_k`, `max_steps`, `beam_width`, `traversal_budget` et tailles similaires suivent un contrat entier strictement positif ; les booléens sont refusés. Les compteurs autorisant zéro le documentent explicitement.
- Les nouvelles frontières copient les `Document` et les métadonnées imbriquées aux entrées/sorties sensibles. Les objets de résultat restent des objets Python mutables : appeler les méthodes de snapshot prévues lorsqu'une copie indépendante supplémentaire est nécessaire.
- Une provenance riche facilite l'audit, mais ne démontre pas qu'une preuve est vraie. L'évaluation claim-level et les règles de citation doivent être calibrées sur le domaine.
- Les heuristiques offline sont conçues pour être déterministes, testables et remplaçables. Elles ne doivent pas recevoir les performances ou le nom d'un modèle entraîné absent.
- Avant production, mesurer au minimum rappel retrieval, précision/citations, faithfulness avec un juge validé, latence, budget tokens, coût, stabilité temporelle et isolation des accès sur un jeu représentatif.

## Références primaires

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs](https://arxiv.org/abs/2406.15319)
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity](https://arxiv.org/abs/2403.14403)
- [Active Retrieval Augmented Generation (FLARE)](https://arxiv.org/abs/2305.06983)
- [Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions (IRCoT)](https://arxiv.org/abs/2212.10509)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884)
- [Ragas: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)
- [RAGChecker: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation](https://arxiv.org/abs/2408.08067)
