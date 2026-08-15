# Architectures RAG — v1.2.0

Cheragh 1.2 ajoute quatre architectures expérimentales. Elles respectent les contrats communs du package (`Document`, `BaseRetriever`, `top_k`, snapshots défensifs et provenance), mais ne prétendent pas reproduire intégralement les systèmes de recherche cités en référence.

| Architecture | Point d'entrée | Rôle |
|---|---|---|
| Community GraphRAG | `CommunityGraphRAGEngine` | Partitionner un graphe d'entités, produire des rapports de communautés et servir une recherche globale ou locale |
| ColPali | `ColPaliRetriever` | Retrouver des pages-images par interaction tardive multi-vecteurs MaxSim |
| Temporal RAG | `TemporalRetriever` | Filtrer les preuves dans le temps, sélectionner les versions et pondérer la fraîcheur |
| Retrieval-aware training | `RetrievalTrainingPipeline` | Préparer des exemples, miner des négatifs difficiles, distiller un enseignant et appeler un entraîneur injecté |

## Community GraphRAG

`CommunityGraphRAGEngine` accepte des documents et, facultativement, un `KnowledgeGraph` déjà construit. Il détecte des communautés déterministes, génère un `CommunityReport` par communauté, puis expose deux chemins :

- `global_search()` / `ask_global()` classent les rapports pour les questions transverses ;
- `local_search()` / `ask_local()` partent des entités reconnues et reviennent aux documents sources.

```python
from cheragh import Document
from cheragh.community_graph import CommunityGraphRAGEngine
from cheragh.graph import KnowledgeGraph, KnowledgeTriple

documents = [
    Document("Alpha finance le programme solaire.", doc_id="climat"),
    Document("Beta mesure les émissions du programme.", doc_id="mesures"),
    Document("Delta publie les revenus annuels.", doc_id="finance"),
]

graph = KnowledgeGraph()
graph.add_triple(KnowledgeTriple("Alpha", "finance", "Solaire", "climat"))
graph.add_triple(KnowledgeTriple("Beta", "mesure", "Solaire", "mesures"))
graph.add_triple(KnowledgeTriple("Delta", "publie", "Revenus", "finance"))

engine = CommunityGraphRAGEngine(documents, graph=graph, top_k=2)

rapports = engine.global_search("Quels thèmes ressortent ?", top_k=2)
preuves = engine.local_search("Que fait Alpha ?", top_k=1)
reponse = engine.ask_global("Résume les thèmes du corpus", top_k=2)

print([document.doc_id for document in rapports])  # community:<id>
print(preuves[0].metadata["source_doc_id"])
print(reponse.metadata["selected_communities"])
```

Un résumeur spécifique peut être injecté avec `summarizer=...`. Il doit être appelable avec `(community, documents)` ou implémenter `summarize(community, documents) -> str`. Sans injection, `DeterministicCommunitySummarizer` produit un rapport local et reproductible. Les métadonnées d'un rapport contiennent notamment `source_doc_ids` et la provenance vers les documents d'origine.

Limites : l'implémentation effectue une partition déterministe mono-niveau, proche d'une passe locale de modularité. Elle n'emploie ni Leiden hiérarchique, ni extraction LLM complète des entités et assertions, ni évaluation et réduction LLM map/reduce des rapports. C'est une baseline Community GraphRAG, pas une reproduction de Microsoft GraphRAG. Le workflow général est inspiré de [From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130).

## ColPali : recherche visuelle par MaxSim

`ColPaliRetriever` indexe des `MultimodalDocument` dont la modalité est `Modality.IMAGE`. L'inférence est séparée derrière `VisualLateInteractionEncoder` : l'application peut injecter ses propres fonctions ou charger l'adaptateur officiel optionnel.

### Encodeur injectable, sans modèle téléchargé

```python
import numpy as np

from cheragh.multimodal import Modality, MultimodalDocument
from cheragh.multimodal.colpali import (
    CallableVisualLateInteractionEncoder,
    ColPaliRetriever,
)

pages = [
    MultimodalDocument(
        "page avec un tableau de revenus",
        doc_id="page-1",
        modality=Modality.IMAGE,
        uri="page-1.png",
        metadata={"patches": [[1.0, 0.0], [0.0, 1.0]]},
    ),
    MultimodalDocument(
        "page de garde",
        doc_id="page-2",
        modality=Modality.IMAGE,
        uri="page-2.png",
        metadata={"patches": [[0.4, 0.4]]},
    ),
]

encoder = CallableVisualLateInteractionEncoder(
    page_encoder=lambda batch: [page.metadata["patches"] for page in batch],
    query_encoder=lambda queries: [np.array([[1.0, 0.0], [0.0, 1.0]]) for _ in queries],
    fingerprint="mon-encodeur-visuel",
)
retriever = ColPaliRetriever(pages, encoder, normalize_vectors=False)

resultats = retriever.retrieve_pages("revenus du tableau", top_k=2)
print(resultats[0].doc_id)
print(resultats[0].metadata["maxsim_patch_indices"])
```

Le score exact est la somme, pour chaque vecteur-token de la requête, de sa similarité maximale avec les vecteurs-patches de la page. `retrieve_pages(..., filters={...})` applique aussi les filtres de métadonnées Cheragh et conserve les indices de patches gagnants dans la provenance.

### Adaptateur `colpali-engine` optionnel

```bash
pip install 'cheragh[colpali]'
```

```python
from cheragh.multimodal.colpali import ColPaliEngineAdapter, ColPaliRetriever

encoder = ColPaliEngineAdapter("vidore/colpali-v1.3", device="cuda")
retriever = ColPaliRetriever(pages, encoder)
resultats = retriever.retrieve_pages("où se trouve le total ?", top_k=3)
```

L'adaptateur charge le modèle et le processeur de `colpali-engine` seulement à son instanciation. Le modèle officiel, PyTorch et le traitement d'images restent des dépendances lourdes optionnelles.

Limites : l'index conserve toutes les matrices de patches en mémoire et calcule un MaxSim exact contre chaque page. Il n'inclut ni index multi-vecteurs ANN, ni compression, ni génération de candidats à grande échelle. L'architecture d'interaction tardive visuelle suit [ColPali: Efficient Document Retrieval with Vision Language Models](https://arxiv.org/abs/2407.01449), tandis que la qualité réelle dépend du modèle ou de l'encodeur injecté.

## Temporal RAG

`TemporalRetriever` enveloppe n'importe quel `BaseRetriever`. Il ne devine pas les dates dans la requête : il applique des contraintes explicites `as_of`, `start` et `end` aux métadonnées validées des documents.

```python
from cheragh.hybrid_search import BM25Retriever
from cheragh.temporal import TemporalDocument, TemporalRetriever

documents = [
    TemporalDocument(
        "L'ancienne politique autorise 30 jours.",
        "2025-01-01T00:00:00+00:00",
        valid_from="2025-01-01T00:00:00+00:00",
        valid_to="2025-03-01T00:00:00+00:00",
        version_id="v1",
        version_group="politique-remboursement",
        doc_id="politique-v1",
    ),
    TemporalDocument(
        "La politique courante autorise 14 jours.",
        "2025-03-01T00:00:00+00:00",
        valid_from="2025-03-01T00:00:00+00:00",
        version_id="v2",
        version_group="politique-remboursement",
        doc_id="politique-v2",
    ),
]

retriever = TemporalRetriever(
    BM25Retriever(documents),
    candidate_top_k=10,
    conflict_resolution="latest",
)

avant = retriever.snapshot(
    "politique jours",
    as_of="2025-02-01T00:00:00+00:00",
    top_k=1,
)
maintenant = retriever.snapshot(
    "politique jours",
    as_of="2025-04-01T00:00:00+00:00",
    top_k=1,
)

print(avant[0].doc_id)      # politique-v1
print(maintenant[0].doc_id) # politique-v2
print(maintenant[0].metadata["temporal_provenance"])
```

API principale :

- `TemporalDocument` et `temporal_metadata()` normalisent les dates conscientes de leur fuseau en ISO-8601 UTC ; les dates naïves sont refusées ;
- `snapshot(query, as_of=...)` réalise une vue à un instant donné ;
- `between(query, start=..., end=...)` réalise une recherche par intervalle ;
- `conflict_resolution` accepte `latest`, `highest_score`, `all` ou `raise` pour les documents du même `version_group` ;
- `missing_timestamp="fail"` est le comportement sûr par défaut ; `allow` conserve explicitement une preuve non vérifiée ;
- `freshness_half_life` et `freshness_weight` activent une décroissance exponentielle de fraîcheur et consignent le détail du calcul dans `temporal_provenance`.

Limites : la combinaison pertinence/fraîcheur suppose que les scores fournis par le retriever de base sont comparables et, idéalement, calibrés ou normalisés. Cheragh ne fait pas cette calibration automatiquement. Cette API n'est ni un graphe temporel, ni un parseur de dates en langage naturel, ni une reproduction de la boucle retrieve–rewrite de [TimeR4 / Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering](https://aclanthology.org/2024.emnlp-main.394/). Elle fournit une couche fiable de métadonnées, filtrage, versions et provenance sur un retriever existant.

## Retrieval-aware training

`RetrievalTrainingPipeline` couvre la partie indépendante du framework : exemples défensifs, mining de négatifs difficiles, distillation de scores et passage à un adaptateur d'entraînement. Il ne télécharge ni ne modifie de poids par lui-même.

```python
from cheragh import Document
from cheragh.hybrid_search import BM25Retriever
from cheragh.training import (
    HardNegativeMiner,
    RAFTDatasetBuilder,
    RetrievalTrainingExample,
    RetrievalTrainingPipeline,
    TeacherScoreDistiller,
)

oracle = Document("Paris est la capitale de la France.", doc_id="oracle")
corpus = [
    oracle,
    Document("Lyon est une métropole française.", doc_id="negatif-1"),
    Document("Berlin est la capitale de l'Allemagne.", doc_id="negatif-2"),
]
source = RetrievalTrainingExample(
    query="Quelle est la capitale de la France ?",
    positive_documents=(oracle,),
    answer="Paris. [source: oracle]",
)

miner = HardNegativeMiner(
    BM25Retriever(corpus),
    candidate_top_k=10,
    negatives_per_query=2,
)
distiller = TeacherScoreDistiller(
    lambda query, docs: [1.0 if doc.doc_id == "oracle" else 0.2 for doc in docs],
    temperature=1.0,
)
pipeline = RetrievalTrainingPipeline(miner=miner, distiller=distiller)

class MonEntraineur:
    def fit(self, examples, **kwargs):
        # Adapter ici Sentence Transformers, PyTorch ou un service hébergé.
        return {"examples": len(examples), "epochs": kwargs.get("epochs", 1)}

rapport = pipeline.fit([source], MonEntraineur(), epochs=2)
record_raft = RAFTDatasetBuilder(oracle_probability=1.0, seed=7).build([source])[0]

print(rapport)
print(record_raft.render_prompt())
```

Les briques publiques sont complémentaires :

- `HardNegativeMiner` récupère des documents bien classés mais non positifs, déduplique et accepte un `exclusion_filter` pour éviter les faux négatifs connus ;
- `TeacherScoreDistiller` transforme les scores d'un enseignant en probabilités avec température ;
- `contrastive_retrieval_loss()` fournit une perte InfoNCE NumPy de référence ;
- `RAFTDatasetBuilder` produit des enregistrements open-book avec oracles et distracteurs ;
- `RetrievalTrainerProtocol.fit()` est la frontière d'intégration vers le framework choisi.

Limites : le pipeline ne fournit aucun poids de modèle, optimizer, boucle de rétropropagation, checkpoint, batching GPU ou entraînement distribué. `RAFTDatasetBuilder` prépare des données de style RAFT ; il n'effectue pas le fine-tuning RAFT. La distillation et le mining sont des primitives, pas une reproduction de RankRAG. Les références primaires utiles pour construire un entraînement complet sont [RAFT](https://arxiv.org/abs/2403.10131), [In Defense of Dual-Encoders for Neural Ranking](https://arxiv.org/abs/2104.08051) pour les négatifs difficiles, et [RankRAG](https://arxiv.org/abs/2407.02485).

## Garanties et points de vigilance communs

- `top_k` et les tailles de pools doivent être des entiers strictement positifs ; les booléens sont refusés.
- Les documents, métadonnées et résultats sont copiés aux frontières d'index afin d'éviter qu'une mutation externe ne corrompe l'état interne.
- Les métadonnées de provenance exposent la méthode de récupération, les sources d'origine ou les détails de scoring pertinents à l'architecture.
- Les quatre composants sont des baselines expérimentales : évaluez rappel, précision, calibration, groundedness, latence et coût sur votre propre corpus avant un déploiement en production.
