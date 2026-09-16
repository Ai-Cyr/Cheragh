# Vérification logicielle et modèles exécutés

Vérification du **16 septembre 2026**, sur CPU. Les résultats ci-dessous sont des tests logiciels et des smokes de modèles ; aucun benchmark de publication ou score métier n’a été reproduit.

## Résultats intégrés

| Configuration locale | Résultat |
|---|---|
| Python 3.12, extras de recherche et checkpoints en cache | **1 016 tests + 546 sous-tests passent**, aucun test ignoré |
| Python 3.12, noyau et serveur sans Torch/Transformers | **953 tests + 527 sous-tests passent**, 52 tests optionnels ignorés |
| Python 3.13, noyau et serveur sans Torch/Transformers | **953 tests + 527 sous-tests passent**, 52 tests optionnels ignorés |
| Ruff, code/tests/scripts | Aucun défaut |
| mypy, périmètre configuré de 41 fichiers | Aucun défaut |
| Imports publics sans extras de recherche | Les 383 exports se chargent ; les modèles restent optionnels |
| Dépendances de recherche, `pip check` | Aucun conflit |

Deux avertissements de dépréciation proviennent du client de test Starlette/httpx et de l’alias AnyIO. Ils ne correspondent pas à des tests en échec. La CI contient le noyau Python 3.10–3.13, un job CPU avec Torch/Transformers/UMAP/graspologic, la construction wheel/sdist, les smokes des distributions installées, le conteneur et l’audit des dépendances. Le conteneur n’a pas été exécuté localement : Docker est absent de cet environnement.

Environnement de recherche : Python 3.12, Torch 2.14.0+cpu, Transformers 5.17.0, NumPy 1.26.4, scikit-learn 1.9.1, umap-learn 0.5.12, graspologic 3.4.4. La version NumPy satisfait ici la contrainte de graspologic. Les adaptateurs appris et les nouveaux cas de découpage ont également été testés avec Transformers 4.57.6 ; ce second environnement ne constitue pas une exécution de toute la suite.

## Checkpoints réellement chargés

| Modèle | Preuve et résultat observé | Identité locale |
|---|---|---|
| `colbert-ir/colbertv2.0` | ~109,6 M paramètres ; projection 128D, augmentation à 32 vecteurs de requête, MaxSim recalculé indépendamment. Paris est classé premier parmi trois documents pour la capitale de la France. | `c1e84128e85ef755c096a95bdb06b47793b13acf` |
| `naver/splade-cocondenser-ensembledistil` | ~109,5 M paramètres ; vecteurs de vocabulaire 30 522D, pooling vérifié contre log1p(ReLU) masqué. Le document France est classé premier. | Cache actif `49cf4c7b0db5b870a401ddf5e2669993ef3699c7` |
| `cross-encoder/nli-MiniLM2-L6-H768` | ~82 M paramètres ; paraphrase positive et contradiction explicite reconnues. Smoke initial : entailment ≈0,9826 et contradiction ≈0,9988 sur les phrases de test. | `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` |
| `cross-encoder/ms-marco-TinyBERT-L2-v2` | ~4,39 M paramètres ; calibration logistique sur six paires, ranking Tokyo/bruit puis sélection des strips. Des faux négatifs hors de ces exemples ont aussi été observés : cette calibration n’est pas une validation métier. | SHA256 du `model.safetensors` : `b1ce0f765792309c17d9113be8c9dac3e963f3c726ae6a9311888069f5db16a6` |

Sources des modèles : [ColBERTv2](https://huggingface.co/colbert-ir/colbertv2.0), [SPLADE](https://huggingface.co/naver/splade-cocondenser-ensembledistil), [NLI](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768), [TinyBERT](https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L2-v2).

Les résultats sur trois documents ou quelques phrases sont des contrôles de fonctionnement. Ils ne donnent ni rappel moyen, ni robustesse hors domaine, ni qualité de réponse en production.

## Modèles minuscules et tests indépendants

- Self-RAG : Llama initialisé localement, vrai décodage avec cache KV, logits aux positions de réflexion, round-trip de checkpoint, faisceau, budgets et provenance.
- Adaptive-RAG : T5 et BERT entraînent effectivement les trois routes ; baisse de perte, classement des exemples d’apprentissage et rechargement vérifiés.
- RAFT/RankRAG : GPT2 et T5 optimisés sur CPU ; perte comparée à une cross-entropy calculée séparément, prompt/padding masqués par position, EOS conservé lorsque PAD=EOS, réponse apprise et probabilités identiques après rechargement. RankRAG sélectionne une preuve puis répond avec les mêmes poids.
- Retrieval/TimeR4 : gradients contrastifs réels, loss décroissante, hard negatives temporels, paramètres effectivement mis à jour.
- Chunking : vrais tokenizers WordPiece et ByteLevelBPE, offsets répétés Unicode, textes avec CRLF, préambules courts et conservation des queues de documents.
- ACL : parents reconstruits, caches, résumés RAPTOR et rapports GraphRAG évaluent les dépendances documentaires complètes.

Les grands modèles Self-RAG, ColPali et Dense X n’ont pas été exécutés. Les données et checkpoints TKS/RAFT/RankRAG métier restent à fournir ou entraîner. Les composants qui emploient un LLM injectable ont été validés avec des sorties contrôlées ; la qualité d’un fournisseur distant n’a pas été mesurée.

## Reproduire

Installer PyTorch pour le matériel visé, puis les dépendances de recherche :

```bash
python -m pip install -e '.[dev,fastapi,raptor,graphrag,training,research]'
python -m pip check
ruff check src tests scripts
mypy --no-incremental
pytest -q
```

La suite standard ne télécharge pas les modèles publiés. Après les avoir téléchargés dans un cache Hugging Face, activer les quatre smokes locaux :

```bash
CHERAGH_REAL_MODELS_CACHE=/chemin/cache-huggingface \
CHERAGH_CRAG_MODEL_PATH=/chemin/checkpoint-tinybert \
OMP_NUM_THREADS=4 \
pytest -q tests/test_learned_model_fidelity.py \
  tests/test_semantic_evaluation.py tests/test_crag_semantic.py
```

Ces tests chargent avec `local_files_only=True` et vérifient les sorties attendues sans réseau. Conserver les révisions et hashes des poids pour répéter exactement l’expérience. Pour les artefacts de livraison, la CI reconstruit le wheel/sdist, les installe séparément, exécute `scripts/smoke_distribution.py` et conserve leurs `SHA256SUMS` ; voir le [guide de production](production.md).
