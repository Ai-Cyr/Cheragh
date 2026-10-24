# Interface graphique de Cheragh

L'interface est incluse dans la wheel et la source distribution. Elle utilise
le serveur FastAPI de Cheragh et fonctionne sans téléchargement de ressources
web externes. Son affichage s'adapte aux ordinateurs et aux mobiles.

## Démarrer

Depuis le dépôt :

```bash
python -m pip install ".[fastapi,config]"
cheragh index ./docs --output .cheragh_index
cheragh ui --index .cheragh_index
```

Ouvrez **http://127.0.0.1:8000/ui/**. La racine du serveur redirige aussi vers
l'interface. `cheragh ui` est un alias de `cheragh serve` : les mêmes options
de configuration, d'authentification et de limites s'appliquent.

Pour utiliser vos fournisseurs et réglages existants :

```bash
cheragh ui --config rag.yaml
```

Les fournisseurs, modèles et architectures actifs sont définis par le serveur
et sa configuration. Le catalogue de l'interface décrit les techniques et leurs
limites ; parcourir une fiche ne change pas le moteur en cours d'exécution.

## Utiliser l'espace de travail

- **Conversation** : poser une question, choisir le nombre de sources recherchées,
  lire la réponse et consulter les passages utilisés.
- **Documents** : consulter l'état de l'index et, si le serveur l'autorise,
  indexer un fichier ou un dossier présent sur le serveur.
- **Architectures** : rechercher une technique et filtrer par maturité, avec ses
  prérequis et limites.

L'interface affiche les données renvoyées par le serveur. Le nombre de documents
correspond aux fragments exposés par le store, pas nécessairement au nombre de
fichiers d'origine. Certains retrievers ne fournissent pas ce compteur.
Une nouvelle question est indépendante des précédentes : l'historique visuel
n'est pas envoyé comme contexte conversationnel au moteur.

## Connexion et confidentialité

Si `CHERAGH_API_KEY` est configurée, saisissez cette clé dans l'écran de connexion.
Sans clé configurée, la connexion locale fonctionne avec le champ vide.
La clé reste en mémoire dans l'onglet et est envoyée uniquement au même serveur
dans l'en-tête `X-API-Key`. Elle n'est enregistrée ni dans une URL, ni dans le
stockage du navigateur. La déconnexion efface les réponses et les données de
session affichées ; le rechargement de la page demande une nouvelle connexion.

La page et ses ressources statiques sont publiques ; les données de session,
les questions, l'indexation et le catalogue utilisent la même authentification
que l'API. Les passages et réponses sont affichés comme texte, sans exécuter de
HTML fourni par les documents. Une politique CSP limite scripts, styles et
connexions à l'origine du serveur et interdit l'intégration dans une iframe.

Pour un accès distant, configurez HTTPS et les protections décrites dans le
[guide du serveur](production_server.md). Les appels de l'interface utilisent
des URL relatives, compatibles avec un préfixe de reverse proxy correctement
transmis au serveur ASGI.

## Indexer et activer un index local

L'indexation HTTP est désactivée par défaut. Elle exige une clé et un répertoire
autorisé. Exemple avec `corpus/` déjà présent sous ce répertoire :

```bash
export CHERAGH_API_KEY="$(your-secret-manager read cheragh-api-key)"
cheragh ui --index /srv/cheragh/index \
  --enable-indexing --index-root /srv/cheragh
```

Dans **Documents**, les chemins sont relatifs au répertoire autorisé du serveur.
Le formulaire accepte les paramètres de découpage et l'indexation incrémentale.
Il ne téléverse pas les fichiers de l'ordinateur depuis lequel vous consultez
l'interface.

Lorsque le serveur utilise `--index` avec les embeddings locaux `HashingEmbedding`,
l'option d'activation est disponible pour réindexer ce même index local.
Les nouvelles requêtes utilisent le nouveau
snapshot après une indexation et un chargement réussis ; les requêtes déjà en
cours conservent leur moteur. Une erreur, une annulation serveur ou un timeout
n'active pas un snapshot tardif. Un travail synchrone déjà démarré peut néanmoins
terminer son écriture sur disque après le timeout.

Les moteurs fournis par `--config` ou par Python, et les index utilisant un
modèle d'embedding personnalisé, ne sont pas remplacés depuis l'interface.
Pour ceux-ci, un index créé doit être raccordé par la configuration
et le processus de déploiement. L'activation concerne un seul processus serveur :
plusieurs workers nécessitent un déploiement coordonné.

## Vérification de l'interface

La CI construit et installe la wheel, puis lance Chromium contre un vrai serveur
et un corpus temporaire. Elle vérifie la connexion, une réponse avec ses sources,
l'affichage sûr des documents, l'activation de l'index, la déconnexion et la
navigation mobile sous un préfixe d'URL.

Pour exécuter ce contrôle dans un environnement de développement :

```bash
python -m pip install ".[fastapi,config]" "playwright==1.58.0"
python -m playwright install chromium
python -I scripts/smoke_ui.py
```
