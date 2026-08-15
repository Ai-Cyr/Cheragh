# Politique de sécurité

## Versions prises en charge

Cheragh est encore en bêta. Les correctifs de sécurité ciblent la dernière ligne
mineure publiée :

| Version | Correctifs de sécurité |
| --- | --- |
| `1.4.x` | Oui |
| `<= 1.3.x` | Non |
| branche `main` | Préversion, sans garantie de stabilité |

Cette table est mise à jour lors de chaque nouvelle ligne mineure. Les versions
Python officiellement testées sont celles annoncées dans `pyproject.toml` et la
matrice CI. Une dépendance optionnelle n'est prise en charge que dans sa plage de
versions déclarée.

## Signaler une vulnérabilité

N'ouvrez pas d'issue publique contenant des détails exploitables. Utilisez le
[signalement privé GitHub](https://github.com/Ai-Cyr/Cheragh/security/advisories/new)
et fournissez, sans inclure de secret ni de donnée de production :

- la version de Cheragh, de Python et des dépendances concernées ;
- le composant, la configuration et le scénario d'attaque ;
- l'impact attendu et les préconditions ;
- une reproduction minimale ou un test en échec ;
- toute mesure de contournement connue ;
- vos préférences de coordination pour la divulgation.

Les mainteneurs visent un accusé de réception sous trois jours ouvrés et une
première qualification sous sept jours ouvrés. Ces délais sont des objectifs,
pas une garantie de correction. La date de publication est coordonnée avec le
rapporteur afin de laisser un délai raisonnable de mise à niveau.

## Périmètre

Sont notamment pertinents :

- contournement d'authentification, d'ACL ou d'isolation tenant/collection ;
- traversée de chemin, écriture ou lecture hors des racines autorisées ;
- exécution de code, désérialisation dangereuse ou injection de commande ;
- exposition de clés, prompts, documents, traces ou métadonnées sensibles ;
- défaut permettant à un document non fiable d'obtenir des capacités système ;
- déni de service reproductible avec un impact mesurable sous les limites
  documentées ;
- vulnérabilité d'une dépendance exploitable dans une configuration supportée.

Une hallucination, une citation inexacte ou une injection de prompt sans rupture
d'une frontière de sécurité n'est pas, à elle seule, une vulnérabilité logicielle.
Signalez-la comme bug de qualité. Les installations non supportées, les anciennes
versions et les services tiers relèvent de leurs mainteneurs, sauf si Cheragh les
utilise d'une manière qui crée directement l'impact.

## Attentes de déploiement

Le serveur Cheragh ne fournit pas à lui seul TLS, identité utilisateur, quotas,
sandbox de modèles ou sauvegarde. Un déploiement supporté suit le
[guide de production](docs/production.md), garde l'indexation HTTP désactivée ou
isolée, place l'API derrière une passerelle et injecte les secrets depuis un
gestionnaire dédié.

Le dépôt n'offre actuellement aucun programme de bug bounty. Ne testez jamais
une infrastructure ou des données qui ne vous appartiennent pas sans autorisation
explicite.
