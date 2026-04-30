"""Rule primitives for query routing."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping, Pattern


@dataclass(frozen=True)
class RouteRule:
    """A transparent keyword/regex rule that scores one route."""

    route: str
    query_type: str
    reason: str
    patterns: tuple[str, ...]
    weight: float = 0.9
    description_boost_terms: tuple[str, ...] = field(default_factory=tuple)

    def score(self, query: str, route_descriptions: Mapping[str, str] | None = None) -> float:
        text = query.lower()
        hits = 0
        for pattern in self.patterns:
            if re.search(pattern, text, flags=re.I | re.UNICODE):
                hits += 1
        if hits == 0:
            return 0.0
        score = self.weight + min(0.08 * (hits - 1), 0.1)
        description = (route_descriptions or {}).get(self.route, "").lower()
        if description and any(term.lower() in description for term in self.description_boost_terms):
            score += 0.05
        return min(1.0, score)


def default_rules() -> list[RouteRule]:
    """Default French/English route rules.

    The canonical route names are intentionally simple: ``qa``, ``summary``,
    ``sql``, ``analytics``, ``fallback`` and ``multi_step``. Users can supply
    custom rules for project-specific route names.
    """

    return [
        RouteRule(
            route="summary",
            query_type="summary",
            reason="summary intent detected",
            patterns=(
                r"\b(résume|resume|summari[sz]e|synthèse|synthese|tl;?dr|en bref|points clés|key points)\b",
                r"\b(fais|donne).{0,25}\b(résumé|resume|synthèse|synthese)\b",
            ),
            description_boost_terms=("summary", "résumé", "synthèse"),
        ),
        RouteRule(
            route="sql",
            query_type="sql",
            reason="structured data or SQL intent detected",
            patterns=(
                r"\b(sql|requête|requete|query|select|join|table|base de données|database)\b",
                r"\b(ventes|sales|revenu|revenue|ca|chiffre d'affaires|q[1-4]|trimestre|quarter)\b",
                r"\b(compare|comparer|comparaison).{0,40}\b(q[1-4]|ventes|sales|revenu|revenue)\b",
            ),
            description_boost_terms=("sql", "database", "table", "structured", "données"),
        ),
        RouteRule(
            route="analytics",
            query_type="analytics",
            reason="analytical comparison intent detected",
            patterns=(
                r"\b(analyse|analyser|compare|comparer|comparaison|évolution|evolution|trend|tendance|pourquoi|why)\b",
                r"\b(impact|corrélation|correlation|cause|root cause|variance|écart|ecart)\b",
            ),
            weight=0.82,
            description_boost_terms=("analysis", "analytics", "analyse"),
        ),
        RouteRule(
            route="multi_step",
            query_type="multi_step",
            reason="multi-step reasoning intent detected",
            patterns=(
                r"\b(décompose|decompose|étape par étape|step by step|multi[- ]?hop|raisonnement|raisonne)\b",
                r"\b(compare).{0,50}\b(et|avec|versus|vs\.?|contre)\b",
            ),
            weight=0.78,
        ),
        RouteRule(
            route="fallback",
            query_type="fallback",
            reason="likely out-of-corpus/current information intent detected",
            patterns=(
                r"\b(météo|meteo|weather|actualité|actualités|news|aujourd'hui|today|maintenant|current|latest|dernières nouvelles)\b",
                r"\b(hors corpus|pas dans les documents|outside the corpus)\b",
            ),
            weight=0.88,
        ),
        RouteRule(
            route="qa",
            query_type="qa",
            reason="factual QA intent detected",
            patterns=(
                r"\b(qui|quoi|quand|où|ou|comment|combien|what|who|when|where|how many|quelle?|quel)\b",
                r"\?",
            ),
            weight=0.55,
        ),
    ]
