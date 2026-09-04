"""Experimental inference-time Self-RAG components.

These exports provide modular inference orchestration only; they do not
implement Self-RAG model training or reflection-token learning.
"""

from .engine import (
    AlwaysRetrieveGate,
    EvidenceCritic,
    EvidenceRelevance,
    LexicalEvidenceCritic,
    RelevanceAssessment,
    RetrievalDecision,
    RetrievalGate,
    ScriptedEvidenceCritic,
    SelfRAGEngine,
    SelfRAGIteration,
    SelfRAGResult,
    SelfRAGTrace,
    StaticRetrievalGate,
    SupportAssessment,
)
from .reflection import (
    ReflectionScore,
    ReflectionTokenDistribution,
    ReflectionTokenGroup,
    ReflectionTokenRetrievalGate,
    ReflectionTokenScorer,
)

__all__ = [
    "AlwaysRetrieveGate",
    "EvidenceCritic",
    "EvidenceRelevance",
    "LexicalEvidenceCritic",
    "RelevanceAssessment",
    "RetrievalDecision",
    "RetrievalGate",
    "ScriptedEvidenceCritic",
    "SelfRAGEngine",
    "SelfRAGIteration",
    "SelfRAGResult",
    "SelfRAGTrace",
    "StaticRetrievalGate",
    "SupportAssessment",
    "ReflectionScore",
    "ReflectionTokenDistribution",
    "ReflectionTokenGroup",
    "ReflectionTokenRetrievalGate",
    "ReflectionTokenScorer",
]
