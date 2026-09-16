"""Experimental inference-time Self-RAG components.

``SelfRAGEngine`` preserves the lightweight critic/refinement baseline.
``SegmentedSelfRAGEngine`` provides the explicit model-conditioned segment
search path, with ``TransformersSelfRAGDecoder`` for trained checkpoints.
Neither path trains reflection tokens or claims benchmark equivalence.
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
from .segmented import (
    ReflectionDecoder,
    ReflectionSegment,
    RetrievalAction,
    ScoredSegment,
    SegmentedSelfRAGEngine,
    SegmentedSelfRAGResult,
    SegmentSearchTrace,
)
from .transformers import TransformersSelfRAGDecoder

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
    "ReflectionDecoder",
    "ReflectionSegment",
    "RetrievalAction",
    "ScoredSegment",
    "SegmentedSelfRAGEngine",
    "SegmentedSelfRAGResult",
    "SegmentSearchTrace",
    "TransformersSelfRAGDecoder",
]
