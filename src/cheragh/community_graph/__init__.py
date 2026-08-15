"""Dependency-light Community GraphRAG baseline.

The implementation is intentionally explicit about being a single-level,
deterministic baseline rather than a full reproduction of Microsoft GraphRAG.
See :mod:`cheragh.community_graph.engine` for the supported workflow and its
limitations.
"""

from .engine import (
    Community,
    CommunityGraphRAGEngine,
    CommunityReport,
    CommunitySummarizer,
    DeterministicCommunitySummarizer,
    detect_communities,
)

__all__ = [
    "Community",
    "CommunityGraphRAGEngine",
    "CommunityReport",
    "CommunitySummarizer",
    "DeterministicCommunitySummarizer",
    "detect_communities",
]
