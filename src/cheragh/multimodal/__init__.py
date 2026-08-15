"""Multimodal retrieval primitives.

The module is dependency-light at import time.  Text/image embeddings can be
provided by any :class:`MultimodalEmbeddingModel`; the optional
``CLIPMultimodalEmbedding`` loads its heavy dependencies only when instantiated.
"""

from .retrieval import (
    CLIPMultimodalEmbedding,
    CallableMultimodalEmbedding,
    Modality,
    MultimodalDocument,
    MultimodalEmbeddingModel,
    MultimodalQuery,
    MultimodalRAGEngine,
    MultimodalRetriever,
)
from .colpali import (
    CallableVisualLateInteractionEncoder,
    ColPaliEngineAdapter,
    ColPaliRetriever,
    VisualLateInteractionEncoder,
)

__all__ = [
    "CLIPMultimodalEmbedding",
    "CallableMultimodalEmbedding",
    "CallableVisualLateInteractionEncoder",
    "Modality",
    "MultimodalDocument",
    "MultimodalEmbeddingModel",
    "MultimodalQuery",
    "MultimodalRAGEngine",
    "MultimodalRetriever",
    "ColPaliEngineAdapter",
    "ColPaliRetriever",
    "VisualLateInteractionEncoder",
]
