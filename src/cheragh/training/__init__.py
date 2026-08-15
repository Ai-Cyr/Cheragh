"""Framework-neutral retrieval-aware training helpers."""

from .data import (
    DistilledRetrievalExample,
    HardNegativeMiner,
    RAFTDatasetBuilder,
    RAFTTrainingRecord,
    RetrievalTrainerProtocol,
    RetrievalTrainingExample,
    RetrievalTrainingPipeline,
    TeacherScoreDistiller,
    contrastive_retrieval_loss,
)

__all__ = [
    "DistilledRetrievalExample",
    "HardNegativeMiner",
    "RAFTDatasetBuilder",
    "RAFTTrainingRecord",
    "RetrievalTrainerProtocol",
    "RetrievalTrainingExample",
    "RetrievalTrainingPipeline",
    "TeacherScoreDistiller",
    "contrastive_retrieval_loss",
]
