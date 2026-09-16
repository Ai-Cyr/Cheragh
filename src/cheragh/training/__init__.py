"""Retrieval-aware datasets and an optional differentiable PyTorch trainer."""

from .data import (
    DistilledRetrievalExample,
    HardNegativeMiner,
    RAFTDatasetBuilder,
    RAFTGeneratedAnswer,
    RAFTTrainingRecord,
    RetrievalTrainerProtocol,
    RetrievalTrainingExample,
    RetrievalTrainingPipeline,
    TeacherScoreDistiller,
    contrastive_retrieval_loss,
)
from .torch_trainer import TorchRetrievalTrainer
from .generative import (
    GenerativeTrainingExample,
    GenerativeTrainingReport,
    TransformersGenerativeTrainer,
    RankRAGDatasetBuilder,
    RankRAGModel,
    RankRAGEngine,
)

__all__ = [
    "GenerativeTrainingExample",
    "GenerativeTrainingReport",
    "TransformersGenerativeTrainer",
    "RankRAGDatasetBuilder",
    "RankRAGModel",
    "RankRAGEngine",
    "DistilledRetrievalExample",
    "HardNegativeMiner",
    "RAFTDatasetBuilder",
    "RAFTGeneratedAnswer",
    "RAFTTrainingRecord",
    "RetrievalTrainerProtocol",
    "RetrievalTrainingExample",
    "RetrievalTrainingPipeline",
    "TeacherScoreDistiller",
    "TorchRetrievalTrainer",
    "contrastive_retrieval_loss",
]
