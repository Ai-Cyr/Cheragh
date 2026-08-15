from __future__ import annotations

from importlib import import_module
import unittest

import cheragh
from cheragh import (
    AdaptiveRAGEngine,
    BM25Retriever,
    ClaimEvaluator,
    CommunityGraphRAGEngine,
    ColPaliRetriever,
    LongContextPacker,
    ReciprocalRankFusionRetriever,
    RetrievalTrainingPipeline,
    TemporalRetriever,
)
from cheragh.catalog import TECHNIQUES, TechniqueStatus, list_techniques
from cheragh.config import validate_config


class ArchitectureCatalogueTests(unittest.TestCase):
    def test_every_catalogue_entry_is_available_and_public(self) -> None:
        self.assertEqual(len(TECHNIQUES), 44)
        self.assertEqual(list_techniques(status=TechniqueStatus.PLANNED), [])

        for technique in TECHNIQUES:
            with self.subTest(technique=technique.id):
                self.assertTrue(technique.available)
                self.assertIsNotNone(technique.implementation)
                module_name, attribute = technique.implementation.rsplit(".", 1)
                module = import_module(module_name)
                self.assertTrue(hasattr(module, attribute))
                public_names = getattr(module, "__all__", ())
                self.assertIn(attribute, public_names)

    def test_new_architecture_boundaries_are_exported_from_root(self) -> None:
        for architecture in (
            AdaptiveRAGEngine,
            BM25Retriever,
            ClaimEvaluator,
            CommunityGraphRAGEngine,
            ColPaliRetriever,
            LongContextPacker,
            ReciprocalRankFusionRetriever,
            RetrievalTrainingPipeline,
            TemporalRetriever,
        ):
            with self.subTest(architecture=architecture.__name__):
                self.assertIs(getattr(cheragh, architecture.__name__), architecture)
                self.assertIn(architecture.__name__, cheragh.__all__)

    def test_bm25_configuration_is_sparse_only_and_validated(self) -> None:
        config = validate_config(
            {
                "retriever": {
                    "type": "bm25",
                    "top_k": 7,
                    "bm25_k1": 1.2,
                    "bm25_b": 0.6,
                }
            }
        )

        self.assertEqual(config.retriever.type, "bm25")
        self.assertEqual(config.retriever.top_k, 7)
        self.assertEqual(config.retriever.bm25_k1, 1.2)
        self.assertEqual(config.retriever.bm25_b, 0.6)

    def test_catalogue_maturity_counts_are_explicit(self) -> None:
        counts = {
            status: len(list_techniques(status=status))
            for status in TechniqueStatus
        }

        self.assertEqual(counts[TechniqueStatus.STABLE], 6)
        self.assertEqual(counts[TechniqueStatus.BETA], 12)
        self.assertEqual(counts[TechniqueStatus.EXPERIMENTAL], 26)
        self.assertEqual(counts[TechniqueStatus.PLANNED], 0)
        self.assertEqual(cheragh.__version__, "1.4.0")


if __name__ == "__main__":
    unittest.main()
