import unittest

import cheragh


class V110PublicAPITests(unittest.TestCase):
    def test_new_root_exports_resolve(self):
        names = (
            "AgenticRAGEngine",
            "AlwaysRetrieveGate",
            "ColBERTRetriever",
            "LLMJSONPlanner",
            "MultimodalRAGEngine",
            "RetrievalDecision",
            "RetrievalToolAdapter",
            "RAGStream",
            "SPLADERetriever",
            "SelfRAGEngine",
            "ToolRegistry",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(cheragh, name))

    def test_version_matches_release(self):
        self.assertEqual(cheragh.__version__, "1.1.0")


if __name__ == "__main__":
    unittest.main()
