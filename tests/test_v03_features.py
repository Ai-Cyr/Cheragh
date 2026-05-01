import unittest
from types import SimpleNamespace

from cheragh import (
    Document,
    HashingEmbedding,
    KeywordOverlapReranker,
    OpenAIEmbedding,
    RAGEngine,
    StaticLLMClient,
    extract_citations,
    validate_citations,
)


class _FakeEmbeddings:
    def create(self, model, input):
        data = []
        for idx, text in enumerate(input):
            # deterministic 3D embedding based on content
            vec = [1.0 if "python" in text.lower() else 0.0, 1.0 if "rag" in text.lower() else 0.0, 0.5]
            data.append(SimpleNamespace(index=idx, embedding=vec))
        return SimpleNamespace(data=data)


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()


class V03FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document("Python packaging uses pyproject.toml.", doc_id="python"),
            Document("RAG retrieves context before generation.", doc_id="rag"),
            Document("Coffee is a hot drink.", doc_id="coffee"),
        ]

    def test_openai_embedding_accepts_injected_client(self):
        embedder = OpenAIEmbedding(client=_FakeOpenAIClient(), model="fake")
        vectors = embedder.embed_documents(["python", "rag"])
        self.assertEqual(vectors.shape, (2, 3))
        self.assertAlmostEqual(float((vectors[0] ** 2).sum()), 1.0, places=5)

    def test_keyword_reranker_can_be_used_by_engine(self):
        llm = StaticLLMClient("Python est cité. [source: python]")
        engine = RAGEngine.from_documents(
            self.docs,
            embedding_model=HashingEmbedding(dimension=64),
            llm_client=llm,
            retriever_type="vector",
            reranker=KeywordOverlapReranker(),
            top_k=1,
            first_stage_top_k=3,
            require_citations=True,
        )
        response = engine.ask("python packaging")
        self.assertEqual(response.sources[0].doc_id, "python")
        self.assertEqual(response.citations, ["python"])
        self.assertEqual(response.grounded_score, 1.0)

    def test_citation_validation_flags_unknown_and_unsourced(self):
        answer = "Le RAG récupère du contexte. Python package. [source: missing]"
        result = validate_citations(answer, self.docs[:2], require_citations=True, flag_unsourced_sentences=True)
        self.assertIn("missing", result.unknown_citations)
        self.assertTrue(result.unsourced_claims)
        self.assertIn("unknown_citations", result.warnings)

    def test_extract_citations(self):
        self.assertEqual(extract_citations("ok [source: a] puis [source: b]"), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
