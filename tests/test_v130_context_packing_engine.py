import json
import unittest

from cheragh import Document, RAGEngine, StaticLLMClient
from cheragh.compression import ContextCompressor
from cheragh.context_packing import LongContextPacker


def _whitespace_tokens(text: str) -> int:
    return len(text.split())


def _content_only(document: Document) -> str:
    return document.content


class _FixedRetriever:
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self.documents[:top_k]


class _ReplacingCompressor(ContextCompressor):
    def compress(self, query: str, documents: list[Document]) -> list[Document]:
        return [
            Document(
                f"compressed {document.content}",
                metadata={**document.metadata, "compressed_by_test": True},
                doc_id=document.doc_id,
                score=document.score,
            )
            for document in documents
        ]


class ContextPackingEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.documents = [
            Document(
                "alpha evidence",
                metadata={"source": "A", "nested": {"version": 1}},
                doc_id="a",
                score=3.0,
            ),
            Document("beta evidence", metadata={"source": "B"}, doc_id="b", score=2.0),
            Document("gamma evidence", metadata={"source": "C"}, doc_id="c", score=1.0),
        ]
        self.packer = LongContextPacker(
            20,
            token_estimator=_whitespace_tokens,
            formatter=_content_only,
            separator=" <SEP> ",
            ordering="lost_in_the_middle",
        )

    def test_ask_uses_exact_packed_text_documents_and_serializable_diagnostics(self) -> None:
        expected = self.packer.pack(self.documents)
        llm = StaticLLMClient("answer")
        engine = RAGEngine(
            _FixedRetriever(self.documents),
            llm_client=llm,
            answer_prompt="BEFORE\n{context}\nAFTER\n{query}",
            context_packer=self.packer,
        )

        response = engine.ask("question", top_k=3)

        self.assertEqual(response.prompt, f"BEFORE\n{expected.text}\nAFTER\nquestion")
        self.assertEqual(llm.prompts, [response.prompt])
        self.assertEqual(
            [document.doc_id for document in response.retrieved_documents],
            [document.doc_id for document in expected.documents],
        )
        diagnostics = response.metadata["context_packing"]
        self.assertEqual(diagnostics["selected_documents"], 3)
        self.assertEqual(diagnostics["tokens_used"], expected.diagnostics.tokens_used)
        self.assertEqual(diagnostics["ordering"], "lost_in_the_middle")
        self.assertEqual(diagnostics, response.trace.metadata["context_packing"])
        self.assertEqual(
            [step.name for step in response.trace.steps],
            ["retrieval", "context_packing", "generation"],
        )
        packing_step = response.trace.steps[1]
        self.assertEqual(packing_step.metadata["diagnostics"], diagnostics)
        json.dumps(response.metadata)
        json.dumps(response.trace.to_dict())

    def test_packing_runs_after_compression_and_before_generation(self) -> None:
        llm = StaticLLMClient("answer")
        packer = LongContextPacker(
            20,
            token_estimator=_whitespace_tokens,
            formatter=_content_only,
            ordering="relevance",
        )
        engine = RAGEngine(
            _FixedRetriever([self.documents[0]]),
            llm_client=llm,
            answer_prompt="{context}",
            compressor=_ReplacingCompressor(),
            context_packer=packer,
        )

        response = engine.ask("alpha")

        self.assertEqual(response.prompt, "compressed alpha evidence")
        self.assertEqual(response.retrieved_documents[0].content, "compressed alpha evidence")
        self.assertTrue(response.retrieved_documents[0].metadata["compressed_by_test"])
        self.assertEqual(
            [step.name for step in response.trace.steps],
            ["retrieval", "compression", "context_packing", "generation"],
        )

    def test_sync_and_stream_have_packing_parity_and_independent_snapshots(self) -> None:
        engine = RAGEngine(
            _FixedRetriever(self.documents),
            llm_client=StaticLLMClient("same answer"),
            answer_prompt="{context}\nQ:{query}",
            context_packer=self.packer,
        )
        direct = engine.ask("question", top_k=3)
        direct.retrieved_documents[0].content = "mutated result"
        direct.retrieved_documents[0].metadata["nested"]["version"] = 99

        stream = engine.stream_with_response("question", top_k=3)
        self.assertEqual("".join(stream), "same answer")
        streamed = stream.response
        self.assertIsNotNone(streamed)
        assert streamed is not None

        expected = self.packer.pack(self.documents)
        self.assertEqual(streamed.prompt, f"{expected.text}\nQ:question")
        self.assertEqual(streamed.metadata, direct.metadata)
        self.assertEqual(
            [document.doc_id for document in streamed.retrieved_documents],
            [document.doc_id for document in expected.documents],
        )
        self.assertEqual(streamed.retrieved_documents[0].content, "alpha evidence")
        self.assertEqual(streamed.retrieved_documents[0].metadata["nested"]["version"], 1)
        self.assertEqual(
            [step.name for step in streamed.trace.steps],
            ["retrieval", "context_packing", "generation"],
        )

    def test_strict_grounding_short_circuits_when_packing_eliminates_every_document(self) -> None:
        llm = StaticLLMClient("must not run")
        packer = LongContextPacker(
            1,
            token_estimator=_whitespace_tokens,
            formatter=_content_only,
        )
        engine = RAGEngine(
            _FixedRetriever([Document("two tokens", doc_id="too-large", score=1.0)]),
            llm_client=llm,
            context_packer=packer,
            strict_grounding=True,
        )

        direct = engine.ask("question")
        stream = engine.stream_with_response("question")
        self.assertEqual("".join(stream), direct.answer)
        streamed = stream.response
        self.assertIsNotNone(streamed)
        assert streamed is not None

        self.assertEqual(llm.prompts, [])
        self.assertEqual(direct.sources, [])
        self.assertEqual(direct.retrieved_documents, [])
        self.assertEqual(direct.metadata["context_packing"]["input_documents"], 1)
        self.assertEqual(direct.metadata["context_packing"]["selected_documents"], 0)
        self.assertIn("no_relevant_documents", direct.warnings)
        self.assertIn("context_packing_empty", direct.warnings)
        self.assertEqual(
            [step.name for step in direct.trace.steps],
            ["retrieval", "context_packing"],
        )
        direct_payload = direct.to_dict()
        streamed_payload = streamed.to_dict()
        direct_payload.pop("trace")
        streamed_payload.pop("trace")
        direct_payload.pop("response_id")
        streamed_payload.pop("response_id")
        self.assertEqual(streamed_payload, direct_payload)

    def test_strict_grounding_treats_empty_rendered_context_as_no_context(self) -> None:
        llm = StaticLLMClient("must not run")
        packer = LongContextPacker(10, formatter=lambda _document: "")
        response = RAGEngine(
            _FixedRetriever([Document("evidence", doc_id="hidden")]),
            llm_client=llm,
            context_packer=packer,
            strict_grounding=True,
        ).ask("question")

        self.assertEqual(llm.prompts, [])
        self.assertEqual(response.retrieved_documents, [])
        self.assertIn("context_packing_empty", response.warnings)

    def test_constructor_validates_packer_and_from_documents_forwards_it(self) -> None:
        with self.assertRaises(TypeError):
            RAGEngine(_FixedRetriever([]), context_packer=object())  # type: ignore[arg-type]

        class _NonCallablePack:
            pack = 1

        with self.assertRaises(TypeError):
            RAGEngine(_FixedRetriever([]), context_packer=_NonCallablePack())  # type: ignore[arg-type]

        class _InvalidResultPacker:
            def pack(self, documents: object) -> object:
                return object()

        invalid_engine = RAGEngine(
            _FixedRetriever([Document("alpha", doc_id="a")]),
            context_packer=_InvalidResultPacker(),  # type: ignore[arg-type]
        )
        with self.assertRaisesRegex(TypeError, "must return PackedContext"):
            invalid_engine.ask("alpha")

        engine = RAGEngine.from_documents(
            self.documents,
            retriever_type="memory",
            context_packer=self.packer,
        )
        self.assertIs(engine.context_packer, self.packer)


if __name__ == "__main__":
    unittest.main()
