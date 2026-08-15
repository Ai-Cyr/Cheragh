import math
import unittest

from cheragh.base import Document
from cheragh.context_packing import (
    ContextPacker,
    DroppedDocument,
    LongContextPacker,
    PackedContext,
    PackingDiagnostics,
    SourceTokenUsage,
    approximate_token_count,
    lost_in_the_middle_order,
    pack_context,
)


def whitespace_tokens(text: str) -> int:
    return len(text.split())


def content_only(document: Document) -> str:
    return document.content


class LongContextPackerTests(unittest.TestCase):
    def test_strict_budget_skips_large_candidate_and_fills_with_smaller_evidence(self) -> None:
        documents = [
            Document("one two three four five six", doc_id="large", score=1.0),
            Document("one two three", doc_id="medium", score=0.8),
            Document("four five", doc_id="small", score=0.7),
        ]
        packed = LongContextPacker(
            5,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            separator=" ",
            ordering="relevance",
        ).pack(documents)

        self.assertEqual([document.doc_id for document in packed.documents], ["medium", "small"])
        self.assertEqual(packed.text, "one two three four five")
        self.assertEqual(packed.token_count, 5)
        self.assertEqual(packed.diagnostics.remaining_tokens, 0)
        self.assertEqual(packed.diagnostics.dropped[0].reason, "token_budget")

    def test_formatter_separator_and_citation_headers_are_inside_exact_budget(self) -> None:
        documents = [
            Document("alpha", doc_id="a", score=2.0),
            Document("beta", doc_id="b", score=1.0),
        ]
        first_only = LongContextPacker(17, token_estimator=len, separator="||", ordering="relevance").pack(
            documents
        )
        both = LongContextPacker(35, token_estimator=len, separator="||", ordering="relevance").pack(documents)

        self.assertEqual(first_only.text, "[source: a]\nalpha")
        self.assertLessEqual(len(first_only.text), 17)
        self.assertEqual(len(first_only.documents), 1)
        self.assertEqual(both.text, "[source: a]\nalpha||[source: b]\nbeta")
        self.assertEqual(both.token_count, len(both.text))
        self.assertLessEqual(both.token_count, 35)

    def test_relevance_is_placed_at_both_edges_to_reduce_lost_in_the_middle(self) -> None:
        documents = [
            Document(letter, doc_id=letter, score=float(5 - index))
            for index, letter in enumerate("ABCDE")
        ]
        packed = ContextPacker(
            5,
            token_estimator=len,
            formatter=content_only,
            separator="",
        )(documents)

        self.assertEqual([document.doc_id for document in packed.documents], ["A", "C", "E", "D", "B"])
        self.assertEqual(packed.text, "ACEDB")
        standalone = lost_in_the_middle_order(documents)
        self.assertEqual([document.doc_id for document in standalone], ["A", "C", "E", "D", "B"])
        self.assertIsNot(standalone[0], documents[0])

    def test_deduplication_is_normalized_and_retains_highest_scored_snapshot(self) -> None:
        documents = [
            Document("Same   Evidence", metadata={"winner": False}, doc_id="low", score=0.1),
            Document(" same evidence ", metadata={"winner": True}, doc_id="high", score=0.9),
            Document("different", doc_id="other", score=0.5),
        ]
        packed = LongContextPacker(
            20,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            ordering="relevance",
        ).pack(documents)

        self.assertEqual([document.doc_id for document in packed.documents], ["high", "other"])
        self.assertTrue(packed.documents[0].metadata["winner"])
        self.assertEqual(packed.diagnostics.unique_documents, 2)
        self.assertEqual(packed.diagnostics.duplicate_count, 1)
        duplicate = packed.diagnostics.dropped[0]
        self.assertEqual(duplicate.doc_id, "low")
        self.assertEqual(duplicate.duplicate_of, "high")

    def test_deduplication_can_be_disabled_or_injected(self) -> None:
        documents = [
            Document("same", doc_id="one", score=2.0),
            Document("same", doc_id="two", score=1.0),
        ]
        disabled = LongContextPacker(
            2,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            separator=" ",
            deduplicate=False,
        ).pack(documents)
        custom = LongContextPacker(
            2,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            deduplication_key=lambda document: document.doc_id,
        ).pack(documents)

        self.assertEqual(len(disabled.documents), 2)
        self.assertEqual(len(custom.documents), 2)

    def test_per_source_budget_prevents_one_source_from_dominating(self) -> None:
        documents = [
            Document("a one", metadata={"source": "A"}, doc_id="a1", score=1.0),
            Document("a two", metadata={"source": "A"}, doc_id="a2", score=0.9),
            Document("b one", metadata={"source": "B"}, doc_id="b1", score=0.8),
        ]
        packed = LongContextPacker(
            10,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            separator=" ",
            per_source_token_budget=2,
            ordering="relevance",
        ).pack(documents)

        self.assertEqual([document.doc_id for document in packed.documents], ["a1", "b1"])
        self.assertEqual([item.tokens for item in packed.diagnostics.source_usage], [2, 2])
        self.assertEqual([item.token_budget for item in packed.diagnostics.source_usage], [2, 2])
        self.assertEqual(
            [item.doc_id for item in packed.diagnostics.dropped if item.reason == "source_budget"],
            ["a2"],
        )

    def test_source_budget_mapping_leaves_unspecified_sources_uncapped(self) -> None:
        documents = [
            Document("one", metadata={"source": "blocked"}, doc_id="x", score=2.0),
            Document("one two three", metadata={"source": "free"}, doc_id="y", score=1.0),
        ]
        packed = LongContextPacker(
            4,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            separator=" ",
            per_source_token_budget={"blocked": 0},
            ordering="relevance",
        ).pack(documents)

        self.assertEqual([document.doc_id for document in packed.documents], ["y"])
        self.assertEqual(packed.diagnostics.dropped[0].reason, "source_budget")

    def test_source_ids_are_normalized_before_budget_lookup(self) -> None:
        packed = LongContextPacker(
            10,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            per_source_token_budget={"source-a": 1},
            source_resolver=lambda _document: "  source-a  ",
        ).pack([Document("two tokens", doc_id="a")])

        self.assertEqual(packed.documents, ())
        self.assertEqual(packed.diagnostics.dropped[0].reason, "source_budget")
        with self.assertRaises(ValueError):
            LongContextPacker(
                10,
                per_source_token_budget={"source-a": 1, " source-a ": 2},
            )

    def test_doc_ids_citations_offsets_and_nested_provenance_are_preserved_as_snapshots(self) -> None:
        source = Document(
            "grounded evidence",
            metadata={
                "source": "guide.md",
                "page": 4,
                "source_char_start": 10,
                "source_char_end": 27,
                "provenance": {"pipeline": ["ocr", "chunk"]},
            },
            doc_id="guide-4",
            score=0.8,
        )
        packed = LongContextPacker(20).pack([source])
        source.content = "mutated"
        source.metadata["provenance"]["pipeline"].append("caller mutation")

        self.assertEqual(packed.documents[0].content, "grounded evidence")
        self.assertEqual(packed.documents[0].metadata["provenance"]["pipeline"], ["ocr", "chunk"])
        self.assertIn("[source: guide-4]", packed.text)
        self.assertEqual(packed.citation_ids, ("guide-4",))

        second_snapshot = packed.snapshot()
        packed.documents[0].metadata["provenance"]["pipeline"].append("result mutation")
        self.assertEqual(second_snapshot.documents[0].metadata["provenance"]["pipeline"], ["ocr", "chunk"])

    def test_optional_truncation_fills_budget_and_adjusts_character_provenance(self) -> None:
        source = Document(
            "one two three four five",
            metadata={"source_char_start": 100, "source_char_end": 123, "nested": {"v": 1}},
            doc_id="long",
            score=1.0,
        )
        packed = LongContextPacker(
            3,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            truncate_oversized=True,
        ).pack([source])

        self.assertEqual(packed.text, "one two three")
        self.assertEqual(packed.token_count, 3)
        self.assertEqual(packed.diagnostics.truncated_document_ids, ("long",))
        output = packed.documents[0]
        self.assertEqual(output.doc_id, "long")
        self.assertEqual(output.metadata["source_char_start"], 100)
        self.assertEqual(output.metadata["source_char_end"], 113)
        self.assertEqual(output.metadata["nested"], {"v": 1})
        self.assertEqual(output.metadata["context_packing"]["original_characters"], 23)
        self.assertEqual(source.metadata["source_char_end"], 123)
        self.assertNotIn("context_packing", source.metadata)

    def test_input_order_is_available_after_score_aware_selection(self) -> None:
        documents = [
            Document("first", doc_id="first", score=0.1),
            Document("second", doc_id="second", score=0.9),
        ]
        packed = LongContextPacker(
            2,
            token_estimator=whitespace_tokens,
            formatter=content_only,
            separator=" ",
            ordering="input",
        ).pack(documents)
        self.assertEqual([document.doc_id for document in packed.documents], ["first", "second"])

    def test_empty_input_and_one_shot_helper_are_deterministic(self) -> None:
        empty = LongContextPacker(5).pack([])
        empty_render = LongContextPacker(5, formatter=lambda _document: "").pack(
            [Document("hidden by formatter", doc_id="hidden")]
        )
        one_shot = pack_context(
            [Document("alpha", doc_id="a", score=1.0)],
            1,
            token_estimator=whitespace_tokens,
            formatter=content_only,
        )

        self.assertEqual(empty.text, "")
        self.assertEqual(empty.documents, ())
        self.assertEqual(empty.token_count, 0)
        self.assertEqual(empty.diagnostics.remaining_tokens, 5)
        self.assertEqual(empty_render.text, "")
        self.assertEqual(empty_render.documents, ())
        self.assertEqual(empty_render.diagnostics.dropped[0].reason, "empty_content")
        self.assertEqual(one_shot.text, "alpha")

    def test_configuration_validation_is_strict(self) -> None:
        for invalid, exception in (
            (True, TypeError),
            (1.5, TypeError),
            ("5", TypeError),
            (0, ValueError),
            (-1, ValueError),
        ):
            with self.subTest(token_budget=invalid):
                with self.assertRaises(exception):
                    LongContextPacker(invalid)  # type: ignore[arg-type]

        for kwargs, exception in (
            ({"token_estimator": 1}, TypeError),
            ({"formatter": 1}, TypeError),
            ({"source_resolver": 1}, TypeError),
            ({"deduplicate": 1}, TypeError),
            ({"deduplication_key": 1}, TypeError),
            ({"separator": 1}, TypeError),
            ({"ordering": "middle"}, ValueError),
            ({"truncate_oversized": 1}, TypeError),
            ({"per_source_token_budget": True}, TypeError),
            ({"per_source_token_budget": -1}, ValueError),
            ({"per_source_token_budget": {1: 2}}, TypeError),
            ({"per_source_token_budget": {"": 2}}, ValueError),
            ({"per_source_token_budget": {"A": True}}, TypeError),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(exception):
                    LongContextPacker(5, **kwargs)  # type: ignore[arg-type]

    def test_runtime_collaborator_and_document_validation_is_strict(self) -> None:
        with self.assertRaises(TypeError):
            LongContextPacker(5).pack("not documents")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            LongContextPacker(5).pack([object()])  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            LongContextPacker(5).pack([Document("x", score=True)])
        with self.assertRaises(ValueError):
            LongContextPacker(5).pack([Document("x", score=math.nan)])
        with self.assertRaises(TypeError):
            LongContextPacker(5, token_estimator=lambda text: 1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LongContextPacker(5, token_estimator=lambda text: -1)
        with self.assertRaises(TypeError):
            LongContextPacker(5, formatter=lambda document: 1).pack(  # type: ignore[arg-type]
                [Document("x")]
            )
        with self.assertRaises(TypeError):
            LongContextPacker(5, source_resolver=lambda document: 1).pack(  # type: ignore[arg-type]
                [Document("x")]
            )
        with self.assertRaises(ValueError):
            LongContextPacker(5, source_resolver=lambda document: " ").pack([Document("x")])
        with self.assertRaises(TypeError):
            LongContextPacker(5, deduplication_key=lambda document: []).pack(  # type: ignore[arg-type]
                [Document("x")]
            )
        with self.assertRaises(TypeError):
            LongContextPacker(5).pack([Document("x", doc_id=1)])  # type: ignore[arg-type]

    def test_direct_result_construction_rejects_incoherent_or_mutable_payloads(self) -> None:
        dropped = DroppedDocument(
            input_index=0,
            doc_id="dropped",
            source_id="source",
            reason="token_budget",
            estimated_tokens=2,
        )
        diagnostics = PackingDiagnostics(
            input_documents=2,
            unique_documents=2,
            selected_documents=1,
            token_budget=3,
            tokens_used=1,
            remaining_tokens=2,
            ordering="relevance",
            dropped=(dropped,),
            source_usage=(SourceTokenUsage("source", 1, 3, 1),),
        )
        source = Document("one", metadata={"nested": {"version": 1}}, doc_id="kept")
        packed = PackedContext("one", (source,), diagnostics)
        source.metadata["nested"]["version"] = 2
        self.assertEqual(packed.documents[0].metadata["nested"]["version"], 1)

        with self.assertRaises(ValueError):
            PackingDiagnostics(
                input_documents=1,
                unique_documents=1,
                selected_documents=1,
                token_budget=3,
                tokens_used=2,
                remaining_tokens=2,
                ordering="relevance",
                dropped=(),
                source_usage=(SourceTokenUsage("source", 2, 3, 1),),
            )
        with self.assertRaises(TypeError):
            PackingDiagnostics(
                input_documents=0,
                unique_documents=0,
                selected_documents=0,
                token_budget=3,
                tokens_used=0,
                remaining_tokens=3,
                ordering="relevance",
                dropped=[],  # type: ignore[arg-type]
                source_usage=(),
            )
        with self.assertRaises(ValueError):
            PackedContext("one", (), diagnostics)
        with self.assertRaises(ValueError):
            PackedContext("", (source,), diagnostics)

    def test_default_estimator_is_dependency_free_and_punctuation_aware(self) -> None:
        self.assertEqual(approximate_token_count("Bonjour, monde!"), 4)
        self.assertEqual(approximate_token_count(""), 0)
        with self.assertRaises(TypeError):
            approximate_token_count(1)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
