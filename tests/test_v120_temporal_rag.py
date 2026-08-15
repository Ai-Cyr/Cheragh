from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from cheragh.base import BaseRetriever, Document
from cheragh.temporal import (
    ConflictResolution,
    MissingTimestampPolicy,
    TemporalDocument,
    TemporalRetriever,
    temporal_metadata,
    version_metadata,
)


UTC = timezone.utc


class _StaticRetriever(BaseRetriever):
    def __init__(self, documents: list[Document]) -> None:
        self.documents = documents
        self.calls: list[int] = []

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        self.calls.append(top_k)
        return self.documents[:top_k]


def _time(day: int, *, hour: int = 0) -> datetime:
    return datetime(2025, 1, day, hour, tzinfo=UTC)


class TemporalDocumentTests(unittest.TestCase):
    def test_reserved_temporal_metadata_must_use_validated_arguments(self):
        for key, value in (
            ("timestamp", "2025-01-01T00:00:00+00:00"),
            ("valid_from", "not-a-date"),
            ("valid_to", "not-a-date"),
            ("version_id", True),
            ("version_group", []),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                TemporalDocument(
                    "fact",
                    "2025-01-01T00:00:00+00:00",
                    metadata={key: value},
                )

    def test_helpers_normalize_aware_datetimes_and_copy_metadata(self) -> None:
        nested = {"labels": ["policy"]}
        document = TemporalDocument(
            "Version courante",
            "2025-01-01T01:00:00+01:00",
            valid_from="2025-01-01T00:00:00Z",
            valid_to="2026-01-01T00:00:00Z",
            version_id=2,
            version_group="policy",
            metadata=nested,
            doc_id="policy-v2",
        )

        nested["labels"].append("caller-mutation")

        self.assertEqual(document.timestamp, datetime(2025, 1, 1, tzinfo=UTC))
        self.assertEqual(document.metadata["timestamp"], "2025-01-01T00:00:00+00:00")
        self.assertEqual(document.metadata["version_id"], "2")
        self.assertEqual(document.version_group, "policy")
        self.assertEqual(document.metadata["labels"], ["policy"])

    def test_from_document_takes_an_independent_snapshot(self) -> None:
        source = Document(
            "fact",
            metadata={
                **temporal_metadata(_time(1), version_id="v1", version_group="fact"),
                "nested": {"owner": "source"},
            },
            doc_id="fact-v1",
            score=0.8,
        )

        temporal = TemporalDocument.from_document(source)
        temporal.metadata["nested"]["owner"] = "result"

        self.assertEqual(source.metadata["nested"]["owner"], "source")
        self.assertEqual(temporal.version_id, "v1")
        self.assertEqual(temporal.score, 0.8)

        overridden = TemporalDocument.from_document(
            source,
            timestamp=_time(2),
            version_id="v2",
        )
        self.assertEqual(overridden.timestamp, _time(2))
        self.assertEqual(overridden.version_id, "v2")
        self.assertEqual(overridden.version_group, "fact")

    def test_helpers_reject_naive_or_incoherent_dates(self) -> None:
        with self.assertRaisesRegex(ValueError, "timezone"):
            temporal_metadata(datetime(2025, 1, 1))
        with self.assertRaisesRegex(ValueError, "valid_from"):
            version_metadata(valid_from=_time(2), valid_to=_time(1))
        with self.assertRaises(TypeError):
            version_metadata(version_id=True)


class TemporalFilteringTests(unittest.TestCase):
    def test_allowed_missing_timestamp_still_honours_known_validity(self):
        base = _StaticRetriever(
            [
                Document(
                    "future policy",
                    doc_id="future",
                    metadata={"valid_from": "2030-01-01T00:00:00+00:00"},
                ),
                Document(
                    "expired policy",
                    doc_id="expired",
                    metadata={"valid_to": "2024-01-01T00:00:00+00:00"},
                ),
                Document("unverified", doc_id="unverified"),
            ]
        )
        retriever = TemporalRetriever(base, missing_timestamp="allow")

        results = retriever.snapshot(
            "policy",
            as_of="2025-01-01T00:00:00+00:00",
            top_k=3,
        )

        self.assertEqual([document.doc_id for document in results], ["unverified"])

    def test_as_of_filters_future_evidence_and_respects_validity_interval(self) -> None:
        documents = [
            TemporalDocument(
                "ancienne règle",
                _time(1),
                valid_from=_time(1),
                valid_to=_time(15),
                version_id="v1",
                version_group="rule",
                doc_id="rule-v1",
                score=0.5,
            ),
            TemporalDocument(
                "nouvelle règle",
                _time(15),
                valid_from=_time(15),
                version_id="v2",
                version_group="rule",
                doc_id="rule-v2",
                score=0.4,
            ),
            TemporalDocument("publication future", _time(20), doc_id="future", score=1.0),
        ]
        retriever = TemporalRetriever(_StaticRetriever(documents), candidate_top_k=10)

        before_change = retriever.snapshot("règle", as_of=_time(14), top_k=3)
        at_change = retriever.retrieve("règle", as_of=_time(15), top_k=3)

        self.assertEqual([doc.doc_id for doc in before_change], ["rule-v1"])
        self.assertEqual([doc.doc_id for doc in at_change], ["rule-v2"])
        self.assertEqual(at_change[0].metadata["temporal_provenance"]["as_of"], _time(15).isoformat())

    def test_start_and_end_are_inclusive(self) -> None:
        documents = [
            TemporalDocument("one", _time(1), doc_id="one", score=0.9),
            TemporalDocument("two", _time(2), doc_id="two", score=0.8),
            TemporalDocument("three", _time(3), doc_id="three", score=0.7),
        ]
        retriever = TemporalRetriever(_StaticRetriever(documents), candidate_top_k=10)

        result = retriever.between("q", start=_time(2), end=_time(3), top_k=5)

        self.assertEqual([doc.doc_id for doc in result], ["two", "three"])

    def test_constructor_window_is_used_and_call_window_overrides_it(self) -> None:
        documents = [
            TemporalDocument("one", _time(1), doc_id="one", score=0.9),
            TemporalDocument("three", _time(3), doc_id="three", score=0.8),
        ]
        retriever = TemporalRetriever(
            _StaticRetriever(documents), as_of=_time(2), candidate_top_k=10
        )

        self.assertEqual([doc.doc_id for doc in retriever.retrieve("q")], ["one"])
        self.assertEqual(
            [doc.doc_id for doc in retriever.retrieve("q", start=_time(3))],
            ["three"],
        )

    def test_missing_timestamp_policy_is_explicit(self) -> None:
        missing = Document("unverified", metadata={"kind": "legacy"}, doc_id="legacy", score=1.0)
        strict = TemporalRetriever(_StaticRetriever([missing]), candidate_top_k=1)
        permissive = TemporalRetriever(
            _StaticRetriever([missing]),
            missing_timestamp=MissingTimestampPolicy.ALLOW,
            candidate_top_k=1,
        )

        with self.assertRaisesRegex(ValueError, "missing temporal metadata"):
            strict.retrieve("q", as_of=_time(1))
        result = permissive.retrieve("q", as_of=_time(1), top_k=1)

        provenance = result[0].metadata["temporal_provenance"]
        self.assertEqual(provenance["timestamp_status"], "missing_allowed")
        self.assertEqual(provenance["missing_timestamp_policy"], "allow")


class TemporalRankingTests(unittest.TestCase):
    def test_freshness_decay_reranks_and_exposes_score_provenance(self) -> None:
        old = TemporalDocument("old", _time(1), doc_id="old", score=0.5)
        recent = TemporalDocument("recent", _time(21), doc_id="recent", score=0.5)
        retriever = TemporalRetriever(
            _StaticRetriever([old, recent]),
            freshness_half_life=timedelta(days=10),
            freshness_weight=0.5,
            reference_time=_time(31),
            candidate_top_k=2,
        )

        result = retriever.retrieve("q", top_k=2)

        self.assertEqual([doc.doc_id for doc in result], ["recent", "old"])
        recent_provenance = result[0].metadata["temporal_provenance"]
        old_provenance = result[1].metadata["temporal_provenance"]
        self.assertAlmostEqual(recent_provenance["freshness_score"], 0.5)
        self.assertAlmostEqual(old_provenance["freshness_score"], 0.125)
        self.assertGreater(result[0].score, result[1].score)
        self.assertEqual(recent_provenance["base_score"], 0.5)
        self.assertEqual(recent_provenance["final_score"], result[0].score)

    def test_latest_version_is_default_conflict_resolution(self) -> None:
        older = TemporalDocument(
            "old", _time(1), version_id="v1", version_group="policy", doc_id="v1", score=0.99
        )
        newer = TemporalDocument(
            "new", _time(2), version_id="v2", version_group="policy", doc_id="v2", score=0.1
        )
        retriever = TemporalRetriever(_StaticRetriever([older, newer]), candidate_top_k=2)

        result = retriever.retrieve("policy", top_k=2)

        self.assertEqual([doc.doc_id for doc in result], ["v2"])
        provenance = result[0].metadata["temporal_provenance"]
        self.assertEqual(provenance["versions_considered"], 2)
        self.assertEqual(provenance["discarded_version_ids"], ["v1"])

    def test_highest_score_all_and_raise_conflict_strategies(self) -> None:
        versions = [
            TemporalDocument(
                "old", _time(1), version_id="v1", version_group="policy", doc_id="v1", score=0.9
            ),
            TemporalDocument(
                "new", _time(2), version_id="v2", version_group="policy", doc_id="v2", score=0.4
            ),
        ]

        highest = TemporalRetriever(
            _StaticRetriever(versions),
            conflict_resolution=ConflictResolution.HIGHEST_SCORE,
            candidate_top_k=2,
        )
        all_versions = TemporalRetriever(
            _StaticRetriever(versions), conflict_resolution="all", candidate_top_k=2
        )
        strict = TemporalRetriever(
            _StaticRetriever(versions), conflict_resolution="raise", candidate_top_k=2
        )

        self.assertEqual(highest.retrieve("q", top_k=2)[0].doc_id, "v1")
        self.assertEqual(len(all_versions.retrieve("q", top_k=2)), 2)
        with self.assertRaisesRegex(ValueError, "multiple versions"):
            strict.retrieve("q", top_k=2)

    def test_results_and_nested_provenance_are_defensive_snapshots(self) -> None:
        source = TemporalDocument(
            "source",
            _time(1),
            metadata={"nested": {"owner": "base"}, "retrieval_method": "dense"},
            doc_id="source",
            score=0.7,
        )
        base = _StaticRetriever([source])
        retriever = TemporalRetriever(base, candidate_top_k=1)

        first = retriever.retrieve("q", top_k=1)[0]
        first.metadata["nested"]["owner"] = "caller"
        first.metadata["temporal_provenance"]["base_score"] = -1
        second = retriever.retrieve("q", top_k=1)[0]

        self.assertNotIn("temporal_provenance", source.metadata)
        self.assertEqual(source.metadata["retrieval_method"], "dense")
        self.assertEqual(second.metadata["nested"]["owner"], "base")
        self.assertEqual(second.metadata["temporal_provenance"]["base_score"], 0.7)
        self.assertEqual(second.metadata["temporal_provenance"]["base_retrieval_method"], "dense")


class TemporalValidationTests(unittest.TestCase):
    def test_top_k_is_strict_and_candidate_pool_is_forwarded(self) -> None:
        base = _StaticRetriever([TemporalDocument("one", _time(1), doc_id="one")])
        retriever = TemporalRetriever(base, candidate_top_k=7)

        retriever.retrieve("q", top_k=1)

        self.assertEqual(base.calls, [7])
        for value, error in ((0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)):
            with self.subTest(top_k=value), self.assertRaises(error):
                retriever.retrieve("q", top_k=value)  # type: ignore[arg-type]

    def test_invalid_windows_configuration_and_metadata_fail_early(self) -> None:
        base = _StaticRetriever([])
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            TemporalRetriever(base, as_of=_time(2), start=_time(1))
        with self.assertRaisesRegex(ValueError, "start"):
            TemporalRetriever(base, start=_time(2), end=_time(1))
        with self.assertRaisesRegex(ValueError, "timezone"):
            TemporalRetriever(base, as_of=datetime(2025, 1, 1))
        with self.assertRaises(ValueError):
            TemporalRetriever(base, missing_timestamp="ignore")
        with self.assertRaises(ValueError):
            TemporalRetriever(base, conflict_resolution="merge")
        with self.assertRaises(ValueError):
            TemporalRetriever(base, candidate_top_k=0)

        malformed = TemporalRetriever(
            _StaticRetriever([Document("bad", metadata={"timestamp": "yesterday"})]),
            candidate_top_k=1,
        )
        with self.assertRaisesRegex(ValueError, "ISO-8601"):
            malformed.retrieve("q")

    def test_invalid_freshness_inputs_are_rejected(self) -> None:
        base = _StaticRetriever([])
        for half_life, error in (
            (timedelta(0), ValueError),
            (0, ValueError),
            (True, TypeError),
            (float("inf"), ValueError),
        ):
            with self.subTest(half_life=half_life), self.assertRaises(error):
                TemporalRetriever(base, freshness_half_life=half_life)
        for weight, error in ((-0.1, ValueError), (1.1, ValueError), (True, TypeError)):
            with self.subTest(weight=weight), self.assertRaises(error):
                TemporalRetriever(base, freshness_weight=weight)


if __name__ == "__main__":
    unittest.main()
