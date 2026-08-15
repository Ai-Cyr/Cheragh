import unittest

from cheragh import Document
from cheragh.base import LLMClient
from cheragh.self_rag import (
    LexicalEvidenceCritic,
    ScriptedEvidenceCritic,
    SelfRAGEngine,
    StaticRetrievalGate,
    SupportAssessment,
)


class RecordingRetriever:
    def __init__(self, documents):
        self.documents = list(documents)
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return self.documents[:top_k]


class QueueLLM(LLMClient):
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return self.answers.pop(0)


class SelfRAGTests(unittest.TestCase):
    def test_retrieves_critiques_and_refines_until_supported(self):
        retriever = RecordingRetriever(
            [Document("Paris est la capitale de la France.", doc_id="france")]
        )
        llm = QueueLLM(["Paris et Rome sont en France.", "Paris est la capitale de la France."])
        critic = ScriptedEvidenceCritic(
            support_assessments=(
                SupportAssessment(
                    0.4,
                    False,
                    "unsupported_claim",
                    ("Rome est en France.",),
                ),
                SupportAssessment(1.0, True, "supported"),
            )
        )
        engine = SelfRAGEngine(
            retriever,
            llm,
            evidence_critic=critic,
            max_refinements=2,
        )

        result = engine.ask("Quelle est la capitale de la France ?", top_k=3)

        self.assertEqual(result.status, "supported")
        self.assertTrue(result.supported)
        self.assertEqual(retriever.calls, [("Quelle est la capitale de la France ?", 3)])
        self.assertEqual(len(result.trace.iterations), 2)
        self.assertEqual(result.trace.iterations[1].kind, "refinement")
        self.assertIn("Rome est en France", llm.prompts[1])
        self.assertEqual(result.trace.stop_reason, "answer_supported")
        self.assertEqual(result.to_dict()["maturity"], "experimental")

    def test_gate_can_skip_retrieval_and_support_is_not_claimed(self):
        llm = QueueLLM(["Réponse prudente."])
        engine = SelfRAGEngine(
            None,
            llm,
            retrieval_gate=StaticRetrievalGate(False, reason="simple_query"),
        )

        result = engine.ask("Bonjour")

        self.assertEqual(result.status, "completed_without_retrieval")
        self.assertIsNone(result.supported)
        self.assertFalse(result.trace.retrieval.should_retrieve)
        self.assertEqual(len(llm.prompts), 1)

    def test_irrelevant_evidence_fails_closed_without_generation(self):
        retriever = RecordingRetriever([Document("Un texte sans rapport", doc_id="other")])
        llm = QueueLLM(["ne doit pas être appelée"])
        critic = ScriptedEvidenceCritic(relevant=False, relevance_score=0.0)
        engine = SelfRAGEngine(retriever, llm, evidence_critic=critic)

        result = engine.ask("Capitales européennes")

        self.assertEqual(result.status, "insufficient_evidence")
        self.assertFalse(result.supported)
        self.assertEqual(llm.prompts, [])
        self.assertEqual(result.trace.stop_reason, "insufficient_relevant_evidence")

    def test_lexical_critic_is_deterministic(self):
        critic = LexicalEvidenceCritic(relevance_threshold=0.2, support_threshold=0.5)
        documents = [
            Document("Mercure est la planète la plus proche du Soleil.", doc_id="mercury"),
            Document("Les baleines vivent dans les océans.", doc_id="whales"),
        ]

        relevance = critic.assess_relevance("Quelle planète est proche du Soleil ?", documents)
        support = critic.assess_support(
            "Quelle planète est proche du Soleil ?",
            "Mercure est la planète la plus proche du Soleil.",
            [documents[0]],
        )

        self.assertEqual(relevance.relevant_indices, (0,))
        self.assertTrue(support.supported)

    def test_refinement_limit_is_strict(self):
        retriever = RecordingRetriever([Document("preuve", doc_id="d1")])
        llm = QueueLLM(["première", "seconde"])
        unsupported = SupportAssessment(0.0, False, "unsupported", ("affirmation",))
        critic = ScriptedEvidenceCritic(support_assessments=(unsupported,))
        engine = SelfRAGEngine(retriever, llm, evidence_critic=critic, max_refinements=1)

        result = engine.ask("preuve")

        self.assertEqual(result.status, "unsupported")
        self.assertEqual(len(result.trace.iterations), 2)
        self.assertEqual(len(llm.prompts), 2)


if __name__ == "__main__":
    unittest.main()
