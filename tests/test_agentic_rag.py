import unittest

from cheragh import Document, StaticLLMClient
from cheragh.agentic import (
    AgentAction,
    AgentTool,
    AgenticRAGEngine,
    LLMJSONPlanner,
    RetrievalToolAdapter,
    ScriptedPlanner,
    ToolRegistry,
)


class RecordingRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=5):
        self.calls.append((query, top_k))
        return [Document("La preuve demandée.", doc_id="proof", score=0.9)]


class AgenticRAGTests(unittest.TestCase):
    def test_registered_retrieval_tool_and_final_answer(self):
        retriever = RecordingRetriever()
        retrieval_tool = RetrievalToolAdapter(retriever, top_k=2).as_tool()
        planner = ScriptedPlanner(
            [
                AgentAction.call_tool("retrieve", "preuve demandée"),
                AgentAction.finish("Réponse fondée sur la preuve."),
            ]
        )
        engine = AgenticRAGEngine(planner, [retrieval_tool], max_steps=3)

        result = engine.ask("Trouve la preuve")

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.answer, "Réponse fondée sur la preuve.")
        self.assertEqual(retriever.calls, [("preuve demandée", 2)])
        observation = result.trace.steps[0].observation
        self.assertTrue(observation.ok)
        self.assertEqual(observation.output["documents"][0]["doc_id"], "proof")
        self.assertEqual(result.to_dict()["maturity"], "experimental")

    def test_unregistered_tool_is_denied_and_never_executed(self):
        called = []
        registry = ToolRegistry([AgentTool("retrieve", "safe retrieval", lambda value: called.append(value))])
        planner = ScriptedPlanner(
            [
                AgentAction.call_tool("python", "import os"),
                AgentAction.finish("Appel refusé."),
            ]
        )

        result = AgenticRAGEngine(planner, registry, max_steps=2).ask("Exécute du code")

        self.assertEqual(result.status, "completed")
        self.assertEqual(called, [])
        self.assertFalse(result.trace.steps[0].observation.ok)
        self.assertEqual(result.trace.steps[0].observation.error, "tool_not_registered")

    def test_max_steps_bounds_tool_calls(self):
        calls = []
        tool = AgentTool("lookup", "bounded lookup", lambda value: calls.append(value) or value)
        planner = ScriptedPlanner(
            [
                AgentAction.call_tool("lookup", "one"),
                AgentAction.call_tool("lookup", "two"),
                AgentAction.call_tool("lookup", "three"),
            ]
        )

        result = AgenticRAGEngine(planner, [tool], max_steps=2).ask("continue")

        self.assertEqual(result.status, "max_steps_exceeded")
        self.assertEqual(calls, ["one", "two"])
        self.assertEqual(len(result.trace.steps), 2)
        self.assertEqual(result.trace.stop_reason, "max_steps_reached")

    def test_tool_input_limit_is_enforced_before_handler(self):
        calls = []
        registry = ToolRegistry(
            [AgentTool("lookup", "small lookup", lambda value: calls.append(value))],
            max_input_chars=3,
        )

        observation = registry.execute("lookup", "large")

        self.assertFalse(observation.ok)
        self.assertEqual(observation.error, "tool_input_too_large")
        self.assertEqual(calls, [])

    def test_llm_planner_uses_strict_json_and_registry_still_enforces_names(self):
        llm = StaticLLMClient('{"kind":"tool","tool":"missing","input":"x"}')
        planner = LLMJSONPlanner(llm)
        action = planner.plan("question", (), ())

        self.assertEqual(action.tool_name, "missing")
        self.assertEqual(ToolRegistry().execute(action.tool_name, action.tool_input).error, "tool_not_registered")

    def test_invalid_planner_output_becomes_structured_error(self):
        planner = LLMJSONPlanner(StaticLLMClient("not json"))

        result = AgenticRAGEngine(planner, [], max_steps=1).ask("question")

        self.assertEqual(result.status, "planner_error")
        self.assertEqual(result.trace.stop_reason, "planner_error")
        self.assertEqual(result.trace.error, "ValueError")


if __name__ == "__main__":
    unittest.main()
