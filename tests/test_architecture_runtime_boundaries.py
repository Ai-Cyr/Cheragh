"""Regression checks for isolation and work limits in composed architectures."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Barrier

import pytest

from cheragh import Document
from cheragh.schema import RAGResponse
from cheragh.conversation import ConversationalRAGEngine, ConversationTurn, InMemoryConversationStore
from cheragh.federated import FederatedRAGEngine
from cheragh.structured import SQLRAGEngine
from cheragh.security import AccessControlledRAGEngine, AccessPolicy, Principal


def sql_engine(**kwargs):
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE public_t(value TEXT);"
        "INSERT INTO public_t VALUES ('public');"
        "CREATE TABLE secret_t(value TEXT);"
        "INSERT INTO secret_t VALUES ('private');"
    )
    return SQLRAGEngine(connection=connection, table_allowlist=["public_t"], **kwargs)


@pytest.mark.parametrize("query", [
    "SELECT * FROM 'secret_t'",
    "SELECT * FROM /*comment*/ secret_t",
    "WITH secret_t AS (SELECT * FROM main.secret_t) SELECT * FROM secret_t",
])
def test_sqlite_authorizer_enforces_allowlist_after_sql_parsing(query):
    engine = sql_engine()
    with pytest.raises((ValueError, sqlite3.DatabaseError)):
        engine.execute_sql(query)
    assert engine.execute_sql("SELECT * FROM public_t").rows == [{"value": "public"}]
    assert [table.name for table in engine.schema()] == ["public_t"]


def test_sql_budget_remains_active_while_fetching_rows_and_recovers():
    engine = sql_engine(max_rows=100_000, max_sql_steps=1000)
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        engine.execute_sql(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 100000) "
            "SELECT x FROM n LIMIT 100000"
        )
    assert engine.execute_sql("SELECT * FROM public_t").rows == [{"value": "public"}]
    with pytest.raises(sqlite3.DatabaseError):
        engine.connection.execute("DELETE FROM public_t")


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_sql_step_budget_rejects_invalid_values(limit):
    with pytest.raises((ValueError, TypeError)):
        SQLRAGEngine(max_sql_steps=limit)


def test_sql_readonly_uri_escapes_filename(tmp_path):
    path = tmp_path / "records?archive#1.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE records(value TEXT); INSERT INTO records VALUES ('kept');")
    connection.close()
    engine = SQLRAGEngine(database=path, table_allowlist=["records"])
    assert engine.execute_sql("SELECT * FROM records").rows == [{"value": "kept"}]


@pytest.mark.parametrize("schema", ["temp", "extra"])
@pytest.mark.parametrize("selection", ["COUNT(*)", "1"])
def test_sql_allowlist_does_not_authorize_ambiguous_schema_reads(schema, selection):
    connection = sqlite3.connect(":memory:")
    if schema == "extra":
        connection.execute("ATTACH ':memory:' AS extra")
    else:
        connection.execute("CREATE TABLE main.public_t(value TEXT)")
    connection.execute(f"CREATE TABLE {schema}.public_t(value TEXT)")
    connection.execute(f"INSERT INTO {schema}.public_t VALUES ('private')")
    connection.commit()
    engine = SQLRAGEngine(connection=connection, table_allowlist=["public_t"])
    with pytest.raises(sqlite3.DatabaseError):
        engine.execute_sql(f"SELECT {selection} FROM public_t")


def test_federated_concurrent_requests_do_not_share_evidence_or_diagnostics():
    source_barrier = Barrier(2)
    generation_barrier = Barrier(2)

    def source(query):
        source_barrier.wait(timeout=5)
        return [Document(query, doc_id=query)]

    class LLM:
        def generate(self, prompt, **kwargs):
            generation_barrier.wait(timeout=5)
            return "answer"

    engine = FederatedRAGEngine({"records": source}, llm_client=LLM())
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(engine.ask, ["tenant-a", "tenant-b"]))
    for query, result in zip(["tenant-a", "tenant-b"], results):
        assert len(result.source_results) == 1
        assert [doc.content for doc in result.source_results[0].documents] == [query]
        assert result.response.metadata["sources_queried"] == ["records"]
    assert engine.retriever.last_results == []


def test_federated_nested_retrieval_preserves_outer_results():
    class LLM:
        def generate(self, prompt, **kwargs):
            engine.retriever.retrieve("nested")
            return "answer"

    engine = FederatedRAGEngine(
        {"records": lambda query: [Document(query, doc_id=query)]}, llm_client=LLM(),
    )
    result = engine.ask("outer")
    assert result.source_results[0].documents[0].content == "outer"


def test_federated_nested_source_retrieval_preserves_outer_diagnostics():
    def source(query):
        if query == "outer":
            engine.retriever.retrieve("private-auxiliary-query")
        return [Document(query, doc_id=query)]

    engine = FederatedRAGEngine({"records": source})
    result = engine.ask("outer")
    assert len(result.source_results) == 1
    assert result.source_results[0].documents[0].content == "outer"
    assert "private-auxiliary-query" not in str(result.to_dict())


def test_federated_provider_errors_do_not_expose_exception_payloads():
    def source(query):
        raise RuntimeError("credential=private-provider-token")

    result = FederatedRAGEngine({"records": source}).ask("q")
    assert result.source_results[0].error == "RuntimeError"
    assert "private-provider-token" not in str(result.to_dict())


def test_access_controlled_federation_does_not_restore_raw_source_answers():
    class LLM:
        def generate(self, prompt, **kwargs):
            assert "private-evidence" not in prompt
            assert "private-source-answer" not in prompt
            return "public-answer"

    documents = [
        Document("private-evidence", {"allowed_users": ["other"]}, doc_id="private"),
        Document("public-evidence", {"allowed_users": ["alice"]}, doc_id="public"),
    ]
    engine = FederatedRAGEngine(
        {"records": lambda query: {"documents": documents, "answer": "private-source-answer"}},
        llm_client=LLM(),
    )
    protected = AccessControlledRAGEngine(
        engine, policy=AccessPolicy(require_tenant_match=False), default_principal=Principal("alice"),
    )
    result = protected.ask("question", top_k=1)
    assert [doc.content for doc in result.retrieved_documents] == ["public-evidence"]
    assert [doc.content for doc in result.source_results[0].documents] == ["public-evidence"]
    assert result.source_results[0].answer is None


def test_conversation_storage_is_bounded_and_owns_its_values():
    memory = InMemoryConversationStore(max_sessions=2, max_turns_per_session=2)
    turn = ConversationTurn("q", "a", metadata={"nested": {"value": "original"}})
    memory.append("a", turn)
    turn.metadata["nested"]["value"] = "changed"
    memory.append("b", turn)
    assert memory.get("a")[0].metadata["nested"]["value"] == "original"
    memory.append("c", turn)
    assert memory.get("b") == []
    memory.append("a", ConversationTurn("second", "a"))
    memory.append("a", ConversationTurn("third", "a"))
    assert [turn.user for turn in memory.get("a")] == ["second", "third"]
    exported = memory.to_dict()
    exported["c"][0]["metadata"]["nested"]["value"] = "external"
    assert memory.get("c")[0].metadata["nested"]["value"] == "changed"


def test_conversation_closes_provider_stream_without_remembering_partial_answer():
    class Stream:
        closed = False

        def __iter__(self):
            return self

        def __next__(self):
            return "partial"

        def close(self):
            self.closed = True

    provider_stream = Stream()

    class Engine:
        def stream(self, query, **kwargs):
            return provider_stream

    engine = ConversationalRAGEngine(Engine())
    stream = engine.stream("question")
    assert next(stream) == "partial"
    stream.close()
    assert provider_stream.closed
    assert engine.history() == []


def test_conversation_turn_indices_remain_monotone_when_history_is_evicted():
    class Engine:
        def ask(self, query, **kwargs):
            return RAGResponse(query=query, answer="answer", sources=[], retrieved_documents=[], prompt="")

    memory = InMemoryConversationStore(max_turns_per_session=1)
    engine = ConversationalRAGEngine(Engine(), memory=memory)
    indices = [engine.ask("q").metadata["conversation"]["turn_index"] for _ in range(3)]
    assert indices == [1, 2, 3]
    engine.clear()
    assert engine.ask("q").metadata["conversation"]["turn_index"] == 1


def test_conversation_closes_stream_owner_when_iterator_is_separate():
    class Owner:
        closed = False

        def __iter__(self):
            return iter(["first", "second"])

        def close(self):
            self.closed = True

    owner = Owner()

    class Engine:
        def stream(self, query, **kwargs):
            return owner

    stream = ConversationalRAGEngine(Engine()).stream("q")
    assert next(stream) == "first"
    stream.close()
    assert owner.closed
