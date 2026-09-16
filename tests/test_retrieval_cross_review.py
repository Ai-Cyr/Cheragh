"""Cross-review regressions for token boundaries and source expansion."""
import inspect
import os

import numpy as np
import pytest

from cheragh import Document, HashingEmbedding, StaticLLMClient
from cheragh.cache import CachedRetriever, MemoryCache
from cheragh.contextual_compression import ContextualCompressionRetriever
from cheragh.hyde import HyDERetriever
from cheragh.ingestion.chunkers import SemanticChunker, TokenTextChunker
from cheragh.ingestion.chunkers.structured import HTMLSectionChunker, MarkdownHeaderChunker
from cheragh.retrieval.parent_child import ParentChildRetriever
from cheragh.security import AccessControlledRetriever, Principal
from cheragh.self_query import SelfQueryRetriever


class OrderedRetriever:
    def __init__(self, documents):
        self.documents = documents

    def retrieve(self, query, top_k=5):
        return self.documents[:top_k]


def _child(content, doc_id, parent="p", **metadata):
    return Document(content, {"chunk_role": "child_chunk", "parent_section_id": parent, **metadata}, doc_id)


def test_reconstructed_parent_preserves_access_metadata_and_owns_snapshot():
    children = [_child("private first", "c1", tenant_id="a", allowed_users=["alice"], chunk_index=0),
                _child("private second", "c2", tenant_id="a", allowed_users=["alice"], chunk_index=1)]
    source = ParentChildRetriever.from_hierarchical_chunks(children)
    parent = source.retrieve("private", 1)[0]
    assert parent.metadata["tenant_id"] == "a"
    assert parent.metadata["allowed_users"] == ["alice"]
    parent.metadata["allowed_users"].append("bob")
    assert children[0].metadata["allowed_users"] == ["alice"]
    wrapped = AccessControlledRetriever(source, Principal("bob", tenant_ids={"a"}))
    assert wrapped.retrieve("private") == []


@pytest.mark.parametrize("field,values", [("tenant_id", ["a", "b"]), ("classification", ["public", "secret"]),
                                          ("custom_permission", ["read", "deny"])])
def test_reconstruction_rejects_incompatible_child_authorization(field, values):
    children = [_child("first", "c1", **{field: values[0]}), _child("second", "c2", **{field: values[1]})]
    with pytest.raises(ValueError, match="conflicting source metadata"):
        ParentChildRetriever.from_hierarchical_chunks(children)


@pytest.mark.parametrize("cached", [False, True])
def test_child_authorization_cannot_grant_access_to_secret_parent(cached):
    children = [_child("Public excerpt", "c1", classification="public", tenant_id="a")]
    secured = AccessControlledRetriever(OrderedRetriever(children), Principal("alice", tenant_ids={"a"}))
    if cached:
        secured = CachedRetriever(secured, MemoryCache())
    source = ParentChildRetriever([Document("SECRET parent body", {"tenant_id": "a", "classification": "secret"}, "p")],
                                  children, child_retriever=secured)
    assert source.retrieve("query", 1) == []


def test_parent_authorization_does_not_starve_allowed_later_parent():
    children = [_child("Public excerpt", "c1", classification="public", tenant_id="a"),
                _child("Allowed excerpt", "c2", parent="ok", classification="public", tenant_id="a")]
    principal = Principal("alice", tenant_ids={"a"})
    source = ParentChildRetriever([Document("Secret body", {"tenant_id": "a", "classification": "secret"}, "p"),
                                   Document("Allowed body", {"tenant_id": "a", "classification": "public"}, "ok")],
                                  children, child_retriever=OrderedRetriever(children), principal=principal, top_k_children=1)
    assert [doc.doc_id for doc in source.retrieve("query", 1)] == ["ok"]


def test_unknown_external_parent_hit_does_not_displace_known_parent():
    children = [_child("Unknown", "x", parent="missing"), _child("Known", "y")]
    children[0].score, children[1].score = 100, 1
    source = ParentChildRetriever([Document("Parent", doc_id="p")], child_retriever=OrderedRetriever(children))
    assert [doc.doc_id for doc in source.retrieve("query", 1)] == ["p"]


class BoundaryTokenizer:
    def __call__(self, text, **kwargs):
        if kwargs.get("return_offsets_mapping"):
            return {"input_ids": list(range(len(text))), "offset_mapping": [(i, i + 1) for i in range(len(text))]}
        return {"input_ids": [1] * (len(text) * 2)}


def _check_chunks(tokenizer, text, *, size=4):
    chunks = TokenTextChunker(chunk_size=size, chunk_overlap=1, tokenizer=tokenizer).split_documents([Document(text, doc_id="src")])
    spans = []
    covered = set()
    for doc in chunks:
        start, end = doc.metadata["source_char_start"], doc.metadata["source_char_end"]
        assert doc.content == text[start:end]
        assert len(tokenizer(doc.content, add_special_tokens=False)["input_ids"]) <= size
        covered.update(range(start, end))
        spans.append((start, end))
    assert len(spans) == len(set(spans)), "shared byte offsets must not create duplicate chunks"
    assert all(index in covered for index, char in enumerate(text) if not char.isspace())


def test_injected_tokenizer_rechecks_rendered_tokens_without_losing_characters():
    _check_chunks(BoundaryTokenizer(), "abcdef", size=3)


@pytest.mark.parametrize("offsets", [[(2, 3), (0, 1)], [(0, 2), (1, 1.5)], [(0, 4)], [(2, 1)]])
def test_tokenizer_rejects_malformed_offsets(offsets):
    def tokenizer(text, **kwargs):
        return {"input_ids": [1] * len(offsets), "offset_mapping": offsets}

    with pytest.raises(ValueError, match="offset"):
        TokenTextChunker(chunk_size=2, chunk_overlap=0, tokenizer=tokenizer).split_text("abc")


def test_real_wordpiece_tokenizer_preserves_unicode_offsets_and_budget(tmp_path):
    transformers = pytest.importorskip("transformers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world", "cafe", "e", "##lan", "中", "文", "."]
    if "vocab" in inspect.signature(transformers.BertTokenizer).parameters:
        tokenizer = transformers.BertTokenizer(vocab={token: index for index, token in enumerate(vocab)})
    else:
        path = tmp_path / "vocab.txt"
        path.write_text("\n".join(vocab), encoding="utf-8")
        tokenizer = transformers.BertTokenizerFast(vocab_file=str(path))
    _check_chunks(tokenizer, "Hello café élan 中文. Hello world.")


def test_real_byte_bpe_tokenizer_avoids_duplicate_unicode_chunks():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.ByteLevelBPETokenizer()
    backend.train_from_iterator(["hello world", "café 中文 🚀🙂 multiwordwordword"], vocab_size=256, min_frequency=2)
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend._tokenizer)
    _check_chunks(tokenizer, "café 中文 🚀🙂 multiwordwordword")


@pytest.mark.skipif(not os.environ.get("CHERAGH_REAL_MODELS_CACHE"), reason="opt-in cached model tokenizers")
@pytest.mark.parametrize("name", ["colbert-ir/colbertv2.0", "cross-encoder/nli-MiniLM2-L6-H768"])
def test_pretrained_tokenizers_cover_non_ascii_source(name):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(name, cache_dir=os.environ["CHERAGH_REAL_MODELS_CACHE"], local_files_only=True)
    _check_chunks(tokenizer, "café élan 中文 🚀🙂 multiwordwordword")


def test_wrapped_line_does_not_allow_compression_to_drop_negation():
    source = OrderedRetriever([Document("This action is not\npermitted.")])
    compressor = ContextualCompressionRetriever(source, StaticLLMClient("permitted."), min_compressed_length=0)
    with pytest.raises(ValueError, match="verbatim"):
        compressor.retrieve("Allowed?")


class LargeEmbeddings(HashingEmbedding):
    def embed_documents(self, texts):
        return np.array([[1e300, 1e300] for _ in texts])


def test_hyde_cosine_stays_finite_for_large_embeddings():
    source = HyDERetriever([Document("source")], LargeEmbeddings(), StaticLLMClient("hypothesis"), n_hypotheses=3, similarity="cosine")
    assert source.retrieve("query")[0].score == pytest.approx(1.0)


def test_hyde_rejects_unrepresentable_dot_score():
    source = HyDERetriever([Document("source")], LargeEmbeddings(), StaticLLMClient("hypothesis"))
    with pytest.raises(ValueError, match="finite numeric"):
        source.retrieve("query")


def test_semantic_boundaries_are_invariant_to_extreme_embedding_scale():
    chunks = SemanticChunker(LargeEmbeddings(), max_chunk_size=500, min_chunk_size=0).split_text("First. Second. Third.")
    assert len(chunks) == 1
    assert chunks[0]["avg_adjacent_similarity"] == pytest.approx(1.0)


@pytest.mark.parametrize("raw", ['{}', '{"cleaned_query":"fact"}',
                                '{"cleaned_query":"fact","filters":{"year":2024},"filters":{}}',
                                '{"cleaned_query":"fact","filters":{"year":NaN}}'])
def test_self_query_does_not_silently_remove_missing_or_duplicate_filters(raw):
    source = SelfQueryRetriever([Document("fact", {"year": 2024})], HashingEmbedding(), StaticLLMClient(raw), {"year": "integer"})
    with pytest.raises(ValueError):
        source.retrieve("In 2024?")


@pytest.mark.parametrize("chunker,text", [(MarkdownHeaderChunker(), "No.\n# Topic\nA longer text follows here."),
                                         (HTMLSectionChunker(), "<p>No.</p><h1>Topic</h1><p>A longer text follows here.</p>")])
def test_structured_chunkers_do_not_discard_short_preamble_facts(chunker, text):
    chunks = chunker.split_documents([Document(text, {"nested": [1]}, "src")])
    assert chunks[0].content == "No."
    chunks[0].metadata["nested"].append(2)
    assert chunks[1].metadata["nested"] == [1]
