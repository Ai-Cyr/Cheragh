"""True contextual token pooling, preserved vectors and optional real encoders."""
from __future__ import annotations

import inspect
import re
from unittest.mock import patch

import numpy as np
import pytest

from cheragh import Document, EmbeddingModel, HashingEmbedding, MemoryVectorStore
from cheragh.ingestion.chunkers.late import LateChunker, TransformersTokenEmbedding


class ContextualTokens(EmbeddingModel):
    dimension = 2

    def __init__(self):
        self.calls = []

    def encode_tokens(self, text):
        self.calls.append(text)
        tokens = list(re.finditer(r"\S+", text))
        # The same passage changes representation with its document context.
        direction = [2.0, 1.0] if re.search(r"\bred\b", text) else [1.0, 2.0]
        return {"embeddings": np.tile(direction, (len(tokens), 1)),
                "offsets": [(token.start(), token.end()) for token in tokens]}

    def embed_documents(self, texts):
        raise AssertionError("Late chunks must not be independently re-embedded")

    def embed_query(self, text):
        vector = np.asarray([2.0, 1.0] if re.search(r"\bred\b", text) else [1.0, 2.0])
        return vector / np.linalg.norm(vector)


def test_full_document_encoding_precedes_pooling_and_identical_text_retains_its_context(tmp_path):
    model = ContextualTokens()
    docs = [Document("red context. Shared passage.", doc_id="red"),
            Document("blue context. Shared passage.", doc_id="blue")]
    chunks, embeddings = LateChunker(model, chunk_size=15, chunk_overlap=0).chunk_and_embed(docs)
    assert model.calls == [doc.content for doc in docs]
    positions = [index for index, chunk in enumerate(chunks) if chunk.content == "Shared passage."]
    assert len(positions) == 2
    assert not np.allclose(embeddings[positions[0]], embeddings[positions[1]])
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1)
    for chunk in chunks:
        parent = next(doc for doc in docs if doc.doc_id == chunk.metadata["parent_doc_id"])
        assert parent.content[chunk.metadata["source_char_start"]:chunk.metadata["source_char_end"]] == chunk.content
        assert chunk.metadata["chunker"] == "late"
        assert chunk.metadata["embedding_context"] == "document"
        assert chunk.metadata["token_start"] < chunk.metadata["token_end"]
    store = MemoryVectorStore(model)
    store.add_embeddings(chunks, embeddings)
    np.testing.assert_allclose(store.embeddings, embeddings)
    store.save(tmp_path / "index")
    restored = MemoryVectorStore.load(tmp_path / "index", model)
    assert restored.similarity_search("red", top_k=1)[0].metadata["parent_doc_id"] == "red"
    assert model.calls == [doc.content for doc in docs], "Persisting/searching must not re-encode document chunks"


def test_late_split_documents_fails_instead_of_discarding_vectors():
    with pytest.raises(ValueError, match="chunk_and_embed.*add_embeddings"):
        LateChunker(ContextualTokens()).split_documents([Document("Text.")])
    with pytest.raises(TypeError, match="encode_tokens"):
        LateChunker(HashingEmbedding())


def test_uncovered_chunk_fails_instead_of_accepting_silent_prefix_encoding():
    model = ContextualTokens()
    model.encode_tokens = lambda text: {"embeddings": [[1, 0]], "offsets": [(0, 4)]}
    with pytest.raises(ValueError, match="no contextual tokens"):
        LateChunker(model, chunk_size=8, chunk_overlap=0).chunk_and_embed([Document("One. Missing tail.")])


@pytest.mark.parametrize("output", [
    {}, {"embeddings": [], "offsets": []},
    {"embeddings": [[float("nan"), 0]], "offsets": [(0, 1)]},
    {"embeddings": [[True, False]], "offsets": [(0, 1)]},
    {"embeddings": [[1j, 1]], "offsets": [(0, 1)]},
    {"embeddings": [[1, 1]], "offsets": [(True, 1)]},
    {"embeddings": [[1, 1]], "offsets": [(0, 0)]},
    {"embeddings": [[1, 1]], "offsets": [(-1, 2)]},
    {"embeddings": [[1, 1]], "offsets": [(0, 100)]},
    {"embeddings": [[1, 1]], "offsets": [(0.0, 1.0)]},
    {"embeddings": [[1, 1]], "offsets": np.asarray(1)},
    {"embeddings": [[1, 1]], "offsets": [(0, 1), (2, 3)]},
    {"embeddings": [[1, 1], [1, 1]], "offsets": [(2, 4), (0, 2)]},
])
def test_token_provider_matrix_and_offsets_are_validated(output):
    model = ContextualTokens()
    model.encode_tokens = lambda text: output
    with pytest.raises(ValueError):
        LateChunker(model).chunk_and_embed([Document("Text.")])


def test_dimension_change_is_rejected_and_extreme_values_pool_without_overflow():
    model = ContextualTokens()
    model.encode_tokens = lambda text: {"embeddings": [[1, 2, 3]], "offsets": [(0, len(text))]}
    with pytest.raises(ValueError, match="dimension"):
        LateChunker(model).chunk_and_embed([Document("Text.")])
    model.encode_tokens = lambda text: {"embeddings": [[1e308, 1e308], [1e308, 1e308]], "offsets": [(0, 2), (2, 5)]}
    _, embeddings = LateChunker(model).chunk_and_embed([Document("Text.")])
    np.testing.assert_allclose(embeddings, [[2 ** -0.5, 2 ** -0.5]])


def test_empty_sources_and_metadata_snapshots_are_isolated_from_provider_callbacks():
    model = ContextualTokens()
    first = Document("red context. Shared passage.", metadata={"acl": {"users": ["alice"]}}, doc_id="first")
    second = Document("blue context. Shared passage.", metadata={"acl": {"users": ["bob"]}}, doc_id="second")
    original = model.encode_tokens

    def mutate_input(text):
        first.metadata["acl"]["users"].append("mallory")
        second.content = "replacement"
        return original(text)

    model.encode_tokens = mutate_input
    chunks, matrix = LateChunker(model, chunk_size=15, chunk_overlap=0).chunk_and_embed([first, second])
    assert model.calls == ["red context. Shared passage.", "blue context. Shared passage."]
    assert chunks[0].metadata["acl"]["users"] == ["alice"]
    chunks[0].metadata["acl"]["users"].append("changed")
    assert chunks[1].metadata["acl"]["users"] == ["alice"]
    assert matrix.shape == (4, 2)
    empty, vectors = LateChunker(model).chunk_and_embed([Document(" \r\n")])
    assert empty == [] and vectors.shape == (0, 2)


def test_token_embedding_is_lazy_and_fingerprint_identifies_pooling_and_capacity():
    with patch.dict("sys.modules", {"torch": None, "transformers": None}):
        encoder = TransformersTokenEmbedding("model", revision="pinned", max_tokens=2048)
        assert encoder.model is None
        assert encoder.get_fingerprint() != TransformersTokenEmbedding("model", max_tokens=4096).get_fingerprint()
        with pytest.raises(ImportError, match="torch and transformers"):
            encoder.embed_query("text")
    assert "content-token-mean-l2" in encoder.get_fingerprint()


@pytest.mark.parametrize("options", [
    {"max_tokens": False}, {"max_tokens": 0}, {"max_tokens": 3.1},
    {"model_name": ""}, {"revision": ""}, {"device": ""}, {"tokenizer": object()},
])
def test_transformer_options_fail_before_any_optional_loading(options):
    with pytest.raises((TypeError, ValueError)):
        TransformersTokenEmbedding(**options)


@pytest.fixture
def tiny_encoder(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "red", "blue", "context", "shared", "passage", "."]
    path = tmp_path / "tiny-context-encoder"
    path.mkdir()
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    if "vocab" in inspect.signature(transformers.BertTokenizer).parameters:
        tokenizer = transformers.BertTokenizer(vocab={token: index for index, token in enumerate(vocab)})
    else:
        tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocab_file))
    config = transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, hidden_dropout_prob=0,
        attention_probs_dropout_prob=0, max_position_embeddings=32,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = transformers.BertModel(config)
    tokenizer.save_pretrained(path)
    model.save_pretrained(path)
    return tokenizer, model, path


def test_real_transformer_encodes_full_context_once_and_excludes_special_tokens(tiny_encoder):
    torch = pytest.importorskip("torch")
    tokenizer, model, _ = tiny_encoder
    encoder = TransformersTokenEmbedding("tiny", tokenizer=tokenizer, model=model, max_tokens=32)
    forwards = []
    hook = model.register_forward_pre_hook(lambda *args: forwards.append(1))
    docs = [Document("red context. Shared passage.", doc_id="red"),
            Document("blue context. Shared passage.", doc_id="blue")]
    chunks, matrix = LateChunker(encoder, chunk_size=15, chunk_overlap=0).chunk_and_embed(docs)
    hook.remove()
    assert len(forwards) == 2, "Late pooling must not invoke a transformer per chunk"
    positions = [index for index, chunk in enumerate(chunks) if chunk.content == "Shared passage."]
    assert len(positions) == 2
    assert not np.allclose(matrix[positions[0]], matrix[positions[1]], atol=1e-7, rtol=0)
    text = docs[0].content
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, return_special_tokens_mask=True)
    offsets, special = encoded.pop("offset_mapping"), encoded.pop("special_tokens_mask")
    with torch.inference_mode():
        raw = model(**encoded).last_hidden_state[0]
    span = chunks[positions[0]].metadata
    selected = (~special[0].bool()) & (offsets[0, :, 0] < span["source_char_end"]) & (offsets[0, :, 1] > span["source_char_start"])
    expected = raw[selected].mean(dim=0).numpy()
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(matrix[positions[0]], expected, atol=1e-6)
    whole, whole_vectors = LateChunker(encoder).chunk_and_embed([Document(text)])
    assert len(whole) == 1
    np.testing.assert_allclose(encoder.embed_query(text), whole_vectors[0], atol=1e-7)


def test_real_model_limit_rejects_before_forward_without_truncating(tiny_encoder):
    tokenizer, model, _ = tiny_encoder
    encoder = TransformersTokenEmbedding("tiny", tokenizer=tokenizer, model=model, max_tokens=1000)
    with patch.object(model, "forward", side_effect=AssertionError("must not forward oversized input")):
        with pytest.raises(ValueError, match="never truncates"):
            encoder.encode_tokens("shared " * 40)
    short_budget = TransformersTokenEmbedding("tiny", tokenizer=tokenizer, model=model, max_tokens=3)
    with patch.object(model, "forward", side_effect=AssertionError("must not forward oversized input")):
        with pytest.raises(ValueError, match="encoder limit is 3"):
            short_budget.encode_tokens("shared passage")


def test_auto_model_and_tokenizer_load_the_local_checkpoint_without_remote_code(tiny_encoder):
    _, _, path = tiny_encoder
    encoder = TransformersTokenEmbedding(str(path), max_tokens=32)
    actual = encoder.embed_documents(["shared passage", "red context"])
    assert actual.shape == (2, 8) and encoder.dimension == 8
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
    with pytest.raises(ValueError, match="non-empty"):
        encoder.encode_tokens("")
