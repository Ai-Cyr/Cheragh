"""Public strategy selection preserves capabilities through supported caches."""
import re

import numpy as np
import pytest

from cheragh import (
    CachedEmbeddingModel, Document, EmbeddingModel, HashingEmbedding,
    MemoryCache, RAGEngine, build_chunker, list_chunking_strategies,
)


class TokenEncoder(EmbeddingModel):
    def __init__(self):
        self.encoded = []
        self.queries = 0

    def encode_tokens(self, text):
        self.encoded.append(text)
        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        return {"offsets": offsets, "embeddings": np.tile([1., len(text)], (len(offsets), 1))}

    def embed_documents(self, texts):
        raise AssertionError("Must preserve full-document token embeddings")

    def embed_query(self, text):
        self.queries += 1
        return np.array([1., 1.])


def test_explicit_nested_embedding_cache_preserves_late_pooling_and_query_cache():
    model = TokenEncoder()
    wrapped = CachedEmbeddingModel(CachedEmbeddingModel(model, MemoryCache()), MemoryCache())
    text = "First sentence has background. Second sentence needs that background."
    engine = RAGEngine.from_documents(
        [Document(text)], embedding_model=wrapped, retriever_type="memory",
        chunk=True, chunking_strategy="late", chunk_size=35, chunk_overlap=0,
    )
    assert model.encoded == [text]
    store = engine.retriever.store
    assert len(store.documents) > 1
    vector = np.array([1., len(text)])
    vector /= np.linalg.norm(vector)
    np.testing.assert_allclose(store.embeddings, np.tile(vector, (len(store.documents), 1)))
    assert store.similarity_search("background", top_k=1)
    assert store.similarity_search("background", top_k=1)
    assert model.queries == 1


def test_nested_cache_does_not_make_hashing_semantic():
    wrapped = CachedEmbeddingModel(CachedEmbeddingModel(HashingEmbedding(), MemoryCache()), MemoryCache())
    with pytest.raises(ValueError, match="semantic embeddings"):
        build_chunker("semantic", embedding_model=wrapped)


def test_cyclic_cache_is_rejected_before_provider_work():
    wrapped = CachedEmbeddingModel(TokenEncoder(), MemoryCache())
    wrapped.model = wrapped
    with pytest.raises(ValueError, match="Cyclic"):
        build_chunker("late", embedding_model=wrapped)


def test_strategy_descriptions_are_independent_and_expose_advanced_modes():
    first = list_chunking_strategies()
    assert {"semantic", "late", "contextual"} <= {entry["id"] for entry in first}
    first[0]["label"] = "caller mutation"
    assert list_chunking_strategies()[0]["label"] != "caller mutation"
