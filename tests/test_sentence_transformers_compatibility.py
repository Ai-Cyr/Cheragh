"""Exercise real ST 5/6 APIs with tiny local checkpoints, without hub access.

Random weights test adapter compatibility, batching and masks, not relevance.
"""

import inspect
import json

import numpy as np
import pytest

from cheragh.base import Document, SentenceTransformerEmbedding
from cheragh.multimodal import CLIPMultimodalEmbedding, Modality, MultimodalDocument, MultimodalQuery
from cheragh.reranking import CrossEncoderReranker
from cheragh.retrieval.learned import SentenceTransformerTokenEncoder

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
st = pytest.importorskip("sentence_transformers")


@pytest.fixture
def bert_checkpoint(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "cat", "dog", "red", "blue", "query", "document", ":"]
    path = tmp_path / "bert"
    path.mkdir()
    vocabulary = path / "vocab.txt"
    vocabulary.write_text("\n".join(vocab), encoding="utf-8")
    if "vocab" in inspect.signature(transformers.BertTokenizer).parameters:
        tokenizer = transformers.BertTokenizer(vocab={token: i for i, token in enumerate(vocab)})
    else:
        tokenizer = transformers.BertTokenizer(vocab_file=str(vocabulary))
    tokenizer.save_pretrained(path)
    config = transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, hidden_dropout_prob=0,
        attention_probs_dropout_prob=0, max_position_embeddings=32,
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        transformers.BertModel(config).save_pretrained(path)
    return path


@pytest.fixture
def sentence_checkpoint(tmp_path, bert_checkpoint):
    # Use the common public compatibility imports to write each version's own
    # checkpoint layout, then exercise Cheragh's actual from-path constructor.
    from sentence_transformers import models

    model = st.SentenceTransformer(
        modules=[models.Transformer(str(bert_checkpoint), max_seq_length=16),
                 models.Pooling(8, include_prompt=False)],
        prompts={"query": "query: ", "document": "document: "},
        device="cpu",
    )
    path = tmp_path / "sentence"
    model.save(str(path))
    return path


def test_sentence_embedding_loads_and_matches_real_encode(sentence_checkpoint):
    encoder = SentenceTransformerEmbedding(str(sentence_checkpoint), device="cpu", local_files_only=True)
    texts = ["cat", "red dog"]
    actual = encoder.embed_documents(texts)
    expected = encoder.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    assert actual.shape == (2, 8)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(encoder.embed_query(texts[0]), actual[0], atol=1e-6)


@pytest.mark.parametrize("role,prompt", [("query", "query: "), ("document", "document: ")])
def test_token_encoder_preserves_pre_pooling_tokens_and_attention(sentence_checkpoint, role, prompt):
    encoder = SentenceTransformerTokenEncoder(
        str(sentence_checkpoint), model_kwargs={"device": "cpu", "local_files_only": True},
    )
    encoder.model.eval()
    texts = ["cat", "red dog"]
    encode = encoder.encode_queries if role == "query" else encoder.encode_documents
    tokens, mask = encode(texts)
    features = encoder.model.tokenize([prompt + text for text in texts])
    expected_mask = features["attention_mask"].clone()
    with torch.inference_mode():
        expected = encoder.model.forward(features)["token_embeddings"]
    torch.testing.assert_close(tokens, expected)
    torch.testing.assert_close(mask, expected_mask)
    assert mask[0].sum() < mask[1].sum(), "padding must not enter late-interaction scoring"
    assert mask[:, :3].all(), "manual role prompts belong to this pre-pooling token contract"
    assert not tokens.requires_grad
    # include_prompt=False only changes pooling when prompt_length is supplied.
    # ST6 stopped mutating that pooling mask; Cheragh deliberately returns the
    # encoder attention mask and does not depend on the old side effect.
    assert encoder.model[1].include_prompt is False
    assert "prompt_length" not in features


def test_cross_encoder_loads_and_preserves_predictions_and_sources(tmp_path, bert_checkpoint):
    tokenizer = transformers.AutoTokenizer.from_pretrained(bert_checkpoint, local_files_only=True)
    config = transformers.BertConfig.from_pretrained(bert_checkpoint, num_labels=1)
    with torch.random.fork_rng():
        torch.manual_seed(43)
        model = transformers.BertForSequenceClassification(config)
    path = tmp_path / "cross-encoder"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    reranker = CrossEncoderReranker(str(path), device="cpu", local_files_only=True)
    docs = [Document("cat", doc_id="a", metadata={"source": "one"}),
            Document("red dog", doc_id="b", metadata={"source": "two"})]
    expected = np.asarray(reranker.model.predict([("cat", doc.content) for doc in docs])).reshape(-1)
    result = reranker.rerank("cat", docs, top_k=2)
    order = np.argsort(-expected, kind="stable")
    assert [doc.doc_id for doc in result] == [docs[i].doc_id for i in order]
    np.testing.assert_allclose([doc.score for doc in result], expected[order], atol=1e-6)
    assert all(doc.metadata["source"] == docs[i].metadata["source"] for doc, i in zip(result, order))
    assert all(doc.score is None and "rerank_score" not in doc.metadata for doc in docs)


@pytest.fixture
def clip_checkpoint(tmp_path):
    from sentence_transformers import models

    pytest.importorskip("PIL")
    path = tmp_path / "clip-base"
    path.mkdir()
    tokens = ["<|startoftext|>", "<|endoftext|>"]
    tokens += list("abcdefghijklmnopqrstuvwxyz")
    tokens += [letter + "</w>" for letter in "abcdefghijklmnopqrstuvwxyz"]
    vocab = {token: i for i, token in enumerate(tokens)}
    (path / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    if "vocab" in inspect.signature(transformers.CLIPTokenizer).parameters:
        tokenizer = transformers.CLIPTokenizer(vocab=vocab, merges=[], model_max_length=16)
    else:
        tokenizer = transformers.CLIPTokenizer(
            vocab_file=str(path / "vocab.json"), merges_file=str(path / "merges.txt"), model_max_length=16,
        )
    image_processor = transformers.CLIPImageProcessor(size={"shortest_edge": 8}, crop_size={"height": 8, "width": 8})
    processor = transformers.CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)
    config = transformers.CLIPConfig(
        text_config={"vocab_size": len(tokenizer), "hidden_size": 8, "intermediate_size": 16,
                     "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 16,
                     "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
                     "pad_token_id": tokenizer.pad_token_id},
        vision_config={"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                       "num_attention_heads": 2, "image_size": 8, "patch_size": 4},
        projection_dim=4,
    )
    with torch.random.fork_rng():
        torch.manual_seed(44)
        transformers.CLIPModel(config).save_pretrained(path)
    processor.save_pretrained(path)
    model = st.SentenceTransformer(modules=[models.CLIPModel(str(path))], device="cpu")
    checkpoint = tmp_path / "clip-sentence"
    model.save(str(checkpoint))
    return checkpoint


def test_clip_loads_and_preserves_mixed_text_image_order(clip_checkpoint, tmp_path):
    from PIL import Image

    encoder = CLIPMultimodalEmbedding(str(clip_checkpoint), device="cpu", local_files_only=True)
    path = tmp_path / "red.png"
    Image.new("RGB", (8, 8), color="red").save(path)
    docs = [MultimodalDocument("cat"),
            MultimodalDocument("red", modality=Modality.IMAGE, uri=str(path)),
            MultimodalDocument("dog")]
    vectors = encoder.embed_documents(docs)
    assert vectors.shape == (3, 4)
    assert np.isfinite(vectors).all()
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
    for index, doc in enumerate(docs):
        query = MultimodalQuery(text=doc.content, modality=doc.modality, uri=doc.uri)
        np.testing.assert_allclose(encoder.embed_query(query), vectors[index], atol=1e-6)
    with Image.open(path) as image:
        image_vector = encoder.model.encode([image.convert("RGB")], normalize_embeddings=True)[0]
    text_vectors = encoder.model.encode(["cat", "dog"], normalize_embeddings=True)
    expected = np.stack([text_vectors[0], image_vector, text_vectors[1]])
    np.testing.assert_allclose(vectors, expected, atol=1e-6)
    np.testing.assert_allclose(encoder.embed_documents([docs[0], docs[2]]), text_vectors, atol=1e-6)
    np.testing.assert_allclose(encoder.embed_documents([docs[1], docs[1]]), [image_vector, image_vector], atol=1e-6)
    reordered = [docs[1], docs[2], docs[0], docs[1]]
    np.testing.assert_allclose(encoder.embed_documents(reordered), expected[[1, 2, 0, 1]], atol=1e-6)
    # Image documents without an asset deliberately use the caption as text.
    caption = MultimodalDocument("cat", modality=Modality.IMAGE)
    np.testing.assert_allclose(encoder.embed_documents([caption, docs[1]]), expected[:2], atol=1e-6)
    assert encoder.embed_documents([]).size == 0
