"""Real tiny-model inference and mathematical contracts; no model downloads."""
from types import SimpleNamespace
import inspect
import os

import numpy as np
import pytest

from cheragh import Document
from cheragh.retrieval.learned import (
    ColBERTRetriever,
    ColBERTTokenEncoder,
    SPLADEEncoder,
    SPLADERetriever,
    _batched_maxsim,
    _colbert_checkpoint_class,
)
from cheragh.multimodal import Modality, MultimodalDocument
from cheragh.multimodal.colpali import ColPaliEngineAdapter, ColPaliRetriever, _split_embeddings

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture
def tokenizer(tmp_path):
    vocab = ["[PAD]", "[unused0]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "cat", "dog", ".", "?", "query", ":"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab), encoding="utf-8")
    if "vocab" in inspect.signature(transformers.BertTokenizer).parameters:
        return transformers.BertTokenizer(vocab={token: index for index, token in enumerate(vocab)})
    return transformers.BertTokenizer(vocab_file=str(path))


def _config(tokenizer):
    return transformers.BertConfig(
        vocab_size=len(tokenizer), hidden_size=8, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=16, hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
    )


@pytest.mark.parametrize("pooling", ["max", "sum"])
def test_splade_uses_mlm_log_relu_and_masked_pooling(tokenizer, pooling):
    class MLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask, **kwargs):
            logits = torch.tensor([[-2., 3., 0.], [1., 0., 2.], [2., 1., -4.]])
            result = logits[torch.arange(input_ids.shape[1]) % 3].repeat(len(input_ids), 1, 1)
            result[~attention_mask.bool()] = 999
            return SimpleNamespace(logits=result)

    encoder = SPLADEEncoder(model=MLM(), tokenizer=tokenizer, pooling=pooling)
    texts = ["cat", "cat dog"]
    batch = tokenizer(texts, padding=True, return_tensors="pt")
    raw = encoder.model(**batch).logits
    expected = torch.log1p(torch.relu(raw)) * batch["attention_mask"].unsqueeze(-1)
    expected = expected.amax(dim=1) if pooling == "max" else expected.sum(dim=1)
    np.testing.assert_allclose(encoder.encode_documents(texts), expected.numpy())
    np.testing.assert_allclose(encoder.encode_queries(texts), expected.numpy())
    assert encoder.encode_queries(["cat"]).max() < np.log1p(999)


def test_splade_loads_actual_local_mlm_checkpoint(tmp_path, tokenizer):
    checkpoint = tmp_path / "splade"
    model = transformers.BertForMaskedLM(_config(tokenizer)).eval()
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    encoder = SPLADEEncoder(str(checkpoint), model_kwargs={"local_files_only": True})
    vectors = encoder.encode_documents(["cat", "dog"])
    assert vectors.shape == (2, len(tokenizer))
    assert np.isfinite(vectors).all() and (vectors >= 0).all()
    retriever = SPLADERetriever([Document("cat"), Document("dog")], encoder=encoder)
    assert len(retriever.retrieve("cat")) == 2


class ProjectedTokens(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.batches = []

    def forward(self, input_ids, attention_mask, **kwargs):
        self.batches.append({"input_ids": input_ids.clone(), "attention_mask": attention_mask.clone()})
        return torch.stack((input_ids.float() + 1, torch.ones_like(input_ids)), dim=-1)


def test_colbert_distinguishes_role_markers_and_scoring_masks(tokenizer):
    model = ProjectedTokens()
    encoder = ColBERTTokenEncoder(model=model, tokenizer=tokenizer, query_max_length=8, document_max_length=8)
    queries = encoder.encode_queries(["cat"])
    query_batch = model.batches[-1]
    assert query_batch["input_ids"][0, 1].item() == tokenizer.convert_tokens_to_ids("[unused0]")
    assert query_batch["input_ids"][0, -1].item() == tokenizer.mask_token_id
    assert query_batch["attention_mask"][0, -1].item() == 0
    assert queries[0].shape == (8, 2), "non-attended MASK augmentation must still contribute to MaxSim"
    np.testing.assert_allclose(np.linalg.norm(queries[0], axis=1), 1)
    documents = encoder.encode_documents(["cat .", "cat dog ."])
    assert model.batches[-1]["input_ids"][0, 1].item() == tokenizer.convert_tokens_to_ids("[unused1]")
    assert [len(row) for row in documents] == [4, 5], "punctuation and real padding must be excluded"


def test_colbert_checkpoint_projection_is_loaded_and_dense_models_rejected(tmp_path, tokenizer):
    config = _config(tokenizer)
    checkpoint = tmp_path / "colbert"
    model = _colbert_checkpoint_class(torch, transformers, 4)(config).eval()
    model.linear.weight.data.fill_(0.5)
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    encoder = ColBERTTokenEncoder(str(checkpoint), projection_dimension=4, model_kwargs={"local_files_only": True})
    torch.testing.assert_close(encoder.model.linear.weight, model.linear.weight)
    assert encoder.encode_queries(["cat"])[0].shape == (32, 4)
    assert len(ColBERTRetriever([Document("cat")], token_encoder=encoder).retrieve("cat")) == 1

    dense = tmp_path / "dense"
    transformers.BertModel(config).save_pretrained(dense)
    tokenizer.save_pretrained(dense)
    with pytest.raises(ValueError, match="trained ColBERT weights"):
        ColBERTTokenEncoder(str(dense), projection_dimension=4, model_kwargs={"local_files_only": True})


def test_masked_batched_maxsim_matches_unpadded_reference():
    query = np.asarray([[1., 0.], [0., 1.]])
    documents = [np.asarray([[-1., -2.]]), np.asarray([[-3., -4.], [-4., -3.]])]
    expected = [float((query @ matrix.T).max(axis=1).sum()) for matrix in documents]
    assert _batched_maxsim(query, documents, batch_size=2) == expected == [-3., -6.]
    assert _batched_maxsim(query, documents, batch_size=1) == expected


def test_colpali_bfloat16_and_padding_cannot_inflate_negative_scores():
    embeddings = torch.tensor([[[-1., 0.], [0., 0.]], [[-2., 0.], [-3., 0.]]], dtype=torch.bfloat16)
    rows = _split_embeddings(embeddings, attention_mask=torch.tensor([[1, 0], [1, 1]]))
    assert [row.shape for row in rows] == [(1, 2), (2, 2)]
    assert _batched_maxsim(np.asarray([[1., 0.]]), rows, batch_size=2) == [-1., -2.]


def test_colpali_adapter_batches_queries_and_preserves_reasoning_tokens():
    class Processor:
        def __init__(self):
            self.batches = []

        def process_queries(self, queries):
            self.batches.append(queries)
            return {"input_ids": torch.tensor([[1, 2, 0]] * len(queries)), "attention_mask": torch.tensor([[1, 1, 0]] * len(queries))}

    model = ProjectedTokens()
    processor = Processor()
    adapter = ColPaliEngineAdapter(model=model, processor=processor, batch_size=2, query_format="processor")
    vectors = adapter.embed_queries(["one", "two", "three"])
    assert [len(batch) for batch in processor.batches] == [2, 1]
    assert [len(row) for row in vectors] == [2, 2, 2]
    assert not model.training


def test_colpali_v13_render_matches_checkpoint_and_materializes_attention_mask():
    class Tokenizer:
        bos_token = "<bos>"
        pad_token = "<pad>"

        def __call__(self, texts, **kwargs):
            self.texts = texts
            self.kwargs = kwargs
            return {"input_ids": torch.tensor([[1, 0, 0]]), "attention_mask": torch.tensor([[1, 1, 0]])}

    processor = SimpleNamespace(tokenizer=Tokenizer())
    model = ProjectedTokens()
    adapter = ColPaliEngineAdapter(model=model, processor=processor)
    rows = adapter.embed_queries(["cat"])
    assert processor.tokenizer.texts == ["<bos>Query: cat" + "<pad>" * 10 + "\n"]
    assert processor.tokenizer.kwargs["return_token_type_ids"] is True
    assert rows[0].shape == (2, 2), "attended augmentation PAD is not padding"


def test_colpali_retriever_batches_indexing_and_scores_without_padding_bias():
    class Encoder:
        def __init__(self):
            self.batch_sizes = []

        def embed_pages(self, pages):
            self.batch_sizes.append(len(pages))
            return [np.array([[-float(page.doc_id), 0.]] * int(page.doc_id)) for page in pages]

        def embed_queries(self, queries):
            return [np.asarray([[1., 0.]]) for _ in queries]

        def get_fingerprint(self):
            return "test"

    encoder = Encoder()
    pages = [MultimodalDocument("page", doc_id=str(i), modality=Modality.IMAGE) for i in (1, 2, 3)]
    retriever = ColPaliRetriever(pages, encoder, normalize_vectors=False, batch_size=2, score_batch_size=2)
    assert encoder.batch_sizes == [2, 1]
    assert [doc.score for doc in retriever.retrieve("query", top_k=3)] == [-1., -2., -3.]


@pytest.mark.skipif(not os.environ.get("CHERAGH_REAL_MODELS_CACHE"), reason="optional cached official checkpoints")
@pytest.mark.parametrize("kind", ["colbert", "splade"])
def test_official_checkpoint_local_smoke_and_reference_pooling(kind):
    """Opt in after downloading official weights; this test never downloads."""
    torch.set_num_threads(4)
    options = {"cache_dir": os.environ["CHERAGH_REAL_MODELS_CACHE"], "local_files_only": True}
    encoder = ColBERTTokenEncoder(model_kwargs=options) if kind == "colbert" else SPLADEEncoder(model_kwargs=options)
    documents = [
        Document("Paris is the capital of France and is located on the river Seine.", doc_id="france"),
        Document("Mitochondria produce energy in living cells through cellular respiration.", doc_id="biology"),
        Document("Python is a programming language with indentation-based syntax.", doc_id="python"),
    ]
    query = "What city is the capital of France?"
    retriever = ColBERTRetriever(documents, token_encoder=encoder) if kind == "colbert" else SPLADERetriever(documents, encoder=encoder)
    assert retriever.retrieve(query)[0].doc_id == "france"
    if kind == "splade":
        batch = encoder.tokenizer([query], return_tensors="pt")
        with torch.inference_mode():
            logits = encoder.model(**batch).logits
            reference = (torch.log1p(logits.relu()) * batch["attention_mask"].unsqueeze(-1)).max(dim=1).values.numpy()
        np.testing.assert_allclose(encoder.encode_queries([query]), reference, atol=1e-6)
    else:
        tokenizer = encoder.tokenizer
        raw_ids = tokenizer.encode(query, add_special_tokens=False)[:29]
        attended = [tokenizer.cls_token_id, tokenizer.convert_tokens_to_ids("[unused0]"), *raw_ids, tokenizer.sep_token_id]
        ids = attended + [tokenizer.mask_token_id] * (32 - len(attended))
        mask = [1] * len(attended) + [0] * (32 - len(attended))
        with torch.inference_mode():
            reference = encoder.model(input_ids=torch.tensor([ids]), attention_mask=torch.tensor([mask]))
            reference = torch.nn.functional.normalize(reference, p=2, dim=-1)[0].numpy()
        np.testing.assert_allclose(encoder.encode_queries([query])[0], reference, atol=1e-6)
