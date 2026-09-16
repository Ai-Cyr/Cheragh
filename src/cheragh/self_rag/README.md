# Model-conditioned Self-RAG inference

`SelfRAGEngine` retains the existing lightweight relevance/refinement API.
`SegmentedSelfRAGEngine` is a separate inference path using actual model
reflection probabilities. Its algorithm follows segment generation conditioned
independently on each retrieved passage, three retrieval states, and a beam
of candidate continuations. The score uses normalized relevance, half credit
for partial support, signed utility, and optionally geometric-mean sequence
probability. See [Self-RAG, Algorithm 1 and Appendix A.3](https://arxiv.org/abs/2310.11511).

The product of successive positive segment scores follows the authors'
[long-form inference implementation](https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_long_form_static.py).
This implementation retains branch-specific evidence for Continue, keeps
completed hypotheses independently, reads utility from utility-token logits,
and accumulates products as log scores. It does not reproduce the reference
script's utility-group indexing mistake. Non-positive segment scores are
explicitly pruned because signed utility otherwise permits negative path
products to reverse ranking on later steps.

```python
from cheragh.self_rag import SegmentedSelfRAGEngine, TransformersSelfRAGDecoder

# An explicit checkpoint load; provide a pinned revision in production.
decoder = TransformersSelfRAGDecoder.from_pretrained(
    "selfrag/selfrag_llama2_7b",
    model_kwargs={"device_map": "auto", "dtype": "auto"},
)
engine = SegmentedSelfRAGEngine(
    retriever,
    decoder,
    top_k=5,
    beam_width=2,
    max_segments=6,
    max_new_tokens=256,
    retrieval_threshold=0.2,
)
result = engine.ask("How did the US states get their names?")
print(result.answer)
print(result.to_dict())
```

Torch and Transformers are optional dependencies. Loading with `device_map`
may also require Accelerate. The adapter validates atomic, distinct reflection
token IDs and their compatibility with the vocabulary, uses greedy full-logit
decoding with a KV cache, and stops at retrieval/EOS/token boundaries. It
never manufactures probabilities from generated textual scores. The included
factory supports ordinary Transformers checkpoint revision and local-cache
options, and disables remote custom model code. The official released
[Self-RAG checkpoint](https://huggingface.co/selfrag/selfrag_llama2_7b) supplies
the learned token behavior; adding token strings to an ordinary model alone
does not train that behavior.

Contract details:

- Initial decisions exclude Continue because no passage exists. With current
  evidence, a most-likely Continue reuses that branch's passage. Otherwise an
  optional threshold applies strictly to Yes/(Yes+No); equality selects No.
- A retrieval query contains the task and preceding segment. Retrieved
  passages are capped and snapshotted before candidate generation. A missing
  relevance token on a newly retrieved passage is an error. Missing support
  or utility tokens contribute zero and remain visible in diagnostics.
- Source citations are attached by the engine from the selected branch, not
  accepted from arbitrary generated markers. They identify the conditioning
  passage. Reflection support is a model assessment, not factual verification.
- Model calls, retrieval calls, generated tokens, segment depth, and beam
  widths are bounded. Tokens spent on discarded candidates still count.
  Reaching a bound returns an explicit status; a length-limited generation
  is never silently declared complete. Oversized prompts fail instead of
  silently discarding evidence or history.
- Generation and reflection scoring are sequential per candidate in this
  adapter; GPU batching and distributed inference are not implemented.

Tests include independently enumerated candidate-path products, continuation
state isolation, early EOS handling, provenance, budgets, scripted torch logits
at reflection positions, real tiny-Llama cached decoding, and a local
`save_pretrained`/`from_pretrained` round-trip. These tests validate inference
mechanics, not output quality. Training the critic/generator, loading the full
7B/13B checkpoints, reproducing the paper's corpus/retriever, and running its
QA, citation, or factuality benchmarks remain separate requirements.
