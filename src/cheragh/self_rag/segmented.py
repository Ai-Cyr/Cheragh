"""Model-conditioned Self-RAG segment decoding and passage-level beam search.

This is an explicit alternative to the lexical/refinement ``SelfRAGEngine``.
It implements Algorithm 1 and Appendix A.3 of arXiv:2310.11511 using a generator
that actually predicts reflection tokens. No lexical critic stands in for the
model. The authors' product of segment scores is accumulated in log space;
non-positive candidates are pruned (signed utility can make scores negative).

Primary references:
https://arxiv.org/abs/2310.11511
https://github.com/AkariAsai/self-rag/blob/main/retrieval_lm/run_long_form_static.py
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from itertools import islice
import math
import re
from typing import Any, Protocol

from ..base import BaseRetriever, Document, _snapshot_document, _validate_top_k
from .reflection import (
    ReflectionScore,
    ReflectionTokenDistribution as Distribution,
    ReflectionTokenGroup as Group,
    ReflectionTokenScorer,
    _normalize,
    _real,
)


class RetrievalAction(str, Enum):
    RETRIEVE = "[Retrieval]"
    NO_RETRIEVAL = "[No Retrieval]"
    CONTINUE = "[Continue to Use Evidence]"


@dataclass(frozen=True)
class ReflectionSegment:
    """A segment and actual distributions observed at its reflection positions.

    ``continuation`` contains generated text AND critique tokens, excluding
    the next retrieval token/EOS. It is replayed verbatim for autoregressive
    conditioning. Missing support/utility tokens remain ``None``; utility is
    usually emitted only at the end of an entire response. ``generated_tokens``
    includes the boundary token and all critique tokens for budget accounting.
    """

    text: str
    continuation: str
    generated_tokens: int
    mean_sequence_logprob: float
    stop_reason: str
    relevance: Distribution | None = None
    support: Distribution | None = None
    utility: Distribution | None = None
    next_retrieval: Distribution | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not isinstance(self.continuation, str):
            raise TypeError("segment text and continuation must be strings")
        _validate_top_k(self.generated_tokens, name="generated_tokens")
        logprob = _real(self.mean_sequence_logprob, name="mean_sequence_logprob")
        if logprob > 0:
            raise ValueError("mean_sequence_logprob must be <= 0")
        if self.stop_reason not in {"eos", "retrieval", "length"}:
            raise ValueError("stop_reason must be eos, retrieval, or length")
        for name, expected in (("relevance", Group.RELEVANCE), ("support", Group.SUPPORT),
                               ("utility", Group.UTILITY), ("next_retrieval", Group.RETRIEVAL)):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, Distribution) or value.group != expected):
                raise ValueError(f"{name} requires the {expected.value} reflection distribution")
        if (self.stop_reason == "retrieval") != (self.next_retrieval is not None):
            raise ValueError("a retrieval boundary requires next_retrieval, and only that boundary may supply it")


class ReflectionDecoder(Protocol):
    """A trained Self-RAG model, not a text-only generator or external critic."""

    def retrieval_distribution(self, prompt: str) -> Distribution:
        """Read next-token probabilities for the complete retrieval group."""

    def decode_segment(self, prompt: str, *, max_new_tokens: int) -> ReflectionSegment:
        """Generate up to a retrieval boundary, EOS, or the explicit token limit."""


@dataclass(frozen=True)
class ScoredSegment:
    number: int
    text: str
    action: RetrievalAction
    retrieval: Distribution
    document: Document | None
    score: ReflectionScore
    missing_reflections: tuple[str, ...]
    generated_tokens: int
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number, "text": self.text, "action": self.action.value,
            "retrieval": self.retrieval.to_dict(),
            "doc_id": self.document.doc_id if self.document is not None else None,
            "score": self.score.to_dict(), "missing_reflections": list(self.missing_reflections),
            "generated_tokens": self.generated_tokens, "stop_reason": self.stop_reason,
        }


@dataclass
class SegmentSearchTrace:
    model_calls: int = 0
    retrieval_calls: int = 0
    generated_tokens: int = 0
    expanded_candidates: int = 0
    pruned_nonpositive: int = 0
    beam_sizes: list[int] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    stop_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_calls": self.model_calls, "retrieval_calls": self.retrieval_calls,
            "generated_tokens": self.generated_tokens, "expanded_candidates": self.expanded_candidates,
            "pruned_nonpositive": self.pruned_nonpositive, "beam_sizes": list(self.beam_sizes),
            "retrieval_queries": list(self.retrieval_queries), "stop_reason": self.stop_reason,
            "score_aggregation": "log_product_positive_segment_scores",
        }


@dataclass
class SegmentedSelfRAGResult:
    query: str
    answer: str
    segments: tuple[ScoredSegment, ...]
    documents: list[Document]
    trace: SegmentSearchTrace
    log_score: float
    status: str
    maturity: str = "experimental_model_conditioned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query, "answer": self.answer, "status": self.status, "maturity": self.maturity,
            "segments": [segment.to_dict() for segment in self.segments], "log_score": self.log_score,
            "documents": [{"doc_id": doc.doc_id, "content": doc.content,
                           "metadata": dict(doc.metadata), "score": doc.score} for doc in self.documents],
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True)
class _Beam:
    prompt: str
    segments: tuple[ScoredSegment, ...] = ()
    evidence: Document | None = None
    relevance: Distribution | None = None
    next_retrieval: Distribution | None = None
    log_score: float = 0.0
    stop_reason: str = ""


def _action(distribution: Distribution, *, has_evidence: bool, threshold: float | None) -> RetrievalAction:
    if not isinstance(distribution, Distribution) or distribution.group not in (Group.RETRIEVAL, Group.INITIAL_RETRIEVAL):
        raise TypeError("decoder must return a retrieval reflection distribution")
    probabilities = distribution.probabilities
    if has_evidence and distribution.group == Group.RETRIEVAL:
        if max(probabilities, key=lambda token: probabilities[token]) == RetrievalAction.CONTINUE.value:
            return RetrievalAction.CONTINUE
    yes_no = (RetrievalAction.RETRIEVE.value, RetrievalAction.NO_RETRIEVAL.value)
    logs = distribution._log_probabilities
    conditional = _normalize(
        {token: logs[token] if logs is not None else probabilities[token] for token in yes_no},
        log_probabilities=logs is not None,
    )
    cutoff = 0.5 if threshold is None else threshold
    return RetrievalAction.RETRIEVE if conditional[yes_no[0]] > cutoff else RetrievalAction.NO_RETRIEVAL


class SegmentedSelfRAGEngine:
    """Passage-conditioned generation, three-state retrieval, and segment beams.

    Each retrieved passage produces one independently conditioned continuation.
    A beam carries its own evidence and complete reflection-token history, so
    Continue reuses exactly that branch's passage without issuing retrieval.
    Retrieval queries use the task and previous segment (Algorithm 1).

    Set ``retrieval_threshold=None`` for model decisions, or a threshold for
    the paper's strict Yes/(Yes+No) policy. With prior evidence, a most-likely
    Continue token takes precedence; the binary threshold applies otherwise.
    Missing critique groups contribute zero, never invented probabilities.
    Citations identify conditioning passages, not an independent factuality
    guarantee. Token limits, candidate budgets and pruned scores are reported.
    """

    top_k: int
    beam_width: int
    max_segments: int
    max_new_tokens: int
    max_model_calls: int
    max_retrieval_calls: int
    max_generated_tokens: int

    def __init__(
        self, retriever: BaseRetriever | None, decoder: ReflectionDecoder, *,
        top_k: int = 5, beam_width: int = 2, max_segments: int = 6,
        max_new_tokens: int = 256, max_model_calls: int = 128,
        max_retrieval_calls: int = 32, max_generated_tokens: int = 16_384,
        retrieval_threshold: float | None = None,
        scorer: ReflectionTokenScorer | None = None, use_sequence_score: bool = True,
    ) -> None:
        if not callable(getattr(decoder, "retrieval_distribution", None)) or not callable(getattr(decoder, "decode_segment", None)):
            raise TypeError("decoder must implement retrieval_distribution and decode_segment")
        if retriever is not None and not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must implement retrieve")
        for name, value in (("top_k", top_k), ("beam_width", beam_width), ("max_segments", max_segments),
                            ("max_new_tokens", max_new_tokens), ("max_model_calls", max_model_calls),
                            ("max_retrieval_calls", max_retrieval_calls), ("max_generated_tokens", max_generated_tokens)):
            setattr(self, name, _validate_top_k(value, name=name))
        if retrieval_threshold is not None:
            retrieval_threshold = _real(retrieval_threshold, name="retrieval_threshold")
            if not 0 <= retrieval_threshold <= 1:
                raise ValueError("retrieval_threshold must be between 0 and 1")
        if not isinstance(use_sequence_score, bool):
            raise TypeError("use_sequence_score must be bool")
        if scorer is not None and not isinstance(scorer, ReflectionTokenScorer):
            raise TypeError("scorer must be ReflectionTokenScorer")
        self.retriever, self.decoder = retriever, decoder
        self.retrieval_threshold, self.use_sequence_score = retrieval_threshold, use_sequence_score
        self.scorer = scorer or ReflectionTokenScorer()

    def ask(self, query: str, *, top_k: int | None = None) -> SegmentedSelfRAGResult:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        query = query.strip()
        if not query:
            raise ValueError("query must not be blank")
        k = self.top_k if top_k is None else _validate_top_k(top_k)
        trace = SegmentSearchTrace()
        beams = [_Beam(f"### Instruction:\n{query}\n\n### Response:\n")]
        completed: list[_Beam] = []
        observed_documents: dict[str, Document] = {}
        budget_stop = ""
        for depth in range(self.max_segments):
            candidates: list[_Beam] = []
            interrupted: list[_Beam] = []
            for beam_index, beam in enumerate(beams):
                if trace.model_calls >= self.max_model_calls:
                    budget_stop = "model_call_limit"
                    interrupted.extend(beams[beam_index:])
                    break
                distribution = beam.next_retrieval
                if distribution is None:
                    trace.model_calls += 1
                    distribution = self.decoder.retrieval_distribution(beam.prompt)
                action = _action(distribution, has_evidence=beam.evidence is not None, threshold=self.retrieval_threshold)
                documents: list[Document | None]
                if action == RetrievalAction.RETRIEVE:
                    if self.retriever is None:
                        raise RuntimeError("model requested retrieval, but no retriever is configured")
                    if trace.retrieval_calls >= self.max_retrieval_calls:
                        budget_stop = "retrieval_call_limit"
                        interrupted.extend(beams[beam_index:])
                        break
                    retrieval_query = query + ("\n" + beam.segments[-1].text if beam.segments else "")
                    trace.retrieval_calls += 1
                    trace.retrieval_queries.append(retrieval_query)
                    documents = [self._document(doc, observed_documents)
                                 for doc in islice(self.retriever.retrieve(retrieval_query, top_k=k), k)]
                    if not documents:
                        completed.append(_Beam(**{**beam.__dict__, "stop_reason": "no_evidence"}))
                        continue
                else:
                    documents = [beam.evidence if action == RetrievalAction.CONTINUE else None]
                for document in documents:
                    remaining = self.max_generated_tokens - trace.generated_tokens
                    if trace.model_calls >= self.max_model_calls or remaining <= 0:
                        budget_stop = "model_call_limit" if trace.model_calls >= self.max_model_calls else "generated_token_limit"
                        interrupted.extend(beams[beam_index:])
                        break
                    evidence = self._document(document, observed_documents) if document is not None else None
                    suffix = action.value
                    if action == RetrievalAction.RETRIEVE:
                        assert evidence is not None
                        title = str(evidence.metadata.get("title", ""))
                        suffix += f"<paragraph>{title}\n{evidence.content}</paragraph>"
                    prompt = beam.prompt + suffix
                    allowance = min(self.max_new_tokens, remaining)
                    trace.model_calls += 1
                    prediction = self.decoder.decode_segment(prompt, max_new_tokens=allowance)
                    if not isinstance(prediction, ReflectionSegment):
                        raise TypeError("decode_segment must return ReflectionSegment")
                    if prediction.generated_tokens > allowance:
                        raise ValueError("decoder exceeded the requested token budget")
                    trace.generated_tokens += prediction.generated_tokens
                    trace.expanded_candidates += 1
                    relevance = prediction.relevance or (beam.relevance if action == RetrievalAction.CONTINUE else None)
                    if action == RetrievalAction.RETRIEVE and relevance is None:
                        raise ValueError("retrieved-passage generation omitted its relevance reflection token")
                    score, missing = self._score(prediction, relevance, evidence is not None)
                    if score.total <= 0:
                        trace.pruned_nonpositive += 1
                        continue
                    text = re.sub(r"\[source:\s*[^\]]*\]", "", prediction.text, flags=re.IGNORECASE).strip()
                    record = ScoredSegment(depth + 1, text, action, distribution, evidence, score, missing,
                                           prediction.generated_tokens, prediction.stop_reason)
                    node = _Beam(
                        prompt + prediction.continuation, (*beam.segments, record),
                        evidence if evidence is not None else beam.evidence,
                        relevance if evidence is not None else beam.relevance,
                        prediction.next_retrieval, beam.log_score + math.log(score.total),
                        prediction.stop_reason if prediction.stop_reason != "retrieval" else "",
                    )
                    (completed if node.stop_reason else candidates).append(node)
                if budget_stop:
                    break
            # Budget interruption leaves these parents unexplored, so their
            # generated prefixes remain valid alternatives to completed paths.
            # Fully expanded/pruned parents must not be resurrected here.
            candidates.extend(node for node in interrupted if node.segments)
            candidates.sort(key=lambda node: node.log_score, reverse=True)
            # Keep the completed frontier bounded too; a single EOS does not
            # terminate unrelated hypotheses that may improve on later steps.
            completed.sort(key=lambda node: node.log_score, reverse=True)
            completed = completed[:self.beam_width]
            beams = candidates[:self.beam_width]
            trace.beam_sizes.append(len(beams))
            if budget_stop:
                break
            if not beams:
                break
        choices = [*completed, *beams]
        best = max(choices, key=lambda node: node.log_score) if choices else _Beam("")
        trace.stop_reason = budget_stop or best.stop_reason or ("segment_limit" if best.segments else "no_positive_candidates")
        unique: dict[str, Document] = {}
        answer: list[str] = []
        for segment in best.segments:
            if not segment.text:
                continue
            citation = ""
            if segment.document is not None:
                doc = segment.document
                assert doc.doc_id is not None
                unique.setdefault(doc.doc_id, _snapshot_document(doc))
                citation = f" [source: {doc.doc_id}]"
            answer.append(segment.text + citation)
        return SegmentedSelfRAGResult(query, " ".join(answer), best.segments, list(unique.values()), trace,
                                      best.log_score, trace.stop_reason)

    def run(self, query: str, **kwargs: Any) -> SegmentedSelfRAGResult:
        return self.ask(query, **kwargs)

    @staticmethod
    def _document(document: Document, observed: dict[str, Document]) -> Document:
        if not isinstance(document, Document) or not isinstance(document.content, str):
            raise TypeError("retriever must yield Document objects with text content")
        result = _snapshot_document(document)
        if result.doc_id is not None and not isinstance(result.doc_id, str):
            raise TypeError("document IDs must be strings or None")
        if result.doc_id is not None:
            result.doc_id = result.doc_id.strip()
        if not result.doc_id:
            result.doc_id = "selfrag-" + hashlib.sha256(result.content.encode()).hexdigest()[:20]
        if not isinstance(result.doc_id, str) or any(character in result.doc_id for character in "\r\n[]"):
            raise ValueError("document IDs must be safe single-line citation identifiers")
        previous = observed.get(result.doc_id)
        if previous is not None and previous.content != result.content:
            raise ValueError("the same document ID refers to different passage contents")
        observed[result.doc_id] = result
        return result

    def _score(self, prediction: ReflectionSegment, relevance: Distribution | None,
               has_evidence: bool) -> tuple[ReflectionScore, tuple[str, ...]]:
        rel = relevance.probabilities["[Relevant]"] if relevance is not None and has_evidence else 0.0
        support = prediction.support
        sup = (support.probabilities["[Fully supported]"] + 0.5 * support.probabilities["[Partially supported]"]
               if support is not None and has_evidence else 0.0)
        utility = prediction.utility
        use = (math.fsum(weight * utility.probabilities[f"[Utility:{index}]"]
                        for index, weight in enumerate((-1., -.5, 0., .5, 1.), 1)) if utility is not None else 0.0)
        sequence = math.exp(prediction.mean_sequence_logprob) if self.use_sequence_score else None
        total = math.fsum((self.scorer.relevance_weight * rel, self.scorer.support_weight * sup,
                          self.scorer.utility_weight * use, sequence or 0.0))
        missing = tuple(name for name, value in (("relevance", relevance), ("support", support), ("utility", utility))
                        if value is None and (has_evidence or name == "utility"))
        models = tuple(dict.fromkeys(value.model_id for value in (relevance, support, utility)
                                     if value is not None and value.model_id is not None))
        return ReflectionScore(rel, sup, use, total, sequence, models), missing
