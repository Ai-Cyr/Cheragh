"""Reusable tokenization utilities for retrieval quality.

The default tokenizer is intentionally dependency-free but handles common RAG
corpora better than a naive ``str.split``: unicode words, apostrophes, hyphenated
terms, accent folding, optional stop words, and optional word n-grams.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import unicodedata
from typing import Iterable, Sequence

_TOKEN_RE = re.compile(r"[\w]+(?:[\-'’][\w]+)*", flags=re.UNICODE)

DEFAULT_STOPWORDS_FR = frozenset(
    {
        "a",
        "afin",
        "ai",
        "ainsi",
        "alors",
        "au",
        "aucun",
        "aussi",
        "aux",
        "avec",
        "ce",
        "ces",
        "cet",
        "cette",
        "comme",
        "dans",
        "de",
        "des",
        "du",
        "elle",
        "en",
        "et",
        "etre",
        "fait",
        "il",
        "ils",
        "je",
        "la",
        "le",
        "les",
        "leur",
        "lui",
        "mais",
        "ne",
        "nous",
        "ou",
        "par",
        "pas",
        "pour",
        "que",
        "qui",
        "se",
        "ses",
        "son",
        "sur",
        "tu",
        "un",
        "une",
        "vous",
    }
)

DEFAULT_STOPWORDS_EN = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)

DEFAULT_STOPWORDS = DEFAULT_STOPWORDS_FR | DEFAULT_STOPWORDS_EN


@dataclass(frozen=True)
class RetrievalTokenizer:
    """Dependency-free tokenizer tuned for hybrid sparse+dense retrieval.

    Parameters
    ----------
    lowercase:
        Lowercase text before tokenization.
    strip_accents:
        Fold accents so ``sécurité`` and ``securite`` match.
    keep_hyphenated:
        Keep the full hyphenated token and also index its parts. For example,
        ``read-only`` becomes ``read-only``, ``read`` and ``only``.
    stopwords:
        Optional stopword set after normalization. Pass an empty set to disable.
    ngram_range:
        Inclusive word n-gram range. ``(1, 2)`` adds bigrams, useful for
        technical expressions such as ``vector store`` or ``read only``.
    min_token_length:
        Drop very short tokens, except digits.
    """

    lowercase: bool = True
    strip_accents: bool = True
    keep_hyphenated: bool = True
    stopwords: frozenset[str] = field(default_factory=lambda: DEFAULT_STOPWORDS)
    ngram_range: tuple[int, int] = (1, 2)
    min_token_length: int = 2

    def __post_init__(self) -> None:
        if self.ngram_range[0] <= 0 or self.ngram_range[0] > self.ngram_range[1]:
            raise ValueError("ngram_range must be like (1, 2)")
        if self.min_token_length < 1:
            raise ValueError("min_token_length must be >= 1")

    def normalize(self, text: str) -> str:
        value = text.lower() if self.lowercase else text
        # Normalize apostrophes before regex so French elisions are stable.
        value = value.replace("’", "'")
        if self.strip_accents:
            value = unicodedata.normalize("NFKD", value)
            value = "".join(ch for ch in value if not unicodedata.combining(ch))
        return value

    def tokenize(self, text: str) -> list[str]:
        normalized = self.normalize(text)
        tokens: list[str] = []
        for match in _TOKEN_RE.finditer(normalized):
            raw = match.group(0).strip("_-'’")
            if not raw:
                continue
            candidates = [raw]
            if self.keep_hyphenated and ("-" in raw or "'" in raw):
                candidates.extend(part for part in re.split(r"[-']+", raw) if part)
            for token in candidates:
                token = token.strip("_")
                if not token:
                    continue
                if len(token) < self.min_token_length and not token.isdigit():
                    continue
                if token in self.stopwords:
                    continue
                tokens.append(token)

        min_n, max_n = self.ngram_range
        if max_n > 1 and tokens:
            base = list(tokens)
            for n in range(max(2, min_n), max_n + 1):
                if len(base) < n:
                    continue
                tokens.extend(" ".join(base[i : i + n]) for i in range(0, len(base) - n + 1))
        return tokens


def tokenize(text: str, *, tokenizer: RetrievalTokenizer | None = None) -> list[str]:
    """Tokenize text with the default retrieval tokenizer."""

    return (tokenizer or RetrievalTokenizer()).tokenize(text)


def normalize_token(text: str, *, tokenizer: RetrievalTokenizer | None = None) -> str:
    """Normalize a single token or short phrase for comparisons/tests."""

    return (tokenizer or RetrievalTokenizer()).normalize(text).strip()


def ngrams(tokens: Sequence[str], n: int) -> Iterable[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return zip(*(tokens[i:] for i in range(n)))
