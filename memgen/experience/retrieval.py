"""Deterministic, answer-blind retrieval primitives for experience memory.

E0 builds and serializes the lexical index; E1 will consume the same objects
online.  Keeping this module independent of Torch makes index/query behavior
cheap to audit and stable across model-runtime changes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

from memgen.experience.memory import (
    MemoryRecord,
    MemoryRecordRejected,
    PayloadSanitizer,
    TokenizerLike,
)
from memgen.experience.phase1 import canonical_json_sha256


BM25_INDEX_SCHEMA = "experience-memory-bm25-index-v1"
RETRIEVAL_QUERY_SCHEMA = "experience-memory-retrieval-query-v1"

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)
_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)",
    re.IGNORECASE,
)

DEFAULT_BM25_STOPWORDS = frozenset(
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
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "when",
        "with",
        "facing",
        "prefer",
        "avoid",
        "question",
        "answer",
    }
)


@dataclass(frozen=True)
class TextAnalyzerConfig:
    lowercase: bool = True
    remove_numeric_literals: bool = True
    stopwords: tuple[str, ...] = tuple(sorted(DEFAULT_BM25_STOPWORDS))


class TextAnalyzer:
    """Versioned lexical analyzer shared by documents and online queries."""

    def __init__(self, config: TextAnalyzerConfig | None = None):
        self.config = config or TextAnalyzerConfig()
        self._stopwords = frozenset(self.config.stopwords)

    def analyze(self, value: str) -> list[str]:
        text = unicodedata.normalize("NFKC", value)
        if self.config.remove_numeric_literals:
            text = re.sub(r"\d+(?:[.,/]\d+)*", " ", text)
        if self.config.lowercase:
            text = text.casefold()
        return [
            match.group(0)
            for match in _WORD_RE.finditer(text)
            if match.group(0) not in self._stopwords
        ]


@dataclass(frozen=True)
class BM25Config:
    k1: float = 1.2
    b: float = 0.75
    minimum_score_exclusive: float = 0.0

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("BM25 k1 must be positive")
        if not 0 <= self.b <= 1:
            raise ValueError("BM25 b must be in [0, 1]")
        if self.minimum_score_exclusive < 0:
            raise ValueError("BM25 minimum score must be non-negative")


@dataclass(frozen=True)
class BM25Hit:
    memory_id: str
    score: float
    rank: int
    payload_hash: str
    token_count: int


class BM25MemoryIndex:
    """Small deterministic BM25 implementation with a serializable index."""

    def __init__(
        self,
        *,
        records: Sequence[MemoryRecord],
        analyzer: TextAnalyzer | None = None,
        config: BM25Config | None = None,
    ):
        if not records:
            raise ValueError("BM25 index requires at least one memory record")
        self.records = tuple(records)
        self.analyzer = analyzer or TextAnalyzer()
        self.config = config or BM25Config()
        self.document_tokens = tuple(
            tuple(self.analyzer.analyze(record.sanitized_retrieval_key))
            for record in self.records
        )
        if any(not tokens for tokens in self.document_tokens):
            raise ValueError("BM25 index contains an empty sanitized retrieval key")
        self.term_frequencies = tuple(Counter(tokens) for tokens in self.document_tokens)
        self.document_frequencies = Counter(
            term for tokens in self.document_tokens for term in set(tokens)
        )
        self.average_document_length = sum(map(len, self.document_tokens)) / len(
            self.document_tokens
        )

    def search(self, query: str, *, top_k: int = 2) -> list[BM25Hit]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query_terms = self.analyzer.analyze(query)
        if not query_terms:
            return []
        scores = [self._score(query_terms, index) for index in range(len(self.records))]
        ranked = sorted(
            (
                index
                for index in range(len(self.records))
                if scores[index] > self.config.minimum_score_exclusive
            ),
            key=lambda index: (-scores[index], self.records[index].memory_id),
        )
        return [
            BM25Hit(
                memory_id=self.records[index].memory_id,
                score=scores[index],
                rank=rank,
                payload_hash=self.records[index].payload_hash,
                token_count=self.records[index].token_count,
            )
            for rank, index in enumerate(ranked[:top_k], start=1)
        ]

    def _score(self, query_terms: Sequence[str], document_index: int) -> float:
        document_length = len(self.document_tokens[document_index])
        frequencies = self.term_frequencies[document_index]
        score = 0.0
        for term in set(query_terms):
            frequency = frequencies.get(term, 0)
            if frequency == 0:
                continue
            document_frequency = self.document_frequencies[term]
            inverse_document_frequency = math.log(
                1
                + (len(self.records) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequency + self.config.k1 * (
                1
                - self.config.b
                + self.config.b * document_length / self.average_document_length
            )
            score += inverse_document_frequency * (
                frequency * (self.config.k1 + 1) / denominator
            )
        return score

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": BM25_INDEX_SCHEMA,
            "config": asdict(self.config),
            "analyzer": asdict(self.analyzer.config),
            "average_document_length": self.average_document_length,
            "document_frequencies": dict(sorted(self.document_frequencies.items())),
            "documents": [
                {
                    "memory_id": record.memory_id,
                    "payload_hash": record.payload_hash,
                    "payload_token_count": record.token_count,
                    "retrieval_tokens": list(tokens),
                    "retrieval_key_sha256": canonical_json_sha256(
                        record.sanitized_retrieval_key
                    ),
                }
                for record, tokens in zip(self.records, self.document_tokens)
            ],
        }
        payload["index_sha256"] = canonical_json_sha256(payload)
        return payload

    @classmethod
    def from_dict(
        cls,
        *,
        records: Sequence[MemoryRecord],
        value: Mapping[str, Any],
    ) -> "BM25MemoryIndex":
        """Rebuild and verify an index artifact against its MemoryRecords."""

        if value.get("schema_version") != BM25_INDEX_SCHEMA:
            raise ValueError("Unexpected BM25 index schema_version")
        expected_hash = value.get("index_sha256")
        unhashed = {key: item for key, item in value.items() if key != "index_sha256"}
        if expected_hash != canonical_json_sha256(unhashed):
            raise ValueError("BM25 index artifact hash mismatch")
        analyzer_value = dict(value.get("analyzer") or {})
        if "stopwords" in analyzer_value:
            analyzer_value["stopwords"] = tuple(analyzer_value["stopwords"])
        index = cls(
            records=records,
            analyzer=TextAnalyzer(TextAnalyzerConfig(**analyzer_value)),
            config=BM25Config(**dict(value.get("config") or {})),
        )
        if index.to_dict()["index_sha256"] != expected_hash:
            raise ValueError("BM25 index does not match the supplied MemoryRecords")
        return index


@dataclass(frozen=True)
class RetrievalQueryConfig:
    partial_cot_window_tokens: int = 96

    def __post_init__(self) -> None:
        if self.partial_cot_window_tokens <= 0:
            raise ValueError("partial_cot_window_tokens must be positive")


@dataclass(frozen=True)
class RetrievalQuery:
    normalized_question: str
    normalized_partial_cot: str
    query_text: str
    query_hash: str
    analyzed_terms: tuple[str, ...]
    schema_version: str = RETRIEVAL_QUERY_SCHEMA

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "query_hash": self.query_hash,
            "analyzed_terms": list(self.analyzed_terms),
        }
        if include_text:
            value.update(
                {
                    "normalized_question": self.normalized_question,
                    "normalized_partial_cot": self.normalized_partial_cot,
                    "query_text": self.query_text,
                }
            )
        return value


class RetrievalQueryBuilder:
    """Build the frozen ``question + partial CoT`` lexical query."""

    def __init__(
        self,
        *,
        tokenizer: TokenizerLike,
        analyzer: TextAnalyzer | None = None,
        config: RetrievalQueryConfig | None = None,
    ):
        self.tokenizer = tokenizer
        self.analyzer = analyzer or TextAnalyzer()
        self.config = config or RetrievalQueryConfig()

    def build(
        self,
        *,
        question: str,
        partial_cot_token_ids: Sequence[int],
    ) -> RetrievalQuery:
        window = partial_cot_token_ids[-self.config.partial_cot_window_tokens :]
        partial = self.tokenizer.decode(window, skip_special_tokens=True)
        if _ANSWER_MARKER_RE.search(partial):
            raise MemoryRecordRejected(["retrieval_query_contains_final_answer_marker"])
        normalized_question = PayloadSanitizer.normalize_text(question)
        normalized_partial = PayloadSanitizer.normalize_text(partial)
        query_text = f"{normalized_question}\n{normalized_partial}".strip()
        terms = tuple(self.analyzer.analyze(query_text))
        if not terms:
            raise MemoryRecordRejected(["retrieval_query_has_no_effective_terms"])
        query_hash = canonical_json_sha256(
            {
                "schema_version": RETRIEVAL_QUERY_SCHEMA,
                "config": asdict(self.config),
                "analyzer": asdict(self.analyzer.config),
                "query_text": query_text,
            }
        )
        return RetrievalQuery(
            normalized_question=normalized_question,
            normalized_partial_cot=normalized_partial,
            query_text=query_text,
            query_hash=query_hash,
            analyzed_terms=terms,
        )
