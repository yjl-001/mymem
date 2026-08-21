"""Pure contracts for the staged E1 experience-memory experiments.

The module intentionally has no Torch or Transformers dependency.  It owns
representative-bank selection, completion-aware lexical queries, immutable
retrieval assignments, and prompt text assembly.  Runtime model execution is
kept in :mod:`memgen.model.e1_runtime`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
import random
import re
from typing import Any, Callable, Mapping, Sequence

from memgen.experience.e1 import MatchedMemoryDeranger, MemoryChoice
from memgen.experience.memory import MemoryRecord, PayloadSanitizer
from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.retrieval import RetrievalQuery, TextAnalyzer


E1A_CATALOG_MANIFEST_SCHEMA = "experience-memory-e1a-catalog-manifest-v1"
E1A_RESULTS_SCHEMA = "experience-memory-e1a-results-v1"
E1A_SUMMARY_SCHEMA = "experience-memory-e1a-summary-v1"
E1B_ASSIGNMENT_SCHEMA = "experience-memory-e1b-assignment-v1"
E1B_MANIFEST_SCHEMA = "experience-memory-e1b-assignment-manifest-v1"
E1B_RESULTS_SCHEMA = "experience-memory-e1b-results-v1"
E1B_SUMMARY_SCHEMA = "experience-memory-e1b-summary-v1"
E1C_RESULTS_SCHEMA = "experience-memory-e1c-results-v1"
E1C_SUMMARY_SCHEMA = "experience-memory-e1c-summary-v1"

E1A_RANDOM_SEEDS = (17, 42, 73)
E1A_CATALOG_TOKEN_BUDGET = 2048
E1B_SHUFFLE_SEED = 42
E1C_MEMORY_SCORE_NORMALIZATION = "log_valid_slots"

_PREANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed\s*\{|\\fbox\s*\{|final\s+answer|answer\s+is)",
    re.IGNORECASE,
)
_MATH_LITERAL_RE = re.compile(
    r"(?:\d+(?:[.,/]\d+)*|[+*/=<>≤≥^]|(?<![A-Za-z])-\s*\d)"
)
_CONTROL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")


def render_experience_catalog(records: Sequence[MemoryRecord]) -> str:
    """Render a fixed multi-record catalog with no task-specific selection."""

    if not records:
        raise ValueError("An experience catalog must contain at least one record")
    entries = "\n\n".join(
        f"Experience {index}:\n{record.sanitized_contrast_payload.strip()}"
        for index, record in enumerate(records, start=1)
    )
    return (
        "General experience guidance:\n"
        "Use the following general problem-solving experiences when relevant. "
        "Ignore any experience that does not apply.\n\n"
        f"{entries}"
    )


def render_single_experience(record: MemoryRecord) -> str:
    """Render one retrieved record using the same neutral instruction style."""

    return (
        "General experience guidance:\n"
        "Use the following general problem-solving experience when relevant. "
        "Ignore it if it does not apply.\n\n"
        f"{record.sanitized_contrast_payload.strip()}"
    )


def build_memory_augmented_messages(
    *,
    question: str,
    memory_text: str | None,
) -> list[dict[str, str]]:
    """Build the frozen GSM8K user message with optional answer-blind memory."""

    from memgen.experience.phase2 import FORMAT_INSTRUCTION

    content = f"{FORMAT_INSTRUCTION}\nQuestion: {question.strip()}\n"
    if memory_text:
        content += f"\n{memory_text.strip()}\n"
    return [{"role": "user", "content": content}]


def _tfidf_vectors(
    records: Sequence[MemoryRecord], analyzer: TextAnalyzer
) -> tuple[dict[str, float], ...]:
    documents = [analyzer.analyze(record.sanitized_retrieval_key) for record in records]
    if any(not document for document in documents):
        raise ValueError("Representative-bank input contains an empty retrieval key")
    frequencies = Counter(term for document in documents for term in set(document))
    count = len(documents)
    vectors: list[dict[str, float]] = []
    for document in documents:
        term_counts = Counter(document)
        vector = {
            term: (1.0 + math.log(frequency))
            * (math.log((1.0 + count) / (1.0 + frequencies[term])) + 1.0)
            for term, frequency in term_counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        vectors.append({term: value / norm for term, value in vector.items()})
    return tuple(vectors)


def _cosine_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    similarity = sum(value * right.get(term, 0.0) for term, value in left.items())
    return max(0.0, min(2.0, 1.0 - similarity))


@dataclass(frozen=True)
class ExperienceCatalog:
    name: str
    method: str
    memory_ids: tuple[str, ...]
    payload_hashes: tuple[str, ...]
    rendered_text: str
    rendered_text_sha256: str
    token_count: int
    token_budget: int
    seed: int | None = None
    objective: float | None = None
    clusters: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.memory_ids:
            raise ValueError("Catalog requires a name and at least one memory")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("Catalog memory IDs must be unique")
        if len(self.payload_hashes) != len(self.memory_ids):
            raise ValueError("Catalog payload metadata is inconsistent")
        if self.token_count <= 0 or self.token_count > self.token_budget:
            raise ValueError("Catalog exceeds its frozen token budget")
        if canonical_json_sha256(self.rendered_text) != self.rendered_text_sha256:
            raise ValueError("Catalog rendered-text hash mismatch")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["memory_ids"] = list(self.memory_ids)
        value["payload_hashes"] = list(self.payload_hashes)
        value["clusters"] = [dict(item) for item in self.clusters]
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperienceCatalog":
        data = dict(value)
        data["memory_ids"] = tuple(str(item) for item in data["memory_ids"])
        data["payload_hashes"] = tuple(str(item) for item in data["payload_hashes"])
        data["clusters"] = tuple(dict(item) for item in data.get("clusters", []))
        return cls(**data)


class ConstrainedKMedoidsCatalogBuilder:
    """Select real MemoryRecord medoids under an exact rendered-token budget."""

    def __init__(
        self,
        *,
        records: Sequence[MemoryRecord],
        token_counter: Callable[[str], int],
        analyzer: TextAnalyzer | None = None,
        token_budget: int = E1A_CATALOG_TOKEN_BUDGET,
        maximum_iterations: int = 20,
    ):
        if not records or token_budget <= 0 or maximum_iterations <= 0:
            raise ValueError("Invalid constrained k-medoids configuration")
        self.records = tuple(sorted(records, key=lambda item: item.memory_id))
        if len({record.memory_id for record in self.records}) != len(self.records):
            raise ValueError("Memory IDs must be unique")
        self.token_counter = token_counter
        self.analyzer = analyzer or TextAnalyzer()
        self.token_budget = token_budget
        self.maximum_iterations = maximum_iterations
        self._rendered_count_cache: dict[tuple[int, ...], int] = {}
        self._objective_cache: dict[tuple[int, ...], float] = {}
        vectors = _tfidf_vectors(self.records, self.analyzer)
        self.distances = tuple(
            tuple(_cosine_distance(left, right) for right in vectors)
            for left in vectors
        )

    def _rendered_count(self, medoids: Sequence[int]) -> int:
        cache_key = tuple(sorted(medoids))
        if cache_key in self._rendered_count_cache:
            return self._rendered_count_cache[cache_key]
        ordered = [self.records[index] for index in sorted(
            medoids, key=lambda item: self.records[item].memory_id
        )]
        value = int(self.token_counter(render_experience_catalog(ordered)))
        self._rendered_count_cache[cache_key] = value
        return value

    def _objective(self, medoids: Sequence[int]) -> float:
        cache_key = tuple(sorted(medoids))
        if cache_key not in self._objective_cache:
            self._objective_cache[cache_key] = sum(
                min(self.distances[index][medoid] for medoid in medoids)
                for index in range(len(self.records))
            )
        return self._objective_cache[cache_key]

    def _initial_medoids(self, count: int) -> tuple[int, ...]:
        first = min(
            range(len(self.records)),
            key=lambda index: (
                sum(self.distances[other][index] for other in range(len(self.records))),
                self.records[index].memory_id,
            ),
        )
        selected = [first]
        while len(selected) < count:
            candidate = min(
                (index for index in range(len(self.records)) if index not in selected),
                key=lambda index: (
                    -min(self.distances[index][medoid] for medoid in selected),
                    self.records[index].memory_id,
                ),
            )
            selected.append(candidate)
        return tuple(sorted(selected))

    def _make_feasible(self, medoids: Sequence[int]) -> tuple[int, ...] | None:
        current = tuple(sorted(medoids))
        if self._rendered_count(current) <= self.token_budget:
            return current
        shortest = tuple(sorted(
            sorted(
                range(len(self.records)),
                key=lambda index: (
                    self.token_counter(
                        self.records[index].sanitized_contrast_payload
                    ),
                    self.records[index].memory_id,
                ),
            )[: len(current)]
        ))
        return shortest if self._rendered_count(shortest) <= self.token_budget else None

    def _refine(self, medoids: Sequence[int]) -> tuple[int, ...]:
        current = tuple(sorted(medoids))
        for _ in range(self.maximum_iterations):
            members: dict[int, list[int]] = {medoid: [] for medoid in current}
            for index in range(len(self.records)):
                owner = min(
                    current,
                    key=lambda medoid: (
                        self.distances[index][medoid],
                        0 if medoid == index else 1,
                        self.records[medoid].memory_id,
                    ),
                )
                members[owner].append(index)
            updated = list(current)
            for position, medoid in enumerate(current):
                ranked = sorted(
                    members[medoid],
                    key=lambda candidate: (
                        sum(
                            self.distances[index][candidate]
                            for index in members[medoid]
                        ),
                        self.records[candidate].memory_id,
                    ),
                )
                for candidate in ranked:
                    proposed = tuple(sorted(
                        updated[:position] + [candidate] + updated[position + 1 :]
                    ))
                    if len(set(proposed)) != len(current):
                        continue
                    if self._rendered_count(proposed) <= self.token_budget:
                        updated[position] = candidate
                        break
            candidate = tuple(sorted(updated))
            if candidate == current:
                break
            if self._objective(candidate) > self._objective(current) + 1e-12:
                break
            current = candidate
        return current

    def build_representative(self) -> ExperienceCatalog:
        feasible: list[tuple[int, float, tuple[int, ...]]] = []
        shortest_records = sorted(
            range(len(self.records)),
            key=lambda index: (
                self.token_counter(self.records[index].sanitized_contrast_payload),
                self.records[index].memory_id,
            ),
        )
        maximum_count = 0
        for count in range(1, len(self.records) + 1):
            if self._rendered_count(shortest_records[:count]) <= self.token_budget:
                maximum_count = count
            else:
                break
        if maximum_count == 0:
            raise ValueError("No MemoryRecord fits the E1-A catalog token budget")
        for count in range(1, maximum_count + 1):
            medoids = self._make_feasible(self._initial_medoids(count))
            if medoids is None:
                continue
            medoids = self._refine(medoids)
            feasible.append((count, self._objective(medoids), medoids))
        if not feasible:
            raise ValueError("No feasible representative catalog was found")
        largest_count = max(item[0] for item in feasible)
        _, objective, medoids = min(
            (item for item in feasible if item[0] == largest_count),
            key=lambda item: (
                item[1],
                tuple(self.records[index].memory_id for index in item[2]),
            ),
        )
        ordered = tuple(sorted(medoids, key=lambda index: self.records[index].memory_id))
        cluster_members: dict[int, list[int]] = {medoid: [] for medoid in ordered}
        for index in range(len(self.records)):
            medoid = min(
                ordered,
                key=lambda candidate: (
                    self.distances[index][candidate],
                    0 if candidate == index else 1,
                    self.records[candidate].memory_id,
                ),
            )
            cluster_members[medoid].append(index)
        clusters = tuple(
            {
                "medoid_memory_id": self.records[medoid].memory_id,
                "size": len(cluster_members[medoid]),
                "mean_distance": sum(
                    self.distances[index][medoid] for index in cluster_members[medoid]
                ) / len(cluster_members[medoid]),
                "max_distance": max(
                    self.distances[index][medoid] for index in cluster_members[medoid]
                ),
                "covered_memory_ids": [
                    self.records[index].memory_id for index in cluster_members[medoid]
                ],
            }
            for medoid in ordered
        )
        return self._catalog(
            name="representative_bank_text",
            method="deterministic-constrained-tfidf-cosine-k-medoids-v1",
            indices=ordered,
            objective=objective,
            clusters=clusters,
        )

    def build_random_control(
        self,
        *,
        representative: ExperienceCatalog,
        seed: int,
    ) -> ExperienceCatalog:
        count = len(representative.memory_ids)
        target_tokens = representative.token_count
        rng = random.Random(seed)
        candidates: list[tuple[int, int, tuple[str, ...], tuple[int, ...]]] = []
        seen: set[tuple[int, ...]] = set()
        indices = list(range(len(self.records)))
        for draw_index in range(5000):
            selected = tuple(sorted(rng.sample(indices, count)))
            if selected in seen:
                continue
            seen.add(selected)
            ids = tuple(self.records[index].memory_id for index in selected)
            if ids == representative.memory_ids:
                continue
            token_count = self._rendered_count(selected)
            if token_count <= self.token_budget:
                candidates.append((
                    abs(token_count - target_tokens), draw_index, ids, selected
                ))
                if abs(token_count - target_tokens) <= 2 and len(candidates) >= 8:
                    break
        if not candidates:
            raise ValueError(f"No random catalog fits the budget for seed {seed}")
        _, _, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
        return self._catalog(
            name=f"random_bank_text_seed{seed}",
            method="seeded-equal-count-nearest-token-budget-random-bank-v1",
            indices=selected,
            seed=seed,
        )

    def _catalog(
        self,
        *,
        name: str,
        method: str,
        indices: Sequence[int],
        seed: int | None = None,
        objective: float | None = None,
        clusters: Sequence[Mapping[str, Any]] = (),
    ) -> ExperienceCatalog:
        records = tuple(self.records[index] for index in indices)
        rendered = render_experience_catalog(records)
        return ExperienceCatalog(
            name=name,
            method=method,
            memory_ids=tuple(record.memory_id for record in records),
            payload_hashes=tuple(record.payload_hash for record in records),
            rendered_text=rendered,
            rendered_text_sha256=canonical_json_sha256(rendered),
            token_count=int(self.token_counter(rendered)),
            token_budget=self.token_budget,
            seed=seed,
            objective=objective,
            clusters=tuple(clusters),
        )


class CompletionAwareRetrievalQueryBuilder:
    """Build ``question + sanitized complete preanswer`` BM25 queries."""

    def __init__(self, *, analyzer: TextAnalyzer | None = None):
        self.analyzer = analyzer or TextAnalyzer()

    @staticmethod
    def sanitize_preanswer(completion: str) -> str:
        marker = _PREANSWER_MARKER_RE.search(completion)
        prefix = completion[: marker.start()] if marker else completion
        prefix = _CONTROL_TOKEN_RE.sub(" ", prefix)
        prefix = _MATH_LITERAL_RE.sub(" ", prefix)
        return PayloadSanitizer.normalize_text(prefix)

    def build(self, *, question: str, completion: str) -> RetrievalQuery:
        normalized_question = PayloadSanitizer.normalize_text(question)
        normalized_preanswer = self.sanitize_preanswer(completion)
        query_text = f"{normalized_question}\n{normalized_preanswer}".strip()
        terms = tuple(self.analyzer.analyze(query_text))
        if not terms:
            raise ValueError("Completion-aware retrieval query has no effective terms")
        query_hash = canonical_json_sha256({
            "schema_version": "experience-memory-completion-retrieval-query-v1",
            "policy": "question-plus-sanitized-complete-preanswer-v1",
            "query_text": query_text,
            "analyzed_terms": list(terms),
        })
        return RetrievalQuery(
            normalized_question=normalized_question,
            normalized_partial_cot=normalized_preanswer,
            query_text=query_text,
            query_hash=query_hash,
            analyzed_terms=terms,
            schema_version="experience-memory-completion-retrieval-query-v1",
        )


@dataclass(frozen=True)
class E1BRetrievalAssignment:
    """One immutable E1-B assignment produced without answer/reward access."""

    sample_id: str
    logical_split: str
    dataset_split: str
    source_index: int
    question_sha256: str
    base_prompt_token_ids_sha256: str
    base_prompt_token_count: int
    preanswer_completion_token_ids: tuple[int, ...]
    preanswer_completion_token_ids_sha256: str
    preanswer_completion_text_sha256: str
    retrieval_query: Mapping[str, Any]
    matched_memory: MemoryChoice
    shuffled_memory: MemoryChoice | None = None
    answer_or_reward_used: bool = False
    schema_version: str = E1B_ASSIGNMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != E1B_ASSIGNMENT_SCHEMA:
            raise ValueError("Unexpected E1-B assignment schema")
        if self.logical_split not in {"calibration-val", "dev-test"}:
            raise ValueError("E1-B cannot use final-test")
        if self.dataset_split != "train" or self.source_index < 0:
            raise ValueError("E1-B requires a frozen train source index")
        if self.answer_or_reward_used:
            raise ValueError("E1-B assignment must be answer-blind")
        if self.base_prompt_token_count <= 0 or not self.preanswer_completion_token_ids:
            raise ValueError("E1-B requires a base prompt and complete preanswer")
        if canonical_json_sha256(list(self.preanswer_completion_token_ids)) != (
            self.preanswer_completion_token_ids_sha256
        ):
            raise ValueError("E1-B preanswer token hash mismatch")
        if not self.retrieval_query.get("query_hash"):
            raise ValueError("E1-B assignment requires a frozen retrieval query")
        if self.shuffled_memory is not None:
            if self.shuffled_memory.memory_id == self.matched_memory.memory_id:
                raise ValueError("E1-B shuffled memory must differ from matched memory")
            if self.shuffled_memory.payload_hash == self.matched_memory.payload_hash:
                raise ValueError("E1-B shuffled payload must differ from matched payload")

    @property
    def assigned(self) -> bool:
        return self.shuffled_memory is not None

    def with_shuffled_memory(self, choice: MemoryChoice) -> "E1BRetrievalAssignment":
        return replace(self, shuffled_memory=choice)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["preanswer_completion_token_ids"] = list(
            self.preanswer_completion_token_ids
        )
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "E1BRetrievalAssignment":
        data = dict(value)
        data["preanswer_completion_token_ids"] = tuple(
            int(item) for item in data["preanswer_completion_token_ids"]
        )
        data["matched_memory"] = MemoryChoice.from_dict(data["matched_memory"])
        if data.get("shuffled_memory") is not None:
            data["shuffled_memory"] = MemoryChoice.from_dict(data["shuffled_memory"])
        return cls(**data)


class E1BRetrievalDeranger:
    """Reuse the proven multiset-preserving derangement for E1-B assignments."""

    def __init__(self, *, seed: int = E1B_SHUFFLE_SEED):
        self.seed = seed

    def assign(
        self, assignments: Sequence[E1BRetrievalAssignment]
    ) -> tuple[tuple[E1BRetrievalAssignment, ...], dict[str, Any]]:
        # The E1-v1 deranger only relies on sample_id, matched_memory,
        # with_shuffled_memory, and assigned; E1-B deliberately implements the
        # same small protocol so the causal control cannot drift.
        output, report = MatchedMemoryDeranger(seed=self.seed).assign(
            assignments  # type: ignore[arg-type]
        )
        return tuple(output), report  # type: ignore[return-value]
