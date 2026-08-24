"""Pure, auditable contracts for the frozen full-system causal experiment.

This module deliberately has no Torch dependency.  It owns the immutable
assignment records, deterministic shuffled-memory control, and paired binary
effect summaries.  Model execution lives in :mod:`memgen.model.e1_runtime`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import math
import random
from typing import Any, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256


E1_ASSIGNMENT_SCHEMA = "experience-memory-e1-assignment-v2"
E1_MANIFEST_SCHEMA = "experience-memory-e1-assignment-manifest-v2"
E1_RESULTS_SCHEMA = "experience-memory-e1-results-v2"
E1_SUMMARY_SCHEMA = "experience-memory-e1-summary-v2"
E1_CONDITIONS = (
    "vanilla",
    "gate_observation_only",
    "matched_persistent_memory",
    "shuffled_persistent_memory",
)


@dataclass(frozen=True)
class MemoryChoice:
    """One text-retrieval result joined to its compiled side-KV metadata."""

    memory_id: str
    payload_hash: str
    token_count: int
    kv_valid_slot_count: int
    retrieval_score: float | None = None
    retrieval_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.memory_id or not self.payload_hash:
            raise ValueError("MemoryChoice requires memory and payload IDs")
        if self.token_count <= 0 or self.kv_valid_slot_count <= 0:
            raise ValueError("MemoryChoice token and slot counts must be positive")
        if self.token_count != self.kv_valid_slot_count:
            raise ValueError("E1-v1 requires payload tokens and valid KV slots to match")
        if self.retrieval_rank is not None and self.retrieval_rank <= 0:
            raise ValueError("retrieval_rank must be positive")
        if self.retrieval_score is not None and (
            not math.isfinite(self.retrieval_score) or self.retrieval_score <= 0
        ):
            raise ValueError("retrieval_score must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MemoryChoice":
        return cls(**dict(value))


@dataclass(frozen=True)
class GateObservation:
    """Frozen observation at the first joint entropy-and-risk trigger."""

    generated_boundary_index: int
    boundary_token_id: int
    entropy: float
    entropy_threshold: float
    persistence_risk_score: float
    persistence_risk_threshold: float

    def __post_init__(self) -> None:
        if self.generated_boundary_index < 0 or self.boundary_token_id < 0:
            raise ValueError("Gate boundary indices must be non-negative")
        values = (
            self.entropy,
            self.entropy_threshold,
            self.persistence_risk_score,
            self.persistence_risk_threshold,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Gate observations must be finite")
        if self.entropy < self.entropy_threshold:
            raise ValueError("Frozen trigger does not pass the entropy threshold")
        if self.persistence_risk_score <= self.persistence_risk_threshold:
            raise ValueError("Frozen trigger does not pass the risk threshold")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GateObservation":
        return cls(**dict(value))


@dataclass(frozen=True)
class E1Assignment:
    """One answer-blind sample assignment shared by every E1 condition."""

    sample_id: str
    logical_split: str
    dataset_split: str
    source_index: int
    question_sha256: str
    prompt_token_count: int
    prompt_token_ids_sha256: str
    observation_completion_token_ids: tuple[int, ...]
    observation_completion_token_ids_sha256: str
    gate_observation: GateObservation | None
    prefix_token_ids: tuple[int, ...]
    prefix_token_ids_sha256: str | None
    retrieval_query: Mapping[str, Any] | None
    matched_memory: MemoryChoice | None
    shuffled_memory: MemoryChoice | None
    abstain_reason: str | None
    answer_or_reward_used: bool = False
    schema_version: str = E1_ASSIGNMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != E1_ASSIGNMENT_SCHEMA:
            raise ValueError("Unexpected E1 assignment schema")
        if not self.sample_id or not self.question_sha256:
            raise ValueError("E1 assignment requires sample and question IDs")
        if self.logical_split not in {"calibration-val", "dev-test"}:
            raise ValueError("E1 development assignments cannot use final-test")
        if self.dataset_split != "train" or self.source_index < 0:
            raise ValueError("E1 development assignments require a train source index")
        if self.prompt_token_count <= 0:
            raise ValueError("prompt_token_count must be positive")
        if self.answer_or_reward_used:
            raise ValueError("E1 assignment construction must be answer-blind")
        if not self.observation_completion_token_ids:
            raise ValueError("Observation-only completion must not be empty")
        completion_hash = canonical_json_sha256(
            list(self.observation_completion_token_ids)
        )
        if completion_hash != self.observation_completion_token_ids_sha256:
            raise ValueError("Observation-only completion hash mismatch")

        triggered = self.gate_observation is not None
        if triggered:
            if not self.prefix_token_ids or self.prefix_token_ids_sha256 is None:
                raise ValueError("Triggered assignment requires an immutable prefix")
            if canonical_json_sha256(list(self.prefix_token_ids)) != self.prefix_token_ids_sha256:
                raise ValueError("Triggered assignment prefix hash mismatch")
            expected_prefix_length = (
                self.prompt_token_count
                + self.gate_observation.generated_boundary_index
                + 1
            )
            if len(self.prefix_token_ids) != expected_prefix_length:
                raise ValueError("Triggered assignment prefix length is inconsistent")
            if self.prefix_token_ids[-1] != self.gate_observation.boundary_token_id:
                raise ValueError("Triggered assignment boundary token mismatch")
            if self.retrieval_query is None:
                raise ValueError("Triggered assignment requires a retrieval query")
        elif self.prefix_token_ids or self.prefix_token_ids_sha256 is not None:
            raise ValueError("Untriggered assignment must not carry a treatment prefix")

        if self.matched_memory is None:
            if self.shuffled_memory is not None:
                raise ValueError("Shuffled memory requires a matched-memory assignment")
            if self.abstain_reason is None:
                raise ValueError("Unassigned sample requires an abstain reason")
        else:
            if not triggered:
                raise ValueError("Matched memory requires a frozen trigger")
            if self.abstain_reason is not None:
                raise ValueError("Assigned sample cannot also abstain")
            if self.shuffled_memory is not None:
                self._validate_shuffle()

    @property
    def triggered(self) -> bool:
        return self.gate_observation is not None

    @property
    def assigned(self) -> bool:
        return self.matched_memory is not None and self.shuffled_memory is not None

    def with_shuffled_memory(self, choice: MemoryChoice) -> "E1Assignment":
        if self.matched_memory is None:
            raise ValueError("Cannot shuffle an assignment without matched memory")
        updated = replace(self, shuffled_memory=choice)
        updated._validate_shuffle()
        return updated

    def _validate_shuffle(self) -> None:
        assert self.matched_memory is not None
        assert self.shuffled_memory is not None
        if self.matched_memory.memory_id == self.shuffled_memory.memory_id:
            raise ValueError("Shuffled memory must differ from matched memory")
        if self.matched_memory.payload_hash == self.shuffled_memory.payload_hash:
            raise ValueError("Shuffled payload must differ from matched payload")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["observation_completion_token_ids"] = list(
            self.observation_completion_token_ids
        )
        value["prefix_token_ids"] = list(self.prefix_token_ids)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "E1Assignment":
        data = dict(value)
        data["observation_completion_token_ids"] = tuple(
            int(item) for item in data["observation_completion_token_ids"]
        )
        data["prefix_token_ids"] = tuple(
            int(item) for item in data.get("prefix_token_ids", [])
        )
        if data.get("gate_observation") is not None:
            data["gate_observation"] = GateObservation.from_dict(
                data["gate_observation"]
            )
        for field_name in ("matched_memory", "shuffled_memory"):
            if data.get(field_name) is not None:
                data[field_name] = MemoryChoice.from_dict(data[field_name])
        return cls(**data)


class MatchedMemoryDeranger:
    """Deterministically permute matched IDs while preserving their multiset.

    The assignments are grouped by matched memory ID and donors are rotated.
    Every valid rotation preserves the exact global memory/slot distribution;
    the chosen rotation minimizes the paired absolute KV-slot difference.
    """

    def __init__(self, *, seed: int = 42):
        self.seed = seed

    def assign(
        self, assignments: Sequence[E1Assignment]
    ) -> tuple[tuple[E1Assignment, ...], dict[str, Any]]:
        assigned = [item for item in assignments if item.matched_memory is not None]
        if len(assigned) < 2:
            raise ValueError("At least two matched assignments are required to shuffle")
        counts = Counter(item.matched_memory.memory_id for item in assigned)
        if max(counts.values()) * 2 > len(assigned):
            raise ValueError(
                "Matched retrieval is too concentrated for a no-self derangement: "
                f"counts={dict(sorted(counts.items()))}"
            )

        ordered = sorted(
            assigned,
            key=lambda item: (
                item.matched_memory.memory_id,
                item.matched_memory.kv_valid_slot_count,
                item.sample_id,
            ),
        )
        valid: list[tuple[int, int, float]] = []
        for shift in range(1, len(ordered)):
            donors = ordered[shift:] + ordered[:shift]
            if any(
                target.matched_memory.memory_id == donor.matched_memory.memory_id
                for target, donor in zip(ordered, donors)
            ):
                continue
            absolute_differences = [
                abs(
                    target.matched_memory.kv_valid_slot_count
                    - donor.matched_memory.kv_valid_slot_count
                )
                for target, donor in zip(ordered, donors)
            ]
            tie_break = random.Random(self.seed + shift).random()
            valid.append((sum(absolute_differences), shift, tie_break))
        if not valid:
            raise ValueError("No deterministic matched-memory derangement exists")
        _, selected_shift, _ = min(valid, key=lambda item: (item[0], item[2], item[1]))
        donors = ordered[selected_shift:] + ordered[:selected_shift]
        donor_by_sample_id = {
            target.sample_id: replace(
                donor.matched_memory,
                retrieval_score=None,
                retrieval_rank=None,
            )
            for target, donor in zip(ordered, donors)
        }
        output = tuple(
            item.with_shuffled_memory(donor_by_sample_id[item.sample_id])
            if item.sample_id in donor_by_sample_id
            else item
            for item in assignments
        )
        paired_differences = [
            abs(
                item.matched_memory.kv_valid_slot_count
                - item.shuffled_memory.kv_valid_slot_count
            )
            for item in output
            if item.assigned
        ]
        matched_multiset = Counter(
            item.matched_memory.memory_id for item in output if item.assigned
        )
        shuffled_multiset = Counter(
            item.shuffled_memory.memory_id for item in output if item.assigned
        )
        if matched_multiset != shuffled_multiset:
            raise RuntimeError("Shuffled memory does not preserve the matched multiset")
        report = {
            "method": "minimum-slot-cost-cyclic-derangement-v1",
            "seed": self.seed,
            "selected_shift": selected_shift,
            "assigned_count": len(paired_differences),
            "memory_id_counts": dict(sorted(matched_multiset.items())),
            "matched_memory_multiset_sha256": canonical_json_sha256(
                sorted(matched_multiset.elements())
            ),
            "shuffled_memory_multiset_sha256": canonical_json_sha256(
                sorted(shuffled_multiset.elements())
            ),
            "exact_slot_match_count": sum(value == 0 for value in paired_differences),
            "mean_absolute_slot_difference": sum(paired_differences)
            / len(paired_differences),
            "max_absolute_slot_difference": max(paired_differences),
        }
        return output, report


def quantile(values: Sequence[float], probability: float) -> float:
    if not values or not 0 <= probability <= 1:
        raise ValueError("quantile expects non-empty values and p in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_binary_effect(
    treatment: Mapping[str, bool | int | float],
    control: Mapping[str, bool | int | float],
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    """Summarize a paired binary treatment-minus-control effect."""

    if resamples <= 0:
        raise ValueError("resamples must be positive")
    shared = sorted(set(treatment) & set(control))
    treatment_only = sorted(set(treatment) - set(control))
    control_only = sorted(set(control) - set(treatment))
    pairs = [(bool(treatment[key]), bool(control[key])) for key in shared]
    differences = [float(left) - float(right) for left, right in pairs]
    treatment_wins = sum(left and not right for left, right in pairs)
    control_wins = sum(right and not left for left, right in pairs)
    discordant = treatment_wins + control_wins
    exact_p = 1.0
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(treatment_wins, control_wins) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2.0 * tail)
    interval = None
    if differences:
        generator = random.Random(seed)
        size = len(differences)
        means = [
            sum(differences[generator.randrange(size)] for _ in range(size)) / size
            for _ in range(resamples)
        ]
        interval = [quantile(means, 0.025), quantile(means, 0.975)]
    return {
        "paired_sample_count": len(shared),
        "treatment_only_count": len(treatment_only),
        "control_only_count": len(control_only),
        "treatment_accuracy": (
            sum(float(left) for left, _ in pairs) / len(pairs) if pairs else None
        ),
        "control_accuracy": (
            sum(float(right) for _, right in pairs) / len(pairs) if pairs else None
        ),
        "mean_treatment_minus_control": (
            sum(differences) / len(differences) if differences else None
        ),
        "bootstrap_95_ci": interval,
        "treatment_correct_control_wrong": treatment_wins,
        "treatment_wrong_control_correct": control_wins,
        "mcnemar_exact_two_sided_p": exact_p,
    }
