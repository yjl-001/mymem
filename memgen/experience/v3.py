"""Pure, versioned contracts for the V3 online experience-memory system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from memgen.experience.e1 import MemoryChoice


V3_SYSTEM_PROFILE_SCHEMA = "experience-memory-system-profile-v3"
V3_RETRIEVAL_DECISION_SCHEMA = "embedding-memory-retrieval-decision-v1"
V3_OFFLINE_REPORT_SCHEMA = "experience-memory-v3-offline-report-v1"
V3_GENERATION_RESULT_SCHEMA = "experience-memory-v3-generation-result-v1"


@dataclass(frozen=True)
class ExperienceMemoryV3Profile:
    """Frozen V3 behavior, excluding thresholds loaded from the risk artifact."""

    layer_number: int = 24
    query_context: str = "question_plus_full_partial_cot"
    query_encoder_state: str = "pure_prefix_reencode_side_kv_disabled"
    query_pooling: str = "last_valid_token"
    query_normalization: str = "l2"
    retrieval_method: str = "exact_cosine"
    retrieval_abstention_policy: str = "disabled"
    retrieval_top_k: int = 2
    selected_memory_count: int = 1
    max_retrieval_attempts: int = 3
    gate_policy: str = "entropy_hysteresis_rearm"
    risk_role: str = "diagnostic_only"
    boundary_policy: str = "pre_answer_comma_period_newline"
    rearm_policy: str = "low_boundary_rearms_next_future_high_boundary"
    replacement_policy: str = "replace_current_memory"
    duplicate_policy: str = "consume_attempt_keep_current_memory"
    abstain_policy: str = "consume_attempt_keep_current_memory"
    injection_policy: str = "persistent_until_replace_or_eos"
    memory_score_normalization: str = "log_valid_slots"
    memory_score_bias: float = math.log(10.0)
    attention_backend: str = "sdpa"
    schema_version: str = V3_SYSTEM_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V3_SYSTEM_PROFILE_SCHEMA:
            raise ValueError("Unexpected V3 system profile schema")
        if self.layer_number != 24:
            raise ValueError("The current V3 system is frozen to layer 24")
        expected = {
            "query_context": "question_plus_full_partial_cot",
            "query_encoder_state": "pure_prefix_reencode_side_kv_disabled",
            "query_pooling": "last_valid_token",
            "query_normalization": "l2",
            "retrieval_method": "exact_cosine",
            "retrieval_abstention_policy": "disabled",
            "gate_policy": "entropy_hysteresis_rearm",
            "risk_role": "diagnostic_only",
            "boundary_policy": "pre_answer_comma_period_newline",
            "rearm_policy": "low_boundary_rearms_next_future_high_boundary",
            "replacement_policy": "replace_current_memory",
            "duplicate_policy": "consume_attempt_keep_current_memory",
            "abstain_policy": "consume_attempt_keep_current_memory",
            "injection_policy": "persistent_until_replace_or_eos",
            "memory_score_normalization": "log_valid_slots",
            "attention_backend": "sdpa",
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"Unexpected V3 {field_name}")
        if self.retrieval_top_k < 2:
            raise ValueError("V3 retrieval must retain top-2 diagnostics")
        if self.selected_memory_count != 1:
            raise ValueError("V3 supports exactly one active memory")
        if self.max_retrieval_attempts != 3:
            raise ValueError("The current V3 attempt budget is frozen to three")
        if not math.isfinite(self.memory_score_bias):
            raise ValueError("V3 memory score bias must be finite")

    @property
    def memory_odds_multiplier(self) -> float:
        return math.exp(self.memory_score_bias)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "memory_odds_multiplier": self.memory_odds_multiplier,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperienceMemoryV3Profile":
        if value.get("schema_version") != V3_SYSTEM_PROFILE_SCHEMA:
            raise ValueError("Missing or unexpected V3 system profile schema")
        data = dict(value)
        multiplier = data.pop("memory_odds_multiplier", None)
        profile = cls(**data)
        if multiplier is not None and not math.isclose(
            float(multiplier), profile.memory_odds_multiplier, rel_tol=1e-12
        ):
            raise ValueError("V3 memory odds multiplier drifted")
        return profile


@dataclass(frozen=True)
class EmbeddingRetrievalDecision:
    """Auditable exact-cosine result for one full-prefix query embedding."""

    status: str
    query: Mapping[str, Any]
    hits: tuple[Mapping[str, Any], ...]
    matched_memory: MemoryChoice | None
    schema_version: str = V3_RETRIEVAL_DECISION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V3_RETRIEVAL_DECISION_SCHEMA:
            raise ValueError("Unexpected V3 retrieval decision schema")
        if self.status not in {"selected", "empty_bank"}:
            raise ValueError(f"Unexpected V3 retrieval status: {self.status}")
        if self.status == "selected":
            if self.matched_memory is None or not self.hits:
                raise ValueError("Selected V3 retrieval requires hits and memory")
            if self.hits[0].get("memory_id") != self.matched_memory.memory_id:
                raise ValueError("Top embedding hit and selected memory differ")
        elif self.matched_memory is not None or self.hits:
            raise ValueError("Empty-bank retrieval cannot select a memory")

    @property
    def selected(self) -> bool:
        return self.matched_memory is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "query": dict(self.query),
            "hits": [dict(hit) for hit in self.hits],
            "matched_memory": (
                self.matched_memory.to_dict()
                if self.matched_memory is not None
                else None
            ),
        }
