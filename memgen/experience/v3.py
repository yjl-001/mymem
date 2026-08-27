"""Pure, versioned contracts for the V3 online experience-memory system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

from memgen.experience.e1 import MemoryChoice


V3_SYSTEM_PROFILE_SCHEMA = "experience-memory-system-profile-v3"
V34_SYSTEM_PROFILE_SCHEMA = "experience-memory-system-profile-v3.4"
V3_RETRIEVAL_DECISION_SCHEMA = "embedding-memory-retrieval-decision-v1"
V3_OFFLINE_REPORT_SCHEMA = "experience-memory-v3-offline-report-v1"
V3_GENERATION_RESULT_SCHEMA = "experience-memory-v3-generation-result-v1"
V34_GENERATION_RESULT_SCHEMA = "experience-memory-v3.4-generation-result-v1"
V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE = "none"
V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED = (
    "key_bank_centroid_center_l2"
)
V3_RETRIEVAL_EMBEDDING_TRANSFORMS = frozenset({
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
})
V3_QUERY_POOLING_BOUNDARY_LAST = "last_valid_token"
V3_QUERY_POOLING_PRE_BOUNDARY = "last_token_before_trigger_boundary"
V34_QUERY_POOLING_CURRENT_TOKEN = "current_generated_token"
V3_QUERY_POOLING_METHODS = frozenset({
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_PRE_BOUNDARY,
    V34_QUERY_POOLING_CURRENT_TOKEN,
})


def query_embedding_token_index(*, token_count: int, pooling: str) -> int:
    """Resolve the full-prefix token selected by a V3 query-pooling policy."""

    if pooling not in V3_QUERY_POOLING_METHODS:
        raise ValueError("Unexpected V3 query_pooling")
    offset = -2 if pooling == V3_QUERY_POOLING_PRE_BOUNDARY else -1
    index = token_count + offset
    if index < 0 or index >= token_count:
        raise ValueError("V3 query prefix is too short for its pooling policy")
    return index


@dataclass(frozen=True)
class ExperienceMemoryV3Profile:
    """Frozen V3 behavior, excluding thresholds loaded from the risk artifact."""

    layer_number: int = 24
    query_context: str = "question_plus_full_partial_cot"
    query_encoder_state: str = "pure_prefix_reencode_side_kv_disabled"
    query_pooling: str = V3_QUERY_POOLING_BOUNDARY_LAST
    query_normalization: str = "l2"
    retrieval_method: str = "exact_cosine"
    retrieval_embedding_transform: str = (
        V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE
    )
    retrieval_abstention_policy: str = "disabled"
    retrieval_min_top1_top2_margin: float | None = None
    retrieval_top_k: int = 2
    selected_memory_count: int = 1
    max_retrieval_attempts: int = 3
    gate_policy: str = "entropy_hysteresis_rearm"
    risk_role: str = "diagnostic_only"
    boundary_policy: str = "pre_answer_comma_period_newline"
    rearm_policy: str = "low_boundary_rearms_next_future_high_boundary"
    rearm_low_entropy_token_count: int = 1
    replacement_policy: str = "replace_current_memory"
    duplicate_policy: str = "consume_attempt_keep_current_memory"
    abstain_policy: str = "consume_attempt_keep_current_memory"
    injection_policy: str = "persistent_until_replace_or_eos"
    memory_score_normalization: str = "log_valid_slots"
    memory_score_bias: float = math.log(10.0)
    attention_backend: str = "sdpa"
    schema_version: str = V3_SYSTEM_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version not in {
            V3_SYSTEM_PROFILE_SCHEMA,
            V34_SYSTEM_PROFILE_SCHEMA,
        }:
            raise ValueError("Unexpected V3 system profile schema")
        if self.layer_number != 24:
            raise ValueError("The current V3 system is frozen to layer 24")
        expected = {
            "query_context": "question_plus_full_partial_cot",
            "query_encoder_state": "pure_prefix_reencode_side_kv_disabled",
            "query_normalization": "l2",
            "retrieval_method": "exact_cosine",
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
        if self.schema_version == V3_SYSTEM_PROFILE_SCHEMA:
            versioned_expected = {
                "gate_policy": "entropy_hysteresis_rearm",
                "risk_role": "diagnostic_only",
                "boundary_policy": "pre_answer_comma_period_newline",
                "rearm_policy": (
                    "low_boundary_rearms_next_future_high_boundary"
                ),
                "rearm_low_entropy_token_count": 1,
            }
        else:
            versioned_expected = {
                "query_pooling": V34_QUERY_POOLING_CURRENT_TOKEN,
                "gate_policy": "continuous_token_entropy_risk_hysteresis",
                "risk_role": "online_joint_control",
                "boundary_policy": "none_pre_answer_every_generated_token",
                "rearm_policy": (
                    "two_consecutive_low_entropy_tokens_rearm_without_trigger"
                ),
                "rearm_low_entropy_token_count": 2,
            }
        for field_name, expected_value in versioned_expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"Unexpected V3 {field_name}")
        if (
            self.schema_version == V3_SYSTEM_PROFILE_SCHEMA
            and self.query_pooling == V34_QUERY_POOLING_CURRENT_TOKEN
        ):
            raise ValueError(
                "V3.4 current-token pooling requires the V3.4 profile schema"
            )
        if self.query_pooling not in V3_QUERY_POOLING_METHODS:
            raise ValueError("Unexpected V3 query_pooling")
        if self.retrieval_embedding_transform not in (
            V3_RETRIEVAL_EMBEDDING_TRANSFORMS
        ):
            raise ValueError("Unexpected V3 retrieval_embedding_transform")
        if self.retrieval_abstention_policy == "disabled":
            if self.retrieval_min_top1_top2_margin is not None:
                raise ValueError(
                    "Disabled V3 retrieval abstention cannot set a margin"
                )
        elif self.retrieval_abstention_policy == "top1_top2_margin":
            threshold = self.retrieval_min_top1_top2_margin
            if (
                threshold is None
                or not math.isfinite(threshold)
                or threshold < 0.0
            ):
                raise ValueError(
                    "V3 margin abstention needs a finite non-negative threshold"
                )
        else:
            raise ValueError("Unexpected V3 retrieval_abstention_policy")
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
    def continuous_token_joint(
        cls,
        *,
        retrieval_embedding_transform: str = (
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE
        ),
        retrieval_abstention_policy: str = "disabled",
        retrieval_min_top1_top2_margin: float | None = None,
    ) -> "ExperienceMemoryV3Profile":
        """Build the frozen V3.4 no-boundary, joint entropy-risk profile."""

        return cls(
            query_pooling=V34_QUERY_POOLING_CURRENT_TOKEN,
            retrieval_embedding_transform=retrieval_embedding_transform,
            retrieval_abstention_policy=retrieval_abstention_policy,
            retrieval_min_top1_top2_margin=(
                retrieval_min_top1_top2_margin
            ),
            gate_policy="continuous_token_entropy_risk_hysteresis",
            risk_role="online_joint_control",
            boundary_policy="none_pre_answer_every_generated_token",
            rearm_policy=(
                "two_consecutive_low_entropy_tokens_rearm_without_trigger"
            ),
            rearm_low_entropy_token_count=2,
            schema_version=V34_SYSTEM_PROFILE_SCHEMA,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperienceMemoryV3Profile":
        if value.get("schema_version") not in {
            V3_SYSTEM_PROFILE_SCHEMA,
            V34_SYSTEM_PROFILE_SCHEMA,
        }:
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
        if self.status not in {"selected", "below_margin", "empty_bank"}:
            raise ValueError(f"Unexpected V3 retrieval status: {self.status}")
        if self.status == "selected":
            if self.matched_memory is None or not self.hits:
                raise ValueError("Selected V3 retrieval requires hits and memory")
            if self.hits[0].get("memory_id") != self.matched_memory.memory_id:
                raise ValueError("Top embedding hit and selected memory differ")
        elif self.status == "below_margin":
            if self.matched_memory is not None or len(self.hits) < 2:
                raise ValueError(
                    "Margin-abstained V3 retrieval requires top-2 hits and no memory"
                )
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
