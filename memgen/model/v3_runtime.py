"""Versioned V3 generation, including V3.5 applicability and safe teardown."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
import time
from typing import Any, Mapping, Protocol, Sequence

import torch

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v3 import (
    V34_GENERATION_RESULT_SCHEMA,
    V34_SYSTEM_PROFILE_SCHEMA,
    V35_GENERATION_RESULT_SCHEMA,
    V35_SYSTEM_PROFILE_SCHEMA,
    V3_GENERATION_RESULT_SCHEMA,
    EmbeddingRetrievalDecision,
    ExperienceMemoryV3Profile,
)
from memgen.model.e1_runtime import (
    EntropyRiskGate,
    GateProbe,
    GreedyDecodingPolicy,
    clone_cache,
    logits_kl,
)
from memgen.model.retrieval_keys import (
    EmbeddingMemoryRetriever,
    FullPrefixQueryEncoder,
)
from memgen.model.side_kv import (
    SideKVAttentionController,
    SideKVAttentionTrace,
    SideKVBankLoader,
    SideKVMemory,
)


_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)", re.IGNORECASE
)


class V35StaticSelectionProtocol(Protocol):
    """Loose runtime boundary for the independently versioned V3.5 selector."""

    available: bool
    unavailable_reason: str | None

    def to_dict(self) -> dict[str, Any]: ...


class V35RetrievalDecisionProtocol(Protocol):
    """Fields shared by legacy and applicability-aware retrieval decisions."""

    status: str
    selected: bool
    query: Mapping[str, Any]
    hits: tuple[Mapping[str, Any], ...]
    matched_memory: Any | None

    def to_dict(self) -> dict[str, Any]: ...


class V35RetrieverProtocol(Protocol):
    """Runtime-facing two-stage selector API; implementation stays decoupled."""

    profile: ExperienceMemoryV3Profile | None
    shortlist_k: int
    applicability_score_floor: float
    dynamic_min_top1_top2_margin: float | None

    def prepare_question(self, question: str) -> V35StaticSelectionProtocol: ...

    def retrieve(
        self,
        *,
        query_embedding: torch.Tensor,
        query_token_ids: Sequence[int],
        prompt_token_count: int,
        static_context: V35StaticSelectionProtocol,
    ) -> V35RetrievalDecisionProtocol: ...


@dataclass(frozen=True)
class EntropyHysteresisConfig:
    layer_number: int
    sink_token_count: int
    high_entropy_threshold: float
    low_entropy_threshold: float
    risk_threshold: float
    risk_role: str = "diagnostic_only"
    rearm_low_entropy_token_count: int = 1

    def __post_init__(self) -> None:
        if self.layer_number != 24 or self.sink_token_count < 0:
            raise ValueError("V3 hysteresis requires layer 24 and a valid sink count")
        if not all(math.isfinite(value) for value in (
            self.high_entropy_threshold,
            self.low_entropy_threshold,
            self.risk_threshold,
        )):
            raise ValueError("V3 hysteresis thresholds must be finite")
        if self.low_entropy_threshold > self.high_entropy_threshold:
            raise ValueError("V3 low entropy threshold exceeds the high threshold")
        if self.risk_role not in {"diagnostic_only", "online_joint_control"}:
            raise ValueError("Unexpected V3 risk-control role")
        if self.rearm_low_entropy_token_count <= 0:
            raise ValueError("V3 re-arm token count must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntropyHysteresisGate:
    """Qualified entropy hysteresis with diagnostic or joint risk control."""

    def __init__(
        self,
        *,
        diagnostic_gate: EntropyRiskGate,
        config: EntropyHysteresisConfig,
    ):
        if diagnostic_gate.config.layer_number != config.layer_number:
            raise ValueError("V3 gate and risk diagnostic layers differ")
        if diagnostic_gate.config.sink_token_count != config.sink_token_count:
            raise ValueError("V3 gate and risk diagnostic sink counts differ")
        if not math.isclose(
            diagnostic_gate.config.entropy_threshold,
            config.high_entropy_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V3 high threshold differs from the risk artifact")
        if not math.isclose(
            diagnostic_gate.config.risk_threshold,
            config.risk_threshold,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V3 risk diagnostic threshold drifted")
        self.diagnostic_gate = diagnostic_gate
        self.config = config

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]) -> "EntropyHysteresisGate":
        value = dict(artifact)
        diagnostic_gate = EntropyRiskGate.from_artifact(value)
        construction = value.get("construction", {})
        low = construction.get("low_entropy_threshold")
        if not isinstance(low, (int, float)) or isinstance(low, bool):
            raise ValueError("Risk artifact has no frozen low-entropy re-arm threshold")
        return cls(
            diagnostic_gate=diagnostic_gate,
            config=EntropyHysteresisConfig(
                layer_number=diagnostic_gate.config.layer_number,
                sink_token_count=diagnostic_gate.config.sink_token_count,
                high_entropy_threshold=diagnostic_gate.config.entropy_threshold,
                low_entropy_threshold=float(low),
                risk_threshold=diagnostic_gate.config.risk_threshold,
            ),
        )

    @classmethod
    def from_token_artifact(
        cls, artifact: Mapping[str, Any]
    ) -> "EntropyHysteresisGate":
        from memgen.experience.risk import TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA

        value = dict(artifact)
        if (
            value.get("schema_version") != TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA
            or value.get("status") != "passed"
        ):
            raise ValueError(
                "V3.4/V3.5 require a qualified token-risk artifact"
            )
        construction = value.get("construction", {})
        qualification = value.get("qualification", {})
        if (
            construction.get("observation_scope")
            != "every_pre_answer_generated_token"
            or construction.get("label_policy")
            != "stable_low_recovery_within_frozen_horizon"
            or int(construction.get("stable_low_token_count", -1)) != 2
            or qualification.get("passed") is not True
        ):
            raise ValueError(
                "V3.4/V3.5 token-risk construction did not qualify"
            )
        diagnostic_gate = EntropyRiskGate.from_artifact(value)
        low = construction.get("low_entropy_threshold")
        if not isinstance(low, (int, float)) or isinstance(low, bool):
            raise ValueError("Token-risk artifact has no frozen low threshold")
        return cls(
            diagnostic_gate=diagnostic_gate,
            config=EntropyHysteresisConfig(
                layer_number=diagnostic_gate.config.layer_number,
                sink_token_count=diagnostic_gate.config.sink_token_count,
                high_entropy_threshold=diagnostic_gate.config.entropy_threshold,
                low_entropy_threshold=float(low),
                risk_threshold=diagnostic_gate.config.risk_threshold,
                risk_role="online_joint_control",
                rearm_low_entropy_token_count=2,
            ),
        )

    def probe(self, **kwargs: Any) -> GateProbe:
        return self.diagnostic_gate.probe(**kwargs)

    def trigger_qualified(self, probe: GateProbe) -> bool:
        if probe.entropy < self.config.high_entropy_threshold:
            return False
        return (
            self.config.risk_role == "diagnostic_only"
            or probe.risk_score > self.config.risk_threshold
        )


@dataclass(frozen=True)
class V3BoundaryTrace:
    generated_boundary_index: int
    boundary_token_id: int
    boundary_token_text: str
    state_before: str
    state_after: str
    entropy: float
    high_entropy_threshold: float
    low_entropy_threshold: float
    persistence_risk_score: float
    persistence_risk_threshold: float
    risk_role: str
    action: str
    retrieval_attempt_count_before: int
    retrieval_attempt_count_after: int
    active_memory_id_before: str | None
    active_memory_id_after: str | None
    trace_scope: str
    vocabulary_entropy: float
    top1_top2_logit_margin: float
    low_entropy_streak_before: int
    low_entropy_streak_after: int
    joint_trigger_qualified: bool
    active_memory_conditioned: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({
            "generated_observation_index": self.generated_boundary_index,
            "observation_token_id": self.boundary_token_id,
            "observation_token_text": self.boundary_token_text,
            "affects_generated_token_index": self.generated_boundary_index + 1,
        })
        return value


@dataclass(frozen=True)
class V3RetrievalAttemptTrace:
    attempt_number: int
    generated_boundary_index: int
    boundary_token_id: int
    boundary_token_text: str
    query_embedding_token_id: int
    query_embedding_token_text: str
    outcome: str
    previous_memory_id: str | None
    selected_memory_id: str | None
    active_memory_id_after: str | None
    retrieval_decision: EmbeddingRetrievalDecision | V35RetrievalDecisionProtocol
    query_encoding_seconds: float
    retrieval_seconds: float
    memory_load_seconds: float | None
    activation_forward_seconds: float | None
    attempt_total_seconds: float
    activation_first_step_logits_kl: float | None
    activation_first_step_top1_changed: bool | None
    activation_baseline_first_token_id: int | None
    affects_generated_token_index: int
    terminal_abstain: bool | None = None
    memory_cleared_on_abstain: bool | None = None
    cleared_memory_id: str | None = None
    deactivation_forward_seconds: float | None = None
    deactivation_first_step_logits_kl: float | None = None
    deactivation_first_step_top1_changed: bool | None = None
    deactivation_baseline_first_token_id: int | None = None
    deactivation_native_first_token_id: int | None = None
    clear_affects_generated_token_index: int | None = None
    actual_path_after_abstain: str | None = None
    actual_path_memory_id_after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update({
            "generated_observation_index": self.generated_boundary_index,
            "observation_token_id": self.boundary_token_id,
            "observation_token_text": self.boundary_token_text,
        })
        value["retrieval_decision"] = self.retrieval_decision.to_dict()
        if self.terminal_abstain is None:
            for name in (
                "terminal_abstain",
                "memory_cleared_on_abstain",
                "cleared_memory_id",
                "deactivation_forward_seconds",
                "deactivation_first_step_logits_kl",
                "deactivation_first_step_top1_changed",
                "deactivation_baseline_first_token_id",
                "deactivation_native_first_token_id",
                "clear_affects_generated_token_index",
                "actual_path_after_abstain",
                "actual_path_memory_id_after",
            ):
                value.pop(name)
        return value


@dataclass(frozen=True)
class V3MemoryTransition:
    generated_boundary_index: int
    transition: str
    previous_memory_id: str | None
    next_memory_id: str | None
    attempt_number: int
    affects_generated_token_index: int

    def __post_init__(self) -> None:
        if self.transition == "deactivated_on_terminal_abstain":
            if self.previous_memory_id is None or self.next_memory_id is not None:
                raise ValueError("V3.5 terminal clear transition metadata drifted")
        elif not self.next_memory_id:
            raise ValueError("Legacy V3 memory transitions require a next memory")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V3AttentionStepTrace:
    generated_input_index: int
    processed_prefix_token_count: int
    trace: SideKVAttentionTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_input_index": self.generated_input_index,
            "processed_prefix_token_count": self.processed_prefix_token_count,
            **self.trace.to_dict(),
        }


@dataclass(frozen=True)
class V3GenerationResult:
    completion_token_ids: tuple[int, ...]
    boundary_traces: tuple[V3BoundaryTrace, ...]
    retrieval_attempts: tuple[V3RetrievalAttemptTrace, ...]
    memory_transitions: tuple[V3MemoryTransition, ...]
    attention_traces: tuple[V3AttentionStepTrace, ...]
    final_gate_state: str
    final_memory_id: str | None
    answer_marker_seen: bool
    query_embeddings: tuple[torch.Tensor, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    schema_version: str = V3_GENERATION_RESULT_SCHEMA
    static_selector_trace: Mapping[str, Any] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def generated_token_count(self) -> int:
        return len(self.completion_token_ids)

    @property
    def retrieval_attempt_count(self) -> int:
        return len(self.retrieval_attempts)

    @property
    def rearm_count(self) -> int:
        return sum(trace.action == "rearmed" for trace in self.boundary_traces)

    @property
    def replacement_count(self) -> int:
        return sum(
            attempt.outcome == "replaced" for attempt in self.retrieval_attempts
        )

    @property
    def duplicate_count(self) -> int:
        return sum(
            attempt.outcome == "duplicate" for attempt in self.retrieval_attempts
        )

    @property
    def abstain_count(self) -> int:
        return sum(
            attempt.outcome == "abstained" for attempt in self.retrieval_attempts
        )

    @property
    def terminal_abstain_count(self) -> int:
        return sum(
            attempt.terminal_abstain is True
            for attempt in self.retrieval_attempts
        )

    @property
    def clear_on_terminal_abstain_count(self) -> int:
        return sum(
            attempt.memory_cleared_on_abstain is True
            for attempt in self.retrieval_attempts
        )

    def _stale_attention_after_terminal_clear_count(self) -> int:
        clear_points = [
            (
                int(attempt.generated_boundary_index),
                str(attempt.cleared_memory_id),
            )
            for attempt in self.retrieval_attempts
            if attempt.memory_cleared_on_abstain is True
            and attempt.cleared_memory_id is not None
        ]
        return sum(
            trace.generated_input_index >= clear_index
            and trace.trace.memory_id == memory_id
            for clear_index, memory_id in clear_points
            for trace in self.attention_traces
        )

    def _no_rearm_after_terminal_abstain(self) -> bool:
        terminal_indices = [
            attempt.generated_boundary_index
            for attempt in self.retrieval_attempts
            if attempt.terminal_abstain is True
        ]
        if not terminal_indices:
            return True
        first_terminal = min(terminal_indices)
        return not any(
            trace.generated_boundary_index >= first_terminal
            and trace.action == "rearmed"
            for trace in self.boundary_traces
        )

    def _two_low_rearm_respected(self) -> bool:
        return all(
            trace.action != "rearmed"
            or (
                trace.state_before == "DISARMED"
                and trace.low_entropy_streak_before == 1
                and trace.retrieval_attempt_count_before
                == trace.retrieval_attempt_count_after
            )
            for trace in self.boundary_traces
        )

    @property
    def native_gate_observation_count(self) -> int:
        return sum(
            not trace.active_memory_conditioned for trace in self.boundary_traces
        )

    @property
    def memory_conditioned_gate_observation_count(self) -> int:
        return sum(
            trace.active_memory_conditioned for trace in self.boundary_traces
        )

    def _memory_activation_spans(self) -> list[dict[str, Any]]:
        spans: list[dict[str, Any]] = []
        for item in self.attention_traces:
            if not spans or spans[-1]["memory_id"] != item.trace.memory_id:
                spans.append({
                    "memory_id": item.trace.memory_id,
                    "start_generated_input_index": item.generated_input_index,
                    "end_generated_input_index": item.generated_input_index,
                    "attention_step_count": 1,
                    "memory_attention_mass_sum": float(
                        item.trace.memory_attention_mass
                    ),
                })
            else:
                span = spans[-1]
                span["end_generated_input_index"] = item.generated_input_index
                span["attention_step_count"] += 1
                span["memory_attention_mass_sum"] += float(
                    item.trace.memory_attention_mass
                )
        for span in spans:
            span["mean_memory_attention_mass"] = (
                span.pop("memory_attention_mass_sum")
                / span["attention_step_count"]
            )
        return spans

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "completion_token_ids": list(self.completion_token_ids),
            "completion_token_ids_sha256": canonical_json_sha256(
                list(self.completion_token_ids)
            ),
            "generated_token_count": self.generated_token_count,
            "gate_trace_storage_field": "boundary_traces",
            "gate_trace_scope": (
                self.boundary_traces[0].trace_scope
                if self.boundary_traces
                else None
            ),
            "boundary_traces": [trace.to_dict() for trace in self.boundary_traces],
            "retrieval_attempts": [
                attempt.to_dict() for attempt in self.retrieval_attempts
            ],
            "memory_transitions": [
                transition.to_dict() for transition in self.memory_transitions
            ],
            "memory_activation_spans": self._memory_activation_spans(),
            "attention_traces": [trace.to_dict() for trace in self.attention_traces],
            "final_gate_state": self.final_gate_state,
            "final_memory_id": self.final_memory_id,
            "answer_marker_seen": self.answer_marker_seen,
            "summary": {
                "retrieval_attempt_count": self.retrieval_attempt_count,
                "rearm_count": self.rearm_count,
                "replacement_count": self.replacement_count,
                "duplicate_count": self.duplicate_count,
                "memory_attention_step_count": len(self.attention_traces),
                "gate_observation_count": len(self.boundary_traces),
                "joint_trigger_qualified_count": sum(
                    trace.joint_trigger_qualified
                    for trace in self.boundary_traces
                ),
                "native_gate_observation_count": (
                    self.native_gate_observation_count
                ),
                "memory_conditioned_gate_observation_count": (
                    self.memory_conditioned_gate_observation_count
                ),
            },
        }
        if self.schema_version == V35_GENERATION_RESULT_SCHEMA:
            static_trace = (
                dict(self.static_selector_trace)
                if self.static_selector_trace is not None
                else None
            )
            static_unavailable = bool(
                static_trace is not None
                and static_trace.get("static_selector_unavailable", False)
            )
            stale_count = self._stale_attention_after_terminal_clear_count()
            value["static_selector_trace"] = static_trace
            value["summary"].update({
                "abstain_count": self.abstain_count,
                "terminal_abstain_count": self.terminal_abstain_count,
                "clear_on_terminal_abstain_count": (
                    self.clear_on_terminal_abstain_count
                ),
                "static_selector_unavailable": static_unavailable,
                "initial_gate_state": (
                    "EXHAUSTED" if static_unavailable else "ARMED"
                ),
                "static_selector_unavailable_zero_attempt": (
                    not static_unavailable
                    or self.retrieval_attempt_count == 0
                ),
                "max_three_attempts_respected": self.retrieval_attempt_count <= 3,
                "two_low_rearm_respected": self._two_low_rearm_respected(),
                "second_low_rearms_without_trigger": (
                    self._two_low_rearm_respected()
                ),
                "no_rearm_after_terminal_abstain": (
                    self._no_rearm_after_terminal_abstain()
                ),
                "stale_memory_attention_after_terminal_clear_count": (
                    stale_count
                ),
                "terminal_clear_attention_safe": stale_count == 0,
                "final_gate_state": self.final_gate_state,
                "final_memory_id": self.final_memory_id,
            })
        return value


class OnlineExperienceMemorySystemV3:
    """Generate with at most three entropy-triggered memory replacements."""

    VALID_STATES = frozenset({"ARMED", "DISARMED", "EXHAUSTED", "CLOSED"})

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str,
        max_new_tokens: int,
        gate: EntropyHysteresisGate,
        query_encoder: FullPrefixQueryEncoder,
        retriever: EmbeddingMemoryRetriever | V35RetrieverProtocol,
        loader: SideKVBankLoader,
        controller: SideKVAttentionController,
        profile: ExperienceMemoryV3Profile,
    ):
        if max_new_tokens <= 0:
            raise ValueError("V3 online generation needs a positive token budget")
        if gate.config.layer_number != profile.layer_number:
            raise ValueError("V3 gate and profile layers differ")
        if (
            gate.config.risk_role != profile.risk_role
            or gate.config.rearm_low_entropy_token_count
            != profile.rearm_low_entropy_token_count
        ):
            raise ValueError("V3 gate and profile control policies differ")
        if query_encoder.layer_number != profile.layer_number:
            raise ValueError("V3 query encoder and profile layers differ")
        if query_encoder.query_pooling != profile.query_pooling:
            raise ValueError("V3 query encoder and profile pooling differ")
        if controller.layer_number != profile.layer_number:
            raise ValueError("V3 side-KV controller and profile layers differ")
        if profile.schema_version == V35_SYSTEM_PROFILE_SCHEMA:
            if not callable(getattr(retriever, "prepare_question", None)):
                raise ValueError(
                    "V3.5 retriever must implement question applicability selection"
                )
            if not callable(getattr(retriever, "retrieve", None)):
                raise ValueError("V3.5 retriever must implement dynamic reranking")
            retriever_profile = getattr(retriever, "profile", None)
            if retriever_profile is not None and retriever_profile != profile:
                raise ValueError("V3.5 retriever and online profiles differ")
            if (
                getattr(retriever, "shortlist_k", None)
                != profile.applicability_shortlist_k
            ):
                raise ValueError("V3.5 runtime and retriever shortlist k differ")
            retriever_floor = getattr(
                retriever, "applicability_score_floor", None
            )
            if (
                retriever_floor is None
                or not math.isclose(
                    float(retriever_floor),
                    float(profile.applicability_score_floor),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "V3.5 runtime and retriever applicability floors differ"
                )
            retriever_margin = getattr(
                retriever, "dynamic_min_top1_top2_margin", None
            )
            profile_margin = profile.retrieval_min_top1_top2_margin
            if (retriever_margin is None) != (profile_margin is None) or (
                retriever_margin is not None
                and not math.isclose(
                    float(retriever_margin),
                    float(profile_margin),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "V3.5 runtime and retriever dynamic margins differ"
                )
        elif retriever.profile != profile:
            raise ValueError("V3 retriever and online profiles differ")
        if (
            controller.memory_score_normalization
            != profile.memory_score_normalization
            or not math.isclose(
                controller.memory_score_bias,
                profile.memory_score_bias,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("V3 side-KV strength differs from the profile")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.gate = gate
        self.query_encoder = query_encoder
        self.retriever = retriever
        self.loader = loader
        self.controller = controller
        self.profile = profile
        self.decoding = GreedyDecodingPolicy(tokenizer=tokenizer, device=device)

    @property
    def _is_v35(self) -> bool:
        return self.profile.schema_version == V35_SYSTEM_PROFILE_SCHEMA

    @property
    def _uses_continuous_token_gate(self) -> bool:
        return self.profile.schema_version in {
            V34_SYSTEM_PROFILE_SCHEMA,
            V35_SYSTEM_PROFILE_SCHEMA,
        }

    @staticmethod
    def _static_selection_dict(
        selection: V35StaticSelectionProtocol | Mapping[str, Any],
    ) -> dict[str, Any]:
        serializer = getattr(selection, "to_dict", None)
        if callable(serializer):
            value = serializer()
        elif isinstance(selection, Mapping):
            value = dict(selection)
        else:
            raise TypeError("V3.5 static selection is not serializable")
        if not isinstance(value, Mapping):
            raise TypeError("V3.5 static selection serializer returned no mapping")
        return dict(value)

    @classmethod
    def _static_selection_available(
        cls,
        selection: V35StaticSelectionProtocol | Mapping[str, Any],
    ) -> bool:
        available = getattr(selection, "available", None)
        if isinstance(available, bool):
            return available
        value = cls._static_selection_dict(selection)
        if isinstance(value.get("static_selector_unavailable"), bool):
            return not bool(value["static_selector_unavailable"])
        if isinstance(value.get("shortlist_nonempty"), bool):
            return bool(value["shortlist_nonempty"])
        raise ValueError("V3.5 static selection has no availability flag")

    @classmethod
    def _static_shortlist_memory_ids(
        cls,
        selection: V35StaticSelectionProtocol | Mapping[str, Any],
    ) -> tuple[str, ...]:
        memory_ids = getattr(selection, "memory_ids", None)
        if memory_ids is None:
            memory_ids = cls._static_selection_dict(selection).get(
                "shortlist_memory_ids"
            )
        if not isinstance(memory_ids, (list, tuple)):
            raise ValueError("V3.5 static selection has no shortlist memory IDs")
        values = tuple(str(memory_id) for memory_id in memory_ids)
        if any(not memory_id for memory_id in values) or len(set(values)) != len(values):
            raise ValueError("V3.5 static shortlist IDs are invalid")
        return values

    def _tensor(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    def _is_delimiter(self, token_id: int) -> bool:
        text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        return text.rstrip(" \t").endswith((",", ".", "\n"))

    def _should_observe(self, *, generation_step: int, state: str, token_id: int) -> bool:
        if generation_step <= 0 or state == "CLOSED":
            return False
        if self._uses_continuous_token_gate:
            return True
        return self._is_delimiter(token_id)

    @staticmethod
    def _logit_diagnostics(logits: torch.Tensor) -> tuple[float, float]:
        values = logits[0, -1, :].detach().float()
        log_probabilities = torch.log_softmax(values, dim=-1)
        probabilities = log_probabilities.exp()
        vocabulary_entropy = float(
            (-(probabilities * log_probabilities).sum()).item()
        )
        top2 = torch.topk(values, k=2).values
        margin = float((top2[0] - top2[1]).item())
        if not math.isfinite(vocabulary_entropy) or not math.isfinite(margin):
            raise RuntimeError("V3 logit diagnostics are non-finite")
        return vocabulary_entropy, margin

    @staticmethod
    def _cache_sequence_length(cache: Any) -> int | None:
        if cache is None:
            return 0
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            return int(getter())
        try:
            return int(cache[0][0].shape[-2])
        except (IndexError, KeyError, TypeError, AttributeError):
            return None

    @classmethod
    def _restore_cache_length(cls, cache: Any, *, expected_length: int) -> Any:
        """Rollback a mutable live cache after a counterfactual probe."""

        actual = cls._cache_sequence_length(cache)
        if actual == expected_length:
            return cache
        crop = getattr(cache, "crop", None)
        if actual is None or actual < expected_length or not callable(crop):
            raise RuntimeError(
                "Unable to restore continuous-gate live cache after probing"
            )
        crop(expected_length)
        if cls._cache_sequence_length(cache) != expected_length:
            raise RuntimeError("Continuous-gate live cache rollback length drifted")
        return cache

    def _load_selected_memory(
        self,
        decision: EmbeddingRetrievalDecision | V35RetrievalDecisionProtocol,
    ) -> SideKVMemory:
        choice = decision.matched_memory
        if choice is None:
            raise ValueError("Cannot load memory from an abstaining V3 decision")
        memory = self.loader.get(
            choice.memory_id,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        if (
            memory.payload_hash != choice.payload_hash
            or memory.valid_slot_count != choice.kv_valid_slot_count
            or memory.layer_number != self.profile.layer_number
        ):
            raise ValueError("V3 embedding selection and side-KV metadata differ")
        return memory

    def _retrieve(
        self,
        *,
        query_embedding: torch.Tensor,
        query_token_ids: Sequence[int],
        prompt_token_count: int,
        static_selection: V35StaticSelectionProtocol | Mapping[str, Any] | None,
    ) -> EmbeddingRetrievalDecision | V35RetrievalDecisionProtocol:
        kwargs: dict[str, Any] = {
            "query_embedding": query_embedding,
            "query_token_ids": query_token_ids,
            "prompt_token_count": prompt_token_count,
        }
        if self._is_v35:
            if static_selection is None:
                raise RuntimeError("V3.5 dynamic retrieval has no static shortlist")
            kwargs["static_context"] = static_selection
        decision = self.retriever.retrieve(**kwargs)
        if self._is_v35 and not all(
            hasattr(decision, name)
            for name in (
                "status",
                "selected",
                "matched_memory",
                "query",
                "hits",
                "to_dict",
            )
        ):
            raise TypeError(
                "V3.5 retrieval decision does not satisfy the runtime contract"
            )
        if self._is_v35 and decision.selected:
            choice = decision.matched_memory
            if choice is None:
                raise RuntimeError("Selected V3.5 decision has no matched memory")
            shortlist_ids = self._static_shortlist_memory_ids(static_selection)
            if choice.memory_id not in shortlist_ids:
                raise RuntimeError(
                    "V3.5 dynamic selection escaped the static applicability shortlist"
                )
        return decision

    def _record_actual_attention(
        self,
        *,
        trace_start: int,
        ids: Sequence[int],
        prompt_length: int,
        destination: list[V3AttentionStepTrace],
    ) -> None:
        traces = self.controller.traces[trace_start:]
        if self.controller.active_memory is None:
            if traces:
                raise RuntimeError("V3 inactive side-KV path unexpectedly emitted traces")
            return
        if len(traces) != 1:
            raise RuntimeError("V3 expects exactly one side-KV trace per decode step")
        trace = traces[0]
        if trace.native_key_length != len(ids):
            raise RuntimeError("V3 side-KV trace includes non-native cache positions")
        if trace.memory_id != self.controller.active_memory.memory_id:
            raise RuntimeError("V3 attention trace memory ID drifted")
        if (
            trace.memory_slot_count
            != self.controller.active_memory.valid_slot_count
            or trace.memory_score_normalization
            != self.profile.memory_score_normalization
            or not math.isclose(
                trace.memory_score_bias,
                self.profile.memory_score_bias,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isfinite(trace.memory_attention_mass)
            or trace.memory_attention_mass <= 0.0
            or not math.isfinite(trace.native_attention_mass)
        ):
            raise RuntimeError("V3 side-KV attention trace metadata drifted")
        destination.append(V3AttentionStepTrace(
            generated_input_index=len(ids) - prompt_length - 1,
            processed_prefix_token_count=len(ids),
            trace=trace,
        ))

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompt_token_ids: Sequence[int],
        question: str | None = None,
    ) -> V3GenerationResult:
        ids = list(prompt_token_ids)
        if not ids:
            raise ValueError("V3 online system requires a non-empty prompt")
        prompt_length = len(ids)
        past = None
        state = "ARMED"
        attempt_count = 0
        current_memory: SideKVMemory | None = None
        boundary_traces: list[V3BoundaryTrace] = []
        attempts: list[V3RetrievalAttemptTrace] = []
        transitions: list[V3MemoryTransition] = []
        attention_traces: list[V3AttentionStepTrace] = []
        query_embeddings: list[torch.Tensor] = []
        answer_marker_seen = False
        low_entropy_streak = 0
        static_selection: V35StaticSelectionProtocol | Mapping[str, Any] | None = None
        static_selector_trace: dict[str, Any] | None = None
        static_selector_unavailable = False
        self.controller.deactivate()
        self.controller.clear_traces()
        try:
            if self._is_v35:
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(
                        "V3.5 online generation requires the original question text"
                    )
                with self.controller.suspend_memory():
                    static_selection = self.retriever.prepare_question(question)
                if (
                    self.controller.active_memory is not None
                    or self.controller.traces
                ):
                    raise RuntimeError(
                        "V3.5 static question encoding was not side-KV isolated"
                    )
                static_selector_trace = self._static_selection_dict(
                    static_selection
                )
                static_selector_unavailable = not self._static_selection_available(
                    static_selection
                )
                shortlist_ids = self._static_shortlist_memory_ids(static_selection)
                if static_selector_unavailable:
                    if shortlist_ids and len(shortlist_ids) >= 2:
                        raise RuntimeError(
                            "V3.5 unavailable static selection retained a full shortlist"
                        )
                    state = "EXHAUSTED"
                elif len(shortlist_ids) < 2:
                    raise RuntimeError(
                        "V3.5 available static selection has fewer than two memories"
                    )
            for generation_step in range(self.max_new_tokens):
                full = self._tensor(ids)
                attention_mask = torch.ones_like(full)
                completion_text = self.tokenizer.decode(
                    ids[prompt_length:], skip_special_tokens=False
                )
                if _ANSWER_MARKER_RE.search(completion_text):
                    answer_marker_seen = True
                    state = "CLOSED"

                trace_start = len(self.controller.traces)
                probe: GateProbe | None = None
                if (
                    not static_selector_unavailable
                    and self._should_observe(
                        generation_step=generation_step,
                        state=state,
                        token_id=int(ids[-1]),
                    )
                ):
                    past_length_before_probe: int | None = None
                    if self._uses_continuous_token_gate:
                        past_length_before_probe = self._cache_sequence_length(past)
                        if past_length_before_probe is None:
                            raise RuntimeError(
                                "Unable to audit continuous-gate native cache length"
                            )
                    probe = self.gate.probe(
                        model=self.model,
                        boundary_token=full[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past,
                        clone_past_key_values=(
                            not self._uses_continuous_token_gate
                        ),
                    )
                    state_before = state
                    attempts_before = attempt_count
                    memory_before = (
                        current_memory.memory_id if current_memory is not None else None
                    )
                    low_streak_before = low_entropy_streak
                    action = "observed"
                    output = probe.output
                    vocabulary_entropy, logit_margin = self._logit_diagnostics(
                        probe.output.logits
                    )
                    joint_trigger_qualified = self.gate.trigger_qualified(probe)

                    if state == "ARMED" and joint_trigger_qualified:
                        low_entropy_streak = 0
                        attempt_started = time.perf_counter()
                        attempt_count += 1
                        encoding_started = time.perf_counter()
                        with self.controller.suspend_memory():
                            query_embedding = self.query_encoder.encode(ids)
                        query_encoding_seconds = time.perf_counter() - encoding_started
                        query_embeddings.append(query_embedding.detach().float().cpu())
                        retrieval_started = time.perf_counter()
                        decision = self._retrieve(
                            query_embedding=query_embedding,
                            query_token_ids=ids,
                            prompt_token_count=prompt_length,
                            static_selection=static_selection,
                        )
                        retrieval_seconds = time.perf_counter() - retrieval_started
                        selected_id = (
                            decision.matched_memory.memory_id
                            if decision.matched_memory is not None
                            else None
                        )
                        load_seconds: float | None = None
                        activation_forward_seconds: float | None = None
                        activation_kl: float | None = None
                        top1_changed: bool | None = None
                        baseline_first_token: int | None = None
                        terminal_abstain: bool | None = (
                            False if self._is_v35 else None
                        )
                        memory_cleared_on_abstain: bool | None = (
                            False if self._is_v35 else None
                        )
                        cleared_memory_id: str | None = None
                        deactivation_forward_seconds: float | None = None
                        deactivation_kl: float | None = None
                        deactivation_top1_changed: bool | None = None
                        deactivation_baseline_first_token: int | None = None
                        deactivation_native_first_token: int | None = None
                        clear_affects_generated_token_index: int | None = None
                        actual_path_after_abstain: str | None = None
                        if not decision.selected:
                            outcome = "abstained"
                            if self._is_v35:
                                terminal_abstain = True
                                actual_path_after_abstain = "native"
                                if current_memory is not None:
                                    cleared_memory_id = current_memory.memory_id
                                    memory_cleared_on_abstain = True
                                    clear_affects_generated_token_index = (
                                        len(ids) - prompt_length
                                    )
                                    baseline_scores = (
                                        self.decoding.processed_scores(
                                            token_ids=ids,
                                            logits=probe.output.logits,
                                        )
                                    )
                                    # The active-memory probe is counterfactual
                                    # after a terminal abstain.  Remove its trace,
                                    # unload memory, roll back the native cache,
                                    # and reprocess this token on the native path.
                                    self.controller.truncate_traces(trace_start)
                                    self.controller.deactivate()
                                    current_memory = None
                                    native_past = self._restore_cache_length(
                                        past,
                                        expected_length=int(
                                            past_length_before_probe
                                        ),
                                    )
                                    deactivation_started = time.perf_counter()
                                    native_output = self.model(
                                        input_ids=full[:, -1:],
                                        attention_mask=attention_mask,
                                        past_key_values=native_past,
                                        use_cache=True,
                                        return_dict=True,
                                    )
                                    deactivation_forward_seconds = (
                                        time.perf_counter()
                                        - deactivation_started
                                    )
                                    output = native_output
                                    native_scores = (
                                        self.decoding.processed_scores(
                                            token_ids=ids,
                                            logits=native_output.logits,
                                        )
                                    )
                                    deactivation_kl = logits_kl(
                                        baseline_scores, native_scores
                                    )
                                    if not math.isfinite(deactivation_kl):
                                        raise RuntimeError(
                                            "V3.5 deactivation logits KL is non-finite"
                                        )
                                    deactivation_baseline_first_token = int(
                                        baseline_scores.argmax(dim=-1).item()
                                    )
                                    deactivation_native_first_token = int(
                                        native_scores.argmax(dim=-1).item()
                                    )
                                    deactivation_top1_changed = bool(
                                        deactivation_baseline_first_token
                                        != deactivation_native_first_token
                                    )
                                    transitions.append(V3MemoryTransition(
                                        generated_boundary_index=(
                                            len(ids) - prompt_length - 1
                                        ),
                                        transition=(
                                            "deactivated_on_terminal_abstain"
                                        ),
                                        previous_memory_id=cleared_memory_id,
                                        next_memory_id=None,
                                        attempt_number=attempt_count,
                                        affects_generated_token_index=(
                                            len(ids) - prompt_length
                                        ),
                                    ))
                        elif selected_id == memory_before:
                            if self._is_v35:
                                choice = decision.matched_memory
                                if (
                                    choice is None
                                    or current_memory is None
                                    or choice.payload_hash
                                    != current_memory.payload_hash
                                    or choice.kv_valid_slot_count
                                    != current_memory.valid_slot_count
                                    or current_memory.layer_number
                                    != self.profile.layer_number
                                ):
                                    raise ValueError(
                                        "V3.5 duplicate selection and active "
                                        "side-KV differ"
                                    )
                            outcome = "duplicate"
                        else:
                            load_started = time.perf_counter()
                            next_memory = self._load_selected_memory(decision)
                            load_seconds = time.perf_counter() - load_started
                            baseline_scores = self.decoding.processed_scores(
                                token_ids=ids, logits=probe.output.logits
                            )
                            # The old-memory probe is counterfactual once replacement
                            # occurs, so retain only the new-memory actual-path trace.
                            self.controller.truncate_traces(trace_start)
                            self.controller.activate(next_memory)
                            treatment_past = (
                                self._restore_cache_length(
                                    past,
                                    expected_length=int(
                                        past_length_before_probe
                                    ),
                                )
                                if self._uses_continuous_token_gate
                                else clone_cache(past)
                            )
                            activation_started = time.perf_counter()
                            treatment = self.model(
                                input_ids=full[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=treatment_past,
                                use_cache=True,
                                return_dict=True,
                            )
                            activation_forward_seconds = (
                                time.perf_counter() - activation_started
                            )
                            output = treatment
                            treatment_scores = self.decoding.processed_scores(
                                token_ids=ids, logits=treatment.logits
                            )
                            activation_kl = logits_kl(
                                baseline_scores, treatment_scores
                            )
                            if not math.isfinite(activation_kl):
                                raise RuntimeError(
                                    "V3 activation logits KL is non-finite"
                                )
                            baseline_first_token = int(
                                baseline_scores.argmax(dim=-1).item()
                            )
                            top1_changed = bool(
                                baseline_first_token
                                != int(treatment_scores.argmax(dim=-1).item())
                            )
                            outcome = "activated" if current_memory is None else "replaced"
                            transitions.append(V3MemoryTransition(
                                generated_boundary_index=(
                                    len(ids) - prompt_length - 1
                                ),
                                transition=outcome,
                                previous_memory_id=memory_before,
                                next_memory_id=next_memory.memory_id,
                                attempt_number=attempt_count,
                                affects_generated_token_index=(
                                    len(ids) - prompt_length
                                ),
                            ))
                            current_memory = next_memory
                        state = (
                            "EXHAUSTED"
                            if (
                                terminal_abstain is True
                                or attempt_count
                                >= self.profile.max_retrieval_attempts
                            )
                            else "DISARMED"
                        )
                        action = "retrieval_attempt"
                        attempts.append(V3RetrievalAttemptTrace(
                            attempt_number=attempt_count,
                            generated_boundary_index=len(ids) - prompt_length - 1,
                            boundary_token_id=int(ids[-1]),
                            boundary_token_text=self.tokenizer.decode(
                                [int(ids[-1])], skip_special_tokens=False
                            ),
                            query_embedding_token_id=int(
                                decision.query["query_embedding_token_id"]
                            ),
                            query_embedding_token_text=self.tokenizer.decode(
                                [
                                    int(
                                        decision.query[
                                            "query_embedding_token_id"
                                        ]
                                    )
                                ],
                                skip_special_tokens=False,
                            ),
                            outcome=outcome,
                            previous_memory_id=memory_before,
                            selected_memory_id=selected_id,
                            active_memory_id_after=(
                                current_memory.memory_id
                                if current_memory is not None
                                else None
                            ),
                            retrieval_decision=decision,
                            query_encoding_seconds=query_encoding_seconds,
                            retrieval_seconds=retrieval_seconds,
                            memory_load_seconds=load_seconds,
                            activation_forward_seconds=(
                                activation_forward_seconds
                            ),
                            attempt_total_seconds=(
                                time.perf_counter() - attempt_started
                            ),
                            activation_first_step_logits_kl=activation_kl,
                            activation_first_step_top1_changed=top1_changed,
                            activation_baseline_first_token_id=baseline_first_token,
                            affects_generated_token_index=(
                                len(ids) - prompt_length
                            ),
                            terminal_abstain=terminal_abstain,
                            memory_cleared_on_abstain=(
                                memory_cleared_on_abstain
                            ),
                            cleared_memory_id=cleared_memory_id,
                            deactivation_forward_seconds=(
                                deactivation_forward_seconds
                            ),
                            deactivation_first_step_logits_kl=(
                                deactivation_kl
                            ),
                            deactivation_first_step_top1_changed=(
                                deactivation_top1_changed
                            ),
                            deactivation_baseline_first_token_id=(
                                deactivation_baseline_first_token
                            ),
                            deactivation_native_first_token_id=(
                                deactivation_native_first_token
                            ),
                            clear_affects_generated_token_index=(
                                clear_affects_generated_token_index
                            ),
                            actual_path_after_abstain=(
                                actual_path_after_abstain
                            ),
                            actual_path_memory_id_after=(
                                current_memory.memory_id
                                if current_memory is not None
                                else None
                            ) if self._is_v35 else None,
                        ))
                    elif state == "DISARMED" and (
                        probe.entropy <= self.gate.config.low_entropy_threshold
                    ):
                        low_entropy_streak += 1
                        if (
                            low_entropy_streak
                            >= self.gate.config.rearm_low_entropy_token_count
                        ):
                            state = "ARMED"
                            action = "rearmed"
                            low_entropy_streak = 0
                        else:
                            action = "observed_disarmed_low_streak"
                    elif state == "DISARMED":
                        low_entropy_streak = 0
                        action = "observed_disarmed"
                    elif state == "EXHAUSTED":
                        low_entropy_streak = 0
                        action = "observed_exhausted"
                    else:
                        low_entropy_streak = 0
                        action = "observed_armed_below_high"

                    boundary_traces.append(V3BoundaryTrace(
                        generated_boundary_index=len(ids) - prompt_length - 1,
                        boundary_token_id=int(ids[-1]),
                        boundary_token_text=self.tokenizer.decode(
                            [int(ids[-1])], skip_special_tokens=False
                        ),
                        state_before=state_before,
                        state_after=state,
                        entropy=probe.entropy,
                        high_entropy_threshold=(
                            self.gate.config.high_entropy_threshold
                        ),
                        low_entropy_threshold=(
                            self.gate.config.low_entropy_threshold
                        ),
                        persistence_risk_score=probe.risk_score,
                        persistence_risk_threshold=self.gate.config.risk_threshold,
                        risk_role=self.gate.config.risk_role,
                        action=action,
                        retrieval_attempt_count_before=attempts_before,
                        retrieval_attempt_count_after=attempt_count,
                        active_memory_id_before=memory_before,
                        active_memory_id_after=(
                            current_memory.memory_id
                            if current_memory is not None
                            else None
                        ),
                        trace_scope=(
                            "every_pre_answer_generated_token"
                            if self._uses_continuous_token_gate
                            else "trigger_boundary_only"
                        ),
                        vocabulary_entropy=vocabulary_entropy,
                        top1_top2_logit_margin=logit_margin,
                        low_entropy_streak_before=low_streak_before,
                        low_entropy_streak_after=low_entropy_streak,
                        joint_trigger_qualified=joint_trigger_qualified,
                        active_memory_conditioned=memory_before is not None,
                    ))
                else:
                    kwargs: dict[str, Any] = {
                        "attention_mask": attention_mask,
                        "use_cache": True,
                        "return_dict": True,
                        "input_ids": full if past is None else full[:, -1:],
                    }
                    if past is not None:
                        kwargs["past_key_values"] = past
                    output = self.model(**kwargs)

                self._record_actual_attention(
                    trace_start=trace_start,
                    ids=ids,
                    prompt_length=prompt_length,
                    destination=attention_traces,
                )
                next_token = self.decoding.next_token(
                    token_ids=ids, logits=output.logits
                )
                ids.append(next_token)
                past = output.past_key_values
                # Legacy V3/V3.4 discover an answer marker at the beginning of
                # the following decode step.  V3.5 keeps that generation
                # behavior, but also audits the just-appended token so a marker
                # completed by EOS or the final token budget is not missed.
                if self._is_v35 and not answer_marker_seen:
                    final_completion_text = self.tokenizer.decode(
                        ids[prompt_length:], skip_special_tokens=False
                    )
                    if _ANSWER_MARKER_RE.search(final_completion_text):
                        answer_marker_seen = True
                        state = "CLOSED"
                if self.decoding.is_eos(next_token):
                    state = "CLOSED"
                    break
        finally:
            self.controller.deactivate()

        if state not in self.VALID_STATES:
            raise RuntimeError("V3 gate ended in an unknown state")
        return V3GenerationResult(
            completion_token_ids=tuple(ids[prompt_length:]),
            boundary_traces=tuple(boundary_traces),
            retrieval_attempts=tuple(attempts),
            memory_transitions=tuple(transitions),
            attention_traces=tuple(attention_traces),
            final_gate_state=state,
            final_memory_id=(
                current_memory.memory_id if current_memory is not None else None
            ),
            answer_marker_seen=answer_marker_seen,
            query_embeddings=tuple(query_embeddings),
            schema_version=(
                V35_GENERATION_RESULT_SCHEMA
                if self._is_v35
                else (
                    V34_GENERATION_RESULT_SCHEMA
                    if self.profile.schema_version
                    == V34_SYSTEM_PROFILE_SCHEMA
                    else V3_GENERATION_RESULT_SCHEMA
                )
            ),
            static_selector_trace=static_selector_trace,
        )
