"""Executable one-stage, gate-scoped online runtime for MemGen V4.

The V3.4 entropy/risk gate and canonical side-KV attention mechanism are
reused without modifying their frozen implementation files.  V4 replaces the
old retrieval and persistent-to-EOS policy with a one-stage source-state
selector and a bounded target-memory episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Mapping, Sequence

import torch

from memgen.experience.phase1 import canonical_json_sha256
from memgen.model.e1_runtime import GreedyDecodingPolicy, logits_kl
from memgen.model.side_kv import SideKVAttentionController, SideKVAttentionTrace
from memgen.model.v3_runtime import EntropyHysteresisGate
from memgen.model.v4_runtime import (
    V4EpisodeTransition,
    V4MemoryEpisodeController,
)
from memgen.model.v4_selector import (
    V4RepairSelector,
    V4SelectionDecision,
    pool_v4_local_reasoning_query,
    v4_tensor_sha256,
)
from memgen.model.v4_side_kv import (
    V4_MEMORY_SCORE_BIAS,
    V4_MEMORY_SCORE_NORMALIZATION,
    V4SideKVBankLoader,
)


V4_ONLINE_PROFILE_SCHEMA = "memgen-v4-online-profile-v1"
V4_GATE_TRACE_SCHEMA = "memgen-v4-gate-trace-v1"
V4_ATTEMPT_TRACE_SCHEMA = "memgen-v4-selector-attempt-trace-v1"
V4_ATTENTION_TRACE_SCHEMA = "memgen-v4-attention-step-trace-v1"
V4_GENERATION_RESULT_SCHEMA = "memgen-v4-generation-result-v1"

_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)", re.IGNORECASE
)


@dataclass(frozen=True)
class V4OnlineProfile:
    layer_number: int = 24
    hidden_state_tuple_index: int = 24
    query_pooling: str = "layer24_local_reasoning_window_mean_16"
    selector_structure: str = "one_stage_state_to_bank"
    gate_source: str = "qualified_v3_4_entropy_risk_joint_gate"
    memory_role_online: str = "target_only"
    reference_online_injectable: bool = False
    auxiliary_banks_materialized: bool = False
    memory_score_normalization: str = V4_MEMORY_SCORE_NORMALIZATION
    memory_score_bias: float = V4_MEMORY_SCORE_BIAS
    schema_version: str = V4_ONLINE_PROFILE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != V4_ONLINE_PROFILE_SCHEMA:
            raise ValueError("Unexpected V4 online profile schema")
        if self.layer_number != 24 or self.hidden_state_tuple_index != 24:
            raise ValueError("V4 initial online runtime is frozen at layer 24")
        if self.query_pooling != "layer24_local_reasoning_window_mean_16":
            raise ValueError("V4 initial online query pooling is frozen to local16")
        if self.selector_structure != "one_stage_state_to_bank":
            raise ValueError("V4 online selector must remain one-stage")
        if self.gate_source != "qualified_v3_4_entropy_risk_joint_gate":
            raise ValueError("V4 online gate provenance drifted")
        if self.memory_role_online != "target_only":
            raise ValueError("V4 online runtime may inject target memory only")
        if self.reference_online_injectable is not False:
            raise ValueError("V4 reference memory cannot be injected online")
        if self.auxiliary_banks_materialized is not False:
            raise ValueError("V4 auxiliary bank is schema-only in the initial system")
        if self.memory_score_normalization != "log_valid_slots":
            raise ValueError("V4 memory slot normalization drifted")
        if not math.isclose(
            self.memory_score_bias,
            V4_MEMORY_SCORE_BIAS,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("V4 memory total prior drifted")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4GateTrace:
    generated_input_index: int
    token_id: int
    token_text: str
    state_before: str
    state_after: str
    active_bank_before: str | None
    active_bank_after: str | None
    predecision_entropy: float
    predecision_risk: float
    actual_path_entropy: float
    actual_path_risk: float
    high_entropy_threshold: float
    low_entropy_threshold: float
    risk_threshold: float
    joint_trigger_qualified: bool
    action: str
    selector_attempt_count_after: int
    active_step_count_after: int
    low_entropy_streak_after: int
    episode_transition: Mapping[str, Any] | None
    schema_version: str = V4_GATE_TRACE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V4SelectorAttemptTrace:
    attempt_number: int
    generated_input_index: int
    query_embedding_sha256: str
    query_embedding_norm: float
    query_window_token_count: int
    decision: V4SelectionDecision
    episode_transition: V4EpisodeTransition
    activation_first_step_logits_kl: float | None
    activation_first_step_top1_changed: bool | None
    activation_baseline_first_token_id: int | None
    activation_target_first_token_id: int | None
    schema_version: str = V4_ATTEMPT_TRACE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "decision": self.decision.to_dict(),
            "episode_transition": self.episode_transition.to_dict(),
        }


@dataclass(frozen=True)
class V4AttentionStepTrace:
    generated_input_index: int
    processed_prefix_token_count: int
    trace: SideKVAttentionTrace
    schema_version: str = V4_ATTENTION_TRACE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_input_index": self.generated_input_index,
            "processed_prefix_token_count": self.processed_prefix_token_count,
            "trace": self.trace.to_dict(),
        }


@dataclass(frozen=True)
class V4GenerationResult:
    completion_token_ids: tuple[int, ...]
    gate_traces: tuple[V4GateTrace, ...]
    selector_attempts: tuple[V4SelectorAttemptTrace, ...]
    attention_traces: tuple[V4AttentionStepTrace, ...]
    lifecycle_summary: Mapping[str, Any]
    answer_marker_seen: bool
    final_state: str
    profile: V4OnlineProfile
    query_embeddings: tuple[torch.Tensor, ...] = field(
        default_factory=tuple, repr=False, compare=False
    )
    schema_version: str = V4_GENERATION_RESULT_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        activation_ids = [
            attempt.decision.selected_bank_id
            for attempt in self.selector_attempts
            if attempt.decision.selected_bank_id is not None
        ]
        active_counts: dict[str, int] = {}
        for trace in self.attention_traces:
            active_counts[trace.trace.memory_id] = (
                active_counts.get(trace.trace.memory_id, 0) + 1
            )
        return {
            "schema_version": self.schema_version,
            "completion_token_ids": list(self.completion_token_ids),
            "completion_token_ids_sha256": canonical_json_sha256(
                list(self.completion_token_ids)
            ),
            "generated_token_count": len(self.completion_token_ids),
            "gate_traces": [item.to_dict() for item in self.gate_traces],
            "selector_attempts": [
                item.to_dict() for item in self.selector_attempts
            ],
            "attention_traces": [item.to_dict() for item in self.attention_traces],
            "lifecycle_summary": dict(self.lifecycle_summary),
            "answer_marker_seen": self.answer_marker_seen,
            "final_state": self.final_state,
            "profile": self.profile.to_dict(),
            "summary": {
                "selector_attempt_count": len(self.selector_attempts),
                "selection_count": len(activation_ids),
                "abstain_count": sum(
                    attempt.decision.selected_bank_id is None
                    for attempt in self.selector_attempts
                ),
                "selected_bank_ids": activation_ids,
                "attention_step_count": len(self.attention_traces),
                "attention_steps_by_bank": active_counts,
                "positive_attention_mass_every_active_step": all(
                    trace.trace.memory_attention_mass > 0.0
                    for trace in self.attention_traces
                ),
                "max_three_attempts_respected": len(self.selector_attempts) <= 3,
                "target_only_memory_ids": all(
                    "::reference" not in trace.trace.memory_id
                    for trace in self.attention_traces
                ),
            },
        }


class OnlineExperienceMemorySystemV4:
    """Greedy generation with state routing and bounded target-memory episodes."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str,
        max_new_tokens: int,
        gate: EntropyHysteresisGate,
        selector: V4RepairSelector,
        loader: V4SideKVBankLoader,
        controller: SideKVAttentionController,
        profile: V4OnlineProfile | None = None,
    ) -> None:
        profile = profile or V4OnlineProfile()
        if max_new_tokens <= 0:
            raise ValueError("V4 online generation needs a positive token budget")
        if gate.config.layer_number != 24 or gate.config.risk_role != "online_joint_control":
            raise ValueError("V4 requires the qualified layer-24 joint gate")
        if gate.config.rearm_low_entropy_token_count != 2:
            raise ValueError("V4 gate recovery hysteresis must use two low tokens")
        if selector.config.layer_number != 24:
            raise ValueError("V4 selector layer differs from runtime")
        selector_ids = tuple(bank.bank_id for bank in selector.banks)
        if not set(selector_ids).issubset(set(loader.bank_ids)):
            raise ValueError("V4 selector contains a bank absent from side-KV")
        if controller.layer_number != 24:
            raise ValueError("V4 side-KV controller layer differs from runtime")
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
            raise ValueError("V4 side-KV strength differs from the online profile")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.gate = gate
        self.selector = selector
        self.loader = loader
        self.controller = controller
        self.profile = profile
        self.decoding = GreedyDecodingPolicy(tokenizer=tokenizer, device=device)

    def _tensor(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

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
        actual = cls._cache_sequence_length(cache)
        if actual == expected_length:
            return cache
        crop = getattr(cache, "crop", None)
        if actual is None or actual < expected_length or not callable(crop):
            raise RuntimeError("Unable to restore V4 live cache after native probe")
        crop(expected_length)
        if cls._cache_sequence_length(cache) != expected_length:
            raise RuntimeError("V4 live cache rollback length drifted")
        return cache

    def _layer_state(self, output: Any) -> torch.Tensor:
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) <= 24:
            raise RuntimeError("V4 runtime output has no layer-24 state")
        state = hidden_states[24][0, -1, :].detach().float()
        if not torch.isfinite(state).all():
            raise RuntimeError("V4 runtime layer-24 state is non-finite")
        return state

    def _load_target(self, bank_id: str) -> Any:
        return self.loader.get_target(
            bank_id,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )

    def _new_attention_traces(
        self,
        *,
        trace_start: int,
        generated_input_index: int,
        prefix_length: int,
        expected_memory_id: str | None,
    ) -> list[V4AttentionStepTrace]:
        traces = self.controller.traces[trace_start:]
        if expected_memory_id is None:
            if traces:
                raise RuntimeError("V4 inactive side-KV path emitted attention traces")
            return []
        if len(traces) != 1:
            raise RuntimeError("V4 expects exactly one side-KV trace per active step")
        trace = traces[0]
        if (
            trace.memory_id != expected_memory_id
            or trace.memory_attention_mass <= 0.0
            or not math.isfinite(trace.memory_attention_mass)
            or trace.memory_score_normalization != V4_MEMORY_SCORE_NORMALIZATION
            or not math.isclose(
                trace.memory_score_bias,
                V4_MEMORY_SCORE_BIAS,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise RuntimeError("V4 active attention trace failed authentication")
        return [
            V4AttentionStepTrace(
                generated_input_index=generated_input_index,
                processed_prefix_token_count=prefix_length,
                trace=trace,
            )
        ]

    @torch.inference_mode()
    def generate(self, *, prompt_token_ids: Sequence[int]) -> V4GenerationResult:
        ids = [int(value) for value in prompt_token_ids]
        if not ids:
            raise ValueError("V4 online generation requires a non-empty prompt")
        prompt_length = len(ids)
        past = None
        episode = V4MemoryEpisodeController()
        reasoning_states: list[torch.Tensor] = []
        gate_traces: list[V4GateTrace] = []
        attempts: list[V4SelectorAttemptTrace] = []
        attention_traces: list[V4AttentionStepTrace] = []
        query_embeddings: list[torch.Tensor] = []
        answer_marker_seen = False
        self.controller.deactivate()
        self.controller.clear_traces()
        try:
            for generation_step in range(self.max_new_tokens):
                completion_text = self.tokenizer.decode(
                    ids[prompt_length:], skip_special_tokens=False
                )
                if not answer_marker_seen and _ANSWER_MARKER_RE.search(completion_text):
                    answer_marker_seen = True
                    episode.close(reason="answer_marker")
                    self.controller.deactivate()

                full = self._tensor(ids)
                attention_mask = torch.ones_like(full)
                should_probe = generation_step > 0 and episode.state in {
                    "ARMED",
                    "ACTIVE",
                    "COOLDOWN",
                }
                if not should_probe:
                    kwargs: dict[str, Any] = {
                        "input_ids": full if past is None else full[:, -1:],
                        "attention_mask": attention_mask,
                        "use_cache": True,
                        "return_dict": True,
                    }
                    if past is not None:
                        kwargs["past_key_values"] = past
                    output = self.model(**kwargs)
                else:
                    generated_input_index = len(ids) - prompt_length - 1
                    state_before = episode.state
                    active_before = episode.active_bank_id
                    trace_start = len(self.controller.traces)
                    cache_length = self._cache_sequence_length(past)
                    if cache_length is None:
                        raise RuntimeError("V4 cannot audit the live cache length")
                    native_probe = self.gate.probe(
                        model=self.model,
                        boundary_token=full[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past,
                        clone_past_key_values=False,
                    )
                    output = native_probe.output
                    actual_probe = native_probe
                    actual_memory_id = active_before
                    reasoning_states.append(self._layer_state(native_probe.output))
                    if len(reasoning_states) > 16:
                        del reasoning_states[:-16]
                    joint = self.gate.trigger_qualified(native_probe)
                    action = "observed"
                    transition: V4EpisodeTransition | None = None

                    if episode.state == "ARMED" and joint:
                        query = pool_v4_local_reasoning_query(
                            torch.stack(reasoning_states, dim=0),
                            reasoning_start_index=0,
                        )
                        query_embeddings.append(query.detach().cpu())
                        decision = self.selector.select(query)
                        transition = episode.apply_selection(decision.selected_bank_id)
                        activation_kl: float | None = None
                        activation_changed: bool | None = None
                        baseline_first: int | None = None
                        target_first: int | None = None
                        if decision.selected_bank_id is not None:
                            baseline_scores = self.decoding.processed_scores(
                                token_ids=ids, logits=native_probe.output.logits
                            )
                            self.controller.activate(
                                self._load_target(decision.selected_bank_id)
                            )
                            past = self._restore_cache_length(
                                past, expected_length=cache_length
                            )
                            actual_probe = self.gate.probe(
                                model=self.model,
                                boundary_token=full[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=past,
                                clone_past_key_values=False,
                            )
                            output = actual_probe.output
                            actual_memory_id = decision.selected_bank_id
                            reasoning_states[-1] = self._layer_state(output)
                            target_scores = self.decoding.processed_scores(
                                token_ids=ids, logits=output.logits
                            )
                            activation_kl = logits_kl(baseline_scores, target_scores)
                            if not math.isfinite(activation_kl):
                                raise RuntimeError("V4 activation logits KL is non-finite")
                            baseline_first = int(baseline_scores.argmax(dim=-1).item())
                            target_first = int(target_scores.argmax(dim=-1).item())
                            activation_changed = baseline_first != target_first
                            active_transition = episode.observe_decoded_token(
                                low_entropy=(
                                    actual_probe.entropy
                                    <= self.gate.config.low_entropy_threshold
                                )
                            )
                            if active_transition is not None:
                                transition = active_transition
                                if active_transition.deactivate_memory:
                                    self.controller.deactivate()
                            action = "selected_target"
                        else:
                            action = "selector_abstained"
                        attempts.append(
                            V4SelectorAttemptTrace(
                                attempt_number=episode.attempt_count,
                                generated_input_index=generated_input_index,
                                query_embedding_sha256=v4_tensor_sha256(query),
                                query_embedding_norm=float(query.float().norm().item()),
                                query_window_token_count=len(reasoning_states),
                                decision=decision,
                                episode_transition=transition,
                                activation_first_step_logits_kl=activation_kl,
                                activation_first_step_top1_changed=activation_changed,
                                activation_baseline_first_token_id=baseline_first,
                                activation_target_first_token_id=target_first,
                            )
                        )
                    elif episode.state in {"ACTIVE", "COOLDOWN"}:
                        transition = episode.observe_decoded_token(
                            low_entropy=(
                                actual_probe.entropy
                                <= self.gate.config.low_entropy_threshold
                            )
                        )
                        if transition is not None:
                            action = transition.event
                            if transition.deactivate_memory:
                                self.controller.deactivate()
                        elif episode.state == "ACTIVE":
                            action = "active_target"
                        else:
                            action = "cooldown"
                    elif episode.state == "ARMED":
                        action = "armed_below_joint_gate"

                    attention_traces.extend(
                        self._new_attention_traces(
                            trace_start=trace_start,
                            generated_input_index=generated_input_index,
                            prefix_length=len(ids),
                            expected_memory_id=actual_memory_id,
                        )
                    )
                    gate_traces.append(
                        V4GateTrace(
                            generated_input_index=generated_input_index,
                            token_id=int(ids[-1]),
                            token_text=self.tokenizer.decode(
                                [int(ids[-1])], skip_special_tokens=False
                            ),
                            state_before=state_before,
                            state_after=episode.state,
                            active_bank_before=active_before,
                            active_bank_after=episode.active_bank_id,
                            predecision_entropy=native_probe.entropy,
                            predecision_risk=native_probe.risk_score,
                            actual_path_entropy=actual_probe.entropy,
                            actual_path_risk=actual_probe.risk_score,
                            high_entropy_threshold=self.gate.config.high_entropy_threshold,
                            low_entropy_threshold=self.gate.config.low_entropy_threshold,
                            risk_threshold=self.gate.config.risk_threshold,
                            joint_trigger_qualified=joint,
                            action=action,
                            selector_attempt_count_after=episode.attempt_count,
                            active_step_count_after=episode.active_step_count,
                            low_entropy_streak_after=episode.low_entropy_streak,
                            episode_transition=(
                                transition.to_dict() if transition is not None else None
                            ),
                        )
                    )

                next_token = self.decoding.next_token(
                    token_ids=ids, logits=output.logits
                )
                ids.append(next_token)
                past = output.past_key_values
                final_text = self.tokenizer.decode(
                    ids[prompt_length:], skip_special_tokens=False
                )
                if not answer_marker_seen and _ANSWER_MARKER_RE.search(final_text):
                    answer_marker_seen = True
                    episode.close(reason="answer_marker")
                    self.controller.deactivate()
                if self.decoding.is_eos(next_token):
                    episode.close(reason="eos")
                    self.controller.deactivate()
                    break
        finally:
            self.controller.deactivate()

        if episode.state != "CLOSED":
            episode.close(reason="max_new_tokens")

        if episode.attempt_count != len(attempts):
            raise RuntimeError("V4 lifecycle and selector attempt counts differ")
        if any(
            "::reference" in trace.trace.memory_id for trace in attention_traces
        ):
            raise RuntimeError("V4 reference memory appeared on the online path")
        return V4GenerationResult(
            completion_token_ids=tuple(ids[prompt_length:]),
            gate_traces=tuple(gate_traces),
            selector_attempts=tuple(attempts),
            attention_traces=tuple(attention_traces),
            lifecycle_summary=episode.summary(),
            answer_marker_seen=answer_marker_seen,
            final_state=episode.state,
            profile=self.profile,
            query_embeddings=tuple(query_embeddings),
        )


__all__ = [
    "OnlineExperienceMemorySystemV4",
    "V4GenerationResult",
    "V4OnlineProfile",
]
