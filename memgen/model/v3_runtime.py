"""Online V3 generation: entropy re-arm, embedding retrieval, KV replacement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
import time
from typing import Any, Mapping, Sequence

import torch

from memgen.experience.phase1 import canonical_json_sha256
from memgen.experience.v3 import (
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


@dataclass(frozen=True)
class EntropyHysteresisConfig:
    layer_number: int
    sink_token_count: int
    high_entropy_threshold: float
    low_entropy_threshold: float
    risk_threshold: float
    risk_role: str = "diagnostic_only"

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
        if self.risk_role != "diagnostic_only":
            raise ValueError("V3 risk score must remain diagnostic-only")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntropyHysteresisGate:
    """Reuse the qualified observer/centers but gate only on entropy."""

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

    def probe(self, **kwargs: Any) -> GateProbe:
        return self.diagnostic_gate.probe(**kwargs)


@dataclass(frozen=True)
class V3BoundaryTrace:
    generated_boundary_index: int
    boundary_token_id: int
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class V3RetrievalAttemptTrace:
    attempt_number: int
    generated_boundary_index: int
    boundary_token_id: int
    outcome: str
    previous_memory_id: str | None
    selected_memory_id: str | None
    active_memory_id_after: str | None
    retrieval_decision: EmbeddingRetrievalDecision
    query_encoding_seconds: float
    retrieval_seconds: float
    memory_load_seconds: float | None
    activation_forward_seconds: float | None
    attempt_total_seconds: float
    activation_first_step_logits_kl: float | None
    activation_first_step_top1_changed: bool | None
    activation_baseline_first_token_id: int | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["retrieval_decision"] = self.retrieval_decision.to_dict()
        return value


@dataclass(frozen=True)
class V3MemoryTransition:
    generated_boundary_index: int
    transition: str
    previous_memory_id: str | None
    next_memory_id: str
    attempt_number: int

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
        return {
            "schema_version": self.schema_version,
            "completion_token_ids": list(self.completion_token_ids),
            "completion_token_ids_sha256": canonical_json_sha256(
                list(self.completion_token_ids)
            ),
            "generated_token_count": self.generated_token_count,
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
            },
        }


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
        retriever: EmbeddingMemoryRetriever,
        loader: SideKVBankLoader,
        controller: SideKVAttentionController,
        profile: ExperienceMemoryV3Profile,
    ):
        if max_new_tokens <= 0:
            raise ValueError("V3 online generation needs a positive token budget")
        if gate.config.layer_number != profile.layer_number:
            raise ValueError("V3 gate and profile layers differ")
        if query_encoder.layer_number != profile.layer_number:
            raise ValueError("V3 query encoder and profile layers differ")
        if controller.layer_number != profile.layer_number:
            raise ValueError("V3 side-KV controller and profile layers differ")
        if retriever.profile != profile:
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

    def _tensor(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    def _is_delimiter(self, token_id: int) -> bool:
        text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        return text.rstrip(" \t").endswith((",", ".", "\n"))

    def _load_selected_memory(
        self, decision: EmbeddingRetrievalDecision
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
        self.controller.deactivate()
        self.controller.clear_traces()
        try:
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
                    generation_step > 0
                    and state != "CLOSED"
                    and self._is_delimiter(ids[-1])
                ):
                    probe = self.gate.probe(
                        model=self.model,
                        boundary_token=full[:, -1:],
                        attention_mask=attention_mask,
                        past_key_values=past,
                    )
                    state_before = state
                    attempts_before = attempt_count
                    memory_before = (
                        current_memory.memory_id if current_memory is not None else None
                    )
                    action = "observed"
                    output = probe.output

                    if state == "ARMED" and (
                        probe.entropy >= self.gate.config.high_entropy_threshold
                    ):
                        attempt_started = time.perf_counter()
                        attempt_count += 1
                        encoding_started = time.perf_counter()
                        with self.controller.suspend_memory():
                            query_embedding = self.query_encoder.encode(ids)
                        query_encoding_seconds = time.perf_counter() - encoding_started
                        query_embeddings.append(query_embedding.detach().float().cpu())
                        retrieval_started = time.perf_counter()
                        decision = self.retriever.retrieve(
                            query_embedding=query_embedding,
                            query_token_ids=ids,
                            prompt_token_count=prompt_length,
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
                        if not decision.selected:
                            outcome = "abstained"
                        elif selected_id == memory_before:
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
                            activation_started = time.perf_counter()
                            treatment = self.model(
                                input_ids=full[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=clone_cache(past),
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
                            ))
                            current_memory = next_memory
                        state = (
                            "EXHAUSTED"
                            if attempt_count >= self.profile.max_retrieval_attempts
                            else "DISARMED"
                        )
                        action = "retrieval_attempt"
                        attempts.append(V3RetrievalAttemptTrace(
                            attempt_number=attempt_count,
                            generated_boundary_index=len(ids) - prompt_length - 1,
                            boundary_token_id=int(ids[-1]),
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
                        ))
                    elif state == "DISARMED" and (
                        probe.entropy <= self.gate.config.low_entropy_threshold
                    ):
                        state = "ARMED"
                        action = "rearmed"
                    elif state == "DISARMED":
                        action = "observed_disarmed"
                    elif state == "EXHAUSTED":
                        action = "observed_exhausted"
                    else:
                        action = "observed_armed_below_high"

                    boundary_traces.append(V3BoundaryTrace(
                        generated_boundary_index=len(ids) - prompt_length - 1,
                        boundary_token_id=int(ids[-1]),
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
        )
