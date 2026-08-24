"""End-to-end online orchestration for experience-backed side-KV memory."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Sequence

import torch

from memgen.experience.e1 import GateObservation, MemoryChoice
from memgen.experience.system import (
    ExperienceMemorySystemProfile,
    SemanticMemoryRetriever,
    SemanticRetrievalDecision,
)
from memgen.model.e1_runtime import EntropyRiskGate, clone_cache, logits_kl
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
class GateCandidateTrace:
    """One pre-answer delimiter inspected by the entropy-risk gate."""

    generated_boundary_index: int
    boundary_token_id: int
    entropy: float
    entropy_threshold: float
    persistence_risk_score: float
    persistence_risk_threshold: float
    triggered: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperienceMemoryGenerationResult:
    """Complete online generation plus gate, retrieval, and side-KV evidence."""

    completion_token_ids: tuple[int, ...]
    gate_candidates: tuple[GateCandidateTrace, ...]
    gate_observation: GateObservation | None
    retrieval_decision: SemanticRetrievalDecision | None
    matched_memory: MemoryChoice | None
    attention_traces: tuple[SideKVAttentionTrace, ...]
    activation_first_step_logits_kl: float | None
    activation_first_step_top1_changed: bool | None
    activation_baseline_first_token_id: int | None

    @property
    def triggered(self) -> bool:
        return self.gate_observation is not None

    @property
    def side_kv_applied(self) -> bool:
        return self.matched_memory is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "completion_token_ids": list(self.completion_token_ids),
            "gate_candidates": [item.to_dict() for item in self.gate_candidates],
            "gate_observation": (
                self.gate_observation.to_dict()
                if self.gate_observation is not None
                else None
            ),
            "retrieval_decision": (
                self.retrieval_decision.to_dict()
                if self.retrieval_decision is not None
                else None
            ),
            "matched_memory": (
                self.matched_memory.to_dict()
                if self.matched_memory is not None
                else None
            ),
            "side_kv_applied": self.side_kv_applied,
            "attention_traces": [trace.to_dict() for trace in self.attention_traces],
            "activation_first_step_logits_kl": (
                self.activation_first_step_logits_kl
            ),
            "activation_first_step_top1_changed": (
                self.activation_first_step_top1_changed
            ),
            "activation_baseline_first_token_id": (
                self.activation_baseline_first_token_id
            ),
        }


class OnlineExperienceMemorySystem:
    """Generate once with online gate, retrieval, and persistent side-KV.

    The first pre-answer delimiter satisfying both frozen gate thresholds
    consumes the single retrieval opportunity.  A selected memory is evaluated
    at that same boundary and remains visible through EOS.  Memory K/V never
    enter the Hugging Face native cache.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str,
        max_new_tokens: int,
        gate: EntropyRiskGate,
        retriever: SemanticMemoryRetriever,
        loader: SideKVBankLoader,
        controller: SideKVAttentionController,
        profile: ExperienceMemorySystemProfile,
    ):
        if max_new_tokens <= 0:
            raise ValueError("Online experience system needs a positive token budget")
        if gate.config.layer_number != profile.layer_number:
            raise ValueError("Gate and system profile layers differ")
        if controller.layer_number != profile.layer_number:
            raise ValueError("Side-KV controller and system profile layers differ")
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
            raise ValueError("Side-KV controller and system strength profile differ")
        if retriever.profile != profile:
            raise ValueError("Retriever and online system profiles differ")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.gate = gate
        self.retriever = retriever
        self.loader = loader
        self.controller = controller
        self.profile = profile

    def _tensor(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    def _is_delimiter(self, token_id: int) -> bool:
        text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        return text.rstrip(" \t").endswith((",", ".", "\n"))

    def _load_selected_memory(self, choice: MemoryChoice) -> SideKVMemory:
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
            raise ValueError("Retrieved text and side-KV memory metadata differ")
        return memory

    @torch.inference_mode()
    def generate(
        self,
        *,
        question: str,
        prompt_token_ids: Sequence[int],
    ) -> ExperienceMemoryGenerationResult:
        ids = list(prompt_token_ids)
        if not ids:
            raise ValueError("Online experience system requires a non-empty prompt")
        prompt_length = len(ids)
        past = None
        eos = self.tokenizer.eos_token_id
        gate_candidates: list[GateCandidateTrace] = []
        selected_gate: GateObservation | None = None
        retrieval_decision: SemanticRetrievalDecision | None = None
        retrieval_attempted = False

        for generation_step in range(self.max_new_tokens):
            full = self._tensor(ids)
            attention_mask = torch.ones_like(full)
            probe = None
            completion_text = self.tokenizer.decode(
                ids[prompt_length:], skip_special_tokens=False
            )
            can_observe = (
                not retrieval_attempted
                and not _ANSWER_MARKER_RE.search(completion_text)
            )
            if (
                generation_step > 0
                and can_observe
                and self._is_delimiter(ids[-1])
            ):
                probe = self.gate.probe(
                    model=self.model,
                    boundary_token=full[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past,
                )
                triggered = self.gate.triggered(probe)
                boundary_index = len(ids) - prompt_length - 1
                gate_candidates.append(GateCandidateTrace(
                    generated_boundary_index=boundary_index,
                    boundary_token_id=int(ids[-1]),
                    entropy=probe.entropy,
                    entropy_threshold=self.gate.config.entropy_threshold,
                    persistence_risk_score=probe.risk_score,
                    persistence_risk_threshold=self.gate.config.risk_threshold,
                    triggered=triggered,
                ))
                if triggered:
                    retrieval_attempted = True
                    selected_gate = GateObservation(
                        generated_boundary_index=boundary_index,
                        boundary_token_id=int(ids[-1]),
                        entropy=probe.entropy,
                        entropy_threshold=self.gate.config.entropy_threshold,
                        persistence_risk_score=probe.risk_score,
                        persistence_risk_threshold=self.gate.config.risk_threshold,
                    )
                    partial_cot_ids = ids[prompt_length:]
                    retrieval_decision = self.retriever.retrieve(
                        question=question,
                        partial_cot_token_ids=partial_cot_ids,
                    )
                    if retrieval_decision.selected:
                        assert retrieval_decision.matched_memory is not None
                        memory = self._load_selected_memory(
                            retrieval_decision.matched_memory
                        )
                        baseline_logits = probe.output.logits[:, -1, :]
                        self.controller.clear_traces()
                        with self.controller.use_memory(memory):
                            treatment = self.model(
                                input_ids=full[:, -1:],
                                attention_mask=attention_mask,
                                past_key_values=clone_cache(past),
                                use_cache=True,
                                return_dict=True,
                            )
                            return self._finish_with_active_memory(
                                ids=ids,
                                prompt_length=prompt_length,
                                initial_output=treatment,
                                baseline_logits=baseline_logits,
                                gate_candidates=gate_candidates,
                                gate_observation=selected_gate,
                                retrieval_decision=retrieval_decision,
                                memory=memory,
                            )

            if probe is not None:
                output = probe.output
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
            next_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
            ids.append(next_token)
            past = output.past_key_values
            if eos is not None and next_token == eos:
                break

        return ExperienceMemoryGenerationResult(
            completion_token_ids=tuple(ids[prompt_length:]),
            gate_candidates=tuple(gate_candidates),
            gate_observation=selected_gate,
            retrieval_decision=retrieval_decision,
            matched_memory=None,
            attention_traces=(),
            activation_first_step_logits_kl=None,
            activation_first_step_top1_changed=None,
            activation_baseline_first_token_id=None,
        )

    def _finish_with_active_memory(
        self,
        *,
        ids: list[int],
        prompt_length: int,
        initial_output: Any,
        baseline_logits: torch.Tensor,
        gate_candidates: Sequence[GateCandidateTrace],
        gate_observation: GateObservation,
        retrieval_decision: SemanticRetrievalDecision,
        memory: SideKVMemory,
    ) -> ExperienceMemoryGenerationResult:
        """Finish decoding while ``controller.use_memory`` remains active."""

        treatment_logits = initial_output.logits[:, -1, :]
        next_token = int(treatment_logits.argmax(dim=-1).item())
        ids.append(next_token)
        past = initial_output.past_key_values
        eos = self.tokenizer.eos_token_id
        while len(ids) - prompt_length < self.max_new_tokens:
            if eos is not None and ids[-1] == eos:
                break
            output = self.model(
                input_ids=self._tensor([ids[-1]]),
                attention_mask=torch.ones(
                    (1, len(ids)), dtype=torch.long, device=self.device
                ),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            next_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
            ids.append(next_token)
            past = output.past_key_values

        traces = self.controller.traces
        activation_prefix_length = prompt_length + gate_observation.generated_boundary_index + 1
        generated_after_activation = (
            len(ids) - prompt_length - gate_observation.generated_boundary_index - 1
        )
        expected_native_lengths = tuple(
            activation_prefix_length + index
            for index in range(generated_after_activation)
        )
        if len(traces) != generated_after_activation:
            raise RuntimeError("Persistent online side-KV trace count drifted")
        if tuple(trace.native_key_length for trace in traces) != expected_native_lengths:
            raise RuntimeError("Persistent online native cache length drifted")
        if any(trace.memory_id != memory.memory_id for trace in traces):
            raise RuntimeError("Persistent online memory ID changed during decoding")
        if any(
            trace.memory_slot_count != memory.valid_slot_count
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
            for trace in traces
        ):
            raise RuntimeError("Persistent online side-KV trace metadata drifted")
        return ExperienceMemoryGenerationResult(
            completion_token_ids=tuple(ids[prompt_length:]),
            gate_candidates=tuple(gate_candidates),
            gate_observation=gate_observation,
            retrieval_decision=retrieval_decision,
            matched_memory=retrieval_decision.matched_memory,
            attention_traces=traces,
            activation_first_step_logits_kl=logits_kl(
                baseline_logits, treatment_logits
            ),
            activation_first_step_top1_changed=bool(
                baseline_logits.argmax(dim=-1).item()
                != treatment_logits.argmax(dim=-1).item()
            ),
            activation_baseline_first_token_id=int(
                baseline_logits.argmax(dim=-1).item()
            ),
        )
