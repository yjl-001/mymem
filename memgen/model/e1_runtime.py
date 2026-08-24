"""Runtime primitives for E1 observation and persistent side-KV branches."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import re
from typing import Any, Sequence

import torch

from memgen.experience.e1 import GateObservation
from memgen.model.side_kv import (
    SideKVAttentionController,
    SideKVAttentionTrace,
    SideKVMemory,
)


_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)",
    re.IGNORECASE,
)


def clone_cache(cache: Any) -> Any:
    """Clone a mutable Hugging Face cache before a counterfactual branch."""

    if cache is None:
        return None
    try:
        return copy.deepcopy(cache)
    except Exception:
        legacy = cache.to_legacy_cache()
        cloned = tuple(
            tuple(value.clone() for value in layer) for layer in legacy
        )
        constructor = getattr(type(cache), "from_legacy_cache", None)
        if not callable(constructor):
            raise RuntimeError("Unable to clone the native KV cache")
        return constructor(cloned)


def logits_kl(reference: torch.Tensor, treatment: torch.Tensor) -> float:
    reference_log_probs = torch.log_softmax(reference.float(), dim=-1)
    treatment_log_probs = torch.log_softmax(treatment.float(), dim=-1)
    reference_probs = reference_log_probs.exp()
    value = (
        reference_probs * (reference_log_probs - treatment_log_probs)
    ).sum(dim=-1)
    return float(value.mean().item())


@dataclass(frozen=True)
class EntropyRiskGateConfig:
    layer_number: int
    sink_token_count: int
    entropy_threshold: float
    risk_threshold: float

    def __post_init__(self) -> None:
        if self.layer_number <= 0 or self.sink_token_count < 0:
            raise ValueError("Invalid entropy-risk layer or sink count")
        if not math.isfinite(self.entropy_threshold) or not math.isfinite(
            self.risk_threshold
        ):
            raise ValueError("Entropy-risk thresholds must be finite")


@dataclass(frozen=True)
class GateProbe:
    entropy: float
    risk_score: float
    output: Any


class EntropyRiskGate:
    """Frozen high-entropy plus persistence-risk observation policy."""

    def __init__(
        self,
        *,
        recovery_center: torch.Tensor,
        persistence_center: torch.Tensor,
        config: EntropyRiskGateConfig,
    ):
        if recovery_center.ndim != 1 or persistence_center.shape != recovery_center.shape:
            raise ValueError("Risk centers must be equal-width vectors")
        if not torch.isfinite(recovery_center).all() or not torch.isfinite(
            persistence_center
        ).all():
            raise ValueError("Risk centers contain non-finite values")
        self.recovery_center = recovery_center.detach().float().cpu()
        self.persistence_center = persistence_center.detach().float().cpu()
        self.config = config

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> "EntropyRiskGate":
        construction = artifact.get("construction", {})
        risk = artifact.get("risk_gate", {})
        required = (
            "layer",
            "recovery_center",
            "persistence_center",
            "threshold",
        )
        missing = [name for name in required if name not in risk]
        if missing:
            raise ValueError(f"Risk artifact is missing fields: {missing}")
        entropy_threshold = construction.get("high_entropy_threshold")
        sink_count = construction.get("sink_token_count")
        if not isinstance(entropy_threshold, (int, float)) or not isinstance(
            sink_count, int
        ):
            raise ValueError("Risk artifact has no frozen entropy policy")
        return cls(
            recovery_center=risk["recovery_center"],
            persistence_center=risk["persistence_center"],
            config=EntropyRiskGateConfig(
                layer_number=int(risk["layer"]),
                sink_token_count=sink_count,
                entropy_threshold=float(entropy_threshold),
                risk_threshold=float(risk["threshold"]),
            ),
        )

    @property
    def config_dict(self) -> dict[str, Any]:
        return {
            "layer_number": self.config.layer_number,
            "sink_token_count": self.config.sink_token_count,
            "entropy_threshold": self.config.entropy_threshold,
            "risk_threshold": self.config.risk_threshold,
            "selection_policy": "first_joint_entropy_and_risk_boundary",
        }

    def triggered(self, probe: GateProbe) -> bool:
        return bool(
            probe.entropy >= self.config.entropy_threshold
            and probe.risk_score > self.config.risk_threshold
        )

    @torch.inference_mode()
    def probe(
        self,
        *,
        model: Any,
        boundary_token: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Any,
    ) -> GateProbe:
        output = model(
            input_ids=boundary_token,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            past_key_values=clone_cache(past_key_values),
            use_cache=True,
            return_dict=True,
        )
        attentions = output.attentions
        if not attentions or attentions[-1] is None:
            raise RuntimeError("Entropy-risk gate requires eager attention weights")
        valid_positions = attention_mask[0].nonzero(as_tuple=True)[0]
        keys = valid_positions[self.config.sink_token_count :]
        raw = attentions[-1][0, :, -1, :].float()
        if raw.shape[-1] != valid_positions.numel():
            raise RuntimeError("Attention key length differs from the live prefix")
        if keys.numel() == 0:
            raise RuntimeError("Sink mask removed every attention key")
        probabilities = raw.index_select(1, keys)
        normalizer = probabilities.sum(dim=-1, keepdim=True)
        if torch.any(normalizer <= 0):
            raise RuntimeError("Attention mass after sink masking is zero")
        probabilities = probabilities / normalizer
        entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
        ).sum(dim=-1)
        entropy_value = float(entropy.mean().item())
        hidden_states = output.hidden_states
        if hidden_states is None or self.config.layer_number >= len(hidden_states):
            raise RuntimeError("Risk-gate hidden state layer is unavailable")
        state = hidden_states[self.config.layer_number][0, -1, :].detach().float()
        recovery = self.recovery_center.to(device=state.device)
        persistence = self.persistence_center.to(device=state.device)
        recovery_similarity = torch.nn.functional.cosine_similarity(
            state, recovery, dim=0
        )
        persistence_similarity = torch.nn.functional.cosine_similarity(
            state, persistence, dim=0
        )
        risk_score = float((persistence_similarity - recovery_similarity).item())
        if not math.isfinite(entropy_value) or not math.isfinite(risk_score):
            raise RuntimeError("Entropy-risk observation is non-finite")
        return GateProbe(
            entropy=entropy_value,
            risk_score=risk_score,
            output=output,
        )


@dataclass(frozen=True)
class ObservationRolloutResult:
    completion_token_ids: tuple[int, ...]
    gate_observation: GateObservation | None
    prefix_token_ids: tuple[int, ...]
    candidate_boundary_count: int


@dataclass(frozen=True)
class TriggeredPersistentMemoryGenerationResult:
    """Frozen-prefix continuation with memory visible from trigger to EOS."""

    completion_token_ids: tuple[int, ...]
    attention_traces: tuple[SideKVAttentionTrace, ...]
    first_step_logits_kl: float
    first_step_top1_changed: bool
    baseline_first_token_id: int


class GreedyE1Runtime:
    """Deterministic batch-one generation used by every E1 condition."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str,
        max_new_tokens: int,
    ):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens

    def _tensor(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    def _is_delimiter(self, token_id: int) -> bool:
        text = self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        return text.rstrip(" \t").endswith((",", ".", "\n"))

    @torch.inference_mode()
    def generate_vanilla(self, prompt_token_ids: Sequence[int]) -> tuple[int, ...]:
        ids = list(prompt_token_ids)
        prompt_length = len(ids)
        past = None
        eos = self.tokenizer.eos_token_id
        for _ in range(self.max_new_tokens):
            full = self._tensor(ids)
            kwargs: dict[str, Any] = {
                "attention_mask": torch.ones_like(full),
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
        return tuple(ids[prompt_length:])

    @torch.inference_mode()
    def generate_observation_only(
        self,
        *,
        prompt_token_ids: Sequence[int],
        gate: EntropyRiskGate,
    ) -> ObservationRolloutResult:
        ids = list(prompt_token_ids)
        prompt_length = len(ids)
        past = None
        eos = self.tokenizer.eos_token_id
        selected: GateObservation | None = None
        selected_prefix: tuple[int, ...] = ()
        candidate_count = 0
        for generation_step in range(self.max_new_tokens):
            full = self._tensor(ids)
            attention_mask = torch.ones_like(full)
            probe: GateProbe | None = None
            completion_text = self.tokenizer.decode(
                ids[prompt_length:], skip_special_tokens=False
            )
            can_observe = selected is None and not _ANSWER_MARKER_RE.search(
                completion_text
            )
            if (
                generation_step > 0
                and can_observe
                and self._is_delimiter(ids[-1])
            ):
                candidate_count += 1
                probe = gate.probe(
                    model=self.model,
                    boundary_token=full[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past,
                )
                if gate.triggered(probe):
                    boundary_index = len(ids) - prompt_length - 1
                    selected = GateObservation(
                        generated_boundary_index=boundary_index,
                        boundary_token_id=int(ids[-1]),
                        entropy=probe.entropy,
                        entropy_threshold=gate.config.entropy_threshold,
                        persistence_risk_score=probe.risk_score,
                        persistence_risk_threshold=gate.config.risk_threshold,
                    )
                    selected_prefix = tuple(ids)

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
        return ObservationRolloutResult(
            completion_token_ids=tuple(ids[prompt_length:]),
            gate_observation=selected,
            prefix_token_ids=selected_prefix,
            candidate_boundary_count=candidate_count,
        )

    @torch.inference_mode()
    def generate_from_trigger_with_persistent_memory(
        self,
        *,
        prefix_token_ids: Sequence[int],
        prompt_token_count: int,
        memory: SideKVMemory,
        controller: SideKVAttentionController,
    ) -> TriggeredPersistentMemoryGenerationResult:
        """Replay a frozen trigger prefix and keep memory active through EOS."""

        prefix = list(prefix_token_ids)
        if prompt_token_count <= 0 or len(prefix) <= prompt_token_count:
            raise ValueError("Treatment prefix must contain partial reasoning")
        partial_length = len(prefix) - prompt_token_count
        if partial_length >= self.max_new_tokens:
            raise ValueError("Treatment boundary leaves no generation budget")
        before_boundary = self._tensor(prefix[:-1])
        boundary = self._tensor([prefix[-1]])
        prefill_mask = torch.ones_like(before_boundary)
        full_mask = torch.ones(
            (1, len(prefix)), dtype=prefill_mask.dtype, device=self.device
        )
        prefill = self.model(
            input_ids=before_boundary,
            attention_mask=prefill_mask,
            use_cache=True,
            return_dict=True,
        )
        baseline = self.model(
            input_ids=boundary,
            attention_mask=full_mask,
            past_key_values=clone_cache(prefill.past_key_values),
            use_cache=True,
            return_dict=True,
        )
        controller.clear_traces()
        with controller.use_memory(memory):
            treatment = self.model(
                input_ids=boundary,
                attention_mask=full_mask,
                past_key_values=clone_cache(prefill.past_key_values),
                use_cache=True,
                return_dict=True,
            )
            baseline_logits = baseline.logits[:, -1, :]
            treatment_logits = treatment.logits[:, -1, :]
            first_token = int(treatment_logits.argmax(dim=-1).item())
            ids = prefix + [first_token]
            past = treatment.past_key_values
            eos = self.tokenizer.eos_token_id
            remaining = self.max_new_tokens - partial_length - 1
            if eos is None or first_token != eos:
                for _ in range(remaining):
                    full = self._tensor(ids)
                    output = self.model(
                        input_ids=full[:, -1:],
                        attention_mask=torch.ones_like(full),
                        past_key_values=past,
                        use_cache=True,
                        return_dict=True,
                    )
                    next_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
                    ids.append(next_token)
                    past = output.past_key_values
                    if eos is not None and next_token == eos:
                        break

        traces = controller.traces
        generated_after_trigger = len(ids) - len(prefix)
        expected_native_lengths = tuple(
            len(prefix) + index for index in range(generated_after_trigger)
        )
        if len(traces) != generated_after_trigger:
            raise RuntimeError(
                "Triggered persistent side-KV requires one trace per new token"
            )
        if tuple(trace.native_key_length for trace in traces) != expected_native_lengths:
            raise RuntimeError("Triggered persistent native cache length drifted")
        if any(trace.memory_id != memory.memory_id for trace in traces):
            raise RuntimeError("Triggered persistent memory ID changed during generation")
        return TriggeredPersistentMemoryGenerationResult(
            completion_token_ids=tuple(ids[prompt_token_count:]),
            attention_traces=traces,
            first_step_logits_kl=logits_kl(baseline_logits, treatment_logits),
            first_step_top1_changed=bool(
                baseline_logits.argmax(dim=-1).item()
                != treatment_logits.argmax(dim=-1).item()
            ),
            baseline_first_token_id=int(baseline_logits.argmax(dim=-1).item()),
        )
