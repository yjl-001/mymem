"""Exact-prefix three-branch runtime for offline V4 causal qualification.

This module is intentionally separate from the online loader/runtime.  It can
materialize a reference role only through an explicitly offline API and uses
the frozen V4 nonpersistent episode policy after the initial oracle-selected
injection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Mapping, Sequence

import torch

from memgen.experience.phase1 import canonical_json_sha256
from memgen.model.e1_runtime import GreedyDecodingPolicy, clone_cache, logits_kl
from memgen.model.side_kv import (
    SideKVAttentionController,
    SideKVAttentionTrace,
    SideKVMemory,
)
from memgen.model.v3_runtime import EntropyHysteresisGate
from memgen.model.v4_runtime import V4MemoryEpisodeController
from memgen.model.v4_side_kv import V4SideKVBankLoader


_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)", re.IGNORECASE
)


@dataclass(frozen=True)
class V4OracleBranchResult:
    role: str
    continuation_token_ids: tuple[int, ...]
    memory_id: str | None
    attention_traces: tuple[SideKVAttentionTrace, ...]
    lifecycle: Mapping[str, Any] | None
    first_step_logits_kl: float
    first_step_top1_changed: bool
    baseline_top1_token_id: int
    branch_top1_token_id: int
    baseline_top1_rank_under_branch: int
    branch_top1_rank_under_baseline: int
    baseline_top1_log_probability_delta: float
    branch_top1_log_probability_delta: float
    initial_cache_length: int
    first_output_cache_length: int
    answer_marker_seen: bool
    eos_seen: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "continuation_token_ids": list(self.continuation_token_ids),
            "continuation_token_ids_sha256": canonical_json_sha256(
                list(self.continuation_token_ids)
            ),
            "attention_traces": [trace.to_dict() for trace in self.attention_traces],
        }


class V4OfflineSideKVRoleBankLoader:
    """Compose the authenticated online loader with offline reference access."""

    def __init__(self, *, manifest_path: Any) -> None:
        self.target_loader = V4SideKVBankLoader(manifest_path=manifest_path)
        self.manifest = self.target_loader.manifest
        self._reference_entry_by_bank_id = {
            str(entry["bank_id"]): entry
            for entry in self.manifest["records"]
            if entry.get("role") == "reference"
        }
        if set(self._reference_entry_by_bank_id) != set(self.target_loader.bank_ids):
            raise ValueError("V4 offline role loader lacks reference coverage")
        if any(
            entry.get("online_injectable") is not False
            for entry in self._reference_entry_by_bank_id.values()
        ):
            raise ValueError("V4 reference role was incorrectly marked online injectable")

    @property
    def bank_ids(self) -> tuple[str, ...]:
        return self.target_loader.bank_ids

    def get_target(
        self, bank_id: str, *, device: torch.device | str, dtype: torch.dtype
    ) -> SideKVMemory:
        return self.target_loader.get_target(bank_id, device=device, dtype=dtype)

    def get_reference_offline(
        self, bank_id: str, *, device: torch.device | str, dtype: torch.dtype
    ) -> SideKVMemory:
        """Materialize the non-online reference role for a causal control only."""

        entry = self._reference_entry_by_bank_id.get(bank_id)
        if entry is None:
            raise KeyError(f"Unknown V4 reference bank ID: {bank_id}")
        if entry.get("role") != "reference" or entry.get("online_injectable") is not False:
            raise ValueError("V4 offline reference role authentication failed")
        index = int(entry["index"])
        slots = int(entry["kv_valid_slot_count"])
        return SideKVMemory(
            memory_id=f"{bank_id}::reference",
            payload_hash=str(entry["payload_hash"]),
            keys=self.target_loader.keys[index, :, :slots, :].to(
                device=device, dtype=dtype
            ),
            values=self.target_loader.values[index, :, :slots, :].to(
                device=device, dtype=dtype
            ),
            slot_mask=self.target_loader.slot_mask[index, :slots].to(device=device),
            layer_number=24,
            relative_phase_delta=0,
        )


class V4OracleExactPrefixRuntime:
    """Clone one replayed native cache into baseline/target/reference branches."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: str,
        gate: EntropyHysteresisGate,
        controller: SideKVAttentionController,
        maximum_continuation_tokens: int = 32,
    ) -> None:
        if maximum_continuation_tokens != 32:
            raise ValueError("V4 oracle continuation is initially frozen at 32 tokens")
        if gate.config.layer_number != 24 or gate.config.risk_role != "online_joint_control":
            raise ValueError("V4 oracle requires the qualified layer-24 joint gate")
        if gate.config.rearm_low_entropy_token_count != 2:
            raise ValueError("V4 oracle recovery hysteresis requires two low tokens")
        if controller.layer_number != 24:
            raise ValueError("V4 oracle controller must inject layer 24")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.gate = gate
        self.controller = controller
        self.maximum_continuation_tokens = maximum_continuation_tokens
        self.decoding = GreedyDecodingPolicy(tokenizer=tokenizer, device=device)

    def _tensor(self, token_ids: Sequence[int]) -> torch.Tensor:
        return torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)

    @staticmethod
    def _cache_sequence_length(cache: Any) -> int:
        if cache is None:
            return 0
        getter = getattr(cache, "get_seq_length", None)
        if callable(getter):
            return int(getter())
        try:
            return int(cache[0][0].shape[-2])
        except (IndexError, KeyError, TypeError, AttributeError) as exc:
            raise RuntimeError("Unable to audit V4 oracle cache length") from exc

    @staticmethod
    def _cache_tensors(cache: Any) -> tuple[torch.Tensor, ...]:
        """Flatten a native or legacy cache into its tensor leaves."""

        legacy = cache.to_legacy_cache() if hasattr(cache, "to_legacy_cache") else cache
        tensors: list[torch.Tensor] = []

        def collect(value: Any) -> None:
            if isinstance(value, torch.Tensor):
                tensors.append(value)
                return
            if isinstance(value, Mapping):
                for nested in value.values():
                    collect(nested)
                return
            if isinstance(value, (tuple, list)):
                for nested in value:
                    collect(nested)

        collect(legacy)
        if not tensors:
            raise RuntimeError("Unable to audit V4 oracle cache tensor leaves")
        return tuple(tensors)

    @classmethod
    def _caches_exactly_equal(cls, first: Any, second: Any) -> bool:
        first_tensors = cls._cache_tensors(first)
        second_tensors = cls._cache_tensors(second)
        if len(first_tensors) != len(second_tensors):
            return False
        return all(
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left, right)
            for left, right in zip(first_tensors, second_tensors, strict=True)
        )

    @classmethod
    def _caches_have_independent_storage(cls, first: Any, second: Any) -> bool:
        first_storage = {
            tensor.untyped_storage().data_ptr() for tensor in cls._cache_tensors(first)
        }
        second_storage = {
            tensor.untyped_storage().data_ptr() for tensor in cls._cache_tensors(second)
        }
        return first_storage.isdisjoint(second_storage)

    @torch.inference_mode()
    def _replay_prefix_cache(
        self, *, prefix_token_ids: Sequence[int], prompt_token_count: int
    ) -> Any:
        prefix = [int(value) for value in prefix_token_ids]
        if prompt_token_count <= 0 or len(prefix) <= prompt_token_count:
            raise ValueError("V4 oracle prefix must include partial reasoning")
        self.controller.deactivate()
        prompt = self._tensor(prefix[:prompt_token_count])
        output = self.model(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            use_cache=True,
            return_dict=True,
        )
        past = output.past_key_values
        replayed_length = prompt_token_count
        for token_id in prefix[prompt_token_count:-1]:
            replayed_length += 1
            token = self._tensor([token_id])
            output = self.model(
                input_ids=token,
                attention_mask=torch.ones(
                    (1, replayed_length), dtype=torch.long, device=self.device
                ),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = output.past_key_values
        if self._cache_sequence_length(past) != len(prefix) - 1:
            raise RuntimeError("V4 oracle replayed cache length drifted")
        return past

    @staticmethod
    def _token_rank(scores: torch.Tensor, token_id: int) -> int:
        values = scores[0].float()
        token_score = values[int(token_id)]
        token_ids = torch.arange(values.shape[0], device=values.device)
        better = (values > token_score) | (
            (values == token_score) & (token_ids < int(token_id))
        )
        return int(better.sum().item()) + 1

    @staticmethod
    def _log_probability(scores: torch.Tensor, token_id: int) -> float:
        return float(torch.log_softmax(scores.float(), dim=-1)[0, int(token_id)].item())

    @torch.inference_mode()
    def _run_branch(
        self,
        *,
        role: str,
        prefix_token_ids: Sequence[int],
        prompt_token_count: int,
        initial_cache: Any,
        baseline_first_scores: torch.Tensor | None,
        memory: SideKVMemory | None,
    ) -> tuple[V4OracleBranchResult, torch.Tensor]:
        if role not in {"baseline", "target", "reference"}:
            raise ValueError("Unknown V4 oracle branch role")
        if (role == "baseline") != (memory is None):
            raise ValueError("V4 oracle baseline/memory branch role mismatch")
        ids = [int(value) for value in prefix_token_ids]
        initial_length = self._cache_sequence_length(initial_cache)
        expected_initial = len(ids) - 1
        if initial_length != expected_initial:
            raise RuntimeError("V4 oracle branch initial cache length drifted")
        past = initial_cache
        self.controller.deactivate()
        self.controller.clear_traces()
        episode: V4MemoryEpisodeController | None = None
        if memory is not None:
            episode = V4MemoryEpisodeController()
            episode.apply_selection(memory.memory_id)
            self.controller.activate(memory)
        first_scores: torch.Tensor | None = None
        first_output_cache_length = -1
        answer_marker_seen = False
        eos_seen = False
        try:
            for step in range(self.maximum_continuation_tokens):
                full = self._tensor(ids)
                probe = self.gate.probe(
                    model=self.model,
                    boundary_token=full[:, -1:],
                    attention_mask=torch.ones_like(full),
                    past_key_values=past,
                    clone_past_key_values=False,
                )
                scores = self.decoding.processed_scores(
                    token_ids=ids, logits=probe.output.logits
                ).detach().float()
                if step == 0:
                    first_scores = scores
                    first_output_cache_length = self._cache_sequence_length(
                        probe.output.past_key_values
                    )
                if episode is not None and episode.state == "ACTIVE":
                    transition = episode.observe_decoded_token(
                        low_entropy=(
                            probe.entropy <= self.gate.config.low_entropy_threshold
                        )
                    )
                    if transition is not None and transition.deactivate_memory:
                        self.controller.deactivate()
                next_token = self.decoding.next_token(
                    token_ids=ids, logits=probe.output.logits
                )
                ids.append(next_token)
                past = probe.output.past_key_values
                completion_text = self.tokenizer.decode(
                    ids[prompt_token_count:], skip_special_tokens=False
                )
                if not answer_marker_seen and _ANSWER_MARKER_RE.search(completion_text):
                    answer_marker_seen = True
                    if episode is not None and episode.state != "CLOSED":
                        transition = episode.close(reason="answer_marker")
                        if transition.deactivate_memory:
                            self.controller.deactivate()
                if self.decoding.is_eos(next_token):
                    eos_seen = True
                    if episode is not None and episode.state != "CLOSED":
                        transition = episode.close(reason="eos")
                        if transition.deactivate_memory:
                            self.controller.deactivate()
                    break
        finally:
            self.controller.deactivate()
        if episode is not None and episode.state != "CLOSED":
            episode.close(reason="maximum_continuation_tokens")
        if first_scores is None:
            raise RuntimeError("V4 oracle branch produced no first-step logits")
        reference_scores = first_scores if baseline_first_scores is None else baseline_first_scores
        baseline_top1 = int(reference_scores.argmax(dim=-1).item())
        branch_top1 = int(first_scores.argmax(dim=-1).item())
        traces = self.controller.traces
        if memory is None and traces:
            raise RuntimeError("V4 oracle baseline emitted side-KV traces")
        if memory is not None:
            if not traces or any(
                trace.memory_id != memory.memory_id
                or not math.isfinite(float(trace.memory_attention_mass))
                or float(trace.memory_attention_mass) <= 0.0
                or trace.canonical_rope_score_relative_error is None
                or not math.isfinite(
                    float(trace.canonical_rope_score_relative_error)
                )
                for trace in traces
            ):
                raise RuntimeError("V4 oracle memory attention integrity failed")
            expected_lengths = tuple(
                len(prefix_token_ids) + index for index in range(len(traces))
            )
            if tuple(trace.native_key_length for trace in traces) != expected_lengths:
                raise RuntimeError("V4 oracle memory trace cache lengths drifted")
        result = V4OracleBranchResult(
            role=role,
            continuation_token_ids=tuple(ids[len(prefix_token_ids) :]),
            memory_id=memory.memory_id if memory is not None else None,
            attention_traces=traces,
            lifecycle=episode.summary() if episode is not None else None,
            first_step_logits_kl=logits_kl(reference_scores, first_scores),
            first_step_top1_changed=baseline_top1 != branch_top1,
            baseline_top1_token_id=baseline_top1,
            branch_top1_token_id=branch_top1,
            baseline_top1_rank_under_branch=self._token_rank(first_scores, baseline_top1),
            branch_top1_rank_under_baseline=self._token_rank(reference_scores, branch_top1),
            baseline_top1_log_probability_delta=(
                self._log_probability(first_scores, baseline_top1)
                - self._log_probability(reference_scores, baseline_top1)
            ),
            branch_top1_log_probability_delta=(
                self._log_probability(first_scores, branch_top1)
                - self._log_probability(reference_scores, branch_top1)
            ),
            initial_cache_length=initial_length,
            first_output_cache_length=first_output_cache_length,
            answer_marker_seen=answer_marker_seen,
            eos_seen=eos_seen,
        )
        return result, first_scores

    @torch.inference_mode()
    def run_three_branches(
        self,
        *,
        prefix_token_ids: Sequence[int],
        prompt_token_count: int,
        target: SideKVMemory,
        reference: SideKVMemory,
    ) -> tuple[dict[str, V4OracleBranchResult], dict[str, Any]]:
        prefix = tuple(int(value) for value in prefix_token_ids)
        if target.memory_id.endswith("::reference"):
            raise ValueError("V4 oracle target role has a reference memory ID")
        if reference.memory_id != f"{target.memory_id}::reference":
            raise ValueError("V4 oracle target/reference bank identities differ")
        replayed = self._replay_prefix_cache(
            prefix_token_ids=prefix, prompt_token_count=prompt_token_count
        )
        base_length = self._cache_sequence_length(replayed)
        cloned = {role: clone_cache(replayed) for role in ("baseline", "target", "reference")}
        clone_lengths = {role: self._cache_sequence_length(cache) for role, cache in cloned.items()}
        if set(clone_lengths.values()) != {base_length}:
            raise RuntimeError("V4 oracle cloned cache lengths differ")
        role_pairs = (
            ("baseline", "target"),
            ("baseline", "reference"),
            ("target", "reference"),
        )
        initial_tensor_parity = all(
            self._caches_exactly_equal(cloned[left], cloned[right])
            for left, right in role_pairs
        )
        independent_storage = all(
            self._caches_have_independent_storage(cloned[left], cloned[right])
            for left, right in role_pairs
        )
        if not initial_tensor_parity:
            raise RuntimeError("V4 oracle cloned cache tensors differ before branching")
        if not independent_storage:
            raise RuntimeError("V4 oracle cloned caches share mutable tensor storage")
        cache_geometry = [
            {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for tensor in self._cache_tensors(replayed)
        ]
        baseline, baseline_scores = self._run_branch(
            role="baseline",
            prefix_token_ids=prefix,
            prompt_token_count=prompt_token_count,
            initial_cache=cloned["baseline"],
            baseline_first_scores=None,
            memory=None,
        )
        target_result, _ = self._run_branch(
            role="target",
            prefix_token_ids=prefix,
            prompt_token_count=prompt_token_count,
            initial_cache=cloned["target"],
            baseline_first_scores=baseline_scores,
            memory=target,
        )
        reference_result, _ = self._run_branch(
            role="reference",
            prefix_token_ids=prefix,
            prompt_token_count=prompt_token_count,
            initial_cache=cloned["reference"],
            baseline_first_scores=baseline_scores,
            memory=reference,
        )
        results = {
            "baseline": baseline,
            "target": target_result,
            "reference": reference_result,
        }
        first_lengths = {
            role: result.first_output_cache_length for role, result in results.items()
        }
        if set(first_lengths.values()) != {len(prefix)}:
            raise RuntimeError("V4 oracle first-output cache lengths differ")
        parity = {
            "prefix_token_count": len(prefix),
            "prefix_token_ids_sha256": canonical_json_sha256(list(prefix)),
            "replayed_cache_length": base_length,
            "branch_initial_cache_lengths": clone_lengths,
            "branch_first_output_cache_lengths": first_lengths,
            "all_branches_share_exact_prefix": True,
            "all_branches_start_from_cloned_cache": True,
            "initial_cache_lengths_equal": True,
            "initial_cache_tensors_exactly_equal": initial_tensor_parity,
            "branch_cache_storage_is_independent": independent_storage,
            "replayed_cache_geometry_sha256": canonical_json_sha256(cache_geometry),
        }
        return results, parity


__all__ = [
    "V4OfflineSideKVRoleBankLoader",
    "V4OracleBranchResult",
    "V4OracleExactPrefixRuntime",
]
