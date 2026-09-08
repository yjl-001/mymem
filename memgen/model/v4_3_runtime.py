"""Unified-memory four-branch and visible-prompt full-answer audit runtime."""
from __future__ import annotations

from itertools import combinations
import math
import re
from typing import Any, Mapping, Sequence

import torch

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.experience.v4_3_bank import canonical_hash, text_hash
from memgen.model.e1_runtime import clone_cache, logits_kl
from memgen.model.side_kv import SideKVMemory
from memgen.model.v4_oracle import V4OracleExactPrefixRuntime, _complete_boxed_answer_seen

_MARKER = re.compile(r"\\boxed|\\fbox|final\s+answer|answer\s+is", re.I)


class V43UnifiedRuntime(V4OracleExactPrefixRuntime):
    """Reuse audited cache/decoding utilities, with independent unified branches.

    The legacy target/reference loader and three-branch entrypoint are never
    called. Episode accounting here only permits a single activation.
    """

    @torch.inference_mode()
    def replay(self, prefix: Sequence[int], prompt_count: int):
        if len(prefix) < prompt_count or prompt_count < 2:
            raise ValueError("Invalid native prompt/prefix")
        if len(prefix) > prompt_count:
            return self._replay_prefix_cache(prefix_token_ids=prefix, prompt_token_count=prompt_count)
        self.controller.deactivate()
        ids = self._tensor(prefix[:-1])
        cache = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True).past_key_values
        if self._cache_sequence_length(cache) != len(prefix) - 1:
            raise RuntimeError("Prompt-end cache length mismatch")
        return cache

    @torch.inference_mode()
    def decode(self, *, prefix: Sequence[int], prompt_count: int, cache: Any,
               memory: SideKVMemory | None, baseline_scores: torch.Tensor | None = None):
        ids = list(prefix)
        budget = 1024 - (len(prefix) - prompt_count)
        if budget <= 0 or self._cache_sequence_length(cache) != len(prefix) - 1:
            raise ValueError("Prefix budget/cache alignment invalid")
        limit = getattr(self.model.config, "max_position_embeddings", None) if hasattr(self.model, "config") else None
        if limit and len(prefix) + budget > limit:
            raise ValueError("Full 1024-token completion would exceed model context")
        if _complete_boxed_answer_seen(self.tokenizer.decode(ids[prompt_count:], skip_special_tokens=False)):
            raise ValueError("Audit prefix already contains a completed answer")
        self.controller.deactivate()
        self.controller.clear_traces()
        active = memory is not None
        if active:
            if not memory.memory_id.startswith("v43-bank-") or "::" in memory.memory_id:
                raise ValueError("Only unified V4.3 memories can be activated")
            self.controller.activate(memory)
        active_steps = low_streak = 0
        close_reason = None
        first_scores = None
        stop_reason = "maximum_completion_tokens"
        initial_length = self._cache_sequence_length(cache)
        try:
            for step in range(budget):
                full = self._tensor(ids)
                if active:
                    probe = self.gate.probe(model=self.model, boundary_token=full[:, -1:],
                        attention_mask=torch.ones_like(full), past_key_values=cache, clone_past_key_values=False)
                    output = probe.output
                    if not math.isfinite(float(probe.entropy)):
                        raise RuntimeError("Nonfinite attention entropy")
                    low_streak = low_streak + 1 if probe.entropy <= self.gate.config.low_entropy_threshold else 0
                    active_steps += 1
                else:
                    output = self.model(input_ids=full[:, -1:], attention_mask=torch.ones_like(full),
                                        past_key_values=cache, use_cache=True, return_dict=True)
                if self._cache_sequence_length(output.past_key_values) != len(ids):
                    raise RuntimeError("Side-KV changed native causal cache length")
                scores = self.decoding.processed_scores(token_ids=ids, logits=output.logits).detach().float()
                if not torch.isfinite(scores.max()):
                    raise RuntimeError("No finite next-token score")
                if first_scores is None:
                    first_scores = scores
                token = self.decoding.next_token(token_ids=ids, logits=output.logits)
                ids.append(token)
                cache = output.past_key_values
                completion = self.tokenizer.decode(ids[prompt_count:], skip_special_tokens=False)
                eos = self.decoding.is_eos(token)
                boxed = _complete_boxed_answer_seen(completion)
                if active:
                    if eos or boxed or _MARKER.search(completion):
                        close_reason = "eos" if eos else "answer_marker"
                    elif low_streak >= 2:
                        close_reason = "recovery_low_entropy_hysteresis"
                    elif active_steps >= 32:
                        close_reason = "maximum_active_window"
                    if close_reason:
                        self.controller.deactivate()
                        active = False
                if eos or boxed:
                    stop_reason = "eos" if eos else "completed_boxed_answer"
                    break
        finally:
            self.controller.deactivate()
        traces = [t.to_dict() for t in self.controller.traces]
        if len(traces) != active_steps or (memory is not None and active_steps == 0):
            raise RuntimeError("Exactly one trace required per active step")
        for i, trace in enumerate(traces):
            if (trace["memory_id"] != memory.memory_id or trace["native_key_length"] != len(prefix) + i
                    or not 0 < trace["memory_attention_mass"] <= 1
                    or not math.isclose(trace["memory_attention_mass"] + trace["native_attention_mass"], 1.0, abs_tol=0.003)
                    or trace["canonical_rope_score_relative_error"] is None
                    or not math.isfinite(trace["canonical_rope_score_relative_error"])):
                raise RuntimeError("Side-KV attention integrity failed")
        compared = first_scores if baseline_scores is None else baseline_scores
        base_token, branch_token = int(compared.argmax(-1)), int(first_scores.argmax(-1))
        continuation = ids[len(prefix):]
        return {
            "memory_id": memory.memory_id if memory is not None else None,
            "continuation_token_ids": continuation, "continuation_token_ids_sha256": canonical_hash(continuation),
            "local_continuation_token_ids": continuation[:32], "attention_traces": traces,
            "active_step_count": active_steps, "memory_close_reason": close_reason or (stop_reason if memory is not None else None),
            "post_memory_native_step_count": len(continuation) - active_steps,
            "activation_count": int(memory is not None), "maximum_completion_tokens": 1024,
            "prefix_completion_token_count": len(prefix) - prompt_count,
            "generation_budget_from_prefix": budget, "stop_reason": stop_reason,
            "first_step_logits_kl": max(0.0, logits_kl(compared, first_scores)),
            "first_step_top1_changed": base_token != branch_token,
            "baseline_top1_token_id": base_token, "branch_top1_token_id": branch_token,
            "initial_cache_length": initial_length, "first_output_cache_length": len(prefix),
            "final_cache_length": self._cache_sequence_length(cache),
            "prefix_token_ids_sha256": canonical_hash(list(prefix)),
        }, first_scores

    @torch.inference_mode()
    def run_latent(self, *, prefix: Sequence[int], prompt_count: int, memories: Mapping[str, SideKVMemory | None]):
        if list(memories)[0] != "baseline" or memories["baseline"] is not None or any(m is None for k, m in memories.items() if k != "baseline"):
            raise ValueError("Latent branches require baseline followed by unified memories")
        replayed = self.replay(prefix, prompt_count)
        # All branch initial states are materialized and checked before any
        # branch generation, including exhaustive sweeps.
        clones = {name: clone_cache(replayed) for name in memories}
        pairs = list(combinations(clones, 2))
        exact = all(self._caches_exactly_equal(clones[a], clones[b]) for a, b in pairs)
        independent = all(self._caches_have_independent_storage(clones[a], clones[b]) for a, b in pairs)
        lengths = {name: self._cache_sequence_length(c) for name, c in clones.items()}
        if not exact or not independent or set(lengths.values()) != {len(prefix) - 1}:
            raise RuntimeError("Exact-prefix independent cache clone validation failed")
        results, baseline_scores = {}, None
        for name, memory in memories.items():
            result, scores = self.decode(prefix=prefix, prompt_count=prompt_count, cache=clones.pop(name),
                                         memory=memory, baseline_scores=baseline_scores)
            result["condition_memory_id"] = memory.memory_id if memory is not None else None
            results[name] = result
            if name == "baseline":
                baseline_scores = scores
        return results, {"all_branches_share_exact_prefix": True, "initial_cache_tensors_exactly_equal": exact,
            "branch_cache_storage_is_independent": independent, "cache_length_parity": True,
            "branch_initial_cache_lengths": lengths, "prefix_token_ids_sha256": canonical_hash(list(prefix)),
            "prompt_end_boundary_policy": "replay_native_prefix_except_last_token_then_branch_on_last_live_query"}

    def visible_prefix(self, question: str, descriptor: str | None) -> list[int]:
        if descriptor is None:
            return GSM8K_PROMPT_CONTRACT.token_ids(self.tokenizer, question)
        messages = [{"role": "system", "content": "Use this reusable reasoning guidance only when applicable:\n" + descriptor}]
        messages += GSM8K_PROMPT_CONTRACT.messages(question)
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return list(self.tokenizer.encode(rendered, add_special_tokens=False))

    @torch.inference_mode()
    def run_visible(self, *, question: str, descriptors: Mapping[str, str | None], memory_ids: Mapping[str, str | None]):
        if list(descriptors)[0] != "baseline" or descriptors["baseline"] is not None:
            raise ValueError("Visible audit requires plain baseline first")
        results, scores = {}, None
        for name, descriptor in descriptors.items():
            prefix = self.visible_prefix(question, descriptor)
            result, first = self.decode(prefix=prefix, prompt_count=len(prefix), cache=self.replay(prefix, len(prefix)),
                                        memory=None, baseline_scores=scores)
            result["condition_memory_id"] = memory_ids[name]
            result["visible_descriptor_sha256"] = None if descriptor is None else text_hash(descriptor)
            results[name] = result
            if name == "baseline":
                scores = first
        return results, {"all_branches_share_exact_prefix": False, "first_step_kl_context": "different_visible_prompts",
                         "branch_prefix_sha256": {name: r["prefix_token_ids_sha256"] for name, r in results.items()}}
