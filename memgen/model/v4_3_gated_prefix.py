"""Read offline prefix KV at virtual negative positions without rewriting history."""
import math
import types

import torch
import torch.nn.functional as F

from memgen.experience.v4_3_bank import canonical_hash
from memgen.model.side_kv import (
    QwenAttentionProtocol, _native_additive_mask, apply_rotary_pos_emb, repeat_kv, rotate_half,
)
from memgen.model.v4_3_prefix_equivalence import validate_tensors
from memgen.model.v4_3_runtime import _MARKER
from memgen.model.v4_oracle import _complete_boxed_answer_seen


def validate_model(model):
    config = model.config
    rotary = model.model.rotary_emb
    if (config.model_type != "qwen2" or config._attn_implementation != "sdpa"
            or getattr(rotary, "rope_type", None) != "default"
            or float(rotary.attention_scaling) != 1.
            or any(getattr(l.self_attn, "sliding_window", None) is not None for l in model.model.layers)
            or model.training):
        raise ValueError("Virtual prefix requires frozen Qwen2 SDPA, default fixed-frequency RoPE and full attention")
    return rotary.inv_freq.detach().float().cpu()


def virtual_prefix(model, token_ids, tensors):
    """Shift only stored post-RoPE memory keys by -M. Source tensors stay intact."""
    inv = validate_model(model)
    validate_tensors(tensors, len(token_ids), model)
    phase = -len(token_ids) * torch.cat((inv, inv))
    cos, sin = phase.cos().view(1, 1, 1, -1), phase.sin().view(1, 1, 1, -1)
    pairs = []
    for i in range(model.config.num_hidden_layers):
        k, v = tensors[f"{i}.k"], tensors[f"{i}.v"]
        x = k.detach().float().cpu()
        shifted = (x * cos + rotate_half(x) * sin).to(k.dtype)
        if not torch.isfinite(shifted).all():
            raise ValueError("Nonfinite rephased memory")
        pairs.append((shifted, v.detach().cpu().clone()))
    return tuple(pairs)


class VirtualPrefixReader:
    """Install native+memory joint SDPA only after the last unconditioned probe."""
    def __init__(self, model, memory):
        validate_model(model)
        self.modules = [layer.self_attn for layer in model.model.layers]
        if len(memory) != len(self.modules):
            raise ValueError("Memory must cover every decoder layer")
        self.memory = memory
        self.originals = [module.forward for module in self.modules]
        self.protocols = [QwenAttentionProtocol(forward) for forward in self.originals]
        self.active = False
        self.counts = [0] * len(memory)
        self.first_read = {}

    def activate(self):
        if self.active:
            raise RuntimeError("Virtual prefix may be enabled only once per reader")
        self.active = True
        self.memory = tuple(tuple(t.to(device=module.q_proj.weight.device) for t in pair)
                            for module, pair in zip(self.modules, self.memory))
        for i, module in enumerate(self.modules):
            def patched(module_self, *args, _index=i, **kwargs):
                return self.forward(_index, module_self, self.protocols[_index].bind(args, kwargs))
            module.forward = types.MethodType(patched, module)

    def close(self):
        if self.active:
            for module, original in zip(self.modules, self.originals):
                module.forward = original
            self.active = False

    def forward(self, index, module, call):
        hidden = call.hidden_states
        batch, length, _ = hidden.shape
        if batch != 1 or call.position_embeddings is None or call.past_key_value is None:
            raise ValueError("Virtual prefix requires batch-one cached native attention")
        shape = (batch, length, -1, module.head_dim)
        q = module.q_proj(hidden).view(shape).transpose(1, 2)
        k = module.k_proj(hidden).view(shape).transpose(1, 2)
        v = module.v_proj(hidden).view(shape).transpose(1, 2)
        cos, sin = call.position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # Only the newly processed real token(s) enter the native cache.
        k, v = call.past_key_value.update(k, v, module.layer_idx,
            {"cos": cos, "sin": sin, "cache_position": call.cache_position})
        native_length = k.shape[-2]
        mk, mv = self.memory[index]
        if mk.dtype != k.dtype or mv.dtype != v.dtype:
            raise ValueError("Memory/model dtype mismatch")
        k = repeat_kv(torch.cat((k, mk), dim=-2), module.num_key_value_groups)
        v = repeat_kv(torch.cat((v, mv), dim=-2), module.num_key_value_groups)
        native_mask = _native_additive_mask(attention_mask=call.attention_mask, batch_size=batch,
            query_length=length, native_key_length=native_length, device=q.device, dtype=q.dtype)
        memory_mask = torch.zeros((batch, 1, length, mk.shape[-2]), device=q.device, dtype=q.dtype)
        mask = torch.cat((native_mask, memory_mask), dim=-1)
        output = F.scaled_dot_product_attention(q.contiguous(), k.contiguous(), v.contiguous(),
            attn_mask=mask, dropout_p=0., scale=float(module.scaling), is_causal=False)
        if not self.counts[index]:
            scores = (q.float() @ k.float().transpose(-1, -2)) * float(module.scaling) + mask.float()
            mass = (scores[..., native_length:].logsumexp(-1) - scores.logsumexp(-1)).exp().mean()
            value = float(mass)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise RuntimeError("Invalid first-step memory attention mass")
            self.first_read[str(index+1)] = {"memory_attention_mass": value,
                "native_key_length": native_length, "memory_key_length": mk.shape[-2]}
        self.counts[index] += 1
        output = output.transpose(1, 2).contiguous().reshape(batch, length, -1)
        return module.o_proj(output), None


def snapshot_history(cache):
    return tuple(tuple(t.detach().cpu().clone() for t in pair) for pair in cache.to_legacy_cache())


def history_equal(cache, snapshot):
    layers = cache.to_legacy_cache()
    return len(layers) == len(snapshot) and all(
        torch.equal(actual[..., :old.shape[-2], :].detach().cpu(), old)
        for pair, saved in zip(layers, snapshot) for actual, old in zip(pair, saved))


def abstained_result(baseline):
    return {**baseline, "activation_count": 0, "gate_trigger_count": 0, "gate_traces": [],
        "activation": None, "active_forward_steps": 0, "first_read_by_layer": {}, "layer_read_counts": [],
        "history_kv_preserved": None, "inserted_token_count": 0, "replayed_token_count": 0,
        "non_activation_reason": "selector_abstained", "offline_prefix_kv_read": True}


@torch.inference_mode()
def generate_gated(runtime, question, memory, *, maximum_completion_tokens=1024):
    if not 0 < maximum_completion_tokens <= 1024:
        raise ValueError("Invalid completion budget")
    # Do not wrap attention during native gate observations.
    runtime.controller.close()
    reader = VirtualPrefixReader(runtime.model, memory)
    prefix = runtime.visible_prefix(question, None)
    memory_length = memory[0][0].shape[-2]
    if len(prefix)+memory_length+maximum_completion_tokens > runtime.model.config.max_position_embeddings:
        raise ValueError("Native+virtual prefix relative span exceeds context; no truncation")
    ids, completion, traces = list(prefix), [], []
    cache = runtime.replay(prefix, len(prefix))
    initial_length = runtime._cache_sequence_length(cache)
    trigger = activation = snapshot = None
    pending, marker_seen = False, False
    active_steps = 0
    stop = "maximum_completion_tokens"
    try:
        for step in range(maximum_completion_tokens):
            if runtime._cache_sequence_length(cache) != len(ids)-1:
                raise RuntimeError("Native cache length/position drift")
            if pending:
                snapshot = snapshot_history(cache)
                activation = {"trigger_generated_input_index": trigger["generated_input_index"],
                    "unchanged_generated_token_count": len(completion),
                    "first_memory_query_generated_index": len(completion)-1,
                    "native_cache_length_before": len(ids)-1,
                    "unchanged_completion_sha256": canonical_hash(completion)}
                reader.activate()
                pending = False
            marker_seen = marker_seen or bool(_MARKER.search(runtime.tokenizer.decode(completion, skip_special_tokens=False)))
            eligible = bool(completion) and trigger is None and not marker_seen
            full = runtime._tensor(ids)
            if eligible:
                probe = runtime.gate.probe(model=runtime.model, boundary_token=full[:, -1:],
                    attention_mask=torch.ones_like(full), past_key_values=cache, clone_past_key_values=False)
                if not all(math.isfinite(float(v)) for v in (probe.entropy, probe.risk_score)):
                    raise RuntimeError("Nonfinite gate observation")
                output = probe.output
                observed = {"generated_input_index": len(completion)-1, "entropy": float(probe.entropy),
                            "risk_score": float(probe.risk_score), "triggered": bool(runtime.gate.trigger_qualified(probe))}
                traces.append(observed)
                if observed["triggered"]:
                    trigger = observed
                    pending = True
            else:
                output = runtime.model(input_ids=full[:, -1:], attention_mask=torch.ones_like(full),
                                       past_key_values=cache, use_cache=True, return_dict=True)
            cache = output.past_key_values
            if runtime._cache_sequence_length(cache) != len(ids):
                raise RuntimeError("Memory changed native cache length")
            if reader.active:
                active_steps += 1
                if active_steps == 1 and not history_equal(cache, snapshot):
                    raise RuntimeError("First memory read rewrote historical KV")
            if not torch.isfinite(runtime.decoding.processed_scores(token_ids=ids, logits=output.logits).max()):
                raise RuntimeError("No finite next-token score")
            token = runtime.decoding.next_token(token_ids=ids, logits=output.logits)
            ids.append(token)
            completion.append(token)
            if runtime.decoding.is_eos(token):
                stop = "eos"
                break
            if _complete_boxed_answer_seen(runtime.tokenizer.decode(completion, skip_special_tokens=False)):
                stop = "completed_boxed_answer"
                break
        preserved = history_equal(cache, snapshot) if snapshot is not None else None
        if snapshot is not None and not preserved:
            raise RuntimeError("Historical KV changed after activation")
        if reader.counts != [active_steps]*len(memory):
            raise RuntimeError("Every active forward must read memory in all layers")
        return prefix, {"continuation_token_ids": completion, "continuation_token_ids_sha256": canonical_hash(completion),
            "local_continuation_token_ids": completion[:32], "maximum_completion_tokens": maximum_completion_tokens,
            "stop_reason": stop, "activation_count": int(activation is not None), "activation": activation,
            "gate_trigger_count": int(trigger is not None), "gate_traces": traces, "active_forward_steps": active_steps,
            "layer_read_counts": reader.counts, "first_read_by_layer": reader.first_read,
            "history_kv_preserved": preserved, "initial_cache_length": initial_length,
            "final_cache_length": runtime._cache_sequence_length(cache), "prefix_token_ids_sha256": canonical_hash(prefix),
            "inserted_token_count": 0, "replayed_token_count": 0, "memory_token_count": memory_length,
            "offline_prefix_kv_read": True, "non_activation_reason": None if activation else (
                "generation_stopped_before_first_memory_read" if trigger else
                "answer_marker_before_trigger" if marker_seen else "no_joint_trigger_before_stop")}
    finally:
        reader.close()
