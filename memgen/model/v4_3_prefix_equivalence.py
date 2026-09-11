"""Native all-layer prefix KV reuse and paired numerical diagnostics.

This is a separate reference experiment, not a change to frozen side-KV.
"""
from pathlib import Path
import os
import tempfile

import torch
from safetensors.torch import load_file, save_file
from transformers import DynamicCache

from memgen.experience.v4_3_artifacts import atomic_json, read_json
from memgen.experience.v4_3_bank import authenticate, canonical_hash, file_hash, seal
from memgen.model.e1_runtime import clone_cache, logits_kl
from memgen.model.v4_3_runtime import V43UnifiedRuntime


def memory_prefix_ids(tokenizer, descriptor):
    messages = [{"role": "system", "content": "Use this reusable reasoning guidance only when applicable:\n" + descriptor}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return list(tokenizer.encode(rendered, add_special_tokens=False))


def split_prefix(runtime, question, descriptor, memory_ids):
    full = runtime.visible_prefix(question, descriptor)
    if not memory_ids or full[:len(memory_ids)] != list(memory_ids) or len(full) <= len(memory_ids) + 1:
        raise ValueError("Memory token prefix does not exactly match the visible prompt; no retokenization fallback")
    return full


def cache_tensors(cache):
    legacy = cache.to_legacy_cache()
    return {f"{i}.{kind}": tensor.detach().cpu().contiguous()
            for i, pair in enumerate(legacy) for kind, tensor in zip(("k", "v"), pair)}


def restore_cache(tensors, device):
    if not tensors or len(tensors) % 2:
        raise ValueError("Incomplete all-layer KV")
    layers = len(tensors) // 2
    if set(tensors) != {f"{i}.{kind}" for i in range(layers) for kind in ("k", "v")}:
        raise ValueError("Noncontiguous cache layer namespace")
    # Clone even on CPU: consuming a Bank must never mutate its stored prefix.
    return DynamicCache.from_legacy_cache(tuple(tuple(tensors[f"{i}.{kind}"].to(device).clone()
                                                for kind in ("k", "v")) for i in range(layers)))


def validate_tensors(tensors, token_count, model):
    config = model.config
    shape = (1, config.num_key_value_heads, token_count, config.hidden_size // config.num_attention_heads)
    dtype = next(model.parameters()).dtype
    expected = {f"{i}.{kind}" for i in range(config.num_hidden_layers) for kind in ("k", "v")}
    if set(tensors) != expected or any(tuple(t.shape) != shape or t.dtype != dtype or not torch.isfinite(t).all()
                                       for t in tensors.values()):
        raise ValueError("Native prefix cache geometry/dtype/finiteness mismatch")


@torch.inference_mode()
def prefix_bank(directory, record, runtime, profile_sha256, *, validate_only=False):
    directory = Path(directory)
    bid = record["bank_id"]
    if Path(bid).name != bid or not bid.startswith("v43-bank-"):
        raise ValueError("Unsafe Bank ID")
    ids = memory_prefix_ids(runtime.tokenizer, record["descriptor"])
    tensor_path = directory / (bid + ".safetensors")
    manifest_path = directory / (bid + ".json")
    identity = {"schema_version": "v43-native-prefix-kv-v1", "profile_sha256": profile_sha256,
                "bank_id": bid, "source_record_sha256": record["record_sha256"],
                "prefix_token_ids": ids, "prefix_token_ids_sha256": canonical_hash(ids),
                "all_layers": True, "position_policy": "native_absolute_prefix_positions",
                "includes_role_wrapper": True, "question_independent": True}
    if tensor_path.is_symlink() or manifest_path.is_symlink() or directory.is_symlink():
        raise ValueError("Refusing symlink prefix cache")
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        authenticate(manifest, "manifest_sha256", "native prefix KV")
        if any(manifest.get(k) != v for k, v in identity.items()) or file_hash(tensor_path) != manifest["tensor_sha256"]:
            raise ValueError("Native prefix cache identity/hash drift")
        tensors = load_file(str(tensor_path))
    else:
        if validate_only:
            raise ValueError("Missing native prefix cache manifest")
        runtime.controller.deactivate()
        x = runtime._tensor(ids)
        if not ids or len(ids) >= runtime.model.config.max_position_embeddings:
            raise ValueError("Invalid memory prefix length")
        out = runtime.model(input_ids=x, attention_mask=torch.ones_like(x),
                            cache_position=torch.arange(len(ids), device=x.device), use_cache=True, return_dict=True)
        tensors = cache_tensors(out.past_key_values)
        directory.mkdir(parents=True, exist_ok=True)
        if tensor_path.exists():
            old = load_file(str(tensor_path))
            if set(old) != set(tensors) or any(not torch.equal(old[k], tensors[k]) for k in tensors):
                raise ValueError("Interrupted prefix cache differs from recomputation")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".prefix-", dir=directory)
            os.close(fd)
            try:
                save_file(tensors, temporary)
                os.link(temporary, tensor_path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        atomic_json(manifest_path, seal({**identity, "tensor_sha256": file_hash(tensor_path)}, "manifest_sha256"), immutable=True)
    validate_tensors(tensors, len(ids), runtime.model)
    return ids, tensors


def tensor_difference(left, right, atol, rtol):
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("Paired tensor geometry/dtype mismatch")
    a, b = left.float(), right.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite paired tensor")
    delta = a - b
    return {"max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / a.norm().clamp_min(1e-12)),
            "exact": bool(torch.equal(left, right)),
            "within_tolerance": bool(torch.allclose(a, b, atol=atol, rtol=rtol))}


def compare_caches(left, right, atol, rtol):
    a = V43UnifiedRuntime._cache_tensors(left)
    b = V43UnifiedRuntime._cache_tensors(right)
    if len(a) != len(b):
        raise ValueError("Cache layer count mismatch")
    by_layer = [{kind: tensor_difference(a[2*i+j], b[2*i+j], atol, rtol)
                 for j, kind in enumerate(("k", "v"))} for i in range(len(a)//2)]
    return {"by_layer": by_layer, "within_tolerance": all(d["within_tolerance"] for l in by_layer for d in l.values()),
            "max_abs": max(d["max_abs"] for l in by_layer for d in l.values())}


@torch.inference_mode()
def paired_states(runtime, question, descriptor, memory_ids, tensors):
    prefix = split_prefix(runtime, question, descriptor, memory_ids)
    runtime.controller.deactivate()
    text_cache = runtime.replay(prefix, len(prefix))
    cached = restore_cache(tensors, runtime.device)
    suffix = prefix[len(memory_ids):-1]
    ids = runtime._tensor(suffix)
    output = runtime.model(input_ids=ids, attention_mask=torch.ones((1, len(prefix)-1), dtype=torch.long, device=ids.device),
                           past_key_values=cached, cache_position=torch.arange(len(memory_ids), len(prefix)-1, device=ids.device),
                           use_cache=True, return_dict=True)
    cached = output.past_key_values
    if runtime._cache_sequence_length(cached) != len(prefix)-1:
        raise ValueError("Cached-question prefill length mismatch")
    if not runtime._caches_have_independent_storage(text_cache, cached):
        raise ValueError("Paired caches share storage")
    return prefix, text_cache, cached


@torch.inference_mode()
def forced_diagnostic(runtime, prefix, text_cache, cached_cache, reference_tokens, *, atol, rtol):
    """Compare each prediction along the entire freely generated TEXT trajectory."""
    caches = [clone_cache(text_cache), clone_cache(cached_cache)]
    runtime.controller.deactivate()
    initial = compare_caches(caches[0], caches[1], atol, rtol)
    steps = []
    for i in range(len(reference_tokens)):
        token = prefix[-1] if i == 0 else reference_tokens[i-1]
        position = len(prefix)-1+i
        outputs = [runtime.model(input_ids=runtime._tensor([token]),
                    attention_mask=torch.ones((1, position+1), dtype=torch.long, device=runtime.device),
                    past_key_values=c, cache_position=torch.tensor([position], device=runtime.device),
                    use_cache=True, return_dict=True) for c in caches]
        caches = [o.past_key_values for o in outputs]
        a, b = [o.logits[:, -1, :] for o in outputs]
        diff = tensor_difference(a, b, atol, rtol)
        top = a.float().topk(2).values
        steps.append({"step": i, **diff, "kl": max(0., logits_kl(a, b)),
                      "top1_equal": int(a.argmax()) == int(b.argmax()),
                      "text_top1_margin": float(top[0, 0]-top[0, 1])})
    if not steps:
        raise ValueError("Empty text reference trajectory")
    final = compare_caches(caches[0], caches[1], atol, rtol)
    return {"atol": atol, "rtol": rtol, "comparison": "raw_logits_on_full_text_reference_trajectory",
            "prefill_cache": initial, "final_forced_cache": final, "steps": steps,
            "step_count": len(steps), "top1_mismatch_count": sum(not s["top1_equal"] for s in steps),
            "max_logits_abs": max(s["max_abs"] for s in steps), "max_kl": max(s["kl"] for s in steps),
            "numerical_pass": initial["within_tolerance"] and final["within_tolerance"] and all(s["within_tolerance"] for s in steps)}


@torch.inference_mode()
def run_equivalence(runtime, question, descriptor, memory_ids, tensors, *, atol, rtol):
    prefix, text_state, kv_state = paired_states(runtime, question, descriptor, memory_ids, tensors)
    text, _ = runtime.decode(prefix=prefix, prompt_count=len(prefix), cache=clone_cache(text_state), memory=None)
    cached, _ = runtime.decode(prefix=prefix, prompt_count=len(prefix), cache=clone_cache(kv_state), memory=None)
    diagnostics = forced_diagnostic(runtime, prefix, text_state, kv_state, text["continuation_token_ids"], atol=atol, rtol=rtol)
    diagnostics["free_tokens_equal"] = text["continuation_token_ids"] == cached["continuation_token_ids"]
    diagnostics["free_stop_equal"] = text["stop_reason"] == cached["stop_reason"]
    diagnostics["behavioral_pass"] = diagnostics["free_tokens_equal"] and diagnostics["free_stop_equal"] and diagnostics["top1_mismatch_count"] == 0
    return prefix, text, cached, diagnostics
