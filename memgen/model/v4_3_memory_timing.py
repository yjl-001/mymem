"""Context-aware memory append; frozen selector/prefix/gate implementations stay intact."""
import math
import torch

from memgen.experience.v4_3_bank import canonical_hash
from memgen.model.v4_3_question_selector import generate
from memgen.model.v4_3_runtime import _MARKER
from memgen.model.v4_oracle import _complete_boxed_answer_seen

WRAPPER = ("\n[Reusable reasoning guidance]\n"
           "Use this reusable reasoning guidance only when applicable:\n{descriptor}\n"
           "[End reusable reasoning guidance]\nContinue solving the problem.\n")


def memory_tokens(runtime, descriptor):
    ids = list(runtime.tokenizer.encode(WRAPPER.format(descriptor=descriptor), add_special_tokens=False))
    if not ids:
        raise ValueError("Empty contextual memory")
    return ids


@torch.inference_mode()
def append_tokens(runtime, cache, tokens):
    """Consume each appended token once at its natural position, in every layer."""
    start = runtime._cache_sequence_length(cache)
    x = runtime._tensor(tokens)
    output = runtime.model(input_ids=x, past_key_values=cache,
        attention_mask=torch.ones((1, start + len(tokens)), dtype=torch.long, device=runtime.device),
        cache_position=torch.arange(start, start + len(tokens), device=runtime.device),
        use_cache=True, return_dict=True)
    if runtime._cache_sequence_length(output.past_key_values) != start + len(tokens):
        raise RuntimeError("Contextual memory cache growth mismatch")
    return output


@torch.inference_mode()
def generate_delayed(runtime, question, descriptor, mode, *, maximum_completion_tokens=1024):
    if mode not in {"prompt_end", "entropy_gate"}:
        raise ValueError("Unknown memory timing")
    if not 0 < maximum_completion_tokens <= 1024:
        raise ValueError("Invalid completion budget")
    # This runtime only uses native causal attention. Remove the unused legacy
    # side-KV wrapper so the frozen entropy observer sees the model signature.
    # close() is idempotent; the shared vanilla decoder can still deactivate it.
    runtime.controller.close()
    # A selector abstention uses the unchanged vanilla consumer, without probes.
    if descriptor is None:
        if maximum_completion_tokens != 1024:
            raise ValueError("Abstention uses the frozen 1024-token decoder")
        prefix, result = generate(runtime, question)
        return prefix, {**result, "timing": mode, "injected_token_count": 0,
                        "injection_generated_token_count": None, "gate_traces": [],
                        "injection": None, "non_activation_reason": "selector_abstained"}
    runtime.controller.deactivate()
    prefix = runtime.visible_prefix(question, None)
    memory = memory_tokens(runtime, descriptor)
    if len(prefix) + len(memory) + maximum_completion_tokens > runtime.model.config.max_position_embeddings:
        raise ValueError("Full question+memory+completion exceeds context; no truncation")
    context, completion, traces = list(prefix), [], []
    cache = runtime.replay(prefix, len(prefix))
    initial_length = runtime._cache_sequence_length(cache)
    injected, marker_seen, injection = False, False, None
    stop = "maximum_completion_tokens"
    for _ in range(maximum_completion_tokens):
        if runtime._cache_sequence_length(cache) != len(context) - 1:
            raise RuntimeError("Pending live-token/cache alignment drift")
        marker_seen = marker_seen or bool(_MARKER.search(runtime.tokenizer.decode(completion, skip_special_tokens=False)))
        eligible = mode == "entropy_gate" and bool(completion) and not injected and not marker_seen
        activate = mode == "prompt_end" and not injected
        if eligible:
            probe = runtime.gate.probe(model=runtime.model, boundary_token=runtime._tensor(context[-1:]),
                attention_mask=torch.ones((1, len(context)), dtype=torch.long, device=runtime.device),
                past_key_values=cache, clone_past_key_values=False)
            if not all(math.isfinite(float(v)) for v in (probe.entropy, probe.risk_score)):
                raise RuntimeError("Nonfinite gate observation")
            output = probe.output
            activate = bool(runtime.gate.trigger_qualified(probe))
            traces.append({"generated_input_index": len(completion)-1,
                           "entropy": float(probe.entropy), "risk_score": float(probe.risk_score),
                           "triggered": activate})
        else:
            output = append_tokens(runtime, cache, context[-1:])
        cache = output.past_key_values
        if runtime._cache_sequence_length(cache) != len(context):
            raise RuntimeError("Gate probe consumed an incorrect number of tokens")
        if activate:
            before = len(context)
            output = append_tokens(runtime, cache, memory)
            cache = output.past_key_values
            context.extend(memory)
            injected = True
            injection = {"generated_token_count": len(completion), "cache_length_before": before,
                         "cache_length_after": len(context), "memory_token_count": len(memory),
                         "memory_token_ids_sha256": canonical_hash(memory),
                         "pre_injection_completion_sha256": canonical_hash(completion)}
        if not torch.isfinite(runtime.decoding.processed_scores(token_ids=context, logits=output.logits).max()):
            raise RuntimeError("No finite next-token score")
        token = runtime.decoding.next_token(token_ids=context, logits=output.logits)
        context.append(token)
        # Injected guidance is input, never a generated answer or scoring target.
        completion.append(token)
        if runtime.decoding.is_eos(token):
            stop = "eos"
            break
        if _complete_boxed_answer_seen(runtime.tokenizer.decode(completion, skip_special_tokens=False)):
            stop = "completed_boxed_answer"
            break
    return prefix, {"continuation_token_ids": completion,
        "continuation_token_ids_sha256": canonical_hash(completion), "local_continuation_token_ids": completion[:32],
        "timing": mode, "activation_count": int(injected), "injection": injection,
        "injected_token_count": len(memory) if injected else 0,
        "injection_generated_token_count": injection["generated_token_count"] if injected else None,
        "gate_traces": traces, "non_activation_reason": None if injected else (
            "answer_marker_before_trigger" if marker_seen else "no_joint_trigger_before_stop"),
        "maximum_completion_tokens": maximum_completion_tokens, "stop_reason": stop,
        "initial_cache_length": initial_length, "final_cache_length": runtime._cache_sequence_length(cache),
        "prefix_token_ids_sha256": canonical_hash(prefix), "memory_persistence": "all_layers_until_generation_end",
        "offline_kv_splice": False}
