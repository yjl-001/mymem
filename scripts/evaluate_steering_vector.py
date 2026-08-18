#!/usr/bin/env python3
"""Evaluate one Phase 2 steering condition on a frozen GSM8K split.

This is deliberately a standalone autoregressive loop.  Unlike the existing
MemGen Weaver loop it injects no latent tokens: at a selected decoder block it
adds a bounded residual only to the hidden state of a triggered delimiter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.utils.math_utils import diagnose_gsm8k_completion
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.phase1 import canonical_json_sha256, text_sha256, write_jsonl
from memgen.experience.phase2 import (
    STEERING_VECTOR_ARTIFACT_SCHEMA,
    build_gsm8k_messages,
    soft_entropy_gate,
    stable_uniform,
)


CONDITIONS = (
    "vanilla",
    "entropy_only",
    "real_vector",
    "random_boundary",
    "random_vector",
    "reversed_vector",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--logical-split", choices=("calibration-val", "dev-test", "final-test"), required=True)
    parser.add_argument("--layer", type=int, required=True, help="1-based decoder block index")
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--entropy-threshold", type=float, required=True)
    parser.add_argument("--gate-slope", type=float, default=0.15)
    parser.add_argument("--max-injections", type=int, default=3)
    parser.add_argument("--r-max", type=float, default=0.10)
    parser.add_argument("--sink-token-count", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected split examples")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many manifest-ordered split examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-boundary-rate", type=float, default=0.15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--attn-implementation",
        default="eager",
        help="Must support output_attentions; eager is the portable default.",
    )
    return parser.parse_args()


def processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_decoder_layers(model: Any) -> Any:
    """Find the Transformer block list for common HF causal-LM wrappers."""

    candidates = (
        getattr(getattr(model, "model", None), "layers", None),  # Qwen/Llama/Mistral
        getattr(getattr(model, "transformer", None), "h", None),  # GPT-2 family
        getattr(getattr(getattr(model, "model", None), "decoder", None), "layers", None),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise ValueError("Unsupported model architecture: unable to locate decoder layers")


def current_rms(tensor: Any) -> float:
    return float(tensor.float().square().mean().sqrt().item())


class ResidualVectorHook:
    """A one-shot, bounded residual injection into one decoder layer output."""

    def __init__(self, layer_module: Any, vector: Any, r_max: float):
        self.vector = vector.detach()
        self.vector_rms = current_rms(self.vector)
        if not math.isfinite(self.vector_rms) or self.vector_rms <= 0:
            raise ValueError("Steering vector has invalid RMS")
        self.r_max = r_max
        self.pending: dict[str, Any] | None = None
        self.last_result: dict[str, Any] | None = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def close(self) -> None:
        self.handle.remove()

    def arm(self, *, alpha: float, gate: float) -> None:
        self.pending = {"alpha": alpha, "gate": gate}
        self.last_result = None

    def disarm(self) -> None:
        self.pending = None

    def _hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.pending is None:
            return output
        pending = self.pending
        self.pending = None
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            self.last_result = {"applied": False, "disabled_reason": "unexpected_hidden_shape"}
            return output
        state = hidden[:, -1, :]
        state_rms = current_rms(state)
        desired_scale = float(pending["alpha"]) * float(pending["gate"])
        if not math.isfinite(state_rms) or state_rms <= 0 or not math.isfinite(desired_scale):
            self.last_result = {"applied": False, "disabled_reason": "nonfinite_state_or_scale"}
            return output
        # Both terms use RMS normalization, so desired_scale is the intended
        # relative perturbation.  We still measure the exact ratio below and
        # leave an audit trail instead of silently clipping it.
        if desired_scale > self.r_max:
            self.last_result = {
                "applied": False,
                "disabled_reason": "relative_perturbation_exceeds_r_max",
                "requested_relative_delta_norm": desired_scale,
            }
            return output
        delta = (
            self.vector.to(device=state.device, dtype=state.dtype) / self.vector_rms
        ) * (desired_scale * state_rms)
        relative_delta_norm = float(
            delta.float().norm().item() / state.float().norm().clamp_min(1e-12).item()
        )
        if not math.isfinite(relative_delta_norm) or relative_delta_norm > self.r_max:
            self.last_result = {
                "applied": False,
                "disabled_reason": "measured_relative_perturbation_exceeds_r_max",
                "relative_delta_norm": relative_delta_norm,
            }
            return output
        edited = hidden.clone()
        edited[:, -1, :] = state + delta
        self.last_result = {
            "applied": True,
            "state_rms": state_rms,
            "relative_delta_norm": relative_delta_norm,
        }
        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited


def is_delimiter(tokenizer: Any, token_id: int) -> bool:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).rstrip(" \t").endswith(
        (",", ".", "\n")
    )


def sink_masked_entropy(
    model: Any, input_ids: Any, attention_mask: Any, sink_count: int
) -> tuple[float, Any]:
    """Compute the FlashMem-style last-layer entropy without leading sinks."""

    import torch

    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    attentions = output.attentions
    if not attentions or attentions[-1] is None:
        raise RuntimeError(
            "Model did not return attentions. Phase 2 entropy gating requires "
            "--attn-implementation eager (or another attention backend with weights)."
        )
    valid_positions = attention_mask[0].nonzero(as_tuple=True)[0]
    if valid_positions.numel() < 2:
        raise ValueError("Need at least two valid tokens for entropy")
    query_index = int(valid_positions[-1].item())
    raw = attentions[-1][0, :, query_index, :].float()
    keys = valid_positions[sink_count:]
    if keys.numel() == 0:
        raise ValueError("Sink mask removed every attention key")
    probs = raw.index_select(1, keys)
    normalizer = probs.sum(dim=-1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise RuntimeError("Attention mass after sink masking is zero")
    probs = probs / normalizer
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
    value = float(entropy.mean().item())
    if not math.isfinite(value):
        raise RuntimeError("Non-finite sink-masked entropy")
    # The caller uses these logits as the no-injection counterfactual for the
    # same prefix.  The attention forward already exists for entropy gating, so
    # recording it adds no extra model pass.
    return value, output.logits[:, -1, :].detach().float()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_sha256")
    actual = canonical_json_sha256(
        {key: value for key, value in manifest.items() if key not in {"created_at", "manifest_sha256"}}
    )
    if expected != actual or not manifest.get("overlap_check", {}).get("passed"):
        raise ValueError("Invalid or overlapping split manifest")
    return manifest


def build_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    prompt = tokenizer.apply_chat_template(
        build_gsm8k_messages(question), tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(prompt, add_special_tokens=False)


def generate_one(
    *,
    model: Any,
    tokenizer: Any,
    hook: ResidualVectorHook,
    prompt_ids: list[int],
    condition: str,
    sample_id: str,
    alpha: float,
    entropy_threshold: float,
    gate_slope: float,
    max_injections: int,
    sink_token_count: int,
    max_new_tokens: int,
    seed: int,
    random_boundary_rate: float,
    device: str,
) -> tuple[list[int], list[dict[str, Any]]]:
    import torch

    ids = list(prompt_ids)
    events: list[dict[str, Any]] = []
    past = None
    eos = tokenizer.eos_token_id
    injection_count = 0
    for generation_step in range(max_new_tokens):
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        event: dict[str, Any] | None = None
        baseline_logits = None
        should_arm = False
        gate = 0.0
        if generation_step > 0 and is_delimiter(tokenizer, ids[-1]):
            event = {
                "generation_step": generation_step,
                "candidate": True,
                "injection_count_before": injection_count,
            }
            if condition != "vanilla":
                entropy, baseline_logits = sink_masked_entropy(
                    model, input_ids, attention_mask, sink_token_count
                )
                triggered = entropy >= entropy_threshold and injection_count < max_injections
                gate = soft_entropy_gate(entropy, entropy_threshold, gate_slope)
                event.update(
                    {
                        "entropy": entropy,
                        "entropy_threshold": entropy_threshold,
                        "soft_gate": gate,
                        "entropy_triggered": triggered,
                    }
                )
                if condition in {"real_vector", "random_vector", "reversed_vector"}:
                    should_arm = triggered
                elif condition == "random_boundary":
                    random_draw = stable_uniform(seed, sample_id, str(generation_step))
                    should_arm = random_draw < random_boundary_rate and injection_count < max_injections
                    gate = 1.0
                    event.update(
                        {
                            "random_boundary_draw": random_draw,
                            "random_boundary_rate": random_boundary_rate,
                        }
                    )
            else:
                event["entropy_triggered"] = False
        if should_arm:
            hook.arm(alpha=alpha, gate=gate)
        else:
            hook.disarm()

        model_kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "return_dict": True,
        }
        if past is None:
            model_kwargs["input_ids"] = input_ids
        else:
            model_kwargs["input_ids"] = input_ids[:, -1:]
            model_kwargs["past_key_values"] = past
        with torch.inference_mode():
            output = model(**model_kwargs)
        hook_result = hook.last_result if should_arm else None
        hook.disarm()
        if event is not None:
            event["injection"] = hook_result or {"applied": False}
            if hook_result and hook_result.get("applied"):
                injection_count += 1
                if baseline_logits is not None:
                    baseline_log_probs = baseline_logits.log_softmax(dim=-1)
                    injected_log_probs = output.logits[:, -1, :].float().log_softmax(dim=-1)
                    baseline_probs = baseline_log_probs.exp()
                    event["logits_kl_baseline_to_injected"] = float(
                        (baseline_probs * (baseline_log_probs - injected_log_probs)).sum().item()
                    )
                    event["top1_token_changed"] = bool(
                        baseline_logits.argmax(dim=-1).item()
                        != output.logits[:, -1, :].argmax(dim=-1).item()
                    )
            events.append(event)
        next_token = int(output.logits[:, -1, :].argmax(dim=-1).item())
        ids.append(next_token)
        past = output.past_key_values
        if eos is not None and next_token == eos:
            break
    # A candidate's next candidate boundary is an observable, post-intervention
    # outcome.  It is recorded after generation rather than used to trigger an
    # online intervention, so runtime remains causal.
    entropy_event_indices = [index for index, event in enumerate(events) if "entropy" in event]
    for current_index, next_index in zip(entropy_event_indices, entropy_event_indices[1:]):
        current = events[current_index]
        following = events[next_index]
        current_entropy = float(current["entropy"])
        next_entropy = float(following["entropy"])
        current["next_candidate_entropy"] = next_entropy
        current["entropy_delta_to_next_candidate"] = next_entropy - current_entropy
        current["entropy_decreased_to_next_candidate"] = next_entropy < current_entropy
    return ids[len(prompt_ids) :], events


def main() -> None:
    args = parse_args()
    if args.layer <= 0 or args.max_injections < 0 or args.max_new_tokens <= 0:
        raise ValueError("layer/max-new-tokens must be positive and max-injections non-negative")
    if args.alpha < 0 or args.r_max <= 0 or args.sink_token_count < 0:
        raise ValueError("alpha/sink-token-count/r-max values are invalid")
    if not 0 <= args.random_boundary_rate <= 1:
        raise ValueError("random-boundary-rate must be between 0 and 1")
    if args.limit < 0 or args.offset < 0:
        raise ValueError("limit and offset must be non-negative")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("schema_version") != STEERING_VECTOR_ARTIFACT_SCHEMA:
        raise ValueError("Unexpected steering vector artifact schema")
    if args.layer not in artifact.get("vectors", {}):
        available = sorted(int(key) for key in artifact.get("vectors", {}))
        raise ValueError(f"Layer {args.layer} absent from artifact; available={available}")
    artifact_model = artifact.get("reasoner", {}).get("model_name")
    if artifact_model != args.model:
        raise ValueError(f"Artifact model {artifact_model!r} differs from requested model {args.model!r}")

    manifest = load_manifest(args.split_manifest)
    if manifest.get("dataset", {}).get("revision") != args.dataset_revision:
        raise ValueError("dataset revision differs from split manifest")
    selected = [
        sample for sample in manifest["samples"] if sample["logical_split"] == args.logical_split
    ]
    selected = selected[args.offset :]
    if args.limit:
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("No selected evaluation samples")
    dataset_split = "test" if args.logical_split == "final-test" else "train"
    dataset = load_dataset(
        "openai/gsm8k", "main", split=dataset_split, revision=args.dataset_revision
    )

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()
    decoder_layers = resolve_decoder_layers(model)
    if args.layer > len(decoder_layers):
        raise ValueError(f"Requested layer {args.layer} exceeds {len(decoder_layers)} decoder blocks")

    vector = artifact["vectors"][args.layer].float()
    if args.condition == "reversed_vector":
        vector = -vector
    elif args.condition == "random_vector":
        generator = torch.Generator(device="cpu").manual_seed(args.seed + args.layer)
        vector = torch.randn(vector.shape, generator=generator, dtype=torch.float32)
        vector = vector / current_rms(vector) * float(artifact["vector_rms"][str(args.layer)])
    hook = ResidualVectorHook(decoder_layers[args.layer - 1], vector, args.r_max)

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    try:
        for index, entry in enumerate(selected, start=1):
            source = dataset[int(entry["source_index"])]
            question = str(source["question"]).strip()
            if text_sha256(question) != entry["question_sha256"]:
                raise ValueError(f"Question hash mismatch for {entry['sample_id']}")
            prompt_ids = build_prompt_ids(tokenizer, question)
            completion_ids, events = generate_one(
                model=model,
                tokenizer=tokenizer,
                hook=hook,
                prompt_ids=prompt_ids,
                condition=args.condition,
                sample_id=str(entry["sample_id"]),
                alpha=args.alpha,
                entropy_threshold=args.entropy_threshold,
                gate_slope=args.gate_slope,
                max_injections=args.max_injections,
                sink_token_count=args.sink_token_count,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                random_boundary_rate=args.random_boundary_rate,
                device=args.device,
            )
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            diagnosis = diagnose_gsm8k_completion(
                completion, processed_solution(str(source["answer"]).strip())
            )
            injections = [event["injection"] for event in events if event["injection"].get("applied")]
            disabled = [
                event["injection"].get("disabled_reason")
                for event in events
                if event["injection"].get("disabled_reason")
            ]
            records.append(
                {
                    "sample_id": entry["sample_id"],
                    "logical_split": args.logical_split,
                    "condition": args.condition,
                    "artifact_id": artifact["artifact_id"],
                    "layer": args.layer,
                    "alpha": args.alpha,
                    "entropy_threshold": args.entropy_threshold,
                    "gate_slope": args.gate_slope,
                    "r_max": args.r_max,
                    "completion": completion,
                    "verifier": diagnosis,
                    "final_reward": diagnosis["reward"],
                    "generation_length": len(completion_ids),
                    "candidate_boundary_count": len(events),
                    "injection_applied_count": len(injections),
                    "max_relative_delta_norm": max(
                        (float(item.get("relative_delta_norm", 0.0)) for item in injections), default=0.0
                    ),
                    "disabled_injection_reasons": disabled,
                    "intervention_trace": events,
                }
            )
            if index % 10 == 0 or index == len(selected):
                print(f"[phase2-eval] {index}/{len(selected)} {entry['sample_id']}", flush=True)
    finally:
        hook.close()

    result_path = output_dir / "results.jsonl"
    write_jsonl(result_path, records)
    total = len(records)
    accuracy = sum(float(item["final_reward"]) for item in records) / total
    format_accuracy = sum(bool(item["verifier"].get("format_valid")) for item in records) / total
    applied = sum(int(item["injection_applied_count"]) for item in records)
    candidate = sum(int(item["candidate_boundary_count"]) for item in records)
    disabled = sum(len(item["disabled_injection_reasons"]) for item in records)
    injection_events = [
        event
        for item in records
        for event in item["intervention_trace"]
        if event.get("injection", {}).get("applied")
    ]
    logit_kls = [
        float(event["logits_kl_baseline_to_injected"])
        for event in injection_events
        if "logits_kl_baseline_to_injected" in event
    ]

    def entropy_recovery_summary(predicate: Any) -> dict[str, Any]:
        matched = [
            event
            for item in records
            for event in item["intervention_trace"]
            if predicate(event) and "entropy_delta_to_next_candidate" in event
        ]
        deltas = [float(event["entropy_delta_to_next_candidate"]) for event in matched]
        return {
            "measured_count": len(deltas),
            "mean_delta_to_next_candidate": sum(deltas) / len(deltas) if deltas else None,
            "entropy_decreased_rate": (
                sum(bool(event["entropy_decreased_to_next_candidate"]) for event in matched)
                / len(deltas)
                if deltas
                else None
            ),
        }
    summary = {
        "schema_version": "steering-evaluation-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "artifact": {
            "path": str(args.artifact),
            "artifact_id": artifact["artifact_id"],
            "artifact_sha256": __import__("hashlib").sha256(args.artifact.read_bytes()).hexdigest(),
        },
        "config": {
            "logical_split": args.logical_split,
            "layer": args.layer,
            "alpha": args.alpha,
            "entropy_threshold": args.entropy_threshold,
            "gate_slope": args.gate_slope,
            "max_injections": args.max_injections,
            "r_max": args.r_max,
            "sink_token_count": args.sink_token_count,
            "max_new_tokens": args.max_new_tokens,
            "offset": args.offset,
            "random_boundary_rate": args.random_boundary_rate,
            "seed": args.seed,
            "attn_implementation": args.attn_implementation,
        },
        "sample_count": total,
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
        "mean_generation_length": sum(item["generation_length"] for item in records) / total,
        "mean_injections": applied / total,
        "candidate_boundary_count": candidate,
        "injection_applied_count": applied,
        "disabled_injection_count": disabled,
        "max_observed_relative_delta_norm": max(
            (float(item["max_relative_delta_norm"]) for item in records), default=0.0
        ),
        "injection_logit_diagnostics": {
            "measured_count": len(logit_kls),
            "mean_kl_baseline_to_injected": sum(logit_kls) / len(logit_kls) if logit_kls else 0.0,
            "max_kl_baseline_to_injected": max(logit_kls, default=0.0),
            "top1_token_changed_count": sum(
                bool(event.get("top1_token_changed")) for event in injection_events
            ),
        },
        "entropy_recovery_diagnostics": {
            # Triggered is available for entropy-only controls; injected is the
            # matched mechanistic outcome for vector conditions.
            "triggered": entropy_recovery_summary(
                lambda event: bool(event.get("entropy_triggered", False))
            ),
            "injected": entropy_recovery_summary(
                lambda event: bool(event.get("injection", {}).get("applied", False))
            ),
        },
        "results": {"path": result_path.name, "sha256": __import__("hashlib").sha256(result_path.read_bytes()).hexdigest()},
        "git_revision": git_revision(),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[phase2-eval] condition={args.condition} accuracy={accuracy:.4f} "
        f"format={format_accuracy:.4f} injections={applied} summary={summary_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
