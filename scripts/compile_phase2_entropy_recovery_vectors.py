#!/usr/bin/env python3
"""Compile condition-matched entropy-recovery steering vectors without new AI.

The offline evidence condition mirrors online use: a vector is constructed only
from sink-masked high-attention-entropy reasoning boundaries.  Positive states
are successful target trajectories whose next boundary recovers to low entropy;
negative states are failed references whose next boundary remains diffuse.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.phase1 import file_sha256, iter_jsonl, write_jsonl
from memgen.experience.phase2 import (
    PHASE2_ELIGIBLE_EXPERIENCE_TYPES,
    STEERING_VECTOR_ARTIFACT_SCHEMA,
    approved_experiences,
    build_gsm8k_messages,
    entropy_quantile,
    entropy_recovery_label,
    parse_csv_numbers,
    parse_csv_strings,
)


METHODS = (
    "entropy_recovery_state_delta",
    "entropy_recovery_displacement_delta",
)


@dataclass(frozen=True)
class TrajectoryTokens:
    ids: list[int]
    boundaries: list[int]


@dataclass(frozen=True)
class TokenizedPair:
    experience: dict[str, Any]
    target: TrajectoryTokens
    reference: TrajectoryTokens


@dataclass(frozen=True)
class RecoveryEvent:
    pair_index: int
    side: str
    boundary_index: int
    next_boundary_index: int
    entropy: float
    next_entropy: float
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--layers", default="8,16,24")
    parser.add_argument("--experience-types", default="answer_correctness")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sink-token-count", type=int, default=4)
    parser.add_argument("--high-entropy-quantile", type=float, default=0.85)
    parser.add_argument("--low-entropy-quantile", type=float, default=0.50)
    parser.add_argument("--min-recovery-events", type=int, default=50)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    return parser.parse_args()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_context_limit(model: Any) -> int | None:
    candidates = (
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(model.config, "max_sequence_length", None),
    )
    values = [int(value) for value in candidates if isinstance(value, int) and value > 0]
    return min(values) if values else None


def is_delimiter(tokenizer: Any, token_id: int) -> bool:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).rstrip(" \t").endswith(
        (",", ".", "\n")
    )


def tokenize_trajectory(tokenizer: Any, prompt_ids: list[int], completion: str) -> TrajectoryTokens:
    ids = prompt_ids + tokenizer.encode(completion, add_special_tokens=False)
    box_offset = completion.find("\\boxed")
    if box_offset < 0:
        box_offset = completion.find("\\fbox")
    upper = len(ids) - 1
    if box_offset >= 0:
        upper = len(prompt_ids) + len(tokenizer.encode(completion[:box_offset], add_special_tokens=False)) - 1
    boundaries = [
        index
        for index in range(len(prompt_ids), min(len(ids), upper + 1))
        if is_delimiter(tokenizer, ids[index])
    ]
    return TrajectoryTokens(ids=ids, boundaries=boundaries)


def tokenize_pair(tokenizer: Any, experience: dict[str, Any]) -> TokenizedPair:
    prompt = tokenizer.apply_chat_template(
        build_gsm8k_messages(str(experience["context"])), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return TokenizedPair(
        experience=experience,
        target=tokenize_trajectory(tokenizer, prompt_ids, str(experience["trajectory"])),
        reference=tokenize_trajectory(tokenizer, prompt_ids, str(experience["reference_trajectory"])),
    )


def pad_batch(tokenizer: Any, pairs: Iterable[TokenizedPair], device: str):
    import torch

    rows: list[tuple[TokenizedPair, str, list[int]]] = []
    for pair in pairs:
        rows.extend(((pair, "target", pair.target.ids), (pair, "reference", pair.reference.ids)))
    length = max(len(ids) for _, _, ids in rows)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    input_ids, masks, offsets = [], [], []
    for _, _, ids in rows:
        pad = length - len(ids)
        input_ids.append([tokenizer.pad_token_id] * pad + ids)
        masks.append([0] * pad + [1] * len(ids))
        offsets.append(pad)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        offsets,
    )


def sink_masked_entropy_at(attention: Any, attention_mask: Any, row: int, query_index: int, sink_count: int) -> float:
    """Compute the same last-layer sink-masked entropy used by online gating."""

    import torch

    valid = attention_mask[row].nonzero(as_tuple=True)[0]
    keys = valid[sink_count:]
    if keys.numel() == 0:
        raise ValueError("Sink mask removed every attention key")
    raw = attention[row, :, query_index, :].float().index_select(1, keys)
    normalizer = raw.sum(dim=-1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise RuntimeError("Attention mass after sink masking is zero")
    probs = raw / normalizer
    entropy = -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)
    value = float(entropy.mean().item())
    if not math.isfinite(value):
        raise RuntimeError("Non-finite sink-masked entropy")
    return value


def rms(value: Any) -> float:
    return float(value.float().square().mean().sqrt().item())


def main() -> None:
    args = parse_args()
    if args.attn_implementation != "eager":
        raise ValueError("Entropy-recovery compilation requires --attn-implementation eager")
    if args.batch_size <= 0 or args.limit < 0 or args.sink_token_count < 0 or args.min_recovery_events <= 0:
        raise ValueError("Invalid batch/limit/sink/minimum configuration")
    if not 0.0 <= args.low_entropy_quantile <= args.high_entropy_quantile <= 1.0:
        raise ValueError("Require 0 <= low-entropy-quantile <= high-entropy-quantile <= 1")
    methods = parse_csv_strings(args.methods)
    if set(methods) - set(METHODS) or len(set(methods)) != len(methods):
        raise ValueError(f"Unsupported or repeated methods: {methods}")
    layers = list(parse_csv_numbers(args.layers, integer=True))
    if any(layer <= 0 for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("layers must be distinct positive block indices")
    types = parse_csv_strings(args.experience_types)
    if set(types) - PHASE2_ELIGIBLE_EXPERIENCE_TYPES:
        raise ValueError("Unsupported experience type")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected, selection_report = approved_experiences(
        list(iter_jsonl(args.approved_bank)), list(iter_jsonl(args.experiences)), allowed_experience_types=types
    )
    if args.limit:
        selected = selected[:args.limit]
        selection_report["selected_count_after_limit"] = len(selected)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, torch_dtype=dtype, attn_implementation="eager"
    ).to(args.device)
    model.eval()
    context_limit = args.max_sequence_length or model_context_limit(model)
    pairs: list[TokenizedPair] = []
    trace: list[dict[str, Any]] = []
    for experience in selected:
        pair = tokenize_pair(tokenizer, experience)
        if context_limit and max(len(pair.target.ids), len(pair.reference.ids)) > context_limit:
            trace.append({"experience_id": experience["experience_id"], "status": "skipped_context_limit"})
            continue
        pairs.append(pair)
    if not pairs:
        raise RuntimeError("No pairs remain within the model context limit")

    # Pass 1: identify high-entropy conditions using only bank-source states.
    entropy_by_trajectory: dict[tuple[int, str], list[float]] = {}
    all_entropies: list[float] = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        ids, mask, offsets = pad_batch(tokenizer, batch, args.device)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False, return_dict=True)
        attentions = output.attentions
        if not attentions or attentions[-1] is None:
            raise RuntimeError("Model did not return eager attention weights")
        final_attention = attentions[-1]
        for offset, pair in enumerate(batch):
            for row_offset, side, trajectory in (
                (offset * 2, "target", pair.target),
                (offset * 2 + 1, "reference", pair.reference),
            ):
                values = [
                    sink_masked_entropy_at(final_attention, mask, row_offset, offsets[row_offset] + boundary, args.sink_token_count)
                    for boundary in trajectory.boundaries
                ]
                entropy_by_trajectory[(start + offset, side)] = values
                all_entropies.extend(values)
        if start and start % max(args.batch_size * 50, 1) == 0:
            print(f"[entropy-recovery] entropy pass {start}/{len(pairs)}", flush=True)
    if not all_entropies:
        raise RuntimeError("No pre-boxed delimiter boundaries were available for entropy recovery")
    high_threshold = entropy_quantile(all_entropies, args.high_entropy_quantile)
    low_threshold = entropy_quantile(all_entropies, args.low_entropy_quantile)

    events: list[RecoveryEvent] = []
    labels: Counter[str] = Counter()
    selected_pair_indices: set[int] = set()
    for pair_index, pair in enumerate(pairs):
        for side, trajectory in (("target", pair.target), ("reference", pair.reference)):
            values = entropy_by_trajectory[(pair_index, side)]
            for index in range(len(trajectory.boundaries) - 1):
                label = entropy_recovery_label(
                    side=side,
                    current_entropy=values[index],
                    next_entropy=values[index + 1],
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                )
                trace.append({
                    "experience_id": pair.experience["experience_id"],
                    "side": side,
                    "boundary_rank": index,
                    "boundary_token_index": trajectory.boundaries[index],
                    "next_boundary_token_index": trajectory.boundaries[index + 1],
                    "entropy": values[index],
                    "next_entropy": values[index + 1],
                    "label": label,
                    "status": "candidate",
                })
                if label is not None:
                    labels[label] += 1
                    selected_pair_indices.add(pair_index)
                    events.append(RecoveryEvent(
                        pair_index=pair_index,
                        side=side,
                        boundary_index=trajectory.boundaries[index],
                        next_boundary_index=trajectory.boundaries[index + 1],
                        entropy=values[index],
                        next_entropy=values[index + 1],
                        label=label,
                    ))
    trace_path = args.output_dir.expanduser() / "entropy_recovery_evidence_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(trace_path, trace)
    required_labels = ("successful_recovery", "failed_persistence")
    if any(labels[label] < args.min_recovery_events for label in required_labels):
        raise RuntimeError(
            "Insufficient entropy-recovery evidence: "
            + ", ".join(f"{label}={labels[label]}" for label in required_labels)
            + f" required={args.min_recovery_events}"
        )

    # Pass 2: only now extract hidden states from the same condition-matched boundaries.
    state_sums: dict[str, dict[int, Any]] = {label: {} for label in required_labels}
    displacement_sums: dict[str, dict[int, Any]] = {label: {} for label in required_labels}

    def initialize(hidden_size: int) -> None:
        if state_sums["successful_recovery"]:
            return
        for label in required_labels:
            for layer in layers:
                state_sums[label][layer] = torch.zeros(hidden_size, dtype=torch.float64)
                displacement_sums[label][layer] = torch.zeros(hidden_size, dtype=torch.float64)

    events_by_pair: dict[int, list[RecoveryEvent]] = {}
    for event in events:
        events_by_pair.setdefault(event.pair_index, []).append(event)
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        ids, mask, offsets = pad_batch(tokenizer, batch, args.device)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True)
        states = output.hidden_states
        if states is None or any(layer >= len(states) for layer in layers):
            raise RuntimeError("Model hidden states do not cover requested layers")
        initialize(int(states[layers[0]].shape[-1]))
        for local_index, _pair in enumerate(batch):
            pair_index = start + local_index
            for event in events_by_pair.get(pair_index, []):
                row = local_index * 2 + (0 if event.side == "target" else 1)
                for layer in layers:
                    current = states[layer][row, offsets[row] + event.boundary_index].float()
                    following = states[layer][row, offsets[row] + event.next_boundary_index].float()
                    state_sums[event.label][layer] += current.detach().to(dtype=torch.float64, device="cpu")
                    displacement_sums[event.label][layer] += (following - current).detach().to(dtype=torch.float64, device="cpu")
        if start and start % max(args.batch_size * 50, 1) == 0:
            print(f"[entropy-recovery] state pass {start}/{len(pairs)}", flush=True)

    vectors: dict[str, dict[int, Any]] = {}
    if "entropy_recovery_state_delta" in methods:
        vectors["entropy_recovery_state_delta"] = {
            layer: (
                state_sums["successful_recovery"][layer] / labels["successful_recovery"]
                - state_sums["failed_persistence"][layer] / labels["failed_persistence"]
            ).float()
            for layer in layers
        }
    if "entropy_recovery_displacement_delta" in methods:
        vectors["entropy_recovery_displacement_delta"] = {
            layer: (
                displacement_sums["successful_recovery"][layer] / labels["successful_recovery"]
                - displacement_sums["failed_persistence"][layer] / labels["failed_persistence"]
            ).float()
            for layer in layers
        }
    if any(rms(vector) <= 0 or not torch.isfinite(vector).all() for method in vectors.values() for vector in method.values()):
        raise RuntimeError("A final entropy-recovery vector is non-finite or zero")

    output_dir = args.output_dir.expanduser()
    common = {
        "schema_version": STEERING_VECTOR_ARTIFACT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reasoner": {
            "model_name": args.model,
            "model_revision": str(getattr(model.config, "_commit_hash", None) or args.model_revision),
            "tokenizer_revision": str(getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or args.model_revision),
        },
        "construction": {
            "source": "frozen_phase1_ai_approved_bank_no_phase2_ai",
            "boundary_definition": "preboxed_delimiter_with_next_reasoning_boundary",
            "condition": "sink_masked_high_entropy_then_successful_recovery_vs_failed_persistence",
            "sink_token_count": args.sink_token_count,
            "high_entropy_quantile": args.high_entropy_quantile,
            "high_entropy_threshold": high_threshold,
            "low_entropy_quantile": args.low_entropy_quantile,
            "low_entropy_threshold": low_threshold,
            "layers": layers,
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.experiences),
            "selection": selection_report,
            "compiler_git_revision": git_revision(),
        },
    }
    artifacts = []
    source_episode_ids = [
        {"target": pairs[index].experience["target_episode_id"], "reference": pairs[index].experience["reference_episode_id"]}
        for index in sorted(selected_pair_indices)
    ]
    for method, method_vectors in vectors.items():
        artifact_id = f"phase2-{method}-answer_correctness"
        artifact_path = output_dir / f"{artifact_id}.pt"
        payload = {
            **common,
            "artifact_id": artifact_id,
            "experience_type": "answer_correctness",
            "construction": {**common["construction"], "method": method},
            "evidence_count": min(labels[label] for label in required_labels),
            "recovery_event_counts": dict(sorted(labels.items())),
            "vectors": method_vectors,
            "vector_rms": {str(layer): rms(vector) for layer, vector in method_vectors.items()},
            "source_episode_ids": source_episode_ids,
        }
        torch.save(payload, artifact_path)
        artifacts.append({
            "method": method,
            "artifact_id": artifact_id,
            "path": artifact_path.name,
            "sha256": file_sha256(artifact_path),
            "vector_rms": payload["vector_rms"],
        })
    report = {
        "schema_version": "phase2-entropy-recovery-compilation-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_pair_count": len(selected),
        "context_eligible_pair_count": len(pairs),
        "candidate_boundary_count": len(all_entropies),
        "entropy_thresholds": {
            "high_quantile": args.high_entropy_quantile,
            "high_threshold": high_threshold,
            "low_quantile": args.low_entropy_quantile,
            "low_threshold": low_threshold,
        },
        "recovery_event_counts": dict(sorted(labels.items())),
        "artifacts": artifacts,
        "evidence_trace": {"path": trace_path.name, "sha256": file_sha256(trace_path)},
        "inputs": common["inputs"],
    }
    report_path = output_dir / "entropy_recovery_compilation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[entropy-recovery] artifacts={len(artifacts)} report={report_path}", flush=True)


if __name__ == "__main__":
    main()
