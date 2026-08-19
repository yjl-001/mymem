#!/usr/bin/env python3
"""Audit whether Phase-1 pairs can form H3 conditional local actions.

This is a read-only, offline feasibility check.  It never produces a steering
artifact, changes generation, or makes a Phase-2 model/API call.  It replays
the frozen student trajectories at layer 24, labels high-entropy transitions,
and counts experience-local pairs of:

    reference persistence state -> target recovery state.

The resulting report determines whether the H3 action-bank definition has
enough evidence to justify an implementation; it does not tune an intervention.
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
    approved_experiences,
    cosine_similarity,
    deterministic_train_partition,
    entropy_quantile,
    entropy_transition_label,
    leave_one_out_nearest_cosines,
    select_max_cosine_event_pair,
    build_gsm8k_messages,
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
    is_train: bool


@dataclass(frozen=True)
class TransitionEvent:
    pair_index: int
    side: str
    boundary_rank: int
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
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sink-token-count", type=int, default=4)
    parser.add_argument("--high-entropy-quantile", type=float, default=0.85)
    parser.add_argument("--low-entropy-quantile", type=float, default=0.50)
    parser.add_argument("--risk-train-fraction", type=float, default=0.5)
    parser.add_argument("--risk-split-seed", type=int, default=42)
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
    values = [
        int(value)
        for value in (
            getattr(model.config, "max_position_embeddings", None),
            getattr(model.config, "n_positions", None),
            getattr(model.config, "max_sequence_length", None),
        )
        if isinstance(value, int) and value > 0
    ]
    return min(values) if values else None


def is_delimiter(tokenizer: Any, token_id: int) -> bool:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).rstrip(" \t").endswith((",", ".", "\n"))


def tokenize_trajectory(tokenizer: Any, prompt_ids: list[int], completion: str) -> TrajectoryTokens:
    ids = prompt_ids + tokenizer.encode(completion, add_special_tokens=False)
    box_offset = completion.find("\\boxed")
    if box_offset < 0:
        box_offset = completion.find("\\fbox")
    upper = len(ids) - 1
    if box_offset >= 0:
        upper = len(prompt_ids) + len(tokenizer.encode(completion[:box_offset], add_special_tokens=False)) - 1
    return TrajectoryTokens(
        ids=ids,
        boundaries=[
            index for index in range(len(prompt_ids), min(len(ids), upper + 1))
            if is_delimiter(tokenizer, ids[index])
        ],
    )


def tokenize_pair(tokenizer: Any, experience: dict[str, Any], *, is_train: bool) -> TokenizedPair:
    prompt = tokenizer.apply_chat_template(
        build_gsm8k_messages(str(experience["context"])), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return TokenizedPair(
        experience=experience,
        target=tokenize_trajectory(tokenizer, prompt_ids, str(experience["trajectory"])),
        reference=tokenize_trajectory(tokenizer, prompt_ids, str(experience["reference_trajectory"])),
        is_train=is_train,
    )


def pad_batch(tokenizer: Any, pairs: Iterable[TokenizedPair], device: str):
    import torch

    rows: list[list[int]] = []
    for pair in pairs:
        rows.extend((pair.target.ids, pair.reference.ids))
    length = max(len(ids) for ids in rows)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    input_ids, masks, offsets = [], [], []
    for ids in rows:
        padding = length - len(ids)
        input_ids.append([tokenizer.pad_token_id] * padding + ids)
        masks.append([0] * padding + [1] * len(ids))
        offsets.append(padding)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        offsets,
    )


def sink_masked_entropy_at(attention: Any, attention_mask: Any, row: int, query_index: int, sink_count: int) -> float:
    import torch

    valid = attention_mask[row].nonzero(as_tuple=True)[0]
    keys = valid[sink_count:]
    if keys.numel() == 0:
        raise ValueError("Sink mask removed every attention key")
    raw = attention[row, :, query_index, :].float().index_select(1, keys)
    normalizer = raw.sum(dim=-1, keepdim=True)
    if torch.any(normalizer <= 0):
        raise RuntimeError("Attention mass after sink masking is zero")
    probabilities = raw / normalizer
    entropy = -(probabilities * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()).sum(dim=-1)
    value = float(entropy.mean().item())
    if not math.isfinite(value):
        raise RuntimeError("Non-finite sink-masked entropy")
    return value


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None, "p05": None, "p50": None, "p95": None}
    return {
        "count": len(values), "min": min(values), "mean": sum(values) / len(values), "max": max(values),
        "p05": entropy_quantile(values, 0.05), "p50": entropy_quantile(values, 0.50),
        "p95": entropy_quantile(values, 0.95),
    }


def vector_rms(vector: Any) -> float:
    import torch

    value = float(vector.float().square().mean().sqrt().item())
    if not math.isfinite(value):
        raise RuntimeError("Non-finite action RMS")
    return value


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.attn_implementation != "eager":
        raise ValueError("Conditional-action audit requires --attn-implementation eager")
    if args.layer <= 0 or args.batch_size <= 0 or args.limit < 0 or args.sink_token_count < 0:
        raise ValueError("Invalid layer/batch/limit/sink configuration")
    if not 0.0 <= args.low_entropy_quantile <= args.high_entropy_quantile <= 1.0:
        raise ValueError("Require 0 <= low-entropy-quantile <= high-entropy-quantile <= 1")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected, selection_report = approved_experiences(
        list(iter_jsonl(args.approved_bank)), list(iter_jsonl(args.experiences)),
        allowed_experience_types=("answer_correctness",),
    )
    if args.limit:
        selected = selected[: args.limit]
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
        pair = tokenize_pair(
            tokenizer, experience,
            is_train=deterministic_train_partition(
                str(experience["experience_id"]), seed=args.risk_split_seed,
                train_fraction=args.risk_train_fraction,
            ),
        )
        if context_limit and max(len(pair.target.ids), len(pair.reference.ids)) > context_limit:
            trace.append({"experience_id": experience["experience_id"], "status": "skipped_context_limit"})
            continue
        pairs.append(pair)
    if not pairs:
        raise RuntimeError("No pairs remain within the model context limit")

    entropy_by_trajectory: dict[tuple[int, str], list[float]] = {}
    train_entropies: list[float] = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        ids, mask, offsets = pad_batch(tokenizer, batch, args.device)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False, return_dict=True)
        if not output.attentions or output.attentions[-1] is None:
            raise RuntimeError("Model did not return eager attention weights")
        final_attention = output.attentions[-1]
        for local_index, pair in enumerate(batch):
            pair_index = start + local_index
            for row, side, trajectory in (
                (local_index * 2, "target", pair.target),
                (local_index * 2 + 1, "reference", pair.reference),
            ):
                values = [
                    sink_masked_entropy_at(final_attention, mask, row, offsets[row] + boundary, args.sink_token_count)
                    for boundary in trajectory.boundaries
                ]
                entropy_by_trajectory[(pair_index, side)] = values
                if pair.is_train:
                    train_entropies.extend(values)
    if not train_entropies:
        raise RuntimeError("No bank-train delimiter boundaries were available")
    high_threshold = entropy_quantile(train_entropies, args.high_entropy_quantile)
    low_threshold = entropy_quantile(train_entropies, args.low_entropy_quantile)

    events: list[TransitionEvent] = []
    eligible_events: list[TransitionEvent] = []
    events_by_pair: dict[int, list[TransitionEvent]] = {}
    four_cells: Counter[tuple[str, str]] = Counter()
    for pair_index, pair in enumerate(pairs):
        partition = "train" if pair.is_train else "holdout"
        for side, trajectory in (("target", pair.target), ("reference", pair.reference)):
            values = entropy_by_trajectory[(pair_index, side)]
            for rank in range(len(trajectory.boundaries) - 1):
                label = entropy_transition_label(
                    current_entropy=values[rank], next_entropy=values[rank + 1],
                    high_threshold=high_threshold, low_threshold=low_threshold,
                )
                if label is None:
                    continue
                event = TransitionEvent(
                    pair_index=pair_index, side=side, boundary_rank=rank,
                    boundary_index=trajectory.boundaries[rank], next_boundary_index=trajectory.boundaries[rank + 1],
                    entropy=values[rank], next_entropy=values[rank + 1], label=label,
                )
                events.append(event)
                events_by_pair.setdefault(pair_index, []).append(event)
                four_cells[(side, label)] += 1
                trace.append({
                    "experience_id": pair.experience["experience_id"], "partition": partition, "side": side,
                    "boundary_rank": rank, "boundary_token_index": event.boundary_index,
                    "next_boundary_token_index": event.next_boundary_index, "entropy": event.entropy,
                    "next_entropy": event.next_entropy, "transition_label": label, "status": "transition_event",
                })
                if (side, label) in {("reference", "persistence"), ("target", "recovery")}:
                    eligible_events.append(event)

    states_by_event: dict[TransitionEvent, Any] = {}
    eligible_by_pair: dict[int, list[TransitionEvent]] = {}
    for event in eligible_events:
        eligible_by_pair.setdefault(event.pair_index, []).append(event)
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        ids, mask, offsets = pad_batch(tokenizer, batch, args.device)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True)
        if output.hidden_states is None or args.layer >= len(output.hidden_states):
            raise RuntimeError("Model hidden states do not cover the requested layer")
        states = output.hidden_states[args.layer]
        for local_index, _pair in enumerate(batch):
            pair_index = start + local_index
            for event in eligible_by_pair.get(pair_index, []):
                row = local_index * 2 + (0 if event.side == "target" else 1)
                states_by_event[event] = states[row, offsets[row] + event.boundary_index].detach().float().cpu()

    candidates: list[dict[str, Any]] = []
    eligibility_by_partition: Counter[str] = Counter()
    for pair_index, pair in enumerate(pairs):
        reference = [
            event for event in eligible_by_pair.get(pair_index, [])
            if event.side == "reference" and event.label == "persistence"
        ]
        target = [
            event for event in eligible_by_pair.get(pair_index, [])
            if event.side == "target" and event.label == "recovery"
        ]
        partition = "train" if pair.is_train else "holdout"
        if reference:
            eligibility_by_partition[f"{partition}_reference_persistence_experiences"] += 1
        if target:
            eligibility_by_partition[f"{partition}_target_recovery_experiences"] += 1
        selected_pair = select_max_cosine_event_pair(
            [(event.boundary_rank, states_by_event[event].tolist()) for event in reference],
            [(event.boundary_rank, states_by_event[event].tolist()) for event in target],
        )
        if selected_pair is None:
            continue
        reference_rank, target_rank, alignment_similarity = selected_pair
        reference_event = next(event for event in reference if event.boundary_rank == reference_rank)
        target_event = next(event for event in target if event.boundary_rank == target_rank)
        action = states_by_event[target_event] - states_by_event[reference_event]
        action_rms = vector_rms(action)
        key = states_by_event[reference_event].tolist()
        candidate = {
            "experience_id": pair.experience["experience_id"], "partition": partition,
            "reference_boundary_rank": reference_event.boundary_rank,
            "reference_boundary_token_index": reference_event.boundary_index,
            "reference_entropy": reference_event.entropy,
            "reference_next_entropy": reference_event.next_entropy,
            "target_boundary_rank": target_event.boundary_rank,
            "target_boundary_token_index": target_event.boundary_index,
            "target_entropy": target_event.entropy,
            "target_next_entropy": target_event.next_entropy,
            "state_alignment_cosine": alignment_similarity,
            "raw_action_rms": action_rms,
            "status": "eligible_conditional_action",
            "_key": key,
        }
        candidates.append(candidate)
        trace.append({key: value for key, value in candidate.items() if key != "_key"})
        eligibility_by_partition[f"{partition}_eligible_action_experiences"] += 1

    train_candidates = [candidate for candidate in candidates if candidate["partition"] == "train"]
    holdout_candidates = [candidate for candidate in candidates if candidate["partition"] == "holdout"]
    train_keys = [candidate["_key"] for candidate in train_candidates]
    leave_one_out = leave_one_out_nearest_cosines(train_keys)
    similarity_threshold = entropy_quantile(leave_one_out, 0.05) if leave_one_out else None
    holdout_top1: list[float] = []
    holdout_accepted = 0
    for candidate in holdout_candidates:
        if not train_keys:
            break
        score = max(cosine_similarity(candidate["_key"], key) for key in train_keys)
        holdout_top1.append(score)
        if similarity_threshold is not None and score >= similarity_threshold:
            holdout_accepted += 1

    output_dir = args.output_dir.expanduser()
    trace_path = output_dir / "conditional_action_candidate_trace.jsonl"
    write_jsonl(trace_path, trace)
    report = {
        "schema_version": "phase2-conditional-action-feasibility-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if similarity_threshold is not None else "insufficient_train_candidates_for_threshold",
        "purpose": "offline_only_no_vector_artifact_no_online_injection",
        "construction": {
            "method": "same_experience_reference_persistence_to_target_recovery_contrast",
            "layer": args.layer, "candidate_limit": args.limit,
            "tie_break": "higher_alignment_cosine_then_earlier_reference_then_earlier_target",
            "key": "normalized_reference_persistence_hidden_state",
            "action": "normalized_target_recovery_hidden_state_minus_reference_persistence_hidden_state",
            "similarity_threshold": "q05_leave_one_out_nearest_train_key_cosine",
            "source": "frozen_phase1_ai_approved_answer_correctness_no_phase2_ai",
        },
        "entropy_thresholds": {
            "source_partition": "bank-train", "high_quantile": args.high_entropy_quantile,
            "high_threshold": high_threshold, "low_quantile": args.low_entropy_quantile,
            "low_threshold": low_threshold,
        },
        "selection": selection_report,
        "context_eligible_pair_count": len(pairs),
        "four_cell_counts": {
            side: {label: int(four_cells[(side, label)]) for label in ("recovery", "persistence")}
            for side in ("target", "reference")
        },
        "candidate_eligibility_counts": {
            partition: {
                "reference_persistence_experiences": eligibility_by_partition[
                    f"{partition}_reference_persistence_experiences"
                ],
                "target_recovery_experiences": eligibility_by_partition[
                    f"{partition}_target_recovery_experiences"
                ],
                "eligible_action_experiences": eligibility_by_partition[
                    f"{partition}_eligible_action_experiences"
                ],
            }
            for partition in ("train", "holdout")
        },
        "candidate_counts": {"total": len(candidates), "train": len(train_candidates), "holdout": len(holdout_candidates)},
        "state_alignment_cosine": {
            "all": summary([float(candidate["state_alignment_cosine"]) for candidate in candidates]),
            "train": summary([float(candidate["state_alignment_cosine"]) for candidate in train_candidates]),
            "holdout": summary([float(candidate["state_alignment_cosine"]) for candidate in holdout_candidates]),
        },
        "raw_action_rms": {
            "all": summary([float(candidate["raw_action_rms"]) for candidate in candidates]),
            "train": summary([float(candidate["raw_action_rms"]) for candidate in train_candidates]),
            "holdout": summary([float(candidate["raw_action_rms"]) for candidate in holdout_candidates]),
            "zero_count": sum(float(candidate["raw_action_rms"]) == 0.0 for candidate in candidates),
        },
        "retrieval_feasibility": {
            "train_leave_one_out_nearest_cosine": summary(leave_one_out),
            "similarity_threshold_q05": similarity_threshold,
            "holdout_top1_to_train_cosine": summary(holdout_top1),
            "holdout_above_threshold_count": holdout_accepted if similarity_threshold is not None else 0,
            "holdout_above_threshold_rate": (
                holdout_accepted / len(holdout_top1) if similarity_threshold is not None and holdout_top1 else None
            ),
        },
        "trace": {"path": trace_path.name, "sha256": file_sha256(trace_path)},
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.experiences),
            "audit_git_revision": git_revision(), "risk_split_seed": args.risk_split_seed,
            "risk_train_fraction": args.risk_train_fraction, "model": args.model,
            "model_revision": args.model_revision,
        },
    }
    report_path = output_dir / "conditional_action_feasibility_report.json"
    write_report(report_path, report)
    print(f"[conditional-action-audit] status={report['status']} output={report_path}")


if __name__ == "__main__":
    main()
