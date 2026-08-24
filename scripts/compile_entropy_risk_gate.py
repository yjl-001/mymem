#!/usr/bin/env python3
"""Compile the qualified high-entropy persistence-risk gate artifact.

The serialized schema remains stable so qualified server artifacts stay
loadable. The compiler emits risk prototypes and diagnostics only; it does not
authorize or evaluate a residual-vector intervention.
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
from memgen.experience.risk import (
    ENTROPY_RISK_ARTIFACT_SCHEMA,
    approved_experiences,
    binary_average_precision,
    binary_roc_auc,
    build_gsm8k_messages,
    deterministic_train_partition,
    entropy_quantile,
    entropy_transition_label,
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
    parser.add_argument("--experience-types", default="answer_correctness")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sink-token-count", type=int, default=4)
    parser.add_argument("--high-entropy-quantile", type=float, default=0.85)
    parser.add_argument("--low-entropy-quantile", type=float, default=0.50)
    parser.add_argument("--risk-train-fraction", type=float, default=0.5)
    parser.add_argument("--risk-split-seed", type=int, default=42)
    parser.add_argument("--min-events-per-label", type=int, default=50)
    parser.add_argument("--min-heldout-roc-auc", type=float, default=0.60)
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


def cosine_score(state: Any, *, recovery: Any, persistence: Any) -> float:
    import torch

    state = state.float()
    recovery = recovery.float()
    persistence = persistence.float()
    recovery_similarity = torch.nn.functional.cosine_similarity(state, recovery, dim=0)
    persistence_similarity = torch.nn.functional.cosine_similarity(state, persistence, dim=0)
    return float((persistence_similarity - recovery_similarity).item())


def count_labels(events: Iterable[TransitionEvent]) -> dict[str, int]:
    return dict(sorted(Counter(event.label for event in events).items()))


def write_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.attn_implementation != "eager":
        raise ValueError("Entropy-risk compilation requires --attn-implementation eager")
    if args.layer <= 0 or args.batch_size <= 0 or args.limit < 0 or args.sink_token_count < 0:
        raise ValueError("Invalid layer/batch/limit/sink configuration")
    if args.min_events_per_label <= 0 or not 0.0 <= args.min_heldout_roc_auc <= 1.0:
        raise ValueError("Invalid minimum risk-diagnostic requirement")
    if not 0.0 <= args.low_entropy_quantile <= args.high_entropy_quantile <= 1.0:
        raise ValueError("Require 0 <= low-entropy-quantile <= high-entropy-quantile <= 1")
    if args.experience_types != "answer_correctness":
        raise ValueError("The minimal risk diagnostic is deliberately restricted to answer_correctness")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected, selection_report = approved_experiences(
        list(iter_jsonl(args.approved_bank)),
        list(iter_jsonl(args.experiences)),
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
        args.model,
        revision=args.model_revision,
        dtype=dtype,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()
    context_limit = args.max_sequence_length or model_context_limit(model)
    pairs: list[TokenizedPair] = []
    trace: list[dict[str, Any]] = []
    for experience in selected:
        pair = tokenize_pair(
            tokenizer,
            experience,
            is_train=deterministic_train_partition(
                str(experience["experience_id"]), seed=args.risk_split_seed, train_fraction=args.risk_train_fraction
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
    all_entropies: list[float] = []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        ids, mask, offsets = pad_batch(tokenizer, batch, args.device)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False, return_dict=True)
        if not output.attentions or output.attentions[-1] is None:
            raise RuntimeError("Model did not return eager attention weights")
        final_attention = output.attentions[-1]
        for offset, pair in enumerate(batch):
            for row, side, trajectory in ((offset * 2, "target", pair.target), (offset * 2 + 1, "reference", pair.reference)):
                values = [
                    sink_masked_entropy_at(final_attention, mask, row, offsets[row] + boundary, args.sink_token_count)
                    for boundary in trajectory.boundaries
                ]
                entropy_by_trajectory[(start + offset, side)] = values
                all_entropies.extend(values)
                if pair.is_train:
                    train_entropies.extend(values)
    if not train_entropies:
        raise RuntimeError("No training delimiter boundaries were available for the risk diagnostic")
    high_threshold = entropy_quantile(train_entropies, args.high_entropy_quantile)
    low_threshold = entropy_quantile(train_entropies, args.low_entropy_quantile)

    events: list[TransitionEvent] = []
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
                record = {
                    "experience_id": pair.experience["experience_id"], "side": side, "partition": partition,
                    "boundary_rank": rank, "boundary_token_index": trajectory.boundaries[rank],
                    "next_boundary_token_index": trajectory.boundaries[rank + 1], "entropy": values[rank],
                    "next_entropy": values[rank + 1], "transition_label": label, "status": "candidate",
                }
                trace.append(record)
                if label is not None:
                    four_cells[(side, label)] += 1
                    events.append(TransitionEvent(
                        pair_index=pair_index, side=side, boundary_rank=rank,
                        boundary_index=trajectory.boundaries[rank], next_boundary_index=trajectory.boundaries[rank + 1],
                        entropy=values[rank], next_entropy=values[rank + 1], label=label,
                    ))
    output_dir = args.output_dir.expanduser()
    trace_path = output_dir / "entropy_risk_evidence_trace.jsonl"
    write_jsonl(trace_path, trace)
    train_events = [event for event in events if pairs[event.pair_index].is_train]
    holdout_events = [event for event in events if not pairs[event.pair_index].is_train]
    event_counts = {"train": count_labels(train_events), "holdout": count_labels(holdout_events)}
    four_cell_counts = {
        side: {label: int(four_cells[(side, label)]) for label in ("recovery", "persistence")}
        for side in ("target", "reference")
    }
    insufficient = [
        f"{partition}_{label}={counts.get(label, 0)}"
        for partition, counts in event_counts.items()
        for label in ("recovery", "persistence")
        if counts.get(label, 0) < args.min_events_per_label
    ]
    base_report = {
        "schema_version": "entropy-risk-diagnostic-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_pair_count": len(selected), "context_eligible_pair_count": len(pairs),
        "candidate_boundary_count": len(all_entropies),
        "entropy_thresholds": {
            "source_partition": "bank-train", "high_quantile": args.high_entropy_quantile,
            "high_threshold": high_threshold, "low_quantile": args.low_entropy_quantile,
            "low_threshold": low_threshold,
        },
        "four_cell_counts": four_cell_counts,
        "event_counts": event_counts,
        "evidence_trace": {"path": trace_path.name, "sha256": file_sha256(trace_path)},
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.experiences), "selection": selection_report,
            "compiler_git_revision": git_revision(), "risk_split_seed": args.risk_split_seed,
            "risk_train_fraction": args.risk_train_fraction, "layer": args.layer,
        },
    }
    report_path = output_dir / "entropy_risk_diagnostic_report.json"
    if insufficient:
        base_report.update({"status": "insufficient_evidence", "failure_reason": ", ".join(insufficient)})
        write_report(report_path, base_report)
        raise RuntimeError(f"Insufficient entropy-risk evidence: {base_report['failure_reason']}")

    states_by_event: dict[TransitionEvent, Any] = {}
    events_by_pair: dict[int, list[TransitionEvent]] = {}
    for event in events:
        events_by_pair.setdefault(event.pair_index, []).append(event)
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        ids, mask, offsets = pad_batch(tokenizer, batch, args.device)
        with torch.inference_mode():
            output = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True)
        if output.hidden_states is None or args.layer >= len(output.hidden_states):
            raise RuntimeError("Model hidden states do not cover the fixed risk layer")
        states = output.hidden_states[args.layer]
        for local_index, _pair in enumerate(batch):
            pair_index = start + local_index
            for event in events_by_pair.get(pair_index, []):
                row = local_index * 2 + (0 if event.side == "target" else 1)
                states_by_event[event] = states[row, offsets[row] + event.boundary_index].detach().float().cpu()

    recovery_train = [states_by_event[event] for event in train_events if event.label == "recovery"]
    persistence_train = [states_by_event[event] for event in train_events if event.label == "persistence"]
    recovery_center = torch.stack(recovery_train).mean(dim=0)
    persistence_center = torch.stack(persistence_train).mean(dim=0)
    heldout_scores = [
        cosine_score(states_by_event[event], recovery=recovery_center, persistence=persistence_center)
        for event in holdout_events
    ]
    heldout_labels = [event.label == "persistence" for event in holdout_events]
    auc = binary_roc_auc(heldout_labels, heldout_scores)
    persistence_prevalence = sum(heldout_labels) / len(heldout_labels)
    average_precision = binary_average_precision(heldout_labels, heldout_scores)
    predicted = [score > 0.0 for score in heldout_scores]
    positives = sum(heldout_labels)
    negatives = len(heldout_labels) - positives
    true_positive = sum(truth and guess for truth, guess in zip(heldout_labels, predicted))
    true_negative = sum((not truth) and (not guess) for truth, guess in zip(heldout_labels, predicted))
    diagnostic = {
        "risk_score": "cosine(current,persistence_center) - cosine(current,recovery_center)",
        "risk_threshold": 0.0,
        "heldout_roc_auc": auc,
        "heldout_average_precision": average_precision,
        "heldout_persistence_prevalence": persistence_prevalence,
        "heldout_average_precision_lift_over_prevalence": average_precision - persistence_prevalence,
        "heldout_balanced_accuracy_at_zero": 0.5 * (true_positive / positives + true_negative / negatives),
        "heldout_event_count": len(holdout_events),
        "heldout_persistence_count": positives,
        "heldout_recovery_count": negatives,
        "minimum_heldout_roc_auc": args.min_heldout_roc_auc,
    }
    if auc < args.min_heldout_roc_auc:
        base_report.update({"status": "not_discriminative", "risk_diagnostic": diagnostic,
                            "failure_reason": "heldout_roc_auc_below_requirement"})
        write_report(report_path, base_report)
        raise RuntimeError(f"Held-out entropy-risk ROC AUC {auc:.4f} is below {args.min_heldout_roc_auc:.4f}")

    artifact_path = output_dir / "entropy-risk-gate-answer_correctness.pt"
    artifact = {
        "schema_version": ENTROPY_RISK_ARTIFACT_SCHEMA,
        "artifact_id": "entropy-risk-gate-answer_correctness",
        "experience_type": "answer_correctness",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reasoner": {
            "model_name": args.model,
            "model_revision": str(getattr(model.config, "_commit_hash", None) or args.model_revision),
            "tokenizer_revision": str(getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or args.model_revision),
        },
        "construction": {
            "source": "frozen_phase1_ai_approved_bank_no_additional_ai",
            "method": "entropy_risk_prototype_classifier",
            "fixed_layer": args.layer,
            "boundary_definition": "preboxed_delimiter_with_next_reasoning_boundary",
            "condition": "high_entropy_recovery_vs_persistence_all_outcomes_train_only",
            "sink_token_count": args.sink_token_count,
            "high_entropy_quantile": args.high_entropy_quantile, "high_entropy_threshold": high_threshold,
            "low_entropy_quantile": args.low_entropy_quantile, "low_entropy_threshold": low_threshold,
        },
        "risk_gate": {
            "layer": args.layer, "recovery_center": recovery_center, "persistence_center": persistence_center,
            "score_definition": diagnostic["risk_score"], "threshold": 0.0,
            "fit_partition": "bank-train", "heldout_diagnostic": diagnostic,
        },
        "evidence_count": min(len(recovery_train), len(persistence_train)),
        "recovery_event_counts": event_counts,
        "source_episode_ids": [
            {"target": pair.experience["target_episode_id"], "reference": pair.experience["reference_episode_id"]}
            for pair in pairs if pair.is_train
        ],
        "inputs": base_report["inputs"],
    }
    torch.save(artifact, artifact_path)
    base_report.update({
        "status": "passed", "risk_diagnostic": diagnostic,
        "artifact": {
            "path": artifact_path.name,
            "sha256": file_sha256(artifact_path),
        },
    })
    write_report(report_path, base_report)
    print(f"[entropy-risk] passed auc={auc:.4f} artifact={artifact_path} report={report_path}", flush=True)


if __name__ == "__main__":
    main()
