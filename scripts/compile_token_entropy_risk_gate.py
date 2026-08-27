#!/usr/bin/env python3
"""Compile the V3.4 every-token entropy-risk gate artifact.

The compiler is answer-blind after the approved trajectory bank is selected.
It splits whole experiences, derives entropy thresholds and a recovery horizon
from bank-train only, fits layer-24 recovery/persistence prototypes, and
qualifies them on the held-out experience partition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.phase1 import file_sha256, iter_jsonl, write_jsonl
from memgen.experience.risk import (
    TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA,
    approved_experiences,
    binary_average_precision,
    binary_balanced_accuracy,
    binary_roc_auc,
    deterministic_train_partition,
    entropy_quantile,
    select_recovery_horizon,
    stable_low_recovery_offset,
    token_entropy_transition_label,
)


REPORT_SCHEMA = "token-entropy-risk-diagnostic-v3.4"


@dataclass(frozen=True)
class TrajectoryTokens:
    ids: list[int]
    reasoning_indices: list[int]


@dataclass(frozen=True)
class TokenizedPair:
    experience: dict[str, Any]
    target: TrajectoryTokens
    reference: TrajectoryTokens
    is_train: bool


@dataclass(frozen=True)
class TokenEvent:
    pair_index: int
    side: str
    reasoning_rank: int
    token_index: int
    entropy: float
    vocabulary_entropy: float
    top1_top2_logit_margin: float
    recovery_offset: int | None
    label: str
    token_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--layer", type=int, default=24)
    parser.add_argument("--experience-types", default="answer_correctness")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sink-token-count", type=int, default=4)
    parser.add_argument("--high-entropy-quantile", type=float, default=0.85)
    parser.add_argument("--low-entropy-quantile", type=float, default=0.50)
    parser.add_argument("--risk-train-fraction", type=float, default=0.5)
    parser.add_argument("--risk-split-seed", type=int, default=42)
    parser.add_argument("--stable-low-token-count", type=int, default=2)
    parser.add_argument("--horizon-quantile", type=float, default=0.75)
    parser.add_argument("--maximum-recovery-horizon", type=int, default=32)
    parser.add_argument("--min-events-per-label", type=int, default=40)
    parser.add_argument("--reference-boundary-roc-auc", type=float, default=0.8026)
    parser.add_argument(
        "--reference-boundary-balanced-accuracy", type=float, default=0.7180
    )
    parser.add_argument("--allowed-reference-regression", type=float, default=0.03)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
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


def tokenize_trajectory(
    tokenizer: Any, prompt_ids: list[int], completion: str
) -> TrajectoryTokens:
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    ids = prompt_ids + completion_ids
    marker_offsets = [
        offset
        for marker in ("\\boxed", "\\fbox", "final answer", "answer is")
        if (offset := completion.lower().find(marker.lower())) >= 0
    ]
    upper = len(ids)
    if marker_offsets:
        prefix = completion[: min(marker_offsets)]
        upper = len(prompt_ids) + len(
            tokenizer.encode(prefix, add_special_tokens=False)
        )
    return TrajectoryTokens(
        ids=ids,
        reasoning_indices=list(range(len(prompt_ids), upper)),
    )


def tokenize_pair(
    tokenizer: Any, experience: dict[str, Any], *, is_train: bool
) -> TokenizedPair:
    prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(
        tokenizer, str(experience["context"])
    )
    return TokenizedPair(
        experience=experience,
        target=tokenize_trajectory(
            tokenizer, prompt_ids, str(experience["trajectory"])
        ),
        reference=tokenize_trajectory(
            tokenizer, prompt_ids, str(experience["reference_trajectory"])
        ),
        is_train=is_train,
    )


def pad_batch(
    tokenizer: Any, pairs: Iterable[TokenizedPair], device: str
) -> tuple[Any, Any, list[int]]:
    import torch

    rows: list[list[int]] = []
    for pair in pairs:
        rows.extend((pair.target.ids, pair.reference.ids))
    length = max(len(ids) for ids in rows)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    input_ids: list[list[int]] = []
    masks: list[list[int]] = []
    offsets: list[int] = []
    for ids in rows:
        pad = length - len(ids)
        input_ids.append([int(tokenizer.pad_token_id)] * pad + ids)
        masks.append([0] * pad + [1] * len(ids))
        offsets.append(pad)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(masks, dtype=torch.long, device=device),
        offsets,
    )


def token_kind(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if not text or text.isspace():
        return "whitespace"
    stripped = text.strip()
    if stripped and all(character.isdigit() or character in ".,-+" for character in stripped):
        return "numeric"
    if stripped and all(not character.isalnum() for character in stripped):
        return "punctuation"
    if stripped.startswith("<|") and stripped.endswith("|>"):
        return "special"
    return "other"


def sequence_logit_diagnostics(
    logits: Any,
    *,
    row: int,
    positions: Sequence[int],
    chunk_size: int = 32,
) -> tuple[list[float], list[float]]:
    """Compute exact vocabulary diagnostics without a full float32 copy."""

    import torch

    vocabulary_entropies: list[float] = []
    margins: list[float] = []
    for start in range(0, len(positions), chunk_size):
        selected = positions[start : start + chunk_size]
        values = logits[row, selected, :].detach().float()
        log_probabilities = torch.log_softmax(values, dim=-1)
        probabilities = log_probabilities.exp()
        vocabulary_entropies.extend(
            float(value)
            for value in (
                -(probabilities * log_probabilities).sum(dim=-1)
            ).cpu().tolist()
        )
        top2 = torch.topk(values, k=2, dim=-1).values
        margins.extend(
            float(value)
            for value in (top2[:, 0] - top2[:, 1]).cpu().tolist()
        )
    return vocabulary_entropies, margins


def cosine_score(state: Any, *, recovery: Any, persistence: Any) -> float:
    import torch

    state = state.float()
    recovery = recovery.float()
    persistence = persistence.float()
    recovery_similarity = torch.nn.functional.cosine_similarity(
        state, recovery, dim=0
    )
    persistence_similarity = torch.nn.functional.cosine_similarity(
        state, persistence, dim=0
    )
    return float((persistence_similarity - recovery_similarity).item())


def label_counts(events: Sequence[TokenEvent]) -> dict[str, int]:
    return dict(sorted(Counter(event.label for event in events).items()))


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    normalized = [float(value) for value in values]
    return {
        "count": len(normalized),
        "min": min(normalized),
        "mean": sum(normalized) / len(normalized),
        "median": entropy_quantile(normalized, 0.5),
        "p05": entropy_quantile(normalized, 0.05),
        "p95": entropy_quantile(normalized, 0.95),
        "max": max(normalized),
    }


def write_report(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.attn_implementation != "sdpa" or args.layer != 24:
        raise ValueError("V3.4 is frozen to SDPA and layer 24")
    if (
        args.batch_size <= 0
        or args.limit < 0
        or args.sink_token_count < 0
        or args.stable_low_token_count != 2
        or args.maximum_recovery_horizon <= 0
        or args.min_events_per_label <= 0
    ):
        raise ValueError("Invalid V3.4 compiler limits")
    if not 0.0 <= args.low_entropy_quantile <= args.high_entropy_quantile <= 1.0:
        raise ValueError("Entropy quantiles must satisfy 0 <= low <= high <= 1")
    if not 0.0 <= args.horizon_quantile <= 1.0:
        raise ValueError("Horizon quantile must be in [0, 1]")
    if not 0.0 <= args.allowed_reference_regression <= 1.0:
        raise ValueError("Allowed reference regression must be in [0, 1]")
    if args.experience_types != "answer_correctness":
        raise ValueError("V3.4 is restricted to approved answer_correctness data")


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from memgen.model.side_kv import SDPAAttentionEntropyObserver

    selected, selection_report = approved_experiences(
        list(iter_jsonl(args.approved_bank)),
        list(iter_jsonl(args.experiences)),
        allowed_experience_types=("answer_correctness",),
    )
    if args.limit:
        selected = selected[: args.limit]
        selection_report["selected_count_after_limit"] = len(selected)

    tokenizer_revision_request = args.tokenizer_revision or args.model_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=tokenizer_revision_request
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    context_limit = args.max_sequence_length or model_context_limit(model)

    pairs: list[TokenizedPair] = []
    skipped: list[dict[str, Any]] = []
    for experience in selected:
        pair = tokenize_pair(
            tokenizer,
            experience,
            is_train=deterministic_train_partition(
                str(experience["experience_id"]),
                seed=args.risk_split_seed,
                train_fraction=args.risk_train_fraction,
            ),
        )
        if not pair.target.reasoning_indices or not pair.reference.reasoning_indices:
            skipped.append({
                "experience_id": experience["experience_id"],
                "status": "skipped_no_pre_answer_tokens",
            })
            continue
        if context_limit and max(len(pair.target.ids), len(pair.reference.ids)) > context_limit:
            skipped.append({
                "experience_id": experience["experience_id"],
                "status": "skipped_context_limit",
            })
            continue
        pairs.append(pair)
    if not pairs or not any(pair.is_train for pair in pairs) or not any(
        not pair.is_train for pair in pairs
    ):
        raise RuntimeError("V3.4 needs non-empty train and holdout experience partitions")

    entropy_sequences: dict[tuple[int, str], list[float]] = {}
    vocabulary_sequences: dict[tuple[int, str], list[float]] = {}
    margin_sequences: dict[tuple[int, str], list[float]] = {}
    all_train_entropies: list[float] = []
    all_entropies: list[float] = []
    observer = SDPAAttentionEntropyObserver(
        model=model, sink_token_count=args.sink_token_count
    )
    try:
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start : start + args.batch_size]
            input_ids, attention_mask, offsets = pad_batch(
                tokenizer, batch, args.device
            )
            with torch.inference_mode(), observer.capture():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
            entropy_by_query = observer.observation.entropy_by_query
            for local_index, pair in enumerate(batch):
                pair_index = start + local_index
                for row, side, trajectory in (
                    (local_index * 2, "target", pair.target),
                    (local_index * 2 + 1, "reference", pair.reference),
                ):
                    positions = [offsets[row] + index for index in trajectory.reasoning_indices]
                    entropies = [
                        float(entropy_by_query[row, position].item())
                        for position in positions
                    ]
                    vocabulary_entropies, margins = (
                        sequence_logit_diagnostics(
                            output.logits,
                            row=row,
                            positions=positions,
                        )
                    )
                    if not all(math.isfinite(value) for value in (
                        entropies + vocabulary_entropies + margins
                    )):
                        raise RuntimeError("V3.4 token diagnostics contain non-finite values")
                    key = (pair_index, side)
                    entropy_sequences[key] = entropies
                    vocabulary_sequences[key] = vocabulary_entropies
                    margin_sequences[key] = margins
                    all_entropies.extend(entropies)
                    if pair.is_train:
                        all_train_entropies.extend(entropies)
    finally:
        observer.close()

    high_threshold = entropy_quantile(
        all_train_entropies, args.high_entropy_quantile
    )
    low_threshold = entropy_quantile(
        all_train_entropies, args.low_entropy_quantile
    )
    train_sequences = [
        entropy_sequences[(pair_index, side)]
        for pair_index, pair in enumerate(pairs)
        if pair.is_train
        for side in ("target", "reference")
    ]
    horizon = select_recovery_horizon(
        train_sequences,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        stable_token_count=args.stable_low_token_count,
        quantile=args.horizon_quantile,
        maximum_horizon=args.maximum_recovery_horizon,
    )
    recovery_horizon = int(horizon["recovery_horizon"])

    events: list[TokenEvent] = []
    evidence_rows: list[dict[str, Any]] = list(skipped)
    token_kind_counts: Counter[str] = Counter()
    censored_count = 0
    for pair_index, pair in enumerate(pairs):
        partition = "train" if pair.is_train else "holdout"
        for side, trajectory in (
            ("target", pair.target),
            ("reference", pair.reference),
        ):
            entropies = entropy_sequences[(pair_index, side)]
            vocab_entropies = vocabulary_sequences[(pair_index, side)]
            margins = margin_sequences[(pair_index, side)]
            for rank, token_index in enumerate(trajectory.reasoning_indices):
                label = token_entropy_transition_label(
                    entropies,
                    current_index=rank,
                    high_threshold=high_threshold,
                    low_threshold=low_threshold,
                    recovery_horizon=recovery_horizon,
                    stable_token_count=args.stable_low_token_count,
                )
                kind = token_kind(tokenizer, trajectory.ids[token_index])
                is_high = entropies[rank] >= high_threshold
                recovery_offset = None
                if label == "recovery":
                    recovery_offset = stable_low_recovery_offset(
                        entropies,
                        current_index=rank,
                        low_threshold=low_threshold,
                        stable_token_count=args.stable_low_token_count,
                        maximum_horizon=recovery_horizon,
                    )
                if is_high:
                    token_kind_counts[kind] += 1
                    if label is None:
                        censored_count += 1
                evidence_rows.append({
                    "experience_id": pair.experience["experience_id"],
                    "side": side,
                    "partition": partition,
                    "reasoning_rank": rank,
                    "token_index": token_index,
                    "token_id": int(trajectory.ids[token_index]),
                    "token_text": tokenizer.decode(
                        [int(trajectory.ids[token_index])],
                        skip_special_tokens=False,
                    ),
                    "token_kind": kind,
                    "attention_entropy": entropies[rank],
                    "vocabulary_entropy": vocab_entropies[rank],
                    "top1_top2_logit_margin": margins[rank],
                    "high_entropy_candidate": is_high,
                    "transition_label": label,
                    "recovery_offset": recovery_offset,
                    "status": (
                        "labeled" if label is not None else
                        "right_censored" if is_high else "below_high_threshold"
                    ),
                })
                if label is not None:
                    events.append(TokenEvent(
                        pair_index=pair_index,
                        side=side,
                        reasoning_rank=rank,
                        token_index=token_index,
                        entropy=entropies[rank],
                        vocabulary_entropy=vocab_entropies[rank],
                        top1_top2_logit_margin=margins[rank],
                        recovery_offset=recovery_offset,
                        label=label,
                        token_kind=kind,
                    ))

    output_dir = args.output_dir.expanduser()
    trace_path = output_dir / "token_entropy_risk_evidence.jsonl"
    write_jsonl(trace_path, evidence_rows)
    train_events = [event for event in events if pairs[event.pair_index].is_train]
    holdout_events = [event for event in events if not pairs[event.pair_index].is_train]
    event_counts = {
        "train": label_counts(train_events),
        "holdout": label_counts(holdout_events),
    }
    insufficient = [
        f"{partition}_{label}={counts.get(label, 0)}"
        for partition, counts in event_counts.items()
        for label in ("recovery", "persistence")
        if counts.get(label, 0) < args.min_events_per_label
    ]
    base_report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "compiling",
        "selected_pair_count": len(selected),
        "context_eligible_pair_count": len(pairs),
        "pre_answer_token_count": len(all_entropies),
        "high_token_kind_counts": dict(sorted(token_kind_counts.items())),
        "right_censored_high_token_count": censored_count,
        "entropy_thresholds": {
            "source_partition": "bank-train",
            "high_quantile": args.high_entropy_quantile,
            "high_threshold": high_threshold,
            "low_quantile": args.low_entropy_quantile,
            "low_threshold": low_threshold,
        },
        "recovery_horizon": horizon,
        "event_counts": event_counts,
        "diagnostic_summaries": {
            "attention_entropy": numeric_summary(all_entropies),
            "labeled_vocabulary_entropy": numeric_summary(
                [event.vocabulary_entropy for event in events]
            ),
            "labeled_top1_top2_logit_margin": numeric_summary(
                [event.top1_top2_logit_margin for event in events]
            ),
        },
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "evidence_trace": {
            "path": trace_path.name,
            "sha256": file_sha256(trace_path),
            "row_count": len(evidence_rows),
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.experiences),
            "selection": selection_report,
            "compiler_git_revision": git_revision(),
            "risk_split_seed": args.risk_split_seed,
            "risk_train_fraction": args.risk_train_fraction,
            "layer": args.layer,
            "attention_implementation": args.attn_implementation,
            "tokenizer_revision_request": tokenizer_revision_request,
        },
    }
    report_path = output_dir / "token_entropy_risk_report.json"
    if insufficient:
        base_report.update({
            "status": "insufficient_evidence",
            "failure_reason": ", ".join(insufficient),
        })
        write_report(report_path, base_report)
        raise RuntimeError(
            f"Insufficient V3.4 token-risk evidence: {base_report['failure_reason']}"
        )

    events_by_pair: dict[int, list[TokenEvent]] = defaultdict(list)
    for event in events:
        events_by_pair[event.pair_index].append(event)
    states_by_event: dict[TokenEvent, Any] = {}
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start : start + args.batch_size]
        input_ids, attention_mask, offsets = pad_batch(
            tokenizer, batch, args.device
        )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        if output.hidden_states is None or args.layer >= len(output.hidden_states):
            raise RuntimeError("Model hidden states do not cover layer 24")
        states = output.hidden_states[args.layer]
        for local_index, _pair in enumerate(batch):
            pair_index = start + local_index
            for event in events_by_pair.get(pair_index, []):
                row = local_index * 2 + (0 if event.side == "target" else 1)
                states_by_event[event] = states[
                    row, offsets[row] + event.token_index
                ].detach().float().cpu()

    recovery_train = [
        states_by_event[event]
        for event in train_events
        if event.label == "recovery"
    ]
    persistence_train = [
        states_by_event[event]
        for event in train_events
        if event.label == "persistence"
    ]
    recovery_center = torch.stack(recovery_train).mean(dim=0)
    persistence_center = torch.stack(persistence_train).mean(dim=0)
    holdout_scores = [
        cosine_score(
            states_by_event[event],
            recovery=recovery_center,
            persistence=persistence_center,
        )
        for event in holdout_events
    ]
    holdout_labels = [event.label == "persistence" for event in holdout_events]
    predictions = [score > 0.0 for score in holdout_scores]
    auc = binary_roc_auc(holdout_labels, holdout_scores)
    balanced_accuracy = binary_balanced_accuracy(holdout_labels, predictions)
    average_precision = binary_average_precision(holdout_labels, holdout_scores)
    prevalence = sum(holdout_labels) / len(holdout_labels)
    minimum_auc = (
        args.reference_boundary_roc_auc - args.allowed_reference_regression
    )
    minimum_balanced_accuracy = (
        args.reference_boundary_balanced_accuracy
        - args.allowed_reference_regression
    )
    diagnostic = {
        "risk_score": (
            "cosine(current,persistence_center) - "
            "cosine(current,recovery_center)"
        ),
        "risk_threshold": 0.0,
        "heldout_roc_auc": auc,
        "heldout_average_precision": average_precision,
        "heldout_persistence_prevalence": prevalence,
        "heldout_average_precision_lift_over_prevalence": (
            average_precision - prevalence
        ),
        "heldout_balanced_accuracy_at_zero": balanced_accuracy,
        "heldout_event_count": len(holdout_events),
        "heldout_persistence_count": sum(holdout_labels),
        "heldout_recovery_count": len(holdout_labels) - sum(holdout_labels),
        "reference_boundary_roc_auc": args.reference_boundary_roc_auc,
        "reference_boundary_balanced_accuracy": (
            args.reference_boundary_balanced_accuracy
        ),
        "allowed_reference_regression": args.allowed_reference_regression,
        "minimum_heldout_roc_auc": minimum_auc,
        "minimum_heldout_balanced_accuracy": minimum_balanced_accuracy,
    }
    qualification = {
        "passed": (
            auc >= minimum_auc
            and balanced_accuracy >= minimum_balanced_accuracy
        ),
        "heldout_roc_auc_passed": auc >= minimum_auc,
        "heldout_balanced_accuracy_passed": (
            balanced_accuracy >= minimum_balanced_accuracy
        ),
        "minimum_events_per_partition_label_passed": True,
    }
    base_report["risk_diagnostic"] = diagnostic
    base_report["qualification"] = qualification
    if not qualification["passed"]:
        base_report.update({
            "status": "not_qualified",
            "failure_reason": "heldout_token_risk_below_reference_tolerance",
        })
        write_report(report_path, base_report)
        raise RuntimeError(
            "V3.4 token-risk artifact did not meet both held-out metrics"
        )

    artifact_path = output_dir / "token-entropy-risk-gate-v3.4.pt"
    artifact = {
        "schema_version": TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA,
        "artifact_id": "token-entropy-risk-gate-v3.4-answer-correctness",
        "status": "passed",
        "experience_type": "answer_correctness",
        "created_at": utc_now(),
        "reasoner": {
            "model_name": args.model,
            "model_revision": str(
                getattr(model.config, "_commit_hash", None)
                or args.model_revision
            ),
            "tokenizer_revision": str(
                getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
                or tokenizer_revision_request
            ),
            "attention_implementation": args.attn_implementation,
        },
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "construction": {
            "source": "frozen_phase1_ai_approved_bank_no_additional_ai",
            "method": "token_entropy_risk_prototype_classifier",
            "fixed_layer": args.layer,
            "observation_scope": "every_pre_answer_generated_token",
            "query_pooling": "current_generated_token",
            "label_policy": "stable_low_recovery_within_frozen_horizon",
            "right_censoring_policy": "exclude_without_full_horizon_or_recovery",
            "stable_low_token_count": args.stable_low_token_count,
            "recovery_horizon": recovery_horizon,
            "horizon_selection": horizon,
            "minimum_events_per_partition_label": args.min_events_per_label,
            "sink_token_count": args.sink_token_count,
            "high_entropy_quantile": args.high_entropy_quantile,
            "high_entropy_threshold": high_threshold,
            "low_entropy_quantile": args.low_entropy_quantile,
            "low_entropy_threshold": low_threshold,
            "online_effect_timing": "observed_token_t_affects_token_t_plus_1",
        },
        "risk_gate": {
            "layer": args.layer,
            "recovery_center": recovery_center,
            "persistence_center": persistence_center,
            "score_definition": diagnostic["risk_score"],
            "threshold": 0.0,
            "fit_partition": "bank-train",
            "heldout_diagnostic": diagnostic,
        },
        "qualification": qualification,
        "evidence_count": min(len(recovery_train), len(persistence_train)),
        "event_counts": event_counts,
        "token_diagnostics": base_report["diagnostic_summaries"],
        "source_episode_ids": [
            {
                "target": pair.experience["target_episode_id"],
                "reference": pair.experience["reference_episode_id"],
            }
            for pair in pairs
            if pair.is_train
        ],
        "inputs": base_report["inputs"],
    }
    torch.save(artifact, artifact_path)
    base_report.update({
        "status": "passed",
        "artifact": {
            "path": artifact_path.name,
            "sha256": file_sha256(artifact_path),
        },
    })
    write_report(report_path, base_report)
    print(
        f"[v3.4-token-risk] passed auc={auc:.4f} "
        f"balanced_accuracy={balanced_accuracy:.4f} "
        f"horizon={recovery_horizon} artifact={artifact_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
