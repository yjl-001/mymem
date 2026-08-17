#!/usr/bin/env python3
"""Compile audited Phase 1 target/reference pairs into student-space vectors.

The compiler intentionally never feeds Teacher prose to the student model.  It
joins Pro-approved bank records back to ``verified_experiences.jsonl`` and
extracts hidden-state differences from the frozen student trajectories.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
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
    last_completion_boundary,
    parse_csv_numbers,
    parse_csv_strings,
)


@dataclass(frozen=True)
class TokenizedPair:
    experience: dict[str, Any]
    target_ids: list[int]
    target_boundary: int
    reference_ids: list[int]
    reference_boundary: int
    mechanism_cluster: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument(
        "--evidence-anchors",
        type=Path,
        help="Optional Pro-reviewed exact evidence anchors. Formal Phase 2 runs require this.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--layers",
        default="8,16,24",
        help="1-based transformer block indices, comma separated.",
    )
    parser.add_argument(
        "--experience-types",
        default=",".join(sorted(PHASE2_ELIGIBLE_EXPERIENCE_TYPES)),
        help="Approved verifier types to compile, comma separated.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected pairs")
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=0,
        help="0 uses the model context limit; overlong evidence is skipped, never truncated.",
    )
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
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(model.config, "max_sequence_length", None),
    ]
    valid = [int(value) for value in values if isinstance(value, int) and value > 0]
    return min(valid) if valid else None


def tokenize_pair(
    tokenizer: Any, experience: dict[str, Any], anchor_record: dict[str, Any] | None = None
) -> TokenizedPair | None:
    prompt = tokenizer.apply_chat_template(
        build_gsm8k_messages(str(experience["context"])),
        tokenize=False,
        add_generation_prompt=True,
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)

    def completion_ids(value: str, quote: str | None) -> tuple[list[int], int | None]:
        ids = prompt_ids + tokenizer.encode(value, add_special_tokens=False)
        if anchor_record is not None and not isinstance(quote, str):
            return ids, None
        if quote is not None:
            quote_start = value.find(quote)
            if quote_start < 0 or value.find(quote, quote_start + len(quote)) >= 0:
                return ids, None
            prefix = value[: quote_start + len(quote)]
            boundary = len(prompt_ids) + len(tokenizer.encode(prefix, add_special_tokens=False)) - 1
            if boundary < len(prompt_ids) or boundary >= len(ids):
                return ids, None
            token_text = tokenizer.decode([ids[boundary]], skip_special_tokens=False)
            if not token_text.rstrip(" \t").endswith((",", ".", "\n")):
                return ids, None
            return ids, boundary
        boundary = last_completion_boundary(
            ids,
            completion_start=len(prompt_ids),
            decode_token=lambda token_id: tokenizer.decode(
                [token_id], skip_special_tokens=False
            ),
        )
        return ids, boundary

    review = anchor_record.get("anchor_review", {}) if anchor_record else {}
    target_anchor = review.get("target_anchor") if isinstance(review, dict) else None
    reference_anchor = review.get("reference_anchor") if isinstance(review, dict) else None
    target_quote = target_anchor.get("quote") if isinstance(target_anchor, dict) else None
    reference_quote = reference_anchor.get("quote") if isinstance(reference_anchor, dict) else None
    target_ids, target_boundary = completion_ids(str(experience["trajectory"]), target_quote)
    reference_ids, reference_boundary = completion_ids(
        str(experience["reference_trajectory"]), reference_quote
    )
    if target_boundary is None or reference_boundary is None:
        return None
    return TokenizedPair(
        experience=experience,
        target_ids=target_ids,
        target_boundary=target_boundary,
        reference_ids=reference_ids,
        reference_boundary=reference_boundary,
        mechanism_cluster=(
            str(review.get("mechanism_cluster")) if isinstance(review, dict) else None
        ),
    )


def pad_batch(tokenizer: Any, pairs: Iterable[TokenizedPair], device: str):
    """Left-pad target/reference sequences and retain their boundary indices."""

    import torch

    rows: list[tuple[TokenizedPair, str, list[int], int]] = []
    for pair in pairs:
        rows.extend(
            [
                (pair, "target", pair.target_ids, pair.target_boundary),
                (pair, "reference", pair.reference_ids, pair.reference_boundary),
            ]
        )
    max_length = max(len(ids) for _, _, ids, _ in rows)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    input_ids = []
    attention_mask = []
    boundaries = []
    for _, _, ids, boundary in rows:
        pad = max_length - len(ids)
        input_ids.append([pad_token_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))
        boundaries.append(pad + boundary)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        boundaries,
        rows,
    )


def rms(vector: Any) -> float:
    return float(vector.float().square().mean().sqrt().item())


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.limit < 0 or args.max_sequence_length < 0:
        raise ValueError("batch-size must be positive; limit/max-sequence-length must be non-negative")
    layers = list(parse_csv_numbers(args.layers, integer=True))
    if any(layer <= 0 for layer in layers) or len(set(layers)) != len(layers):
        raise ValueError("layers must be distinct positive 1-based transformer block indices")
    requested_types = parse_csv_strings(args.experience_types)
    unsupported = set(requested_types) - PHASE2_ELIGIBLE_EXPERIENCE_TYPES
    if unsupported:
        raise ValueError(f"Unsupported Phase 2 experience type(s): {sorted(unsupported)}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    approved = list(iter_jsonl(args.approved_bank))
    experience_rows = list(iter_jsonl(args.experiences))
    selected, selection_report = approved_experiences(
        approved,
        experience_rows,
        allowed_experience_types=requested_types,
    )
    if args.limit:
        selected = selected[: args.limit]
        selection_report["selected_count_after_limit"] = len(selected)
    if not selected:
        raise ValueError("No selected experiences")
    anchors_by_id: dict[str, dict[str, Any]] = {}
    if args.evidence_anchors is not None:
        for anchor in iter_jsonl(args.evidence_anchors):
            experience_id = str(anchor.get("experience_id", ""))
            if not experience_id or experience_id in anchors_by_id:
                raise ValueError(
                    f"Missing or duplicate evidence-anchor experience_id: {experience_id!r}"
                )
            anchors_by_id[experience_id] = anchor
        for experience in selected:
            anchor = anchors_by_id.get(str(experience["experience_id"]))
            if anchor is not None and anchor.get("route") == "anchored":
                if anchor.get("experience_provenance_sha256") != experience.get("provenance_sha256"):
                    raise ValueError(
                        f"Evidence anchor provenance mismatch for {experience['experience_id']}"
                    )

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    ).to(args.device)
    model.eval()

    resolved_revision = str(getattr(model.config, "_commit_hash", None) or args.model_revision)
    tokenizer_revision = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash") or args.model_revision
    )
    context_limit = args.max_sequence_length or model_context_limit(model)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = output_dir / "compiler_evidence_trace.jsonl"

    cluster_variants = sorted(
        {
            f"mechanism-{record['anchor_review']['mechanism_cluster']}"
            for record in anchors_by_id.values()
            if record.get("route") == "anchored"
            and isinstance(record.get("anchor_review"), dict)
            and isinstance(record["anchor_review"].get("mechanism_cluster"), str)
        }
    )
    vector_variants = ["all_selected", *requested_types, *cluster_variants]
    sums: dict[str, dict[int, Any]] = {}
    counts: Counter[str] = Counter()
    trace: list[dict[str, Any]] = []
    pending: list[TokenizedPair] = []

    def initialize(hidden_size: int) -> None:
        if sums:
            return
        for vector_type in vector_variants:
            sums[vector_type] = {
                layer: torch.zeros(hidden_size, dtype=torch.float64, device="cpu")
                for layer in layers
            }

    def consume(batch: list[TokenizedPair]) -> None:
        input_ids, attention_mask, boundaries, rows = pad_batch(tokenizer, batch, args.device)
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        max_layer = len(hidden_states) - 1
        if any(layer > max_layer for layer in layers):
            raise ValueError(
                f"Requested layer exceeds model depth: requested={layers}, transformer_blocks={max_layer}"
            )
        initialize(int(hidden_states[layers[0]].shape[-1]))
        for pair_offset, pair in enumerate(batch):
            target_row = pair_offset * 2
            reference_row = target_row + 1
            vector_type = str(pair.experience["experience_type"])
            per_layer_rms: dict[str, float] = {}
            for layer in layers:
                target = hidden_states[layer][target_row, boundaries[target_row]].float()
                reference = hidden_states[layer][reference_row, boundaries[reference_row]].float()
                delta = target - reference
                delta_rms = rms(delta)
                if not torch.isfinite(delta).all() or delta_rms == 0.0:
                    raise RuntimeError(
                        f"Non-finite or zero hidden-state difference for {pair.experience['experience_id']} layer {layer}"
                    )
                normalized = (delta / delta_rms).detach().to(dtype=torch.float64, device="cpu")
                sums["all_selected"][layer] += normalized
                sums[vector_type][layer] += normalized
                if pair.mechanism_cluster:
                    sums[f"mechanism-{pair.mechanism_cluster}"][layer] += normalized
                per_layer_rms[str(layer)] = delta_rms
            counts["all_selected"] += 1
            counts[vector_type] += 1
            if pair.mechanism_cluster:
                counts[f"mechanism-{pair.mechanism_cluster}"] += 1
            trace.append(
                {
                    "experience_id": pair.experience["experience_id"],
                    "experience_type": vector_type,
                    "source_episode_ids": {
                        "target": pair.experience["target_episode_id"],
                        "reference": pair.experience["reference_episode_id"],
                    },
                    "provenance_sha256": pair.experience["provenance_sha256"],
                    "target_boundary_token_index": pair.target_boundary,
                    "reference_boundary_token_index": pair.reference_boundary,
                    "target_sequence_length": len(pair.target_ids),
                    "reference_sequence_length": len(pair.reference_ids),
                    "mechanism_cluster": pair.mechanism_cluster,
                    "per_layer_difference_rms": per_layer_rms,
                    "status": "compiled",
                }
            )

    for index, experience in enumerate(selected, start=1):
        anchor = anchors_by_id.get(str(experience["experience_id"])) if anchors_by_id else None
        if anchors_by_id and (anchor is None or anchor.get("route") != "anchored"):
            trace.append(
                {
                    "experience_id": experience["experience_id"],
                    "experience_type": experience["experience_type"],
                    "provenance_sha256": experience["provenance_sha256"],
                    "status": "skipped_no_pro_anchored_evidence",
                }
            )
            continue
        tokenized = tokenize_pair(tokenizer, experience, anchor)
        if tokenized is None:
            trace.append(
                {
                    "experience_id": experience["experience_id"],
                    "experience_type": experience["experience_type"],
                    "provenance_sha256": experience["provenance_sha256"],
                    "status": (
                        "skipped_unusable_pro_anchored_boundary"
                        if anchors_by_id
                        else "skipped_no_completion_delimiter"
                    ),
                }
            )
            continue
        if context_limit and max(len(tokenized.target_ids), len(tokenized.reference_ids)) > context_limit:
            trace.append(
                {
                    "experience_id": experience["experience_id"],
                    "experience_type": experience["experience_type"],
                    "provenance_sha256": experience["provenance_sha256"],
                    "status": "skipped_context_limit",
                    "context_limit": context_limit,
                    "target_sequence_length": len(tokenized.target_ids),
                    "reference_sequence_length": len(tokenized.reference_ids),
                }
            )
            continue
        pending.append(tokenized)
        if len(pending) == args.batch_size:
            consume(pending)
            pending = []
        if index % 100 == 0:
            print(f"[vector-compiler] prepared {index}/{len(selected)} pairs", flush=True)
    if pending:
        consume(pending)

    write_jsonl(evidence_path, trace)
    if counts["all_selected"] == 0:
        raise RuntimeError(
            "No selected pair contained a usable completion delimiter; no steering vector was written"
        )
    artifacts: list[dict[str, Any]] = []
    common = {
        "schema_version": STEERING_VECTOR_ARTIFACT_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reasoner": {
            "model_name": args.model,
            "model_revision": resolved_revision,
            "tokenizer_revision": tokenizer_revision,
        },
        "construction": {
            "boundary_definition": (
                "pro_reviewed_exact_evidence_span_boundary"
                if args.evidence_anchors is not None
                else "last_completion_delimiter_token"
            ),
            "delimiters": [",", ".", "\\n"],
            "difference": "mean(normalize(target_hidden - reference_hidden))",
            "layers": layers,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "max_sequence_length": context_limit,
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.experiences),
            "evidence_anchors_sha256": (
                file_sha256(args.evidence_anchors) if args.evidence_anchors is not None else None
            ),
            "selection": selection_report,
            "compiler_git_revision": git_revision(),
        },
    }
    for vector_type in vector_variants:
        evidence_count = counts[vector_type]
        if evidence_count == 0:
            continue
        vectors = {
            layer: (sums[vector_type][layer] / evidence_count).to(dtype=torch.float32)
            for layer in layers
        }
        vector_rms = {str(layer): rms(vector) for layer, vector in vectors.items()}
        artifact_id = f"global-stay-on-track-{vector_type}"
        artifact_path = output_dir / f"{artifact_id}.pt"
        payload = {
            **common,
            "artifact_id": artifact_id,
            "experience_type": vector_type,
            "evidence_count": evidence_count,
            "vectors": vectors,
            "vector_rms": vector_rms,
            "source_episode_ids": [
                item["source_episode_ids"]
                for item in trace
                if item.get("status") == "compiled"
                and (
                    vector_type == "all_selected"
                    or item["experience_type"] == vector_type
                    or item.get("mechanism_cluster") == vector_type.removeprefix("mechanism-")
                )
            ],
        }
        torch.save(payload, artifact_path)
        artifacts.append(
            {
                "artifact_id": artifact_id,
                "experience_type": vector_type,
                "path": artifact_path.name,
                "sha256": file_sha256(artifact_path),
                "evidence_count": evidence_count,
                "vector_rms": vector_rms,
            }
        )

    report = {
        "schema_version": "steering-vector-compilation-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_pair_count": len(selected),
        "compiled_pair_count": int(counts["all_selected"]),
        "skipped_counts": dict(
            sorted(Counter(item["status"] for item in trace if item["status"] != "compiled").items())
        ),
        "compiler_evidence_trace": {
            "path": evidence_path.name,
            "sha256": file_sha256(evidence_path),
        },
        "artifacts": artifacts,
        "inputs": common["inputs"],
    }
    report_path = output_dir / "vector_compilation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"[vector-compiler] compiled={counts['all_selected']} artifacts={len(artifacts)} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
