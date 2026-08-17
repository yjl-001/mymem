#!/usr/bin/env python3
"""Compile no-new-AI Phase 2 vector-construction ablations.

All vectors are derived from the frozen, Phase-1-approved target/reference
trajectories of the *student* model.  This script deliberately makes no API
calls and never places Teacher/Reviewer prose in the student's prompt.  The
already-approved bank text is used only by a deterministic mechanism bucket
rule for one ablation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
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
    parse_csv_numbers,
    parse_csv_strings,
    phase1_mechanism_cluster,
)


METHODS = (
    "pair_normalized_delta",
    "global_centroid_delta",
    "mechanism_balanced_centroid_delta",
    "seal_style_execution_minus_nonexecution",
)
REFLECTION_RE = re.compile(
    r"\b(wait|verify|verification|double[ -]?check|hold on|think again|let me check|make sure|seems (right|wrong)|incorrect)\b",
    re.I,
)
TRANSITION_RE = re.compile(
    r"\b(alternatively|another (way|approach|method|solution|strategy|technique)|different (way|approach|method))\b",
    re.I,
)


@dataclass(frozen=True)
class TrajectoryTokens:
    ids: list[int]
    last_reasoning_boundary: int | None
    thought_boundaries: list[tuple[int, str]]


@dataclass(frozen=True)
class TokenizedPair:
    experience: dict[str, Any]
    target: TrajectoryTokens
    reference: TrajectoryTokens
    mechanism_cluster: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--layers", default="8,16,24")
    parser.add_argument("--experience-types", default="answer_correctness")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected pairs")
    parser.add_argument("--min-evidence-count", type=int, default=50)
    parser.add_argument("--min-cluster-evidence-count", type=int, default=25)
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
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(model.config, "max_sequence_length", None),
    ]
    valid = [int(value) for value in values if isinstance(value, int) and value > 0]
    return min(valid) if valid else None


def is_delimiter_text(value: str) -> bool:
    return value.rstrip(" \t").endswith((",", ".", "\n"))


def classify_thought(text: str) -> str:
    if TRANSITION_RE.search(text):
        return "transition"
    if REFLECTION_RE.search(text):
        return "reflection"
    return "execution"


def thought_spans(completion: str) -> list[tuple[int, str, str]]:
    """Return SEAL-style blocks; blank lines are primary, lines are fallback.

    SEAL itself uses ``\\n\\n``.  GSM8K rollouts may have only one newline, so
    the fallback is explicit in the evidence trace rather than silently using
    sentence punctuation as a thought boundary.
    """

    separator = "\n\n" if "\n\n" in completion else "\n"
    spans: list[tuple[int, str, str]] = []
    cursor = 0
    for fragment in completion.split(separator):
        end = cursor + len(fragment)
        if fragment.strip():
            spans.append((end, fragment, "blank_line" if separator == "\n\n" else "line_fallback"))
        cursor = end + len(separator)
    return spans


def tokenize_trajectory(tokenizer: Any, prompt_ids: list[int], completion: str) -> TrajectoryTokens:
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    ids = prompt_ids + completion_ids
    # Do not extract from the answer-formatting suffix.  The precise value is
    # derived with the same tokenizer, so it remains auditable even for merged
    # tokens.  If no box exists the entire completion remains eligible.
    boxed_at = completion.find("\\boxed")
    if boxed_at < 0:
        boxed_at = completion.find("\\fbox")
    prefix_limit = len(ids) - 1
    if boxed_at >= 0:
        before_box = completion[:boxed_at]
        prefix_limit = len(prompt_ids) + len(tokenizer.encode(before_box, add_special_tokens=False)) - 1

    boundaries = [
        index
        for index in range(len(prompt_ids), min(len(ids), prefix_limit + 1))
        if is_delimiter_text(tokenizer.decode([ids[index]], skip_special_tokens=False))
    ]
    last = boundaries[-1] if boundaries else None
    thoughts: list[tuple[int, str]] = []
    previous = len(prompt_ids) - 1
    for char_end, text, _segmentation in thought_spans(completion):
        token_end = len(prompt_ids) + len(tokenizer.encode(completion[:char_end], add_special_tokens=False)) - 1
        candidates = [index for index in boundaries if previous < index <= token_end]
        if candidates:
            thoughts.append((candidates[-1], classify_thought(text)))
            previous = candidates[-1]
    return TrajectoryTokens(ids=ids, last_reasoning_boundary=last, thought_boundaries=thoughts)


def tokenize_pair(tokenizer: Any, experience: dict[str, Any]) -> TokenizedPair:
    prompt = tokenizer.apply_chat_template(
        build_gsm8k_messages(str(experience["context"])), tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return TokenizedPair(
        experience=experience,
        target=tokenize_trajectory(tokenizer, prompt_ids, str(experience["trajectory"])),
        reference=tokenize_trajectory(tokenizer, prompt_ids, str(experience["reference_trajectory"])),
        mechanism_cluster=phase1_mechanism_cluster(experience),
    )


def pad_batch(tokenizer: Any, pairs: Iterable[TokenizedPair], device: str):
    import torch

    rows: list[tuple[TokenizedPair, str, list[int]]] = []
    for pair in pairs:
        rows.extend(((pair, "target", pair.target.ids), (pair, "reference", pair.reference.ids)))
    max_length = max(len(ids) for _, _, ids in rows)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")
    input_ids, attention_mask, offsets = [], [], []
    for _, _, ids in rows:
        pad = max_length - len(ids)
        input_ids.append([tokenizer.pad_token_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))
        offsets.append(pad)
    return (
        torch.tensor(input_ids, dtype=torch.long, device=device),
        torch.tensor(attention_mask, dtype=torch.long, device=device),
        offsets,
    )


def rms(vector: Any) -> float:
    return float(vector.float().square().mean().sqrt().item())


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.limit < 0 or args.min_evidence_count <= 0 or args.min_cluster_evidence_count <= 0:
        raise ValueError("batch-size/minimum counts must be positive; limit must be non-negative")
    layers = list(parse_csv_numbers(args.layers, integer=True))
    methods = parse_csv_strings(args.methods)
    if set(methods) - set(METHODS):
        raise ValueError(f"Unknown methods: {sorted(set(methods) - set(METHODS))}")
    if len(set(methods)) != len(methods) or any(layer <= 0 for layer in layers):
        raise ValueError("methods and layers must be unique; layers must be positive")
    requested_types = parse_csv_strings(args.experience_types)
    if set(requested_types) - PHASE2_ELIGIBLE_EXPERIENCE_TYPES:
        raise ValueError("Unsupported Phase 2 experience type")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    selected, selection_report = approved_experiences(
        list(iter_jsonl(args.approved_bank)), list(iter_jsonl(args.experiences)), allowed_experience_types=requested_types
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
        args.model, revision=args.model_revision, torch_dtype=dtype, attn_implementation=args.attn_implementation
    ).to(args.device)
    model.eval()
    context_limit = args.max_sequence_length or model_context_limit(model)
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace: list[dict[str, Any]] = []

    pair_sums: dict[int, Any] = {}
    target_sums: dict[int, Any] = {}
    reference_sums: dict[int, Any] = {}
    mechanism_sums: dict[str, dict[str, dict[int, Any]]] = {}
    seal_sums: dict[str, dict[int, Any]] = {"execution": {}, "nonexecution": {}}
    pair_count = 0
    mechanism_counts: Counter[str] = Counter()
    seal_counts: Counter[str] = Counter()

    def initialize(hidden_size: int) -> None:
        if pair_sums:
            return
        for layer in layers:
            pair_sums[layer] = torch.zeros(hidden_size, dtype=torch.float64)
            target_sums[layer] = torch.zeros(hidden_size, dtype=torch.float64)
            reference_sums[layer] = torch.zeros(hidden_size, dtype=torch.float64)
            seal_sums["execution"][layer] = torch.zeros(hidden_size, dtype=torch.float64)
            seal_sums["nonexecution"][layer] = torch.zeros(hidden_size, dtype=torch.float64)

    def add_mechanism(cluster: str, side: str, layer: int, state: Any) -> None:
        if cluster not in mechanism_sums:
            mechanism_sums[cluster] = {"target": {}, "reference": {}}
            for candidate_layer in layers:
                mechanism_sums[cluster]["target"][candidate_layer] = torch.zeros_like(pair_sums[candidate_layer])
                mechanism_sums[cluster]["reference"][candidate_layer] = torch.zeros_like(pair_sums[candidate_layer])
        mechanism_sums[cluster][side][layer] += state.detach().to(dtype=torch.float64, device="cpu")

    def consume(batch: list[TokenizedPair]) -> None:
        nonlocal pair_count
        input_ids, attention_mask, offsets = pad_batch(tokenizer, batch, args.device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states")
        if any(layer >= len(hidden_states) for layer in layers):
            raise ValueError("Requested layer exceeds model depth")
        initialize(int(hidden_states[layers[0]].shape[-1]))
        for pair_offset, pair in enumerate(batch):
            target_row, reference_row = pair_offset * 2, pair_offset * 2 + 1
            target_boundary = pair.target.last_reasoning_boundary
            reference_boundary = pair.reference.last_reasoning_boundary
            has_pair = target_boundary is not None and reference_boundary is not None
            if has_pair:
                pair_count += 1
                if pair.mechanism_cluster:
                    mechanism_counts[pair.mechanism_cluster] += 1
            for layer in layers:
                if has_pair:
                    target = hidden_states[layer][target_row, offsets[target_row] + target_boundary].float()
                    reference = hidden_states[layer][reference_row, offsets[reference_row] + reference_boundary].float()
                    delta = target - reference
                    delta_rms = rms(delta)
                    if delta_rms <= 0 or not torch.isfinite(delta).all():
                        raise RuntimeError(f"Invalid pair delta for {pair.experience['experience_id']} layer {layer}")
                    pair_sums[layer] += (delta / delta_rms).detach().to(dtype=torch.float64, device="cpu")
                    target_sums[layer] += target.detach().to(dtype=torch.float64, device="cpu")
                    reference_sums[layer] += reference.detach().to(dtype=torch.float64, device="cpu")
                    if pair.mechanism_cluster:
                        add_mechanism(pair.mechanism_cluster, "target", layer, target)
                        add_mechanism(pair.mechanism_cluster, "reference", layer, reference)
                for row, trajectory in ((target_row, pair.target), (reference_row, pair.reference)):
                    for boundary, label in trajectory.thought_boundaries:
                        bucket = "execution" if label == "execution" else "nonexecution"
                        seal_sums[bucket][layer] += hidden_states[layer][row, offsets[row] + boundary].float().detach().to(dtype=torch.float64, device="cpu")
                        seal_counts[bucket] += 1 if layer == layers[0] else 0
            trace.append({
                "experience_id": pair.experience["experience_id"],
                "experience_type": pair.experience["experience_type"],
                "provenance_sha256": pair.experience["provenance_sha256"],
                "mechanism_cluster": pair.mechanism_cluster,
                "target_last_reasoning_boundary": target_boundary,
                "reference_last_reasoning_boundary": reference_boundary,
                "target_thought_boundary_count": len(pair.target.thought_boundaries),
                "reference_thought_boundary_count": len(pair.reference.thought_boundaries),
                "status": "compiled_pair" if has_pair else "skipped_no_preboxed_delimiter",
            })

    pending: list[TokenizedPair] = []
    for index, experience in enumerate(selected, start=1):
        pair = tokenize_pair(tokenizer, experience)
        if context_limit and max(len(pair.target.ids), len(pair.reference.ids)) > context_limit:
            trace.append({"experience_id": experience["experience_id"], "status": "skipped_context_limit", "context_limit": context_limit})
            continue
        pending.append(pair)
        if len(pending) == args.batch_size:
            consume(pending)
            pending = []
        if index % 100 == 0:
            print(f"[phase2-ablation-compiler] prepared {index}/{len(selected)} pairs", flush=True)
    if pending:
        consume(pending)

    trace_path = output_dir / "compiler_evidence_trace.jsonl"
    write_jsonl(trace_path, trace)
    if pair_count < args.min_evidence_count:
        raise RuntimeError(f"Usable paired evidence below minimum: compiled={pair_count} required={args.min_evidence_count}")

    vectors_by_method: dict[str, dict[int, Any]] = {}
    if "pair_normalized_delta" in methods:
        vectors_by_method["pair_normalized_delta"] = {layer: (pair_sums[layer] / pair_count).float() for layer in layers}
    if "global_centroid_delta" in methods:
        vectors_by_method["global_centroid_delta"] = {layer: ((target_sums[layer] - reference_sums[layer]) / pair_count).float() for layer in layers}
    eligible_clusters = sorted(cluster for cluster, count in mechanism_counts.items() if count >= args.min_cluster_evidence_count)
    if "mechanism_balanced_centroid_delta" in methods and eligible_clusters:
        vectors_by_method["mechanism_balanced_centroid_delta"] = {}
        for layer in layers:
            deltas = []
            for cluster in eligible_clusters:
                delta = (mechanism_sums[cluster]["target"][layer] - mechanism_sums[cluster]["reference"][layer]) / mechanism_counts[cluster]
                value = rms(delta)
                if value > 0 and torch.isfinite(delta).all():
                    deltas.append(delta / value)
            if deltas:
                vectors_by_method["mechanism_balanced_centroid_delta"][layer] = (sum(deltas) / len(deltas)).float()
        if len(vectors_by_method["mechanism_balanced_centroid_delta"]) != len(layers):
            del vectors_by_method["mechanism_balanced_centroid_delta"]
    if "seal_style_execution_minus_nonexecution" in methods and min(seal_counts.values()) >= args.min_evidence_count:
        vectors_by_method["seal_style_execution_minus_nonexecution"] = {
            layer: (seal_sums["execution"][layer] / seal_counts["execution"] - seal_sums["nonexecution"][layer] / seal_counts["nonexecution"]).float()
            for layer in layers
        }

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
            "pair_boundary_definition": "last_preboxed_completion_delimiter",
            "seal_boundary_definition": "blank_line_thought_boundary_with_line_fallback",
            "delimiters": [",", ".", "\\n"],
            "layers": layers,
            "min_evidence_count": args.min_evidence_count,
            "min_cluster_evidence_count": args.min_cluster_evidence_count,
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.experiences),
            "selection": selection_report,
            "compiler_git_revision": git_revision(),
        },
    }
    artifacts = []
    for method, vectors in vectors_by_method.items():
        if any(rms(vector) == 0 or not torch.isfinite(vector).all() for vector in vectors.values()):
            raise RuntimeError(f"Invalid final vector for method {method}")
        artifact_id = f"phase2-{method}-answer_correctness"
        path = output_dir / f"{artifact_id}.pt"
        payload = {
            **common,
            "artifact_id": artifact_id,
            "experience_type": "answer_correctness",
            "construction": {**common["construction"], "method": method, "eligible_mechanism_clusters": eligible_clusters},
            "evidence_count": pair_count,
            "method_evidence_counts": {"paired": pair_count, "mechanism": dict(sorted(mechanism_counts.items())), "seal_thoughts": dict(sorted(seal_counts.items()))},
            "vectors": vectors,
            "vector_rms": {str(layer): rms(vector) for layer, vector in vectors.items()},
            "source_episode_ids": [
                {"target": item["target_episode_id"], "reference": item["reference_episode_id"]}
                for item in selected
            ],
        }
        torch.save(payload, path)
        artifacts.append({"method": method, "artifact_id": artifact_id, "path": path.name, "sha256": file_sha256(path), "vector_rms": payload["vector_rms"]})
    unavailable = sorted(set(methods) - set(vectors_by_method))
    report = {
        "schema_version": "phase2-vector-construction-ablation-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_pair_count": len(selected),
        "paired_evidence_count": pair_count,
        "mechanism_counts": dict(sorted(mechanism_counts.items())),
        "eligible_mechanism_clusters": eligible_clusters,
        "seal_thought_counts": dict(sorted(seal_counts.items())),
        "unavailable_methods": unavailable,
        "artifacts": artifacts,
        "compiler_evidence_trace": {"path": trace_path.name, "sha256": file_sha256(trace_path)},
        "inputs": common["inputs"],
    }
    report_path = output_dir / "vector_construction_ablation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[phase2-ablation-compiler] paired={pair_count} artifacts={len(artifacts)} report={report_path}", flush=True)


if __name__ == "__main__":
    main()
