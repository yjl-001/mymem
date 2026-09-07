#!/usr/bin/env python3
"""Extract an authenticated, answer-blind V4 layer-24 source-state cache.

This is the only V4 source-state component that loads the reasoner.  It runs
one teacher-forced success/failure pair per construction sample, retains raw
latest-32 windows for every counterfactual gate attempt, and writes an
offline-only cache for subsequent CPU selector research.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.phase1 import canonical_json_sha256, file_sha256, text_sha256
from memgen.experience.v4_source_state import (
    V4_SOURCE_STATE_MAX_WINDOW,
    save_source_state_cache,
)
from scripts.build_v4_repair_bank import _validate_split_manifest, load_v4_experiences
from scripts.compile_v4_selector_anchors import (
    _construction_bank_membership,
    _load_records,
    _model_context_limit,
    _normalized_progress_rank,
    _pad_pair,
    _risk_scores,
    _tokenize_trajectory,
)


EXPECTED_CURATED_BANK_COUNT = 17
EXPECTED_CONSTRUCTION_PAIR_COUNT = 116


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--bank-records", type=Path, required=True)
    parser.add_argument("--bank-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--token-risk-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--max-sequence-length", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Deterministic construction-sample cap; zero extracts the full curated bank.",
    )
    return parser.parse_args()


def _git_revision() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("V4 source-state extraction requires a git revision") from exc
    if not value:
        raise RuntimeError("V4 source-state extraction resolved an empty git revision")
    return value


def _question_token_indices(
    *, tokenizer: Any, question: str, prompt_ids: Sequence[int]
) -> tuple[int, ...]:
    """Locate question content with tokenizer offsets, with a token fallback."""

    rendered = GSM8K_PROMPT_CONTRACT.render(tokenizer, question)
    normalized = str(question).strip()
    character_start = rendered.rfind(normalized)
    if character_start < 0:
        raise ValueError("Canonical GSM8K prompt lost the question text")
    character_end = character_start + len(normalized)
    try:
        encoded = tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids = [int(value) for value in encoded["input_ids"]]
        offsets = encoded["offset_mapping"]
        if ids == list(prompt_ids):
            selected = tuple(
                index
                for index, (start, end) in enumerate(offsets)
                if int(end) > character_start and int(start) < character_end
            )
            if selected:
                return selected
    except (KeyError, TypeError, ValueError, NotImplementedError):
        pass

    question_ids = [
        int(value)
        for value in tokenizer.encode(normalized, add_special_tokens=False)
    ]
    candidates = [
        start
        for start in range(len(prompt_ids) - len(question_ids) + 1)
        if list(prompt_ids[start : start + len(question_ids)]) == question_ids
    ]
    if not candidates:
        raise ValueError("Unable to identify GSM8K question tokens in canonical prompt")
    start = candidates[-1]
    return tuple(range(start, start + len(question_ids)))


def _right_aligned_window(states: Any, *, rank: int) -> tuple[Any, Any, int]:
    import torch

    if states.ndim != 2 or rank < 0 or rank >= int(states.shape[0]):
        raise ValueError("V4 source-state window rank is outside the trajectory")
    selected = states[max(0, rank + 1 - V4_SOURCE_STATE_MAX_WINDOW) : rank + 1]
    length = int(selected.shape[0])
    window = torch.zeros(
        (V4_SOURCE_STATE_MAX_WINDOW, int(states.shape[1])),
        dtype=states.dtype,
        device=states.device,
    )
    mask = torch.zeros(
        V4_SOURCE_STATE_MAX_WINDOW, dtype=torch.bool, device=states.device
    )
    window[-length:] = selected
    mask[-length:] = True
    return window.detach().cpu(), mask.detach().cpu(), length


def _logit_summary(logits: Any) -> dict[str, float]:
    import torch

    values = logits.detach().float()
    if values.ndim != 1 or not torch.isfinite(values).all():
        raise ValueError("V4 source-state logit summary received invalid logits")
    top = torch.topk(values, k=2).values
    log_probs = torch.log_softmax(values, dim=-1)
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum()
    return {
        "maximum_logit": float(top[0].item()),
        "top1_top2_logit_gap": float((top[0] - top[1]).item()),
        "logsumexp": float(torch.logsumexp(values, dim=-1).item()),
        "predictive_entropy": float(entropy.item()),
    }


def _gate_attempts(
    *, states: Any, entropies: Any, gate_artifact: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    from memgen.experience.v3_5_source_alignment import (
        CounterfactualGateObservation,
        counterfactual_attempts,
    )

    risks = _risk_scores(
        states,
        recovery=gate_artifact["risk_gate"]["recovery_center"],
        persistence=gate_artifact["risk_gate"]["persistence_center"],
    ).detach().float().cpu()
    construction = gate_artifact["construction"]
    risk = gate_artifact["risk_gate"]
    return counterfactual_attempts(
        [
            CounterfactualGateObservation(
                reasoning_rank=rank,
                attention_entropy=float(entropies[rank].item()),
                risk_score=float(risks[rank].item()),
            )
            for rank in range(int(states.shape[0]))
        ],
        high_entropy_threshold=float(construction["high_entropy_threshold"]),
        low_entropy_threshold=float(construction["low_entropy_threshold"]),
        risk_threshold=float(risk["threshold"]),
        rearm_low_token_count=2,
        maximum_attempts=3,
    )


def _common_event(
    *, experience: Mapping[str, Any], record: Mapping[str, Any], bank_id: str
) -> dict[str, Any]:
    sample_id = str(experience["sample_id"])
    source = experience["source"]
    medoid_id = str(record["construction"].get("joint_medoid_experience_id", ""))
    curation = record.get("curation", {})
    return {
        "experience_id": str(experience["experience_id"]),
        "sample_id": sample_id,
        "independent_sample_id": canonical_json_sha256(
            {
                "benchmark": "openai/gsm8k",
                "logical_split": str(source["logical_split"]),
                "sample_id": sample_id,
            }
        ),
        "bank_id": bank_id,
        "benchmark": "openai/gsm8k",
        "logical_split": str(source["logical_split"]),
        "dataset_split": str(source["dataset_split"]),
        "source_index": int(source["source_index"]),
        "question_sha256": str(source["question_sha256"]),
        "is_medoid": str(experience["experience_id"]) == medoid_id,
        "curation_tier": str(curation.get("decision", "")),
        "construction_profile_sha256": str(record["construction_profile_sha256"]),
        "bank_record_sha256": str(record["record_sha256"]),
        "contrast_pair": {
            "target_episode_id": str(experience["target_episode_id"]),
            "reference_episode_id": str(experience["reference_episode_id"]),
            "paired_success_failure": True,
        },
        "completion_hashes": {
            "verified_success_completion_sha256": text_sha256(
                str(experience["trajectory"])
            ),
            "verified_failure_completion_sha256": text_sha256(
                str(experience["reference_trajectory"])
            ),
        },
    }


def _gate_event(
    *,
    common: Mapping[str, Any],
    event_kind: str,
    attempt: Mapping[str, Any],
    token_position: int,
    prefix_ids: Sequence[int],
    window_length: int,
    tensor_rows: Mapping[str, int],
    gate_artifact: Mapping[str, Any],
    logits: Any,
    aligned: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    construction = gate_artifact["construction"]
    risk = gate_artifact["risk_gate"]
    attempt_number = int(attempt["attempt_number"])
    event_id = (
        f"{common['experience_id']}::{event_kind}::attempt-{attempt_number}"
    )
    row = {
        **dict(common),
        "event_id": event_id,
        "event_kind": event_kind,
        "online_reachable_safety_negative": event_kind == "success_gate_attempt",
        "attempt_number": attempt_number,
        "reasoning_rank": int(attempt["reasoning_rank"]),
        "candidate_rank": int(attempt["reasoning_rank"]) + 1,
        "token_position": token_position,
        "window_token_count": window_length,
        "tensor_rows": dict(tensor_rows),
        "gate_diagnostics": {
            "gate_eligible": True,
            "gate_rejection_reason": None,
            "attention_entropy": float(attempt["attention_entropy"]),
            "persistence_risk": float(attempt["risk_score"]),
            "high_entropy_threshold": float(construction["high_entropy_threshold"]),
            "low_entropy_threshold": float(construction["low_entropy_threshold"]),
            "risk_threshold": float(risk["threshold"]),
            "state_before": str(attempt["state_before"]),
            "state_after": str(attempt["state_after"]),
            "logit_summary": _logit_summary(logits),
        },
        "prefix_alignment": {
            "prefix_token_count": len(prefix_ids),
            "prefix_token_ids_sha256": canonical_json_sha256(
                [int(value) for value in prefix_ids]
            ),
            "prefix_includes_current_token": True,
            "token_position_matches_prefix_end": token_position == len(prefix_ids) - 1,
        },
    }
    if event_kind == "failure_gate_attempt":
        if aligned is None:
            raise ValueError("Failure gate event requires matched-success alignment")
        row["matched_success_alignment"] = dict(aligned)
    return row


def _validate_inputs(args: argparse.Namespace) -> None:
    if args.limit < 0:
        raise ValueError("V4 source-state extraction limit must be non-negative")
    for path in (
        args.experiences,
        args.split_manifest,
        args.bank_records,
        args.bank_manifest,
        args.side_kv_manifest,
        args.token_risk_artifact,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def main() -> None:
    args = parse_args()
    _validate_inputs(args)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.side_kv import SDPAAttentionEntropyObserver
    from memgen.model.v3_runtime import EntropyHysteresisGate
    from memgen.model.v4_side_kv import (
        V4SideKVBankLoader,
        validate_v4_tensor_free_manifest,
    )

    split_manifest = _validate_split_manifest(
        args.split_manifest, dataset_revision=args.dataset_revision
    )
    experiences = load_v4_experiences(
        args.experiences, split_manifest=split_manifest
    )
    experience_by_id = {str(item["experience_id"]): item for item in experiences}
    records = _load_records(args.bank_records)
    if len(records) != EXPECTED_CURATED_BANK_COUNT:
        raise ValueError(
            "V4 source-state extraction requires the fixed 17-bank curated archive"
        )
    record_by_bank_id = {str(record["bank_id"]): record for record in records}
    construction_manifest = json.loads(args.bank_manifest.read_text(encoding="utf-8"))
    validate_v4_tensor_free_manifest(construction_manifest)
    if [record["bank_id"] for record in records] != construction_manifest["bank_ids"]:
        raise ValueError("V4 source-state records differ from bank manifest order")
    if any(
        construction_manifest.get("record_sha256", {}).get(record["bank_id"])
        != record["record_sha256"]
        for record in records
    ):
        raise ValueError("V4 source-state records are not bound by bank manifest")
    if construction_manifest["inputs"].get("experiences_sha256") != file_sha256(args.experiences):
        raise ValueError("V4 source-state experiences differ from bank construction")
    if construction_manifest["inputs"].get("split_manifest_sha256") != file_sha256(
        args.split_manifest
    ):
        raise ValueError("V4 source-state split differs from bank construction")
    membership = _construction_bank_membership(
        records=records, experience_by_id=experience_by_id
    )
    if len(membership) != EXPECTED_CONSTRUCTION_PAIR_COUNT:
        raise ValueError(
            "V4 source-state extraction requires all 116 construction pairs"
        )

    side_loader = V4SideKVBankLoader(manifest_path=args.side_kv_manifest)
    if tuple(construction_manifest["bank_ids"]) != side_loader.bank_ids:
        raise ValueError("V4 source-state bank and side-KV namespaces differ")
    if (
        side_loader.manifest.get("source", {}).get("bank_manifest_logical_sha256")
        != construction_manifest["manifest_sha256"]
    ):
        raise ValueError("V4 source-state side-KV source binding drifted")

    risk_artifact = torch.load(
        args.token_risk_artifact, map_location="cpu", weights_only=False
    )
    gate = EntropyHysteresisGate.from_token_artifact(risk_artifact)
    if gate.config.layer_number != 24 or gate.config.risk_role != "online_joint_control":
        raise ValueError("V4 source-state cache requires the qualified layer-24 joint gate")
    reasoner = dict(side_loader.manifest["reasoner"])
    for field in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_artifact.get("reasoner", {}).get(field) != reasoner.get(field):
            raise ValueError("V4 source-state gate and side-KV reasoner provenance differ")

    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    resolved_model = str(
        getattr(model.config, "_commit_hash", None) or reasoner["model_revision"]
    )
    resolved_tokenizer = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or reasoner["tokenizer_revision"]
    )
    if (
        resolved_model != reasoner["model_revision"]
        or resolved_tokenizer != reasoner["tokenizer_revision"]
    ):
        raise ValueError("V4 source-state model/tokenizer revision drifted")
    context_limit = args.max_sequence_length or _model_context_limit(model)

    ordered_ids = sorted(membership)
    if args.limit:
        ordered_ids = ordered_ids[: args.limit]
    hidden_width = int(model.config.hidden_size)
    tensor_rows: dict[str, list[Any]] = {
        "prompt_end_states": [],
        "question_mean_states": [],
        "question_boundary_states": [],
        "question_local_windows": [],
        "question_local_masks": [],
        "failure_gate_windows": [],
        "failure_gate_masks": [],
        "aligned_success_windows": [],
        "aligned_success_masks": [],
        "success_gate_windows": [],
        "success_gate_masks": [],
    }
    events: list[dict[str, Any]] = []
    observer = SDPAAttentionEntropyObserver(
        model=model,
        sink_token_count=int(risk_artifact["construction"]["sink_token_count"]),
    )
    try:
        for position, experience_id in enumerate(ordered_ids, start=1):
            experience = experience_by_id[experience_id]
            bank_id = membership[experience_id]
            record = record_by_bank_id[bank_id]
            prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(
                tokenizer, str(experience["context"])
            )
            success = _tokenize_trajectory(
                tokenizer, prompt_ids, str(experience["trajectory"])
            )
            failure = _tokenize_trajectory(
                tokenizer, prompt_ids, str(experience["reference_trajectory"])
            )
            if not success.reasoning_indices or not failure.reasoning_indices:
                raise ValueError(
                    f"V4 source-state sample has no pre-answer tokens: {experience_id}"
                )
            if context_limit and max(len(success.ids), len(failure.ids)) > context_limit:
                raise ValueError(f"V4 source-state sample exceeds model context: {experience_id}")
            input_ids, attention_mask = _pad_pair(
                tokenizer=tokenizer,
                target=success,
                reference=failure,
                device=args.device,
            )
            with torch.inference_mode(), observer.capture():
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            if output.hidden_states is None or len(output.hidden_states) <= 24:
                raise RuntimeError("V4 source-state model has no layer-24 states")
            layer_states = output.hidden_states[24]
            prompt_states = layer_states[0, : len(prompt_ids), :].detach()
            success_positions = list(success.reasoning_indices)
            failure_positions = list(failure.reasoning_indices)
            success_states = layer_states[0, success_positions, :].detach()
            failure_states = layer_states[1, failure_positions, :].detach()
            success_entropies = observer.observation.entropy_by_query[
                0, success_positions
            ].detach().float().cpu()
            failure_entropies = observer.observation.entropy_by_query[
                1, failure_positions
            ].detach().float().cpu()
            failure_attempts = _gate_attempts(
                states=failure_states,
                entropies=failure_entropies,
                gate_artifact=risk_artifact,
            )
            success_attempts = _gate_attempts(
                states=success_states,
                entropies=success_entropies,
                gate_artifact=risk_artifact,
            )

            question_indices = _question_token_indices(
                tokenizer=tokenizer,
                question=str(experience["context"]),
                prompt_ids=prompt_ids,
            )
            question_states = prompt_states[list(question_indices), :]
            question_window, question_mask, _ = _right_aligned_window(
                question_states, rank=int(question_states.shape[0]) - 1
            )
            prompt_row = len(tensor_rows["prompt_end_states"])
            tensor_rows["prompt_end_states"].append(prompt_states[-1].cpu())
            tensor_rows["question_mean_states"].append(
                question_states.float().mean(dim=0).to(dtype=prompt_states.dtype).cpu()
            )
            tensor_rows["question_boundary_states"].append(question_states[-1].cpu())
            tensor_rows["question_local_windows"].append(question_window)
            tensor_rows["question_local_masks"].append(question_mask)
            common = _common_event(experience=experience, record=record, bank_id=bank_id)
            events.append(
                {
                    **common,
                    "event_id": f"{experience_id}::prompt-semantic",
                    "event_kind": "prompt_semantic",
                    "online_reachable_safety_negative": False,
                    "tensor_rows": {name: prompt_row for name in (
                        "prompt_end_states",
                        "question_mean_states",
                        "question_boundary_states",
                        "question_local_windows",
                        "question_local_masks",
                    )},
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
                    "question_token_start": question_indices[0],
                    "question_token_end_exclusive": question_indices[-1] + 1,
                    "question_token_count": len(question_indices),
                    "failure_gate_eligible": bool(failure_attempts),
                    "failure_gate_attempt_count": len(failure_attempts),
                    "failure_gate_rejection_reason": (
                        None if failure_attempts else "failure_has_no_joint_gate"
                    ),
                    "success_gate_eligible": bool(success_attempts),
                    "success_gate_attempt_count": len(success_attempts),
                    "success_gate_rejection_reason": (
                        None if success_attempts else "success_has_no_joint_gate"
                    ),
                }
            )

            for attempt in failure_attempts:
                failure_rank = int(attempt["reasoning_rank"])
                success_rank = _normalized_progress_rank(
                    failure_rank,
                    source_count=len(failure_positions),
                    target_count=len(success_positions),
                )
                failure_window, failure_mask, failure_length = _right_aligned_window(
                    failure_states, rank=failure_rank
                )
                aligned_window, aligned_mask, aligned_length = _right_aligned_window(
                    success_states, rank=success_rank
                )
                row_index = len(tensor_rows["failure_gate_windows"])
                tensor_rows["failure_gate_windows"].append(failure_window)
                tensor_rows["failure_gate_masks"].append(failure_mask)
                tensor_rows["aligned_success_windows"].append(aligned_window)
                tensor_rows["aligned_success_masks"].append(aligned_mask)
                failure_token_position = failure_positions[failure_rank]
                success_token_position = success_positions[success_rank]
                aligned = {
                    "state_role": "offline_repair_direction_control",
                    "online_reachable_safety_negative": False,
                    "alignment_method": "normalized_reasoning_progress_endpoint_preserving",
                    "reasoning_rank": success_rank,
                    "token_position": success_token_position,
                    "window_token_count": aligned_length,
                    "prefix_token_ids_sha256": canonical_json_sha256(
                        list(success.ids[: success_token_position + 1])
                    ),
                }
                events.append(
                    _gate_event(
                        common=common,
                        event_kind="failure_gate_attempt",
                        attempt=attempt,
                        token_position=failure_token_position,
                        prefix_ids=failure.ids[: failure_token_position + 1],
                        window_length=failure_length,
                        tensor_rows={name: row_index for name in (
                            "failure_gate_windows",
                            "failure_gate_masks",
                            "aligned_success_windows",
                            "aligned_success_masks",
                        )},
                        gate_artifact=risk_artifact,
                        logits=output.logits[1, failure_token_position, :],
                        aligned=aligned,
                    )
                )

            for attempt in success_attempts:
                success_rank = int(attempt["reasoning_rank"])
                success_window, success_mask, success_length = _right_aligned_window(
                    success_states, rank=success_rank
                )
                row_index = len(tensor_rows["success_gate_windows"])
                tensor_rows["success_gate_windows"].append(success_window)
                tensor_rows["success_gate_masks"].append(success_mask)
                success_token_position = success_positions[success_rank]
                events.append(
                    _gate_event(
                        common=common,
                        event_kind="success_gate_attempt",
                        attempt=attempt,
                        token_position=success_token_position,
                        prefix_ids=success.ids[: success_token_position + 1],
                        window_length=success_length,
                        tensor_rows={name: row_index for name in (
                            "success_gate_windows", "success_gate_masks"
                        )},
                        gate_artifact=risk_artifact,
                        logits=output.logits[0, success_token_position, :],
                    )
                )
            print(
                f"[v4-source-state] {position}/{len(ordered_ids)} {experience_id} "
                f"bank={bank_id} failure_gates={len(failure_attempts)} "
                f"success_gates={len(success_attempts)}",
                flush=True,
            )
    finally:
        observer.close()

    def stack_or_empty(name: str) -> Any:
        rows = tensor_rows[name]
        if rows:
            return torch.stack(rows).contiguous()
        if name.endswith("_masks"):
            return torch.empty((0, V4_SOURCE_STATE_MAX_WINDOW), dtype=torch.bool)
        if name.endswith("_windows"):
            return torch.empty(
                (0, V4_SOURCE_STATE_MAX_WINDOW, hidden_width), dtype=torch.bfloat16
            )
        return torch.empty((0, hidden_width), dtype=torch.bfloat16)

    tensors = {name: stack_or_empty(name) for name in tensor_rows}
    implementation_paths = (
        "data/gsm8k/prompt.py",
        "memgen/experience/v3_5_source_alignment.py",
        "memgen/experience/v4_source_state.py",
        "memgen/model/e1_runtime.py",
        "memgen/model/side_kv.py",
        "memgen/model/v3_runtime.py",
        "scripts/compile_v4_selector_anchors.py",
        "scripts/extract_v4_source_state_cache.py",
    )
    construction_profiles = sorted(
        {str(record["construction_profile_sha256"]) for record in records}
    )
    if len(construction_profiles) != 1:
        raise ValueError("V4 source-state bank has multiple construction profiles")
    configuration = {
        "layer_number": 24,
        "hidden_state_tuple_index": 24,
        "attention_implementation": "sdpa",
        "dtype": args.dtype,
        "maximum_gate_attempts": 3,
        "maximum_hidden_window": 32,
        "derived_windows": [1, 4, 8, 16, 32],
        "support_unit": "independent_sample",
        "prompt_state_types": [
            "prompt_end",
            "question_token_mean",
            "question_boundary",
            "question_local_raw_window_32",
        ],
        "failure_gate_state_scope": "all_counterfactual_attempts_up_to_three",
        "success_state_roles_separated": True,
        "extraction_scope": "smoke" if args.limit else "full_curated_construction",
        "requested_limit": args.limit,
        "expected_full_construction_count": len(membership),
        "extracted_construction_count": len(ordered_ids),
    }
    reasoner.update(
        {
            "resolved_model_revision": resolved_model,
            "resolved_tokenizer_revision": resolved_tokenizer,
            "attention_implementation": "sdpa",
        }
    )
    provenance = {
        "construction_profile_sha256": construction_profiles[0],
        "bank_manifest_logical_sha256": construction_manifest["manifest_sha256"],
        "side_kv_manifest_logical_sha256": side_loader.manifest["manifest_sha256"],
        "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
            chat_template=CONVERSATION_TEMPLATE
        ),
        "inputs": {
            "experiences_path": str(args.experiences.resolve()),
            "experiences_sha256": file_sha256(args.experiences),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "split_manifest_logical_sha256": split_manifest["manifest_sha256"],
            "bank_records_path": str(args.bank_records.resolve()),
            "bank_records_sha256": file_sha256(args.bank_records),
            "bank_manifest_path": str(args.bank_manifest.resolve()),
            "bank_manifest_file_sha256": file_sha256(args.bank_manifest),
            "side_kv_manifest_path": str(args.side_kv_manifest.resolve()),
            "side_kv_manifest_file_sha256": file_sha256(args.side_kv_manifest),
            "token_risk_artifact_path": str(args.token_risk_artifact.resolve()),
            "token_risk_artifact_sha256": file_sha256(args.token_risk_artifact),
        },
        "gate": gate.config.to_dict(),
        "created_by": "gpu_teacher_forced_source_state_extractor",
        "external_api_calls_made": 0,
    }
    manifest_path, reachability_path = save_source_state_cache(
        output_dir=args.output_dir,
        tensors=tensors,
        events=events,
        repository_revision=_git_revision(),
        reasoner=reasoner,
        configuration=configuration,
        provenance=provenance,
        implementation_sha256={
            path: file_sha256(PROJECT_ROOT / path) for path in implementation_paths
        },
    )
    print(
        f"[v4-source-state] complete samples={len(ordered_ids)} events={len(events)} "
        f"manifest={manifest_path} reachability={reachability_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4-source-state] error: {exc}", file=sys.stderr)
        raise
