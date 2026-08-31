#!/usr/bin/env python3
"""Decompose V3.5 runtime queries at authenticated first-gate anchors.

The diagnostic keeps source trajectories, counterfactual gate anchors, and
dual key banks fixed.  It independently re-encodes each exact prefix to build
four pre-registered query representations, then scores applicability and
dynamic keys over the full bank.  It performs no generation, side-KV
treatment, task-accuracy access, answer/reward access, window search, winner
selection, threshold fitting, or formal V3.5 qualification change.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.v3_5_hubness import (
    anchor_summary,
    compare_variant_rows,
    numeric_summary,
)
from memgen.experience.v3_5_key_components import (
    V35_KEY_COMPONENT_REPORT_SCHEMA,
)
from memgen.experience.v3_5_query_state import (
    V35_QUERY_STATE_BASELINE,
    V35_QUERY_STATE_EVIDENCE_SCHEMA,
    V35_QUERY_STATE_KEY_VARIANTS,
    V35_QUERY_STATE_LOCAL_WINDOW,
    V35_QUERY_STATE_PRIMARY_KEY,
    V35_QUERY_STATE_PRIMARY_SIDE,
    V35_QUERY_STATE_REPORT_SCHEMA,
    V35_QUERY_STATE_TENSOR_SCHEMA,
    V35_QUERY_STATE_VARIANTS,
    compare_query_rows,
)
from memgen.experience.v3_5_source_alignment import score_query
from scripts.audit_v3_5_dynamic_hubness import (
    _clean_evidence,
    _key_geometry,
    _load_queries,
    _normalize_rows,
    _reconstruct_dynamic_texts,
    _validate_source_report,
)
from scripts.audit_v3_5_dynamic_source_alignment import (
    _build_pairs,
    _resolved_revision,
    _validate_inputs,
    model_context_limit,
)


REPORT_FILE = "query_state_report.json"
MARKDOWN_FILE = "query_state_report.md"
EVIDENCE_FILE = "query_state_evidence.jsonl"
TENSOR_FILE = "query_state_embeddings.safetensors"
GEOMETRY_FILE = "query_state_geometry.jsonl"
HUB_TEXT_FILE = "query_state_hub_text_audit.jsonl"
HUB_TEXT_TOP_N = 5
CURRENT_REENCODE_MIN_COSINE = 0.99999
CURRENT_REENCODE_MAX_ABS_DELTA = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--verified-experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--dual-key-manifest", type=Path, required=True)
    parser.add_argument("--v35-offline-report", type=Path, required=True)
    parser.add_argument("--token-risk-artifact", type=Path, required=True)
    parser.add_argument("--source-alignment-report", type=Path, required=True)
    parser.add_argument("--source-alignment-evidence", type=Path, required=True)
    parser.add_argument("--first-gate-queries", type=Path, required=True)
    parser.add_argument("--key-component-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--permutation-count", type=int, default=10_000)
    parser.add_argument("--max-sequence-length", type=int, default=0)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("query-state audit requires a git revision") from exc
    if not revision:
        raise RuntimeError("query-state audit resolved an empty git revision")
    return revision


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
    temporary.replace(path)
    return count


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_n < 2:
        raise ValueError("--top-n must be at least two")
    if args.permutation_count <= 0:
        raise ValueError("--permutation-count must be positive")
    if args.max_sequence_length < 0:
        raise ValueError("--max-sequence-length must be non-negative")
    for path in (
        args.approved_bank,
        args.verified_experiences,
        args.split_manifest,
        args.memory_records,
        args.dual_key_manifest,
        args.v35_offline_report,
        args.token_risk_artifact,
        args.source_alignment_report,
        args.source_alignment_evidence,
        args.first_gate_queries,
        args.key_component_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def _logical_report_is_authenticated(report: Mapping[str, Any]) -> bool:
    stored = report.get("report_sha256")
    logical = dict(report)
    logical.pop("report_sha256", None)
    return isinstance(stored, str) and stored == canonical_json_sha256(logical)


def _implementation_is_current(report: Mapping[str, Any]) -> bool:
    inputs = report.get("inputs", {})
    implementation = inputs.get("implementation_files_sha256", {})
    return (
        isinstance(implementation, Mapping)
        and bool(implementation)
        and all(
            not Path(str(path)).is_absolute()
            and ".." not in Path(str(path)).parts
            and (PROJECT_ROOT / str(path)).is_file()
            and file_sha256(PROJECT_ROOT / str(path)) == str(digest)
            for path, digest in implementation.items()
        )
        and inputs.get("implementation_set_sha256")
        == canonical_json_sha256(dict(implementation))
    )


def _validate_key_component_report(
    *,
    args: argparse.Namespace,
    report: Mapping[str, Any],
    source_report: Mapping[str, Any],
    memory_count: int,
    query_count: int,
) -> None:
    inputs = report.get("inputs", {})
    requirements = report.get("requirements", {})
    if (
        report.get("schema_version") != V35_KEY_COMPONENT_REPORT_SCHEMA
        or report.get("status") != "completed_diagnostic"
        or report.get("diagnostic_only") is not True
        or report.get("formal_v3_5_qualification_changed") is not False
        or report.get("reasoner_forward_or_generation_run") is not False
        or report.get("query_tensors_changed") is not False
        or report.get("task_accuracy_used") is not False
        or report.get("answer_or_reward_used") is not False
        or report.get("variant_selected") is not False
        or not _logical_report_is_authenticated(report)
        or not _implementation_is_current(report)
        or not isinstance(requirements, Mapping)
        or not requirements
        or not all(value is True for value in requirements.values())
        or int(report.get("memory_count", -1)) != memory_count
        or int(report.get("query_count", -1)) != query_count
        or inputs.get("approved_bank_sha256") != file_sha256(args.approved_bank)
        or inputs.get("verified_experiences_sha256")
        != file_sha256(args.verified_experiences)
        or inputs.get("memory_records_sha256") != file_sha256(args.memory_records)
        or inputs.get("dual_key_manifest_sha256")
        != file_sha256(args.dual_key_manifest)
        or inputs.get("source_alignment_report_sha256")
        != file_sha256(args.source_alignment_report)
        or inputs.get("source_alignment_report_logical_sha256")
        != source_report.get("report_sha256")
        or inputs.get("first_gate_queries_sha256")
        != file_sha256(args.first_gate_queries)
        or report.get("configuration", {}).get("fixed_variants")
        != [
            "applicability_key",
            "dynamic_key",
            "paired_decision_residual",
        ]
    ):
        raise ValueError(
            "query-state audit cannot authenticate key-component report"
        )


def _load_first_gate_anchors(
    *,
    evidence_path: Path,
    source_report: Mapping[str, Any],
    ordered_queries: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    artifact = source_report.get("artifacts", {}).get("evidence", {})
    rows = list(iter_jsonl(evidence_path))
    if (
        artifact.get("sha256") != file_sha256(evidence_path)
        or int(artifact.get("row_count", -1)) != len(rows)
    ):
        raise ValueError("query-state source evidence artifact drifted")
    first_gate = [
        row
        for row in rows
        if int(row.get("counterfactual_attempt_number") or 0) == 1
    ]
    anchors: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in first_gate:
        identity = (
            str(row.get("trajectory_side", "")),
            str(row.get("memory_id", "")),
        )
        if (
            identity in anchors
            or identity[0] not in {"reference", "target"}
            or not identity[1]
            or row.get("exact_anchor_reencoded") is not True
            or row.get("query_embedding_source")
            != "independent_exact_full_prefix_reencode"
            or row.get("pooling") != "current_generated_token"
            or row.get("normalization") != "l2"
            or int(row.get("layer_number", -1)) != 24
            or row.get("side_kv_disabled") is not True
        ):
            raise ValueError("query-state first-gate evidence is malformed")
        anchors[identity] = row

    expected = {
        (str(item["trajectory_side"]), str(item["memory_id"])): item
        for item in ordered_queries
    }
    if len(expected) != len(ordered_queries) or anchors.keys() != expected.keys():
        raise ValueError("query-state first-gate anchor coverage drifted")
    for identity, item in expected.items():
        row = anchors[identity]
        if (
            row.get("query_sidecar_tensor_name") != item.get("tensor_name")
            or row.get("source_experience_id")
            != item.get("source_experience_id")
            or row.get("query_embedding_sha256")
            != item.get("query_embedding_sha256")
        ):
            raise ValueError("query-state first-gate anchor identity drifted")
    return anchors


def _validate_anchor_against_trajectory(
    *,
    anchor: Mapping[str, Any],
    pair: Any,
    side: str,
    trajectory: Any,
) -> tuple[int, int, tuple[int, ...]]:
    reasoning_rank = int(anchor["reasoning_rank"])
    if reasoning_rank < 0 or reasoning_rank >= len(trajectory.reasoning_indices):
        raise ValueError("query-state reasoning rank is outside the trajectory")
    token_index = int(trajectory.reasoning_indices[reasoning_rank])
    prompt_token_count = int(trajectory.reasoning_indices[0])
    prefix_ids = tuple(int(value) for value in trajectory.ids[: token_index + 1])
    trajectory_field = (
        "trajectory" if side == "target" else "reference_trajectory"
    )
    if (
        prompt_token_count <= 0
        or anchor.get("source_experience_id")
        != pair.memory_record.source_experience_id
        or int(anchor.get("prompt_token_count", -1)) != prompt_token_count
        or int(anchor.get("partial_cot_token_count", -1))
        != reasoning_rank + 1
        or int(anchor.get("encoded_full_prefix_token_count", -1))
        != len(prefix_ids)
        or int(anchor.get("query_embedding_token_index", -1))
        != len(prefix_ids) - 1
        or int(anchor.get("query_embedding_token_id", -1))
        != prefix_ids[-1]
        or anchor.get("full_prefix_token_ids_sha256")
        != canonical_json_sha256(list(prefix_ids))
        or anchor.get("source_question_sha256")
        != text_sha256(str(pair.experience["context"]).strip())
        or anchor.get("source_trajectory_sha256")
        != text_sha256(str(pair.experience[trajectory_field]))
    ):
        raise ValueError("query-state reconstructed anchor drifted")
    return reasoning_rank, prompt_token_count, prefix_ids


def _normalize_vector(value: Any, *, owner: str) -> Any:
    import torch
    import torch.nn.functional as F

    vector = value.detach().float().reshape(-1)
    if (
        vector.ndim != 1
        or not torch.isfinite(vector).all()
        or float(vector.norm().item()) <= 1e-12
    ):
        raise ValueError(f"query-state {owner} vector is invalid")
    return F.normalize(vector, dim=0).contiguous()


def _encode_query_variants(
    *,
    model: Any,
    device: str,
    query_metadata: Sequence[Mapping[str, Any]],
    authenticated_current_queries: Any,
    anchors: Mapping[tuple[str, str], Mapping[str, Any]],
    pair_by_memory_id: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    import torch
    import torch.nn.functional as F

    from memgen.model.retrieval_keys import tensor_sha256

    vectors: dict[str, list[Any]] = {
        variant: [] for variant in V35_QUERY_STATE_VARIANTS
    }
    sidecar_tensors: dict[str, Any] = {}
    sidecar_order: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    prompt_cache: dict[str, dict[str, Any]] = {}
    for index, metadata in enumerate(query_metadata):
        memory_id = str(metadata["memory_id"])
        side = str(metadata["trajectory_side"])
        pair = pair_by_memory_id.get(memory_id)
        if pair is None:
            raise ValueError(f"query-state lost source pair: {memory_id}")
        trajectory = pair.target if side == "target" else pair.reference
        anchor = anchors[(side, memory_id)]
        reasoning_rank, prompt_count, prefix_ids = _validate_anchor_against_trajectory(
            anchor=anchor,
            pair=pair,
            side=side,
            trajectory=trajectory,
        )

        prompt_ids = tuple(prefix_ids[:prompt_count])
        prompt_ids_hash = canonical_json_sha256(list(prompt_ids))
        cached_prompt = prompt_cache.get(memory_id)
        if cached_prompt is None:
            prompt_inputs = torch.tensor(
                [list(prompt_ids)], dtype=torch.long, device=device
            )
            with torch.inference_mode():
                prompt_output = model(
                    input_ids=prompt_inputs,
                    attention_mask=torch.ones_like(prompt_inputs),
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            prompt_hidden_states = prompt_output.hidden_states
            if prompt_hidden_states is None or len(prompt_hidden_states) <= 24:
                raise RuntimeError(
                    "query-state reasoner has no prompt layer-24 states"
                )
            prompt_cache[memory_id] = {
                "token_ids_sha256": prompt_ids_hash,
                "raw_state": (
                    prompt_hidden_states[24][0, -1, :]
                    .detach()
                    .float()
                    .cpu()
                    .contiguous()
                ),
            }
            del prompt_output, prompt_hidden_states
            cached_prompt = prompt_cache[memory_id]
        elif cached_prompt["token_ids_sha256"] != prompt_ids_hash:
            raise ValueError(
                "query-state target/reference prompt tokenization drifted"
            )

        inputs = torch.tensor(
            [list(prefix_ids)], dtype=torch.long, device=device
        )
        with torch.inference_mode():
            output = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) <= 24:
            raise RuntimeError("query-state reasoner has no layer-24 states")
        states = hidden_states[24][0].detach().float()
        prompt_raw = cached_prompt["raw_state"].to(device)
        current_raw = states[-1]
        window_start = max(
            prompt_count,
            len(prefix_ids) - V35_QUERY_STATE_LOCAL_WINDOW,
        )
        local_raw = states[window_start:].mean(dim=0)
        delta_raw = current_raw - prompt_raw

        reencoded_current = _normalize_vector(
            current_raw, owner="reencoded current-token"
        ).double().cpu()
        authenticated_current_raw = (
            authenticated_current_queries[index]
            .detach()
            .double()
            .reshape(-1)
        )
        if (
            not torch.isfinite(authenticated_current_raw).all()
            or float(authenticated_current_raw.norm().item()) <= 1e-12
        ):
            raise ValueError("query-state authenticated current vector is invalid")
        authenticated_current = F.normalize(
            authenticated_current_raw, dim=0
        ).contiguous().cpu()
        reproduction_cosine = float(
            F.cosine_similarity(
                reencoded_current, authenticated_current, dim=0
            ).item()
        )
        reproduction_max_abs_delta = float(
            (reencoded_current - authenticated_current).abs().max().item()
        )
        if (
            reproduction_cosine < CURRENT_REENCODE_MIN_COSINE
            or reproduction_max_abs_delta > CURRENT_REENCODE_MAX_ABS_DELTA
        ):
            raise ValueError(
                f"query-state current-token reencode drifted: {memory_id} {side}"
            )

        compiled = {
            "prompt_boundary": _normalize_vector(
                prompt_raw, owner="prompt-boundary"
            ).double().cpu(),
            # Preserve the authenticated source sidecar exactly for baseline scores.
            "current_token": authenticated_current,
            "prompt_subtracted_delta": _normalize_vector(
                delta_raw, owner="prompt-subtracted delta"
            ).double().cpu(),
            "local_reasoning_window_16": _normalize_vector(
                local_raw, owner="local reasoning window"
            ).double().cpu(),
        }
        if tuple(compiled) != V35_QUERY_STATE_VARIANTS:
            raise AssertionError("query-state fixed variant order drifted")

        anchor_name = str(metadata["tensor_name"])
        query_hashes: dict[str, str] = {}
        for variant, vector in compiled.items():
            tensor_name = f"{variant}__{anchor_name}"
            sidecar_tensors[tensor_name] = vector.contiguous()
            vectors[variant].append(vector)
            digest = tensor_sha256(vector)
            query_hashes[variant] = digest
            sidecar_order.append({
                "tensor_name": tensor_name,
                "anchor_tensor_name": anchor_name,
                "query_variant": variant,
                "memory_id": memory_id,
                "source_experience_id": str(metadata["source_experience_id"]),
                "trajectory_side": side,
                "query_embedding_sha256": digest,
            })

        prompt = compiled["prompt_boundary"]
        current = compiled["current_token"]
        delta = compiled["prompt_subtracted_delta"]
        local = compiled["local_reasoning_window_16"]
        geometry_rows.append({
            "memory_id": memory_id,
            "source_experience_id": str(metadata["source_experience_id"]),
            "trajectory_side": side,
            "anchor_tensor_name": anchor_name,
            "reasoning_rank": reasoning_rank,
            "prompt_token_count": prompt_count,
            "prompt_token_ids_sha256": prompt_ids_hash,
            "partial_cot_token_count": reasoning_rank + 1,
            "local_window_token_count": len(prefix_ids) - window_start,
            "raw_current_minus_prompt_norm": float(delta_raw.norm().item()),
            "prompt_current_cosine": float((prompt @ current).item()),
            "prompt_local_cosine": float((prompt @ local).item()),
            "current_local_cosine": float((current @ local).item()),
            "current_delta_cosine": float((current @ delta).item()),
            "prompt_delta_cosine": float((prompt @ delta).item()),
            "current_reencode_cosine": reproduction_cosine,
            "current_reencode_max_abs_delta": reproduction_max_abs_delta,
            "query_embedding_sha256": query_hashes,
        })
        del output, hidden_states, states
        if device.startswith("cuda") and (index + 1) % 16 == 0:
            torch.cuda.empty_cache()
        print(
            f"[v3.5-query-state] {index + 1}/{len(query_metadata)} "
            f"{memory_id} {side}",
            flush=True,
        )

    stacked = {
        variant: torch.stack(values, dim=0).double().contiguous()
        for variant, values in vectors.items()
    }
    return stacked, sidecar_tensors, sidecar_order, geometry_rows


def _score_grid(
    *,
    queries_by_variant: Mapping[str, Any],
    keys_by_variant: Mapping[str, Any],
    query_metadata: Sequence[Mapping[str, Any]],
    anchors: Mapping[tuple[str, str], Mapping[str, Any]],
    memory_ids: Sequence[str],
    top_n: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], dict[str, Any]]:
    from memgen.model.retrieval_keys import tensor_sha256

    rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    matrices: dict[str, dict[str, Any]] = {}
    for query_variant in V35_QUERY_STATE_VARIANTS:
        rows[query_variant] = {}
        matrices[query_variant] = {}
        for key_variant in V35_QUERY_STATE_KEY_VARIANTS:
            score_matrix = (
                queries_by_variant[query_variant]
                @ keys_by_variant[key_variant].T
            )
            matrices[query_variant][key_variant] = score_matrix
            grid_rows: list[dict[str, Any]] = []
            for index, metadata in enumerate(query_metadata):
                memory_id = str(metadata["memory_id"])
                side = str(metadata["trajectory_side"])
                anchor = anchors[(side, memory_id)]
                scored = score_query(
                    memory_ids=memory_ids,
                    scores=[
                        float(value) for value in score_matrix[index].tolist()
                    ],
                    own_memory_id=memory_id,
                    top_n=top_n,
                    include_rank_lookup=True,
                )
                grid_rows.append({
                    "schema_version": V35_QUERY_STATE_EVIDENCE_SCHEMA,
                    "query_variant": query_variant,
                    "key_variant": key_variant,
                    "tensor_name": (
                        f"{query_variant}__{metadata['tensor_name']}"
                    ),
                    "anchor_tensor_name": str(metadata["tensor_name"]),
                    "memory_id": memory_id,
                    "source_experience_id": str(
                        metadata["source_experience_id"]
                    ),
                    "trajectory_side": side,
                    "query_embedding_sha256": tensor_sha256(
                        queries_by_variant[query_variant][index]
                    ),
                    "query_embedding_norm": float(
                        queries_by_variant[query_variant][index].norm().item()
                    ),
                    "layer_number": 24,
                    "normalization": "l2_after_variant_construction",
                    "selector_partition": str(anchor["selector_partition"]),
                    "risk_partition": str(anchor["risk_partition"]),
                    "reasoning_rank": int(anchor["reasoning_rank"]),
                    "prompt_token_count": int(anchor["prompt_token_count"]),
                    "partial_cot_token_count": int(
                        anchor["partial_cot_token_count"]
                    ),
                    **scored,
                })
            rows[query_variant][key_variant] = grid_rows
    return rows, matrices


def _summarize_grid(
    *,
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    memory_ids: Sequence[str],
    permutation_count: int,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for query_variant in V35_QUERY_STATE_VARIANTS:
        summaries[query_variant] = {}
        for key_variant in V35_QUERY_STATE_KEY_VARIANTS:
            summaries[query_variant][key_variant] = {}
            for side in ("reference", "target"):
                side_rows = [
                    row
                    for row in rows[query_variant][key_variant]
                    if row["trajectory_side"] == side
                ]
                summaries[query_variant][key_variant][side] = anchor_summary(
                    side_rows,
                    memory_ids=memory_ids,
                    permutation_count=permutation_count,
                )
    return summaries


def _paired_comparisons(
    *,
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    summaries: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    query_comparisons: dict[str, Any] = {}
    for key_variant in V35_QUERY_STATE_KEY_VARIANTS:
        query_comparisons[key_variant] = {}
        for baseline, candidate in itertools.combinations(
            V35_QUERY_STATE_VARIANTS, 2
        ):
            name = f"{candidate}_versus_{baseline}"
            query_comparisons[key_variant][name] = {
                "baseline_query_variant": baseline,
                "candidate_query_variant": candidate,
                "by_side": {},
            }
            for side in ("reference", "target"):
                baseline_rows = [
                    row
                    for row in rows[baseline][key_variant]
                    if row["trajectory_side"] == side
                ]
                candidate_rows = [
                    row
                    for row in rows[candidate][key_variant]
                    if row["trajectory_side"] == side
                ]
                comparison = compare_query_rows(
                    baseline_rows, candidate_rows
                )
                comparison["hubness_delta_candidate_minus_baseline"] = {
                    field: (
                        summaries[candidate][key_variant][side]["hubness"][field]
                        - summaries[baseline][key_variant][side]["hubness"][field]
                    )
                    for field in (
                        "top1_share",
                        "top2_combined_share",
                        "selected_memory_count",
                        "selection_gini_over_full_bank",
                    )
                }
                query_comparisons[key_variant][name]["by_side"][side] = (
                    comparison
                )

    key_comparisons: dict[str, Any] = {}
    for query_variant in V35_QUERY_STATE_VARIANTS:
        key_comparisons[query_variant] = {}
        for side in ("reference", "target"):
            applicability = [
                row
                for row in rows[query_variant]["applicability_key"]
                if row["trajectory_side"] == side
            ]
            dynamic = [
                row
                for row in rows[query_variant]["dynamic_key"]
                if row["trajectory_side"] == side
            ]
            comparison = compare_variant_rows(applicability, dynamic)
            comparison["baseline_key_variant"] = "applicability_key"
            comparison["candidate_key_variant"] = "dynamic_key"
            comparison["hubness_delta_dynamic_minus_applicability"] = {
                field: (
                    summaries[query_variant]["dynamic_key"][side]["hubness"][field]
                    - summaries[query_variant]["applicability_key"][side][
                        "hubness"
                    ][field]
                )
                for field in (
                    "top1_share",
                    "top2_combined_share",
                    "selected_memory_count",
                    "selection_gini_over_full_bank",
                )
            }
            key_comparisons[query_variant][side] = comparison
    return query_comparisons, key_comparisons


def _geometry_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fields = (
        "raw_current_minus_prompt_norm",
        "prompt_current_cosine",
        "prompt_local_cosine",
        "current_local_cosine",
        "current_delta_cosine",
        "prompt_delta_cosine",
        "current_reencode_cosine",
        "current_reencode_max_abs_delta",
        "local_window_token_count",
        "partial_cot_token_count",
    )
    result: dict[str, Any] = {
        "all": {
            field: numeric_summary([float(row[field]) for row in rows])
            for field in fields
        }
    }
    for side in ("reference", "target"):
        side_rows = [row for row in rows if row["trajectory_side"] == side]
        result[side] = {
            field: numeric_summary(
                [float(row[field]) for row in side_rows]
            )
            for field in fields
        }
    by_identity: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_identity.setdefault(str(row["memory_id"]), {})[
            str(row["trajectory_side"])
        ] = row
    paired_prompt_hash_matches: list[bool] = []
    for sides in by_identity.values():
        if set(sides) != {"reference", "target"}:
            continue
        reference_hash = sides["reference"]["query_embedding_sha256"][
            "prompt_boundary"
        ]
        target_hash = sides["target"]["query_embedding_sha256"][
            "prompt_boundary"
        ]
        paired_prompt_hash_matches.append(reference_hash == target_hash)
    exact_match_count = sum(paired_prompt_hash_matches)
    result["paired_target_reference_prompt_boundary"] = {
        "paired_count": len(paired_prompt_hash_matches),
        "exact_hash_match_count": exact_match_count,
        "exact_hash_match_fraction": (
            exact_match_count / len(paired_prompt_hash_matches)
            if paired_prompt_hash_matches
            else None
        ),
    }
    return result


def _hub_text_rows(
    *,
    summaries: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    matrices: Mapping[str, Mapping[str, Any]],
    memory_ids: Sequence[str],
    text_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: set[str] = set()
    top_ids: dict[str, Any] = {}
    for query_variant in V35_QUERY_STATE_VARIANTS:
        top_ids[query_variant] = {}
        for key_variant in V35_QUERY_STATE_KEY_VARIANTS:
            top_ids[query_variant][key_variant] = {}
            for side in ("reference", "target"):
                ids = [
                    str(item["memory_id"])
                    for item in summaries[query_variant][key_variant][side][
                        "hubness"
                    ]["top_memories"][:HUB_TEXT_TOP_N]
                ]
                top_ids[query_variant][key_variant][side] = ids
                selected.update(ids)

    index_by_id = {memory_id: index for index, memory_id in enumerate(memory_ids)}
    audit_rows: list[dict[str, Any]] = []
    for memory_id in sorted(selected):
        index = index_by_id[memory_id]
        grid_statistics: dict[str, Any] = {}
        for query_variant in V35_QUERY_STATE_VARIANTS:
            grid_statistics[query_variant] = {}
            for key_variant in V35_QUERY_STATE_KEY_VARIANTS:
                matrix = matrices[query_variant][key_variant]
                grid_statistics[query_variant][key_variant] = {
                    "mean_query_cosine": float(matrix[:, index].mean().item()),
                    "reference_top1_count": sum(
                        row["trajectory_side"] == "reference"
                        and row["top1_memory_id"] == memory_id
                        for row in rows[query_variant][key_variant]
                    ),
                    "target_top1_count": sum(
                        row["trajectory_side"] == "target"
                        and row["top1_memory_id"] == memory_id
                        for row in rows[query_variant][key_variant]
                    ),
                }
        audit_rows.append({
            **text_by_id[memory_id],
            "grid_statistics": grid_statistics,
        })
    return audit_rows, top_ids


def _metric_delta(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
) -> dict[str, float] | None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return None
    fields = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_32")
    return {field: float(left[field]) - float(right[field]) for field in fields}


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MemGen V3.5 Dynamic Query-State Decomposition Audit",
        "",
        f"- Status: `{report.get('status')}`",
        "- Diagnostic only: `true`",
        "- Formal V3.5 qualification changed: `false`",
        "- Reasoner forward run: `true`",
        "- Generation run: `false`",
        "- Task accuracy used: `false`",
        "- Answer or reward used: `false`",
        f"- Fixed query variants: `{', '.join(V35_QUERY_STATE_VARIANTS)}`",
        f"- Fixed key variants: `{', '.join(V35_QUERY_STATE_KEY_VARIANTS)}`",
        "",
        "| Side | Query | Key | MRR | R@1 | R@5 | R@10 | R@32 | "
        "Top-1 hub | Top-2 hub | Selected keys | Gini |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = report["grid"]
    for side in ("reference", "target"):
        for query_variant in V35_QUERY_STATE_VARIANTS:
            for key_variant in V35_QUERY_STATE_KEY_VARIANTS:
                summary = summaries[query_variant][key_variant][side]
                metrics = summary["all"]
                hubness = summary["hubness"]
                lines.append(
                    f"| {side} | {query_variant} | {key_variant} | "
                    f"{metrics['mrr']:.6f} | {metrics['recall_at_1']:.6f} | "
                    f"{metrics['recall_at_5']:.6f} | "
                    f"{metrics['recall_at_10']:.6f} | "
                    f"{metrics['recall_at_32']:.6f} | "
                    f"{hubness['top1_share']:.6f} | "
                    f"{hubness['top2_combined_share']:.6f} | "
                    f"{hubness['selected_memory_count']} | "
                    f"{hubness['selection_gini_over_full_bank']:.6f} |"
                )
    geometry = report["query_geometry"]["all"]
    lines.extend([
        "",
        "## Query geometry",
        "",
        "- Prompt/current cosine: "
        f"`{json.dumps(geometry['prompt_current_cosine'], sort_keys=True)}`",
        "- Current/local cosine: "
        f"`{json.dumps(geometry['current_local_cosine'], sort_keys=True)}`",
        "- Raw current-minus-prompt norm: "
        f"`{json.dumps(geometry['raw_current_minus_prompt_norm'], sort_keys=True)}`",
        "- Local window token count: "
        f"`{json.dumps(geometry['local_window_token_count'], sort_keys=True)}`",
        "",
        "The current-token row is an authenticated reproduction baseline. Prompt-only",
        "matching current-token indicates static task dominance; delta or local-window",
        "improvement indicates recoverable dynamic-state information. No query or key",
        "variant is selected for online use by this diagnostic.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    _validate_args(args)

    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.retrieval_keys import tensor_sha256
    from memgen.model.v3_5_retrieval import DualRetrievalKeyBankLoader

    approved_records = list(iter_jsonl(args.approved_bank))
    verified_experiences = list(iter_jsonl(args.verified_experiences))
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    key_bank = DualRetrievalKeyBankLoader(manifest_path=args.dual_key_manifest)
    source_report = json.loads(
        args.source_alignment_report.read_text(encoding="utf-8")
    )
    ordered = _validate_source_report(
        args=args,
        report=source_report,
        key_bank=key_bank,
    )
    authenticated_current, query_metadata = _load_queries(
        query_path=args.first_gate_queries,
        ordered=ordered,
        expected_width=int(key_bank.dynamic_embeddings.shape[1]),
    )
    anchors = _load_first_gate_anchors(
        evidence_path=args.source_alignment_evidence,
        source_report=source_report,
        ordered_queries=ordered,
    )
    memory_ids = tuple(str(entry["memory_id"]) for entry in key_bank.entries)
    component_report = json.loads(
        args.key_component_report.read_text(encoding="utf-8")
    )
    _validate_key_component_report(
        args=args,
        report=component_report,
        source_report=source_report,
        memory_count=len(memory_ids),
        query_count=len(query_metadata),
    )

    risk_artifact = torch.load(
        args.token_risk_artifact, map_location="cpu", weights_only=False
    )
    authenticated_inputs = _validate_inputs(
        args=args,
        records=records,
        approved_records=approved_records,
        verified_experiences=verified_experiences,
        key_bank=key_bank,
        risk_artifact=risk_artifact,
    )

    reasoner = key_bank.manifest["reasoner"]
    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    if (
        _resolved_revision(
            getattr(model.config, "_commit_hash", None),
            str(reasoner["model_revision"]),
        )
        != reasoner["model_revision"]
        or _resolved_revision(
            getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
            str(reasoner["tokenizer_revision"]),
        )
        != reasoner["tokenizer_revision"]
    ):
        raise ValueError("query-state resolved reasoner/tokenizer drifted")

    context_limit = args.max_sequence_length or model_context_limit(model)
    risk_seed = int(source_report["configuration"]["risk_split_seed"])
    risk_fraction = float(source_report["configuration"]["risk_train_fraction"])
    pairs, skipped = _build_pairs(
        records=records,
        source_by_id=authenticated_inputs["source_by_id"],
        tokenizer=tokenizer,
        context_limit=context_limit,
        risk_split_seed=risk_seed,
        risk_train_fraction=risk_fraction,
    )
    pair_by_memory_id = {
        pair.memory_record.memory_id: pair for pair in pairs
    }
    required_ids = {str(row["memory_id"]) for row in query_metadata}
    if not required_ids.issubset(pair_by_memory_id):
        raise ValueError("query-state reconstructed source pairs lost an anchor")

    queries_by_variant, sidecar_tensors, sidecar_order, geometry_rows = (
        _encode_query_variants(
            model=model,
            device=args.device,
            query_metadata=query_metadata,
            authenticated_current_queries=authenticated_current,
            anchors=anchors,
            pair_by_memory_id=pair_by_memory_id,
        )
    )
    keys_by_variant = {
        "applicability_key": _normalize_rows(
            key_bank.applicability_embeddings.double(),
            owner="query-state applicability keys",
        ),
        "dynamic_key": _normalize_rows(
            key_bank.dynamic_embeddings.double(),
            owner="query-state dynamic keys",
        ),
    }
    if tuple(keys_by_variant) != V35_QUERY_STATE_KEY_VARIANTS:
        raise AssertionError("query-state fixed key order drifted")

    rows, matrices = _score_grid(
        queries_by_variant=queries_by_variant,
        keys_by_variant=keys_by_variant,
        query_metadata=query_metadata,
        anchors=anchors,
        memory_ids=memory_ids,
        top_n=args.top_n,
    )
    summaries = _summarize_grid(
        rows=rows,
        memory_ids=memory_ids,
        permutation_count=args.permutation_count,
    )

    reproduction_deltas: dict[str, Any] = {}
    for key_variant in V35_QUERY_STATE_KEY_VARIANTS:
        reproduction_deltas[key_variant] = {}
        for side in ("reference", "target"):
            current_summary = summaries["current_token"][key_variant][side]
            component_summary = component_report["variants"][key_variant][side]
            current_metrics = current_summary["all"]
            component_metrics = component_summary["all"]
            delta = _metric_delta(current_metrics, component_metrics)
            if delta is None or any(abs(value) > 1e-15 for value in delta.values()):
                raise ValueError(
                    "query-state current baseline does not reproduce "
                    "key-component metrics"
                )
            hubness_delta = {
                field: (
                    current_summary["hubness"][field]
                    - component_summary["hubness"][field]
                )
                for field in (
                    "top1_share",
                    "top2_combined_share",
                    "selected_memory_count",
                    "selection_gini_over_full_bank",
                )
            }
            if any(abs(value) > 1e-15 for value in hubness_delta.values()):
                raise ValueError(
                    "query-state current baseline does not reproduce "
                    "key-component hubness"
                )
            reproduction_deltas[key_variant][side] = {
                "rank_metrics": delta,
                "hubness": hubness_delta,
            }

    query_comparisons, key_comparisons = _paired_comparisons(
        rows=rows,
        summaries=summaries,
    )
    query_geometry = _geometry_summary(geometry_rows)
    paired_prompt = query_geometry[
        "paired_target_reference_prompt_boundary"
    ]
    if (
        int(paired_prompt["paired_count"]) <= 0
        or paired_prompt["paired_count"]
        != paired_prompt["exact_hash_match_count"]
    ):
        raise ValueError(
            "query-state target/reference prompt-boundary tensors drifted"
        )

    text_by_id = _reconstruct_dynamic_texts(
        approved_records=approved_records,
        verified_experiences=verified_experiences,
        records=records,
        key_bank=key_bank,
    )
    hub_text_rows, top_hub_ids = _hub_text_rows(
        summaries=summaries,
        rows=rows,
        matrices=matrices,
        memory_ids=memory_ids,
        text_by_id=text_by_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / EVIDENCE_FILE
    evidence_count = write_jsonl(
        evidence_path,
        (
            _clean_evidence(row)
            for query_variant in V35_QUERY_STATE_VARIANTS
            for key_variant in V35_QUERY_STATE_KEY_VARIANTS
            for row in rows[query_variant][key_variant]
        ),
    )
    geometry_path = args.output_dir / GEOMETRY_FILE
    geometry_count = write_jsonl(geometry_path, geometry_rows)
    tensor_path = args.output_dir / TENSOR_FILE
    save_file(
        {
            name: value.contiguous().cpu()
            for name, value in sidecar_tensors.items()
        },
        str(tensor_path),
        metadata={
            "schema_version": V35_QUERY_STATE_TENSOR_SCHEMA,
            "query_variant_order_sha256": canonical_json_sha256(
                V35_QUERY_STATE_VARIANTS
            ),
            "ordered_tensors_sha256": canonical_json_sha256(sidecar_order),
        },
    )
    hub_text_path = args.output_dir / HUB_TEXT_FILE
    hub_text_count = write_jsonl(hub_text_path, hub_text_rows)

    implementation_paths = (
        "data/gsm8k/prompt.py",
        "memgen/chat_templates.py",
        "memgen/experience/memory.py",
        "memgen/experience/phase1.py",
        "memgen/experience/risk.py",
        "memgen/experience/v3_5_selector.py",
        "memgen/experience/v3_5_source_alignment.py",
        "memgen/experience/v3_5_hubness.py",
        "memgen/experience/v3_5_key_components.py",
        "memgen/experience/v3_5_query_state.py",
        "memgen/model/retrieval_keys.py",
        "memgen/model/v3_5_retrieval.py",
        "scripts/audit_v3_5_dynamic_source_alignment.py",
        "scripts/audit_v3_5_dynamic_hubness.py",
        "scripts/audit_v3_5_dynamic_query_state.py",
    )
    implementation_hashes = {
        path: file_sha256(PROJECT_ROOT / path) for path in implementation_paths
    }
    report: dict[str, Any] = {
        "schema_version": V35_QUERY_STATE_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "completed_diagnostic",
        "diagnostic_only": True,
        "formal_v3_5_qualification_changed": False,
        "reasoner_forward_run": True,
        "generation_run": False,
        "side_kv_used": False,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "answer_or_reward_scope": (
            "not_used_for_query_construction_ranking_threshold_or_variant_"
            "selection_after_authenticated_source_bank_selection"
        ),
        "verified_success_failure_roles_reused_from_source_provenance": True,
        "query_variant_selected": False,
        "key_variant_selected": False,
        "primary_side": V35_QUERY_STATE_PRIMARY_SIDE,
        "primary_key_variant": V35_QUERY_STATE_PRIMARY_KEY,
        "reproduction_baseline_query_variant": V35_QUERY_STATE_BASELINE,
        "memory_count": len(memory_ids),
        "query_count": len(query_metadata),
        "context_eligible_pair_count": len(pairs),
        "skipped_pair_count": len(skipped),
        "configuration": {
            "fixed_query_variants": list(V35_QUERY_STATE_VARIANTS),
            "fixed_key_variants": list(V35_QUERY_STATE_KEY_VARIANTS),
            "prompt_boundary_definition": (
                "layer24_state_at_token_immediately_before_first_completion_token"
            ),
            "current_token_definition": (
                "authenticated_source_alignment_exact_current_token_sidecar"
            ),
            "prompt_subtracted_delta_definition": (
                "l2_normalize(raw_layer24_current_minus_raw_layer24_prompt_boundary)"
            ),
            "local_reasoning_window_definition": (
                "l2_normalize(mean_raw_layer24_latest_up_to_16_reasoning_tokens_"
                "including_anchor)"
            ),
            "local_reasoning_window_size": V35_QUERY_STATE_LOCAL_WINDOW,
            "local_reasoning_short_prefix_policy": (
                "use_all_available_reasoning_tokens"
            ),
            "layer_number": 24,
            "normalization": "l2_after_variant_construction",
            "current_reencode_min_cosine": CURRENT_REENCODE_MIN_COSINE,
            "current_reencode_max_abs_delta": CURRENT_REENCODE_MAX_ABS_DELTA,
            "current_scoring_tensor_source": (
                "authenticated_source_alignment_sidecar_not_new_reencode"
            ),
            "prompt_state_extraction": (
                "independent_exact_prompt_only_prefix_reencode_cached_per_memory"
            ),
            "retrieval_scope": "all_161_keys_static_shortlist_bypassed",
            "retrieval_method": "exact_cosine",
            "stable_tie_break": "memory_id_ascending",
            "top_n": args.top_n,
            "permutation_count": args.permutation_count,
            "permutation_policy": (
                "shuffle_query_to_own_memory_binding_preserve_grid_score_matrix"
            ),
            "compute_device": args.device,
            "model_compute_dtype": args.dtype,
            "prompt_contract": GSM8K_PROMPT_CONTRACT.metadata(
                chat_template=CONVERSATION_TEMPLATE
            ),
        },
        "query_geometry": query_geometry,
        "key_geometry": {
            variant: _key_geometry(keys)
            for variant, keys in keys_by_variant.items()
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(
                args.verified_experiences
            ),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "memory_records_sha256": file_sha256(args.memory_records),
            "dual_key_manifest_sha256": file_sha256(args.dual_key_manifest),
            "dual_key_manifest_logical_sha256": key_bank.manifest_sha256,
            "v35_offline_report_sha256": file_sha256(args.v35_offline_report),
            "token_risk_artifact_sha256": file_sha256(
                args.token_risk_artifact
            ),
            "source_alignment_report_sha256": file_sha256(
                args.source_alignment_report
            ),
            "source_alignment_report_logical_sha256": source_report[
                "report_sha256"
            ],
            "source_alignment_evidence_sha256": file_sha256(
                args.source_alignment_evidence
            ),
            "first_gate_queries_sha256": file_sha256(args.first_gate_queries),
            "key_component_report_sha256": file_sha256(
                args.key_component_report
            ),
            "key_component_report_logical_sha256": component_report[
                "report_sha256"
            ],
            "git_revision": git_revision(),
            "implementation_files_sha256": implementation_hashes,
            "implementation_set_sha256": canonical_json_sha256(
                implementation_hashes
            ),
        },
        "artifacts": {
            "grid_evidence": {
                "path": evidence_path.name,
                "sha256": file_sha256(evidence_path),
                "row_count": evidence_count,
            },
            "query_state_embeddings": {
                "path": tensor_path.name,
                "sha256": file_sha256(tensor_path),
                "tensor_count": len(sidecar_tensors),
                "ordered_tensors_sha256": canonical_json_sha256(
                    sidecar_order
                ),
                "tensor_set_sha256": canonical_json_sha256(
                    {
                        name: tensor_sha256(value)
                        for name, value in sorted(sidecar_tensors.items())
                    }
                ),
            },
            "query_geometry": {
                "path": geometry_path.name,
                "sha256": file_sha256(geometry_path),
                "row_count": geometry_count,
            },
            "hub_key_text_audit": {
                "path": hub_text_path.name,
                "sha256": file_sha256(hub_text_path),
                "row_count": hub_text_count,
            },
        },
        "current_baseline_reproduction_delta": reproduction_deltas,
        "grid": summaries,
        "paired_query_comparisons": query_comparisons,
        "paired_key_comparisons": key_comparisons,
        "top_hub_ids_by_query_key_side": top_hub_ids,
        "requirements": {
            "source_alignment_report_authenticated": True,
            "source_alignment_evidence_authenticated": True,
            "exact_first_gate_query_sidecar_authenticated": True,
            "key_component_report_authenticated": True,
            "dual_key_bank_authenticated": True,
            "source_trajectory_join_authenticated": True,
            "source_first_gate_anchors_reused_without_reselection": True,
            "current_token_authenticated_sidecar_reused_for_scoring": True,
            "current_token_independent_reencode_within_tolerance": True,
            "current_baseline_metrics_exactly_reproduced": True,
            "paired_target_reference_prompt_boundary_identical": True,
            "four_query_variants_pre_registered": True,
            "two_key_variants_pre_registered": True,
            "local_window_size_fixed_at_16": True,
            "window_size_not_searched": True,
            "query_variant_not_selected": True,
            "key_variant_not_selected": True,
            "threshold_not_fitted": True,
            "static_shortlist_bypassed_for_dynamic_isolation": True,
            "generation_not_run": True,
            "side_kv_not_used": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
            "formal_v3_5_qualification_unchanged": True,
        },
        "interpretation_contract": {
            "prompt_boundary_similar_to_current": (
                "current_query_is_dominated_by_static_prompt_or_task_semantics"
            ),
            "prompt_subtracted_delta_improves_reference_alignment": (
                "prompt_baseline_masks_recoverable_dynamic_state_information"
            ),
            "local_window_improves_reference_alignment": (
                "single_current_token_pooling_discards_recoverable_local_state"
            ),
            "state_queries_help_applicability_not_dynamic": (
                "state_is_recoverable_but_prefer_action_should_not_be_in_key"
            ),
            "all_state_queries_remain_weak": (
                "standalone_text_keys_and_runtime_causal_states_require_"
                "structured_trigger_or_explicit_alignment"
            ),
            "target_strong_reference_weak": (
                "confirmation_retrieval_not_corrective_retrieval"
            ),
            "strong_in_source_result_not_cross_problem_utility": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = args.output_dir / REPORT_FILE
    write_json(report_path, report)
    write_text(args.output_dir / MARKDOWN_FILE, _markdown(report))
    print(
        "[v3.5-query-state] "
        f"status={report['status']} queries={len(query_metadata)} "
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
