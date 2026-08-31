#!/usr/bin/env python3
"""Audit source causal-prefix states as V3.6 retrieval keys.

Reference/failure first-gate states become a diagnostic key bank.  Paired
target/success first-gate states query that bank, so no query is scored against
the exact tensor from which its own key was copied.  A prompt-only identity
control measures same-question leakage explicitly.  The experiment performs
no model forward, generation, side-KV treatment, task scoring, threshold
fitting, or online variant selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
)
from memgen.experience.v3_5_hubness import (
    anchor_summary,
    numeric_summary,
)
from memgen.experience.v3_5_query_state import (
    V35_QUERY_STATE_REPORT_SCHEMA,
    V35_QUERY_STATE_TENSOR_SCHEMA,
    V35_QUERY_STATE_VARIANTS,
)
from memgen.experience.v3_5_source_alignment import score_query
from memgen.experience.v3_6_state_keys import (
    V36_STATE_KEY_BANK_SCHEMA,
    V36_STATE_KEY_EVIDENCE_SCHEMA,
    V36_STATE_KEY_IDENTITY_CONTROL,
    V36_STATE_KEY_PRIMARY_VARIANT,
    V36_STATE_KEY_REPORT_SCHEMA,
    V36_STATE_KEY_TEXT_CONTROL,
    V36_STATE_KEY_TRAJECTORY_KEY_SIDE,
    V36_STATE_KEY_TRAJECTORY_QUERY_SIDE,
    V36_STATE_KEY_VARIANTS,
    compare_state_key_rows,
)
from scripts.audit_v3_5_dynamic_hubness import (
    _key_geometry,
    _logical_report_is_authenticated,
    _normalize_rows,
    _reconstruct_dynamic_texts,
    _validate_source_report,
)
from scripts.audit_v3_5_dynamic_query_state import (
    _load_first_gate_anchors,
)


REPORT_FILE = "state_key_report.json"
MARKDOWN_FILE = "state_key_report.md"
EVIDENCE_FILE = "state_key_evidence.jsonl"
KEY_TENSOR_FILE = "reference_state_key_bank.safetensors"
KEY_MANIFEST_FILE = "reference_state_key_manifest.json"
HUB_TEXT_FILE = "state_key_hub_text_audit.jsonl"
STATE_COMPONENTS = tuple(V35_QUERY_STATE_VARIANTS)
HUB_TEXT_TOP_N = 5


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
    parser.add_argument("--query-state-report", type=Path, required=True)
    parser.add_argument("--query-state-embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--permutation-count", type=int, default=10_000)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_revision() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("state-key audit requires a git revision") from exc
    if not value:
        raise RuntimeError("state-key audit resolved an empty git revision")
    return value


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
        args.query_state_report,
        args.query_state_embeddings,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


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


def _validate_query_state_report(
    *,
    args: argparse.Namespace,
    report: Mapping[str, Any],
    source_report: Mapping[str, Any],
    ordered_queries: Sequence[Mapping[str, Any]],
) -> None:
    inputs = report.get("inputs", {})
    requirements = report.get("requirements", {})
    artifact = report.get("artifacts", {}).get("query_state_embeddings", {})
    if (
        report.get("schema_version") != V35_QUERY_STATE_REPORT_SCHEMA
        or report.get("status") != "completed_diagnostic"
        or report.get("diagnostic_only") is not True
        or report.get("formal_v3_5_qualification_changed") is not False
        or report.get("reasoner_forward_run") is not True
        or report.get("generation_run") is not False
        or report.get("side_kv_used") is not False
        or report.get("task_accuracy_used") is not False
        or report.get("answer_or_reward_used") is not False
        or report.get("query_variant_selected") is not False
        or report.get("key_variant_selected") is not False
        or not _logical_report_is_authenticated(report)
        or not _implementation_is_current(report)
        or not isinstance(requirements, Mapping)
        or not requirements
        or not all(value is True for value in requirements.values())
        or report.get("configuration", {}).get("fixed_query_variants")
        != list(V35_QUERY_STATE_VARIANTS)
        or int(report.get("query_count", -1)) != len(ordered_queries)
        or inputs.get("approved_bank_sha256") != file_sha256(args.approved_bank)
        or inputs.get("verified_experiences_sha256")
        != file_sha256(args.verified_experiences)
        or inputs.get("split_manifest_sha256") != file_sha256(args.split_manifest)
        or inputs.get("memory_records_sha256") != file_sha256(args.memory_records)
        or inputs.get("dual_key_manifest_sha256")
        != file_sha256(args.dual_key_manifest)
        or inputs.get("v35_offline_report_sha256")
        != file_sha256(args.v35_offline_report)
        or inputs.get("token_risk_artifact_sha256")
        != file_sha256(args.token_risk_artifact)
        or inputs.get("source_alignment_report_sha256")
        != file_sha256(args.source_alignment_report)
        or inputs.get("source_alignment_report_logical_sha256")
        != source_report.get("report_sha256")
        or inputs.get("source_alignment_evidence_sha256")
        != file_sha256(args.source_alignment_evidence)
        or inputs.get("first_gate_queries_sha256")
        != file_sha256(args.first_gate_queries)
        or inputs.get("key_component_report_sha256")
        != file_sha256(args.key_component_report)
        or artifact.get("sha256") != file_sha256(args.query_state_embeddings)
        or int(artifact.get("tensor_count", -1))
        != len(V35_QUERY_STATE_VARIANTS) * len(ordered_queries)
    ):
        raise ValueError("state-key audit cannot authenticate query-state report")


def _load_state_tensors(
    *,
    path: Path,
    report: Mapping[str, Any],
    ordered_queries: Sequence[Mapping[str, Any]],
    expected_width: int,
) -> tuple[dict[tuple[str, str, str], Any], list[dict[str, Any]]]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    from memgen.model.retrieval_keys import tensor_sha256

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        names = set(handle.keys())
    tensors = load_file(str(path), device="cpu")
    expected_names = {
        f"{variant}__{item['tensor_name']}"
        for item in ordered_queries
        for variant in V35_QUERY_STATE_VARIANTS
    }
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != V35_QUERY_STATE_TENSOR_SCHEMA
        or metadata.get("query_variant_order_sha256")
        != canonical_json_sha256(V35_QUERY_STATE_VARIANTS)
        or names != expected_names
    ):
        raise ValueError("state-key query-state tensor metadata drifted")

    values: dict[tuple[str, str, str], Any] = {}
    ordered_tensors: list[dict[str, Any]] = []
    for item in ordered_queries:
        side = str(item["trajectory_side"])
        memory_id = str(item["memory_id"])
        anchor_name = str(item["tensor_name"])
        for variant in V35_QUERY_STATE_VARIANTS:
            name = f"{variant}__{anchor_name}"
            vector = tensors[name].detach().double().reshape(-1).contiguous()
            if (
                tuple(vector.shape) != (expected_width,)
                or not torch.isfinite(vector).all()
                or not math.isclose(
                    float(vector.norm().item()),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
            ):
                raise ValueError(f"state-key tensor is invalid: {name}")
            digest = tensor_sha256(vector)
            identity = (side, memory_id, variant)
            if identity in values:
                raise ValueError("state-key tensor identity is duplicated")
            values[identity] = vector
            ordered_tensors.append({
                "tensor_name": name,
                "anchor_tensor_name": anchor_name,
                "query_variant": variant,
                "memory_id": memory_id,
                "source_experience_id": str(item["source_experience_id"]),
                "trajectory_side": side,
                "query_embedding_sha256": digest,
            })

    artifact = report["artifacts"]["query_state_embeddings"]
    tensor_set = {
        name: tensor_sha256(tensors[name]) for name in sorted(tensors)
    }
    if (
        metadata.get("ordered_tensors_sha256")
        != canonical_json_sha256(ordered_tensors)
        or artifact.get("ordered_tensors_sha256")
        != canonical_json_sha256(ordered_tensors)
        or artifact.get("tensor_set_sha256")
        != canonical_json_sha256(tensor_set)
    ):
        raise ValueError("state-key tensor ordering or content drifted")
    return values, ordered_tensors


def _paired_state_inputs(
    *,
    state_tensors: Mapping[tuple[str, str, str], Any],
    ordered_queries: Sequence[Mapping[str, Any]],
    anchors: Mapping[tuple[str, str], Mapping[str, Any]],
    key_bank: Any,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    import torch

    by_identity = {
        (str(item["trajectory_side"]), str(item["memory_id"])): item
        for item in ordered_queries
    }
    reference_ids = tuple(sorted(
        memory_id
        for side, memory_id in by_identity
        if side == V36_STATE_KEY_TRAJECTORY_KEY_SIDE
    ))
    target_ids = {
        memory_id
        for side, memory_id in by_identity
        if side == V36_STATE_KEY_TRAJECTORY_QUERY_SIDE
    }
    paired_ids = tuple(
        memory_id for memory_id in reference_ids if memory_id in target_ids
    )
    if not reference_ids or not paired_ids:
        raise ValueError("state-key audit has no reference keys or paired queries")

    key_vectors = {
        variant: _normalize_rows(
            torch.stack([
                state_tensors[("reference", memory_id, variant)]
                for memory_id in reference_ids
            ], dim=0).double().contiguous(),
            owner=f"state-key reference {variant}",
        )
        for variant in STATE_COMPONENTS
    }
    query_vectors = {
        variant: _normalize_rows(
            torch.stack([
                state_tensors[("target", memory_id, variant)]
                for memory_id in paired_ids
            ], dim=0).double().contiguous(),
            owner=f"state-key target {variant}",
        )
        for variant in STATE_COMPONENTS
    }
    metadata: list[dict[str, Any]] = []
    for memory_id in paired_ids:
        target_item = by_identity[("target", memory_id)]
        reference_item = by_identity[("reference", memory_id)]
        target_anchor = anchors[("target", memory_id)]
        reference_anchor = anchors[("reference", memory_id)]
        if (
            target_item["source_experience_id"]
            != reference_item["source_experience_id"]
            or target_anchor["selector_partition"]
            != reference_anchor["selector_partition"]
            or target_anchor["risk_partition"]
            != reference_anchor["risk_partition"]
        ):
            raise ValueError("state-key paired source identity or partition drifted")
        metadata.append({
            "memory_id": memory_id,
            "source_experience_id": str(target_item["source_experience_id"]),
            "target_tensor_name": str(target_item["tensor_name"]),
            "reference_tensor_name": str(reference_item["tensor_name"]),
            "selector_partition": str(target_anchor["selector_partition"]),
            "risk_partition": str(target_anchor["risk_partition"]),
            "target_reasoning_rank": int(target_anchor["reasoning_rank"]),
            "reference_reasoning_rank": int(reference_anchor["reasoning_rank"]),
            "target_full_prefix_token_ids_sha256": str(
                target_anchor["full_prefix_token_ids_sha256"]
            ),
            "reference_full_prefix_token_ids_sha256": str(
                reference_anchor["full_prefix_token_ids_sha256"]
            ),
        })

    applicability = torch.stack([
        key_bank.applicability_embeddings[key_bank.index_by_id[memory_id]]
        for memory_id in reference_ids
    ], dim=0).double().contiguous()
    applicability = _normalize_rows(
        applicability, owner="state-key applicability control"
    )
    key_vectors["applicability_key"] = applicability
    return reference_ids, paired_ids, key_vectors, query_vectors, metadata


def _variant_specifications() -> Mapping[str, tuple[str, str]]:
    value = {
        "text_applicability__target_current_control": (
            "applicability_key",
            "current_token",
        ),
        "state_prompt__target_prompt_identity_control": (
            "prompt_boundary",
            "prompt_boundary",
        ),
        "state_current__target_current": (
            "current_token",
            "current_token",
        ),
        "state_delta__target_delta": (
            "prompt_subtracted_delta",
            "prompt_subtracted_delta",
        ),
        "state_local16__target_local16": (
            "local_reasoning_window_16",
            "local_reasoning_window_16",
        ),
    }
    if tuple(value) != V36_STATE_KEY_VARIANTS:
        raise AssertionError("state-key fixed variant order drifted")
    return value


def _score_variants(
    *,
    memory_ids: Sequence[str],
    paired_ids: Sequence[str],
    key_vectors: Mapping[str, Any],
    query_vectors: Mapping[str, Any],
    metadata: Sequence[Mapping[str, Any]],
    top_n: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    from memgen.model.retrieval_keys import tensor_sha256

    specifications = _variant_specifications()
    rows: dict[str, list[dict[str, Any]]] = {}
    matrices: dict[str, Any] = {}
    for variant, (key_component, query_component) in specifications.items():
        keys = key_vectors[key_component]
        queries = query_vectors[query_component]
        matrix = queries @ keys.T
        matrices[variant] = matrix
        variant_rows: list[dict[str, Any]] = []
        for index, (memory_id, item) in enumerate(zip(paired_ids, metadata)):
            scored = score_query(
                memory_ids=memory_ids,
                scores=[float(value) for value in matrix[index].tolist()],
                own_memory_id=memory_id,
                top_n=top_n,
                include_rank_lookup=True,
            )
            own_index = memory_ids.index(memory_id)
            variant_rows.append({
                "schema_version": V36_STATE_KEY_EVIDENCE_SCHEMA,
                "variant": variant,
                "tensor_name": str(item["target_tensor_name"]),
                "memory_id": memory_id,
                "source_experience_id": str(item["source_experience_id"]),
                "trajectory_side": "target",
                "key_trajectory_side": (
                    "text" if key_component == "applicability_key" else "reference"
                ),
                "key_component": key_component,
                "query_component": query_component,
                "key_embedding_sha256": tensor_sha256(keys[own_index]),
                "query_embedding_sha256": tensor_sha256(queries[index]),
                "selector_partition": str(item["selector_partition"]),
                "risk_partition": str(item["risk_partition"]),
                "target_reasoning_rank": int(item["target_reasoning_rank"]),
                "reference_reasoning_rank": int(item["reference_reasoning_rank"]),
                "query_and_key_origin_tensor_names_differ": (
                    item["target_tensor_name"] != item["reference_tensor_name"]
                ),
                **scored,
            })
        rows[variant] = variant_rows
    return rows, matrices


def _cross_trajectory_geometry(
    *,
    paired_ids: Sequence[str],
    key_vectors: Mapping[str, Any],
    query_vectors: Mapping[str, Any],
    memory_ids: Sequence[str],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import torch

    values: dict[str, Any] = {}
    for component in STATE_COMPONENTS:
        keys = key_vectors[component]
        queries = query_vectors[component]
        own_keys = [keys[memory_ids.index(memory_id)] for memory_id in paired_ids]
        own_cosines = [
            float((queries[index] @ own_keys[index]).item())
            for index in range(len(paired_ids))
        ]
        exact_count = sum(
            torch.equal(queries[index], own_keys[index])
            for index in range(len(paired_ids))
        )
        values[component] = {
            "own_cosine": numeric_summary(own_cosines),
            "exact_embedding_match_count": exact_count,
            "exact_embedding_match_fraction": exact_count / len(paired_ids),
        }
    prompt = values["prompt_boundary"]
    if (
        prompt["exact_embedding_match_count"] != len(paired_ids)
        or float(prompt["own_cosine"]["minimum"]) < 1.0 - 1e-12
        or float(prompt["own_cosine"]["maximum"]) > 1.0 + 1e-12
    ):
        raise ValueError("state-key paired prompt identity control drifted")
    values["paired_full_prefix_identity"] = {
        "paired_count": len(metadata),
        "exact_full_prefix_token_hash_match_count": sum(
            item["target_full_prefix_token_ids_sha256"]
            == item["reference_full_prefix_token_ids_sha256"]
            for item in metadata
        ),
    }
    return values


def _comparisons(
    rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for baseline, candidate in itertools.combinations(V36_STATE_KEY_VARIANTS, 2):
        result[f"{candidate}_versus_{baseline}"] = {
            "baseline_variant": baseline,
            "candidate_variant": candidate,
            **compare_state_key_rows(rows[baseline], rows[candidate]),
        }
    return result


def _clean_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in row.items() if key != "rank_by_memory_id"
    }


def _hub_text_rows(
    *,
    summaries: Mapping[str, Any],
    text_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    selected: set[str] = set()
    top_ids: dict[str, list[str]] = {}
    for variant in V36_STATE_KEY_VARIANTS:
        ids = [
            str(item["memory_id"])
            for item in summaries[variant]["hubness"]["top_memories"][:HUB_TEXT_TOP_N]
        ]
        top_ids[variant] = ids
        selected.update(ids)
    rows = [
        {
            **text_by_id[memory_id],
            "top1_statistics": {
                variant: next(
                    (
                        item
                        for item in summaries[variant]["hubness"]["top_memories"]
                        if item["memory_id"] == memory_id
                    ),
                    None,
                )
                for variant in V36_STATE_KEY_VARIANTS
            },
        }
        for memory_id in sorted(selected)
    ]
    return rows, top_ids


def _save_state_key_bank(
    *,
    output_dir: Path,
    memory_ids: Sequence[str],
    key_vectors: Mapping[str, Any],
    ordered_queries: Sequence[Mapping[str, Any]],
    anchors: Mapping[tuple[str, str], Mapping[str, Any]],
    key_bank: Any,
    query_state_report: Mapping[str, Any],
) -> tuple[Path, Path, Mapping[str, Any]]:
    from safetensors.torch import save_file
    from memgen.model.retrieval_keys import tensor_sha256

    by_identity = {
        (str(item["trajectory_side"]), str(item["memory_id"])): item
        for item in ordered_queries
    }
    tensors: dict[str, Any] = {}
    entries: list[dict[str, Any]] = []
    for key_index, memory_id in enumerate(memory_ids):
        item = by_identity[("reference", memory_id)]
        anchor = anchors[("reference", memory_id)]
        component_entries: dict[str, Any] = {}
        for component in STATE_COMPONENTS:
            tensor_name = f"reference_{component}__{memory_id}"
            vector = key_vectors[component][key_index].contiguous().cpu()
            tensors[tensor_name] = vector
            component_entries[component] = {
                "tensor_name": tensor_name,
                "embedding_sha256": tensor_sha256(vector),
                "embedding_norm": float(vector.norm().item()),
            }
        bank_entry = key_bank.entry_by_id[memory_id]
        entries.append({
            "index": key_index,
            "memory_id": memory_id,
            "source_experience_id": str(item["source_experience_id"]),
            "payload_hash": str(bank_entry["payload_hash"]),
            "payload_token_count": int(bank_entry["payload_token_count"]),
            "kv_layer": int(bank_entry["kv_layer"]),
            "kv_valid_slot_count": int(bank_entry["kv_valid_slot_count"]),
            "trajectory_side": "reference",
            "anchor_tensor_name": str(item["tensor_name"]),
            "anchor_reasoning_rank": int(anchor["reasoning_rank"]),
            "full_prefix_token_ids_sha256": str(
                anchor["full_prefix_token_ids_sha256"]
            ),
            "state_components": component_entries,
        })

    tensor_path = output_dir / KEY_TENSOR_FILE
    save_file(
        tensors,
        str(tensor_path),
        metadata={
            "schema_version": V36_STATE_KEY_BANK_SCHEMA,
            "record_order_sha256": canonical_json_sha256(list(memory_ids)),
            "component_order_sha256": canonical_json_sha256(STATE_COMPONENTS),
        },
    )
    manifest: dict[str, Any] = {
        "schema_version": V36_STATE_KEY_BANK_SCHEMA,
        "created_at": utc_now(),
        "status": "completed_diagnostic",
        "diagnostic_only": True,
        "qualified_for_online_use": False,
        "key_source": (
            "authenticated_reference_failure_first_gate_full_causal_prefix_state"
        ),
        "value_source": "existing_full_when_facing_prefer_avoid_side_kv_unchanged",
        "memory_count": len(memory_ids),
        "state_components": list(STATE_COMPONENTS),
        "record_order_sha256": canonical_json_sha256(list(memory_ids)),
        "tensor_artifact": {
            "path": tensor_path.name,
            "sha256": file_sha256(tensor_path),
            "tensor_count": len(tensors),
            "tensor_set_sha256": canonical_json_sha256({
                name: tensor_sha256(value) for name, value in sorted(tensors.items())
            }),
        },
        "source": {
            "query_state_report_logical_sha256": query_state_report["report_sha256"],
            "query_state_embeddings_sha256": query_state_report[
                "artifacts"
            ]["query_state_embeddings"]["sha256"],
            "dual_key_manifest_logical_sha256": key_bank.manifest_sha256,
        },
        "records": entries,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = output_dir / KEY_MANIFEST_FILE
    write_json(manifest_path, manifest)
    return tensor_path, manifest_path, manifest


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MemGen V3.6 Source-State Retrieval-Key Audit",
        "",
        f"- Status: `{report['status']}`",
        "- Diagnostic only: `true`",
        "- Qualified for online use: `false`",
        "- Generation or reasoner forward run: `false`",
        "- Task accuracy used: `false`",
        "- Answer or reward used: `false`",
        f"- Reference state-key count: `{report['reference_key_count']}`",
        f"- Paired target query count: `{report['paired_target_query_count']}`",
        "- Target/reference tensor origins are distinct: `true`",
        "- Existing side-KV payload changed: `false`",
        "",
        "| Variant | MRR | R@1 | R@5 | R@10 | R@32 | Top-1 hub | "
        "Top-2 hub | Selected keys | Gini |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in V36_STATE_KEY_VARIANTS:
        summary = report["variants"][variant]
        metrics = summary["all"]
        hubness = summary["hubness"]
        lines.append(
            f"| {variant} | {metrics['mrr']:.6f} | "
            f"{metrics['recall_at_1']:.6f} | {metrics['recall_at_5']:.6f} | "
            f"{metrics['recall_at_10']:.6f} | {metrics['recall_at_32']:.6f} | "
            f"{hubness['top1_share']:.6f} | "
            f"{hubness['top2_combined_share']:.6f} | "
            f"{hubness['selected_memory_count']} | "
            f"{hubness['selection_gini_over_full_bank']:.6f} |"
        )
    geometry = report["cross_trajectory_own_key_cosine"]
    lines.extend([
        "",
        "## Cross-trajectory geometry",
        "",
        "- Prompt identity cosine: "
        f"`{json.dumps(geometry['prompt_boundary'], sort_keys=True)}`",
        "- Current-state cosine: "
        f"`{json.dumps(geometry['current_token'], sort_keys=True)}`",
        "- Prompt-subtracted delta cosine: "
        f"`{json.dumps(geometry['prompt_subtracted_delta'], sort_keys=True)}`",
        "- Local-16 cosine: "
        f"`{json.dumps(geometry['local_reasoning_window_16'], sort_keys=True)}`",
        "",
        "Prompt retrieval is an explicit same-question identity ceiling. A strong",
        "current/local result is only evidence of cross-trajectory state stability;",
        "it is not cross-problem transfer. Delta retaining the gain is the strongest",
        "available indication that matching extends beyond the shared prompt. No",
        "variant is selected for online use by this diagnostic.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    _validate_args(args)

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
    ordered_queries = _validate_source_report(
        args=args,
        report=source_report,
        key_bank=key_bank,
    )
    anchors = _load_first_gate_anchors(
        evidence_path=args.source_alignment_evidence,
        source_report=source_report,
        ordered_queries=ordered_queries,
    )
    query_state_report = json.loads(
        args.query_state_report.read_text(encoding="utf-8")
    )
    _validate_query_state_report(
        args=args,
        report=query_state_report,
        source_report=source_report,
        ordered_queries=ordered_queries,
    )
    state_tensors, _ = _load_state_tensors(
        path=args.query_state_embeddings,
        report=query_state_report,
        ordered_queries=ordered_queries,
        expected_width=int(key_bank.applicability_embeddings.shape[1]),
    )
    (
        memory_ids,
        paired_ids,
        key_vectors,
        query_vectors,
        metadata,
    ) = _paired_state_inputs(
        state_tensors=state_tensors,
        ordered_queries=ordered_queries,
        anchors=anchors,
        key_bank=key_bank,
    )
    rows, _ = _score_variants(
        memory_ids=memory_ids,
        paired_ids=paired_ids,
        key_vectors=key_vectors,
        query_vectors=query_vectors,
        metadata=metadata,
        top_n=args.top_n,
    )
    summaries = {
        variant: anchor_summary(
            rows[variant],
            memory_ids=memory_ids,
            permutation_count=args.permutation_count,
        )
        for variant in V36_STATE_KEY_VARIANTS
    }
    geometry = _cross_trajectory_geometry(
        paired_ids=paired_ids,
        key_vectors=key_vectors,
        query_vectors=query_vectors,
        memory_ids=memory_ids,
        metadata=metadata,
    )
    comparisons = _comparisons(rows)
    text_by_id = _reconstruct_dynamic_texts(
        approved_records=approved_records,
        verified_experiences=verified_experiences,
        records=records,
        key_bank=key_bank,
    )
    hub_text_rows, top_hub_ids = _hub_text_rows(
        summaries=summaries,
        text_by_id=text_by_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / EVIDENCE_FILE
    evidence_count = write_jsonl(
        evidence_path,
        (
            _clean_evidence(row)
            for variant in V36_STATE_KEY_VARIANTS
            for row in rows[variant]
        ),
    )
    hub_path = args.output_dir / HUB_TEXT_FILE
    hub_count = write_jsonl(hub_path, hub_text_rows)
    tensor_path, manifest_path, manifest = _save_state_key_bank(
        output_dir=args.output_dir,
        memory_ids=memory_ids,
        key_vectors=key_vectors,
        ordered_queries=ordered_queries,
        anchors=anchors,
        key_bank=key_bank,
        query_state_report=query_state_report,
    )

    implementation_paths = (
        "memgen/experience/memory.py",
        "memgen/experience/phase1.py",
        "memgen/experience/v3_5_hubness.py",
        "memgen/experience/v3_5_query_state.py",
        "memgen/experience/v3_5_source_alignment.py",
        "memgen/experience/v3_6_state_keys.py",
        "memgen/model/retrieval_keys.py",
        "memgen/model/v3_5_retrieval.py",
        "scripts/audit_v3_5_dynamic_hubness.py",
        "scripts/audit_v3_5_dynamic_query_state.py",
        "scripts/audit_v3_6_source_state_keys.py",
    )
    implementation_hashes = {
        path: file_sha256(PROJECT_ROOT / path) for path in implementation_paths
    }
    report: dict[str, Any] = {
        "schema_version": V36_STATE_KEY_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "completed_diagnostic",
        "diagnostic_only": True,
        "qualified_for_online_use": False,
        "formal_v3_5_qualification_changed": False,
        "reasoner_forward_or_generation_run": False,
        "side_kv_used": False,
        "side_kv_payload_changed": False,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "variant_selected": False,
        "threshold_fitted": False,
        "primary_variant": V36_STATE_KEY_PRIMARY_VARIANT,
        "text_control": V36_STATE_KEY_TEXT_CONTROL,
        "identity_control": V36_STATE_KEY_IDENTITY_CONTROL,
        "full_memory_count": len(key_bank.entries),
        "reference_key_count": len(memory_ids),
        "paired_target_query_count": len(paired_ids),
        "configuration": {
            "fixed_variants": list(V36_STATE_KEY_VARIANTS),
            "state_components": list(STATE_COMPONENTS),
            "key_trajectory_side": V36_STATE_KEY_TRAJECTORY_KEY_SIDE,
            "query_trajectory_side": V36_STATE_KEY_TRAJECTORY_QUERY_SIDE,
            "key_anchor": "reference_first_counterfactual_joint_gate_event",
            "query_anchor": "target_first_counterfactual_joint_gate_event",
            "key_context": "canonical_gsm8k_prompt_question_full_partial_cot",
            "query_context": "canonical_gsm8k_prompt_question_full_partial_cot",
            "layer_number": 24,
            "normalization": "l2",
            "local_window_size": 16,
            "retrieval_scope": "reference_first_gate_eligible_state_key_bank",
            "retrieval_method": "exact_cosine",
            "stable_tie_break": "memory_id_ascending",
            "permutation_count": args.permutation_count,
            "target_and_reference_tensor_origins_distinct": True,
            "exact_cross_trajectory_embedding_matches_measured": True,
            "same_question_prompt_identity_measured": True,
            "value_source": (
                "existing_full_when_facing_prefer_avoid_side_kv_unchanged"
            ),
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.verified_experiences),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "memory_records_sha256": file_sha256(args.memory_records),
            "dual_key_manifest_sha256": file_sha256(args.dual_key_manifest),
            "dual_key_manifest_logical_sha256": key_bank.manifest_sha256,
            "v35_offline_report_sha256": file_sha256(args.v35_offline_report),
            "token_risk_artifact_sha256": file_sha256(args.token_risk_artifact),
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
            "query_state_report_sha256": file_sha256(args.query_state_report),
            "query_state_report_logical_sha256": query_state_report[
                "report_sha256"
            ],
            "query_state_embeddings_sha256": file_sha256(
                args.query_state_embeddings
            ),
            "git_revision": git_revision(),
            "implementation_files_sha256": implementation_hashes,
            "implementation_set_sha256": canonical_json_sha256(
                implementation_hashes
            ),
        },
        "artifacts": {
            "state_key_bank": {
                "tensor_path": tensor_path.name,
                "tensor_sha256": file_sha256(tensor_path),
                "manifest_path": manifest_path.name,
                "manifest_sha256": file_sha256(manifest_path),
                "manifest_logical_sha256": manifest["manifest_sha256"],
                "memory_count": len(memory_ids),
                "tensor_count": len(memory_ids) * len(STATE_COMPONENTS),
            },
            "evidence": {
                "path": evidence_path.name,
                "sha256": file_sha256(evidence_path),
                "row_count": evidence_count,
            },
            "hub_text_audit": {
                "path": hub_path.name,
                "sha256": file_sha256(hub_path),
                "row_count": hub_count,
            },
        },
        "variants": summaries,
        "paired_variant_comparisons": comparisons,
        "cross_trajectory_own_key_cosine": geometry,
        "key_geometry": {
            component: _key_geometry(key_vectors[component])
            for component in STATE_COMPONENTS
        },
        "top_hub_ids_by_variant": top_hub_ids,
        "requirements": {
            "source_alignment_report_authenticated": True,
            "source_alignment_evidence_authenticated": True,
            "query_state_report_authenticated": True,
            "query_state_tensor_sidecar_authenticated": True,
            "reference_failure_states_used_as_keys": True,
            "paired_target_success_states_used_as_queries": True,
            "key_and_query_use_matching_causal_state_components": True,
            "target_and_reference_tensor_origins_distinct": True,
            "exact_cross_trajectory_embedding_matches_measured": True,
            "prompt_identity_control_exact": True,
            "text_applicability_control_retained": True,
            "static_shortlist_bypassed_within_reference_key_universe": True,
            "side_kv_not_loaded_or_changed": True,
            "full_when_facing_prefer_avoid_value_binding_preserved": True,
            "variant_not_selected": True,
            "threshold_not_fitted": True,
            "reasoner_forward_not_run": True,
            "generation_not_run": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
            "formal_v3_5_qualification_unchanged": True,
            "qualified_for_online_use_false": True,
        },
        "interpretation_contract": {
            "prompt_identity_strong": (
                "same_question_episode_identity_ceiling_not_state_transfer"
            ),
            "current_strong_delta_weak": (
                "apparent_state_key_gain_is_largely_shared_prompt_identity"
            ),
            "delta_retains_gain": (
                "cross_trajectory_dynamics_align_beyond_shared_prompt_component"
            ),
            "local_stronger_than_current_without_hub_collapse": (
                "local_reasoning_context_is_a_better_same_space_state_key"
            ),
            "state_variants_weak": (
                "single_source_state_keys_do_not_form_stable_transferable_keys"
            ),
            "strong_result_not_cross_problem_utility": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = args.output_dir / REPORT_FILE
    write_json(report_path, report)
    write_text(args.output_dir / MARKDOWN_FILE, _markdown(report))
    print(
        "[v3.6-state-key] "
        f"status={report['status']} keys={len(memory_ids)} "
        f"paired_queries={len(paired_ids)} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
