#!/usr/bin/env python3
"""Audit which aligned V3.5 key component is readable by runtime queries.

This CPU-only diagnostic reuses authenticated exact first-gate query tensors.
It compares the original applicability keys, current full dynamic keys, and
the normalized per-memory dynamic-minus-applicability direction.  It performs
no model forward, generation, side-KV treatment, answer/reward access, variant
search, threshold fitting, or formal V3.5 qualification change.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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
from memgen.experience.risk import deterministic_train_partition
from memgen.experience.v3_5_hubness import (
    V35_HUBNESS_REPORT_SCHEMA,
    anchor_summary,
    numeric_summary,
)
from memgen.experience.v3_5_key_components import (
    V35_KEY_COMPONENT_CURRENT_VARIANT,
    V35_KEY_COMPONENT_EVIDENCE_SCHEMA,
    V35_KEY_COMPONENT_PRIMARY_SIDE,
    V35_KEY_COMPONENT_REPORT_SCHEMA,
    V35_KEY_COMPONENT_TENSOR_SCHEMA,
    V35_KEY_COMPONENT_VARIANTS,
    pairwise_variant_comparisons,
)
from memgen.experience.v3_5_selector import deterministic_source_pair_partition
from memgen.experience.v3_5_source_alignment import score_query
from scripts.audit_v3_5_dynamic_hubness import (
    _clean_evidence,
    _key_geometry,
    _load_queries,
    _normalize_rows,
    _reconstruct_dynamic_texts,
    _validate_source_report,
)


REPORT_FILE = "key_component_report.json"
MARKDOWN_FILE = "key_component_report.md"
EVIDENCE_FILE = "key_component_evidence.jsonl"
TENSOR_FILE = "key_component_tensors.safetensors"
HUB_TEXT_FILE = "key_component_hub_text_audit.jsonl"
HUB_TEXT_TOP_N = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--verified-experiences", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--dual-key-manifest", type=Path, required=True)
    parser.add_argument("--source-alignment-report", type=Path, required=True)
    parser.add_argument("--first-gate-queries", type=Path, required=True)
    parser.add_argument("--hubness-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--permutation-count", type=int, default=10_000)
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
        raise RuntimeError("key-component audit requires a git revision") from exc
    if not revision:
        raise RuntimeError("key-component audit resolved an empty git revision")
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
    for path in (
        args.approved_bank,
        args.verified_experiences,
        args.memory_records,
        args.dual_key_manifest,
        args.source_alignment_report,
        args.first_gate_queries,
        args.hubness_report,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def _logical_report_is_authenticated(report: Mapping[str, Any]) -> bool:
    stored = report.get("report_sha256")
    logical = dict(report)
    logical.pop("report_sha256", None)
    return isinstance(stored, str) and stored == canonical_json_sha256(logical)


def _validate_hubness_report(
    *,
    args: argparse.Namespace,
    report: Mapping[str, Any],
    source_report: Mapping[str, Any],
    memory_count: int,
    query_count: int,
) -> None:
    inputs = report.get("inputs", {})
    requirements = report.get("requirements", {})
    implementation = inputs.get("implementation_files_sha256", {})
    implementation_ok = (
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
    if (
        report.get("schema_version") != V35_HUBNESS_REPORT_SCHEMA
        or report.get("status") != "completed_diagnostic"
        or report.get("diagnostic_only") is not True
        or report.get("formal_v3_5_qualification_changed") is not False
        or report.get("reasoner_forward_or_generation_run") is not False
        or report.get("task_accuracy_used") is not False
        or report.get("answer_or_reward_used") is not False
        or report.get("variant_selected") is not False
        or not _logical_report_is_authenticated(report)
        or not isinstance(requirements, Mapping)
        or not requirements
        or not all(value is True for value in requirements.values())
        or not implementation_ok
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
    ):
        raise ValueError("key-component audit cannot authenticate hubness report")


def _compile_spaces(
    applicability_keys: Any,
    dynamic_keys: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    applicability = _normalize_rows(
        applicability_keys.double(), owner="applicability keys"
    )
    dynamic = _normalize_rows(dynamic_keys.double(), owner="dynamic keys")
    if applicability.shape != dynamic.shape:
        raise ValueError("key-component aligned key shapes differ")
    raw_residual = dynamic - applicability
    residual_norms = raw_residual.norm(dim=-1)
    if (
        not torch.isfinite(raw_residual).all()
        or not torch.isfinite(residual_norms).all()
        or bool((residual_norms <= 1e-12).any().item())
    ):
        raise ValueError("key-component paired residual has a zero or invalid row")
    residual = _normalize_rows(
        raw_residual, owner="paired dynamic-minus-applicability residual"
    )
    spaces = {
        "applicability_key": applicability,
        "dynamic_key": dynamic,
        "paired_decision_residual": residual,
    }
    if tuple(spaces) != V35_KEY_COMPONENT_VARIANTS:
        raise AssertionError("key-component fixed variant order drifted")
    geometry = {
        "raw_dynamic_minus_applicability": raw_residual.contiguous(),
        "paired_residual_norm": numeric_summary(residual_norms.tolist()),
        "applicability_dynamic_cosine": numeric_summary(
            (applicability * dynamic).sum(dim=-1).tolist()
        ),
        "residual_to_applicability_cosine": numeric_summary(
            (residual * applicability).sum(dim=-1).tolist()
        ),
        "residual_to_dynamic_cosine": numeric_summary(
            (residual * dynamic).sum(dim=-1).tolist()
        ),
    }
    return spaces, geometry


def _score_variants(
    *,
    spaces: Mapping[str, Any],
    queries: Any,
    query_metadata: Sequence[Mapping[str, Any]],
    memory_ids: Sequence[str],
    top_n: int,
    risk_seed: int,
    risk_fraction: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    matrices: dict[str, Any] = {}
    for variant in V35_KEY_COMPONENT_VARIANTS:
        score_matrix = queries @ spaces[variant].T
        matrices[variant] = score_matrix
        rows: list[dict[str, Any]] = []
        for index, metadata in enumerate(query_metadata):
            scored = score_query(
                memory_ids=memory_ids,
                scores=[float(value) for value in score_matrix[index].tolist()],
                own_memory_id=str(metadata["memory_id"]),
                top_n=top_n,
                include_rank_lookup=True,
            )
            rows.append({
                "schema_version": V35_KEY_COMPONENT_EVIDENCE_SCHEMA,
                "variant": variant,
                **metadata,
                "selector_partition": deterministic_source_pair_partition(
                    str(metadata["memory_id"]),
                    str(metadata["source_experience_id"]),
                ),
                "risk_partition": (
                    "train"
                    if deterministic_train_partition(
                        str(metadata["source_experience_id"]),
                        seed=risk_seed,
                        train_fraction=risk_fraction,
                    )
                    else "holdout"
                ),
                **scored,
            })
        rows_by_variant[variant] = rows
    return rows_by_variant, matrices


def _hub_text_rows(
    *,
    summaries: Mapping[str, Any],
    rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    matrices: Mapping[str, Any],
    memory_ids: Sequence[str],
    text_by_id: Mapping[str, Mapping[str, Any]],
    spaces: Mapping[str, Any],
    raw_residual: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: set[str] = set()
    top_ids: dict[str, dict[str, list[str]]] = {}
    for variant in V35_KEY_COMPONENT_VARIANTS:
        top_ids[variant] = {}
        for side in ("reference", "target"):
            ids = [
                str(item["memory_id"])
                for item in summaries[variant][side]["hubness"]["top_memories"][
                    :HUB_TEXT_TOP_N
                ]
            ]
            top_ids[variant][side] = ids
            selected.update(ids)

    index_by_id = {memory_id: index for index, memory_id in enumerate(memory_ids)}
    audit_rows: list[dict[str, Any]] = []
    for memory_id in sorted(selected):
        index = index_by_id[memory_id]
        statistics: dict[str, Any] = {}
        for variant in V35_KEY_COMPONENT_VARIANTS:
            score_matrix = matrices[variant]
            statistics[variant] = {
                "mean_query_cosine": float(score_matrix[:, index].mean().item()),
                "reference_top1_count": sum(
                    row["trajectory_side"] == "reference"
                    and row["top1_memory_id"] == memory_id
                    for row in rows_by_variant[variant]
                ),
                "target_top1_count": sum(
                    row["trajectory_side"] == "target"
                    and row["top1_memory_id"] == memory_id
                    for row in rows_by_variant[variant]
                ),
            }
        applicability = spaces["applicability_key"][index]
        dynamic = spaces["dynamic_key"][index]
        residual = spaces["paired_decision_residual"][index]
        audit_rows.append({
            **text_by_id[memory_id],
            "applicability_dynamic_cosine": float((applicability @ dynamic).item()),
            "raw_dynamic_minus_applicability_norm": float(
                raw_residual[index].norm().item()
            ),
            "residual_to_applicability_cosine": float(
                (residual @ applicability).item()
            ),
            "residual_to_dynamic_cosine": float((residual @ dynamic).item()),
            "variant_statistics": statistics,
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
        "# MemGen V3.5 Dynamic Key-Component Audit",
        "",
        f"- Status: `{report.get('status')}`",
        "- Diagnostic only: `true`",
        "- Formal V3.5 qualification changed: `false`",
        "- Reasoner forward or generation run: `false`",
        "- Query tensors changed: `false`",
        f"- Fixed variants: `{', '.join(V35_KEY_COMPONENT_VARIANTS)}`",
        "",
        "| Side | Key component | MRR | R@1 | R@5 | R@10 | R@32 | "
        "Top-1 hub share | Top-2 share | Selected keys | Gini |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    variants = report["variants"]
    for side in ("reference", "target"):
        for variant in V35_KEY_COMPONENT_VARIANTS:
            summary = variants[variant][side]
            metrics = summary["all"]
            hubness = summary["hubness"]
            lines.append(
                f"| {side} | {variant} | {metrics['mrr']:.6f} | "
                f"{metrics['recall_at_1']:.6f} | {metrics['recall_at_5']:.6f} | "
                f"{metrics['recall_at_10']:.6f} | {metrics['recall_at_32']:.6f} | "
                f"{hubness['top1_share']:.6f} | "
                f"{hubness['top2_combined_share']:.6f} | "
                f"{hubness['selected_memory_count']} | "
                f"{hubness['selection_gini_over_full_bank']:.6f} |"
            )
    lines.extend([
        "",
        "Interpretation is comparative: applicability beating dynamic means the",
        "appended Prefer component harms the current query/key match; a strong paired",
        "residual means decision information exists but is masked in the full key; all",
        "three remaining weak rejects this key construction for the current "
        "runtime query.",
        "No variant is selected for online use by this diagnostic.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    _validate_args(args)

    from safetensors.torch import save_file

    from memgen.model.retrieval_keys import tensor_sha256
    from memgen.model.v3_5_retrieval import DualRetrievalKeyBankLoader

    source_report = json.loads(
        args.source_alignment_report.read_text(encoding="utf-8")
    )
    key_bank = DualRetrievalKeyBankLoader(manifest_path=args.dual_key_manifest)
    ordered = _validate_source_report(
        args=args,
        report=source_report,
        key_bank=key_bank,
    )
    queries, query_metadata = _load_queries(
        query_path=args.first_gate_queries,
        ordered=ordered,
        expected_width=int(key_bank.dynamic_embeddings.shape[1]),
    )
    queries = _normalize_rows(queries.double(), owner="first-gate queries")
    memory_ids = tuple(str(entry["memory_id"]) for entry in key_bank.entries)
    if any(row["memory_id"] not in memory_ids for row in query_metadata):
        raise ValueError("key-component query references a memory outside the bank")

    hubness_report = json.loads(args.hubness_report.read_text(encoding="utf-8"))
    _validate_hubness_report(
        args=args,
        report=hubness_report,
        source_report=source_report,
        memory_count=len(memory_ids),
        query_count=len(query_metadata),
    )

    spaces, component_geometry = _compile_spaces(
        key_bank.applicability_embeddings,
        key_bank.dynamic_embeddings,
    )
    risk_seed = int(source_report["configuration"]["risk_split_seed"])
    risk_fraction = float(source_report["configuration"]["risk_train_fraction"])
    rows_by_variant, score_matrices = _score_variants(
        spaces=spaces,
        queries=queries,
        query_metadata=query_metadata,
        memory_ids=memory_ids,
        top_n=args.top_n,
        risk_seed=risk_seed,
        risk_fraction=risk_fraction,
    )

    summaries: dict[str, Any] = {}
    for variant in V35_KEY_COMPONENT_VARIANTS:
        summaries[variant] = {}
        for side in ("reference", "target"):
            side_rows = [
                row
                for row in rows_by_variant[variant]
                if row["trajectory_side"] == side
            ]
            summaries[variant][side] = anchor_summary(
                side_rows,
                memory_ids=memory_ids,
                permutation_count=args.permutation_count,
            )
        summaries[variant]["key_geometry"] = _key_geometry(spaces[variant])

    hubness_raw_deltas = {
        side: _metric_delta(
            summaries["dynamic_key"][side]["all"],
            hubness_report["variants"]["raw"][side]["all"],
        )
        for side in ("reference", "target")
    }
    if any(
        delta is None
        or any(abs(value) > 1e-15 for value in delta.values())
        for delta in hubness_raw_deltas.values()
    ):
        raise ValueError(
            "key-component dynamic variant does not reproduce hubness raw metrics"
        )

    comparisons = pairwise_variant_comparisons(rows_by_variant)
    for comparison in comparisons.values():
        baseline = str(comparison["baseline_variant"])
        candidate = str(comparison["candidate_variant"])
        for side in ("reference", "target"):
            comparison["by_side"][side][
                "hubness_delta_candidate_minus_baseline"
            ] = {
                field: (
                    summaries[candidate][side]["hubness"][field]
                    - summaries[baseline][side]["hubness"][field]
                )
                for field in (
                    "top1_share",
                    "top2_combined_share",
                    "selected_memory_count",
                    "selection_gini_over_full_bank",
                )
            }

    approved_records = list(iter_jsonl(args.approved_bank))
    verified_experiences = list(iter_jsonl(args.verified_experiences))
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    text_by_id = _reconstruct_dynamic_texts(
        approved_records=approved_records,
        verified_experiences=verified_experiences,
        records=records,
        key_bank=key_bank,
    )
    hub_text_rows, top_hub_ids = _hub_text_rows(
        summaries=summaries,
        rows_by_variant=rows_by_variant,
        matrices=score_matrices,
        memory_ids=memory_ids,
        text_by_id=text_by_id,
        spaces=spaces,
        raw_residual=component_geometry["raw_dynamic_minus_applicability"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / EVIDENCE_FILE
    evidence_count = write_jsonl(
        evidence_path,
        (
            _clean_evidence(row)
            for variant in V35_KEY_COMPONENT_VARIANTS
            for row in rows_by_variant[variant]
        ),
    )
    hub_text_path = args.output_dir / HUB_TEXT_FILE
    hub_text_count = write_jsonl(hub_text_path, hub_text_rows)
    tensor_path = args.output_dir / TENSOR_FILE
    component_tensors = {
        "raw_dynamic_minus_applicability": component_geometry[
            "raw_dynamic_minus_applicability"
        ],
        "paired_decision_residual_embeddings": spaces[
            "paired_decision_residual"
        ],
    }
    save_file(
        {
            name: value.contiguous().cpu()
            for name, value in component_tensors.items()
        },
        str(tensor_path),
        metadata={
            "schema_version": V35_KEY_COMPONENT_TENSOR_SCHEMA,
            "variant_order_sha256": canonical_json_sha256(
                V35_KEY_COMPONENT_VARIANTS
            ),
        },
    )

    implementation_paths = (
        "memgen/experience/memory.py",
        "memgen/experience/phase1.py",
        "memgen/experience/risk.py",
        "memgen/experience/v3_5_selector.py",
        "memgen/experience/v3_5_source_alignment.py",
        "memgen/experience/v3_5_hubness.py",
        "memgen/experience/v3_5_key_components.py",
        "memgen/model/retrieval_keys.py",
        "memgen/model/v3_5_retrieval.py",
        "scripts/audit_v3_5_dynamic_hubness.py",
        "scripts/audit_v3_5_dynamic_key_components.py",
    )
    implementation_hashes = {
        path: file_sha256(PROJECT_ROOT / path) for path in implementation_paths
    }
    source_raw_reference = source_report["primary"]["all"]
    source_raw_target = source_report["secondary"]["target_first_gate"]["all"]
    hubness_raw_reference = hubness_report["variants"]["raw"]["reference"]["all"]
    hubness_raw_target = hubness_report["variants"]["raw"]["target"]["all"]
    report: dict[str, Any] = {
        "schema_version": V35_KEY_COMPONENT_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "completed_diagnostic",
        "diagnostic_only": True,
        "formal_v3_5_qualification_changed": False,
        "reasoner_forward_or_generation_run": False,
        "query_tensors_changed": False,
        "side_kv_used": False,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "variant_selected": False,
        "primary_side": V35_KEY_COMPONENT_PRIMARY_SIDE,
        "current_online_key_variant": V35_KEY_COMPONENT_CURRENT_VARIANT,
        "memory_count": len(memory_ids),
        "query_count": len(query_metadata),
        "configuration": {
            "fixed_variants": list(V35_KEY_COMPONENT_VARIANTS),
            "applicability_definition": "authenticated_v3_when_facing_key",
            "dynamic_definition": (
                "authenticated_v35_when_facing_plus_prefer_key"
            ),
            "paired_decision_residual_definition": (
                "l2_normalize(dynamic_key_i_minus_applicability_key_i)"
            ),
            "paired_decision_residual_semantics": (
                "encoder_direction_induced_by_appending_prefer_not_a_"
                "standalone_decision_text_embedding"
            ),
            "query_transform_for_all_variants": "none_beyond_existing_l2",
            "compute_device": "cpu",
            "compute_dtype": "float64",
            "retrieval_method": "exact_cosine_over_full_bank",
            "stable_tie_break": "memory_id_ascending",
            "top_n": args.top_n,
            "permutation_count": args.permutation_count,
            "permutation_policy": (
                "shuffle_query_to_own_memory_binding_preserve_variant_score_matrix"
            ),
            "hub_text_top_n_per_variant_side": HUB_TEXT_TOP_N,
        },
        "component_geometry": {
            key: value
            for key, value in component_geometry.items()
            if key != "raw_dynamic_minus_applicability"
        },
        "component_tensor_sha256": {
            name: tensor_sha256(value)
            for name, value in component_tensors.items()
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(
                args.verified_experiences
            ),
            "memory_records_sha256": file_sha256(args.memory_records),
            "dual_key_manifest_sha256": file_sha256(args.dual_key_manifest),
            "dual_key_manifest_logical_sha256": key_bank.manifest_sha256,
            "source_alignment_report_sha256": file_sha256(
                args.source_alignment_report
            ),
            "source_alignment_report_logical_sha256": source_report[
                "report_sha256"
            ],
            "first_gate_queries_sha256": file_sha256(args.first_gate_queries),
            "hubness_report_sha256": file_sha256(args.hubness_report),
            "hubness_report_logical_sha256": hubness_report["report_sha256"],
            "git_revision": git_revision(),
            "implementation_files_sha256": implementation_hashes,
            "implementation_set_sha256": canonical_json_sha256(
                implementation_hashes
            ),
        },
        "artifacts": {
            "variant_evidence": {
                "path": evidence_path.name,
                "sha256": file_sha256(evidence_path),
                "row_count": evidence_count,
            },
            "component_tensors": {
                "path": tensor_path.name,
                "sha256": file_sha256(tensor_path),
                "tensor_names": sorted(component_tensors),
            },
            "hub_key_text_audit": {
                "path": hub_text_path.name,
                "sha256": file_sha256(hub_text_path),
                "row_count": hub_text_count,
            },
        },
        "source_and_hubness_raw_metrics": {
            "source_report": {
                "reference": source_raw_reference,
                "target": source_raw_target,
            },
            "hubness_report": {
                "reference": hubness_raw_reference,
                "target": hubness_raw_target,
            },
        },
        "recomputed_dynamic_metric_delta": {
            "reference_minus_source_report": _metric_delta(
                summaries["dynamic_key"]["reference"]["all"],
                source_raw_reference,
            ),
            "target_minus_source_report": _metric_delta(
                summaries["dynamic_key"]["target"]["all"],
                source_raw_target,
            ),
            "reference_minus_hubness_raw": _metric_delta(
                summaries["dynamic_key"]["reference"]["all"],
                hubness_raw_reference,
            ),
            "target_minus_hubness_raw": _metric_delta(
                summaries["dynamic_key"]["target"]["all"],
                hubness_raw_target,
            ),
        },
        "variants": summaries,
        "pairwise_comparisons": comparisons,
        "top_hub_ids_by_variant_side": top_hub_ids,
        "requirements": {
            "source_alignment_report_authenticated": True,
            "hubness_report_authenticated": True,
            "exact_first_gate_query_sidecar_authenticated": True,
            "dual_key_bank_authenticated": True,
            "memory_and_source_text_join_authenticated": True,
            "fixed_three_key_components_only": True,
            "applicability_and_dynamic_rows_paired_by_bank_index": True,
            "residual_is_dynamic_minus_applicability": True,
            "residual_l2_normalized_per_memory": True,
            "query_tensors_identical_across_variants": True,
            "query_labels_not_used_to_construct_keys": True,
            "variant_not_selected": True,
            "threshold_not_fitted": True,
            "reasoner_forward_or_generation_not_run": True,
            "side_kv_not_used": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
            "formal_v3_5_qualification_unchanged": True,
        },
        "interpretation_contract": {
            "applicability_similar_to_dynamic_and_residual_weak": (
                "prefer_information_is_not_readable_from_current_runtime_query_"
                "redesign_key_pooling_or_query_representation"
            ),
            "applicability_stronger_than_dynamic": (
                "appended_prefer_component_harms_current_query_key_alignment"
            ),
            "paired_decision_residual_stronger_than_both_full_keys": (
                "decision_direction_exists_but_is_masked_factorized_decision_"
                "channel_is_warranted"
            ),
            "all_three_weak": (
                "current_abstract_key_construction_is_not_recoverable_from_"
                "runtime_prefix_queries"
            ),
            "strong_in_source_result_not_cross_problem_utility": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = args.output_dir / REPORT_FILE
    write_json(report_path, report)
    write_text(args.output_dir / MARKDOWN_FILE, _markdown(report))
    print(
        "[v3.5-key-components] "
        f"status={report['status']} queries={len(query_metadata)} "
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
