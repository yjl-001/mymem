#!/usr/bin/env python3
"""Decompose V3.5 dynamic retrieval hubness with fixed key-only transforms.

This audit reuses authenticated exact first-gate query tensors.  It performs no
reasoner forward, generation, side-KV treatment, answer/reward access, variant
search, threshold fitting, or formal V3.5 qualification change.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

from memgen.experience.memory import (
    ApprovedMemorySourceSelector,
    MemoryRecord,
    MemorySanitizerConfig,
    PayloadSanitizer,
)
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.risk import deterministic_train_partition
from memgen.experience.v3_5_hubness import (
    V35_HUBNESS_EVIDENCE_SCHEMA,
    V35_HUBNESS_PRIMARY_SIDE,
    V35_HUBNESS_REPORT_SCHEMA,
    V35_HUBNESS_TRANSFORM_SCHEMA,
    V35_HUBNESS_VARIANTS,
    anchor_summary,
    compare_variant_rows,
    numeric_summary,
)
from memgen.experience.v3_5_selector import deterministic_source_pair_partition
from memgen.experience.v3_5_source_alignment import (
    V35_SOURCE_ALIGNMENT_QUERY_SIDECAR_SCHEMA,
    V35_SOURCE_ALIGNMENT_REPORT_SCHEMA,
    score_query,
)


REPORT_FILE = "hubness_decomposition_report.json"
MARKDOWN_FILE = "hubness_decomposition_report.md"
EVIDENCE_FILE = "hubness_variant_evidence.jsonl"
TRANSFORM_FILE = "hubness_transforms.safetensors"
HUB_TEXT_FILE = "hub_key_text_audit.jsonl"
HUB_TEXT_TOP_N = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--verified-experiences", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--dual-key-manifest", type=Path, required=True)
    parser.add_argument("--source-alignment-report", type=Path, required=True)
    parser.add_argument("--first-gate-queries", type=Path, required=True)
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
        raise RuntimeError("hubness audit requires a git revision") from exc
    if not revision:
        raise RuntimeError("hubness audit resolved an empty git revision")
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
    ):
        if not path.is_file():
            raise FileNotFoundError(path)


def _logical_report_is_authenticated(report: Mapping[str, Any]) -> bool:
    stored = report.get("report_sha256")
    logical = dict(report)
    logical.pop("report_sha256", None)
    return isinstance(stored, str) and stored == canonical_json_sha256(logical)


def _validate_source_report(
    *,
    args: argparse.Namespace,
    report: Mapping[str, Any],
    key_bank: Any,
) -> tuple[Mapping[str, Any], ...]:
    requirements = report.get("requirements", {})
    inputs = report.get("inputs", {})
    query_artifact = report.get("artifacts", {}).get(
        "first_gate_query_embeddings", {}
    )
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
    ordered = query_artifact.get("ordered_queries")
    if (
        report.get("schema_version") != V35_SOURCE_ALIGNMENT_REPORT_SCHEMA
        or report.get("status") != "completed_diagnostic"
        or report.get("diagnostic_only") is not True
        or report.get("formal_v3_5_qualification_changed") is not False
        or report.get("task_accuracy_used") is not False
        or report.get("answer_or_reward_used") is not False
        or not _logical_report_is_authenticated(report)
        or not isinstance(requirements, Mapping)
        or not requirements
        or not all(value is True for value in requirements.values())
        or requirements.get("exact_anchor_reencoded") is not True
        or not implementation_ok
        or inputs.get("approved_bank_sha256") != file_sha256(args.approved_bank)
        or inputs.get("verified_experiences_sha256")
        != file_sha256(args.verified_experiences)
        or inputs.get("memory_records_sha256") != file_sha256(args.memory_records)
        or inputs.get("dual_key_manifest_sha256")
        != file_sha256(args.dual_key_manifest)
        or inputs.get("dual_key_manifest_logical_sha256")
        != key_bank.manifest_sha256
        or query_artifact.get("sha256") != file_sha256(args.first_gate_queries)
        or not isinstance(ordered, list)
        or not ordered
        or int(query_artifact.get("tensor_count", -1)) != len(ordered)
        or query_artifact.get("ordered_queries_sha256")
        != canonical_json_sha256(ordered)
    ):
        raise ValueError("hubness audit cannot authenticate source alignment report")
    if any(not isinstance(item, Mapping) for item in ordered):
        raise ValueError("source alignment query order contains a malformed row")
    values = tuple(ordered)
    tensor_names = [str(item.get("tensor_name", "")) for item in values]
    identities = [
        (str(item.get("trajectory_side", "")), str(item.get("memory_id", "")))
        for item in values
    ]
    if (
        any(not name for name in tensor_names)
        or len(set(tensor_names)) != len(tensor_names)
        or len(set(identities)) != len(identities)
        or any(side not in {"target", "reference"} for side, _ in identities)
    ):
        raise ValueError("source alignment query order is malformed")
    expected_counts = {
        "reference": int(report.get("primary", {}).get("eligible_count", -1)),
        "target": int(
            report.get("secondary", {})
            .get("target_first_gate", {})
            .get("eligible_count", -1)
        ),
    }
    actual_counts = Counter(side for side, _ in identities)
    if any(actual_counts[side] != expected for side, expected in expected_counts.items()):
        raise ValueError("source alignment first-gate query counts drifted")
    return values


def _load_queries(
    *,
    query_path: Path,
    ordered: Sequence[Mapping[str, Any]],
    expected_width: int,
) -> tuple[Any, list[dict[str, Any]]]:
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    from memgen.model.retrieval_keys import tensor_sha256

    with safe_open(str(query_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        tensor_names = set(handle.keys())
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version")
        != V35_SOURCE_ALIGNMENT_QUERY_SIDECAR_SCHEMA
        or metadata.get("ordered_queries_sha256")
        != canonical_json_sha256(ordered)
        or tensor_names != {str(item["tensor_name"]) for item in ordered}
    ):
        raise ValueError("hubness audit query sidecar metadata drifted")
    tensors = load_file(str(query_path), device="cpu")
    vectors: list[Any] = []
    rows: list[dict[str, Any]] = []
    for item in ordered:
        name = str(item["tensor_name"])
        vector = tensors[name].detach().float().reshape(-1).contiguous()
        if (
            tuple(vector.shape) != (expected_width,)
            or not torch.isfinite(vector).all()
            or not math.isclose(
                float(vector.norm().item()), 1.0, rel_tol=0.0, abs_tol=1e-5
            )
            or tensor_sha256(vector) != item.get("query_embedding_sha256")
        ):
            raise ValueError(f"hubness query tensor drifted: {name}")
        vectors.append(vector)
        rows.append({
            "tensor_name": name,
            "memory_id": str(item["memory_id"]),
            "source_experience_id": str(item["source_experience_id"]),
            "trajectory_side": str(item["trajectory_side"]),
            "query_embedding_sha256": str(item["query_embedding_sha256"]),
        })
    return torch.stack(vectors, dim=0).double(), rows


def _normalize_rows(value: Any, *, owner: str) -> Any:
    import torch

    norms = value.norm(dim=-1, keepdim=True)
    if (
        value.ndim != 2
        or not torch.isfinite(value).all()
        or not torch.isfinite(norms).all()
        or bool((norms <= 1e-12).any().item())
    ):
        raise ValueError(f"hubness {owner} has an invalid vector")
    return (value / norms).contiguous()


def _compile_spaces(raw_keys: Any, raw_queries: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    raw_keys = _normalize_rows(raw_keys.double(), owner="raw keys")
    raw_queries = _normalize_rows(raw_queries.double(), owner="raw queries")
    centroid = raw_keys.mean(dim=0)
    centered_key_residual = raw_keys - centroid.unsqueeze(0)
    centered_query_residual = raw_queries - centroid.unsqueeze(0)
    centered_keys = _normalize_rows(centered_key_residual, owner="centered keys")
    centered_queries = _normalize_rows(
        centered_query_residual, owner="centered queries"
    )

    _, singular_values, vh = torch.linalg.svd(
        centered_key_residual, full_matrices=False
    )
    pc1 = vh[0].contiguous()
    pivot = int(pc1.abs().argmax().item())
    if float(pc1[pivot].item()) < 0.0:
        pc1 = -pc1
    key_pc1_projection = centered_key_residual @ pc1
    query_pc1_projection = centered_query_residual @ pc1
    removed_keys = _normalize_rows(
        centered_key_residual - key_pc1_projection.unsqueeze(1) * pc1.unsqueeze(0),
        owner="centered PC1-removed keys",
    )
    removed_queries = _normalize_rows(
        centered_query_residual
        - query_pc1_projection.unsqueeze(1) * pc1.unsqueeze(0),
        owner="centered PC1-removed queries",
    )
    explained = singular_values.square()
    pc1_share = float((explained[0] / explained.sum()).item())
    spaces = {
        "raw": (raw_keys, raw_queries),
        "key_centroid_centered": (centered_keys, centered_queries),
        "key_centroid_centered_remove_pc1": (removed_keys, removed_queries),
    }
    if tuple(spaces) != V35_HUBNESS_VARIANTS:
        raise AssertionError("hubness fixed variant order drifted")
    transform = {
        "key_centroid": centroid.contiguous(),
        "centered_key_pc1": pc1.contiguous(),
        "centered_key_singular_values": singular_values.contiguous(),
        "key_pc1_projection": key_pc1_projection.contiguous(),
        "query_pc1_projection": query_pc1_projection.contiguous(),
        "pc1_explained_variance_fraction": pc1_share,
        "pc1_sign_pivot_index": pivot,
    }
    return spaces, transform


def _key_geometry(keys: Any) -> dict[str, Any]:
    import torch

    scores = keys @ keys.T
    count = int(keys.shape[0])
    mask = ~torch.eye(count, dtype=torch.bool)
    off_diagonal = scores[mask].detach().cpu().tolist()
    nearest_other = scores.masked_fill(~mask, -torch.inf).max(dim=1).values
    return {
        "off_diagonal_cosine": numeric_summary(off_diagonal),
        "nearest_other_cosine": numeric_summary(
            nearest_other.detach().cpu().tolist()
        ),
    }


def _score_variants(
    *,
    spaces: Mapping[str, tuple[Any, Any]],
    query_metadata: Sequence[Mapping[str, Any]],
    memory_ids: Sequence[str],
    top_n: int,
    risk_seed: int,
    risk_fraction: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    matrices: dict[str, Any] = {}
    for variant in V35_HUBNESS_VARIANTS:
        keys, queries = spaces[variant]
        score_matrix = queries @ keys.T
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
                "schema_version": V35_HUBNESS_EVIDENCE_SCHEMA,
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


def _nested_text(record: Mapping[str, Any]) -> Any:
    bank = record.get("bank", {})
    target = bank.get("target", {}) if isinstance(bank, Mapping) else {}
    return target.get("transferable_decision") if isinstance(target, Mapping) else None


def _reconstruct_dynamic_texts(
    *,
    approved_records: Sequence[Mapping[str, Any]],
    verified_experiences: Sequence[Mapping[str, Any]],
    records: Sequence[MemoryRecord],
    key_bank: Any,
) -> dict[str, dict[str, Any]]:
    from memgen.model.v3_5_retrieval import sanitize_v35_dynamic_decision_text

    sources, _ = ApprovedMemorySourceSelector().join(
        approved_records, verified_experiences
    )
    source_by_id = {source.experience_id: source for source in sources}
    sanitizer = PayloadSanitizer(
        MemorySanitizerConfig(forbid_numeric_literals=True)
    )
    if [record.memory_id for record in records] != [
        str(entry["memory_id"]) for entry in key_bank.entries
    ]:
        raise ValueError("hubness memory/key record order drifted")
    result: dict[str, dict[str, Any]] = {}
    for record, entry in zip(records, key_bank.entries):
        source = source_by_id.get(record.source_experience_id)
        if source is None:
            raise ValueError(f"hubness source join lost {record.memory_id}")
        raw = _nested_text(source.approved_record)
        decision = sanitizer.sanitize_field(
            path="bank.target.transferable_decision",
            value=raw,
            source=source,
        )
        compiled, canonicalized = sanitize_v35_dynamic_decision_text(
            owner=f"{record.memory_id} transferable_decision",
            text=decision,
        )
        when_facing = str(record.sanitized_fields.get("when_facing", "")).strip()
        dynamic_text = f"When facing: {when_facing}\nPrefer: {compiled}"
        if (
            record.source_record_sha256 != entry.get("source_record_sha256")
            or record.phase1_provenance_sha256
            != entry.get("phase1_provenance_sha256")
            or text_sha256(when_facing)
            != entry.get("dynamic_when_facing_text_sha256")
            or text_sha256(decision)
            != entry.get("dynamic_decision_raw_sanitized_sha256")
            or text_sha256(compiled)
            != entry.get("dynamic_decision_compiled_sha256")
            or text_sha256(dynamic_text) != entry.get("dynamic_key_text_sha256")
            or canonicalized != entry.get("dynamic_decision_v35_canonicalized")
        ):
            raise ValueError(f"hubness dynamic text reconstruction drifted: {record.memory_id}")
        result[record.memory_id] = {
            "memory_id": record.memory_id,
            "source_experience_id": record.source_experience_id,
            "when_facing": when_facing,
            "transferable_decision": compiled,
            "dynamic_key_text": dynamic_text,
            "dynamic_key_text_sha256": text_sha256(dynamic_text),
            "dynamic_decision_v35_canonicalized": canonicalized,
        }
    return result


def _clean_evidence(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "rank_by_memory_id"}


def _metric_delta(
    left: Mapping[str, Any] | None, right: Mapping[str, Any] | None
) -> dict[str, float] | None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return None
    fields = ("mrr", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_32")
    return {field: float(left[field]) - float(right[field]) for field in fields}


def _hub_text_rows(
    *,
    summaries: Mapping[str, Any],
    rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]],
    matrices: Mapping[str, Any],
    memory_ids: Sequence[str],
    text_by_id: Mapping[str, Mapping[str, Any]],
    raw_keys: Any,
    transform: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected: set[str] = set()
    top_ids: dict[str, dict[str, list[str]]] = {}
    for variant in V35_HUBNESS_VARIANTS:
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

    centroid = transform["key_centroid"]
    centroid_direction = centroid / centroid.norm()
    pc1 = transform["centered_key_pc1"]
    centered = raw_keys - centroid.unsqueeze(0)
    index_by_id = {memory_id: index for index, memory_id in enumerate(memory_ids)}
    audit_rows: list[dict[str, Any]] = []
    for memory_id in sorted(selected):
        index = index_by_id[memory_id]
        statistics: dict[str, Any] = {}
        for variant in V35_HUBNESS_VARIANTS:
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
        audit_rows.append({
            **text_by_id[memory_id],
            "raw_key_to_centroid_direction_cosine": float(
                (raw_keys[index] @ centroid_direction).item()
            ),
            "centered_key_pc1_loading": float((centered[index] @ pc1).item()),
            "variant_statistics": statistics,
        })
    return audit_rows, top_ids


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MemGen V3.5 Dynamic Hubness Decomposition Audit",
        "",
        f"- Status: `{report.get('status')}`",
        "- Diagnostic only: `true`",
        "- Formal V3.5 qualification changed: `false`",
        "- Reasoner forward or generation run: `false`",
        f"- Fixed variants: `{', '.join(V35_HUBNESS_VARIANTS)}`",
        "",
        "| Side | Variant | MRR | R@1 | R@5 | R@10 | R@32 | "
        "Top-1 hub share | Top-2 share | Selected keys | Gini |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    variants = report.get("variants", {})
    for side in ("reference", "target"):
        for variant in V35_HUBNESS_VARIANTS:
            summary = variants[variant][side]
            metrics = summary["all"]
            hubness = summary["hubness"]
            lines.append(
                f"| {side} | {variant} | {metrics['mrr']:.6f} | "
                f"{metrics['recall_at_1']:.6f} | {metrics['recall_at_5']:.6f} | "
                f"{metrics['recall_at_10']:.6f} | {metrics['recall_at_32']:.6f} | "
                f"{hubness['top1_share']:.6f} | {hubness['top2_combined_share']:.6f} | "
                f"{hubness['selected_memory_count']} | "
                f"{hubness['selection_gini_over_full_bank']:.6f} |"
            )
    lines.extend([
        "",
        "The audit compares only pre-registered key-bank transforms. It does not",
        "select a winning variant or authorize an online retrieval change.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    _validate_args(args)

    import torch
    from safetensors.torch import save_file

    from memgen.model.retrieval_keys import tensor_sha256
    from memgen.model.v3_5_retrieval import DualRetrievalKeyBankLoader

    source_report = json.loads(
        args.source_alignment_report.read_text(encoding="utf-8")
    )
    key_bank = DualRetrievalKeyBankLoader(manifest_path=args.dual_key_manifest)
    ordered = _validate_source_report(
        args=args, report=source_report, key_bank=key_bank
    )
    raw_queries, query_metadata = _load_queries(
        query_path=args.first_gate_queries,
        ordered=ordered,
        expected_width=int(key_bank.dynamic_embeddings.shape[1]),
    )
    memory_ids = tuple(str(entry["memory_id"]) for entry in key_bank.entries)
    if any(row["memory_id"] not in memory_ids for row in query_metadata):
        raise ValueError("hubness query references a memory outside the bank")

    risk_seed = int(source_report["configuration"]["risk_split_seed"])
    risk_fraction = float(source_report["configuration"]["risk_train_fraction"])
    raw_keys = key_bank.dynamic_embeddings.double().contiguous()
    spaces, transform = _compile_spaces(raw_keys, raw_queries)
    rows_by_variant, score_matrices = _score_variants(
        spaces=spaces,
        query_metadata=query_metadata,
        memory_ids=memory_ids,
        top_n=args.top_n,
        risk_seed=risk_seed,
        risk_fraction=risk_fraction,
    )

    summaries: dict[str, Any] = {}
    for variant in V35_HUBNESS_VARIANTS:
        summaries[variant] = {}
        for side in ("reference", "target"):
            side_rows = [
                row for row in rows_by_variant[variant]
                if row["trajectory_side"] == side
            ]
            summaries[variant][side] = anchor_summary(
                side_rows,
                memory_ids=memory_ids,
                permutation_count=args.permutation_count,
            )
        summaries[variant]["key_geometry"] = _key_geometry(spaces[variant][0])

    comparisons: dict[str, Any] = {}
    for variant in V35_HUBNESS_VARIANTS[1:]:
        comparisons[variant] = {}
        for side in ("reference", "target"):
            raw_rows = [
                row for row in rows_by_variant["raw"]
                if row["trajectory_side"] == side
            ]
            candidate_rows = [
                row for row in rows_by_variant[variant]
                if row["trajectory_side"] == side
            ]
            comparisons[variant][side] = compare_variant_rows(
                raw_rows, candidate_rows
            )
            comparisons[variant][side]["hubness_delta_candidate_minus_raw"] = {
                field: (
                    summaries[variant][side]["hubness"][field]
                    - summaries["raw"][side]["hubness"][field]
                )
                for field in (
                    "top1_share",
                    "top2_combined_share",
                    "selected_memory_count",
                    "selection_gini_over_full_bank",
                )
            }

    pc1_incremental: dict[str, Any] = {}
    centered_name = "key_centroid_centered"
    pc1_name = "key_centroid_centered_remove_pc1"
    for side in ("reference", "target"):
        centered_rows = [
            row for row in rows_by_variant[centered_name]
            if row["trajectory_side"] == side
        ]
        pc1_rows = [
            row for row in rows_by_variant[pc1_name]
            if row["trajectory_side"] == side
        ]
        pc1_incremental[side] = compare_variant_rows(centered_rows, pc1_rows)
        pc1_incremental[side][
            "hubness_delta_pc1_minus_centered"
        ] = {
            field: (
                summaries[pc1_name][side]["hubness"][field]
                - summaries[centered_name][side]["hubness"][field]
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
        raw_keys=spaces["raw"][0],
        transform=transform,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output_dir / EVIDENCE_FILE
    evidence_count = write_jsonl(
        evidence_path,
        (
            _clean_evidence(row)
            for variant in V35_HUBNESS_VARIANTS
            for row in rows_by_variant[variant]
        ),
    )
    hub_text_path = args.output_dir / HUB_TEXT_FILE
    hub_text_count = write_jsonl(hub_text_path, hub_text_rows)
    transform_path = args.output_dir / TRANSFORM_FILE
    transform_tensors = {
        "key_centroid": transform["key_centroid"],
        "centered_key_pc1": transform["centered_key_pc1"],
        "centered_key_singular_values": transform[
            "centered_key_singular_values"
        ],
        "key_centroid_centered_dynamic_keys": spaces[
            "key_centroid_centered"
        ][0],
        "key_centroid_centered_remove_pc1_dynamic_keys": spaces[
            "key_centroid_centered_remove_pc1"
        ][0],
    }
    save_file(
        {name: value.contiguous().cpu() for name, value in transform_tensors.items()},
        str(transform_path),
        metadata={
            "schema_version": V35_HUBNESS_TRANSFORM_SCHEMA,
            "variant_order_sha256": canonical_json_sha256(V35_HUBNESS_VARIANTS),
        },
    )

    source_raw_reference = source_report["primary"]["all"]
    source_raw_target = source_report["secondary"]["target_first_gate"]["all"]
    implementation_paths = (
        "memgen/experience/v3_5_hubness.py",
        "scripts/audit_v3_5_dynamic_hubness.py",
    )
    implementation_hashes = {
        path: file_sha256(PROJECT_ROOT / path) for path in implementation_paths
    }
    report: dict[str, Any] = {
        "schema_version": V35_HUBNESS_REPORT_SCHEMA,
        "created_at": utc_now(),
        "status": "completed_diagnostic",
        "diagnostic_only": True,
        "formal_v3_5_qualification_changed": False,
        "reasoner_forward_or_generation_run": False,
        "side_kv_used": False,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "variant_selected": False,
        "primary_side": V35_HUBNESS_PRIMARY_SIDE,
        "memory_count": len(memory_ids),
        "query_count": len(query_metadata),
        "configuration": {
            "fixed_variants": list(V35_HUBNESS_VARIANTS),
            "transform_fit_source": "authenticated_dynamic_keys_only",
            "key_centroid_definition": "arithmetic_mean_of_unit_dynamic_keys",
            "centered_query_transform": "l2_normalize(query_minus_key_centroid)",
            "pc1_fit_source": "key_centroid_residuals_only",
            "pc1_count": 1,
            "pc1_sign_policy": "largest_absolute_loading_positive",
            "pc1_query_transform": (
                "remove_projection_on_key_fitted_pc1_after_key_centering"
            ),
            "compute_device": "cpu",
            "compute_dtype": "float64",
            "retrieval_method": "exact_cosine",
            "stable_tie_break": "memory_id_ascending",
            "top_n": args.top_n,
            "permutation_count": args.permutation_count,
            "permutation_policy": (
                "shuffle_query_to_own_memory_binding_preserve_variant_score_matrix"
            ),
            "hub_text_top_n_per_variant_side": HUB_TEXT_TOP_N,
        },
        "transform_diagnostics": {
            "key_centroid_norm": float(transform["key_centroid"].norm().item()),
            "pc1_explained_variance_fraction": transform[
                "pc1_explained_variance_fraction"
            ],
            "pc1_sign_pivot_index": transform["pc1_sign_pivot_index"],
            "key_centroid_sha256": tensor_sha256(transform["key_centroid"]),
            "centered_key_pc1_sha256": tensor_sha256(
                transform["centered_key_pc1"]
            ),
            "raw_key_to_centroid_direction_cosine": numeric_summary(
                (
                    spaces["raw"][0]
                    @ (
                        transform["key_centroid"]
                        / transform["key_centroid"].norm()
                    )
                ).tolist()
            ),
            "centered_key_pc1_loading": numeric_summary(
                transform["key_pc1_projection"].tolist()
            ),
            "centered_query_pc1_loading": numeric_summary(
                transform["query_pc1_projection"].tolist()
            ),
        },
        "inputs": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.verified_experiences),
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
            "transforms": {
                "path": transform_path.name,
                "sha256": file_sha256(transform_path),
                "tensor_names": sorted(transform_tensors),
            },
            "hub_key_text_audit": {
                "path": hub_text_path.name,
                "sha256": file_sha256(hub_text_path),
                "row_count": hub_text_count,
            },
        },
        "source_report_raw_metrics": {
            "reference": source_raw_reference,
            "target": source_raw_target,
        },
        "recomputed_raw_metric_delta": {
            "reference_minus_source_report": _metric_delta(
                summaries["raw"]["reference"]["all"], source_raw_reference
            ),
            "target_minus_source_report": _metric_delta(
                summaries["raw"]["target"]["all"], source_raw_target
            ),
        },
        "variants": summaries,
        "paired_comparisons_against_raw": comparisons,
        "pc1_incremental_comparison_against_centered": pc1_incremental,
        "top_hub_ids_by_variant_side": top_hub_ids,
        "requirements": {
            "source_alignment_report_authenticated": True,
            "exact_first_gate_query_sidecar_authenticated": True,
            "dynamic_key_bank_authenticated": True,
            "memory_and_source_text_join_authenticated": True,
            "raw_centered_and_centered_pc1_variants_pre_registered": True,
            "centroid_fitted_from_dynamic_keys_only": True,
            "pc1_fitted_from_centered_dynamic_keys_only": True,
            "exactly_one_pc_removed": True,
            "query_labels_not_used_to_fit_transforms": True,
            "variant_not_selected": True,
            "threshold_not_fitted": True,
            "reasoner_forward_or_generation_not_run": True,
            "side_kv_not_used": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
            "formal_v3_5_qualification_unchanged": True,
        },
        "interpretation_contract": {
            "centering_reduces_hubness_and_improves_source_rank": (
                "shared_embedding_component_obscures_usable_relative_alignment"
            ),
            "centering_and_pc1_do_not_help": (
                "dynamic_key_runtime_query_semantic_mismatch_remains"
            ),
            "pc1_only_incremental_gain": (
                "dominant_centered_key_direction_contributes_to_hubness"
            ),
            "strong_in_source_result_not_cross_problem_utility": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    report_path = args.output_dir / REPORT_FILE
    write_json(report_path, report)
    write_text(args.output_dir / MARKDOWN_FILE, _markdown(report))
    print(
        "[v3.5-hubness] "
        f"status={report['status']} queries={len(query_metadata)} "
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
