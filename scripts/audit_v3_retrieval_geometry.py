#!/usr/bin/env python3
"""Audit V3 retrieval-key anisotropy and hubness without loading the model."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
)
from memgen.experience.v3_selector import (
    numeric_summary,
    selection_concentration,
)


GEOMETRY_AUDIT_SCHEMA = "experience-memory-v3-retrieval-geometry-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-key-manifest", type=Path, required=True)
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def safe_payload(record: MemoryRecord) -> dict[str, Any]:
    return {
        "memory_id": record.memory_id,
        "experience_type": record.experience_type,
        "when_facing": record.sanitized_fields.get("when_facing"),
        "prefer": record.sanitized_fields.get("prefer"),
        "avoid": record.sanitized_fields.get("avoid"),
        "payload_hash": record.payload_hash,
    }


def markdown_report(value: Mapping[str, Any]) -> str:
    geometry = value["geometry"]
    hubs = value["leave_one_out_hubness"]
    rows = [
        "# MemGen V3 retrieval-key geometry audit",
        "",
        f"- Status: `{value['status']}`",
        f"- Memory count: {geometry['memory_count']}",
        f"- Hidden width: {geometry['hidden_width']}",
        f"- Mean-key norm (anisotropy): {geometry['mean_key_vector_norm']}",
        f"- Pairwise cosine mean / median / p95: "
        f"{geometry['off_diagonal_cosine']['mean']} / "
        f"{geometry['off_diagonal_cosine']['median']} / "
        f"{geometry['off_diagonal_cosine']['p95']}",
        f"- Centered effective rank: {geometry['centered_effective_rank']}",
        f"- Centered participation ratio: "
        f"{geometry['centered_participation_ratio']}",
        f"- Leave-one-out top hub share: {hubs['top1_share']}",
        f"- Leave-one-out hubness Gini: {hubs['gini']}",
        "",
        "## Top leave-one-out hubs",
        "",
        "| Memory ID | Count | Share | When facing |",
        "|---|---:|---:|---|",
    ]
    total = int(hubs["selection_count"])
    payloads = {
        str(item["memory_id"]): item for item in value["hub_payloads"]
    }
    for item in hubs["top_by_frequency"][:10]:
        payload = payloads[str(item["memory_id"])]
        when_facing = str(payload.get("when_facing", "")).replace("|", "\\|")
        rows.append(
            f"| {item['memory_id']} | {item['count']} | "
            f"{item['count'] / total if total else 0.0} | {when_facing} |"
        )
    rows.extend([
        "",
        "This is an answer-blind key-bank audit. Hubness does not itself prove that a memory is harmful.",
        "",
    ])
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("Audit top-k must be positive")

    import torch

    from memgen.model.retrieval_keys import RetrievalKeyBankLoader

    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    if len(records) < 2:
        raise ValueError("Geometry audit requires at least two memories")
    bank = RetrievalKeyBankLoader(manifest_path=args.retrieval_key_manifest)
    record_by_id = {record.memory_id: record for record in records}
    memory_ids = [str(entry["memory_id"]) for entry in bank.entries]
    if list(record_by_id) != memory_ids:
        raise ValueError("Memory records and retrieval keys use different order")
    for entry in bank.entries:
        record = record_by_id[str(entry["memory_id"])]
        if entry.get("payload_hash") != record.payload_hash:
            raise ValueError("Retrieval key payload hash differs from MemoryRecord")

    embeddings = bank.embeddings.float()
    similarities = embeddings @ embeddings.transpose(0, 1)
    count = int(embeddings.shape[0])
    diagonal_mask = torch.eye(count, dtype=torch.bool)
    off_diagonal = similarities[~diagonal_mask].tolist()
    ranked = similarities.clone()
    ranked.fill_diagonal_(float("-inf"))
    nearest_scores, nearest_indices = ranked.max(dim=1)
    hub_memory_ids = [memory_ids[int(index)] for index in nearest_indices.tolist()]
    hubness = selection_concentration(
        hub_memory_ids,
        complete_memory_ids=memory_ids,
    )

    unordered = similarities.triu(diagonal=1)
    pair_count = count * (count - 1) // 2
    threshold_counts = {
        str(threshold): int((unordered >= threshold).sum().item())
        for threshold in (0.9, 0.95, 0.99, 0.999)
    }
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    variance_total = float(variance.sum().item())
    if variance_total <= 0.0 or not math.isfinite(variance_total):
        raise ValueError("Centered retrieval keys have no finite variance")
    probabilities = variance / variance.sum()
    nonzero = probabilities[probabilities > 0]
    effective_rank = float(
        torch.exp(-(nonzero * nonzero.log()).sum()).item()
    )
    participation_ratio = float(
        (variance.sum().square() / variance.square().sum()).item()
    )
    nearest_details = []
    for index, (score, neighbor_index) in enumerate(
        zip(nearest_scores.tolist(), nearest_indices.tolist())
    ):
        nearest_details.append({
            "memory_id": memory_ids[index],
            "nearest_memory_id": memory_ids[int(neighbor_index)],
            "cosine": float(score),
        })
    nearest_details.sort(
        key=lambda item: (-float(item["cosine"]), str(item["memory_id"]))
    )
    top_hub_ids = [
        str(item["memory_id"])
        for item in hubness["top_by_frequency"][:args.top_k]
    ]
    report = {
        "schema_version": GEOMETRY_AUDIT_SCHEMA,
        "created_at": utc_now(),
        "status": "passed",
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "implementation": {
            "files_sha256": {
                "memgen/experience/v3_selector.py": file_sha256(
                    PROJECT_ROOT / "memgen/experience/v3_selector.py"
                ),
                "memgen/model/retrieval_keys.py": file_sha256(
                    PROJECT_ROOT / "memgen/model/retrieval_keys.py"
                ),
                "scripts/audit_v3_retrieval_geometry.py": file_sha256(
                    PROJECT_ROOT / "scripts/audit_v3_retrieval_geometry.py"
                ),
            },
        },
        "inputs": {
            "retrieval_key_manifest_sha256": file_sha256(
                args.retrieval_key_manifest
            ),
            "memory_records_sha256": file_sha256(args.memory_records),
        },
        "geometry": {
            "memory_count": count,
            "hidden_width": int(embeddings.shape[1]),
            "mean_key_vector_norm": float(
                embeddings.mean(dim=0).norm().item()
            ),
            "off_diagonal_cosine": numeric_summary(off_diagonal),
            "unordered_pair_count": pair_count,
            "unordered_pair_threshold_counts": threshold_counts,
            "centered_effective_rank": effective_rank,
            "centered_participation_ratio": participation_ratio,
            "centered_top1_variance_fraction": float(
                probabilities[0].item()
            ),
            "centered_top10_variance_fraction": float(
                probabilities[:10].sum().item()
            ),
            "nearest_neighbor_cosine": numeric_summary(
                [float(value) for value in nearest_scores.tolist()]
            ),
        },
        "leave_one_out_hubness": hubness,
        "nearest_neighbor_pairs": nearest_details[:args.top_k],
        "hub_payloads": [
            safe_payload(record_by_id[memory_id]) for memory_id in top_hub_ids
        ],
        "requirements": {
            "key_manifest_authenticated": True,
            "tensor_artifact_authenticated": True,
            "memory_payload_hashes_aligned": True,
            "embeddings_are_finite_unit_norm": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
        },
    }
    report["report_sha256"] = canonical_json_sha256({
        key: value for key, value in report.items() if key != "created_at"
    })
    write_json_atomic(args.output, report)
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(
        f"[v3-key-audit] memories={count} "
        f"mean_cosine={report['geometry']['off_diagonal_cosine']['mean']:.6f} "
        f"top_hub_share={hubness['top1_share']:.4f} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
