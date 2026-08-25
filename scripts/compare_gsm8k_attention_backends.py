#!/usr/bin/env python3
"""Compare two authenticated GSM8K attention-backend base runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl


CONDITIONS = (
    "native_transformers_generate",
    "explicit_live_kv_cache",
)
SUPPORTED_BACKENDS = ("eager", "sdpa", "flash_attention_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-name", choices=SUPPORTED_BACKENDS, required=True
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--candidate-name", choices=SUPPORTED_BACKENDS, required=True
    )
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reference_name == args.candidate_name:
        parser.error("reference and candidate backends must differ")
    return args


@dataclass(frozen=True)
class BackendArtifacts:
    """One authenticated base-parity run for a named attention backend."""

    name: str
    directory: Path
    summary: dict[str, Any]
    records: tuple[dict[str, Any], ...]

    @classmethod
    def load(cls, *, name: str, directory: Path) -> "BackendArtifacts":
        summary_path = directory / "base_parity_summary.json"
        results_path = directory / "results.jsonl"
        if not summary_path.is_file() or not results_path.is_file():
            raise ValueError(f"Incomplete {name} artifact directory: {directory}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        records = tuple(iter_jsonl(results_path))
        if summary.get("schema_version") != "gsm8k-base-generation-parity-report-v3":
            raise ValueError(f"Unexpected {name} summary schema")
        if summary.get("generation_contract", {}).get(
            "attention_implementation"
        ) != name:
            raise ValueError(f"{name} summary identifies a different backend")
        if summary.get("results", {}).get("sha256") != file_sha256(results_path):
            raise ValueError(f"{name} results hash mismatch")
        if len(records) != int(summary.get("sample_count", -1)):
            raise ValueError(f"{name} summary/result count mismatch")
        if any(
            record.get("schema_version")
            != "gsm8k-base-generation-parity-result-v3"
            or record.get("attention_implementation") != name
            for record in records
        ):
            raise ValueError(f"{name} records have inconsistent provenance")
        return cls(name=name, directory=directory, summary=summary, records=records)

    @property
    def records_by_id(self) -> dict[str, dict[str, Any]]:
        indexed = {str(record["sample_id"]): record for record in self.records}
        if len(indexed) != len(self.records):
            raise ValueError(f"{self.name} results contain duplicate sample IDs")
        return indexed


def shared_prefix(reference: list[int], candidate: list[int]) -> int:
    count = 0
    for left, right in zip(reference, candidate):
        if int(left) != int(right):
            break
        count += 1
    return count


def compare_condition(
    *,
    reference: BackendArtifacts,
    candidate: BackendArtifacts,
    condition: str,
) -> dict[str, Any]:
    reference_rows = reference.records_by_id
    candidate_rows = candidate.records_by_id
    sample_ids = sorted(reference_rows)
    mismatches: list[str] = []
    first_token_mismatches: list[str] = []
    shared_lengths: list[int] = []
    candidate_only_correct = 0
    reference_only_correct = 0
    for sample_id in sample_ids:
        reference_row = reference_rows[sample_id]["conditions"][condition]
        candidate_row = candidate_rows[sample_id]["conditions"][condition]
        reference_ids = reference_row["completion_token_ids"]
        candidate_ids = candidate_row["completion_token_ids"]
        shared = shared_prefix(reference_ids, candidate_ids)
        shared_lengths.append(shared)
        if shared != len(reference_ids) or shared != len(candidate_ids):
            mismatches.append(sample_id)
            if shared == 0:
                first_token_mismatches.append(sample_id)
        reference_correct = bool(reference_row["final_reward"])
        candidate_correct = bool(candidate_row["final_reward"])
        candidate_only_correct += candidate_correct and not reference_correct
        reference_only_correct += reference_correct and not candidate_correct
    return {
        "sample_count": len(sample_ids),
        "exact_token_match_count": len(sample_ids) - len(mismatches),
        "token_mismatch_count": len(mismatches),
        "token_mismatch_sample_ids": mismatches,
        "first_token_mismatch_count": len(first_token_mismatches),
        "first_token_mismatch_sample_ids": first_token_mismatches,
        "mean_shared_prefix_length": sum(shared_lengths) / len(shared_lengths),
        "candidate_correct_reference_wrong": candidate_only_correct,
        "reference_correct_candidate_wrong": reference_only_correct,
    }


def metric_deltas(
    *,
    reference: BackendArtifacts,
    candidate: BackendArtifacts,
    condition: str,
) -> dict[str, float]:
    reference_values = reference.summary["conditions"][condition]
    candidate_values = candidate.summary["conditions"][condition]
    fields = (
        "accuracy",
        "diagnostic_answer_accuracy",
        "format_accuracy",
        "mean_generation_length",
    )
    return {
        f"candidate_minus_reference_{field}": (
            float(candidate_values[field]) - float(reference_values[field])
        )
        for field in fields
    }


def validate_shared_contract(
    reference: BackendArtifacts, candidate: BackendArtifacts
) -> None:
    reference_ids = set(reference.records_by_id)
    candidate_ids = set(candidate.records_by_id)
    if reference_ids != candidate_ids:
        raise ValueError("Attention backend runs cover different samples")
    comparable_summary_fields = (
        "logical_split",
        "dataset_split",
        "sample_count",
        "prompt_contract",
        "inputs",
    )
    if any(
        reference.summary.get(field) != candidate.summary.get(field)
        for field in comparable_summary_fields
    ):
        raise ValueError("Attention backend runs do not share one input contract")
    reference_reasoner = dict(reference.summary.get("reasoner", {}))
    candidate_reasoner = dict(candidate.summary.get("reasoner", {}))
    reference_reasoner.pop("attention_implementation", None)
    candidate_reasoner.pop("attention_implementation", None)
    if reference_reasoner != candidate_reasoner:
        raise ValueError("Attention backend runs use different reasoners")
    reference_generation = dict(
        reference.summary.get("generation_contract", {})
    )
    candidate_generation = dict(
        candidate.summary.get("generation_contract", {})
    )
    reference_generation.pop("attention_implementation", None)
    candidate_generation.pop("attention_implementation", None)
    if reference_generation != candidate_generation:
        raise ValueError("Attention backend runs use different decoding contracts")
    if any(
        reference_generation.get(condition, {}).get("batch_size") != 1
        for condition in CONDITIONS
    ):
        raise ValueError("Attention backend diagnostic requires batch_size=1")
    for sample_id in reference_ids:
        reference_row = reference.records_by_id[sample_id]
        candidate_row = candidate.records_by_id[sample_id]
        identity_fields = (
            "logical_split",
            "dataset_split",
            "source_index",
            "question_sha256",
            "prompt_token_count",
            "prompt_token_ids_sha256",
        )
        if any(
            reference_row.get(field) != candidate_row.get(field)
            for field in identity_fields
        ):
            raise ValueError(f"Input identity differs for {sample_id}")


def main() -> None:
    args = parse_args()
    reference = BackendArtifacts.load(
        name=args.reference_name, directory=args.reference_dir
    )
    candidate = BackendArtifacts.load(
        name=args.candidate_name, directory=args.candidate_dir
    )
    validate_shared_contract(reference, candidate)
    comparisons = {
        condition: compare_condition(
            reference=reference,
            candidate=candidate,
            condition=condition,
        )
        for condition in CONDITIONS
    }
    output: dict[str, Any] = {
        "schema_version": "gsm8k-attention-backend-comparison-v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_memory_claim": False,
        "diagnostic_variable": "attention_implementation",
        "reference_backend": reference.name,
        "candidate_backend": candidate.name,
        "fixed_contract": {
            "batch_size": 1,
            "logical_split": reference.summary["logical_split"],
            "dataset_split": reference.summary["dataset_split"],
            "sample_count": reference.summary["sample_count"],
            "prompt_contract": reference.summary["prompt_contract"],
            "reasoner": {
                key: value
                for key, value in reference.summary["reasoner"].items()
                if key != "attention_implementation"
            },
        },
        "backends": {
            reference.name: {
                "within_backend_token_parity": reference.summary[
                    "exact_token_parity"
                ],
                "conditions": reference.summary["conditions"],
            },
            candidate.name: {
                "within_backend_token_parity": candidate.summary[
                    "exact_token_parity"
                ],
                "conditions": candidate.summary["conditions"],
            },
        },
        "candidate_minus_reference": {
            condition: metric_deltas(
                reference=reference,
                candidate=candidate,
                condition=condition,
            )
            for condition in CONDITIONS
        },
        "cross_backend_token_comparison": comparisons,
        "inputs": {
            "reference_summary_sha256": file_sha256(
                args.reference_dir / "base_parity_summary.json"
            ),
            "reference_results_sha256": file_sha256(
                args.reference_dir / "results.jsonl"
            ),
            "candidate_summary_sha256": file_sha256(
                args.candidate_dir / "base_parity_summary.json"
            ),
            "candidate_results_sha256": file_sha256(
                args.candidate_dir / "results.jsonl"
            ),
        },
    }
    output["comparison_sha256"] = canonical_json_sha256(output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[attention-backend] status=passed output={args.output}", flush=True)


if __name__ == "__main__":
    main()
