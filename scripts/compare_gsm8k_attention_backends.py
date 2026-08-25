#!/usr/bin/env python3
"""Compare authenticated eager and FlashAttention2 GSM8K base runs."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eager-dir", type=Path, required=True)
    parser.add_argument("--flash-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


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
    *, eager: BackendArtifacts, flash: BackendArtifacts, condition: str
) -> dict[str, Any]:
    eager_rows = eager.records_by_id
    flash_rows = flash.records_by_id
    sample_ids = sorted(eager_rows)
    mismatches: list[str] = []
    first_token_mismatches: list[str] = []
    shared_lengths: list[int] = []
    flash_only_correct = 0
    eager_only_correct = 0
    for sample_id in sample_ids:
        eager_row = eager_rows[sample_id]["conditions"][condition]
        flash_row = flash_rows[sample_id]["conditions"][condition]
        eager_ids = eager_row["completion_token_ids"]
        flash_ids = flash_row["completion_token_ids"]
        shared = shared_prefix(eager_ids, flash_ids)
        shared_lengths.append(shared)
        if shared != len(eager_ids) or shared != len(flash_ids):
            mismatches.append(sample_id)
            if shared == 0:
                first_token_mismatches.append(sample_id)
        eager_correct = bool(eager_row["final_reward"])
        flash_correct = bool(flash_row["final_reward"])
        flash_only_correct += flash_correct and not eager_correct
        eager_only_correct += eager_correct and not flash_correct
    return {
        "sample_count": len(sample_ids),
        "exact_token_match_count": len(sample_ids) - len(mismatches),
        "token_mismatch_count": len(mismatches),
        "token_mismatch_sample_ids": mismatches,
        "first_token_mismatch_count": len(first_token_mismatches),
        "first_token_mismatch_sample_ids": first_token_mismatches,
        "mean_shared_prefix_length": sum(shared_lengths) / len(shared_lengths),
        "flash_correct_eager_wrong": flash_only_correct,
        "eager_correct_flash_wrong": eager_only_correct,
    }


def metric_deltas(
    *, eager: BackendArtifacts, flash: BackendArtifacts, condition: str
) -> dict[str, float]:
    eager_values = eager.summary["conditions"][condition]
    flash_values = flash.summary["conditions"][condition]
    fields = (
        "accuracy",
        "diagnostic_answer_accuracy",
        "format_accuracy",
        "mean_generation_length",
    )
    return {
        f"flash_minus_eager_{field}": (
            float(flash_values[field]) - float(eager_values[field])
        )
        for field in fields
    }


def validate_shared_contract(
    eager: BackendArtifacts, flash: BackendArtifacts
) -> None:
    eager_ids = set(eager.records_by_id)
    flash_ids = set(flash.records_by_id)
    if eager_ids != flash_ids:
        raise ValueError("Attention backend runs cover different samples")
    comparable_summary_fields = (
        "logical_split",
        "dataset_split",
        "sample_count",
        "prompt_contract",
        "inputs",
    )
    if any(
        eager.summary.get(field) != flash.summary.get(field)
        for field in comparable_summary_fields
    ):
        raise ValueError("Attention backend runs do not share one input contract")
    eager_reasoner = dict(eager.summary.get("reasoner", {}))
    flash_reasoner = dict(flash.summary.get("reasoner", {}))
    eager_reasoner.pop("attention_implementation", None)
    flash_reasoner.pop("attention_implementation", None)
    if eager_reasoner != flash_reasoner:
        raise ValueError("Attention backend runs use different reasoners")
    eager_generation = dict(eager.summary.get("generation_contract", {}))
    flash_generation = dict(flash.summary.get("generation_contract", {}))
    eager_generation.pop("attention_implementation", None)
    flash_generation.pop("attention_implementation", None)
    if eager_generation != flash_generation:
        raise ValueError("Attention backend runs use different decoding contracts")
    if any(
        eager_generation.get(condition, {}).get("batch_size") != 1
        for condition in CONDITIONS
    ):
        raise ValueError("Attention backend diagnostic requires batch_size=1")
    for sample_id in eager_ids:
        eager_row = eager.records_by_id[sample_id]
        flash_row = flash.records_by_id[sample_id]
        identity_fields = (
            "logical_split",
            "dataset_split",
            "source_index",
            "question_sha256",
            "prompt_token_count",
            "prompt_token_ids_sha256",
        )
        if any(eager_row.get(field) != flash_row.get(field) for field in identity_fields):
            raise ValueError(f"Input identity differs for {sample_id}")


def main() -> None:
    args = parse_args()
    eager = BackendArtifacts.load(name="eager", directory=args.eager_dir)
    flash = BackendArtifacts.load(
        name="flash_attention_2", directory=args.flash_dir
    )
    validate_shared_contract(eager, flash)
    comparisons = {
        condition: compare_condition(
            eager=eager, flash=flash, condition=condition
        )
        for condition in CONDITIONS
    }
    output: dict[str, Any] = {
        "schema_version": "gsm8k-attention-backend-comparison-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "formal_memory_claim": False,
        "diagnostic_variable": "attention_implementation",
        "fixed_contract": {
            "batch_size": 1,
            "logical_split": eager.summary["logical_split"],
            "dataset_split": eager.summary["dataset_split"],
            "sample_count": eager.summary["sample_count"],
            "prompt_contract": eager.summary["prompt_contract"],
            "reasoner": {
                key: value
                for key, value in eager.summary["reasoner"].items()
                if key != "attention_implementation"
            },
        },
        "backends": {
            "eager": {
                "within_backend_token_parity": eager.summary["exact_token_parity"],
                "conditions": eager.summary["conditions"],
            },
            "flash_attention_2": {
                "within_backend_token_parity": flash.summary["exact_token_parity"],
                "conditions": flash.summary["conditions"],
            },
        },
        "flash_minus_eager": {
            condition: metric_deltas(
                eager=eager, flash=flash, condition=condition
            )
            for condition in CONDITIONS
        },
        "cross_backend_token_comparison": comparisons,
        "inputs": {
            "eager_summary_sha256": file_sha256(
                args.eager_dir / "base_parity_summary.json"
            ),
            "eager_results_sha256": file_sha256(args.eager_dir / "results.jsonl"),
            "flash_summary_sha256": file_sha256(
                args.flash_dir / "base_parity_summary.json"
            ),
            "flash_results_sha256": file_sha256(args.flash_dir / "results.jsonl"),
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
