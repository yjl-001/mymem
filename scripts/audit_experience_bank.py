#!/usr/bin/env python3
"""Apply Phase 1 provenance and quality gates to teacher experience records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import (
    audit_teacher_record,
    file_sha256,
    iter_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--teacher-records", type=Path, required=True)
    parser.add_argument("--approved-output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--human-review-output", type=Path, required=True)
    parser.add_argument("--human-review-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiences = {
        str(item["experience_id"]): item for item in iter_jsonl(args.experiences)
    }
    teacher_records = list(iter_jsonl(args.teacher_records))
    approved: list[dict] = []
    rejected: list[dict] = []
    seen_ids: set[str] = set()
    reason_counts: dict[str, int] = {}

    for record in teacher_records:
        experience_id = str(record.get("experience_id", ""))
        if experience_id in seen_ids:
            reasons = ["duplicate_experience_id"]
        elif experience_id not in experiences:
            reasons = ["unknown_experience_id"]
        else:
            reasons = audit_teacher_record(record, experiences[experience_id])
        seen_ids.add(experience_id)
        if reasons:
            rejected_record = {**record, "quality_gate": {"approved": False, "reasons": reasons}}
            rejected.append(rejected_record)
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        else:
            approved.append({**record, "quality_gate": {"approved": True, "reasons": []}})

    write_jsonl(args.approved_output, approved)
    write_jsonl(args.rejected_output, rejected)

    review_pool = approved.copy()
    random.Random(args.seed).shuffle(review_pool)
    review_rows = []
    for record in review_pool[: min(args.human_review_size, len(review_pool))]:
        experience = experiences[record["experience_id"]]
        review_rows.append(
            {
                "experience_id": record["experience_id"],
                "target_episode_id": experience["target_episode_id"],
                "reference_episode_id": experience["reference_episode_id"],
                "context": experience["context"],
                "target_trajectory": experience["trajectory"],
                "reference_trajectory": experience["reference_trajectory"],
                "teacher_bank": record["bank"],
                "human_review": {
                    "target_supported": None,
                    "reference_supported": None,
                    "target_reference_distinct": None,
                    "factually_consistent": None,
                    "reviewer_notes": "",
                },
            }
        )
    write_jsonl(args.human_review_output, review_rows)

    missing_teacher_records = sorted(set(experiences) - seen_ids)
    report = {
        "schema_version": "phase1-experience-bank-audit-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experience_count": len(experiences),
        "teacher_record_count": len(teacher_records),
        "approved_count": len(approved),
        "rejected_count": len(rejected),
        "rejection_reason_counts": dict(sorted(reason_counts.items())),
        "missing_teacher_record_count": len(missing_teacher_records),
        "missing_teacher_experience_ids": missing_teacher_records,
        "automatic_quality_gate_passed": bool(approved) and not rejected,
        "human_review": {
            "required_sample_size": args.human_review_size,
            "prepared_sample_size": len(review_rows),
            "required_agreement": 0.9,
            "status": "pending_manual_review",
            "review_file": str(args.human_review_output.resolve()),
        },
        "artifacts": {
            "experiences_sha256": file_sha256(args.experiences),
            "teacher_records_sha256": file_sha256(args.teacher_records),
            "approved_sha256": file_sha256(args.approved_output),
            "rejected_sha256": file_sha256(args.rejected_output),
            "human_review_sha256": file_sha256(args.human_review_output),
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[experience-audit] approved={len(approved)} rejected={len(rejected)} "
        f"manual_review={len(review_rows)} report={args.report_output}"
    )
    if not approved:
        raise RuntimeError("No teacher records passed the automatic Phase 1 quality gate")


if __name__ == "__main__":
    main()
