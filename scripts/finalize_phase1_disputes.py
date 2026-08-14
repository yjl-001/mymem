#!/usr/bin/env python3
"""Merge human resolutions for only the disputed Phase 1 AI-review records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import file_sha256, iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ai-approved", type=Path, required=True)
    parser.add_argument("--ai-rejected", type=Path, required=True)
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--final-approved", type=Path, required=True)
    parser.add_argument("--final-rejected", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    approved = list(iter_jsonl(args.ai_approved))
    rejected = list(iter_jsonl(args.ai_rejected))
    disputes = list(iter_jsonl(args.human_review))
    incomplete_ids: list[str] = []
    human_approved = 0
    human_rejected = 0

    for dispute in disputes:
        experience_id = str(dispute.get("experience_id", ""))
        resolution = dispute.get("human_resolution")
        decision = resolution.get("decision") if isinstance(resolution, dict) else None
        if decision not in {"approve", "reject"}:
            incomplete_ids.append(experience_id)
            continue
        teacher_record = dispute.get("teacher_record")
        if not isinstance(teacher_record, dict):
            raise ValueError(f"Dispute {experience_id} is missing teacher_record")
        final_record: dict[str, Any] = {
            **teacher_record,
            "final_review_gate": {
                "route": "human_approved" if decision == "approve" else "human_rejected",
                "deterministic_audit": dispute.get(
                    "deterministic_audit", dispute.get("automatic_gate")
                ),
                "ai_review": dispute.get("ai_review"),
                "human_resolution": resolution,
            },
        }
        if decision == "approve":
            approved.append(final_record)
            human_approved += 1
        else:
            rejected.append(final_record)
            human_rejected += 1

    write_jsonl(args.final_approved, approved)
    write_jsonl(args.final_rejected, rejected)
    report = {
        "schema_version": "phase1-dispute-finalization-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ai_approved_count": len(approved) - human_approved,
        "ai_rejected_count": len(rejected) - human_rejected,
        "dispute_count": len(disputes),
        "completed_dispute_count": len(disputes) - len(incomplete_ids),
        "human_approved_count": human_approved,
        "human_rejected_count": human_rejected,
        "incomplete_dispute_ids": incomplete_ids,
        "final_approved_count": len(approved),
        "final_rejected_count": len(rejected),
        "passed": not incomplete_ids and bool(approved),
        "artifacts": {
            "human_review_sha256": file_sha256(args.human_review),
            "final_approved_sha256": file_sha256(args.final_approved),
            "final_rejected_sha256": file_sha256(args.final_rejected),
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[dispute-finalize] disputes={len(disputes)} "
        f"completed={len(disputes) - len(incomplete_ids)} "
        f"approved={len(approved)} rejected={len(rejected)} "
        f"passed={report['passed']}",
        flush=True,
    )
    if incomplete_ids:
        raise RuntimeError("Some disputed records still require human resolution")
    if not approved:
        raise RuntimeError("No records were approved after dispute resolution")


if __name__ == "__main__":
    main()
