#!/usr/bin/env python3
"""Validate the completed 30-record Phase 1 human review worksheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import file_sha256, iter_jsonl, summarize_human_review


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-sample-size", type=int, default=30)
    parser.add_argument("--required-agreement", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.audit_report.open(encoding="utf-8") as handle:
        audit_report = json.load(handle)
    expected_hash = audit_report.get("artifacts", {}).get("human_review_sha256")
    # The worksheet is expected to change when a human fills it in.  Preserve
    # the originally prepared hash for provenance rather than requiring equality.
    result = summarize_human_review(
        iter_jsonl(args.review),
        required_sample_size=args.required_sample_size,
        required_agreement=args.required_agreement,
    )
    result.update(
        {
            "audit_report": str(args.audit_report.resolve()),
            "audit_report_sha256": file_sha256(args.audit_report),
            "review_file": str(args.review.resolve()),
            "prepared_review_sha256": expected_hash,
            "completed_review_sha256": file_sha256(args.review),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[human-review] completed={result['completed_count']} "
        f"agreement={result['agreement']:.3f} passed={result['passed']}"
    )
    if not result["passed"]:
        raise RuntimeError("Phase 1 human-review acceptance criterion was not met")


if __name__ == "__main__":
    main()
