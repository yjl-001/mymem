#!/usr/bin/env python3
"""Pair verified GSM8K success/failure rollouts into contrastive experiences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import (
    build_verified_experiences,
    file_sha256,
    iter_jsonl,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--max-pairs-per-sample", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiences, report = build_verified_experiences(
        iter_jsonl(args.rollouts),
        max_pairs_per_sample=args.max_pairs_per_sample,
    )
    if not experiences:
        raise RuntimeError(
            "No verified success/failure pairs were found; increase rollout diversity "
            "or bank-source sample count before teacher construction."
        )
    write_jsonl(args.output, experiences)
    report.update(
        {
            "rollouts_path": str(args.rollouts.resolve()),
            "rollouts_sha256": file_sha256(args.rollouts),
            "experiences_path": str(args.output.resolve()),
            "experiences_sha256": file_sha256(args.output),
        }
    )
    report_path = args.report_output or args.output.with_name("experience_build_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[verified-experiences] {args.output} count={len(experiences)} "
        f"sha256={report['experiences_sha256']}"
    )


if __name__ == "__main__":
    main()
