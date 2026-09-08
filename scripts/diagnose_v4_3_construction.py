#!/usr/bin/env python3
"""Read-only construction diagnostics; no model loading or source mutation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.v4_3_artifacts import load_construction, read_json
from memgen.experience.v4_3_bank import authenticate


def construction_diagnostics(bank):
    rows = bank["candidate_bank_records.jsonl"]
    return {
        "status": bank["construction_report.json"]["status"],
        "qualified_tier_counts": {tier: len(bank[f"{tier}_bank_records.jsonl"]) for tier in ("primary", "conditional")},
        "qualification_failure_counts": dict(sorted(Counter(f for r in rows for f in r["qualification"]["failures"]).items())),
        "banks": [{"source_v42_bank_id": r["source_v42_bank_id"], "quality_tier": r["quality_tier"],
                   "failures": r["qualification"]["failures"],
                   "assembled_card_issues": r["leakage_audit"]["assembled_card_issues"],
                   "fields": {field: {"support_count": s["support_count"], "failure": s["failure"],
                       "generated_clause_issues": s.get("generated_clause_issues", []),
                       "candidate_issue_counts": dict(sorted(Counter(issue for c in s["candidate_audit"] for issue in c["leakage_issues"]).items()))}
                              for field, s in r["clause_support"].items()}} for r in rows],
        "source_text_included": False, "thresholds_modified": False,
    }


def require_qualified(bank, *, both_tiers=False):
    counts = {tier: len(bank[f"{tier}_bank_records.jsonl"]) for tier in ("primary", "conditional")}
    if (both_tiers and not all(counts.values())) or not any(counts.values()):
        requirement = "a qualified Bank in both tiers" if both_tiers else "at least one qualified Bank"
        raise ValueError(f"Construction is not ready: {counts}; requires {requirement}. "
                         "Inspect qualification failures in clause_support_report.json and leakage_audit_report.json. "
                         "No model was loaded; source and construction artifacts are preserved. Do not relax thresholds to force a run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", type=Path, required=True)
    parser.add_argument("--reasoner-manifest", type=Path)
    parser.add_argument("--require-auditable", action="store_true")
    args = parser.parse_args()
    bank = load_construction(args.bank_dir)
    report = construction_diagnostics(bank)
    if args.reasoner_manifest is not None:
        old = read_json(args.reasoner_manifest)
        authenticate(old, "manifest_sha256", "source reasoner")
        report["source_reasoner"] = old["reasoner"]
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.require_auditable:
        require_qualified(bank, both_tiers=True)


if __name__ == "__main__":
    main()
