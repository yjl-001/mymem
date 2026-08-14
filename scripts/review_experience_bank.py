#!/usr/bin/env python3
"""Independently review Phase 1 teacher records for a precision-first bank."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import (
    AI_REVIEW_CRITERIA_FIELDS,
    audit_teacher_record,
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    route_ai_review,
    split_audit_reasons,
    upgrade_verified_experience,
    write_jsonl,
)
from scripts.build_teacher_bank import TeacherClient


PROMPT_VERSION = "phase1-ai-review-v1-independent-evidence"
REVIEW_SCHEMA = "phase1-ai-review-record-v1"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--teacher-records", type=Path, required=True)
    parser.add_argument("--review-records-output", type=Path, required=True)
    parser.add_argument("--approved-output", type=Path, required=True)
    parser.add_argument("--rejected-output", type=Path, required=True)
    parser.add_argument("--deferred-output", type=Path, required=True)
    parser.add_argument("--quarantined-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_REVIEW_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--proxy-retries", type=int, default=20)
    parser.add_argument("--proxy-retry-initial-seconds", type=float, default=30.0)
    parser.add_argument("--proxy-retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def review_provenance(experience: dict[str, Any], teacher_record: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "experience_provenance_sha256": experience.get("provenance_sha256"),
            "teacher_prompt_version": teacher_record.get("prompt_version"),
            "teacher_model": teacher_record.get("teacher", {}).get("model"),
            "teacher_bank": teacher_record.get("bank"),
        }
    )


def reviewer_messages(
    experience: dict[str, Any],
    teacher_record: dict[str, Any],
) -> list[dict[str, str]]:
    teacher_bank = teacher_record.get("bank")
    if isinstance(teacher_bank, dict):
        # Hide the curator's self-assigned quality vote so the reviewer cannot
        # merely echo it. The substantive target/reference/evidence remains.
        teacher_bank = {
            key: value for key, value in teacher_bank.items() if key != "quality"
        }
    system = """You are the independent second-pass auditor for an experience bank.
The first-pass curator was a different, cheaper model. Judge its abstraction
only from the supplied raw trajectories and verifier records. Return JSON only.

Authority and scope:
- The deterministic verifier owns task success/failure. Do not override reward.
- Missing or malformed required boxed output is a real task failure.
- diagnostic_answer_correct is diagnostic evidence, not permission to change reward.
- If a format-only reference has a correct diagnostic answer, the curator must
  discuss output/boxing compliance and must not invent a reasoning error.
- For answer failures, accept only failure mechanisms visibly supported by the
  trajectory. Do not invent hidden intentions or cognitive causes.
- A successful reward alone does not prove every reasoning statement is sound;
  reject lucky, contradictory, or unsupported target abstractions.
- Bank text must be transferable and must not preserve instance-specific names,
  numbers, final answers, or equations.

Evaluate independently. The automatic gate result is intentionally hidden from
you to prevent anchoring. Use decision=approve only when every criterion is true.
Use reject when at least one criterion is definitely false. Use uncertain only
when the supplied evidence cannot resolve the issue. Confidence measures
confidence in your whole decision, not fluency. Keep evidence summaries concise
and factual."""
    payload = {
        "context": experience.get("context"),
        "target_trajectory": experience.get("trajectory"),
        "reference_trajectory": experience.get("reference_trajectory"),
        "experience_type": experience.get("experience_type"),
        "reference_failure_types": experience.get("reference_failure_types"),
        "target_verifier": experience.get("target_verifier"),
        "reference_verifier": experience.get("reference_verifier"),
        "teacher_bank": teacher_bank,
    }
    user = f"""Audit this record:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

Return exactly this JSON shape:
{{
  "decision": "approve|reject|uncertain",
  "confidence": 0.0,
  "criteria": {{
    "target_supported": true,
    "reference_supported": true,
    "target_reference_distinct": true,
    "factually_consistent": true,
    "failure_type_aligned": true,
    "transferable_without_instance_leakage": true
  }},
  "evidence": {{
    "target": "brief evidence-based observation",
    "reference": "brief evidence-based observation"
  }},
  "issues": [],
  "uncertainty_reason": ""
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_review_payload(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Reviewer returned empty content")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    payload = json.loads(cleaned)
    if payload.get("decision") not in {"approve", "reject", "uncertain"}:
        raise ValueError("Reviewer returned invalid decision")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise ValueError("Reviewer returned invalid confidence")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("Reviewer confidence is outside [0, 1]")
    criteria = payload.get("criteria")
    if not isinstance(criteria, dict) or any(
        not isinstance(criteria.get(field), bool) for field in AI_REVIEW_CRITERIA_FIELDS
    ):
        raise ValueError("Reviewer returned invalid criteria")
    evidence = payload.get("evidence")
    if not isinstance(evidence, dict) or any(
        not isinstance(evidence.get(field), str) or not evidence[field].strip()
        for field in ("target", "reference")
    ):
        raise ValueError("Reviewer returned invalid evidence")
    if not isinstance(payload.get("issues"), list):
        raise ValueError("Reviewer returned invalid issues")
    if not isinstance(payload.get("uncertainty_reason"), str):
        raise ValueError("Reviewer returned invalid uncertainty_reason")
    return payload


def _backup_and_filter_resume(
    path: Path,
    *,
    model: str,
    expected_provenance: dict[str, str],
) -> dict[str, dict[str, Any]]:
    compatible: dict[str, dict[str, Any]] = {}
    existing_count = 0
    if path.exists():
        for record in iter_jsonl(path):
            existing_count += 1
            experience_id = str(record.get("experience_id", ""))
            if (
                record.get("schema_version") == REVIEW_SCHEMA
                and record.get("prompt_version") == PROMPT_VERSION
                and record.get("reviewer", {}).get("model") == model
                and record.get("review_provenance_sha256")
                == expected_provenance.get(experience_id)
            ):
                compatible[experience_id] = record
    stale_count = existing_count - len(compatible)
    if stale_count:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(f"{path.name}.stale-{stamp}.bak")
        shutil.copy2(path, backup)
        print(f"[ai-review] backed up {stale_count} stale records to {backup}", flush=True)
    write_jsonl(path, compatible.values())
    return compatible


def deterministic_audit(record: dict[str, Any]) -> dict[str, Any]:
    """Read the current audit field while accepting pre-migration review records."""

    audit = record.get("deterministic_audit")
    if not isinstance(audit, dict):
        audit = record.get("automatic_gate")
    if not isinstance(audit, dict) or not isinstance(audit.get("reasons"), list):
        raise ValueError(
            f"Review record {record.get('experience_id')} has no deterministic audit"
        )
    integrity_reasons, semantic_warnings = split_audit_reasons(audit["reasons"])
    return {
        "integrity_passed": not integrity_reasons,
        "reasons": audit["reasons"],
        "integrity_reasons": integrity_reasons,
        "semantic_warnings": semantic_warnings,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1]")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before running AI review")

    experiences = {
        str(item["experience_id"]): upgrade_verified_experience(item)
        for item in iter_jsonl(args.experiences)
    }
    teacher_records: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(args.teacher_records):
        experience_id = str(record.get("experience_id", ""))
        if not experience_id or experience_id in teacher_records:
            raise ValueError(f"Missing or duplicate teacher experience_id: {experience_id!r}")
        if experience_id not in experiences:
            raise ValueError(f"Unknown teacher experience_id: {experience_id}")
        teacher_records[experience_id] = record

    expected_provenance = {
        experience_id: review_provenance(experiences[experience_id], record)
        for experience_id, record in teacher_records.items()
    }
    args.review_records_output.parent.mkdir(parents=True, exist_ok=True)
    completed = (
        _backup_and_filter_resume(
            args.review_records_output,
            model=args.model,
            expected_provenance=expected_provenance,
        )
        if args.resume
        else {}
    )
    mode = "a" if args.resume else "w"
    created_at = datetime.now(timezone.utc).isoformat()
    with TeacherClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        proxy_retries=args.proxy_retries,
        proxy_retry_initial_seconds=args.proxy_retry_initial_seconds,
        proxy_retry_max_seconds=args.proxy_retry_max_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        read_timeout_seconds=args.read_timeout_seconds,
        thinking=args.thinking,
    ) as client, args.review_records_output.open(mode, encoding="utf-8") as handle:
        for index, (experience_id, teacher_record) in enumerate(
            teacher_records.items(), start=1
        ):
            if experience_id in completed:
                print(f"[ai-review] skip completed {experience_id}", flush=True)
                continue
            experience = experiences[experience_id]
            automatic_reasons = audit_teacher_record(teacher_record, experience)
            integrity_reasons, semantic_warnings = split_audit_reasons(
                automatic_reasons
            )
            if integrity_reasons:
                review = None
                route = "quarantined"
            else:
                review = client.call(
                    reviewer_messages(experience, teacher_record),
                    response_parser=parse_review_payload,
                )
                route = route_ai_review(
                    automatic_reasons,
                    review,
                    confidence_threshold=args.confidence_threshold,
                )
            record = {
                "schema_version": REVIEW_SCHEMA,
                "prompt_version": PROMPT_VERSION,
                "created_at": created_at,
                "experience_id": experience_id,
                "reviewer": {"model": args.model, "base_url": args.base_url},
                "review_provenance_sha256": expected_provenance[experience_id],
                "deterministic_audit": {
                    "integrity_passed": not integrity_reasons,
                    "reasons": automatic_reasons,
                    "integrity_reasons": integrity_reasons,
                    "semantic_warnings": semantic_warnings,
                },
                "ai_review": review,
                "route": route,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            completed[experience_id] = record
            print(
                f"[ai-review] {index}/{len(teacher_records)} {experience_id} -> {route}",
                flush=True,
            )

        for experience_id in teacher_records:
            review_record = dict(completed[experience_id])
            audit = deterministic_audit(review_record)
            review_record.pop("automatic_gate", None)
            review_record["deterministic_audit"] = audit
            automatic_reasons = audit["reasons"]
            integrity_reasons = audit["integrity_reasons"]
            semantic_warnings = audit["semantic_warnings"]
            if integrity_reasons:
                initial_route = "quarantined"
            else:
                first_review = review_record.get("ai_review")
                if not isinstance(first_review, dict):
                    first_review = client.call(
                        reviewer_messages(
                            experiences[experience_id],
                            teacher_records[experience_id],
                        ),
                        response_parser=parse_review_payload,
                    )
                    review_record["ai_review"] = first_review
                    handle.write(
                        json.dumps(review_record, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
                    handle.flush()
                initial_route = route_ai_review(
                    automatic_reasons,
                    first_review,
                    confidence_threshold=args.confidence_threshold,
                )
            review_record["initial_route"] = initial_route
            review_record["integrity_reasons"] = integrity_reasons
            review_record["semantic_warnings"] = semantic_warnings
            review_record.pop("adjudication", None)
            review_record["route"] = initial_route
            completed[experience_id] = review_record

    ordered_reviews = []
    for experience_id in teacher_records:
        review_record = dict(completed[experience_id])
        audit = deterministic_audit(review_record)
        review_record.pop("automatic_gate", None)
        review_record["deterministic_audit"] = audit
        if audit["integrity_reasons"]:
            initial_route = "quarantined"
        else:
            initial_route = route_ai_review(
                audit["reasons"],
                review_record["ai_review"],
                confidence_threshold=args.confidence_threshold,
            )
        review_record["initial_route"] = initial_route
        integrity_reasons = audit["integrity_reasons"]
        semantic_warnings = audit["semantic_warnings"]
        review_record["integrity_reasons"] = integrity_reasons
        review_record["semantic_warnings"] = semantic_warnings
        review_record.pop("adjudication", None)
        review_record["route"] = initial_route
        review_record["routing_confidence_threshold"] = args.confidence_threshold
        review_record.pop("adjudication_confidence_threshold", None)
        completed[experience_id] = review_record
        ordered_reviews.append(review_record)
    write_jsonl(args.review_records_output, ordered_reviews)
    approved: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    for review_record in ordered_reviews:
        experience_id = review_record["experience_id"]
        teacher_record = teacher_records[experience_id]
        route = review_record["route"]
        gated_record = {**teacher_record, "ai_review_gate": review_record}
        if route == "ai_approved":
            approved.append(gated_record)
        elif route == "ai_rejected":
            rejected.append(gated_record)
        elif route == "deferred":
            deferred.append(gated_record)
        elif route == "quarantined":
            quarantined.append(gated_record)
        else:
            raise RuntimeError(f"Unexpected review route for {experience_id}: {route}")

    write_jsonl(args.approved_output, approved)
    write_jsonl(args.rejected_output, rejected)
    write_jsonl(args.deferred_output, deferred)
    write_jsonl(args.quarantined_output, quarantined)
    route_counts = Counter(item["route"] for item in ordered_reviews)
    initial_route_counts = Counter(item["initial_route"] for item in ordered_reviews)
    quarantine_reason_counts = Counter(
        reason for item in ordered_reviews for reason in item["integrity_reasons"]
    )
    semantic_warning_counts = Counter(
        reason for item in ordered_reviews for reason in item["semantic_warnings"]
    )
    deferred_reason_counts = Counter()
    deferred_criterion_false_counts = Counter()
    for item in ordered_reviews:
        if item["route"] != "deferred":
            continue
        first_review = item["ai_review"]
        if first_review["decision"] == "uncertain":
            deferred_reason_counts["reviewer_uncertain"] += 1
        if first_review["confidence"] < args.confidence_threshold:
            deferred_reason_counts["reviewer_low_confidence"] += 1
        for criterion, passed in first_review["criteria"].items():
            if not passed:
                deferred_criterion_false_counts[criterion] += 1

    review_decision_counts = Counter(
        item["ai_review"]["decision"]
        for item in ordered_reviews
        if isinstance(item.get("ai_review"), dict)
    )
    confidence_values_by_route: dict[str, list[float]] = {}
    for item in ordered_reviews:
        if not isinstance(item.get("ai_review"), dict):
            continue
        confidence_values_by_route.setdefault(item["route"], []).append(
            float(item["ai_review"]["confidence"])
        )
    confidence_by_route = {
        route: {
            "count": len(values),
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
        for route, values in sorted(confidence_values_by_route.items())
    }

    experience_type_route_counts: dict[str, Counter[str]] = {}
    for item in ordered_reviews:
        experience_type = str(
            experiences[item["experience_id"]].get(
                "experience_type", "unclassified_task_failure"
            )
        )
        counts = experience_type_route_counts.setdefault(experience_type, Counter())
        counts[item["route"]] += 1
    type_route_report = {
        experience_type: dict(sorted(counts.items()))
        for experience_type, counts in sorted(experience_type_route_counts.items())
    }
    approved_rate_by_experience_type = {
        experience_type: counts.get("ai_approved", 0) / sum(counts.values())
        for experience_type, counts in sorted(experience_type_route_counts.items())
    }
    report = {
        "schema_version": "phase1-ai-review-report-v4",
        "created_at": created_at,
        "teacher_record_count": len(teacher_records),
        "reviewed_count": len(ordered_reviews),
        "pro_reviewed_count": sum(
            isinstance(item.get("ai_review"), dict) for item in ordered_reviews
        ),
        "quarantined_before_pro_count": sum(
            item["route"] == "quarantined" and item.get("ai_review") is None
            for item in ordered_reviews
        ),
        "route_counts": dict(sorted(route_counts.items())),
        "initial_route_counts": dict(sorted(initial_route_counts.items())),
        "quarantine_reason_counts": dict(sorted(quarantine_reason_counts.items())),
        "semantic_warning_counts": dict(sorted(semantic_warning_counts.items())),
        "deferred_reason_counts": dict(sorted(deferred_reason_counts.items())),
        "deferred_criterion_false_counts": dict(
            sorted(deferred_criterion_false_counts.items())
        ),
        "review_decision_counts": dict(sorted(review_decision_counts.items())),
        "confidence_by_route": confidence_by_route,
        "experience_type_route_counts": type_route_report,
        "approved_rate_by_experience_type": approved_rate_by_experience_type,
        "confidence_threshold": args.confidence_threshold,
        "deferred_count": len(deferred),
        "selection_policy": (
            "Integrity/provenance/schema failures are quarantined outside quality "
            "judgment; only first-pass Pro approvals at or above the confidence "
            "threshold enter the bank; uncertain or low-confidence records are "
            "deferred without adjudication or human review."
        ),
        "artifacts": {
            "experiences_sha256": file_sha256(args.experiences),
            "teacher_records_sha256": file_sha256(args.teacher_records),
            "review_records_sha256": file_sha256(args.review_records_output),
            "approved_sha256": file_sha256(args.approved_output),
            "rejected_sha256": file_sha256(args.rejected_output),
            "deferred_sha256": file_sha256(args.deferred_output),
            "quarantined_sha256": file_sha256(args.quarantined_output),
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(
        f"[ai-review] approved={len(approved)} rejected={len(rejected)} "
        f"deferred={len(deferred)} quarantined={len(quarantined)} "
        f"report={args.report_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
