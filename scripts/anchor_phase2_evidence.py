#!/usr/bin/env python3
"""Ask an independent Pro reviewer to anchor Phase 2 vector evidence spans.

This is a post-Phase-1 audit.  It does not alter Phase 1 routing or teacher
records: it only creates exact, provenance-bound quotations used by the Phase 2
compiler instead of its earlier last-delimiter heuristic.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl, write_jsonl
from memgen.experience.phase2 import (
    PHASE2_EVIDENCE_ANCHOR_SCHEMA,
    PHASE2_MECHANISM_CLUSTERS,
    approved_experiences,
    validate_evidence_anchor,
)
from scripts.build_teacher_bank import TeacherClient


PROMPT_VERSION = "phase2-evidence-anchor-v2-exact-span"
LEGACY_REUSABLE_PROMPT_VERSIONS = ("phase2-evidence-anchor-v1-exact-quote",)
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-bank", type=Path, required=True)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_REVIEW_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--proxy-retries", type=int, default=20)
    parser.add_argument("--proxy-retry-initial-seconds", type=float, default=30.0)
    parser.add_argument("--proxy-retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def anchor_provenance(
    experience: Mapping[str, Any], bank_record: Mapping[str, Any], *, prompt_version: str
) -> str:
    return canonical_json_sha256(
        {
            "experience_provenance_sha256": experience.get("provenance_sha256"),
            "approved_bank_provenance_sha256": bank_record.get("provenance_sha256"),
            "approved_review": bank_record.get("ai_review_gate", {}).get("ai_review"),
            "bank": bank_record.get("bank"),
            "prompt_version": prompt_version,
        }
    )


def anchor_messages(experience: Mapping[str, Any], bank_record: Mapping[str, Any]) -> list[dict[str, str]]:
    system = """You are an independent evidence annotator for a steering-vector
experiment. Return JSON only. You are not judging whether the bank is good:
this pair already passed its independent Phase 1 review.

Your task is to identify one exact target quote and one exact reference quote
that end at a reasoning delimiter and are suitable hidden-state evidence.

Rules:
- Copy each quote character-for-character from its supplied trajectory. It must
  occur exactly once. It may be a short equation, a LaTex display block, or a
  sentence; the compiler will use the first online delimiter after the quote.
  Do not add ellipses, Markdown fences, explanation, or escaped replacements.
- Select an execution or verification step which materially bears on the task
  outcome. Prefer a calculation, constraint application, unit conversion,
  counting/rounding decision, temporal relation, or explicit check.
- Do not select generic introductions, generic conclusions, or the final
  `\\boxed{}` answer formatting. The target quote should illustrate the
  successful decision; the reference quote should illustrate the competing
  failed decision or the last visibly unsupported step before the failure.
- If no pair of exact, outcome-relevant, delimiter-terminated quotes is safely
  available, return decision="exclude". Exclusion is correct when uncertain.
- mechanism_cluster classifies the reference failure mechanism, not merely the
  verifier label. Use exactly one allowed value.
"""
    payload = {
        "context": experience.get("context"),
        "experience_type": experience.get("experience_type"),
        "reference_failure_types": experience.get("reference_failure_types"),
        "target_verifier": experience.get("target_verifier"),
        "reference_verifier": experience.get("reference_verifier"),
        "target_trajectory": experience.get("trajectory"),
        "reference_trajectory": experience.get("reference_trajectory"),
        "previously_approved_bank": bank_record.get("bank"),
    }
    user = f"""Anchor this verified contrast:
{json.dumps(payload, ensure_ascii=False, sort_keys=True)}

Return exactly this JSON shape:
{{
  "decision": "anchor|exclude",
  "mechanism_cluster": "arithmetic_or_numeric|unit_or_conversion|counting_or_discreteness|temporal_or_sequence|relation_or_constraint|other_task_reasoning",
  "target_anchor": {{"quote": "exact copied target evidence span"}},
  "reference_anchor": {{"quote": "exact copied reference evidence span"}},
  "rationale": "brief evidence-grounded reason",
  "confidence": 0.0
}}

For decision="exclude", set both quotes to "" and explain why."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_anchor_payload(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    payload = json.loads(cleaned)
    decision = payload.get("decision")
    if decision not in {"anchor", "exclude"}:
        raise ValueError("Anchor reviewer must choose decision=anchor|exclude")
    cluster = payload.get("mechanism_cluster")
    if cluster not in PHASE2_MECHANISM_CLUSTERS:
        raise ValueError("Anchor reviewer returned an invalid mechanism_cluster")
    confidence = payload.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
        raise ValueError("Anchor reviewer returned invalid confidence")
    if not isinstance(payload.get("rationale"), str):
        raise ValueError("Anchor reviewer returned missing rationale")
    for side in ("target", "reference"):
        anchor = payload.get(f"{side}_anchor")
        if not isinstance(anchor, dict) or not isinstance(anchor.get("quote"), str):
            raise ValueError(f"Anchor reviewer returned invalid {side}_anchor")
    return payload


def compatible_resume(
    path: Path, expected: Mapping[str, set[str]], model: str
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    existing_count = 0
    if path.exists():
        for record in iter_jsonl(path):
            existing_count += 1
            experience_id = str(record.get("experience_id", ""))
            if (
                record.get("schema_version") == PHASE2_EVIDENCE_ANCHOR_SCHEMA
                and record.get("prompt_version")
                in {PROMPT_VERSION, *LEGACY_REUSABLE_PROMPT_VERSIONS}
                and record.get("reviewer", {}).get("model") == model
                and record.get("anchor_provenance_sha256") in expected.get(experience_id, set())
            ):
                completed[experience_id] = record
    stale = existing_count - len(completed)
    if stale:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = path.with_name(f"{path.name}.stale-{stamp}.bak")
        shutil.copy2(path, backup)
        print(f"[phase2-anchor] backed up {stale} stale records to {backup}", flush=True)
    write_jsonl(path, completed.values())
    return completed


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before evidence anchoring")
    bank = list(iter_jsonl(args.approved_bank))
    experience_rows = list(iter_jsonl(args.experiences))
    selected, _ = approved_experiences(
        bank, experience_rows, allowed_experience_types=["answer_correctness"]
    )
    if args.limit:
        selected = selected[: args.limit]
    bank_by_id = {str(record["experience_id"]): record for record in bank}
    expected = {
        str(experience["experience_id"]): {
            anchor_provenance(
                experience,
                bank_by_id[str(experience["experience_id"])],
                prompt_version=prompt_version,
            )
            for prompt_version in (PROMPT_VERSION, *LEGACY_REUSABLE_PROMPT_VERSIONS)
        }
        for experience in selected
    }
    current_provenance = {
        str(experience["experience_id"]): anchor_provenance(
            experience,
            bank_by_id[str(experience["experience_id"])],
            prompt_version=PROMPT_VERSION,
        )
        for experience in selected
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = compatible_resume(args.output, expected, args.model) if args.resume else {}
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
    ) as client, args.output.open(mode, encoding="utf-8") as handle:
        for index, experience in enumerate(selected, start=1):
            experience_id = str(experience["experience_id"])
            if experience_id in completed:
                print(f"[phase2-anchor] skip completed {experience_id}", flush=True)
                continue
            review = client.call(
                anchor_messages(experience, bank_by_id[experience_id]),
                response_parser=parse_anchor_payload,
                request_label="phase2-anchor",
                expose_parser_error=True,
            )
            reasons = validate_evidence_anchor(review, experience)
            route = "anchored" if not reasons else "excluded"
            record = {
                "schema_version": PHASE2_EVIDENCE_ANCHOR_SCHEMA,
                "prompt_version": PROMPT_VERSION,
                "created_at": created_at,
                "reviewer": {"model": args.model, "base_url": args.base_url},
                "experience_id": experience_id,
                "experience_provenance_sha256": experience["provenance_sha256"],
                "anchor_provenance_sha256": current_provenance[experience_id],
                "route": route,
                "validation_reasons": reasons,
                "anchor_review": review,
            }
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            print(f"[phase2-anchor] {index}/{len(selected)} {experience_id} -> {route}", flush=True)

    selected_by_id = {str(experience["experience_id"]): experience for experience in selected}
    # Revalidate completed records on every run.  This makes deterministic
    # validator upgrades reusable without another expensive Pro API pass.
    records: list[dict[str, Any]] = []
    for record in iter_jsonl(args.output):
        refreshed = dict(record)
        experience = selected_by_id.get(str(record.get("experience_id", "")))
        if experience is None:
            raise ValueError(f"Anchor output references unknown experience {record.get('experience_id')!r}")
        reasons = validate_evidence_anchor(record.get("anchor_review", {}), experience)
        refreshed["validation_reasons"] = reasons
        refreshed["route"] = "anchored" if not reasons else "excluded"
        records.append(refreshed)
    write_jsonl(args.output, records)
    route_counts = Counter(str(record.get("route")) for record in records)
    mechanism_counts = Counter(
        str(record.get("anchor_review", {}).get("mechanism_cluster"))
        for record in records
        if record.get("route") == "anchored"
    )
    report = {
        "schema_version": "phase2-evidence-anchor-report-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_answer_correctness_count": len(selected),
        "completed_count": len(records),
        "route_counts": dict(sorted(route_counts.items())),
        "anchored_mechanism_cluster_counts": dict(sorted(mechanism_counts.items())),
        "artifacts": {
            "approved_bank_sha256": file_sha256(args.approved_bank),
            "verified_experiences_sha256": file_sha256(args.experiences),
            "anchors_sha256": file_sha256(args.output),
        },
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[phase2-anchor] report={args.report_output} anchored={route_counts['anchored']}", flush=True)


if __name__ == "__main__":
    main()
