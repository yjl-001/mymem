#!/usr/bin/env python3
"""Audit, synthesize, and independently review the small V4.2 repair bank.

The default ``preflight`` stage authenticates every upstream artifact, rejoins
the raw GSM8K construction evidence, applies the explicit semantic policy, and
writes the exact request plan.  It neither reads an API key nor creates a
network client.  The paid stage is reachable only with both ``--stage paid``
and ``--approve-paid-stage``; every accepted response is append-checkpointed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl
from memgen.experience.v4_2_bank import (
    V42ConstructionProfile,
    V42LocalClusterCandidate,
    V42LocalRepairAtom,
    V42ShortlistProfile,
    validate_v4_2_cluster_payload,
)
from memgen.experience.v4_2_semantic_bank import (
    V4_2_COMBINED_BATCH_SCHEMA,
    V4_2_COMBINED_PROMPT_VERSION,
    V4_2_COMBINED_RECORD_SCHEMA,
    V4_2_EVIDENCE_PACKET_SCHEMA,
    V4_2_PAID_PLAN_SCHEMA,
    V4_2_PAID_PREFLIGHT_SCHEMA,
    V4_2_REVIEW_BATCH_SCHEMA,
    V4_2_REVIEW_PROMPT_VERSION,
    V4_2_REVIEW_RECORD_SCHEMA,
    V4_2_SEMANTIC_POLICY_SCHEMA,
    V42CombinedSynthesis,
    V42EvidenceJudgment,
    V42SemanticConstructionProfile,
    build_v4_2_semantic_bank_manifest,
    build_v4_2_semantic_bank_record,
    parse_v4_2_combined_batch,
    parse_v4_2_review_batch,
)
from memgen.experience.v4_bank import (
    V4_PROCESS_CARD_SCHEMA,
    V4CardReview,
    V4RepairSignature,
)
from scripts.build_teacher_bank import TeacherClient, TeacherInvalidResponseError
from scripts.build_v4_1_repair_bank import load_authenticated_signatures
from scripts.build_v4_2_local_clusters import EMBEDDING_VIEW_NAMES
from scripts.build_v4_repair_bank import (
    _parse_json_object,
    _validate_split_manifest,
    attach_official_solutions,
    load_v4_experiences,
)
from scripts.select_v4_2_bank_candidates import (
    SELECTED_CANDIDATE_SCHEMA,
    _profile_record as shortlist_profile_record,
    load_authenticated_local_construction,
    validate_completed_output,
)


SEMANTIC_PROFILE_RECORD_SCHEMA = "memgen-v4.2-semantic-profile-record-v1"
POLICY_EXCLUSION_RECORD_SCHEMA = "memgen-v4.2-policy-exclusion-record-v1"
SEMANTIC_REJECTION_RECORD_SCHEMA = "memgen-v4.2-semantic-rejection-record-v1"
PAID_REPORT_SCHEMA = "memgen-v4.2-paid-stage-report-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--source-signatures", type=Path, required=True)
    parser.add_argument("--source-construction-profile", type=Path, required=True)
    parser.add_argument("--local-construction-dir", type=Path, required=True)
    parser.add_argument("--shortlist-dir", type=Path, required=True)
    parser.add_argument("--semantic-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument("--stage", choices=("preflight", "paid"), default="preflight")
    parser.add_argument(
        "--approve-paid-stage",
        action="store_true",
        help="Required with --stage paid before the credential may be read.",
    )
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--synthesis-max-tokens", type=int, default=9000)
    parser.add_argument("--review-max-tokens", type=int, default=7000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--proxy-retries", type=int, default=20)
    parser.add_argument("--proxy-retry-initial-seconds", type=float, default=30.0)
    parser.add_argument("--proxy-retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=240.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _logical_hash(value: Mapping[str, Any], field: str) -> str:
    return canonical_json_sha256({key: item for key, item in value.items() if key != field})


def _implementation_state() -> dict[str, str]:
    paths = (
        "memgen/experience/v4_2_semantic_bank.py",
        "scripts/build_v4_2_semantic_bank.py",
        "scripts/build_teacher_bank.py",
    )
    return {relative: file_sha256(PROJECT_ROOT / relative) for relative in paths}


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.deepseek.com"
        or parsed.port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path.rstrip("/") not in {"", "/v1"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "--base-url must be the credential-free official DeepSeek HTTPS endpoint"
        )
    return value.rstrip("/")


def validate_cli(args: argparse.Namespace) -> V42SemanticConstructionProfile:
    profile = V42SemanticConstructionProfile()
    if args.model != profile.teacher_model:
        raise ValueError(f"V4.2 semantic construction requires {profile.teacher_model}")
    if args.api_key_env != "DEEPSEEK_API_KEY":
        raise ValueError("V4.2 semantic construction reads only DEEPSEEK_API_KEY")
    if args.temperature != profile.temperature or args.thinking != profile.thinking:
        raise ValueError("V4.2 semantic construction requires temperature zero and disabled thinking")
    _validate_base_url(args.base_url)
    for name in (
        "synthesis_max_tokens",
        "review_max_tokens",
        "retries",
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "proxy_retry_initial_seconds",
        "proxy_retry_max_seconds",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.proxy_retries < 0:
        raise ValueError("--proxy-retries must be non-negative")
    if args.stage == "preflight" and args.approve_paid_stage:
        raise ValueError("--approve-paid-stage is valid only with --stage paid")
    return profile


def load_semantic_policy(
    path: Path,
    *,
    shortlist_profile_sha256: str,
    shortlist_manifest_sha256: str,
    shortlist_report_sha256: str,
    selected_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, str]], str]:
    policy = _load_json(path)
    if policy.get("schema_version") != V4_2_SEMANTIC_POLICY_SCHEMA:
        raise ValueError("Unexpected V4.2 semantic-policy schema")
    if policy.get("benchmark") != "openai/gsm8k":
        raise ValueError("V4.2 semantic policy benchmark mismatch")
    bindings = {
        "source_shortlist_profile_sha256": shortlist_profile_sha256,
        "source_shortlist_manifest_sha256": shortlist_manifest_sha256,
        "source_shortlist_report_sha256": shortlist_report_sha256,
    }
    for field, expected in bindings.items():
        if policy.get(field) != expected:
            raise ValueError(f"V4.2 semantic policy {field} mismatch")

    selected_by_id = {
        str(record["candidate"]["candidate_id"]): record for record in selected_records
    }
    candidate_exclusions: dict[str, str] = {}
    for raw in policy.get("candidate_exclusions", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("V4.2 candidate exclusion must be an object")
        candidate_id = str(raw.get("candidate_id", ""))
        reason = str(raw.get("reason", "")).strip()
        if candidate_id not in selected_by_id or candidate_id in candidate_exclusions or not reason:
            raise ValueError("V4.2 candidate exclusion is unknown, duplicated, or missing a reason")
        candidate_exclusions[candidate_id] = reason

    evidence_exclusions: dict[str, dict[str, str]] = {}
    for raw in policy.get("representative_evidence_exclusions", ()):
        if not isinstance(raw, Mapping):
            raise ValueError("V4.2 evidence exclusion must be an object")
        candidate_id = str(raw.get("candidate_id", ""))
        evidence_id = str(raw.get("evidence_id", ""))
        reason = str(raw.get("reason", "")).strip()
        record = selected_by_id.get(candidate_id)
        if record is None or not reason:
            raise ValueError("V4.2 evidence exclusion is unknown or missing a reason")
        provenance = {
            str(item.get("evidence_id", "")): str(item.get("experience_id", ""))
            for item in record.get("representative_provenance", ())
            if isinstance(item, Mapping)
        }
        experience_id = provenance.get(evidence_id)
        if not experience_id:
            raise ValueError("V4.2 policy evidence ID does not resolve in its shortlist record")
        by_candidate = evidence_exclusions.setdefault(candidate_id, {})
        if experience_id in by_candidate:
            raise ValueError("V4.2 semantic policy repeats an evidence exclusion")
        by_candidate[experience_id] = reason
    overlap = set(candidate_exclusions) & set(evidence_exclusions)
    if overlap:
        raise ValueError(
            "V4.2 semantic policy cannot exclude both a candidate and its evidence"
        )
    return policy, candidate_exclusions, evidence_exclusions, canonical_json_sha256(policy)


def _weighted_similarity(
    left: int,
    right: int,
    *,
    embeddings: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> float:
    return sum(
        weights[name] * float(np.dot(embeddings[name][left], embeddings[name][right]))
        for name in EMBEDDING_VIEW_NAMES
    )


def select_evidence_ids(
    candidate: V42LocalClusterCandidate,
    *,
    atom_index: Mapping[str, int],
    atoms: Mapping[str, V42LocalRepairAtom],
    embeddings: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    excluded_experience_ids: set[str],
    maximum_evidence: int = 8,
) -> tuple[str, ...]:
    """Select all small-cluster evidence, else five diverse plus three medoid-near."""

    eligible = [
        item
        for item in candidate.member_experience_ids
        if item not in excluded_experience_ids
    ]
    by_sample: dict[str, str] = {}
    for experience_id in eligible:
        sample_id = atoms[experience_id].sample_id
        by_sample.setdefault(sample_id, experience_id)
    eligible = list(by_sample.values())
    if len(eligible) <= maximum_evidence:
        return tuple(sorted(eligible))

    selected = [
        item
        for item in candidate.representative_experience_ids
        if item in eligible
    ]
    selected = list(dict.fromkeys(selected))
    remaining = [item for item in eligible if item not in selected]

    # Replace a policy-excluded representative with a farthest-first member so
    # the first five still cover the cluster rather than collapsing to medoids.
    while len(selected) < 5:
        if not remaining:
            raise ValueError("V4.2 candidate cannot restore five diverse evidence items")
        if selected:
            chosen = min(
                remaining,
                key=lambda item: (
                    max(
                        _weighted_similarity(
                            atom_index[item],
                            atom_index[kept],
                            embeddings=embeddings,
                            weights=weights,
                        )
                        for kept in selected
                    ),
                    item,
                ),
            )
        else:
            chosen = min(remaining)
        selected.append(chosen)
        remaining.remove(chosen)

    member_indices = [atom_index[item] for item in eligible]
    centroids: dict[str, np.ndarray] = {}
    for name in EMBEDDING_VIEW_NAMES:
        centroid = np.mean(embeddings[name][member_indices], axis=0, dtype=np.float64)
        norm = float(np.linalg.norm(centroid))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("V4.2 semantic evidence centroid is invalid")
        centroids[name] = centroid / norm

    def centroid_score(experience_id: str) -> float:
        index = atom_index[experience_id]
        return sum(
            weights[name] * float(np.dot(embeddings[name][index], centroids[name]))
            for name in EMBEDDING_VIEW_NAMES
        )

    for experience_id in sorted(
        (item for item in remaining if item not in selected),
        key=lambda item: (-centroid_score(item), item),
    ):
        if len(selected) >= maximum_evidence:
            break
        selected.append(experience_id)
    if len(selected) != maximum_evidence:
        raise ValueError("V4.2 semantic evidence selection did not fill its cap")
    return tuple(selected)


def build_evidence_packet(
    *,
    selected_record: Mapping[str, Any],
    candidate: V42LocalClusterCandidate,
    evidence_ids: Sequence[str],
    atoms: Mapping[str, V42LocalRepairAtom],
    signatures: Mapping[str, V4RepairSignature],
    construction_examples: Mapping[str, Mapping[str, Any]],
    policy_sha256: str,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    for experience_id in evidence_ids:
        atom = atoms[experience_id]
        signature = signatures[experience_id]
        example = construction_examples[experience_id]
        sample_id = str(example["sample_id"])
        if sample_id != atom.sample_id or sample_id != signature.sample_id:
            raise ValueError("V4.2 semantic evidence sample identity drifted")
        if sample_id in seen_samples:
            raise ValueError("V4.2 semantic evidence repeats a construction sample")
        seen_samples.add(sample_id)
        evidence.append(
            {
                "evidence_id": experience_id,
                "sample_id": sample_id,
                "source_experience_type": atom.source_experience_type,
                "semantic_signature": {
                    "problem_structure": atom.problem_structure,
                    "decision_point": atom.decision_point,
                    "failure_mechanism": atom.failure_mechanism,
                    "repair_operator": atom.repair_operator,
                    "verification_operator": atom.verification_operator,
                },
                "question": example["question"],
                "official_solution": example["official_solution"],
                "verified_success_trajectory": example["verified_success_trajectory"],
                "verified_failure_trajectory": example["verified_failure_trajectory"],
                "target_verifier": example["target_verifier"],
                "reference_verifier": example["reference_verifier"],
                "source_provenance_sha256": example["source_provenance_sha256"],
                "source_signature_sha256": signature.signature_sha256,
                "construction_input_sha256": example["construction_input_sha256"],
            }
        )
    packet = {
        "schema_version": V4_2_EVIDENCE_PACKET_SCHEMA,
        "candidate_id": candidate.candidate_id,
        "selection_rank": selected_record["selection_rank"],
        "membership_sha256": candidate.membership_sha256,
        "semantic_policy_sha256": policy_sha256,
        "evidence_selection_rule": "all_up_to_cap_else_five_diverse_plus_medoid_near",
        "evidence_count": len(evidence),
        "evidence": evidence,
        "source_shortlist_record_sha256": selected_record["record_sha256"],
    }
    packet["packet_sha256"] = canonical_json_sha256(packet)
    return packet


def combined_messages(packets: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    system = f"""You are the final semantic auditor and process-card synthesizer for MemGen V4.2. Return JSON only.

Audit every supplied evidence item against its raw question, official solution,
verified-success trajectory, paired verified-failure trajectory, and verifier
records. Judge factual validity separately from whether it supports one shared
failure mechanism, repair operator, and verification operator. An evidence item
is usable only when all four booleans are true. Never repair or reinterpret bad
evidence merely to preserve a candidate.

A candidate is coherent only if at least five usable, distinct-sample items
ground the same precise reasoning-process transition. Similar vocabulary,
story domain, arithmetic surface form, or answer formatting is insufficient.
Reject heterogeneous or overly broad candidates.

Only for a coherent candidate, synthesize one reusable target/reference process
card. Ground target.scope, diagnosis, action, verification, and do_not_use_when
in the official solutions plus verified-success trajectories. Ground the
reference undesired_pattern, failure_signal, failure_mechanism, and
contrast_boundary in the paired verified failures. The card and all shared
process fields must be concise English process text: no names, story objects,
instance quantities, digits, equations, formulas, final answers, or solution
traces. Target and reference must be meaningfully distinct. Do not emit a card
for a rejected candidate.

Return exactly one object with schema_version
{V4_2_COMBINED_BATCH_SCHEMA!r} and a results array in the supplied candidate
order. Each result must have candidate_id, evidence_judgments in the exact
supplied evidence order, shared_process_invariant, shared_failure_mechanism,
shared_repair_operator, shared_verification_operator, valid_distinct_support,
coherent, rejection_reason, and card. Each evidence judgment must have
evidence_id, factually_valid, supports_shared_failure_mechanism,
supports_shared_repair_operator, supports_shared_verification_operator,
rationale, and exclusion_reason. Use null exclusion_reason only when all four
booleans are true. A coherent result uses null rejection_reason and a card;
a rejected result uses a nonempty rejection_reason and null card.

The card object must have exactly this shape:
{{
  "schema_version": "{V4_PROCESS_CARD_SCHEMA}",
  "cluster_key": "the exact candidate_id",
  "target": {{
    "scope": "process-only text",
    "diagnosis": "process-only text",
    "action": "process-only text",
    "verification": "process-only text",
    "do_not_use_when": "process-only text"
  }},
  "reference": {{
    "undesired_pattern": "process-only text",
    "failure_signal": "process-only text",
    "failure_mechanism": "process-only text",
    "contrast_boundary": "process-only text"
  }},
  "support_summary": "process-only text",
  "target_reference_distinction": "process-only text"
}}
Set valid_distinct_support to the exact count of judgments whose four booleans
are all true. Do not add, remove, rename, or reorder candidates or evidence."""
    user = "Candidate evidence packets:\n" + json.dumps(
        list(packets), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def review_messages(items: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    system = f"""You are the independent final reviewer for MemGen V4.2 process cards. Return JSON only.

Do not rewrite cards. For each candidate, independently compare the proposed
card with every retained raw evidence item. Approve only when the target is
grounded in official solutions and verified successes, the reference is
grounded in paired verified failures, the card describes a reusable reasoning
process rather than a domain or answer format, target and reference are
distinct, the operator transfers across all retained examples, and no names,
numbers, equations, answers, story details, or solution traces leak into it.
All six component booleans must be true for approve=true. Approved items must
have an empty issues array; rejected items must have at least one precise issue.

Return exactly one object with schema_version {V4_2_REVIEW_BATCH_SCHEMA!r} and
a results array in supplied order. Each result must have candidate_id,
target_grounded, reference_grounded, process_only,
target_reference_distinct, transferable, leakage_free, approve, evidence, and
issues."""
    user = "Review items:\n" + json.dumps(
        list(items), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def message_characters(messages: Sequence[Mapping[str, str]]) -> int:
    return sum(len(item["role"]) + len(item["content"]) for item in messages)


def pack_requests(
    items: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    max_characters: int,
    message_builder: Callable[[Sequence[Mapping[str, Any]]], list[dict[str, str]]],
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    batches: list[tuple[Mapping[str, Any], ...]] = []
    current: list[Mapping[str, Any]] = []
    for item in items:
        proposed = [*current, item]
        if len(proposed) > batch_size or message_characters(message_builder(proposed)) > max_characters:
            if not current:
                raise ValueError("One V4.2 semantic request item exceeds the character guardrail")
            batches.append(tuple(current))
            current = [item]
            if message_characters(message_builder(current)) > max_characters:
                raise ValueError("One V4.2 semantic request item exceeds the character guardrail")
        else:
            current = proposed
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def _selected_records(path: Path) -> tuple[dict[str, Any], ...]:
    values = tuple(iter_jsonl(path))
    for index, record in enumerate(values, start=1):
        if record.get("schema_version") != SELECTED_CANDIDATE_SCHEMA:
            raise ValueError("Unexpected V4.2 selected-candidate schema")
        if record.get("record_sha256") != _logical_hash(record, "record_sha256"):
            raise ValueError("V4.2 selected-candidate hash mismatch")
        if record.get("selection_rank") != index:
            raise ValueError("V4.2 selected-candidate rank drifted")
    return values


def prepare_preflight(
    *,
    args: argparse.Namespace,
    profile: V42SemanticConstructionProfile,
    split_manifest: Mapping[str, Any],
    experiences: Sequence[Mapping[str, Any]],
    signatures: Sequence[V4RepairSignature],
    source_signature_info: Mapping[str, Any],
    construction_examples: Sequence[Mapping[str, Any]],
    construction_profile: V42ConstructionProfile,
    local_plan: Mapping[str, Any],
    atoms: Sequence[V42LocalRepairAtom],
    candidates: Sequence[V42LocalClusterCandidate],
    embeddings: Mapping[str, np.ndarray],
    local_source_info: Mapping[str, Any],
    shortlist_manifest: Mapping[str, Any],
    shortlist_preflight: Mapping[str, Any],
    selected_records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any]]:
    policy, candidate_exclusions, evidence_exclusions, policy_sha256 = load_semantic_policy(
        args.semantic_policy,
        shortlist_profile_sha256=str(shortlist_manifest["profile_sha256"]),
        shortlist_manifest_sha256=str(shortlist_manifest["manifest_sha256"]),
        shortlist_report_sha256=str(shortlist_preflight["report_sha256"]),
        selected_records=selected_records,
    )
    del policy
    candidates_by_id = {item.candidate_id: item for item in candidates}
    atoms_by_id = {item.experience_id: item for item in atoms}
    signatures_by_id = {item.experience_id: item for item in signatures}
    examples_by_id = {str(item["experience_id"]): item for item in construction_examples}
    atom_index = {item.experience_id: index for index, item in enumerate(atoms)}
    weights = {
        "mechanism": construction_profile.mechanism_weight,
        "repair": construction_profile.repair_weight,
        "applicability": construction_profile.applicability_weight,
    }

    packets: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for selected in selected_records:
        candidate_id = str(selected["candidate"]["candidate_id"])
        candidate = candidates_by_id.get(candidate_id)
        if candidate is None or candidate.to_dict() != selected["candidate"]:
            raise ValueError("V4.2 shortlist candidate differs from authenticated local source")
        if candidate_id in candidate_exclusions:
            record = {
                "schema_version": POLICY_EXCLUSION_RECORD_SCHEMA,
                "candidate_id": candidate_id,
                "selection_rank": selected["selection_rank"],
                "level": "candidate",
                "experience_id": None,
                "reason": candidate_exclusions[candidate_id],
            }
            record["record_sha256"] = canonical_json_sha256(record)
            exclusions.append(record)
            continue
        excluded = evidence_exclusions.get(candidate_id, {})
        for experience_id, reason in sorted(excluded.items()):
            record = {
                "schema_version": POLICY_EXCLUSION_RECORD_SCHEMA,
                "candidate_id": candidate_id,
                "selection_rank": selected["selection_rank"],
                "level": "evidence",
                "experience_id": experience_id,
                "reason": reason,
            }
            record["record_sha256"] = canonical_json_sha256(record)
            exclusions.append(record)
        evidence_ids = select_evidence_ids(
            candidate,
            atom_index=atom_index,
            atoms=atoms_by_id,
            embeddings=embeddings,
            weights=weights,
            excluded_experience_ids=set(excluded),
            maximum_evidence=profile.maximum_evidence_per_candidate,
        )
        if len(evidence_ids) < profile.minimum_valid_distinct_support:
            record = {
                "schema_version": POLICY_EXCLUSION_RECORD_SCHEMA,
                "candidate_id": candidate_id,
                "selection_rank": selected["selection_rank"],
                "level": "candidate",
                "experience_id": None,
                "reason": "fewer than five policy-eligible distinct construction examples",
            }
            record["record_sha256"] = canonical_json_sha256(record)
            exclusions.append(record)
            continue
        packets.append(
            build_evidence_packet(
                selected_record=selected,
                candidate=candidate,
                evidence_ids=evidence_ids,
                atoms=atoms_by_id,
                signatures=signatures_by_id,
                construction_examples=examples_by_id,
                policy_sha256=policy_sha256,
            )
        )

    batches = pack_requests(
        packets,
        batch_size=profile.synthesis_batch_size,
        max_characters=profile.max_request_characters,
        message_builder=combined_messages,
    )
    batch_rows = []
    for index, batch in enumerate(batches):
        messages = combined_messages(batch)
        row = {
            "unit_id": f"combined-{index:04d}",
            "candidate_ids": [str(item["candidate_id"]) for item in batch],
            "packet_sha256": {
                str(item["candidate_id"]): str(item["packet_sha256"]) for item in batch
            },
            "request_characters": message_characters(messages),
            "input_sha256": canonical_json_sha256(batch),
        }
        batch_rows.append(row)

    source_shortlist = {
        "profile_sha256": shortlist_manifest["profile_sha256"],
        "manifest_sha256": shortlist_manifest["manifest_sha256"],
        "report_sha256": shortlist_preflight["report_sha256"],
    }
    inputs = {
        "experiences": {"path": str(args.experiences.resolve()), "sha256": file_sha256(args.experiences), "count": len(experiences)},
        "split_manifest": {"path": str(args.split_manifest.resolve()), "manifest_sha256": split_manifest["manifest_sha256"], "file_sha256": file_sha256(args.split_manifest)},
        "source_signatures": dict(source_signature_info),
        "local_construction": dict(local_source_info),
        "local_cluster_plan_sha256": local_plan["plan_sha256"],
        "source_shortlist": source_shortlist,
        "semantic_policy": {"path": str(args.semantic_policy.resolve()), "file_sha256": file_sha256(args.semantic_policy), "policy_sha256": policy_sha256},
    }
    profile_record = {
        "schema_version": SEMANTIC_PROFILE_RECORD_SCHEMA,
        "construction_version": "v4.2",
        "stage": "semantic_audit_synthesis_review",
        "profile": profile.to_dict(),
        "profile_sha256": profile.profile_sha256,
        "inputs": inputs,
        "teacher": {"model": args.model, "base_url": args.base_url, "temperature": args.temperature, "thinking": args.thinking},
        "request_configuration": {
            "synthesis_max_tokens": args.synthesis_max_tokens,
            "review_max_tokens": args.review_max_tokens,
            "short_retries": args.retries,
            "proxy_retries": args.proxy_retries,
            "proxy_retry_initial_seconds": args.proxy_retry_initial_seconds,
            "proxy_retry_max_seconds": args.proxy_retry_max_seconds,
            "connect_timeout_seconds": args.connect_timeout_seconds,
            "read_timeout_seconds": args.read_timeout_seconds,
        },
        "prompt_versions": {"combined": V4_2_COMBINED_PROMPT_VERSION, "review": V4_2_REVIEW_PROMPT_VERSION},
        "implementation_sha256": _implementation_state(),
    }
    excluded_candidate_count = sum(
        item["level"] == "candidate" for item in exclusions
    )
    plan = {
        "schema_version": V4_2_PAID_PLAN_SCHEMA,
        "construction_version": "v4.2",
        "status": "exact_paid_request_plan_complete",
        "profile_sha256": profile.profile_sha256,
        "semantic_policy_sha256": policy_sha256,
        "source_selected_candidate_count": len(selected_records),
        "authenticated_policy_excluded_candidate_count": len(candidate_exclusions),
        "preflight_excluded_candidate_count": excluded_candidate_count,
        "planned_candidate_count": len(packets),
        "planned_combined_request_count": len(batch_rows),
        "nominal_review_request_count_if_all_coherent": math.ceil(
            len(packets) / profile.review_batch_size
        ),
        "nominal_total_paid_request_count_if_all_coherent": (
            len(batch_rows) + math.ceil(len(packets) / profile.review_batch_size)
        ),
        "maximum_combined_request_units_after_recursive_split": sum(
            2 * len(batch) - 1 for batch in batches
        ),
        "maximum_review_request_units_after_recursive_split": (
            sum(
                2 * len(packets[start : start + profile.review_batch_size]) - 1
                for start in range(0, len(packets), profile.review_batch_size)
            )
        ),
        "request_character_count": sum(item["request_characters"] for item in batch_rows),
        "batches": batch_rows,
    }
    plan["maximum_total_request_units_after_recursive_split"] = (
        plan["maximum_combined_request_units_after_recursive_split"]
        + plan["maximum_review_request_units_after_recursive_split"]
    )
    plan["maximum_short_retry_http_attempts_excluding_proxy_failures"] = (
        plan["maximum_total_request_units_after_recursive_split"] * args.retries
    )
    plan["plan_sha256"] = canonical_json_sha256(plan)
    preflight = {
        "schema_version": V4_2_PAID_PREFLIGHT_SCHEMA,
        "construction_version": "v4.2",
        "status": "semantic_evidence_ready_api_not_started",
        "external_api_calls_made": 0,
        "api_key_read": False,
        "automatic_paid_stage_transition": False,
        "qualified_for_online_use": False,
        "source_selected_candidate_count": len(selected_records),
        "planned_candidate_count": len(packets),
        "authenticated_policy_excluded_candidate_count": len(candidate_exclusions),
        "preflight_excluded_candidate_count": excluded_candidate_count,
        "policy_excluded_evidence_count": sum(len(item) for item in evidence_exclusions.values()),
        "evidence_count": sum(int(item["evidence_count"]) for item in packets),
        "evidence_count_distribution": {
            str(key): value
            for key, value in sorted(
                Counter(int(item["evidence_count"]) for item in packets).items()
            )
        },
        "planned_combined_request_count": len(batch_rows),
        "nominal_review_request_count_if_all_coherent": plan[
            "nominal_review_request_count_if_all_coherent"
        ],
        "nominal_total_paid_request_count_if_all_coherent": plan[
            "nominal_total_paid_request_count_if_all_coherent"
        ],
        "maximum_total_request_units_after_recursive_split": plan[
            "maximum_total_request_units_after_recursive_split"
        ],
        "maximum_short_retry_http_attempts_excluding_proxy_failures": plan[
            "maximum_short_retry_http_attempts_excluding_proxy_failures"
        ],
        "exact_combined_request_characters": plan["request_character_count"],
        "estimated_combined_input_tokens_at_three_chars_per_token": math.ceil(plan["request_character_count"] / 3),
        "profile_sha256": profile.profile_sha256,
        "semantic_policy_sha256": policy_sha256,
        "paid_stage_plan_sha256": plan["plan_sha256"],
        "source_shortlist": source_shortlist,
        "note": "The token estimate is character-based. The paid stage still requires --stage paid --approve-paid-stage and a credential.",
    }
    preflight["report_sha256"] = canonical_json_sha256(preflight)
    return tuple(packets), tuple(exclusions), profile_record, {"plan": plan, "preflight": preflight}


def _write_or_validate_preflight(
    output_dir: Path,
    *,
    packets: Sequence[Mapping[str, Any]],
    exclusions: Sequence[Mapping[str, Any]],
    profile_record: Mapping[str, Any],
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    resume: bool,
) -> None:
    values = {
        "construction_profile.json": profile_record,
        "semantic_evidence_packets.jsonl": packets,
        "policy_exclusions.jsonl": exclusions,
        "paid_stage_plan.json": plan,
        "api_preflight_report.json": preflight,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "construction_profile.json").exists():
        if not resume:
            raise ValueError("V4.2 semantic output exists; pass --resume to authenticate it")
        for name, expected in values.items():
            path = output_dir / name
            if not path.is_file():
                raise ValueError(f"V4.2 semantic preflight artifact is missing: {name}")
            actual: Any = tuple(iter_jsonl(path)) if name.endswith(".jsonl") else _load_json(path)
            if actual != (tuple(expected) if name.endswith(".jsonl") else expected):
                raise ValueError(f"V4.2 semantic preflight artifact drifted: {name}")
        return
    _write_json(output_dir / "construction_profile.json", profile_record)
    _write_jsonl(output_dir / "semantic_evidence_packets.jsonl", packets)
    _write_jsonl(output_dir / "policy_exclusions.jsonl", exclusions)
    _write_json(output_dir / "paid_stage_plan.json", plan)
    _write_json(output_dir / "api_preflight_report.json", preflight)


def _load_checkpoint(path: Path, *, schema: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return result
    for record in iter_jsonl(path):
        if record.get("schema_version") != schema:
            raise ValueError(f"Unexpected checkpoint schema: {path}")
        candidate_id = str(record.get("candidate_id", ""))
        if not candidate_id or candidate_id in result:
            raise ValueError(f"Missing or duplicate checkpoint candidate: {candidate_id!r}")
        if record.get("record_sha256") != _logical_hash(record, "record_sha256"):
            raise ValueError(f"Checkpoint hash mismatch: {candidate_id}")
        result[candidate_id] = record
    return result


def _store_checkpoint(path: Path, records: dict[str, dict[str, Any]], record: dict[str, Any]) -> None:
    records[str(record["candidate_id"])] = record
    _write_jsonl(path, (records[key] for key in sorted(records)))


def _checkpoint_record(
    *,
    schema: str,
    candidate_id: str,
    prompt_version: str,
    input_sha256: str,
    payload: Mapping[str, Any],
    status: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    record = {
        "schema_version": schema,
        "candidate_id": candidate_id,
        "created_at": utc_now(),
        "generation_status": status,
        "prompt_version": prompt_version,
        "input_sha256": input_sha256,
        "teacher": {"model": args.model, "base_url": args.base_url, "temperature": args.temperature, "thinking": args.thinking},
        "payload": dict(payload),
        "payload_sha256": canonical_json_sha256(payload),
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def _checkpoint_matches(
    record: Mapping[str, Any],
    *,
    prompt_version: str,
    input_sha256: str,
    args: argparse.Namespace,
) -> bool:
    return bool(
        record.get("prompt_version") == prompt_version
        and record.get("input_sha256") == input_sha256
        and record.get("teacher") == {"model": args.model, "base_url": args.base_url, "temperature": args.temperature, "thinking": args.thinking}
        and record.get("payload_sha256") == canonical_json_sha256(record.get("payload"))
    )


def _fallback_synthesis(packet: Mapping[str, Any]) -> V42CombinedSynthesis:
    judgments = tuple(
        V42EvidenceJudgment(
            evidence_id=str(item["evidence_id"]),
            factually_valid=False,
            supports_shared_failure_mechanism=False,
            supports_shared_repair_operator=False,
            supports_shared_verification_operator=False,
            rationale="The teacher response remained invalid after bounded retries.",
            exclusion_reason="No schema-valid semantic judgment was returned.",
        )
        for item in packet["evidence"]
    )
    return V42CombinedSynthesis(
        candidate_id=str(packet["candidate_id"]),
        evidence_judgments=judgments,
        shared_process_invariant=None,
        shared_failure_mechanism=None,
        shared_repair_operator=None,
        shared_verification_operator=None,
        valid_distinct_support=0,
        coherent=False,
        rejection_reason="Teacher response remained invalid after bounded retries.",
        card=None,
    )


def run_combined_stage(
    packets: Sequence[Mapping[str, Any]],
    *,
    client: TeacherClient,
    checkpoint_path: Path,
    args: argparse.Namespace,
    profile: V42SemanticConstructionProfile,
    call_counter: list[int],
) -> dict[str, V42CombinedSynthesis]:
    existing = _load_checkpoint(checkpoint_path, schema=V4_2_COMBINED_RECORD_SCHEMA)
    packet_by_id = {str(item["candidate_id"]): item for item in packets}
    if set(existing) - set(packet_by_id):
        raise ValueError("V4.2 combined checkpoint contains an unexpected candidate")
    if existing and not args.resume:
        raise ValueError("V4.2 combined checkpoint exists; pass --resume")
    resolved: dict[str, V42CombinedSynthesis] = {}
    pending: list[Mapping[str, Any]] = []
    for packet in packets:
        candidate_id = str(packet["candidate_id"])
        input_sha256 = str(packet["packet_sha256"])
        record = existing.get(candidate_id)
        if record is not None and _checkpoint_matches(record, prompt_version=V4_2_COMBINED_PROMPT_VERSION, input_sha256=input_sha256, args=args):
            payload = {"schema_version": V4_2_COMBINED_BATCH_SCHEMA, "results": [record["payload"]]}
            resolved[candidate_id] = parse_v4_2_combined_batch(
                payload,
                expected={candidate_id: [str(item["evidence_id"]) for item in packet["evidence"]]},
            )[0]
        else:
            pending.append(packet)

    def resolve(batch: tuple[Mapping[str, Any], ...]) -> None:
        expected = {
            str(packet["candidate_id"]): [str(item["evidence_id"]) for item in packet["evidence"]]
            for packet in batch
        }
        try:
            call_counter[0] += 1
            payload = client.call(
                combined_messages(batch),
                response_parser=lambda content: {
                    "schema_version": V4_2_COMBINED_BATCH_SCHEMA,
                    "results": [
                        item.to_dict()
                        for item in parse_v4_2_combined_batch(_parse_json_object(content), expected=expected)
                    ],
                },
                request_label="v4.2-combined",
                expose_parser_error=True,
                repair_parser_errors=True,
            )
            parsed = parse_v4_2_combined_batch(payload, expected=expected)
            statuses = ["teacher_validated"] * len(parsed)
        except TeacherInvalidResponseError:
            if len(batch) > 1:
                middle = len(batch) // 2
                resolve(batch[:middle])
                resolve(batch[middle:])
                return
            parsed = (_fallback_synthesis(batch[0]),)
            statuses = ["deterministic_rejection_after_invalid_teacher_response"]
        for packet, synthesis, status in zip(batch, parsed, statuses):
            candidate_id = str(packet["candidate_id"])
            record = _checkpoint_record(
                schema=V4_2_COMBINED_RECORD_SCHEMA,
                candidate_id=candidate_id,
                prompt_version=V4_2_COMBINED_PROMPT_VERSION,
                input_sha256=str(packet["packet_sha256"]),
                payload=synthesis.to_dict(),
                status=status,
                args=args,
            )
            _store_checkpoint(checkpoint_path, existing, record)
            resolved[candidate_id] = synthesis
            print(f"[v4.2-semantic] combined {len(resolved)}/{len(packets)} {candidate_id} coherent={synthesis.coherent}", flush=True)

    batches = pack_requests(
        pending,
        batch_size=profile.synthesis_batch_size,
        max_characters=profile.max_request_characters,
        message_builder=combined_messages,
    )
    for batch in batches:
        resolve(batch)
    if set(resolved) != set(packet_by_id):
        raise ValueError("V4.2 combined stage lost candidate coverage")
    return resolved


def _review_item(packet: Mapping[str, Any], synthesis: V42CombinedSynthesis) -> dict[str, Any]:
    valid = set(synthesis.valid_evidence_ids)
    evidence = [item for item in packet["evidence"] if item["evidence_id"] in valid]
    return {
        "candidate_id": synthesis.candidate_id,
        "semantic_audit": synthesis.to_dict(),
        "card": synthesis.card.to_dict() if synthesis.card is not None else None,
        "retained_evidence": evidence,
        "evidence_packet_sha256": packet["packet_sha256"],
    }


def _fallback_review(candidate_id: str) -> V4CardReview:
    return V4CardReview(
        cluster_key=candidate_id,
        target_grounded=False,
        reference_grounded=False,
        process_only=False,
        target_reference_distinct=False,
        transferable=False,
        leakage_free=False,
        approve=False,
        evidence="The independent review response remained invalid after bounded retries.",
        issues=("No schema-valid independent review was returned.",),
    )


def run_review_stage(
    packets: Sequence[Mapping[str, Any]],
    syntheses: Mapping[str, V42CombinedSynthesis],
    *,
    client: TeacherClient,
    checkpoint_path: Path,
    args: argparse.Namespace,
    profile: V42SemanticConstructionProfile,
    call_counter: list[int],
) -> dict[str, V4CardReview]:
    items = [
        _review_item(packet, syntheses[str(packet["candidate_id"])])
        for packet in packets
        if syntheses[str(packet["candidate_id"])].coherent
    ]
    existing = _load_checkpoint(checkpoint_path, schema=V4_2_REVIEW_RECORD_SCHEMA)
    item_by_id = {str(item["candidate_id"]): item for item in items}
    if set(existing) - set(item_by_id):
        raise ValueError("V4.2 review checkpoint contains an unexpected candidate")
    if existing and not args.resume:
        raise ValueError("V4.2 review checkpoint exists; pass --resume")
    resolved: dict[str, V4CardReview] = {}
    pending: list[Mapping[str, Any]] = []
    for item in items:
        candidate_id = str(item["candidate_id"])
        input_sha256 = canonical_json_sha256(item)
        record = existing.get(candidate_id)
        if record is not None and _checkpoint_matches(record, prompt_version=V4_2_REVIEW_PROMPT_VERSION, input_sha256=input_sha256, args=args):
            resolved[candidate_id] = parse_v4_2_review_batch(
                {"schema_version": V4_2_REVIEW_BATCH_SCHEMA, "results": [record["payload"]]},
                expected_candidate_ids=(candidate_id,),
            )[0]
        else:
            pending.append(item)

    def resolve(batch: tuple[Mapping[str, Any], ...]) -> None:
        expected = tuple(str(item["candidate_id"]) for item in batch)
        try:
            call_counter[0] += 1
            payload = client.call(
                review_messages(batch),
                response_parser=lambda content: {
                    "schema_version": V4_2_REVIEW_BATCH_SCHEMA,
                    "results": [
                        {"candidate_id": item.cluster_key, **item.to_dict()}
                        for item in parse_v4_2_review_batch(
                            _parse_json_object(content),
                            expected_candidate_ids=expected,
                        )
                    ],
                },
                request_label="v4.2-review",
                expose_parser_error=True,
                repair_parser_errors=True,
            )
            parsed = parse_v4_2_review_batch(payload, expected_candidate_ids=expected)
            statuses = ["teacher_validated"] * len(parsed)
        except TeacherInvalidResponseError:
            if len(batch) > 1:
                middle = len(batch) // 2
                resolve(batch[:middle])
                resolve(batch[middle:])
                return
            parsed = (_fallback_review(expected[0]),)
            statuses = ["deterministic_rejection_after_invalid_teacher_response"]
        for item, review, status in zip(batch, parsed, statuses):
            candidate_id = str(item["candidate_id"])
            record = _checkpoint_record(
                schema=V4_2_REVIEW_RECORD_SCHEMA,
                candidate_id=candidate_id,
                prompt_version=V4_2_REVIEW_PROMPT_VERSION,
                input_sha256=canonical_json_sha256(item),
                payload={"candidate_id": candidate_id, **review.to_dict()},
                status=status,
                args=args,
            )
            _store_checkpoint(checkpoint_path, existing, record)
            resolved[candidate_id] = review
            print(f"[v4.2-semantic] review {len(resolved)}/{len(items)} {candidate_id} approve={review.approve}", flush=True)

    batches = pack_requests(
        pending,
        batch_size=profile.review_batch_size,
        max_characters=profile.max_request_characters,
        message_builder=review_messages,
    )
    for batch in batches:
        resolve(batch)
    if set(resolved) != set(item_by_id):
        raise ValueError("V4.2 review stage lost candidate coverage")
    return resolved


class _CountingSession(requests.Session):
    def __init__(self, counter: list[int]) -> None:
        super().__init__()
        self._counter = counter

    def request(self, method: str, url: str, *args: Any, **kwargs: Any) -> requests.Response:
        self._counter[0] += 1
        return super().request(method, url, *args, **kwargs)


def _counted_client(
    args: argparse.Namespace,
    *,
    api_key: str,
    max_tokens: int,
    http_counter: list[int],
) -> TeacherClient:
    return TeacherClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        max_tokens=max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        proxy_retries=args.proxy_retries,
        proxy_retry_initial_seconds=args.proxy_retry_initial_seconds,
        proxy_retry_max_seconds=args.proxy_retry_max_seconds,
        connect_timeout_seconds=args.connect_timeout_seconds,
        read_timeout_seconds=args.read_timeout_seconds,
        thinking=args.thinking,
        session=_CountingSession(http_counter),
    )


def finalize_bank(
    *,
    packets: Sequence[Mapping[str, Any]],
    syntheses: Mapping[str, V42CombinedSynthesis],
    reviews: Mapping[str, V4CardReview],
    candidates: Mapping[str, V42LocalClusterCandidate],
    signatures: Mapping[str, V4RepairSignature],
    profile: V42SemanticConstructionProfile,
    profile_record: Mapping[str, Any],
    policy_exclusions: Sequence[Mapping[str, Any]],
    request_unit_count: int,
    external_api_call_count: int,
    output_dir: Path,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any], dict[str, Any]]:
    packet_by_id = {str(item["candidate_id"]): item for item in packets}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = [
        {
            "schema_version": SEMANTIC_REJECTION_RECORD_SCHEMA,
            "candidate_id": item["candidate_id"],
            "selection_rank": item["selection_rank"],
            "stage": "authenticated_semantic_policy",
            "reason": item["reason"],
            "source_record_sha256": item["record_sha256"],
        }
        for item in policy_exclusions
        if item["level"] == "candidate"
    ]
    source_shortlist = profile_record["inputs"]["source_shortlist"]
    policy_sha256 = profile_record["inputs"]["semantic_policy"]["policy_sha256"]
    for packet in packets:
        candidate_id = str(packet["candidate_id"])
        synthesis = syntheses[candidate_id]
        if not synthesis.coherent:
            rejected.append({
                "schema_version": SEMANTIC_REJECTION_RECORD_SCHEMA,
                "candidate_id": candidate_id,
                "selection_rank": packet["selection_rank"],
                "stage": "semantic_audit_and_synthesis",
                "reason": synthesis.rejection_reason,
                "source_record_sha256": packet["packet_sha256"],
            })
            continue
        review = reviews[candidate_id]
        if not review.approve:
            rejected.append({
                "schema_version": SEMANTIC_REJECTION_RECORD_SCHEMA,
                "candidate_id": candidate_id,
                "selection_rank": packet["selection_rank"],
                "stage": "independent_card_review",
                "reason": "; ".join(review.issues),
                "source_record_sha256": canonical_json_sha256(review.to_dict()),
            })
            continue
        accepted.append(
            build_v4_2_semantic_bank_record(
                candidate=candidates[candidate_id],
                synthesis=synthesis,
                review=review,
                evidence_packet=packet_by_id[candidate_id],
                signatures=signatures,
                profile=profile,
                source_shortlist=source_shortlist,
                semantic_policy_sha256=policy_sha256,
            )
        )

    accepted.sort(key=lambda item: int(packet_by_id[item["cluster"]["cluster_key"]]["selection_rank"]))
    overflow = accepted[profile.target_runtime_bank_cap :]
    accepted = accepted[: profile.target_runtime_bank_cap]
    for item in overflow:
        candidate_id = str(item["cluster"]["cluster_key"])
        rejected.append({
            "schema_version": SEMANTIC_REJECTION_RECORD_SCHEMA,
            "candidate_id": candidate_id,
            "selection_rank": packet_by_id[candidate_id]["selection_rank"],
            "stage": "runtime_bank_cap",
            "reason": "approved candidate exceeded the frozen runtime-bank cap",
            "source_record_sha256": item["record_sha256"],
        })
    for item in rejected:
        item["record_sha256"] = canonical_json_sha256(item)
    rejected.sort(key=lambda item: (int(item["selection_rank"]), str(item["candidate_id"])))
    if not accepted:
        raise ValueError("V4.2 semantic construction produced no independently approved bank records")
    teacher = {
        **profile_record["teacher"],
        "combined_prompt_version": V4_2_COMBINED_PROMPT_VERSION,
        "review_prompt_version": V4_2_REVIEW_PROMPT_VERSION,
    }
    manifest = build_v4_2_semantic_bank_manifest(
        records=accepted,
        profile=profile,
        inputs=profile_record["inputs"],
        teacher=teacher,
        archive={
            "semantic_rejection_count": len(rejected),
            "semantic_rejections_path": "semantic_rejections.jsonl",
            "policy_evidence_exclusion_count": sum(item["level"] == "evidence" for item in policy_exclusions),
        },
    )
    report = {
        "schema_version": PAID_REPORT_SCHEMA,
        "construction_version": "v4.2",
        "status": "semantic_bank_constructed_not_tensor_compiled",
        "qualified_for_online_use": False,
        "source_candidate_count": len(packets) + sum(item["level"] == "candidate" for item in policy_exclusions),
        "combined_coherent_count": sum(item.coherent for item in syntheses.values()),
        "combined_rejected_count": sum(not item.coherent for item in syntheses.values()),
        "independent_review_approved_count": sum(item.approve for item in reviews.values()),
        "independent_review_rejected_count": sum(not item.approve for item in reviews.values()),
        "bank_record_count": len(accepted),
        "semantic_rejection_count": len(rejected),
        "teacher_request_units_invoked_this_invocation": request_unit_count,
        "external_api_http_attempts_this_invocation": external_api_call_count,
        "bank_manifest_sha256": manifest["manifest_sha256"],
        "profile_sha256": profile.profile_sha256,
        "next_stage": "compile layer twenty four target/reference side KV and build selector anchors",
    }
    report["report_sha256"] = canonical_json_sha256(report)
    _write_jsonl(output_dir / "bank_records.jsonl", accepted)
    _write_jsonl(output_dir / "semantic_rejections.jsonl", rejected)
    _write_json(output_dir / "bank_manifest.json", manifest)
    _write_json(output_dir / "paid_stage_report.json", report)
    return tuple(accepted), manifest, report


def main() -> None:
    args = parse_args()
    profile = validate_cli(args)
    for path in (
        args.experiences,
        args.split_manifest,
        args.source_signatures,
        args.source_construction_profile,
        args.semantic_policy,
    ):
        if not path.expanduser().is_file():
            raise ValueError(f"Missing V4.2 semantic input: {path}")
    args.experiences = args.experiences.expanduser().resolve()
    args.split_manifest = args.split_manifest.expanduser().resolve()
    args.source_signatures = args.source_signatures.expanduser().resolve()
    args.source_construction_profile = args.source_construction_profile.expanduser().resolve()
    args.local_construction_dir = args.local_construction_dir.expanduser().resolve()
    args.shortlist_dir = args.shortlist_dir.expanduser().resolve()
    args.semantic_policy = args.semantic_policy.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.base_url = _validate_base_url(args.base_url)
    protected_output_directories = {
        PROJECT_ROOT.resolve(),
        args.experiences.parent,
        args.source_signatures.parent,
        args.local_construction_dir,
        args.shortlist_dir,
    }
    if args.output_dir in protected_output_directories:
        raise ValueError("V4.2 semantic output must differ from every input directory")

    split_manifest = _validate_split_manifest(args.split_manifest, dataset_revision=args.dataset_revision)
    experiences = load_v4_experiences(args.experiences, split_manifest=split_manifest)
    signatures, source_signature_info = load_authenticated_signatures(
        args.source_signatures,
        source_profile_path=args.source_construction_profile,
        experiences=experiences,
    )
    (
        construction_profile,
        local_plan,
        atoms,
        candidates,
        _review_packets,
        embeddings,
        local_source_info,
    ) = load_authenticated_local_construction(args.local_construction_dir)
    shortlist_profile_json = _load_json(args.shortlist_dir / "construction_profile.json")
    shortlist_profile = V42ShortlistProfile(**shortlist_profile_json.get("profile", {}))
    expected_shortlist_profile = shortlist_profile_record(shortlist_profile, source_info=local_source_info)
    shortlist_preflight = validate_completed_output(
        args.shortlist_dir,
        expected_profile_record=expected_shortlist_profile,
    )
    if shortlist_preflight is None:
        raise ValueError("V4.2 semantic construction requires a completed shortlist")
    shortlist_manifest = _load_json(args.shortlist_dir / "synthesis_shortlist_manifest.json")
    selected_records = _selected_records(args.shortlist_dir / "selected_synthesis_candidates.jsonl")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The datasets package is required to authenticate GSM8K evidence") from exc
    dataset = load_dataset("openai/gsm8k", "main", split="train", revision=args.dataset_revision)
    construction_examples = attach_official_solutions(
        experiences,
        split_manifest=split_manifest,
        dataset_revision=args.dataset_revision,
        dataset=dataset,
    )
    packets, exclusions, profile_record, prepared = prepare_preflight(
        args=args,
        profile=profile,
        split_manifest=split_manifest,
        experiences=experiences,
        signatures=signatures,
        source_signature_info=source_signature_info,
        construction_examples=construction_examples,
        construction_profile=construction_profile,
        local_plan=local_plan,
        atoms=atoms,
        candidates=candidates,
        embeddings=embeddings,
        local_source_info=local_source_info,
        shortlist_manifest=shortlist_manifest,
        shortlist_preflight=shortlist_preflight,
        selected_records=selected_records,
    )
    _write_or_validate_preflight(
        args.output_dir,
        packets=packets,
        exclusions=exclusions,
        profile_record=profile_record,
        plan=prepared["plan"],
        preflight=prepared["preflight"],
        resume=args.resume,
    )
    print(
        f"[v4.2-semantic] preflight PASS candidates={len(packets)} "
        f"combined_requests={prepared['plan']['planned_combined_request_count']} "
        f"nominal_total_requests={prepared['plan']['nominal_total_paid_request_count_if_all_coherent']} "
        f"split_worst_case_units={prepared['plan']['maximum_total_request_units_after_recursive_split']} "
        "api_key_read=false api_calls=0",
        flush=True,
    )
    print(f"[v4.2-semantic] report={args.output_dir / 'api_preflight_report.json'}", flush=True)
    if args.stage == "preflight":
        return
    if not args.approve_paid_stage:
        raise RuntimeError("Paid V4.2 stage requires explicit --approve-paid-stage")

    # Credential access deliberately occurs only after the complete authenticated
    # preflight is durable and explicit paid approval has passed.
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before the explicitly approved paid stage")
    request_unit_counter = [0]
    http_counter = [0]
    with _counted_client(
        args,
        api_key=api_key,
        max_tokens=args.synthesis_max_tokens,
        http_counter=http_counter,
    ) as client:
        syntheses = run_combined_stage(
            packets,
            client=client,
            checkpoint_path=args.output_dir / "combined_synthesis_records.jsonl",
            args=args,
            profile=profile,
            call_counter=request_unit_counter,
        )
    with _counted_client(
        args,
        api_key=api_key,
        max_tokens=args.review_max_tokens,
        http_counter=http_counter,
    ) as client:
        reviews = run_review_stage(
            packets,
            syntheses,
            client=client,
            checkpoint_path=args.output_dir / "review_records.jsonl",
            args=args,
            profile=profile,
            call_counter=request_unit_counter,
        )
    records, manifest, report = finalize_bank(
        packets=packets,
        syntheses=syntheses,
        reviews=reviews,
        candidates={item.candidate_id: item for item in candidates},
        signatures={item.experience_id: item for item in signatures},
        profile=profile,
        profile_record=profile_record,
        policy_exclusions=exclusions,
        request_unit_count=request_unit_counter[0],
        external_api_call_count=http_counter[0],
        output_dir=args.output_dir,
    )
    print(
        f"[v4.2-semantic] bank complete records={len(records)} "
        f"rejected={report['semantic_rejection_count']} "
        f"request_units={request_unit_counter[0]} http_attempts={http_counter[0]}",
        flush=True,
    )
    print(f"[v4.2-semantic] manifest_sha256={manifest['manifest_sha256']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4.2-semantic] error: {exc}", file=sys.stderr, flush=True)
        raise
