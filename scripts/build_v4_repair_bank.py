#!/usr/bin/env python3
"""Construct the tensor-free MemGen V4 repair bank with DeepSeek V4 Flash.

This script is the first half of V4's offline stage.  It consumes only raw,
verifier-backed ``bank-source`` success/failure pairs plus authenticated GSM8K
official solutions.  It never imports V3 teacher-bank text or V3 side-KV
tensors.

The generated manifest is deliberately marked ``qualified_for_online_use:
false``.  Layer-24 target/reference compilation, state-anchor construction,
and causal qualification must bind their own artifacts before a V4 bank can be
loaded by the online runtime.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import (
    EXPERIENCE_SCHEMA,
    SPLIT_MANIFEST_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
    upgrade_verified_experience,
)
from memgen.experience.v4_bank import (
    V4_CARD_PROMPT_VERSION,
    V4_CARD_REVIEW_PROMPT_VERSION,
    V4_CLUSTER_PLAN_SCHEMA,
    V4_CLUSTER_PROMPT_VERSION,
    V4_MAX_CONSTRUCTION_EXAMPLES,
    V4_MIN_CONSTRUCTION_EXAMPLES,
    V4_SIGNATURE_PROMPT_VERSION,
    V4_TEACHER_MODEL,
    V4_TEACHER_THINKING,
    V4CardReview,
    V4ConstructionProfile,
    V4ProcessCard,
    V4RepairCluster,
    V4RepairSignature,
    build_v4_bank_manifest,
    build_v4_bank_record,
    parse_v4_card_review,
    parse_v4_cluster_plan,
    parse_v4_process_card,
    parse_v4_repair_signature,
)
from scripts.build_teacher_bank import TeacherClient, TeacherInvalidResponseError


SIGNATURE_RECORD_SCHEMA = "memgen-v4-repair-signature-record-v1"
CARD_RECORD_SCHEMA = "memgen-v4-process-card-record-v1"
REVIEW_RECORD_SCHEMA = "memgen-v4-process-card-review-record-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repository_state() -> dict[str, Any]:
    paths = (
        "memgen/experience/v4_bank.py",
        "scripts/build_v4_repair_bank.py",
        "scripts/build_teacher_bank.py",
    )
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("V4 bank construction requires a git revision") from exc
    if not revision:
        raise RuntimeError("V4 bank construction resolved an empty git revision")
    return {
        "git_revision": revision,
        "implementation_sha256": {
            relative: file_sha256(PROJECT_ROOT / relative) for relative in paths
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiences", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="main")
    parser.add_argument(
        "--model",
        default=os.environ.get("DEEPSEEK_TEACHER_MODEL", V4_TEACHER_MODEL),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--signature-max-tokens", type=int, default=900)
    parser.add_argument("--cluster-max-tokens", type=int, default=8000)
    parser.add_argument("--card-max-tokens", type=int, default=2200)
    parser.add_argument("--review-max-tokens", type=int, default=1400)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--proxy-retries", type=int, default=20)
    parser.add_argument("--proxy-retry-initial-seconds", type=float, default=30.0)
    parser.add_argument("--proxy-retry-max-seconds", type=float, default=300.0)
    parser.add_argument("--connect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--read-timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only records bound to the exact prompt/model/input hashes.",
    )
    return parser.parse_args()


def _parse_json_object(content: str) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek returned an empty final content field")
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    value = json.loads(cleaned)
    if not isinstance(value, dict):
        raise ValueError("DeepSeek response must be one JSON object")
    return value


def _validate_split_manifest(path: Path, *, dataset_revision: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        raise ValueError("Unexpected GSM8K split-manifest schema")
    stored_hash = value.get("manifest_sha256")
    logical = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    }
    if stored_hash != canonical_json_sha256(logical):
        raise ValueError("GSM8K split-manifest hash mismatch")
    if not value.get("overlap_check", {}).get("passed"):
        raise ValueError("GSM8K split manifest did not pass overlap checking")
    dataset = value.get("dataset", {})
    if dataset.get("name") != "openai/gsm8k" or dataset.get("configuration") != "main":
        raise ValueError("V4 construction requires the authenticated GSM8K main split")
    if dataset.get("revision") != dataset_revision:
        raise ValueError("V4 dataset revision differs from the split manifest")
    samples = value.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("GSM8K split manifest has no samples")
    return value


def _expected_experience_provenance(experience: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {
            "experience_id": experience.get("experience_id"),
            "target_episode_id": experience.get("target_episode_id"),
            "reference_episode_id": experience.get("reference_episode_id"),
            "source": experience.get("source"),
            "student": experience.get("student"),
            "rollout_configuration": experience.get("rollout_configuration"),
        }
    )


def load_v4_experiences(
    path: Path, *, split_manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    """Load raw V2 contrast pairs; reject teacher-bank or non-construction input."""

    manifest_samples = {
        str(item["sample_id"]): item
        for item in split_manifest["samples"]
        if item.get("logical_split") == "bank-source"
    }
    values: list[dict[str, Any]] = []
    seen_experience_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for raw in iter_jsonl(path):
        value = upgrade_verified_experience(raw)
        if value.get("schema_version") != EXPERIENCE_SCHEMA:
            raise ValueError("V4 requires verifier-backed V2 experience pairs")
        experience_id = str(value.get("experience_id", ""))
        sample_id = str(value.get("sample_id", ""))
        if not experience_id or experience_id in seen_experience_ids:
            raise ValueError(f"Missing or duplicate V4 experience ID: {experience_id!r}")
        if not sample_id or sample_id in seen_sample_ids:
            raise ValueError(f"Missing or duplicate V4 construction sample ID: {sample_id!r}")
        source = value.get("source", {})
        manifest_entry = manifest_samples.get(sample_id)
        if manifest_entry is None:
            raise ValueError(f"V4 construction sample is not bank-source: {sample_id}")
        if source.get("logical_split") != "bank-source":
            raise ValueError("V4 experience source is not bank-source")
        for field in ("dataset_split", "source_index", "question_sha256"):
            if source.get(field) != manifest_entry.get(field):
                raise ValueError(f"V4 experience source {field} differs from manifest")
        if source.get("split_manifest_sha256") != split_manifest.get(
            "manifest_sha256"
        ):
            raise ValueError("V4 experience split-manifest binding drifted")
        if value.get("outcome") != "verified_success" or value.get("reward") != 1.0:
            raise ValueError("V4 target trajectory is not a verified success")
        if value.get("target_verifier", {}).get("reward") != 1.0:
            raise ValueError("V4 target verifier did not accept the trajectory")
        if value.get("reference_verifier", {}).get("reward") != 0.0:
            raise ValueError("V4 reference trajectory is not a verified failure")
        if not value.get("trajectory") or not value.get("reference_trajectory"):
            raise ValueError("V4 experience is missing a contrast trajectory")
        if value.get("provenance_sha256") != _expected_experience_provenance(value):
            raise ValueError("V4 experience provenance hash mismatch")
        seen_experience_ids.add(experience_id)
        seen_sample_ids.add(sample_id)
        values.append(value)
    if len(values) < V4_MIN_CONSTRUCTION_EXAMPLES:
        raise ValueError("V4 construction has fewer than five verified experience pairs")
    return tuple(sorted(values, key=lambda item: str(item["experience_id"])))


def attach_official_solutions(
    experiences: Sequence[Mapping[str, Any]],
    *,
    split_manifest: Mapping[str, Any],
    dataset_revision: str,
    dataset: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Rejoin and authenticate official GSM8K solutions without persisting secrets."""

    expected_fingerprint = split_manifest.get("dataset", {}).get("train_fingerprint")
    actual_fingerprint = getattr(dataset, "_fingerprint", None)
    if expected_fingerprint and actual_fingerprint != expected_fingerprint:
        raise ValueError("V4 GSM8K train fingerprint differs from the split manifest")
    if split_manifest.get("dataset", {}).get("revision") != dataset_revision:
        raise ValueError("V4 official-solution revision mismatch")
    manifest_by_id = {
        str(item["sample_id"]): item for item in split_manifest["samples"]
    }
    result: list[dict[str, Any]] = []
    for experience in experiences:
        sample_id = str(experience["sample_id"])
        entry = manifest_by_id[sample_id]
        source = dataset[int(entry["source_index"])]
        question = str(source["question"]).strip()
        official_solution = str(source["answer"]).strip()
        if text_sha256(question) != entry.get("question_sha256"):
            raise ValueError(f"V4 official question hash mismatch: {sample_id}")
        if text_sha256(official_solution) != entry.get("answer_sha256"):
            raise ValueError(f"V4 official answer hash mismatch: {sample_id}")
        if question != str(experience["context"]).strip():
            raise ValueError(f"V4 experience question differs from dataset: {sample_id}")
        construction = {
            "experience_id": str(experience["experience_id"]),
            "sample_id": sample_id,
            "experience_type": str(experience["experience_type"]),
            "question": question,
            "official_solution": official_solution,
            "verified_success_trajectory": str(experience["trajectory"]).strip(),
            "verified_failure_trajectory": str(
                experience["reference_trajectory"]
            ).strip(),
            "target_verifier": experience["target_verifier"],
            "reference_verifier": experience["reference_verifier"],
            "source_provenance_sha256": str(experience["provenance_sha256"]),
            "source": experience["source"],
        }
        construction["construction_input_sha256"] = canonical_json_sha256(
            construction
        )
        result.append(construction)
    return tuple(result)


def repair_signature_messages(example: Mapping[str, Any]) -> list[dict[str, str]]:
    system = """You are the offline repair-signature curator for MemGen V4.
Return JSON only. Compare one verified-success trajectory, one paired
verified-failure trajectory, and the official solution for the same training
problem. Abstract a single grounded failure mechanism and its corresponding
repair operator. This is categorization, not bank writing.

Do not copy or paraphrase names, numbers, answers, equations, formulas, or the
story setting. Every text field must be reusable natural-language process text
and must contain no digits. Do not treat a different surface calculation as a
different mechanism. Reject the example when the contrast does not ground one
specific repair. Preserve the supplied experience_type exactly."""
    user = f"""Experience ID: {example['experience_id']}
Sample ID: {example['sample_id']}
Required experience_type: {example['experience_type']}

Construction question:
{example['question']}

Official reference solution:
{example['official_solution']}

Verified-success trajectory:
{example['verified_success_trajectory']}

Verified-failure trajectory:
{example['verified_failure_trajectory']}

Reference verifier diagnostics:
{json.dumps(example['reference_verifier'], ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "problem_structure": "instance-free structural description",
  "decision_point": "where the reasoning decision becomes necessary",
  "failure_mechanism": "one grounded undesired reasoning mechanism",
  "repair_operator": "one reusable corrective operation",
  "verification_operator": "one reusable check after the repair",
  "applicable": true,
  "rejection_reason": null
}}

If no specific grounded mechanism is visible, keep all five text fields as
brief instance-free descriptions, set applicable to false, and give a concise
rejection_reason. Do not invent an error merely because the final answer was
rejected."""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def cluster_messages(signatures: Sequence[V4RepairSignature]) -> list[dict[str, str]]:
    eligible = [item.to_dict() for item in signatures if item.applicable]
    system = """You are the offline repair-cluster curator for MemGen V4.
Return JSON only. Group repair signatures by the conjunction of failure
mechanism and repair operator. Surface story, object names, and broad GSM8K
topic are metadata, not cluster keys.

Do not mix experience_type values. Every admitted runtime cluster must contain
at least five distinct experience IDs. Select between five and ten
representatives per cluster, covering the cluster's structural variation.
Assign every supplied experience exactly once: either to one admitted cluster
or to rejected_experience_ids. Reject diffuse or ambiguous groups rather than
creating singleton memories. Cluster titles and descriptions must contain no
digits, equations, answer fragments, or instance-specific details. cluster_key
must use lowercase ASCII letters plus hyphen or underscore."""
    user = f"""Repair signatures:
{json.dumps(eligible, ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "schema_version": "{V4_CLUSTER_PLAN_SCHEMA}",
  "clusters": [
    {{
      "cluster_key": "canonical-key",
      "title": "brief reusable title",
      "failure_mechanism": "shared grounded mechanism",
      "repair_operator": "shared corrective operator",
      "scope_summary": "when the repair is applicable",
      "member_experience_ids": ["..."],
      "representative_experience_ids": ["..."]
    }}
  ],
  "rejected_experience_ids": ["..."],
  "rejection_notes": {{"experience-id": "brief reason"}}
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _representative_evidence(
    cluster: V4RepairCluster,
    *,
    examples_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for experience_id in cluster.representative_experience_ids:
        example = examples_by_id[experience_id]
        result.append(
            {
                "experience_id": experience_id,
                "sample_id": example["sample_id"],
                "question": example["question"],
                "official_solution": example["official_solution"],
                "verified_success_trajectory": example[
                    "verified_success_trajectory"
                ],
                "verified_failure_trajectory": example[
                    "verified_failure_trajectory"
                ],
                "reference_verifier": example["reference_verifier"],
            }
        )
    return result


def process_card_messages(
    cluster: V4RepairCluster,
    *,
    signatures_by_id: Mapping[str, V4RepairSignature],
    examples_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    signatures = [
        signatures_by_id[experience_id].to_dict()
        for experience_id in cluster.member_experience_ids
    ]
    evidence = _representative_evidence(cluster, examples_by_id=examples_by_id)
    system = """You are the offline process-card synthesizer for MemGen V4.
Return JSON only. Synthesize one target process card and one contrastive
reference process card from several independent training problems.

The target must describe only the shared desired process supported by official
solutions and verified-success trajectories. The reference must descriptively
summarize the recurring undesired process supported by paired verified-failure
trajectories; never phrase it as an instruction to make the error. Remove all
question text, names, story details, answers, numbers, equations, constants,
source-specific formulas, and solution-order traces. Every output text field
must contain no digits. Keep method selection, corrective action, applicability
boundaries, and verification. Do not combine multiple unrelated repairs."""
    user = f"""Frozen cluster:
{json.dumps(cluster.to_dict(), ensure_ascii=False, sort_keys=True)}

All member signatures:
{json.dumps(signatures, ensure_ascii=False, sort_keys=True)}

Representative construction evidence:
{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "target": {{
    "scope": "when this process is relevant",
    "diagnosis": "what structural reasoning risk is present",
    "action": "the reusable corrective process",
    "verification": "how to check the repaired reasoning",
    "do_not_use_when": "a concrete applicability boundary"
  }},
  "reference": {{
    "undesired_pattern": "descriptive recurring failed process",
    "failure_signal": "observable sign of that process",
    "failure_mechanism": "why that process fails",
    "contrast_boundary": "how it differs from the target process"
  }},
  "support_summary": "why the card is shared across the construction set",
  "target_reference_distinction": "the exact process-level contrast"
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def card_review_messages(
    cluster: V4RepairCluster,
    card: V4ProcessCard,
    *,
    signatures_by_id: Mapping[str, V4RepairSignature],
    examples_by_id: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    signatures = [
        signatures_by_id[experience_id].to_dict()
        for experience_id in cluster.member_experience_ids
    ]
    evidence = _representative_evidence(cluster, examples_by_id=examples_by_id)
    system = """You are a strict offline auditor for a MemGen V4 process card.
Return JSON only. Do not rewrite or repair the card. Check whether each target
claim is grounded in correct construction evidence, each reference claim is
grounded in failed trajectories, the card is process-only and transferable,
and target/reference express a real contrast. Reject any question-specific
detail, answer, number, equation, unsupported mechanism, or disguised wrong
instruction. Approve exactly when every component check is true; approved
reviews must have no issues, rejected reviews must list at least one issue."""
    user = f"""Cluster:
{json.dumps(cluster.to_dict(), ensure_ascii=False, sort_keys=True)}

Member signatures:
{json.dumps(signatures, ensure_ascii=False, sort_keys=True)}

Representative evidence:
{json.dumps(evidence, ensure_ascii=False, sort_keys=True)}

Candidate card:
{json.dumps(card.to_dict(), ensure_ascii=False, sort_keys=True)}

Return exactly:
{{
  "target_grounded": true,
  "reference_grounded": true,
  "process_only": true,
  "target_reference_distinct": true,
  "transferable": true,
  "leakage_free": true,
  "approve": true,
  "evidence": "concise audit rationale",
  "issues": []
}}"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    """Durably append one record without rewriting an ever-growing JSONL file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _client(args: argparse.Namespace, *, api_key: str, max_tokens: int) -> TeacherClient:
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
    )


def _signature_record(
    signature: V4RepairSignature,
    *,
    example: Mapping[str, Any],
    model: str,
    base_url: str,
    generation_status: str = "teacher_validated",
) -> dict[str, Any]:
    record = {
        "schema_version": SIGNATURE_RECORD_SCHEMA,
        "prompt_version": V4_SIGNATURE_PROMPT_VERSION,
        "created_at": utc_now(),
        "generation_status": generation_status,
        "teacher": {
            "model": model,
            "base_url": base_url,
            "temperature": 0.0,
            "thinking": V4_TEACHER_THINKING,
        },
        "construction_input_sha256": example["construction_input_sha256"],
        "signature": signature.to_dict(),
        "signature_sha256": signature.signature_sha256,
    }
    return record


def _rejected_signature_after_invalid_response(
    example: Mapping[str, Any],
) -> V4RepairSignature:
    """Create a non-applicable audit record without inventing a repair signature."""

    return V4RepairSignature(
        experience_id=str(example["experience_id"]),
        sample_id=str(example["sample_id"]),
        experience_type=str(example["experience_type"]),
        problem_structure="an example whose process abstraction was not validated",
        decision_point="before admitting the example into repair clustering",
        failure_mechanism="no validated transferable failure mechanism was recovered",
        repair_operator="exclude the unvalidated example from repair clustering",
        verification_operator=(
            "require a grounded schema compliant abstraction before admission"
        ),
        applicable=False,
        rejection_reason=(
            "teacher output remained outside the instance free schema after retries"
        ),
        source_provenance_sha256=str(example["source_provenance_sha256"]),
    )


def _card_record(
    cluster: V4RepairCluster,
    card: V4ProcessCard,
    *,
    construction_input_sha256: str,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    return {
        "schema_version": CARD_RECORD_SCHEMA,
        "prompt_version": V4_CARD_PROMPT_VERSION,
        "created_at": utc_now(),
        "teacher": {
            "model": model,
            "base_url": base_url,
            "temperature": 0.0,
            "thinking": V4_TEACHER_THINKING,
        },
        "cluster_key": cluster.cluster_key,
        "construction_input_sha256": construction_input_sha256,
        "card": card.to_dict(),
        "card_sha256": card.card_sha256,
    }


def _review_record(
    cluster: V4RepairCluster,
    card: V4ProcessCard,
    review: V4CardReview,
    *,
    construction_input_sha256: str,
    model: str,
    base_url: str,
) -> dict[str, Any]:
    return {
        "schema_version": REVIEW_RECORD_SCHEMA,
        "prompt_version": V4_CARD_REVIEW_PROMPT_VERSION,
        "created_at": utc_now(),
        "teacher": {
            "model": model,
            "base_url": base_url,
            "temperature": 0.0,
            "thinking": V4_TEACHER_THINKING,
        },
        "cluster_key": cluster.cluster_key,
        "construction_input_sha256": construction_input_sha256,
        "card_sha256": card.card_sha256,
        "review": review.to_dict(),
        "review_sha256": canonical_json_sha256(review.to_dict()),
    }


def _validate_frozen_cli(args: argparse.Namespace) -> V4ConstructionProfile:
    profile = V4ConstructionProfile(
        teacher_model=args.model,
        temperature=args.temperature,
        thinking=args.thinking,
    )
    for owner, value in (
        ("signature-max-tokens", args.signature_max_tokens),
        ("cluster-max-tokens", args.cluster_max_tokens),
        ("card-max-tokens", args.card_max_tokens),
        ("review-max-tokens", args.review_max_tokens),
    ):
        if value <= 0:
            raise ValueError(f"--{owner} must be positive")
    parsed_base_url = urlsplit(args.base_url)
    if (
        parsed_base_url.scheme not in {"http", "https"}
        or not parsed_base_url.hostname
        or parsed_base_url.username is not None
        or parsed_base_url.password is not None
        or parsed_base_url.query
        or parsed_base_url.fragment
    ):
        raise ValueError("V4 DeepSeek base URL is invalid or contains unsafe credentials")
    return profile


def main() -> None:
    args = parse_args()
    profile = _validate_frozen_cli(args)
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before V4 bank construction")

    split_manifest = _validate_split_manifest(
        args.split_manifest, dataset_revision=args.dataset_revision
    )
    experiences = load_v4_experiences(
        args.experiences, split_manifest=split_manifest
    )
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - server dependency path
        raise RuntimeError("datasets is required for V4 official-solution join") from exc
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=args.dataset_revision,
    )
    examples = attach_official_solutions(
        experiences,
        split_manifest=split_manifest,
        dataset_revision=args.dataset_revision,
        dataset=dataset,
    )
    examples_by_id = {str(item["experience_id"]): item for item in examples}

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    signatures_path = output_dir / "repair_signatures.jsonl"
    cluster_path = output_dir / "cluster_plan.json"
    cards_path = output_dir / "process_cards.jsonl"
    reviews_path = output_dir / "card_reviews.jsonl"
    rejected_path = output_dir / "rejected_clusters.jsonl"
    records_path = output_dir / "bank_records.jsonl"
    manifest_path = output_dir / "bank_manifest.json"
    profile_path = output_dir / "construction_profile.json"

    _write_json(
        profile_path,
        {
            "profile": profile.to_dict(),
            "profile_sha256": profile.profile_sha256,
            "teacher": {
                "model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "thinking": args.thinking,
            },
            "prompt_versions": {
                "signature": V4_SIGNATURE_PROMPT_VERSION,
                "cluster": V4_CLUSTER_PROMPT_VERSION,
                "card": V4_CARD_PROMPT_VERSION,
                "review": V4_CARD_REVIEW_PROMPT_VERSION,
            },
        },
    )

    compatible_signatures: dict[str, dict[str, Any]] = {}
    seen_resume_signature_ids: set[str] = set()
    # Nested IDs require an explicit compatibility pass.
    if args.resume and signatures_path.is_file():
        for record in iter_jsonl(signatures_path):
            signature_payload = record.get("signature", {})
            experience_id = str(signature_payload.get("experience_id", ""))
            if experience_id in seen_resume_signature_ids:
                raise ValueError("V4 resume signatures contain duplicate experience IDs")
            seen_resume_signature_ids.add(experience_id)
            example = examples_by_id.get(experience_id)
            if (
                example is not None
                and record.get("schema_version") == SIGNATURE_RECORD_SCHEMA
                and record.get("prompt_version") == V4_SIGNATURE_PROMPT_VERSION
                and record.get("teacher", {}).get("model") == args.model
                and record.get("teacher", {}).get("base_url") == args.base_url
                and record.get("teacher", {}).get("temperature") == args.temperature
                and record.get("teacher", {}).get("thinking") == args.thinking
                and record.get("construction_input_sha256")
                == example.get("construction_input_sha256")
            ):
                signature = parse_v4_repair_signature(
                    signature_payload,
                    experience_id=experience_id,
                    sample_id=str(example["sample_id"]),
                    experience_type=str(example["experience_type"]),
                    source_provenance_sha256=str(
                        example["source_provenance_sha256"]
                    ),
                )
                if record.get("signature_sha256") == signature.signature_sha256:
                    compatible_signatures[experience_id] = record

    signature_records: list[dict[str, Any]] = []
    # Compatible resume records are already held in memory. Rebuild the ordered
    # checkpoint once, then append one durable record per completed example.
    # Rewriting all prior records after every API call is quadratic at bank scale.
    _write_jsonl(signatures_path, ())
    with _client(
        args, api_key=api_key, max_tokens=args.signature_max_tokens
    ) as signature_client:
        for index, example in enumerate(examples, start=1):
            experience_id = str(example["experience_id"])
            record = compatible_signatures.get(experience_id)
            if record is not None and "generation_status" not in record:
                record = {**record, "generation_status": "teacher_validated"}
            if record is None:
                def parse_signature_response(content: str) -> dict[str, Any]:
                    candidate = _parse_json_object(content)
                    parse_v4_repair_signature(
                        candidate,
                        experience_id=experience_id,
                        sample_id=str(example["sample_id"]),
                        experience_type=str(example["experience_type"]),
                        source_provenance_sha256=str(
                            example["source_provenance_sha256"]
                        ),
                    )
                    return candidate

                try:
                    payload = signature_client.call(
                        repair_signature_messages(example),
                        response_parser=parse_signature_response,
                        request_label="v4-signature",
                        expose_parser_error=True,
                        repair_parser_errors=True,
                    )
                    signature = parse_v4_repair_signature(
                        payload,
                        experience_id=experience_id,
                        sample_id=str(example["sample_id"]),
                        experience_type=str(example["experience_type"]),
                        source_provenance_sha256=str(
                            example["source_provenance_sha256"]
                        ),
                    )
                    record = _signature_record(
                        signature,
                        example=example,
                        model=args.model,
                        base_url=args.base_url,
                    )
                except TeacherInvalidResponseError:
                    signature = _rejected_signature_after_invalid_response(example)
                    record = _signature_record(
                        signature,
                        example=example,
                        model=args.model,
                        base_url=args.base_url,
                        generation_status=(
                            "deterministic_rejection_after_invalid_teacher_response"
                        ),
                    )
                    print(
                        f"[v4-bank] signature rejected after invalid responses "
                        f"{experience_id}; continuing",
                        file=sys.stderr,
                        flush=True,
                    )
            signature_records.append(record)
            _append_jsonl(signatures_path, record)
            print(
                f"[v4-bank] signatures {index}/{len(examples)} {experience_id}",
                flush=True,
            )
    signatures = tuple(
        parse_v4_repair_signature(
            record["signature"],
            experience_id=str(record["signature"]["experience_id"]),
            sample_id=str(record["signature"]["sample_id"]),
            experience_type=str(record["signature"]["experience_type"]),
            source_provenance_sha256=str(
                record["signature"]["source_provenance_sha256"]
            ),
        )
        for record in signature_records
    )
    if sum(item.applicable for item in signatures) < V4_MIN_CONSTRUCTION_EXAMPLES:
        raise RuntimeError("DeepSeek produced fewer than five applicable V4 signatures")

    cluster_input_sha256 = canonical_json_sha256(
        [item.to_dict() for item in signatures if item.applicable]
    )
    cluster_payload: dict[str, Any] | None = None
    if args.resume and cluster_path.is_file():
        stored = json.loads(cluster_path.read_text(encoding="utf-8"))
        if (
            stored.get("schema_version") == V4_CLUSTER_PLAN_SCHEMA
            and stored.get("prompt_version") == V4_CLUSTER_PROMPT_VERSION
            and stored.get("teacher", {}).get("model") == args.model
            and stored.get("teacher", {}).get("base_url") == args.base_url
            and stored.get("teacher", {}).get("temperature") == args.temperature
            and stored.get("teacher", {}).get("thinking") == args.thinking
            and stored.get("construction_input_sha256") == cluster_input_sha256
            and isinstance(stored.get("payload"), dict)
            and stored.get("record_sha256")
            == canonical_json_sha256(
                {key: value for key, value in stored.items() if key != "record_sha256"}
            )
        ):
            parse_v4_cluster_plan(stored["payload"], signatures=signatures)
            cluster_payload = stored["payload"]
    if cluster_payload is None:
        def parse_cluster_response(content: str) -> dict[str, Any]:
            candidate = _parse_json_object(content)
            parse_v4_cluster_plan(candidate, signatures=signatures)
            return candidate

        with _client(
            args, api_key=api_key, max_tokens=args.cluster_max_tokens
        ) as cluster_client:
            cluster_payload = cluster_client.call(
                cluster_messages(signatures),
                response_parser=parse_cluster_response,
                request_label="v4-cluster",
                expose_parser_error=True,
                repair_parser_errors=True,
            )
        clusters, rejected_ids = parse_v4_cluster_plan(
            cluster_payload, signatures=signatures
        )
        cluster_record = {
            "schema_version": V4_CLUSTER_PLAN_SCHEMA,
            "prompt_version": V4_CLUSTER_PROMPT_VERSION,
            "created_at": utc_now(),
            "teacher": {
                "model": args.model,
                "base_url": args.base_url,
                "temperature": args.temperature,
                "thinking": args.thinking,
            },
            "construction_input_sha256": cluster_input_sha256,
            "payload": cluster_payload,
            "cluster_count": len(clusters),
            "rejected_experience_ids": list(rejected_ids),
        }
        cluster_record["record_sha256"] = canonical_json_sha256(cluster_record)
        _write_json(cluster_path, cluster_record)
    clusters, rejected_ids = parse_v4_cluster_plan(
        cluster_payload, signatures=signatures
    )
    if not clusters:
        raise RuntimeError("DeepSeek produced no five-example V4 repair clusters")

    signatures_by_id = {item.experience_id: item for item in signatures}
    existing_cards: dict[str, dict[str, Any]] = {}
    if args.resume and cards_path.is_file():
        for record in iter_jsonl(cards_path):
            cluster_key = str(record.get("cluster_key", ""))
            if record.get("schema_version") == CARD_RECORD_SCHEMA:
                if cluster_key in existing_cards:
                    raise ValueError("V4 resume cards contain duplicate cluster keys")
                existing_cards[cluster_key] = record
    existing_reviews: dict[str, dict[str, Any]] = {}
    if args.resume and reviews_path.is_file():
        for record in iter_jsonl(reviews_path):
            cluster_key = str(record.get("cluster_key", ""))
            if record.get("schema_version") == REVIEW_RECORD_SCHEMA:
                if cluster_key in existing_reviews:
                    raise ValueError("V4 resume reviews contain duplicate cluster keys")
                existing_reviews[cluster_key] = record

    card_records: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    rejected_clusters: list[dict[str, Any]] = []
    bank_records: list[dict[str, Any]] = []
    with _client(
        args, api_key=api_key, max_tokens=args.card_max_tokens
    ) as card_client, _client(
        args, api_key=api_key, max_tokens=args.review_max_tokens
    ) as review_client:
        for index, cluster in enumerate(clusters, start=1):
            construction_input_sha256 = canonical_json_sha256(
                {
                    "cluster": cluster.to_dict(),
                    "examples": {
                        experience_id: examples_by_id[experience_id][
                            "construction_input_sha256"
                        ]
                        for experience_id in cluster.member_experience_ids
                    },
                }
            )
            card_record = existing_cards.get(cluster.cluster_key)
            card: V4ProcessCard | None = None
            if (
                card_record is not None
                and card_record.get("prompt_version") == V4_CARD_PROMPT_VERSION
                and card_record.get("teacher", {}).get("model") == args.model
                and card_record.get("teacher", {}).get("base_url") == args.base_url
                and card_record.get("teacher", {}).get("temperature")
                == args.temperature
                and card_record.get("teacher", {}).get("thinking") == args.thinking
                and card_record.get("construction_input_sha256")
                == construction_input_sha256
            ):
                card = parse_v4_process_card(
                    card_record.get("card", {}), cluster_key=cluster.cluster_key
                )
                if card_record.get("card_sha256") != card.card_sha256:
                    card = None
            if card is None:
                def parse_card_response(content: str) -> dict[str, Any]:
                    candidate = _parse_json_object(content)
                    parse_v4_process_card(
                        candidate, cluster_key=cluster.cluster_key
                    )
                    return candidate

                payload = card_client.call(
                    process_card_messages(
                        cluster,
                        signatures_by_id=signatures_by_id,
                        examples_by_id=examples_by_id,
                    ),
                    response_parser=parse_card_response,
                    request_label="v4-card",
                    expose_parser_error=True,
                    repair_parser_errors=True,
                )
                card = parse_v4_process_card(
                    payload, cluster_key=cluster.cluster_key
                )
                card_record = _card_record(
                    cluster,
                    card,
                    construction_input_sha256=construction_input_sha256,
                    model=args.model,
                    base_url=args.base_url,
                )
            card_records.append(card_record)
            _write_jsonl(cards_path, card_records)

            review_record = existing_reviews.get(cluster.cluster_key)
            review: V4CardReview | None = None
            if (
                review_record is not None
                and review_record.get("prompt_version")
                == V4_CARD_REVIEW_PROMPT_VERSION
                and review_record.get("teacher", {}).get("model") == args.model
                and review_record.get("teacher", {}).get("base_url") == args.base_url
                and review_record.get("teacher", {}).get("temperature")
                == args.temperature
                and review_record.get("teacher", {}).get("thinking") == args.thinking
                and review_record.get("construction_input_sha256")
                == construction_input_sha256
                and review_record.get("card_sha256") == card.card_sha256
            ):
                review = parse_v4_card_review(
                    review_record.get("review", {}), cluster_key=cluster.cluster_key
                )
                if review_record.get("review_sha256") != canonical_json_sha256(
                    review.to_dict()
                ):
                    review = None
            if review is None:
                def parse_review_response(content: str) -> dict[str, Any]:
                    candidate = _parse_json_object(content)
                    parse_v4_card_review(
                        candidate, cluster_key=cluster.cluster_key
                    )
                    return candidate

                payload = review_client.call(
                    card_review_messages(
                        cluster,
                        card,
                        signatures_by_id=signatures_by_id,
                        examples_by_id=examples_by_id,
                    ),
                    response_parser=parse_review_response,
                    request_label="v4-card-review",
                    expose_parser_error=True,
                    repair_parser_errors=True,
                )
                review = parse_v4_card_review(
                    payload, cluster_key=cluster.cluster_key
                )
                review_record = _review_record(
                    cluster,
                    card,
                    review,
                    construction_input_sha256=construction_input_sha256,
                    model=args.model,
                    base_url=args.base_url,
                )
            review_records.append(review_record)
            _write_jsonl(reviews_path, review_records)

            if review.approve:
                bank_records.append(
                    build_v4_bank_record(
                        cluster=cluster,
                        card=card,
                        review=review,
                        signatures=signatures,
                        construction_input_sha256=construction_input_sha256,
                        profile=profile,
                    )
                )
            else:
                rejected_clusters.append(
                    {
                        "cluster_key": cluster.cluster_key,
                        "card_sha256": card.card_sha256,
                        "review": review.to_dict(),
                    }
                )
            print(
                f"[v4-bank] cards {index}/{len(clusters)} {cluster.cluster_key} "
                f"approved={review.approve}",
                flush=True,
            )

    _write_jsonl(rejected_path, rejected_clusters)
    _write_jsonl(records_path, bank_records)
    if not bank_records:
        raise RuntimeError("No DeepSeek-generated V4 process cards passed review")
    manifest = build_v4_bank_manifest(
        records=bank_records,
        profile=profile,
        inputs={
            "experiences_path": str(args.experiences.resolve()),
            "experiences_sha256": file_sha256(args.experiences),
            "split_manifest_path": str(args.split_manifest.resolve()),
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "split_manifest_logical_sha256": split_manifest["manifest_sha256"],
            "dataset_revision": args.dataset_revision,
            "construction_example_count": len(examples),
            "teacher_invalid_signature_ids": [
                str(record["signature"]["experience_id"])
                for record in signature_records
                if record.get("generation_status")
                == "deterministic_rejection_after_invalid_teacher_response"
            ],
            "cluster_input_sha256": cluster_input_sha256,
            "rejected_signature_ids": list(rejected_ids),
            "repository": _repository_state(),
        },
        teacher={
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "thinking": args.thinking,
            "prompt_versions": {
                "signature": V4_SIGNATURE_PROMPT_VERSION,
                "cluster": V4_CLUSTER_PROMPT_VERSION,
                "card": V4_CARD_PROMPT_VERSION,
                "review": V4_CARD_REVIEW_PROMPT_VERSION,
            },
        },
    )
    manifest["created_at"] = utc_now()
    # created_at is intentionally outside the authenticated logical manifest.
    _write_json(manifest_path, manifest)
    print(
        f"[v4-bank] complete records={len(bank_records)} manifest={manifest_path} "
        f"sha256={manifest['manifest_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4-bank] error: {exc}", file=sys.stderr)
        raise
