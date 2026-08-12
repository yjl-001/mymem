"""Pure helpers for the verifier-backed Phase 1 experience pipeline.

The model and teacher entry points intentionally live in ``scripts/``.  This
module contains deterministic transformations that can be tested without a GPU,
network access, or API credentials.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence


SPLIT_MANIFEST_SCHEMA = "gsm8k-split-manifest-v1"
ROLLOUT_SCHEMA = "verified-student-rollout-v1"
EXPERIENCE_SCHEMA = "verified-contrastive-experience-v1"
TEACHER_BANK_REQUIRED_FIELDS = {
    "target": (
        "situation_signature",
        "transferable_decision",
        "verification_rule",
        "applicability_boundary",
        "confidence",
    ),
    "reference": (
        "competing_pattern",
        "failure_signal",
        "failure_mechanism",
        "non_reuse_boundary",
        "confidence",
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            yield value


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _sample_entry(
    *, dataset_split: str, logical_split: str, index: int, item: Mapping[str, Any]
) -> dict[str, Any]:
    question = str(item["question"]).strip()
    answer = str(item["answer"]).strip()
    question_hash = text_sha256(question)
    return {
        "sample_id": f"gsm8k-{dataset_split}-{index}-{question_hash[:12]}",
        "logical_split": logical_split,
        "dataset_split": dataset_split,
        "source_index": index,
        "question_sha256": question_hash,
        "answer_sha256": text_sha256(answer),
    }


def create_gsm8k_split_manifest(
    train_records: Sequence[Mapping[str, Any]],
    test_records: Sequence[Mapping[str, Any]],
    *,
    bank_source_size: int,
    calibration_val_size: int,
    seed: int,
    dataset_revision: str,
    train_fingerprint: str | None = None,
    test_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Create stable, disjoint GSM8K splits without copying question text.

    Official GSM8K test examples are always assigned to ``final-test``.  Train
    examples are shuffled once, then assigned to ``bank-source``,
    ``calibration-val``, and the optional remainder ``dev-test``.
    """

    train_count = len(train_records)
    if bank_source_size <= 0:
        raise ValueError("bank_source_size must be positive")
    if calibration_val_size <= 0:
        raise ValueError("calibration_val_size must be positive")
    if bank_source_size + calibration_val_size > train_count:
        raise ValueError(
            "bank_source_size + calibration_val_size exceeds GSM8K train size"
        )

    indices = list(range(train_count))
    random.Random(seed).shuffle(indices)
    bank_indices = indices[:bank_source_size]
    calibration_indices = indices[
        bank_source_size : bank_source_size + calibration_val_size
    ]
    dev_indices = indices[bank_source_size + calibration_val_size :]

    entries: list[dict[str, Any]] = []
    for logical_split, selected in (
        ("bank-source", bank_indices),
        ("calibration-val", calibration_indices),
        ("dev-test", dev_indices),
    ):
        entries.extend(
            _sample_entry(
                dataset_split="train",
                logical_split=logical_split,
                index=index,
                item=train_records[index],
            )
            for index in selected
        )
    entries.extend(
        _sample_entry(
            dataset_split="test",
            logical_split="final-test",
            index=index,
            item=item,
        )
        for index, item in enumerate(test_records)
    )

    question_hashes: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        question_hashes[entry["logical_split"]].add(entry["question_sha256"])
    split_names = sorted(question_hashes)
    overlaps = {
        f"{left}::{right}": sorted(question_hashes[left] & question_hashes[right])
        for offset, left in enumerate(split_names)
        for right in split_names[offset + 1 :]
        if question_hashes[left] & question_hashes[right]
    }
    if overlaps:
        raise ValueError(f"GSM8K split leakage detected: {overlaps}")

    manifest = {
        "schema_version": SPLIT_MANIFEST_SCHEMA,
        "created_at": utc_now(),
        "dataset": {
            "name": "openai/gsm8k",
            "configuration": "main",
            "revision": dataset_revision,
            "train_fingerprint": train_fingerprint,
            "test_fingerprint": test_fingerprint,
            "train_size": train_count,
            "test_size": len(test_records),
        },
        "policy": {
            "seed": seed,
            "bank_source_size": bank_source_size,
            "calibration_val_size": calibration_val_size,
            "dev_test_size": len(dev_indices),
            "final_test_source": "official-test",
        },
        "counts": {
            name: sum(entry["logical_split"] == name for entry in entries)
            for name in ("bank-source", "calibration-val", "dev-test", "final-test")
        },
        "overlap_check": {"passed": True, "overlap_count": 0},
        "samples": entries,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(
        {
            key: value
            for key, value in manifest.items()
            if key not in {"created_at", "manifest_sha256"}
        }
    )
    return manifest


def _normalize_trajectory(value: str) -> str:
    return " ".join(value.lower().split())


def build_verified_experiences(
    rollout_records: Iterable[Mapping[str, Any]],
    *,
    max_pairs_per_sample: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pair verifier-backed success and failure episodes from the same prompt."""

    if max_pairs_per_sample <= 0:
        raise ValueError("max_pairs_per_sample must be positive")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    seen_episode_ids: set[str] = set()
    total_rollouts = 0
    for record in rollout_records:
        total_rollouts += 1
        if record.get("schema_version") != ROLLOUT_SCHEMA:
            raise ValueError("Unexpected rollout schema_version")
        episode_id = str(record.get("episode_id", ""))
        if not episode_id or episode_id in seen_episode_ids:
            raise ValueError(f"Missing or duplicate episode_id: {episode_id!r}")
        seen_episode_ids.add(episode_id)
        source = record.get("source", {})
        if source.get("logical_split") != "bank-source":
            raise ValueError(f"Rollout {episode_id} is not from bank-source")
        reward = record.get("reward")
        outcome = record.get("outcome")
        if isinstance(reward, bool) or reward not in {0.0, 1.0}:
            raise ValueError(f"Rollout {episode_id} has non-binary verifier reward")
        if outcome not in {"verified_success", "verified_failure"}:
            raise ValueError(f"Rollout {episode_id} has invalid verifier outcome")
        if (reward == 1.0) != (outcome == "verified_success"):
            raise ValueError(f"Rollout {episode_id} has inconsistent success label")
        if (reward == 0.0) != (outcome == "verified_failure"):
            raise ValueError(f"Rollout {episode_id} has inconsistent failure label")
        grouped[str(record["sample_id"])].append(record)

    experiences: list[dict[str, Any]] = []
    samples_without_contrast = 0
    success_count = 0
    failure_count = 0
    for sample_id in sorted(grouped):
        records = sorted(grouped[sample_id], key=lambda item: str(item["episode_id"]))
        successes = [item for item in records if item["outcome"] == "verified_success"]
        failures = [item for item in records if item["outcome"] == "verified_failure"]
        success_count += len(successes)
        failure_count += len(failures)
        if not successes or not failures:
            samples_without_contrast += 1
            continue

        pair_count = min(max_pairs_per_sample, max(len(successes), len(failures)))
        for pair_index in range(pair_count):
            target = successes[pair_index % len(successes)]
            reference = failures[pair_index % len(failures)]
            if target["source"] != reference["source"]:
                raise ValueError(f"Source provenance mismatch within {sample_id}")
            if target["context"] != reference["context"]:
                raise ValueError(f"Context mismatch within {sample_id}")
            if target["student"] != reference["student"]:
                raise ValueError(f"Student revision mismatch within {sample_id}")
            if _normalize_trajectory(str(target["trajectory"])) == _normalize_trajectory(
                str(reference["trajectory"])
            ):
                continue
            experience_id = f"{sample_id}-contrast-{pair_index}"
            experience = {
                "schema_version": EXPERIENCE_SCHEMA,
                "experience_id": experience_id,
                "sample_id": sample_id,
                "source": dict(target["source"]),
                "context": target["context"],
                "trajectory": target["trajectory"],
                "reference_trajectory": reference["trajectory"],
                "outcome": "verified_success",
                "reward": 1.0,
                "feedback": target["verifier"]["feedback"],
                "reference_evidence": "verified_failure",
                "target_episode_id": target["episode_id"],
                "reference_episode_id": reference["episode_id"],
                "target_verifier": dict(target["verifier"]),
                "reference_verifier": dict(reference["verifier"]),
                "student": dict(target["student"]),
                "rollout_configuration": {
                    "target": dict(target["rollout_configuration"]),
                    "reference": dict(reference["rollout_configuration"]),
                },
                "created_at": utc_now(),
            }
            experience["provenance_sha256"] = canonical_json_sha256(
                {
                    "experience_id": experience_id,
                    "target_episode_id": target["episode_id"],
                    "reference_episode_id": reference["episode_id"],
                    "source": target["source"],
                    "student": target["student"],
                    "rollout_configuration": {
                        "target": target["rollout_configuration"],
                        "reference": reference["rollout_configuration"],
                    },
                }
            )
            experiences.append(experience)

    report = {
        "schema_version": "verified-experience-build-report-v1",
        "created_at": utc_now(),
        "total_rollouts": total_rollouts,
        "verified_success_rollouts": success_count,
        "verified_failure_rollouts": failure_count,
        "sample_count": len(grouped),
        "samples_without_success_failure_contrast": samples_without_contrast,
        "verified_experience_count": len(experiences),
        "max_pairs_per_sample": max_pairs_per_sample,
    }
    return experiences, report


_WORD_RE = re.compile(r"[a-z]+")
_INSTANCE_LITERAL_RE = re.compile(r"(?:\\boxed|\\frac|\d)")


def _word_set(value: str) -> set[str]:
    return set(_WORD_RE.findall(value.lower()))


def _jaccard(left: str, right: str) -> float:
    left_words = _word_set(left)
    right_words = _word_set(right)
    if not left_words and not right_words:
        return 1.0
    return len(left_words & right_words) / max(len(left_words | right_words), 1)


def audit_teacher_record(
    record: Mapping[str, Any],
    experience: Mapping[str, Any],
) -> list[str]:
    """Return machine-checkable quality-gate rejection reasons."""

    reasons: list[str] = []
    if record.get("reference_evidence") != "verified_failure":
        reasons.append("reference_not_verified_failure")
    if experience.get("source", {}).get("logical_split") != "bank-source":
        reasons.append("source_not_bank_source")
    expected_ids = {
        "target": experience.get("target_episode_id"),
        "reference": experience.get("reference_episode_id"),
    }
    if record.get("source_episode_ids") != expected_ids:
        reasons.append("source_episode_ids_mismatch")
    if record.get("experience_id") != experience.get("experience_id"):
        reasons.append("experience_id_mismatch")
    if record.get("provenance_sha256") != experience.get("provenance_sha256"):
        reasons.append("provenance_hash_mismatch")
    expected_provenance_hash = canonical_json_sha256(
        {
            "experience_id": experience.get("experience_id"),
            "target_episode_id": experience.get("target_episode_id"),
            "reference_episode_id": experience.get("reference_episode_id"),
            "source": experience.get("source"),
            "student": experience.get("student"),
            "rollout_configuration": experience.get("rollout_configuration"),
        }
    )
    if experience.get("provenance_sha256") != expected_provenance_hash:
        reasons.append("experience_provenance_hash_invalid")
    for field in (
        "source",
        "student",
        "rollout_configuration",
        "target_verifier",
        "reference_verifier",
    ):
        if record.get(field) != experience.get(field):
            reasons.append(f"{field}_mismatch")

    bank = record.get("bank")
    if not isinstance(bank, Mapping):
        return reasons + ["missing_bank_object"]
    for section, fields in TEACHER_BANK_REQUIRED_FIELDS.items():
        value = bank.get(section)
        if not isinstance(value, Mapping):
            reasons.append(f"missing_{section}_object")
            continue
        for field in fields:
            field_value = value.get(field)
            if field == "confidence":
                if not isinstance(field_value, (int, float)) or isinstance(field_value, bool):
                    reasons.append(f"invalid_{section}_confidence")
                elif not 0.0 <= float(field_value) <= 1.0:
                    reasons.append(f"invalid_{section}_confidence")
            elif not isinstance(field_value, str) or not field_value.strip():
                reasons.append(f"missing_{section}_{field}")

    quality = bank.get("quality")
    if not isinstance(quality, Mapping):
        reasons.append("missing_teacher_quality_mark")
    else:
        for field in (
            "target_supported",
            "reference_supported",
            "target_reference_distinct",
            "contains_instance_specific_details",
        ):
            if not isinstance(quality.get(field), bool):
                reasons.append(f"invalid_quality_{field}")
        if quality.get("target_supported") is not True:
            reasons.append("teacher_marks_target_unsupported")
        if quality.get("reference_supported") is not True:
            reasons.append("teacher_marks_reference_unsupported")
        if quality.get("target_reference_distinct") is not True:
            reasons.append("teacher_marks_target_reference_equivalent")
        if quality.get("contains_instance_specific_details") is not False:
            reasons.append("teacher_marks_instance_specific_details")
        issues = quality.get("issues")
        if not isinstance(issues, list):
            reasons.append("invalid_quality_issues")
        elif issues:
            reasons.append("teacher_reports_quality_issues")

    target_text = " ".join(
        str(bank.get("target", {}).get(field, ""))
        for field in TEACHER_BANK_REQUIRED_FIELDS["target"]
        if field != "confidence"
    )
    reference_text = " ".join(
        str(bank.get("reference", {}).get(field, ""))
        for field in TEACHER_BANK_REQUIRED_FIELDS["reference"]
        if field != "confidence"
    )
    if _jaccard(target_text, reference_text) >= 0.8:
        reasons.append("target_reference_text_too_similar")
    if _INSTANCE_LITERAL_RE.search(target_text) or _INSTANCE_LITERAL_RE.search(reference_text):
        reasons.append("instance_specific_literal_detected")
    return sorted(set(reasons))


def summarize_human_review(
    review_records: Iterable[Mapping[str, Any]],
    *,
    required_sample_size: int = 30,
    required_agreement: float = 0.9,
) -> dict[str, Any]:
    """Validate the human worksheet and compute the Phase 1 acceptance result."""

    if required_sample_size <= 0:
        raise ValueError("required_sample_size must be positive")
    if not 0.0 <= required_agreement <= 1.0:
        raise ValueError("required_agreement must be in [0, 1]")
    fields = (
        "target_supported",
        "reference_supported",
        "target_reference_distinct",
        "factually_consistent",
    )
    records = list(review_records)
    incomplete_ids: list[str] = []
    passing_ids: list[str] = []
    failing_ids: list[str] = []
    for record in records:
        experience_id = str(record.get("experience_id", ""))
        review = record.get("human_review")
        if not isinstance(review, Mapping) or any(
            not isinstance(review.get(field), bool) for field in fields
        ):
            incomplete_ids.append(experience_id)
            continue
        if all(review[field] is True for field in fields):
            passing_ids.append(experience_id)
        else:
            failing_ids.append(experience_id)
    completed_count = len(passing_ids) + len(failing_ids)
    agreement = len(passing_ids) / completed_count if completed_count else 0.0
    passed = (
        len(records) >= required_sample_size
        and not incomplete_ids
        and completed_count >= required_sample_size
        and agreement >= required_agreement
    )
    return {
        "schema_version": "phase1-human-review-result-v1",
        "created_at": utc_now(),
        "required_sample_size": required_sample_size,
        "required_agreement": required_agreement,
        "worksheet_record_count": len(records),
        "completed_count": completed_count,
        "passing_count": len(passing_ids),
        "failing_count": len(failing_ids),
        "agreement": agreement,
        "incomplete_experience_ids": incomplete_ids,
        "failing_experience_ids": failing_ids,
        "passed": passed,
    }
