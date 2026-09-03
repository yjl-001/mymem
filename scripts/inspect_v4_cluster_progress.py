#!/usr/bin/env python3
"""Inspect V4 map/reduce checkpoints without API calls or model loading.

The report reconstructs the deterministic prototype state at the last fully
completed reduce round.  It never mutates construction checkpoints and writes
only the explicitly requested report file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import heapq
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256, iter_jsonl
from memgen.experience.v4_bank import (
    V4_CLUSTER_PROMPT_VERSION,
    V4_MIN_CONSTRUCTION_EXAMPLES,
    V4_SIGNATURE_PROMPT_VERSION,
    V4ConstructionProfile,
    V4RepairSignature,
    parse_v4_repair_signature,
)
from scripts.build_v4_repair_bank import (
    CLUSTER_UNIT_RECORD_SCHEMA,
    MAX_CLUSTER_REDUCE_ROUNDS,
    SIGNATURE_RECORD_SCHEMA,
    _bounded_batches,
    _load_cluster_unit_records,
    _merge_exact_prototypes,
    _semantic_sort_key,
    _signature_sort_key,
    parse_cluster_map_payload,
    parse_cluster_reduce_payload,
)


REPORT_SCHEMA = "memgen-v4-cluster-progress-report-v1"
DEFAULT_TOP_LIMIT = 50
DEFAULT_SAMPLE_LIMIT = 100
DEFAULT_NEAR_DUPLICATE_LIMIT = 100
_WORD_RE = re.compile(r"[a-z]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "each",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "then",
        "to",
        "when",
        "with",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--construction-dir",
        type=Path,
        required=True,
        help="Directory containing V4 construction checkpoints.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Report path; defaults to "
            "CONSTRUCTION_DIR/reduce_progress_report.json."
        ),
    )
    parser.add_argument("--top-limit", type=int, default=DEFAULT_TOP_LIMIT)
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument(
        "--near-duplicate-limit",
        type=int,
        default=DEFAULT_NEAR_DUPLICATE_LIMIT,
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Missing V4 checkpoint file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _repository_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_profile(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    raw_profile = value.get("profile")
    if not isinstance(raw_profile, Mapping):
        raise ValueError("V4 construction profile is missing profile")
    profile = V4ConstructionProfile(**dict(raw_profile))
    if value.get("profile_sha256") != profile.profile_sha256:
        raise ValueError("V4 construction profile hash mismatch")
    teacher = value.get("teacher")
    if not isinstance(teacher, Mapping):
        raise ValueError("V4 construction profile is missing teacher")
    if teacher.get("model") != profile.teacher_model:
        raise ValueError("V4 construction profile teacher model mismatch")
    if teacher.get("temperature") != profile.temperature:
        raise ValueError("V4 construction profile temperature mismatch")
    if teacher.get("thinking") != profile.thinking:
        raise ValueError("V4 construction profile thinking mismatch")
    prompt_versions = value.get("prompt_versions")
    if not isinstance(prompt_versions, Mapping):
        raise ValueError("V4 construction profile is missing prompt versions")
    if prompt_versions.get("signature") != V4_SIGNATURE_PROMPT_VERSION:
        raise ValueError("V4 signature prompt version is not inspectable by this revision")
    if prompt_versions.get("cluster") != V4_CLUSTER_PROMPT_VERSION:
        raise ValueError("V4 cluster prompt version is not inspectable by this revision")
    clustering = value.get("clustering")
    if not isinstance(clustering, Mapping):
        raise ValueError("V4 construction profile is missing clustering")
    if clustering.get("method") != "bounded_map_reduce":
        raise ValueError("Unexpected V4 clustering method")
    for field in ("map_batch_size", "reduce_batch_size", "max_reduce_rounds"):
        item = clustering.get(field)
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            raise ValueError(f"V4 clustering {field} must be a positive integer")
    if clustering["max_reduce_rounds"] != MAX_CLUSTER_REDUCE_ROUNDS:
        raise ValueError("V4 reduce-round bound differs from this code revision")
    return value


def _load_signatures(
    path: Path,
    *,
    profile: Mapping[str, Any],
) -> tuple[V4RepairSignature, ...]:
    teacher = profile["teacher"]
    signatures: list[V4RepairSignature] = []
    seen: set[str] = set()
    for index, record in enumerate(iter_jsonl(path)):
        if record.get("schema_version") != SIGNATURE_RECORD_SCHEMA:
            raise ValueError(f"Unexpected signature record schema at row {index}")
        if record.get("prompt_version") != V4_SIGNATURE_PROMPT_VERSION:
            raise ValueError(f"Unexpected signature prompt version at row {index}")
        if record.get("teacher") != dict(teacher):
            raise ValueError(f"Signature teacher binding mismatch at row {index}")
        payload = record.get("signature")
        if not isinstance(payload, Mapping):
            raise ValueError(f"Signature record {index} is missing signature")
        experience_id = str(payload.get("experience_id", ""))
        if experience_id in seen:
            raise ValueError(f"Duplicate signature experience ID: {experience_id}")
        seen.add(experience_id)
        signature = parse_v4_repair_signature(
            payload,
            experience_id=experience_id,
            sample_id=str(payload.get("sample_id", "")),
            experience_type=str(payload.get("experience_type", "")),
            source_provenance_sha256=str(
                payload.get("source_provenance_sha256", "")
            ),
        )
        if record.get("signature_sha256") != signature.signature_sha256:
            raise ValueError(f"Signature hash mismatch: {experience_id}")
        signatures.append(signature)
    if not signatures:
        raise ValueError("V4 signature checkpoint is empty")
    return tuple(signatures)


def _validate_unit_record(
    record: Mapping[str, Any],
    *,
    stage: str,
    unit_id: str,
    input_value: Any,
    teacher: Mapping[str, Any],
) -> None:
    if record.get("schema_version") != CLUSTER_UNIT_RECORD_SCHEMA:
        raise ValueError(f"Unexpected V4 cluster-unit schema: {unit_id}")
    if record.get("stage") != stage or record.get("unit_id") != unit_id:
        raise ValueError(f"V4 cluster-unit identity mismatch: {unit_id}")
    if record.get("prompt_version") != V4_CLUSTER_PROMPT_VERSION:
        raise ValueError(f"V4 cluster-unit prompt mismatch: {unit_id}")
    if record.get("teacher") != dict(teacher):
        raise ValueError(f"V4 cluster-unit teacher mismatch: {unit_id}")
    expected_input_hash = canonical_json_sha256(input_value)
    if record.get("construction_input_sha256") != expected_input_hash:
        raise ValueError(f"V4 cluster-unit input hash mismatch: {unit_id}")


def _prototype_row(
    prototype: Mapping[str, Any],
    *,
    signatures_by_id: Mapping[str, V4RepairSignature],
    include_members: bool,
) -> dict[str, Any]:
    members = tuple(sorted(str(item) for item in prototype["member_experience_ids"]))
    distinct_samples = tuple(
        sorted({signatures_by_id[item].sample_id for item in members})
    )
    row: dict[str, Any] = {
        "prototype_id": str(prototype["prototype_id"]),
        "experience_type": str(prototype["experience_type"]),
        "title": str(prototype["title"]),
        "failure_mechanism": str(prototype["failure_mechanism"]),
        "repair_operator": str(prototype["repair_operator"]),
        "scope_summary": str(prototype["scope_summary"]),
        "member_experience_count": len(members),
        "distinct_sample_count": len(distinct_samples),
    }
    if include_members:
        row["member_experience_ids"] = list(members)
        row["distinct_sample_ids"] = list(distinct_samples)
    return row


def _support_bucket(distinct_sample_count: int) -> str:
    if distinct_sample_count == 1:
        return "one"
    if distinct_sample_count <= 4:
        return "two_to_four"
    if distinct_sample_count <= 9:
        return "five_to_nine"
    if distinct_sample_count <= 19:
        return "ten_to_nineteen"
    return "twenty_or_more"


def _state_summary(
    prototypes: Sequence[Mapping[str, Any]],
    *,
    signatures_by_id: Mapping[str, V4RepairSignature],
) -> dict[str, Any]:
    buckets = {
        name: {
            "prototype_count": 0,
            "member_experience_count": 0,
            "distinct_sample_count_sum": 0,
        }
        for name in (
            "one",
            "two_to_four",
            "five_to_nine",
            "ten_to_nineteen",
            "twenty_or_more",
        )
    }
    by_type: dict[str, dict[str, Any]] = {}
    all_members: list[str] = []
    qualified_count = 0
    for prototype in prototypes:
        row = _prototype_row(
            prototype,
            signatures_by_id=signatures_by_id,
            include_members=False,
        )
        bucket = _support_bucket(row["distinct_sample_count"])
        buckets[bucket]["prototype_count"] += 1
        buckets[bucket]["member_experience_count"] += row[
            "member_experience_count"
        ]
        buckets[bucket]["distinct_sample_count_sum"] += row[
            "distinct_sample_count"
        ]
        experience_type = row["experience_type"]
        type_value = by_type.setdefault(
            experience_type,
            {
                "prototype_count": 0,
                "member_experience_count": 0,
                "qualified_prototype_count": 0,
                "support_histogram": {
                    name: 0 for name in buckets
                },
            },
        )
        type_value["prototype_count"] += 1
        type_value["member_experience_count"] += row["member_experience_count"]
        type_value["support_histogram"][bucket] += 1
        if row["distinct_sample_count"] >= V4_MIN_CONSTRUCTION_EXAMPLES:
            qualified_count += 1
            type_value["qualified_prototype_count"] += 1
        all_members.extend(str(item) for item in prototype["member_experience_ids"])
    if len(set(all_members)) != len(all_members):
        raise ValueError("Reconstructed V4 prototype state has overlapping memberships")
    logical_state = [
        {
            **dict(prototype),
            "member_experience_ids": sorted(
                str(item) for item in prototype["member_experience_ids"]
            ),
        }
        for prototype in sorted(
            prototypes,
            key=lambda item: str(item["prototype_id"]),
        )
    ]
    return {
        "prototype_count": len(prototypes),
        "member_experience_count": len(all_members),
        "qualified_prototype_count": qualified_count,
        "support_histogram": buckets,
        "by_experience_type": {
            key: by_type[key] for key in sorted(by_type)
        },
        "prototype_state_sha256": canonical_json_sha256(logical_state),
    }


def _deterministic_sample(
    prototypes: Sequence[Mapping[str, Any]],
    *,
    signatures_by_id: Mapping[str, V4RepairSignature],
    predicate: Any,
    seed: str,
    limit: int,
) -> list[dict[str, Any]]:
    eligible = []
    for prototype in prototypes:
        row = _prototype_row(
            prototype,
            signatures_by_id=signatures_by_id,
            include_members=True,
        )
        if predicate(row["distinct_sample_count"]):
            eligible.append(row)
    eligible.sort(
        key=lambda row: (
            canonical_json_sha256(
                {"seed": seed, "prototype_id": row["prototype_id"]}
            ),
            row["prototype_id"],
        )
    )
    return eligible[:limit]


def _prototype_tokens(prototype: Mapping[str, Any]) -> frozenset[str]:
    text = " ".join(
        str(prototype[field]).lower()
        for field in (
            "failure_mechanism",
            "repair_operator",
            "scope_summary",
        )
    )
    return frozenset(
        token for token in _WORD_RE.findall(text) if token not in _STOPWORDS
    )


def _lexical_near_duplicate_diagnostics(
    prototypes: Sequence[Mapping[str, Any]],
    *,
    limit: int,
) -> dict[str, Any]:
    if not prototypes:
        return {
            "method": "word_token_jaccard_diagnostic_only",
            "best_neighbor_histogram": {},
            "top_pairs": [],
        }
    tokens = [_prototype_tokens(item) for item in prototypes]
    best_scores = [0.0] * len(prototypes)
    best_neighbors: list[int | None] = [None] * len(prototypes)
    top_heap: list[tuple[float, str, str, int, int]] = []
    by_type: dict[str, list[int]] = {}
    for index, prototype in enumerate(prototypes):
        by_type.setdefault(str(prototype["experience_type"]), []).append(index)
    for indices in by_type.values():
        for offset, left_index in enumerate(indices):
            left_tokens = tokens[left_index]
            for right_index in indices[offset + 1 :]:
                right_tokens = tokens[right_index]
                union_count = len(left_tokens | right_tokens)
                score = (
                    len(left_tokens & right_tokens) / union_count
                    if union_count
                    else 0.0
                )
                if score > best_scores[left_index]:
                    best_scores[left_index] = score
                    best_neighbors[left_index] = right_index
                if score > best_scores[right_index]:
                    best_scores[right_index] = score
                    best_neighbors[right_index] = left_index
                if limit <= 0:
                    continue
                left_id = str(prototypes[left_index]["prototype_id"])
                right_id = str(prototypes[right_index]["prototype_id"])
                item = (score, left_id, right_id, left_index, right_index)
                if len(top_heap) < limit:
                    heapq.heappush(top_heap, item)
                elif item > top_heap[0]:
                    heapq.heapreplace(top_heap, item)

    histogram = {
        "at_least_zero_point_nine": 0,
        "zero_point_seven_five_to_zero_point_nine": 0,
        "zero_point_five_to_zero_point_seven_five": 0,
        "below_zero_point_five": 0,
        "without_same_type_neighbor": 0,
    }
    for score, neighbor in zip(best_scores, best_neighbors):
        if neighbor is None:
            histogram["without_same_type_neighbor"] += 1
        elif score >= 0.9:
            histogram["at_least_zero_point_nine"] += 1
        elif score >= 0.75:
            histogram["zero_point_seven_five_to_zero_point_nine"] += 1
        elif score >= 0.5:
            histogram["zero_point_five_to_zero_point_seven_five"] += 1
        else:
            histogram["below_zero_point_five"] += 1

    top_pairs = []
    for score, _left_id, _right_id, left_index, right_index in sorted(
        top_heap,
        reverse=True,
    ):
        left = prototypes[left_index]
        right = prototypes[right_index]
        top_pairs.append(
            {
                "score": round(score, 6),
                "left": {
                    "prototype_id": str(left["prototype_id"]),
                    "failure_mechanism": str(left["failure_mechanism"]),
                    "repair_operator": str(left["repair_operator"]),
                },
                "right": {
                    "prototype_id": str(right["prototype_id"]),
                    "failure_mechanism": str(right["failure_mechanism"]),
                    "repair_operator": str(right["repair_operator"]),
                },
            }
        )
    return {
        "method": "word_token_jaccard_diagnostic_only",
        "decision_use": False,
        "stopwords_removed": sorted(_STOPWORDS),
        "best_neighbor_histogram": histogram,
        "top_pairs": top_pairs,
    }


def _detailed_state(
    prototypes: Sequence[Mapping[str, Any]],
    *,
    signatures_by_id: Mapping[str, V4RepairSignature],
    top_limit: int,
    sample_limit: int,
    near_duplicate_limit: int,
) -> dict[str, Any]:
    rows = [
        _prototype_row(
            prototype,
            signatures_by_id=signatures_by_id,
            include_members=True,
        )
        for prototype in prototypes
    ]
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["distinct_sample_count"],
            -row["member_experience_count"],
            row["prototype_id"],
        ),
    )
    qualified = [
        row
        for row in ranked
        if row["distinct_sample_count"] >= V4_MIN_CONSTRUCTION_EXAMPLES
    ]
    return {
        "summary": _state_summary(
            prototypes,
            signatures_by_id=signatures_by_id,
        ),
        "largest_prototypes": ranked[:top_limit],
        "qualified_prototypes": qualified,
        "deterministic_samples": {
            "support_one": _deterministic_sample(
                prototypes,
                signatures_by_id=signatures_by_id,
                predicate=lambda value: value == 1,
                seed="support-one",
                limit=sample_limit,
            ),
            "support_two_to_four": _deterministic_sample(
                prototypes,
                signatures_by_id=signatures_by_id,
                predicate=lambda value: 2 <= value <= 4,
                seed="support-two-to-four",
                limit=sample_limit,
            ),
        },
        "lexical_near_duplicates": _lexical_near_duplicate_diagnostics(
            prototypes,
            limit=near_duplicate_limit,
        ),
    }


def inspect_cluster_progress(
    construction_dir: Path,
    *,
    top_limit: int = DEFAULT_TOP_LIMIT,
    sample_limit: int = DEFAULT_SAMPLE_LIMIT,
    near_duplicate_limit: int = DEFAULT_NEAR_DUPLICATE_LIMIT,
) -> dict[str, Any]:
    for owner, value in (
        ("top_limit", top_limit),
        ("sample_limit", sample_limit),
        ("near_duplicate_limit", near_duplicate_limit),
    ):
        if value < 0:
            raise ValueError(f"{owner} must be non-negative")

    construction_dir = construction_dir.expanduser().resolve()
    profile_path = construction_dir / "construction_profile.json"
    signatures_path = construction_dir / "repair_signatures.jsonl"
    map_path = construction_dir / "cluster_map_shards.jsonl"
    reduce_path = construction_dir / "cluster_reduce_batches.jsonl"
    for path in (profile_path, signatures_path, map_path, reduce_path):
        if not path.is_file():
            raise ValueError(f"Missing V4 checkpoint file: {path}")

    profile = _load_profile(profile_path)
    clustering = profile["clustering"]
    teacher = profile["teacher"]
    signatures = _load_signatures(signatures_path, profile=profile)
    signatures_by_id = {item.experience_id: item for item in signatures}
    eligible = tuple(item for item in signatures if item.applicable)
    map_records = _load_cluster_unit_records(map_path)
    reduce_records = _load_cluster_unit_records(reduce_path)

    map_batches = _bounded_batches(
        eligible,
        batch_size=int(clustering["map_batch_size"]),
        key=_signature_sort_key,
    )
    map_prototypes: list[dict[str, Any]] = []
    missing_map_units: list[str] = []
    used_map_units: set[str] = set()
    for batch_index, batch in enumerate(map_batches):
        unit_id = f"map-{batch[0].experience_type}-{batch_index:05d}"
        record = map_records.get(unit_id)
        if record is None:
            missing_map_units.append(unit_id)
            continue
        input_value = [item.to_dict() for item in batch]
        _validate_unit_record(
            record,
            stage="map",
            unit_id=unit_id,
            input_value=input_value,
            teacher=teacher,
        )
        prototypes, rejected = parse_cluster_map_payload(
            record["payload"],
            signatures=batch,
            unit_id=unit_id,
        )
        if rejected:
            raise ValueError(f"Assignment-only map unexpectedly rejected IDs: {unit_id}")
        map_prototypes.extend(prototypes)
        used_map_units.add(unit_id)

    base_report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "construction_dir": str(construction_dir),
        "repository_revision": _repository_revision(),
        "profile": profile,
        "inputs": {
            "construction_profile": {
                "path": str(profile_path),
                "sha256": file_sha256(profile_path),
            },
            "repair_signatures": {
                "path": str(signatures_path),
                "sha256": file_sha256(signatures_path),
                "record_count": len(signatures),
                "applicable_count": len(eligible),
                "nonapplicable_count": len(signatures) - len(eligible),
            },
            "cluster_map_shards": {
                "path": str(map_path),
                "sha256": file_sha256(map_path),
                "checkpoint_record_count": len(map_records),
                "expected_unit_count": len(map_batches),
            },
            "cluster_reduce_batches": {
                "path": str(reduce_path),
                "sha256": file_sha256(reduce_path),
                "checkpoint_record_count": len(reduce_records),
            },
        },
        "map": {
            "status": "complete" if not missing_map_units else "incomplete",
            "completed_unit_count": len(used_map_units),
            "expected_unit_count": len(map_batches),
            "missing_unit_count": len(missing_map_units),
            "first_missing_unit_ids": missing_map_units[:20],
            "unexpected_checkpoint_unit_ids": sorted(
                set(map_records) - used_map_units
            )[:20],
        },
    }
    if missing_map_units:
        base_report["reduce"] = {
            "status": "not_reconstructable_until_map_is_complete",
        }
        base_report["created_at"] = datetime.now(timezone.utc).isoformat()
        logical = {
            key: value for key, value in base_report.items() if key != "created_at"
        }
        base_report["report_sha256"] = canonical_json_sha256(logical)
        return base_report

    current = list(_merge_exact_prototypes(map_prototypes, unit_id="post-map"))
    stages: list[dict[str, Any]] = [
        {
            "stage": "post_map_exact_merge",
            **_state_summary(current, signatures_by_id=signatures_by_id),
        }
    ]
    used_reduce_units: set[str] = set()
    completed_round_count = 0
    reduce_status = "max_rounds_exhausted"
    in_progress_round: dict[str, Any] | None = None
    max_rounds = int(clustering["max_reduce_rounds"])
    reduce_batch_size = int(clustering["reduce_batch_size"])

    for round_index in range(max_rounds):
        if not current:
            reduce_status = "global_complete"
            break
        counts_by_type: dict[str, int] = {}
        for prototype in current:
            experience_type = str(prototype["experience_type"])
            counts_by_type[experience_type] = counts_by_type.get(experience_type, 0) + 1
        round_is_global = all(
            count <= reduce_batch_size for count in counts_by_type.values()
        )
        batches = _bounded_batches(
            current,
            batch_size=reduce_batch_size,
            key=_semantic_sort_key,
        )
        next_prototypes: list[dict[str, Any]] = []
        missing_units: list[str] = []
        completed_api_batches = 0
        required_api_batches = sum(1 for batch in batches if len(batch) > 1)
        for batch_index, batch in enumerate(batches):
            if len(batch) == 1:
                next_prototypes.append(dict(batch[0]))
                continue
            experience_type = str(batch[0]["experience_type"])
            unit_id = f"reduce-{round_index:02d}-{experience_type}-{batch_index:05d}"
            record = reduce_records.get(unit_id)
            if record is None:
                missing_units.append(unit_id)
                continue
            input_value = [dict(item) for item in batch]
            _validate_unit_record(
                record,
                stage="reduce",
                unit_id=unit_id,
                input_value=input_value,
                teacher=teacher,
            )
            merged, rejected = parse_cluster_reduce_payload(
                record["payload"],
                prototypes=batch,
                unit_id=unit_id,
            )
            if rejected:
                raise ValueError(
                    f"Assignment-only reduce unexpectedly rejected IDs: {unit_id}"
                )
            next_prototypes.extend(merged)
            completed_api_batches += 1
            used_reduce_units.add(unit_id)

        if missing_units:
            in_progress_round = {
                "round": round_index,
                "start_prototype_count": len(current),
                "batch_count": len(batches),
                "required_api_batch_count": required_api_batches,
                "completed_api_batch_count": completed_api_batches,
                "missing_api_batch_count": len(missing_units),
                "first_missing_unit_ids": missing_units[:20],
                "global_within_experience_type": round_is_global,
            }
            reduce_status = "round_incomplete"
            break

        before_count = len(current)
        current = list(
            _merge_exact_prototypes(
                next_prototypes,
                unit_id=f"post-reduce-{round_index:02d}",
            )
        )
        after_count = len(current)
        completed_round_count += 1
        stages.append(
            {
                "stage": "post_reduce_round",
                "round": round_index,
                "input_prototype_count": before_count,
                "output_prototype_count": after_count,
                "retention_ratio": (
                    round(after_count / before_count, 8) if before_count else 0.0
                ),
                "reduction_ratio": (
                    round(1.0 - after_count / before_count, 8)
                    if before_count
                    else 0.0
                ),
                "batch_count": len(batches),
                "api_batch_count": required_api_batches,
                "global_within_experience_type": round_is_global,
                **_state_summary(current, signatures_by_id=signatures_by_id),
            }
        )
        if round_is_global:
            reduce_status = "global_complete"
            break
        if after_count >= before_count:
            reduce_status = "plateau"
            break

    base_report["map"]["post_map_prototype_count_before_exact_merge"] = len(
        map_prototypes
    )
    base_report["map"]["post_map_prototype_count_after_exact_merge"] = stages[0][
        "prototype_count"
    ]
    base_report["reduce"] = {
        "status": reduce_status,
        "completed_round_count": completed_round_count,
        "last_complete_round": completed_round_count - 1,
        "in_progress_round": in_progress_round,
        "stages": stages,
        "used_checkpoint_unit_count": len(used_reduce_units),
        "unexpected_checkpoint_unit_ids": sorted(
            set(reduce_records) - used_reduce_units
        )[:20],
        "last_complete_state": _detailed_state(
            current,
            signatures_by_id=signatures_by_id,
            top_limit=top_limit,
            sample_limit=sample_limit,
            near_duplicate_limit=near_duplicate_limit,
        ),
    }
    base_report["created_at"] = datetime.now(timezone.utc).isoformat()
    logical = {
        key: value for key, value in base_report.items() if key != "created_at"
    }
    base_report["report_sha256"] = canonical_json_sha256(logical)
    return base_report


def _print_summary(report: Mapping[str, Any], *, output: Path) -> None:
    print(
        "[v4-cluster-inspect] "
        f"map={report['map']['status']} "
        f"map_units={report['map']['completed_unit_count']}/"
        f"{report['map']['expected_unit_count']}"
    )
    reduce = report.get("reduce", {})
    print(
        "[v4-cluster-inspect] "
        f"reduce_status={reduce.get('status')} "
        f"completed_rounds={reduce.get('completed_round_count', 0)}"
    )
    state = reduce.get("last_complete_state")
    if isinstance(state, Mapping):
        summary = state["summary"]
        buckets = summary["support_histogram"]
        print(
            "[v4-cluster-inspect] "
            f"prototypes={summary['prototype_count']} "
            f"qualified={summary['qualified_prototype_count']} "
            f"support_one={buckets['one']['prototype_count']} "
            f"support_two_to_four={buckets['two_to_four']['prototype_count']} "
            "support_five_plus="
            f"{summary['qualified_prototype_count']}"
        )
    print(
        "[v4-cluster-inspect] "
        f"report={output} sha256={report['report_sha256']}"
    )


def main() -> None:
    args = parse_args()
    construction_dir = args.construction_dir.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else construction_dir / "reduce_progress_report.json"
    )
    checkpoint_names = {
        "construction_profile.json",
        "repair_signatures.jsonl",
        "cluster_map_shards.jsonl",
        "cluster_reduce_batches.jsonl",
    }
    if output.parent == construction_dir and output.name in checkpoint_names:
        raise ValueError("Refusing to overwrite a V4 construction checkpoint")
    report = inspect_cluster_progress(
        construction_dir,
        top_limit=args.top_limit,
        sample_limit=args.sample_limit,
        near_duplicate_limit=args.near_duplicate_limit,
    )
    _write_json(output, report)
    _print_summary(report, output=output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[v4-cluster-inspect] error: {exc}", file=sys.stderr)
        raise
