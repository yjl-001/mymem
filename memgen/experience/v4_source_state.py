"""Offline-only V4 source-state cache contracts and CPU analysis helpers.

The cache deliberately stores answer-blind layer-24 states once so selector
ablations can be repeated without loading the reasoner.  It is not a runtime
selector artifact and its loader fails closed if any manifest, metadata, or
tensor binding changes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from memgen.experience.phase1 import canonical_json_sha256, file_sha256


V4_SOURCE_STATE_CACHE_SCHEMA = "memgen-v4-source-state-cache-v1"
V4_SOURCE_STATE_EVENT_SCHEMA = "memgen-v4-source-state-event-v1"
V4_GATE_REACHABILITY_REPORT_SCHEMA = "memgen-v4-gate-reachability-report-v1"
V4_SOURCE_STATE_AUDIT_SCHEMA = "memgen-v4-source-state-cpu-audit-v1"
V4_SOURCE_STATE_MAX_WINDOW = 32
V4_SOURCE_STATE_WINDOWS = (1, 4, 8, 16, 32)

PROMPT_TENSORS = (
    "prompt_end_states",
    "question_mean_states",
    "question_boundary_states",
    "question_local_windows",
    "question_local_masks",
)
FAILURE_TENSORS = (
    "failure_gate_windows",
    "failure_gate_masks",
    "aligned_success_windows",
    "aligned_success_masks",
)
SUCCESS_GATE_TENSORS = (
    "success_gate_windows",
    "success_gate_masks",
)
V4_SOURCE_STATE_TENSORS = PROMPT_TENSORS + FAILURE_TENSORS + SUCCESS_GATE_TENSORS
V4_SOURCE_STATE_EVENT_KINDS = frozenset(
    {"prompt_semantic", "failure_gate_attempt", "success_gate_attempt"}
)

_FORBIDDEN_CACHE_KEYS = frozenset(
    {
        "answer",
        "answer_sha256",
        "ground_truth",
        "reward",
        "target_reward",
        "reference_reward",
        "trajectory",
        "reference_trajectory",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
    temporary.replace(path)
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _logical_hash(value: Mapping[str, Any], hash_field: str) -> str:
    return canonical_json_sha256(
        {
            key: item
            for key, item in value.items()
            if key not in {"created_at", hash_field}
        }
    )


def _walk_forbidden_keys(value: Any, *, path: str = "cache") -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_CACHE_KEYS.intersection(str(key) for key in value)
        if forbidden:
            raise ValueError(
                f"V4 source-state cache contains forbidden answer/reward fields at "
                f"{path}: {sorted(forbidden)}"
            )
        for key, item in value.items():
            _walk_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _walk_forbidden_keys(item, path=f"{path}[{index}]")


def finalize_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and content-address one answer-blind cache metadata event."""

    row = dict(event)
    row.pop("record_sha256", None)
    row.setdefault("schema_version", V4_SOURCE_STATE_EVENT_SCHEMA)
    _walk_forbidden_keys(row)
    _validate_event_shape(row, require_hash=False)
    row["record_sha256"] = canonical_json_sha256(row)
    return row


def _validate_tensor_ref(value: Any, *, expected: Sequence[str]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("V4 source-state event tensor references are incomplete")
    for tensor_name in expected:
        index = value[tensor_name]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("V4 source-state tensor row index is invalid")
    if len({int(value[name]) for name in expected}) != 1:
        raise ValueError("V4 source-state paired tensor rows are not aligned")


def _validate_event_shape(event: Mapping[str, Any], *, require_hash: bool) -> None:
    if event.get("schema_version") != V4_SOURCE_STATE_EVENT_SCHEMA:
        raise ValueError("Unexpected V4 source-state event schema")
    event_kind = event.get("event_kind")
    if event_kind not in V4_SOURCE_STATE_EVENT_KINDS:
        raise ValueError("Unexpected V4 source-state event kind")
    required_strings = (
        "event_id",
        "experience_id",
        "sample_id",
        "independent_sample_id",
        "bank_id",
        "benchmark",
        "logical_split",
        "dataset_split",
        "question_sha256",
        "construction_profile_sha256",
        "bank_record_sha256",
    )
    if any(not isinstance(event.get(field), str) or not event[field] for field in required_strings):
        raise ValueError("V4 source-state event identity is incomplete")
    if not isinstance(event.get("is_medoid"), bool):
        raise ValueError("V4 source-state event medoid flag is invalid")
    if event.get("curation_tier") not in {"primary", "conditional"}:
        raise ValueError("V4 source-state event curation tier is invalid")
    if event.get("online_reachable_safety_negative") not in {True, False}:
        raise ValueError("V4 source-state event safety-negative role is missing")
    contrast_pair = event.get("contrast_pair")
    if (
        not isinstance(contrast_pair, Mapping)
        or contrast_pair.get("paired_success_failure") is not True
        or any(
            not isinstance(contrast_pair.get(field), str)
            or not contrast_pair[field]
            for field in ("target_episode_id", "reference_episode_id")
        )
    ):
        raise ValueError("V4 source-state contrast-pair identity is incomplete")
    completion_hashes = event.get("completion_hashes")
    if (
        not isinstance(completion_hashes, Mapping)
        or any(
            not isinstance(completion_hashes.get(field), str)
            or not completion_hashes[field]
            for field in (
                "verified_success_completion_sha256",
                "verified_failure_completion_sha256",
            )
        )
    ):
        raise ValueError("V4 source-state completion hash binding is incomplete")
    refs = event.get("tensor_rows")
    if event_kind == "prompt_semantic":
        _validate_tensor_ref(refs, expected=PROMPT_TENSORS)
        if event.get("online_reachable_safety_negative") is not False:
            raise ValueError("Prompt state cannot be a gate safety negative")
        prompt_count = event.get("prompt_token_count")
        question_start = event.get("question_token_start")
        question_end = event.get("question_token_end_exclusive")
        question_count = event.get("question_token_count")
        if (
            isinstance(prompt_count, bool)
            or not isinstance(prompt_count, int)
            or prompt_count <= 0
            or not isinstance(event.get("prompt_token_ids_sha256"), str)
            or not event["prompt_token_ids_sha256"]
            or isinstance(question_start, bool)
            or not isinstance(question_start, int)
            or question_start < 0
            or isinstance(question_end, bool)
            or not isinstance(question_end, int)
            or question_end <= question_start
            or question_end > prompt_count
            or isinstance(question_count, bool)
            or not isinstance(question_count, int)
            or question_count != question_end - question_start
        ):
            raise ValueError("V4 prompt/question token alignment is invalid")
        for prefix in ("failure", "success"):
            count = event.get(f"{prefix}_gate_attempt_count")
            eligible = event.get(f"{prefix}_gate_eligible")
            reason = event.get(f"{prefix}_gate_rejection_reason")
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= 3
                or eligible is not (count > 0)
                or (reason is None) is not (count > 0)
            ):
                raise ValueError("V4 prompt gate reachability diagnostics are inconsistent")
    elif event_kind == "failure_gate_attempt":
        _validate_tensor_ref(refs, expected=FAILURE_TENSORS)
        if event.get("online_reachable_safety_negative") is not False:
            raise ValueError("Failure gate state cannot be a success safety negative")
        _validate_gate_event(event)
        aligned = event.get("matched_success_alignment")
        if (
            not isinstance(aligned, Mapping)
            or aligned.get("state_role") != "offline_repair_direction_control"
            or aligned.get("online_reachable_safety_negative") is not False
            or aligned.get("alignment_method")
            != "normalized_reasoning_progress_endpoint_preserving"
            or any(
                isinstance(aligned.get(field), bool)
                or not isinstance(aligned.get(field), int)
                or int(aligned[field]) < 0
                for field in ("reasoning_rank", "token_position")
            )
            or isinstance(aligned.get("window_token_count"), bool)
            or not isinstance(aligned.get("window_token_count"), int)
            or not 1 <= int(aligned["window_token_count"]) <= 32
            or not isinstance(aligned.get("prefix_token_ids_sha256"), str)
            or not aligned["prefix_token_ids_sha256"]
        ):
            raise ValueError("Matched-success aligned state role is ambiguous")
    else:
        _validate_tensor_ref(refs, expected=SUCCESS_GATE_TENSORS)
        if event.get("online_reachable_safety_negative") is not True:
            raise ValueError("Actual success gate must be marked as a safety negative")
        _validate_gate_event(event)
    if require_hash:
        stored = event.get("record_sha256")
        logical = {key: value for key, value in event.items() if key != "record_sha256"}
        if stored != canonical_json_sha256(logical):
            raise ValueError("V4 source-state event hash mismatch")


def _validate_gate_event(event: Mapping[str, Any]) -> None:
    integer_fields = (
        "attempt_number",
        "reasoning_rank",
        "token_position",
        "candidate_rank",
        "window_token_count",
    )
    if any(
        isinstance(event.get(field), bool)
        or not isinstance(event.get(field), int)
        or int(event[field])
        < (
            1
            if field in {"attempt_number", "candidate_rank", "window_token_count"}
            else 0
        )
        for field in integer_fields
    ):
        raise ValueError("V4 gate-event integer diagnostics are invalid")
    if (
        int(event["attempt_number"]) > 3
        or int(event["window_token_count"]) > 32
        or int(event["candidate_rank"]) != int(event["reasoning_rank"]) + 1
        or int(event["window_token_count"])
        != min(32, int(event["reasoning_rank"]) + 1)
    ):
        raise ValueError("V4 gate event exceeds frozen attempt/window bounds")
    gate = event.get("gate_diagnostics")
    if not isinstance(gate, Mapping) or gate.get("gate_eligible") is not True:
        raise ValueError("V4 gate event is not explicitly gate eligible")
    if gate.get("gate_rejection_reason") is not None:
        raise ValueError("Eligible V4 gate event has a rejection reason")
    for field in (
        "attention_entropy",
        "persistence_risk",
        "high_entropy_threshold",
        "low_entropy_threshold",
        "risk_threshold",
    ):
        value = gate.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("V4 gate diagnostics contain non-finite values")
    if any(
        not isinstance(gate.get(field), str) or not gate[field]
        for field in ("state_before", "state_after")
    ):
        raise ValueError("V4 gate state-machine diagnostics are incomplete")
    logit_summary = gate.get("logit_summary")
    if not isinstance(logit_summary, Mapping) or any(
        not isinstance(logit_summary.get(field), (int, float))
        or not math.isfinite(float(logit_summary[field]))
        for field in (
            "maximum_logit",
            "top1_top2_logit_gap",
            "logsumexp",
            "predictive_entropy",
        )
    ):
        raise ValueError("V4 gate logit summary is incomplete")
    alignment = event.get("prefix_alignment")
    if (
        not isinstance(alignment, Mapping)
        or alignment.get("prefix_includes_current_token") is not True
        or alignment.get("token_position_matches_prefix_end") is not True
        or not isinstance(alignment.get("prefix_token_ids_sha256"), str)
        or not alignment["prefix_token_ids_sha256"]
        or int(alignment.get("prefix_token_count", -1))
        != int(event["token_position"]) + 1
    ):
        raise ValueError("V4 gate prefix/token alignment is invalid")


def validate_events(events: Sequence[Mapping[str, Any]]) -> None:
    if not events:
        raise ValueError("V4 source-state cache has no metadata events")
    event_ids: set[str] = set()
    tensor_indices: dict[str, list[int]] = defaultdict(list)
    prompt_samples: set[str] = set()
    for event in events:
        _walk_forbidden_keys(event)
        _validate_event_shape(event, require_hash=True)
        event_id = str(event["event_id"])
        if event_id in event_ids:
            raise ValueError("V4 source-state event IDs are duplicated")
        event_ids.add(event_id)
        for tensor_name, index in event["tensor_rows"].items():
            tensor_indices[str(tensor_name)].append(int(index))
        if event["event_kind"] == "prompt_semantic":
            sample_id = str(event["sample_id"])
            if sample_id in prompt_samples:
                raise ValueError("V4 source-state prompt sample is duplicated")
            prompt_samples.add(sample_id)
    event_samples = {str(event["sample_id"]) for event in events}
    if event_samples != prompt_samples:
        raise ValueError("V4 source-state gate event lacks a prompt identity row")
    prompts_by_sample = {
        str(event["sample_id"]): event
        for event in events
        if event["event_kind"] == "prompt_semantic"
    }
    independent_prompt_ids = {
        str(event["independent_sample_id"]) for event in prompts_by_sample.values()
    }
    if len(independent_prompt_ids) != len(prompts_by_sample):
        raise ValueError("V4 source-state independent sample identities are duplicated")
    for sample_id, prompt in prompts_by_sample.items():
        sample_events = [
            event for event in events if str(event["sample_id"]) == sample_id
        ]
        for event in sample_events:
            for field in (
                "experience_id",
                "independent_sample_id",
                "bank_id",
                "benchmark",
                "logical_split",
                "dataset_split",
                "source_index",
                "question_sha256",
                "is_medoid",
                "curation_tier",
                "construction_profile_sha256",
                "bank_record_sha256",
                "contrast_pair",
                "completion_hashes",
            ):
                if event.get(field) != prompt.get(field):
                    raise ValueError("V4 source-state sample identity drifted across events")
        for prefix, event_kind in (
            ("failure", "failure_gate_attempt"),
            ("success", "success_gate_attempt"),
        ):
            attempts = sorted(
                int(event["attempt_number"])
                for event in events
                if event["event_kind"] == event_kind
                and str(event["sample_id"]) == sample_id
            )
            expected_count = int(prompt[f"{prefix}_gate_attempt_count"])
            if attempts != list(range(1, expected_count + 1)):
                raise ValueError("V4 source-state prompt/event gate attempts differ")
    for tensor_name in V4_SOURCE_STATE_TENSORS:
        indices = tensor_indices.get(tensor_name, [])
        if sorted(indices) != list(range(len(indices))):
            raise ValueError(f"V4 source-state tensor rows are not contiguous: {tensor_name}")


def _dtype_name(tensor: Any) -> str:
    return str(getattr(tensor, "dtype", "unknown")).removeprefix("torch.")


def _tensor_layout(tensors: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if set(tensors) != set(V4_SOURCE_STATE_TENSORS):
        missing = sorted(set(V4_SOURCE_STATE_TENSORS) - set(tensors))
        extra = sorted(set(tensors) - set(V4_SOURCE_STATE_TENSORS))
        raise ValueError(f"V4 source-state tensor set differs: missing={missing} extra={extra}")
    result: dict[str, dict[str, Any]] = {}
    for name in V4_SOURCE_STATE_TENSORS:
        tensor = tensors[name]
        result[name] = {
            "shape": [int(value) for value in tensor.shape],
            "dtype": _dtype_name(tensor),
        }
    return result


def validate_tensor_alignment(
    *,
    events: Sequence[Mapping[str, Any]],
    tensors: Mapping[str, Any],
    layout: Mapping[str, Any] | None = None,
) -> None:
    """Validate tensor/index coverage without depending on model code."""

    validate_events(events)
    observed_layout = _tensor_layout(tensors)
    if layout is not None and dict(layout) != observed_layout:
        raise ValueError("V4 source-state tensor layout differs from manifest")
    hidden_widths: set[int] = set()
    for name, tensor in tensors.items():
        shape = tuple(int(value) for value in tensor.shape)
        if name.endswith("_masks"):
            if len(shape) != 2 or shape[1] != V4_SOURCE_STATE_MAX_WINDOW:
                raise ValueError("V4 source-state window mask shape is invalid")
            if _dtype_name(tensor) != "bool":
                raise ValueError("V4 source-state window mask dtype is invalid")
            continue
        if name.endswith("_windows"):
            if len(shape) != 3 or shape[1] != V4_SOURCE_STATE_MAX_WINDOW:
                raise ValueError("V4 source-state window tensor shape is invalid")
            hidden_widths.add(shape[2])
        else:
            if len(shape) != 2:
                raise ValueError("V4 source-state point tensor shape is invalid")
            hidden_widths.add(shape[1])
        finite = getattr(tensor, "isfinite", None)
        if callable(finite) and not bool(finite().all().item()):
            raise ValueError("V4 source-state tensor contains non-finite values")
    if len(hidden_widths) != 1:
        raise ValueError("V4 source-state hidden widths differ")
    for state_name, mask_name in (
        ("question_local_windows", "question_local_masks"),
        ("failure_gate_windows", "failure_gate_masks"),
        ("aligned_success_windows", "aligned_success_masks"),
        ("success_gate_windows", "success_gate_masks"),
    ):
        if tensors[state_name].shape[:2] != tensors[mask_name].shape:
            raise ValueError("V4 source-state window/mask shape differs")
        mask = tensors[mask_name]
        for row in range(int(mask.shape[0])):
            values = [bool(value) for value in mask[row].tolist()]
            if not any(values) or values != sorted(values):
                raise ValueError("V4 source-state masks must be non-empty suffixes")


def build_gate_reachability_report(
    *, events: Sequence[Mapping[str, Any]], bank_ids: Sequence[str]
) -> dict[str, Any]:
    """Count construction and gate support by independent sample, never event."""

    validate_events(events)
    prompts = [event for event in events if event["event_kind"] == "prompt_semantic"]
    normalized_bank_ids = tuple(str(value) for value in bank_ids)
    if (
        not normalized_bank_ids
        or len(set(normalized_bank_ids)) != len(normalized_bank_ids)
        or set(normalized_bank_ids) != {str(event["bank_id"]) for event in prompts}
    ):
        raise ValueError("V4 gate-reachability bank coverage differs from prompt events")
    failures = [event for event in events if event["event_kind"] == "failure_gate_attempt"]
    successes = [event for event in events if event["event_kind"] == "success_gate_attempt"]
    prompt_by_sample = {
        str(event["independent_sample_id"]): event for event in prompts
    }
    failure_samples = {
        str(event["independent_sample_id"]) for event in failures
    }
    success_samples = {
        str(event["independent_sample_id"]) for event in successes
    }
    per_bank: dict[str, Any] = {}
    for bank_id in normalized_bank_ids:
        bank_prompts = [event for event in prompts if event["bank_id"] == bank_id]
        bank_failures = [event for event in failures if event["bank_id"] == bank_id]
        bank_successes = [event for event in successes if event["bank_id"] == bank_id]
        construction_samples = {
            str(event["independent_sample_id"]) for event in bank_prompts
        }
        reachable_failure_samples = {
            str(event["independent_sample_id"]) for event in bank_failures
        }
        reachable_success_samples = {
            str(event["independent_sample_id"]) for event in bank_successes
        }
        per_bank[str(bank_id)] = {
            "construction_independent_sample_count": len(construction_samples),
            "failure_gate_reachable_independent_sample_count": len(reachable_failure_samples),
            "failure_gate_unreachable_independent_sample_count": len(
                construction_samples - reachable_failure_samples
            ),
            "failure_gate_event_count": len(bank_failures),
            "success_gate_reachable_independent_sample_count": len(reachable_success_samples),
            "success_gate_unreachable_independent_sample_count": len(
                construction_samples - reachable_success_samples
            ),
            "success_gate_event_count": len(bank_successes),
            "medoid_failure_gate_reachable": any(
                event["is_medoid"]
                and event["independent_sample_id"] in reachable_failure_samples
                for event in bank_prompts
            ),
            "failure_rejection_reasons": dict(
                sorted(
                    Counter(
                        str(event.get("failure_gate_rejection_reason") or "reachable")
                        for event in bank_prompts
                    ).items()
                )
            ),
            "success_rejection_reasons": dict(
                sorted(
                    Counter(
                        str(event.get("success_gate_rejection_reason") or "reachable")
                        for event in bank_prompts
                    ).items()
                )
            ),
        }
    report: dict[str, Any] = {
        "schema_version": V4_GATE_REACHABILITY_REPORT_SCHEMA,
        "created_at": utc_now(),
        "offline_only": True,
        "qualified_for_online_use": False,
        "support_unit": "independent_sample",
        "multiple_attempts_do_not_increase_support": True,
        "construction_independent_sample_count": len(prompt_by_sample),
        "failure_gate_reachable_independent_sample_count": len(failure_samples),
        "failure_gate_unreachable_independent_sample_count": len(
            set(prompt_by_sample) - failure_samples
        ),
        "failure_gate_event_count": len(failures),
        "success_gate_reachable_independent_sample_count": len(success_samples),
        "success_gate_unreachable_independent_sample_count": len(
            set(prompt_by_sample) - success_samples
        ),
        "success_gate_event_count": len(successes),
        "per_bank": per_bank,
    }
    report["report_sha256"] = _logical_hash(report, "report_sha256")
    return report


def validate_gate_reachability_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != V4_GATE_REACHABILITY_REPORT_SCHEMA:
        raise ValueError("Unexpected V4 gate-reachability report schema")
    if (
        report.get("offline_only") is not True
        or report.get("qualified_for_online_use") is not False
        or report.get("support_unit") != "independent_sample"
        or report.get("multiple_attempts_do_not_increase_support") is not True
    ):
        raise ValueError("V4 gate-reachability safety contract drifted")
    if report.get("report_sha256") != _logical_hash(report, "report_sha256"):
        raise ValueError("V4 gate-reachability report hash mismatch")


def build_source_state_manifest(
    *,
    tensors: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    reachability_report: Mapping[str, Any],
    tensor_path: Path,
    event_path: Path,
    reachability_path: Path,
    repository_revision: str,
    reasoner: Mapping[str, Any],
    configuration: Mapping[str, Any],
    provenance: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
) -> dict[str, Any]:
    validate_tensor_alignment(events=events, tensors=tensors)
    validate_gate_reachability_report(reachability_report)
    _walk_forbidden_keys(provenance)
    event_kinds = Counter(str(event["event_kind"]) for event in events)
    prompt_events = [event for event in events if event["event_kind"] == "prompt_semantic"]
    bank_ids = sorted({str(event["bank_id"]) for event in prompt_events})
    manifest: dict[str, Any] = {
        "schema_version": V4_SOURCE_STATE_CACHE_SCHEMA,
        "created_at": utc_now(),
        "status": "source_state_cache_built",
        "offline_only": True,
        "qualified_for_online_use": False,
        "contains_reward_or_answer_signal": False,
        "repository_revision": repository_revision,
        "reasoner": dict(reasoner),
        "configuration": dict(configuration),
        "provenance": dict(provenance),
        "implementation_sha256": dict(implementation_sha256),
        "bank_ids": bank_ids,
        "counts": {
            "bank_count": len(bank_ids),
            "independent_sample_count": len(prompt_events),
            "event_count": len(events),
            "event_count_by_kind": dict(sorted(event_kinds.items())),
            "failure_gate_unreachable_independent_sample_count": reachability_report[
                "failure_gate_unreachable_independent_sample_count"
            ],
        },
        "tensor_layout": _tensor_layout(tensors),
        "event_order_sha256": canonical_json_sha256(
            [str(event["event_id"]) for event in events]
        ),
        "event_record_sha256": {
            str(event["event_id"]): str(event["record_sha256"]) for event in events
        },
        "artifacts": {
            "tensors": {"path": tensor_path.name, "sha256": file_sha256(tensor_path)},
            "events": {
                "path": event_path.name,
                "sha256": file_sha256(event_path),
                "row_count": len(events),
            },
            "gate_reachability_report": {
                "path": reachability_path.name,
                "sha256": file_sha256(reachability_path),
                "logical_sha256": reachability_report["report_sha256"],
            },
        },
    }
    manifest["manifest_sha256"] = _logical_hash(manifest, "manifest_sha256")
    return manifest


def validate_source_state_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != V4_SOURCE_STATE_CACHE_SCHEMA:
        raise ValueError("Unexpected V4 source-state cache schema")
    if (
        manifest.get("status") != "source_state_cache_built"
        or manifest.get("offline_only") is not True
        or manifest.get("qualified_for_online_use") is not False
        or manifest.get("contains_reward_or_answer_signal") is not False
    ):
        raise ValueError("V4 source-state cache safety contract drifted")
    if manifest.get("manifest_sha256") != _logical_hash(manifest, "manifest_sha256"):
        raise ValueError("V4 source-state cache manifest hash mismatch")
    configuration = manifest.get("configuration", {})
    if (
        configuration.get("layer_number") != 24
        or configuration.get("attention_implementation") != "sdpa"
        or configuration.get("dtype") != "bfloat16"
        or configuration.get("maximum_gate_attempts") != 3
        or configuration.get("maximum_hidden_window") != 32
        or configuration.get("support_unit") != "independent_sample"
    ):
        raise ValueError("V4 source-state frozen configuration drifted")
    reasoner = manifest.get("reasoner", {})
    if any(
        not reasoner.get(field)
        for field in ("model_name", "model_revision", "tokenizer_revision")
    ):
        raise ValueError("V4 source-state reasoner provenance is incomplete")
    if set(manifest.get("tensor_layout", {})) != set(V4_SOURCE_STATE_TENSORS):
        raise ValueError("V4 source-state manifest tensor set differs")
    if any(
        layout.get("dtype") != ("bool" if name.endswith("_masks") else "bfloat16")
        for name, layout in manifest["tensor_layout"].items()
    ):
        raise ValueError("V4 source-state manifest tensor dtypes differ")
    bank_ids = manifest.get("bank_ids")
    if (
        not isinstance(bank_ids, list)
        or not bank_ids
        or bank_ids != sorted(bank_ids)
        or len(set(bank_ids)) != len(bank_ids)
    ):
        raise ValueError("V4 source-state manifest bank namespace is invalid")
    if set(manifest.get("artifacts", {})) != {
        "tensors",
        "events",
        "gate_reachability_report",
    }:
        raise ValueError("V4 source-state manifest artifact set differs")
    _walk_forbidden_keys(manifest.get("provenance", {}))


@dataclass(frozen=True)
class V4SourceStateCache:
    manifest_path: Path
    manifest: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    reachability_report: Mapping[str, Any]
    tensors: Mapping[str, Any] | None

    @property
    def event_by_id(self) -> dict[str, Mapping[str, Any]]:
        return {str(event["event_id"]): event for event in self.events}


def load_source_state_cache(
    manifest_path: Path, *, load_tensors: bool = True
) -> V4SourceStateCache:
    """Authenticate all cache artifacts; optionally leave tensors unopened."""

    manifest_path = manifest_path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_source_state_manifest(manifest)
    resolved: dict[str, Path] = {}
    for name, artifact in manifest["artifacts"].items():
        relative = Path(str(artifact.get("path", "")))
        if relative.is_absolute() or relative.name != str(relative):
            raise ValueError("V4 source-state artifact path must be a local filename")
        path = (manifest_path.parent / relative).resolve()
        if path.parent != manifest_path.parent:
            raise ValueError("V4 source-state artifact escaped its cache directory")
        if not path.is_file() or file_sha256(path) != artifact.get("sha256"):
            raise ValueError(f"V4 source-state {name} artifact is missing or corrupted")
        resolved[name] = path
    events = tuple(_read_jsonl(resolved["events"]))
    validate_events(events)
    if len(events) != manifest["artifacts"]["events"]["row_count"]:
        raise ValueError("V4 source-state event count differs from manifest")
    if manifest["event_order_sha256"] != canonical_json_sha256(
        [str(event["event_id"]) for event in events]
    ):
        raise ValueError("V4 source-state event order differs from manifest")
    if manifest["event_record_sha256"] != {
        str(event["event_id"]): str(event["record_sha256"]) for event in events
    }:
        raise ValueError("V4 source-state event bindings differ from manifest")
    prompt_events = [
        event for event in events if event["event_kind"] == "prompt_semantic"
    ]
    observed_kind_counts = dict(
        sorted(Counter(str(event["event_kind"]) for event in events).items())
    )
    if (
        manifest["counts"].get("independent_sample_count") != len(prompt_events)
        or manifest["counts"].get("event_count") != len(events)
        or manifest["counts"].get("event_count_by_kind") != observed_kind_counts
        or manifest.get("bank_ids")
        != sorted({str(event["bank_id"]) for event in prompt_events})
    ):
        raise ValueError("V4 source-state manifest counts differ from events")
    tensor_reference_counts = {
        tensor_name: sum(
            tensor_name in event["tensor_rows"] for event in events
        )
        for tensor_name in V4_SOURCE_STATE_TENSORS
    }
    if any(
        manifest["tensor_layout"][tensor_name]["shape"][0] != count
        for tensor_name, count in tensor_reference_counts.items()
    ):
        raise ValueError("V4 source-state tensor counts differ from event references")
    reachability = json.loads(resolved["gate_reachability_report"].read_text(encoding="utf-8"))
    validate_gate_reachability_report(reachability)
    if (
        manifest["counts"].get("bank_count") != len(manifest["bank_ids"])
        or manifest["counts"].get("failure_gate_unreachable_independent_sample_count")
        != reachability["failure_gate_unreachable_independent_sample_count"]
        or reachability["report_sha256"]
        != manifest["artifacts"]["gate_reachability_report"]["logical_sha256"]
    ):
        raise ValueError("V4 source-state reachability binding differs")
    reconstructed_reachability = build_gate_reachability_report(
        events=events, bank_ids=manifest["bank_ids"]
    )
    if {
        key: value
        for key, value in reconstructed_reachability.items()
        if key not in {"created_at", "report_sha256"}
    } != {
        key: value
        for key, value in reachability.items()
        if key not in {"created_at", "report_sha256"}
    }:
        raise ValueError("V4 source-state reachability report differs from events")
    tensors = None
    if load_tensors:
        from safetensors.torch import load_file

        tensors = load_file(str(resolved["tensors"]), device="cpu")
        validate_tensor_alignment(
            events=events,
            tensors=tensors,
            layout=manifest["tensor_layout"],
        )
    return V4SourceStateCache(
        manifest_path=manifest_path,
        manifest=manifest,
        events=events,
        reachability_report=reachability,
        tensors=tensors,
    )


def save_source_state_cache(
    *,
    output_dir: Path,
    tensors: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    repository_revision: str,
    reasoner: Mapping[str, Any],
    configuration: Mapping[str, Any],
    provenance: Mapping[str, Any],
    implementation_sha256: Mapping[str, str],
) -> tuple[Path, Path]:
    """Atomically write safetensors, JSONL, reachability, then manifest."""

    from safetensors.torch import save_file

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    finalized = tuple(finalize_event(event) for event in events)
    validate_tensor_alignment(events=finalized, tensors=tensors)
    tensor_path = output_dir / "v4_source_states.safetensors"
    event_path = output_dir / "v4_source_state_events.jsonl"
    reachability_path = output_dir / "gate_reachability_report.json"
    manifest_path = output_dir / "v4_source_state_manifest.json"
    temporary_tensor = tensor_path.with_name(tensor_path.name + ".tmp")
    save_file(
        {name: tensors[name].detach().cpu().contiguous() for name in V4_SOURCE_STATE_TENSORS},
        str(temporary_tensor),
    )
    temporary_tensor.replace(tensor_path)
    _write_jsonl(event_path, finalized)
    bank_ids = sorted(
        {str(event["bank_id"]) for event in finalized if event["event_kind"] == "prompt_semantic"}
    )
    reachability = build_gate_reachability_report(events=finalized, bank_ids=bank_ids)
    _write_json(reachability_path, reachability)
    manifest = build_source_state_manifest(
        tensors=tensors,
        events=finalized,
        reachability_report=reachability,
        tensor_path=tensor_path,
        event_path=event_path,
        reachability_path=reachability_path,
        repository_revision=repository_revision,
        reasoner=reasoner,
        configuration=configuration,
        provenance=provenance,
        implementation_sha256=implementation_sha256,
    )
    _write_json(manifest_path, manifest)
    load_source_state_cache(manifest_path)
    return manifest_path, reachability_path


def derive_window_mean(window: Any, mask: Any, *, size: int) -> Any:
    """Derive a normalized latest-N vector from a right-aligned raw window."""

    import torch
    import torch.nn.functional as F

    if size not in V4_SOURCE_STATE_WINDOWS:
        raise ValueError("Unsupported V4 source-state window size")
    if window.ndim != 2 or mask.ndim != 1 or window.shape[0] != mask.shape[0]:
        raise ValueError("V4 source-state window/mask dimensions differ")
    selected = window[mask.to(dtype=torch.bool)][-size:].float()
    if selected.numel() == 0:
        raise ValueError("V4 source-state window contains no valid state")
    vector = selected.mean(dim=0)
    if not torch.isfinite(vector).all() or float(vector.norm().item()) <= 0.0:
        raise ValueError("V4 source-state derived vector is invalid")
    return F.normalize(vector, dim=0).cpu().contiguous()


def independent_support(
    events: Sequence[Mapping[str, Any]], *, event_kind: str
) -> dict[str, int]:
    """Return per-bank support after collapsing attempts from the same sample."""

    if event_kind not in V4_SOURCE_STATE_EVENT_KINDS:
        raise ValueError("Unknown V4 source-state event kind")
    grouped: dict[str, set[str]] = defaultdict(set)
    for event in events:
        if event.get("event_kind") == event_kind:
            grouped[str(event["bank_id"])].add(
                str(event["independent_sample_id"])
            )
    return {bank_id: len(sample_ids) for bank_id, sample_ids in sorted(grouped.items())}


__all__ = [
    "FAILURE_TENSORS",
    "PROMPT_TENSORS",
    "SUCCESS_GATE_TENSORS",
    "V4_GATE_REACHABILITY_REPORT_SCHEMA",
    "V4_SOURCE_STATE_AUDIT_SCHEMA",
    "V4_SOURCE_STATE_CACHE_SCHEMA",
    "V4_SOURCE_STATE_EVENT_SCHEMA",
    "V4_SOURCE_STATE_MAX_WINDOW",
    "V4_SOURCE_STATE_TENSORS",
    "V4_SOURCE_STATE_WINDOWS",
    "V4SourceStateCache",
    "build_gate_reachability_report",
    "build_source_state_manifest",
    "derive_window_mean",
    "finalize_event",
    "independent_support",
    "load_source_state_cache",
    "save_source_state_cache",
    "validate_events",
    "validate_source_state_manifest",
    "validate_tensor_alignment",
]
