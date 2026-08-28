#!/usr/bin/env python3
"""Freeze the answer-blind V3.5 dynamic-margin selector.

The calibration consumes only authenticated calibration-val retrieval traces.
It deliberately never reads completions, answers, rewards, strict correctness,
or format correctness.  Static applicability parameters come from the already
qualified V3.5 applicability artifact; this script only freezes the inclusive
top1-top2 dynamic-margin threshold inside each question's fixed shortlist.
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
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memgen.experience.phase1 import canonical_json_sha256, file_sha256
from memgen.experience.v3 import V35_SYSTEM_PROFILE_SCHEMA
from memgen.experience.v3_5_selector import (
    V35_DUAL_KEY_BANK_SCHEMA,
    V35_RETRIEVAL_DECISION_SCHEMA,
    V35_SELECTOR_CALIBRATION_SCHEMA,
    V35_SELECTOR_POLICY,
    load_v35_applicability_calibration,
    retained_dynamic_margin_threshold,
    v35_artifact_sha256,
)


EVALUATION_PROFILE_SCHEMA = "experience-memory-v3.5-evaluation-profile-v1"
EVALUATION_ROW_SCHEMA = "experience-memory-v3.5-evaluation-row-v1"
V35_GENERATION_RESULT_SCHEMA = "experience-memory-v3.5-generation-result-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-profile", type=Path, required=True)
    parser.add_argument(
        "--dual-key-manifest", "--dual-selector-manifest",
        dest="dual_key_manifest", type=Path, required=True,
    )
    parser.add_argument(
        "--applicability-calibration", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-retained-fraction", type=float, default=0.5)
    parser.add_argument(
        "--minimum-first-attempts", "--minimum-triggered-samples",
        dest="minimum_first_attempts", type=int, default=32,
    )
    parser.add_argument(
        "--maximum-insufficient-shortlist-fraction",
        type=float,
        default=0.25,
        help=(
            "Fail closed when too many calibration questions have fewer than "
            "two post-floor candidates."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def evaluation_profile_sha256(value: Mapping[str, Any]) -> str:
    material = {
        key: item
        for key, item in value.items()
        if key not in {"created_at", "repository", "profile_sha256"}
    }
    repository = value.get("repository", {})
    material["code_identity"] = {
        "git_revision": repository.get("git_revision"),
        "tracked_diff_sha256": repository.get("tracked_diff_sha256"),
        "implementation_set_sha256": repository.get(
            "implementation_set_sha256"
        ),
    }
    return canonical_json_sha256(material)


def load_profile(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    system = value.get("system_profile", {})
    inputs = value.get("inputs", {})
    logging = value.get("logging", {})
    decision_contract = value.get("selector_decision_data_contract", {})
    trace_only = value.get("calibration_trace_only")
    if trace_only is None:
        trace_only = system.get("calibration_trace_only")
    task_results_used = value.get("task_results_used_for_selector_decision")
    dynamic_threshold = system.get(
        "minimum_dynamic_top1_top2_margin",
        system.get("retrieval_min_top1_top2_margin"),
    )
    abstention = system.get("retrieval_abstention_policy")
    trace_policy_ok = (
        "retrieval_abstention_policy" in system
        and abstention == "disabled"
        and "retrieval_min_top1_top2_margin" in system
        and dynamic_threshold is None
    )
    if (
        value.get("schema_version") != EVALUATION_PROFILE_SCHEMA
        or value.get("profile_sha256") != evaluation_profile_sha256(value)
        or value.get("system_version") != "v3.5"
        or value.get("logical_split") != "calibration-val"
        or int(value.get("selected_sample_count", -1)) != 64
        or int(value.get("slice", {}).get("offset", -1)) != 0
        or int(value.get("slice", {}).get("limit", -1)) != 64
        or not isinstance(value.get("selected_sample_ids_sha256"), str)
        or not value.get("selected_sample_ids_sha256")
        or system.get("schema_version") != V35_SYSTEM_PROFILE_SCHEMA
        or int(system.get("layer_number", -1)) != 24
        or system.get("query_context")
        != "question_plus_full_partial_cot"
        or system.get("query_encoder_state")
        != "pure_prefix_reencode_side_kv_disabled"
        or system.get("query_pooling") != "current_generated_token"
        or system.get("query_normalization") != "l2"
        or system.get("retrieval_method") != "exact_cosine"
        or system.get("retrieval_embedding_transform") != "none"
        or int(system.get("retrieval_top_k", -1)) != 2
        or int(system.get("selected_memory_count", -1)) != 1
        or system.get("boundary_policy")
        != "none_pre_answer_every_generated_token"
        or system.get("gate_policy")
        != "continuous_token_entropy_risk_hysteresis"
        or system.get("risk_role") != "online_joint_control"
        or system.get("rearm_policy")
        != "two_consecutive_low_entropy_tokens_rearm_without_trigger"
        or int(system.get("rearm_low_entropy_token_count", -1)) != 2
        or int(system.get("max_retrieval_attempts", -1)) != 3
        or system.get("selector_policy") != V35_SELECTOR_POLICY
        or system.get("replacement_policy") != "replace_current_memory"
        or system.get("duplicate_policy")
        != "consume_attempt_keep_current_memory"
        or system.get("abstain_policy")
        != "terminal_consume_attempt_clear_current_memory"
        or system.get("injection_policy")
        != "persistent_until_replace_terminal_abstain_or_eos"
        or system.get("memory_score_normalization") != "log_valid_slots"
        or not math.isclose(
            float(system.get("memory_score_bias", float("nan"))),
            math.log(10.0),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or system.get("attention_backend") != "sdpa"
        or trace_only is not True
        or system.get("calibration_trace_only") is not True
        or task_results_used is not False
        or decision_contract.get("task_accuracy_used") is not False
        or decision_contract.get("answer_or_reward_used") is not False
        or decision_contract.get("first_attempt_dynamic_margins_only") is not True
        or not trace_policy_ok
        or logging.get("query_embeddings_sidecar") is not True
        or logging.get("query_embeddings_sidecar_required_for_calibration")
        is not True
        or logging.get("query_embedding_sidecar_representation")
        != "dynamic_query_l2_normalized_exact_audit"
        or logging.get("raw_query_token_ids_saved") is not True
        or logging.get("calibration_query_hash_reproduction_required") is not True
        or not isinstance(inputs.get("risk_artifact_sha256"), str)
        or not inputs.get("risk_artifact_sha256")
    ):
        raise ValueError(
            "Dynamic calibration requires an authenticated answer-blind V3.5 "
            "calibration-val trace-only profile"
        )
    return value


def load_dual_key_manifest(
    path: Path, *, expected_file_sha256: str
) -> dict[str, Any]:
    if not expected_file_sha256 or file_sha256(path) != expected_file_sha256:
        raise ValueError("Dual-key manifest differs from the calibration run")
    value = json.loads(path.read_text(encoding="utf-8"))
    records = value.get("records")
    expected = value.get("manifest_sha256")
    actual = canonical_json_sha256({
        key: item for key, item in value.items() if key != "manifest_sha256"
    })
    memory_ids = [str(item.get("memory_id", "")) for item in records or []]
    if (
        value.get("schema_version") != V35_DUAL_KEY_BANK_SCHEMA
        or not expected
        or expected != actual
        or not isinstance(records, list)
        or not records
        or len(memory_ids) != len(set(memory_ids))
        or any(not memory_id for memory_id in memory_ids)
        or int(value.get("record_count", -1)) != len(records)
    ):
        raise ValueError("Invalid V3.5 dual-key manifest")
    return value


def load_dynamic_key_embeddings(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> Any:
    """Authenticate and load the CPU dynamic bank used for independent replay."""

    import torch
    from safetensors.torch import load_file
    from memgen.model.retrieval_keys import tensor_sha256

    artifact = manifest.get("tensor_artifact", {})
    relative = Path(str(artifact.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe V3.5 dual-key tensor path")
    tensor_path = (manifest_path.parent / relative).resolve()
    if manifest_path.parent.resolve() not in tensor_path.parents:
        raise ValueError("V3.5 dual-key tensor escaped its artifact directory")
    if (
        not tensor_path.is_file()
        or artifact.get("sha256") != file_sha256(tensor_path)
    ):
        raise ValueError("V3.5 dual-key tensor hash mismatch")
    tensors = load_file(str(tensor_path), device="cpu")
    if set(tensors) != {
        "applicability_key_embeddings",
        "dynamic_key_embeddings",
    }:
        raise ValueError("Unexpected V3.5 dual-key tensor names")
    dynamic = tensors["dynamic_key_embeddings"].detach().float().cpu()
    records = list(manifest["records"])
    if (
        dynamic.ndim != 2
        or dynamic.shape[0] != len(records)
        or not torch.isfinite(dynamic).all()
    ):
        raise ValueError("Invalid V3.5 dynamic embedding matrix")
    for index, record in enumerate(records):
        vector = dynamic[index]
        if (
            tensor_sha256(vector) != record.get("dynamic_key_embedding_sha256")
            or not math.isclose(
                float(vector.norm().item()), 1.0, rel_tol=0.0, abs_tol=1e-5
            )
        ):
            raise ValueError("V3.5 dynamic embedding record/hash mismatch")
    return dynamic


def _row_sha256(row: Mapping[str, Any]) -> str:
    return canonical_json_sha256({
        key: item
        for key, item in row.items()
        if key not in {"created_at", "row_sha256"}
    })


def _runtime(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return row.get("conditions", {}).get("v3", {}).get("runtime_trace", {})


def _shortlist_ids(static_trace: Mapping[str, Any]) -> list[str]:
    direct = static_trace.get("shortlist_memory_ids")
    if isinstance(direct, list):
        return [str(value) for value in direct]
    entries = static_trace.get("post_floor_shortlist", [])
    return [
        str(item.get("memory_id", ""))
        for item in entries
        if isinstance(item, Mapping)
    ]


def _decision_query(decision: Mapping[str, Any]) -> Mapping[str, Any]:
    dynamic = decision.get("dynamic_query")
    return dynamic if isinstance(dynamic, Mapping) else decision.get("query", {})


def _decision_hits(decision: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    hits = decision.get("dynamic_hits", decision.get("hits", []))
    return list(hits) if isinstance(hits, list) else []


def _decision_shortlist_ids(
    decision: Mapping[str, Any], static_trace: Mapping[str, Any]
) -> list[str]:
    values = decision.get("static_shortlist")
    if values is None:
        values = decision.get("static_shortlist_ids")
    if isinstance(values, list):
        return [
            str(item.get("memory_id")) if isinstance(item, Mapping) else str(item)
            for item in values
        ]
    return _shortlist_ids(static_trace)


def _ranked_hits_are_stable(
    hits: Sequence[Mapping[str, Any]], *, score_field: str
) -> bool:
    try:
        pairs = [
            (float(hit[score_field]), str(hit["memory_id"])) for hit in hits
        ]
    except (KeyError, TypeError, ValueError):
        return False
    return (
        all(math.isfinite(score) and memory_id for score, memory_id in pairs)
        and all(
            left_score > right_score
            or (
                left_score == right_score
                and left_memory_id < right_memory_id
            )
            for (left_score, left_memory_id), (
                right_score, right_memory_id
            ) in zip(pairs, pairs[1:])
        )
    )


def _verify_query_sidecar(
    row: Mapping[str, Any],
    *,
    root: Path,
    attempts: Sequence[Mapping[str, Any]],
    sample_id: str,
) -> Mapping[str, Any]:
    from safetensors.torch import load_file
    from memgen.model.retrieval_keys import tensor_sha256

    descriptor = row.get("conditions", {}).get("v3", {}).get(
        "query_embedding_sidecar"
    )
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"Missing V3.5 query embedding sidecar: {sample_id}")
    relative = Path(str(descriptor.get("path", "")))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe V3.5 query sidecar path: {sample_id}")
    sidecar = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root not in sidecar.parents:
        raise ValueError(f"V3.5 query sidecar escaped its run: {sample_id}")
    if (
        not sidecar.is_file()
        or descriptor.get("sha256") != file_sha256(sidecar)
        or int(descriptor.get("attempt_count", -1)) != len(attempts)
    ):
        raise ValueError(f"Invalid V3.5 query sidecar descriptor: {sample_id}")
    tensors = load_file(str(sidecar), device="cpu")
    expected_names = {
        f"attempt_{index:02d}" for index in range(1, len(attempts) + 1)
    }
    if set(tensors) != expected_names:
        raise ValueError(f"V3.5 query sidecar attempt set drifted: {sample_id}")
    for index, attempt in enumerate(attempts, start=1):
        tensor = tensors[f"attempt_{index:02d}"].detach().float().cpu()
        query = _decision_query(attempt.get("retrieval_decision", {}))
        flattened = tensor.reshape(-1)
        norm = float(flattened.norm().item())
        expected_hash = query.get("query_embedding_sha256")
        if (
            tensor.numel() == 0
            or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-5)
            or expected_hash != tensor_sha256(flattened)
            or not math.isclose(
                norm,
                float(query.get("query_embedding_norm", float("nan"))),
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            raise ValueError(
                f"V3.5 query sidecar/hash reproduction failed: {sample_id}"
            )
    return tensors


def retained_margin_threshold(
    margins: Sequence[float], *, target_retained_fraction: float
) -> dict[str, Any]:
    """Compatibility wrapper around the frozen V3.5 pure contract."""

    return retained_dynamic_margin_threshold(
        margins, target_retained_fraction=target_retained_fraction
    )


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    normalized = sorted(float(value) for value in values)
    if not normalized or not all(math.isfinite(value) for value in normalized):
        raise ValueError("Cannot summarize invalid dynamic margins")

    def percentile(q: float) -> float:
        position = (len(normalized) - 1) * q
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return normalized[lower]
        weight = position - lower
        return normalized[lower] * (1.0 - weight) + normalized[upper] * weight

    return {
        "count": len(normalized),
        "min": normalized[0],
        "mean": sum(normalized) / len(normalized),
        "median": percentile(0.5),
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "max": normalized[-1],
    }


def collect_first_attempts(
    path: Path,
    *,
    profile_sha256: str,
    known_memory_ids: set[str],
    expected_shortlist_k: int | None = None,
    expected_score_floor: float | None = None,
    expected_manifest_sha256: str | None = None,
    sidecar_root: Path | None = None,
    dynamic_embeddings: Any | None = None,
    ordered_memory_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    margins: list[float] = []
    selected_memory_ids: list[str] = []
    row_count = 0
    insufficient_shortlist_count = 0
    available_static_count = 0
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            if (
                row.get("schema_version") != EVALUATION_ROW_SCHEMA
                or row.get("profile_sha256") != profile_sha256
                or row.get("row_sha256") != _row_sha256(row)
                or not sample_id
                or sample_id in seen
            ):
                raise ValueError(
                    f"Invalid V3.5 calibration row at line {line_number}"
                )
            seen.add(sample_id)
            row_count += 1
            runtime = _runtime(row)
            if runtime.get("schema_version") != V35_GENERATION_RESULT_SCHEMA:
                raise ValueError(f"Invalid V3.5 runtime trace: {sample_id}")
            static_trace = runtime.get("static_selector_trace")
            if not isinstance(static_trace, Mapping):
                raise ValueError(f"Missing V3.5 static selector trace: {sample_id}")
            shortlist_ids = _shortlist_ids(static_trace)
            static_query = static_trace.get("query", {})
            raw_static_ids = static_query.get("static_question_token_ids")
            post_floor = list(static_trace.get("post_floor_shortlist", []))
            pre_floor = list(static_trace.get("pre_floor_top_k", []))
            try:
                logged_shortlist_k = int(static_trace.get("shortlist_k", -1))
                logged_score_floor = float(
                    static_trace.get("score_floor", float("nan"))
                )
                pre_floor_ids = [
                    str(hit["memory_id"]) for hit in pre_floor
                ]
                post_floor_ids = [
                    str(hit["memory_id"]) for hit in post_floor
                ]
                expected_post_floor = [
                    hit
                    for hit in pre_floor
                    if float(hit["static_score"]) >= logged_score_floor
                ]
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"Invalid static selector reproduction: {sample_id}"
                ) from None
            if len(post_floor) >= 2:
                expected_unavailable_reason = None
            elif not pre_floor:
                expected_unavailable_reason = "empty_bank"
            elif not post_floor:
                expected_unavailable_reason = "below_applicability_floor"
            else:
                expected_unavailable_reason = "insufficient_shortlist"
            static_trace_ok = (
                static_trace.get("schema_version")
                == "experience-memory-v3.5-static-shortlist-v1"
                and isinstance(static_query, Mapping)
                and isinstance(raw_static_ids, list)
                and raw_static_ids
                and all(isinstance(value, int) for value in raw_static_ids)
                and int(static_query.get("static_question_token_count", -1))
                == len(raw_static_ids)
                and static_query.get("static_question_token_ids_sha256")
                == canonical_json_sha256(raw_static_ids)
                and bool(static_query.get("static_question_text_sha256"))
                and static_query.get("static_question_text_sha256")
                == row.get("question_sha256")
                and bool(static_query.get("static_question_embedding_sha256"))
                and math.isclose(
                    float(static_query.get(
                        "static_question_embedding_norm", float("nan")
                    )),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
                and int(static_query.get("layer_number", -1)) == 24
                and static_query.get("schema_version")
                == "experience-memory-v3.5-static-question-query-v1"
                and static_query.get("representation")
                == "decoder_layer_output"
                and static_query.get("pooling") == "last_valid_token"
                and static_query.get("normalization") == "l2"
                and static_query.get("side_kv_disabled") is True
                and static_query.get("chat_wrapper_included") is False
                and static_query.get("prompt_boilerplate_included") is False
                and static_query.get("add_special_tokens") is False
                and static_trace.get("shortlist_fixed_for_generation") is True
                and static_trace.get("retrieval_method") == "exact_cosine"
                and static_trace.get("stable_tie_break")
                == "memory_id_ascending"
                and static_trace.get("score_floor_tie_policy")
                == "retain_score_greater_than_or_equal_to_floor"
                and math.isfinite(logged_score_floor)
                and -1.0 <= logged_score_floor <= 1.0
                and 1 <= logged_shortlist_k <= 32
                and (
                    expected_shortlist_k is None
                    or logged_shortlist_k == expected_shortlist_k
                )
                and (
                    expected_score_floor is None
                    or math.isclose(
                        logged_score_floor,
                        expected_score_floor,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                )
                and (
                    expected_manifest_sha256 is None
                    or static_trace.get("applicability_bank_manifest_sha256")
                    == expected_manifest_sha256
                )
                and len(pre_floor)
                == min(logged_shortlist_k, len(known_memory_ids))
                and len(post_floor) <= logged_shortlist_k
                and pre_floor_ids
                == list(dict.fromkeys(pre_floor_ids))
                and post_floor_ids
                == list(dict.fromkeys(post_floor_ids))
                and all(
                    memory_id in known_memory_ids for memory_id in pre_floor_ids
                )
                and all(
                    memory_id in known_memory_ids for memory_id in post_floor_ids
                )
                and _ranked_hits_are_stable(
                    pre_floor, score_field="static_score"
                )
                and _ranked_hits_are_stable(
                    post_floor, score_field="static_score"
                )
                and [
                    int(hit.get("original_global_rank", -1))
                    for hit in pre_floor
                ]
                == list(range(1, len(pre_floor) + 1))
                and post_floor == expected_post_floor
                and shortlist_ids == post_floor_ids
                and static_trace.get("shortlist_nonempty")
                is bool(post_floor)
                and static_trace.get("unavailable_reason")
                == expected_unavailable_reason
                and static_trace.get("static_selector_unavailable")
                is (expected_unavailable_reason is not None)
            )
            if not static_trace_ok:
                raise ValueError(f"Invalid static selector reproduction: {sample_id}")
            if (
                len(shortlist_ids) != len(set(shortlist_ids))
                or any(memory_id not in known_memory_ids for memory_id in shortlist_ids)
            ):
                raise ValueError(f"Invalid static shortlist: {sample_id}")
            static_unavailable = static_trace.get(
                "static_selector_unavailable"
            ) is True
            if static_unavailable or len(shortlist_ids) < 2:
                insufficient_shortlist_count += 1
                if runtime.get("retrieval_attempts"):
                    raise ValueError(
                        f"Static-unavailable sample has retrieval attempts: {sample_id}"
                    )
                continue
            available_static_count += 1
            attempts = list(runtime.get("retrieval_attempts", []))
            if not attempts:
                continue
            sidecar_tensors: Mapping[str, Any] | None = None
            if sidecar_root is not None:
                sidecar_tensors = _verify_query_sidecar(
                    row,
                    root=sidecar_root,
                    attempts=attempts,
                    sample_id=sample_id,
                )
            first = attempts[0]
            if int(first.get("attempt_number", -1)) != 1:
                raise ValueError(f"First attempt is not numbered one: {sample_id}")
            decision = first.get("retrieval_decision", {})
            query = _decision_query(decision)
            hits = _decision_hits(decision)
            decision_shortlist = _decision_shortlist_ids(decision, static_trace)
            selected_id = first.get(
                "selected_memory_id", decision.get("selected_memory_id")
            )
            if selected_id is None and decision.get("matched_memory"):
                selected_id = decision["matched_memory"].get("memory_id")
            selected_id = str(selected_id) if selected_id is not None else None
            margin = decision.get(
                "dynamic_top1_top2_margin",
                query.get("top1_top2_margin"),
            )
            observation_index = int(first.get(
                "generated_observation_index",
                first.get("generated_boundary_index", -1),
            ))
            prompt_count = int(query.get("prompt_token_count", -1))
            partial_count = int(query.get("partial_cot_token_count", -1))
            query_count = int(query.get("query_token_count", -1))
            raw_query_ids = query.get("query_token_ids")
            encoder_state = query.get("encoder_state")
            side_kv_disabled = query.get("side_kv_disabled")
            if side_kv_disabled is None:
                side_kv_disabled = (
                    encoder_state == "pure_prefix_reencode_side_kv_disabled"
                )
            hit_ids = [str(hit.get("memory_id", "")) for hit in hits]
            hit_scores = [float(hit.get("score", float("nan"))) for hit in hits]
            recomputed_margin = (
                hit_scores[0] - hit_scores[1]
                if len(hit_scores) >= 2
                else float("nan")
            )
            stable_hit_order = all(
                hit_scores[index] > hit_scores[index + 1]
                or (
                    hit_scores[index] == hit_scores[index + 1]
                    and hit_ids[index] < hit_ids[index + 1]
                )
                for index in range(len(hit_scores) - 1)
            )
            logged_top1 = query.get("top1_score")
            logged_top2 = query.get("top2_score")
            independent_reproduction_ok = True
            if dynamic_embeddings is not None or ordered_memory_ids is not None:
                if (
                    dynamic_embeddings is None
                    or ordered_memory_ids is None
                    or sidecar_tensors is None
                ):
                    independent_reproduction_ok = False
                else:
                    import torch

                    memory_index = {
                        str(memory_id): index
                        for index, memory_id in enumerate(ordered_memory_ids)
                    }
                    query_tensor = (
                        sidecar_tensors["attempt_01"]
                        .reshape(-1)
                        .float()
                        .contiguous()
                    )
                    shortlist_indices = [
                        memory_index[memory_id] for memory_id in shortlist_ids
                    ]
                    replay_scores_tensor = torch.mv(
                        dynamic_embeddings[shortlist_indices], query_tensor
                    )
                    replay = sorted(
                        (
                            (float(replay_scores_tensor[position].item()), memory_id)
                            for position, memory_id in enumerate(shortlist_ids)
                        ),
                        key=lambda item: (-item[0], item[1]),
                    )[:2]
                    independent_reproduction_ok = (
                        len(replay) == 2
                        and hit_ids[:2] == [item[1] for item in replay]
                        and hit_scores[:2]
                        == [item[0] for item in replay]
                        and float(margin)
                        == replay[0][0] - replay[1][0]
                    )
            if (
                decision.get("schema_version")
                != V35_RETRIEVAL_DECISION_SCHEMA
                or decision.get("status") != "selected"
                or selected_id is None
                or selected_id not in known_memory_ids
                or decision_shortlist != shortlist_ids
                or selected_id not in shortlist_ids
                or len(hits) < 2
                or hit_ids[0] != selected_id
                or any(memory_id not in shortlist_ids for memory_id in hit_ids)
                or not all(math.isfinite(score) for score in hit_scores)
                or hit_scores != sorted(hit_scores, reverse=True)
                or not stable_hit_order
                or margin is None
                or not math.isfinite(float(margin))
                or float(margin) < 0.0
                or not math.isclose(
                    float(margin),
                    recomputed_margin,
                    rel_tol=1e-7,
                    abs_tol=1e-7,
                )
                or not independent_reproduction_ok
                or logged_top1 is None
                or logged_top2 is None
                or not math.isclose(
                    float(logged_top1), hit_scores[0], rel_tol=1e-7, abs_tol=1e-7
                )
                or not math.isclose(
                    float(logged_top2), hit_scores[1], rel_tol=1e-7, abs_tol=1e-7
                )
                or query_count != prompt_count + partial_count
                or partial_count != observation_index + 1
                or not isinstance(raw_query_ids, list)
                or len(raw_query_ids) != query_count
                or not all(isinstance(value, int) for value in raw_query_ids)
                or query.get("query_token_ids_sha256")
                != canonical_json_sha256(raw_query_ids)
                or canonical_json_sha256(raw_query_ids[:prompt_count])
                != row.get("prompt_token_ids_sha256")
                or raw_query_ids[prompt_count:]
                != list(runtime.get("completion_token_ids", []))[:partial_count]
                or query.get("context") != "question_plus_full_partial_cot"
                or query.get("method")
                != "exact_cosine_within_static_applicability_shortlist"
                or query.get("encoder_state")
                != "pure_prefix_reencode_side_kv_disabled"
                or side_kv_disabled is not True
                or not query.get("query_embedding_sha256")
                or int(query.get("layer_number", -1)) != 24
                or query.get("pooling") != "current_generated_token"
                or query.get("normalization") != "l2"
                or int(query.get("encoded_full_prefix_token_count", -1))
                != query_count
                or int(query.get("query_embedding_token_index", -1))
                != query_count - 1
                or int(query.get(
                    "query_embedding_causal_context_token_count", -1
                ))
                != query_count
                or int(query.get("query_embedding_token_id", -1))
                != raw_query_ids[-1]
                or int(first.get("query_embedding_token_id", -2))
                != raw_query_ids[-1]
                or not math.isclose(
                    float(query.get("query_embedding_norm", float("nan"))),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-5,
                )
                or query.get("static_shortlist_fixed_for_generation") is not True
                or query.get(
                    "dynamic_search_restricted_to_static_shortlist"
                )
                is not True
                or int(query.get("dynamic_search_candidate_count", -1))
                != len(shortlist_ids)
                or query.get("selected_memory_kv_metadata_aligned") is not True
                or not math.isclose(
                    float(query.get(
                        "minimum_applicability_score", float("nan")
                    )),
                    float(static_trace.get("score_floor")),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"First-attempt dynamic query failed reproduction/integrity: {sample_id}"
                )
            margins.append(float(margin))
            selected_memory_ids.append(selected_id)
    return {
        "margins": margins,
        "selected_memory_ids": selected_memory_ids,
        "row_count": row_count,
        "available_static_count": available_static_count,
        "insufficient_shortlist_count": insufficient_shortlist_count,
    }


def markdown_report(value: Mapping[str, Any]) -> str:
    calibration = value["calibration"]
    return "\n".join([
        "# MemGen V3.5 dynamic selector calibration",
        "",
        f"- Status: `{value['status']}`",
        f"- Source split: `{value['source']['logical_split']}`",
        f"- First-attempt sample count: {calibration['sample_count']}",
        f"- Static shortlist k: {calibration['shortlist_k']}",
        f"- Applicability score floor: `{calibration['minimum_applicability_score']}`",
        f"- Dynamic top1-top2 margin: `{calibration['minimum_dynamic_top1_top2_margin']}`",
        f"- Target retained fraction: {calibration['target_retained_fraction']}",
        f"- Actual inclusive retained fraction: {calibration['actual_retained_fraction']}",
        "",
        "The threshold is answer-blind and uses only each question's first "
        "attempt inside its frozen static applicability shortlist. Ties at the "
        "threshold are retained.",
        "",
    ])


def main() -> None:
    args = parse_args()
    if (
        not 0.0 < args.target_retained_fraction <= 1.0
        or args.minimum_first_attempts <= 0
        or not 0.0 <= args.maximum_insufficient_shortlist_fraction < 1.0
    ):
        raise ValueError("Invalid V3.5 dynamic calibration limits")
    profile = load_profile(args.run_profile)
    inputs = profile.get("inputs", {})
    dual_file_sha256 = str(inputs.get("dual_key_manifest_sha256", ""))
    dual_manifest = load_dual_key_manifest(
        args.dual_key_manifest,
        expected_file_sha256=dual_file_sha256,
    )
    applicability_file_sha256 = str(
        inputs.get("applicability_calibration_sha256", "")
    )
    if (
        not applicability_file_sha256
        or file_sha256(args.applicability_calibration)
        != applicability_file_sha256
    ):
        raise ValueError(
            "Applicability calibration differs from the calibration run"
        )
    applicability = load_v35_applicability_calibration(
        args.applicability_calibration
    )
    if (
        applicability.get("status") != "passed"
        or applicability.get("task_accuracy_used") is not False
        or applicability.get("answer_or_reward_used") is not False
        or applicability.get("source", {}).get("dual_key_manifest_sha256")
        not in {dual_file_sha256, dual_manifest.get("manifest_sha256")}
        or not all(applicability.get("requirements", {}).values())
    ):
        raise ValueError("Applicability calibration is not qualified and bound")
    calibration = applicability.get("calibration", {})
    shortlist_k = int(calibration.get("shortlist_k", 0))
    score_floor = calibration.get("minimum_applicability_score")
    if (
        shortlist_k < 1
        or shortlist_k > 32
        or score_floor is None
        or not math.isfinite(float(score_floor))
    ):
        raise ValueError("Invalid frozen applicability parameters")
    system = profile.get("system_profile", {})
    if (
        int(system.get("applicability_shortlist_k", -1)) != shortlist_k
        or not math.isclose(
            float(system.get("applicability_score_floor", float("nan"))),
            float(score_floor),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "Calibration profile static selector differs from applicability artifact"
        )
    known_memory_ids = {
        str(item["memory_id"]) for item in dual_manifest["records"]
    }
    ordered_memory_ids = [
        str(item["memory_id"]) for item in dual_manifest["records"]
    ]
    dynamic_embeddings = load_dynamic_key_embeddings(
        args.dual_key_manifest, dual_manifest
    )
    collected = collect_first_attempts(
        args.results,
        profile_sha256=str(profile["profile_sha256"]),
        known_memory_ids=known_memory_ids,
        expected_shortlist_k=shortlist_k,
        expected_score_floor=float(score_floor),
        expected_manifest_sha256=str(dual_manifest["manifest_sha256"]),
        sidecar_root=args.results.parent,
        dynamic_embeddings=dynamic_embeddings,
        ordered_memory_ids=ordered_memory_ids,
    )
    if collected["row_count"] != int(profile.get("selected_sample_count", -1)):
        raise ValueError("V3.5 calibration results are incomplete")
    margins = collected["margins"]
    if len(margins) < args.minimum_first_attempts:
        raise ValueError("Too few first attempts for V3.5 dynamic calibration")
    insufficient_fraction = (
        collected["insufficient_shortlist_count"] / collected["row_count"]
    )
    if insufficient_fraction > args.maximum_insufficient_shortlist_fraction:
        raise ValueError(
            "Too many calibration questions have an insufficient static shortlist"
        )
    threshold = retained_margin_threshold(
        margins,
        target_retained_fraction=args.target_retained_fraction,
    )
    selection_counts = Counter(collected["selected_memory_ids"])
    artifact: dict[str, Any] = {
        "schema_version": V35_SELECTOR_CALIBRATION_SCHEMA,
        "created_at": utc_now(),
        "status": "passed",
        "policy": V35_SELECTOR_POLICY,
        "task_accuracy_used": False,
        "answer_or_reward_used": False,
        "source": {
            "logical_split": "calibration-val",
            "scope": "first_retrieval_attempt_per_triggered_question",
            "run_profile_sha256": profile["profile_sha256"],
            "run_profile_file_sha256": file_sha256(args.run_profile),
            "results_file_sha256": file_sha256(args.results),
            "dual_key_manifest_sha256": dual_file_sha256,
            "dual_key_manifest_logical_sha256": dual_manifest[
                "manifest_sha256"
            ],
            "applicability_calibration_sha256": (
                applicability_file_sha256
            ),
            "applicability_calibration_artifact_sha256": applicability[
                "artifact_sha256"
            ],
            "risk_artifact_sha256": inputs.get("risk_artifact_sha256"),
            "system_version": "v3.5",
            "system_profile_schema": profile.get("system_profile", {}).get(
                "schema_version"
            ),
            "calibration_trace_only": True,
            "completed_sample_count": collected["row_count"],
        },
        "calibration": {
            "sample_count": len(margins),
            "shortlist_k": shortlist_k,
            "minimum_applicability_score": float(score_floor),
            "applicability_score_floor_tie_policy": calibration[
                "applicability_score_floor_tie_policy"
            ],
            **threshold,
            "static_selector_available_sample_count": collected[
                "available_static_count"
            ],
            "insufficient_shortlist_sample_count": collected[
                "insufficient_shortlist_count"
            ],
            "insufficient_shortlist_fraction": insufficient_fraction,
            "first_attempt_selected_memory_count": len(selection_counts),
            "first_attempt_selected_memory_frequency": [
                {"memory_id": memory_id, "count": count}
                for memory_id, count in sorted(
                    selection_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ],
        },
        "requirements": {
            "source_is_calibration_val": True,
            "source_profile_is_authenticated": True,
            "source_rows_are_complete_and_authenticated": True,
            "source_is_trace_only": True,
            "first_attempt_only": True,
            "task_accuracy_not_used": True,
            "answer_or_reward_not_used": True,
            "dual_key_manifest_is_authenticated_and_bound": True,
            "applicability_calibration_is_authenticated_and_bound": True,
            "static_shortlist_is_fixed_and_bound": True,
            "dynamic_queries_are_full_prefix_and_authenticated": True,
            "dynamic_queries_disable_side_kv": True,
            "dynamic_rerank_is_inside_static_shortlist": True,
            "first_attempt_sample_count_sufficient": True,
            "insufficient_shortlist_fraction_acceptable": True,
            "threshold_is_finite_and_nonnegative": True,
            "inclusive_tie_policy": True,
        },
    }
    artifact["artifact_sha256"] = v35_artifact_sha256(artifact)
    write_json_atomic(args.output, artifact)
    args.output.with_suffix(".md").write_text(
        markdown_report(artifact), encoding="utf-8"
    )
    print(
        f"[v3.5-calibration] first_attempts={len(margins)} "
        f"threshold={threshold['minimum_dynamic_top1_top2_margin']:.9g} "
        f"retained={threshold['actual_retained_fraction']:.4f} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
