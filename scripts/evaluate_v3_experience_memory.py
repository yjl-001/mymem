#!/usr/bin/env python3
"""Resumable vanilla-vs-V3 GSM8K evaluation with full online diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.gsm8k.prompt import GSM8K_PROMPT_CONTRACT
from data.utils.math_utils import GSM8K_VERIFIER_VERSION, diagnose_gsm8k_completion
from memgen.chat_templates import CONVERSATION_TEMPLATE
from memgen.experience.memory import MemoryRecord
from memgen.experience.phase1 import (
    SPLIT_MANIFEST_SCHEMA,
    canonical_json_sha256,
    file_sha256,
    iter_jsonl,
    text_sha256,
)
from memgen.experience.risk import (
    ENTROPY_RISK_ARTIFACT_SCHEMA,
    TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA,
)
from memgen.experience.v3 import (
    ExperienceMemoryV3Profile,
    V34_QUERY_POOLING_CURRENT_TOKEN,
    V3_QUERY_POOLING_BOUNDARY_LAST,
    V3_QUERY_POOLING_METHODS,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
    V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
)
from memgen.experience.v3_selector import (
    load_margin_selector_calibration,
    selector_calibration_query_pooling,
)
from memgen.experience.v3_artifacts import (
    authenticate_e0_inputs,
    load_formal_e0_report,
    load_v3_offline_report,
    validate_cross_bank_metadata,
)
from memgen.experience.v3_eval import summarize_v3_rows


V3_EVAL_PROFILE_SCHEMA = "experience-memory-v3-evaluation-profile-v1"
V3_EVAL_ROW_SCHEMA = "experience-memory-v3-evaluation-row-v1"
V3_EVAL_REPORT_SCHEMA = "experience-memory-v3-evaluation-report-v1"
V35_EVAL_PROFILE_SCHEMA = "experience-memory-v3.5-evaluation-profile-v1"
V35_EVAL_ROW_SCHEMA = "experience-memory-v3.5-evaluation-row-v1"
V35_EVAL_REPORT_SCHEMA = "experience-memory-v3.5-evaluation-report-v1"
V35_QUERY_SIDECAR_REPRESENTATION = (
    "dynamic_query_l2_normalized_exact_audit"
)
LEGACY_QUERY_SIDECAR_REPRESENTATION = (
    "raw_unit_before_retrieval_embedding_transform"
)

_ANSWER_MARKER_RE = re.compile(
    r"(?:\\boxed|\\fbox|final\s+answer|answer\s+is)", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument(
        "--logical-split",
        choices=("calibration-val", "dev-test", "final-test"),
        required=True,
    )
    parser.add_argument("--memory-records", type=Path, required=True)
    parser.add_argument("--retrieval-key-manifest", type=Path, required=True)
    parser.add_argument("--side-kv-manifest", type=Path, required=True)
    parser.add_argument("--v3-offline-report", type=Path, required=True)
    parser.add_argument("--e0-final-report", type=Path, required=True)
    parser.add_argument("--risk-artifact", type=Path, required=True)
    parser.add_argument(
        "--system-version", choices=("v3", "v3.4", "v3.5"), default="v3"
    )
    parser.add_argument("--selector-calibration", type=Path)
    parser.add_argument(
        "--dual-key-manifest",
        type=Path,
        help="Required V3.5 applicability/dynamic dual-key manifest.",
    )
    parser.add_argument(
        "--applicability-calibration",
        type=Path,
        help="Required V3.5 frozen static-shortlist calibration artifact.",
    )
    parser.add_argument(
        "--calibration-trace-only",
        action="store_true",
        help=(
            "Collect answer-blind V3.5 first-attempt margins using the "
            "explicit trace-only profile. Only calibration-val is accepted."
        ),
    )
    parser.add_argument(
        "--retrieval-embedding-transform",
        choices=(
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED,
        ),
        default=V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
    )
    parser.add_argument(
        "--query-pooling",
        choices=tuple(sorted(V3_QUERY_POOLING_METHODS)),
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates the slice remainder")
    parser.add_argument("--parity-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=GSM8K_PROMPT_CONTRACT.max_new_tokens,
    )
    parser.add_argument("--save-query-embeddings", action="store_true")
    return parser.parse_args()


def evaluation_schemas(system_version: str) -> tuple[str, str, str]:
    """Keep V3.5 resumes physically distinct from legacy evaluation rows."""

    if system_version == "v3.5":
        return (
            V35_EVAL_PROFILE_SCHEMA,
            V35_EVAL_ROW_SCHEMA,
            V35_EVAL_REPORT_SCHEMA,
        )
    if system_version in {"v3", "v3.4"}:
        return V3_EVAL_PROFILE_SCHEMA, V3_EVAL_ROW_SCHEMA, V3_EVAL_REPORT_SCHEMA
    raise ValueError("Unexpected V3 evaluation system version")


def _resolve_and_validate_versioned_args(args: argparse.Namespace) -> None:
    """Apply version defaults and fail closed on cross-version artifacts."""

    continuous_version = args.system_version in {"v3.4", "v3.5"}
    if args.query_pooling is None:
        args.query_pooling = (
            V34_QUERY_POOLING_CURRENT_TOKEN
            if continuous_version
            else V3_QUERY_POOLING_BOUNDARY_LAST
        )
    if continuous_version and args.query_pooling != V34_QUERY_POOLING_CURRENT_TOKEN:
        raise ValueError(
            f"{args.system_version} requires current-generated-token query pooling"
        )
    if args.system_version == "v3.5":
        if getattr(args, "dtype", None) != "bfloat16":
            raise ValueError("V3.5 requires --dtype bfloat16")
        if args.logical_split == "final-test":
            raise ValueError(
                "V3.5 final-test is blocked pending separate user authorization"
            )
        if args.dual_key_manifest is None:
            raise ValueError("V3.5 requires --dual-key-manifest")
        if args.applicability_calibration is None:
            raise ValueError("V3.5 requires --applicability-calibration")
        if args.retrieval_embedding_transform != V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE:
            raise ValueError("V3.5 does not permit a legacy retrieval transform")
        if args.calibration_trace_only:
            if args.logical_split != "calibration-val":
                raise ValueError(
                    "V3.5 calibration trace-only mode requires calibration-val"
                )
            if args.selector_calibration is not None:
                raise ValueError(
                    "V3.5 calibration trace-only mode cannot use a final selector artifact"
                )
            if not args.save_query_embeddings:
                raise ValueError(
                    "V3.5 calibration trace-only mode requires --save-query-embeddings"
                )
        elif args.selector_calibration is None:
            raise ValueError("Final V3.5 evaluation requires --selector-calibration")
    elif (
        args.dual_key_manifest is not None
        or args.applicability_calibration is not None
        or args.calibration_trace_only
    ):
        raise ValueError("V3.5-only selector arguments require --system-version v3.5")


def _load_v35_profile_and_artifacts(
    args: argparse.Namespace,
) -> tuple[ExperienceMemoryV3Profile, dict[str, Any], dict[str, Any] | None]:
    """Authenticate V3.5 selector inputs and build the exact runtime profile."""

    from memgen.experience.v3_5_selector import (
        load_v35_applicability_calibration,
        load_v35_selector_calibration,
    )

    assert args.dual_key_manifest is not None
    assert args.applicability_calibration is not None
    dual_manifest_sha256 = file_sha256(args.dual_key_manifest)
    applicability = load_v35_applicability_calibration(
        args.applicability_calibration,
        expected_input_hashes={
            "dual_key_manifest_sha256": dual_manifest_sha256,
        },
    )
    applicability_sha256 = file_sha256(args.applicability_calibration)
    applicability_values = applicability["calibration"]
    if args.calibration_trace_only:
        profile = ExperienceMemoryV3Profile.applicability_aware_continuous(
            applicability_shortlist_k=int(
                applicability_values["shortlist_k"]
            ),
            applicability_score_floor=float(
                applicability_values["minimum_applicability_score"]
            ),
            retrieval_min_top1_top2_margin=None,
            calibration_trace_only=True,
        )
        return profile, applicability, None

    assert args.selector_calibration is not None
    selector = load_v35_selector_calibration(
        args.selector_calibration,
        expected_input_hashes={
            "dual_key_manifest_sha256": dual_manifest_sha256,
            "applicability_calibration_sha256": applicability_sha256,
            "risk_artifact_sha256": file_sha256(args.risk_artifact),
        },
    )
    selector_values = selector["calibration"]
    if (
        int(selector_values["shortlist_k"])
        != int(applicability_values["shortlist_k"])
        or abs(
            float(selector_values["minimum_applicability_score"])
            - float(applicability_values["minimum_applicability_score"])
        )
        > 1e-12
    ):
        raise ValueError(
            "V3.5 final selector differs from its applicability calibration"
        )
    profile = ExperienceMemoryV3Profile.applicability_aware_continuous(
        applicability_shortlist_k=int(selector_values["shortlist_k"]),
        applicability_score_floor=float(
            selector_values["minimum_applicability_score"]
        ),
        retrieval_min_top1_top2_margin=float(
            selector_values["minimum_dynamic_top1_top2_margin"]
        ),
        calibration_trace_only=False,
    )
    return profile, applicability, selector


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


def repository_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return ""

    revision = run("rev-parse", "HEAD") or None
    status = run("status", "--short")
    diff = run("diff", "--binary", "HEAD")
    implementation_paths = (
        "memgen/experience/risk.py",
        "memgen/experience/v3.py",
        "memgen/experience/v3_5_selector.py",
        "memgen/experience/v3_selector.py",
        "memgen/experience/v3_artifacts.py",
        "memgen/experience/v3_eval.py",
        "memgen/model/retrieval_keys.py",
        "memgen/model/v3_5_retrieval.py",
        "memgen/model/e1_runtime.py",
        "memgen/model/side_kv.py",
        "memgen/model/v3_runtime.py",
        "scripts/compile_v3_5_dual_selector.py",
        "scripts/calibrate_v3_5_dynamic_selector.py",
        "scripts/analyze_v3_evaluation.py",
        "scripts/compare_v3_5_applicability_selector.py",
        "scripts/qualify_v3_5_dev.py",
        "scripts/run_online_experience_memory_v3.py",
        "scripts/evaluate_v3_experience_memory.py",
        "scripts/experiments/gsm8k/run_v3_5_applicability_selector_experiment.sh",
    )
    implementation_hashes = {
        relative: file_sha256(PROJECT_ROOT / relative)
        for relative in implementation_paths
        if (PROJECT_ROOT / relative).is_file()
    }
    missing_implementation_paths = [
        relative
        for relative in implementation_paths
        if not (PROJECT_ROOT / relative).is_file()
    ]
    return {
        "git_revision": revision,
        "worktree_dirty": bool(status),
        "git_status_sha256": text_sha256(status),
        "tracked_diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        "implementation_files_sha256": implementation_hashes,
        "missing_implementation_files": missing_implementation_paths,
        "implementation_set_sha256": canonical_json_sha256(
            implementation_hashes
        ),
    }


def _retrieval_embedding_audit(
    *, retriever: Any, key_bank: Any, dual_key_manifest: Path | None
) -> dict[str, Any]:
    for owner in (retriever, key_bank):
        value = getattr(owner, "embedding_space_audit", None)
        if isinstance(value, Mapping):
            return dict(value)
    if dual_key_manifest is None:
        raise ValueError("V3 retriever did not expose its embedding-space audit")
    return {
        "schema_version": "experience-memory-v3.5-dual-retrieval-space-audit-v1",
        "dual_key_manifest_sha256": file_sha256(dual_key_manifest),
        "loader_authenticated": True,
    }


def _dual_key_artifact_identity(
    *, key_bank: Any, manifest_path: Path | None
) -> dict[str, Any] | None:
    if manifest_path is None:
        return None
    manifest = getattr(key_bank, "manifest", None)
    if not isinstance(manifest, Mapping):
        raise ValueError("V3.5 dual-key loader did not expose its manifest")
    tensor_artifact = manifest.get("tensor_artifact")
    input_artifacts = manifest.get("input_artifacts")
    if not isinstance(tensor_artifact, Mapping) or not isinstance(
        input_artifacts, Mapping
    ):
        raise ValueError("V3.5 dual-key artifact identity is incomplete")
    return {
        "schema_version": manifest.get("schema_version"),
        "manifest_file_sha256": file_sha256(manifest_path),
        "manifest_logical_sha256": manifest.get("manifest_sha256"),
        "tensor_artifact": dict(tensor_artifact),
        "input_artifacts": dict(input_artifacts),
    }


def load_split_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SPLIT_MANIFEST_SCHEMA:
        raise ValueError("Unexpected GSM8K split manifest schema")
    expected = value.get("manifest_sha256")
    actual = canonical_json_sha256({
        key: item
        for key, item in value.items()
        if key not in {"created_at", "manifest_sha256"}
    })
    if expected != actual or value.get("overlap_check", {}).get("passed") is not True:
        raise ValueError("GSM8K split manifest hash or overlap audit failed")
    return value


def processed_solution(answer: str) -> str:
    parts = answer.split("\n####")
    return (parts[0] + "\\boxed{" + parts[-1].strip() + "}").strip()


def score_condition(
    *,
    tokenizer: Any,
    completion_token_ids: Sequence[int],
    ground_truth: str,
    runtime_seconds: float,
) -> dict[str, Any]:
    completion_ids = [int(value) for value in completion_token_ids]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    marker_token_index = (
        first_answer_marker_token_index(
            tokenizer=tokenizer,
            completion_token_ids=completion_ids,
        )
        if _ANSWER_MARKER_RE.search(completion)
        else None
    )
    verifier = diagnose_gsm8k_completion(completion, ground_truth)
    numeric_correct_but_format_invalid = bool(
        verifier.get("diagnostic_answer_correct") is True
        and not verifier["format_valid"]
    )
    return {
        "completion": completion,
        "completion_token_ids": completion_ids,
        "completion_token_ids_sha256": canonical_json_sha256(completion_ids),
        "generated_token_count": len(completion_ids),
        "strict_correct": verifier["reward"] == 1.0,
        "format_correct": bool(verifier["format_valid"]),
        "strict_reward": float(verifier["reward"]),
        "numeric_correct_but_format_invalid": (
            numeric_correct_but_format_invalid
        ),
        "answer_marker_seen": marker_token_index is not None,
        "first_answer_marker_token_index": marker_token_index,
        "diagnostic_answer_source": verifier.get("diagnostic_answer_source"),
        "diagnostic_failure_types": list(verifier.get("failure_types", ())),
        "scorer_version": GSM8K_VERIFIER_VERSION,
        "runtime_seconds": runtime_seconds,
    }


def first_answer_marker_token_index(
    *, tokenizer: Any, completion_token_ids: Sequence[int]
) -> int | None:
    """Return the first 0-based token whose decoded prefix contains a marker."""

    token_ids = [int(value) for value in completion_token_ids]
    for index in range(len(token_ids)):
        decoded_prefix = tokenizer.decode(
            token_ids[: index + 1], skip_special_tokens=True
        )
        if _ANSWER_MARKER_RE.search(decoded_prefix):
            return index
    return None


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        converted = to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    return {}


def _static_shortlist_ids(static_trace: Mapping[str, Any]) -> tuple[str, ...]:
    explicit_ids = static_trace.get("shortlist_memory_ids")
    if isinstance(explicit_ids, Sequence) and not isinstance(
        explicit_ids, (str, bytes)
    ):
        return tuple(str(memory_id) for memory_id in explicit_ids)
    candidates = static_trace.get(
        "post_floor_shortlist",
        static_trace.get("shortlist", static_trace.get("shortlist_hits", ())),
    )
    result: list[str] = []
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                memory_id = candidate.get("memory_id")
            else:
                memory_id = candidate
            if memory_id is not None:
                result.append(str(memory_id))
    return tuple(result)


def validate_calibration_reproduction_trace(
    *,
    runtime_trace: Mapping[str, Any],
    tokenizer: Any,
    question: str,
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
) -> None:
    """Recompute every raw-token hash logged by the V3.5 trace-only profile."""

    static_trace = _mapping(runtime_trace.get("static_selector_trace"))
    static_query = _mapping(static_trace.get("query"))
    static_ids = [
        int(value)
        for value in tokenizer.encode(
            question.strip(), add_special_tokens=False
        )
    ]
    logged_static_ids = static_query.get("static_question_token_ids")
    try:
        static_embedding_norm = float(
            static_query.get("static_question_embedding_norm", float("nan"))
        )
    except (TypeError, ValueError):
        static_embedding_norm = float("nan")
    static_embedding_sha256 = static_query.get(
        "static_question_embedding_sha256"
    )
    if (
        logged_static_ids != static_ids
        or static_query.get("static_question_token_count") != len(static_ids)
        or static_query.get("static_question_token_ids_sha256")
        != canonical_json_sha256(static_ids)
        or static_query.get("static_question_text_sha256")
        != text_sha256(question.strip())
        or static_query.get("layer_number") != 24
        or static_query.get("pooling") != "last_valid_token"
        or static_query.get("normalization") != "l2"
        or static_query.get("side_kv_disabled") is not True
        or static_query.get("chat_wrapper_included") is not False
        or static_query.get("prompt_boilerplate_included") is not False
        or static_query.get("add_special_tokens") is not False
        or not isinstance(static_embedding_sha256, str)
        or not static_embedding_sha256
        or not math.isclose(
            static_embedding_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-5,
        )
    ):
        raise RuntimeError(
            "V3.5 calibration static-question token reproduction failed"
        )

    prompt_ids = [int(value) for value in prompt_token_ids]
    completion_ids = [int(value) for value in completion_token_ids]
    attempts = runtime_trace.get("retrieval_attempts")
    if not isinstance(attempts, list):
        raise RuntimeError("V3.5 calibration retrieval trace is missing")
    for attempt in attempts:
        if not isinstance(attempt, Mapping):
            raise RuntimeError("V3.5 calibration attempt trace is malformed")
        observation_index = int(
            attempt.get(
                "generated_observation_index",
                attempt.get("generated_boundary_index", -1),
            )
        )
        if observation_index < 0 or observation_index >= len(completion_ids):
            raise RuntimeError(
                "V3.5 calibration observation index is outside completion"
            )
        expected_ids = prompt_ids + completion_ids[: observation_index + 1]
        decision = _mapping(attempt.get("retrieval_decision"))
        query = _mapping(decision.get("query"))
        logged_ids = query.get("query_token_ids")
        if (
            logged_ids != expected_ids
            or query.get("query_token_count") != len(expected_ids)
            or query.get("prompt_token_count") != len(prompt_ids)
            or query.get("partial_cot_token_count") != observation_index + 1
            or query.get("encoded_full_prefix_token_count") != len(expected_ids)
            or query.get("query_token_ids_sha256")
            != canonical_json_sha256(expected_ids)
            or query.get("context") != "question_plus_full_partial_cot"
            or query.get("encoder_state")
            != "pure_prefix_reencode_side_kv_disabled"
            or query.get("pooling") != V34_QUERY_POOLING_CURRENT_TOKEN
            or query.get("normalization") != "l2"
            or query.get("side_kv_disabled") is not True
            or query.get("query_embedding_token_index")
            != len(expected_ids) - 1
            or query.get("query_embedding_token_id") != expected_ids[-1]
            or query.get("query_embedding_causal_context_token_count")
            != len(expected_ids)
        ):
            raise RuntimeError(
                "V3.5 calibration dynamic full-prefix token reproduction failed"
            )


def prepare_v35_query_sidecar_embeddings(
    *,
    query_embeddings: Sequence[Any],
    runtime_trace: Mapping[str, Any],
) -> tuple[Any, ...]:
    """Validate and return the exact unit vectors hashed by V3.5 retrieval."""

    import torch

    from memgen.model.retrieval_keys import tensor_sha256

    attempts = runtime_trace.get("retrieval_attempts")
    if not isinstance(attempts, list) or len(query_embeddings) != len(attempts):
        raise RuntimeError("V3.5 query embedding sidecar/attempt counts differ")
    canonical_embeddings: list[Any] = []
    for query_embedding, attempt_trace in zip(query_embeddings, attempts):
        if not isinstance(attempt_trace, Mapping):
            raise RuntimeError("V3.5 query embedding attempt trace is malformed")
        query_audit = _mapping(
            _mapping(attempt_trace.get("retrieval_decision")).get("query")
        )
        canonical = (
            query_embedding.detach().float().cpu().reshape(-1).contiguous()
        )
        if (
            not bool(torch.isfinite(canonical).all().item())
            or query_audit.get("query_embedding_sha256")
            != tensor_sha256(canonical)
            or not math.isclose(
                float(canonical.norm().item()),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
        ):
            raise RuntimeError(
                "V3.5 canonical query sidecar differs from retrieval audit"
            )
        canonical_embeddings.append(canonical)
    return tuple(canonical_embeddings)


def answer_marker_distance_diagnostics(
    *,
    runtime_trace: Mapping[str, Any],
    first_marker_token_index: int | None,
) -> dict[str, Any]:
    """Measure each attempt from its first affected token to a later marker."""

    attempts = runtime_trace.get("retrieval_attempts")
    if not isinstance(attempts, list):
        attempts = []
    per_attempt: list[dict[str, Any]] = []
    affects_contract_respected = True
    for ordinal, attempt in enumerate(attempts, start=1):
        attempt_payload = _mapping(attempt)
        observation_value = attempt_payload.get(
            "generated_observation_index",
            attempt_payload.get("generated_boundary_index"),
        )
        affects_value = attempt_payload.get("affects_generated_token_index")
        try:
            observation_index = int(observation_value)
            affects_index = int(affects_value)
        except (TypeError, ValueError):
            observation_index = None
            affects_index = None
            affects_contract_respected = False
        if (
            observation_index is not None
            and affects_index != observation_index + 1
        ):
            affects_contract_respected = False
        distance = None
        if (
            first_marker_token_index is not None
            and affects_index is not None
            and first_marker_token_index >= affects_index
        ):
            distance = first_marker_token_index - affects_index
        per_attempt.append({
            "attempt_number": int(
                attempt_payload.get("attempt_number", ordinal)
            ),
            "generated_observation_index": observation_index,
            "affects_generated_token_index": affects_index,
            "first_answer_marker_token_index": first_marker_token_index,
            "tokens_until_first_answer_marker": distance,
        })
    distances = [
        int(value["tokens_until_first_answer_marker"])
        for value in per_attempt
        if value["tokens_until_first_answer_marker"] is not None
    ]
    return {
        "first_answer_marker_token_index": first_marker_token_index,
        "answer_marker_attempt_distances": per_attempt,
        "attempt_affects_index_contract_respected": (
            affects_contract_respected
        ),
        "attempts_with_subsequent_answer_marker_count": len(distances),
        "late_attempt_within_32_tokens_count": sum(
            0 <= distance <= 32 for distance in distances
        ),
    }


def online_diagnostics(
    result: Any,
    *,
    system_version: str = "v3",
    first_answer_marker_token_index: int | None = None,
) -> dict[str, Any]:
    outcomes = Counter(attempt.outcome for attempt in result.retrieval_attempts)
    attention = tuple(result.attention_traces)
    result_payload = _mapping(result)
    runtime_summary = _mapping(result_payload.get("summary"))
    attempt_payloads = [_mapping(attempt) for attempt in result.retrieval_attempts]

    def full_prefix_query_contract(attempt: Mapping[str, Any]) -> bool:
        decision = _mapping(attempt.get("retrieval_decision"))
        query = _mapping(decision.get("query"))
        try:
            query_count = int(query.get("query_token_count"))
            prompt_count = int(query.get("prompt_token_count"))
            partial_count = int(query.get("partial_cot_token_count"))
            encoded_count = int(query.get("encoded_full_prefix_token_count"))
            observation_index = int(attempt.get(
                "generated_observation_index",
                attempt.get("generated_boundary_index"),
            ))
        except (TypeError, ValueError):
            return False
        return bool(
            prompt_count > 0
            and partial_count > 0
            and query_count == prompt_count + partial_count
            and encoded_count == query_count
            and partial_count == observation_index + 1
            and query.get("context") == "question_plus_full_partial_cot"
            and query.get("encoder_state")
            == "pure_prefix_reencode_side_kv_disabled"
            and (
                system_version != "v3.5"
                or query.get("side_kv_disabled") is True
            )
        )

    diagnostics = {
        "retrieval_attempt_count": result.retrieval_attempt_count,
        "rearm_count": result.rearm_count,
        "activation_count": outcomes["activated"],
        "replacement_count": result.replacement_count,
        "duplicate_count": result.duplicate_count,
        "abstain_count": outcomes["abstained"],
        "memory_attention_step_count": len(attention),
        "gate_observation_count": len(result.boundary_traces),
        "joint_trigger_qualified_count": sum(
            trace.joint_trigger_qualified for trace in result.boundary_traces
        ),
        "native_gate_observation_count": (
            result.native_gate_observation_count
        ),
        "memory_conditioned_gate_observation_count": (
            result.memory_conditioned_gate_observation_count
        ),
        "attempt_budget_respected": (
            result.retrieval_attempt_count <= 3
        ),
        "query_context_is_full_prefix": all(
            full_prefix_query_contract(attempt)
            for attempt in attempt_payloads
        ),
        "native_cache_excludes_memory_slots": all(
            trace.trace.native_key_length == trace.processed_prefix_token_count
            for trace in attention
        ),
        "memory_attention_mass_finite_and_positive": all(
            math.isfinite(float(trace.trace.memory_attention_mass))
            and float(trace.trace.memory_attention_mass) > 0.0
            for trace in attention
        ),
        "final_gate_state": getattr(result, "final_gate_state", None),
        "final_memory_id": getattr(result, "final_memory_id", None),
        "answer_marker_seen": bool(
            getattr(result, "answer_marker_seen", False)
        ),
    }
    if system_version != "v3.5":
        return diagnostics

    static_trace = _mapping(
        getattr(result, "static_selector_trace", None)
        or result_payload.get("static_selector_trace")
    )
    static_query = _mapping(static_trace.get("query"))
    static_side_kv_disabled = (
        static_query.get("side_kv_disabled") is True
        or static_trace.get("side_kv_disabled") is True
    )
    shortlist_ids = _static_shortlist_ids(static_trace)
    decision_payloads = [
        _mapping(attempt.get("retrieval_decision"))
        for attempt in attempt_payloads
    ]
    selected_attempts = [
        attempt
        for attempt in attempt_payloads
        if attempt.get("selected_memory_id") is not None
    ]
    selected_belongs = all(
        str(attempt["selected_memory_id"]) in shortlist_ids
        for attempt in selected_attempts
    )
    kv_alignment_values = [
        _mapping(decision.get("query")).get(
            "selected_memory_kv_metadata_aligned"
        )
        for attempt, decision in zip(attempt_payloads, decision_payloads)
        if attempt.get("selected_memory_id") is not None
    ]
    dynamic_side_kv_disabled = all(
        _mapping(decision.get("query")).get("side_kv_disabled") is True
        for decision in decision_payloads
    )
    dynamic_restricted_to_static_shortlist = all(
        _mapping(decision.get("query")).get(
            "dynamic_search_restricted_to_static_shortlist"
        )
        is True
        and tuple(
            str(memory_id)
            for memory_id in _mapping(decision.get("query")).get(
                "static_shortlist_ids", ()
            )
        )
        == shortlist_ids
        for decision in decision_payloads
    )
    terminal_attempts = [
        attempt for attempt in attempt_payloads if attempt.get("terminal_abstain") is True
    ]
    cleared_terminal_attempts = [
        attempt
        for attempt in terminal_attempts
        if attempt.get("memory_cleared_on_abstain") is True
    ]
    diagnostics.update({
        "static_selector_trace_present": bool(static_trace),
        "static_selector_unavailable": bool(
            static_trace.get("static_selector_unavailable", False)
        ),
        "static_selector_unavailable_reason": static_trace.get(
            "unavailable_reason"
        ),
        "static_shortlist_size": len(shortlist_ids),
        "static_shortlist_ids_sha256": canonical_json_sha256(shortlist_ids),
        "static_shortlist_fixed_for_generation": (
            static_trace.get("shortlist_fixed_for_generation") is True
        ),
        "static_query_side_kv_disabled": (
            static_side_kv_disabled
        ),
        "dynamic_query_side_kv_disabled": dynamic_side_kv_disabled,
        "both_query_encodings_side_kv_disabled": (
            static_side_kv_disabled and dynamic_side_kv_disabled
        ),
        "dynamic_search_restricted_to_static_shortlist": (
            dynamic_restricted_to_static_shortlist
        ),
        "selected_memory_belongs_to_static_shortlist": selected_belongs,
        "selected_outside_static_shortlist_count": sum(
            str(attempt["selected_memory_id"]) not in shortlist_ids
            for attempt in selected_attempts
        ),
        "selected_memory_kv_metadata_aligned": all(
            value is True for value in kv_alignment_values
        ),
        "selected_memory_kv_alignment_unlogged_count": sum(
            value is not True for value in kv_alignment_values
        ),
        "terminal_abstain_count": int(
            runtime_summary.get("terminal_abstain_count", len(terminal_attempts))
        ),
        "clear_on_terminal_abstain_count": int(
            runtime_summary.get(
                "clear_on_terminal_abstain_count",
                sum(
                    attempt.get("memory_cleared_on_abstain") is True
                    for attempt in terminal_attempts
                ),
            )
        ),
        "no_rearm_after_terminal_abstain": bool(
            runtime_summary.get("no_rearm_after_terminal_abstain", False)
        ),
        "two_low_rearm_respected": bool(
            runtime_summary.get("two_low_rearm_respected", False)
        ),
        "second_low_rearms_without_trigger": bool(
            runtime_summary.get("second_low_rearms_without_trigger", False)
        ),
        "stale_memory_attention_after_terminal_clear_count": int(
            runtime_summary.get(
                "stale_memory_attention_after_terminal_clear_count", -1
            )
        ),
        "terminal_clear_attention_safe": bool(
            runtime_summary.get("terminal_clear_attention_safe", False)
        ),
        "terminal_abstain_actual_path_native": all(
            attempt.get("actual_path_after_abstain") == "native"
            and attempt.get("actual_path_memory_id_after") is None
            and attempt.get("active_memory_id_after") is None
            for attempt in terminal_attempts
        ),
        "terminal_clear_native_reforward_audited": all(
            attempt.get("cleared_memory_id") is not None
            and attempt.get("deactivation_forward_seconds") is not None
            and attempt.get("deactivation_first_step_logits_kl") is not None
            and attempt.get("deactivation_first_step_top1_changed") is not None
            and attempt.get("deactivation_baseline_first_token_id") is not None
            and attempt.get("deactivation_native_first_token_id") is not None
            and attempt.get("clear_affects_generated_token_index") is not None
            and attempt.get("clear_affects_generated_token_index")
            == attempt.get("affects_generated_token_index")
            for attempt in cleared_terminal_attempts
        ),
        "static_selector_unavailable_zero_attempt": (
            not static_trace.get("static_selector_unavailable", False)
            or result.retrieval_attempt_count == 0
        ),
    })
    diagnostics.update(answer_marker_distance_diagnostics(
        runtime_trace=result_payload,
        first_marker_token_index=first_answer_marker_token_index,
    ))
    return diagnostics


def summary_row(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop large runtime traces after their authenticated JSONL write."""

    result: dict[str, Any] = {"conditions": {}}
    for condition in ("vanilla", "v3"):
        source = value["conditions"][condition]
        compact = {
            key: source[key]
            for key in (
                "strict_correct",
                "format_correct",
                "generated_token_count",
            )
        }
        for diagnostic_key in (
            "numeric_correct_but_format_invalid",
            "answer_marker_seen",
            "first_answer_marker_token_index",
        ):
            if diagnostic_key in source:
                compact[diagnostic_key] = source[diagnostic_key]
        if condition == "v3":
            compact["online_diagnostics"] = dict(
                source["online_diagnostics"]
            )
        result["conditions"][condition] = compact
    return result


def load_existing_rows(
    *,
    path: Path,
    profile_sha256: str,
    selected_ids: set[str],
    row_schema: str = V3_EVAL_ROW_SCHEMA,
    sidecar_root: Path | None = None,
    require_v35_query_sidecars: bool = False,
) -> tuple[list[dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows: list[dict[str, Any]] = []
    completed: set[str] = set()
    for value in iter_jsonl(path):
        if (
            value.get("schema_version") != row_schema
            or value.get("profile_sha256") != profile_sha256
        ):
            raise ValueError("Existing V3 result row uses a different run profile")
        expected_row_hash = value.get("row_sha256")
        actual_row_hash = canonical_json_sha256({
            key: item
            for key, item in value.items()
            if key not in {"created_at", "row_sha256"}
        })
        if expected_row_hash != actual_row_hash:
            raise ValueError("Existing V3 result row hash mismatch")
        sample_id = str(value.get("sample_id", ""))
        if sample_id not in selected_ids or sample_id in completed:
            raise ValueError("Existing V3 results contain unknown or duplicate samples")
        if require_v35_query_sidecars:
            if sidecar_root is None:
                raise ValueError(
                    "V3.5 calibration resume requires a sidecar root"
                )
            v3_condition = _mapping(
                _mapping(value.get("conditions")).get("v3")
            )
            runtime_trace = _mapping(v3_condition.get("runtime_trace"))
            attempts = runtime_trace.get("retrieval_attempts")
            if not isinstance(attempts, list):
                raise ValueError(
                    "Existing V3.5 calibration row lacks a retrieval trace"
                )
            descriptor = v3_condition.get("query_embedding_sidecar")
            if attempts:
                if not isinstance(descriptor, Mapping):
                    raise ValueError(
                        "Existing V3.5 calibration row lacks its query sidecar"
                    )
                relative = Path(str(descriptor.get("path", "")))
                resolved_root = sidecar_root.resolve()
                resolved_sidecar = (sidecar_root / relative).resolve()
                if (
                    not relative.parts
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or resolved_root not in resolved_sidecar.parents
                    or not resolved_sidecar.is_file()
                    or descriptor.get("sha256")
                    != file_sha256(resolved_sidecar)
                    or isinstance(descriptor.get("attempt_count"), bool)
                    or not isinstance(descriptor.get("attempt_count"), int)
                    or int(descriptor["attempt_count"]) != len(attempts)
                    or descriptor.get("representation")
                    != V35_QUERY_SIDECAR_REPRESENTATION
                ):
                    raise ValueError(
                        "Existing V3.5 calibration query sidecar is invalid"
                    )
            elif descriptor is not None:
                raise ValueError(
                    "Zero-attempt V3.5 calibration row has an unexpected sidecar"
                )
        completed.add(sample_id)
        rows.append(summary_row(value))
    return rows, completed


def progress_report(
    *,
    status: str,
    profile_sha256: str,
    selected_count: int,
    rows: Sequence[Mapping[str, Any]],
    error: Mapping[str, Any] | None = None,
    report_schema: str = V3_EVAL_REPORT_SCHEMA,
) -> dict[str, Any]:
    value = {
        "schema_version": report_schema,
        "updated_at": utc_now(),
        "status": status,
        "profile_sha256": profile_sha256,
        "selected_sample_count": selected_count,
        "completed_sample_count": len(rows),
        "remaining_sample_count": selected_count - len(rows),
        "error": dict(error) if error is not None else None,
        "summary": summarize_v3_rows(rows) if rows else None,
    }
    value["report_sha256"] = canonical_json_sha256(value)
    return value


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


def main() -> None:
    args = parse_args()
    _resolve_and_validate_versioned_args(args)
    profile_schema, row_schema, report_schema = evaluation_schemas(
        args.system_version
    )
    if (
        args.offset < 0
        or args.limit < 0
        or args.parity_samples < 0
        or args.max_new_tokens <= 0
    ):
        raise ValueError("V3 evaluation received invalid limits")
    if args.max_new_tokens != GSM8K_PROMPT_CONTRACT.max_new_tokens:
        raise ValueError("V3 evaluation must use the canonical GSM8K token budget")

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from memgen.model.e1_runtime import GreedyE1Runtime, compare_token_sequences
    from memgen.model.retrieval_keys import (
        EmbeddingMemoryRetriever,
        FullPrefixQueryEncoder,
        RetrievalKeyBankLoader,
    )
    from memgen.model.side_kv import SideKVAttentionController, SideKVBankLoader
    from memgen.model.v3_runtime import (
        EntropyHysteresisGate,
        OnlineExperienceMemorySystemV3,
    )

    split_manifest = load_split_manifest(args.split_manifest)
    selected = [
        item
        for item in split_manifest["samples"]
        if item.get("logical_split") == args.logical_split
    ][args.offset:]
    if args.limit:
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("Selected V3 evaluation slice is empty")
    dataset_splits = {str(item.get("dataset_split", "")) for item in selected}
    if len(dataset_splits) != 1:
        raise ValueError("V3 logical split maps to multiple dataset splits")
    dataset_split = next(iter(dataset_splits))
    if args.logical_split == "final-test" and dataset_split != "test":
        raise ValueError("V3 final-test must be the official GSM8K test split")

    selector_calibration = None
    applicability_calibration = None
    if args.system_version == "v3.5":
        (
            profile,
            applicability_calibration,
            selector_calibration,
        ) = _load_v35_profile_and_artifacts(args)
    elif args.selector_calibration is not None:
        selector_calibration = load_margin_selector_calibration(
            args.selector_calibration
        )
        expected_key_sha256 = selector_calibration.get("source", {}).get(
            "retrieval_key_manifest_sha256"
        )
        if expected_key_sha256 != file_sha256(args.retrieval_key_manifest):
            raise ValueError(
                "V3 selector calibration uses a different retrieval key bank"
            )
        calibration_risk_sha256 = selector_calibration.get("source", {}).get(
            "risk_artifact_sha256"
        )
        if (
            args.system_version == "v3.4"
            and calibration_risk_sha256 != file_sha256(args.risk_artifact)
        ):
            raise ValueError(
                "V3.4 selector calibration uses a different token-risk artifact"
            )
        calibration_system_version = selector_calibration.get(
            "source", {}
        ).get("system_version", "v3")
        if calibration_system_version != args.system_version:
            raise ValueError(
                "Selector calibration uses a different system version"
            )
        calibration_transform = selector_calibration.get("source", {}).get(
            "retrieval_embedding_transform",
            V3_RETRIEVAL_EMBEDDING_TRANSFORM_NONE,
        )
        if calibration_transform != args.retrieval_embedding_transform:
            raise ValueError(
                "Selector calibration uses a different retrieval embedding transform"
            )
        if selector_calibration_query_pooling(
            selector_calibration
        ) != args.query_pooling:
            raise ValueError(
                "Selector calibration uses a different query pooling policy"
            )
        profile_kwargs = {
            "retrieval_embedding_transform": (
                args.retrieval_embedding_transform
            ),
            "retrieval_abstention_policy": "top1_top2_margin",
            "retrieval_min_top1_top2_margin": float(
                selector_calibration["calibration"][
                    "minimum_top1_top2_margin"
                ]
            ),
        }
        profile = (
            ExperienceMemoryV3Profile.continuous_token_joint(**profile_kwargs)
            if args.system_version == "v3.4"
            else ExperienceMemoryV3Profile(
                query_pooling=args.query_pooling, **profile_kwargs
            )
        )
    else:
        profile = (
            ExperienceMemoryV3Profile.continuous_token_joint(
                retrieval_embedding_transform=(
                    args.retrieval_embedding_transform
                )
            )
            if args.system_version == "v3.4"
            else ExperienceMemoryV3Profile(
                query_pooling=args.query_pooling,
                retrieval_embedding_transform=(
                    args.retrieval_embedding_transform
                ),
            )
        )
    e0_report = load_formal_e0_report(args.e0_final_report)
    authenticate_e0_inputs(
        e0_report=e0_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
    )
    load_v3_offline_report(
        args.v3_offline_report,
        memory_records_path=args.memory_records,
        side_kv_manifest_path=args.side_kv_manifest,
        retrieval_key_manifest_path=args.retrieval_key_manifest,
        e0_final_report_path=args.e0_final_report,
    )
    records = tuple(
        MemoryRecord.from_dict(value) for value in iter_jsonl(args.memory_records)
    )
    side_manifest = json.loads(args.side_kv_manifest.read_text(encoding="utf-8"))
    key_manifest = json.loads(
        args.retrieval_key_manifest.read_text(encoding="utf-8")
    )
    alignment = validate_cross_bank_metadata(
        records=records,
        side_manifest=side_manifest,
        key_manifest=key_manifest,
    )
    reasoner = side_manifest["reasoner"]

    risk_artifact = torch.load(
        args.risk_artifact, map_location="cpu", weights_only=False
    )
    continuous_version = args.system_version in {"v3.4", "v3.5"}
    expected_risk_schema = (
        TOKEN_ENTROPY_RISK_ARTIFACT_SCHEMA
        if continuous_version
        else ENTROPY_RISK_ARTIFACT_SCHEMA
    )
    if risk_artifact.get("schema_version") != expected_risk_schema:
        raise ValueError(
            f"{args.system_version} evaluation requires its canonical risk artifact"
        )
    expected_prompt_contract = GSM8K_PROMPT_CONTRACT.metadata(
        chat_template=CONVERSATION_TEMPLATE
    )
    if risk_artifact.get("prompt_contract") != expected_prompt_contract:
        raise ValueError("V3 risk artifact uses a different prompt contract")
    heldout = risk_artifact.get("risk_gate", {}).get("heldout_diagnostic", {})
    if (
        float(heldout.get("heldout_roc_auc", 0.0))
        < float(heldout.get("minimum_heldout_roc_auc", 1.0))
        or continuous_version
        and risk_artifact.get("qualification", {}).get("passed") is not True
    ):
        raise ValueError("V3 risk artifact failed its held-out diagnostic")
    for field_name in ("model_name", "model_revision", "tokenizer_revision"):
        if risk_artifact.get("reasoner", {}).get(field_name) != reasoner.get(
            field_name
        ):
            raise ValueError("V3 risk and memory reasoner provenance differs")
    gate = (
        EntropyHysteresisGate.from_token_artifact(risk_artifact)
        if continuous_version
        else EntropyHysteresisGate.from_artifact(risk_artifact)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        reasoner["model_name"], revision=reasoner["tokenizer_revision"]
    )
    tokenizer.chat_template = CONVERSATION_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        reasoner["model_name"],
        revision=reasoner["model_revision"],
        dtype=dtype,
        attn_implementation="sdpa",
    ).to(args.device)
    model.eval()
    resolved_model = str(
        getattr(model.config, "_commit_hash", None) or reasoner["model_revision"]
    )
    resolved_tokenizer = str(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        or reasoner["tokenizer_revision"]
    )
    if (
        resolved_model != reasoner["model_revision"]
        or resolved_tokenizer != reasoner["tokenizer_revision"]
    ):
        raise ValueError("Resolved V3 evaluation reasoner revision drifted")

    side_loader = SideKVBankLoader(
        manifest_path=args.side_kv_manifest,
        expected_reasoner_name=reasoner["model_name"],
        expected_reasoner_revision=reasoner["model_revision"],
        expected_tokenizer_revision=reasoner["tokenizer_revision"],
    )
    side_entries = {
        str(entry["memory_id"]): entry for entry in side_manifest["records"]
    }
    kv_valid_slot_counts = {
        memory_id: int(entry["kv_valid_slot_count"])
        for memory_id, entry in side_entries.items()
    }
    if args.system_version == "v3.5":
        from memgen.model.v3_5_retrieval import (
            ApplicabilityAwareMemoryRetriever,
            DualRetrievalKeyBankLoader,
            QuestionOnlyEncoder,
        )

        assert args.dual_key_manifest is not None
        RetrievalKeyBankLoader(
            manifest_path=args.retrieval_key_manifest,
            expected_reasoner_name=reasoner["model_name"],
            expected_reasoner_revision=reasoner["model_revision"],
            expected_tokenizer_revision=reasoner["tokenizer_revision"],
        )
        legacy_tensor_sha256 = str(
            key_manifest.get("tensor_artifact", {}).get("sha256", "")
        )
        if not legacy_tensor_sha256:
            raise ValueError("V3 applicability source tensor hash is missing")
        key_bank = DualRetrievalKeyBankLoader(
            manifest_path=args.dual_key_manifest,
            expected_reasoner_name=reasoner["model_name"],
            expected_reasoner_revision=reasoner["model_revision"],
            expected_tokenizer_revision=reasoner["tokenizer_revision"],
            expected_input_hashes={
                "memory_records_sha256": file_sha256(args.memory_records),
                "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
                "e0_final_report_sha256": file_sha256(args.e0_final_report),
                "v3_retrieval_key_manifest_sha256": file_sha256(
                    args.retrieval_key_manifest
                ),
                "v3_retrieval_key_tensor_sha256": legacy_tensor_sha256,
                "v3_offline_report_sha256": file_sha256(
                    args.v3_offline_report
                ),
                "split_manifest_sha256": file_sha256(args.split_manifest),
            },
        )
        retriever = ApplicabilityAwareMemoryRetriever(
            key_bank=key_bank,
            records=records,
            kv_valid_slot_counts=kv_valid_slot_counts,
            question_encoder=QuestionOnlyEncoder(
                model=model,
                tokenizer=tokenizer,
                device=args.device,
                layer_number=profile.layer_number,
            ),
            shortlist_k=int(profile.applicability_shortlist_k),
            applicability_score_floor=float(
                profile.applicability_score_floor
            ),
            dynamic_min_top1_top2_margin=(
                profile.retrieval_min_top1_top2_margin
            ),
            profile=profile,
        )
    else:
        key_bank = RetrievalKeyBankLoader(
            manifest_path=args.retrieval_key_manifest,
            expected_reasoner_name=reasoner["model_name"],
            expected_reasoner_revision=reasoner["model_revision"],
            expected_tokenizer_revision=reasoner["tokenizer_revision"],
        )
        retriever = EmbeddingMemoryRetriever(
            key_bank=key_bank,
            records=records,
            kv_valid_slot_counts=kv_valid_slot_counts,
            profile=profile,
        )
    controller = SideKVAttentionController(
        model=model,
        layer_number=profile.layer_number,
        audit_canonical_rope=False,
        memory_score_normalization=profile.memory_score_normalization,
        memory_score_bias=profile.memory_score_bias,
    )
    baseline_runtime = GreedyE1Runtime(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    v3_runtime = OnlineExperienceMemorySystemV3(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        gate=gate,
        query_encoder=FullPrefixQueryEncoder(
            model=model,
            device=args.device,
            layer_number=profile.layer_number,
            query_pooling=profile.query_pooling,
        ),
        retriever=retriever,
        loader=side_loader,
        controller=controller,
        profile=profile,
    )

    selected_ids = [str(item["sample_id"]) for item in selected]
    retrieval_space_audit = _retrieval_embedding_audit(
        retriever=retriever,
        key_bank=key_bank,
        dual_key_manifest=args.dual_key_manifest,
    )
    dual_key_artifact = _dual_key_artifact_identity(
        key_bank=key_bank,
        manifest_path=args.dual_key_manifest,
    )
    if args.logical_split == "final-test":
        interpretation = "reused_official_test_descriptive_evaluation"
    elif args.system_version == "v3.5" and args.calibration_trace_only:
        interpretation = "answer_blind_v3_5_first_attempt_margin_trace_only"
    elif args.system_version == "v3.5":
        interpretation = (
            "exploratory_matched_dev_v3_5_compound_selector_lifecycle"
        )
    elif args.system_version == "v3.4":
        interpretation = "answer_blind_continuous_token_joint_gate_validation"
    elif profile.query_pooling == "last_token_before_trigger_boundary":
        interpretation = "answer_blind_pre_boundary_query_validation"
    elif (
        profile.retrieval_embedding_transform
        == V3_RETRIEVAL_EMBEDDING_TRANSFORM_CENTERED
    ):
        interpretation = "answer_blind_centered_retrieval_validation"
    elif selector_calibration is not None:
        interpretation = "answer_blind_margin_selector_validation"
    else:
        interpretation = "system_validation"
    run_profile = {
        "schema_version": profile_schema,
        "created_at": utc_now(),
        "repository": repository_state(),
        "evaluation_interpretation": interpretation,
        "independent_final_confirmation": False,
        "logical_split": args.logical_split,
        "system_version": args.system_version,
        "calibration_trace_only": bool(args.calibration_trace_only),
        "task_results_used_for_selector_decision": False,
        "selector_decision_data_contract": {
            "task_accuracy_used": False,
            "answer_or_reward_used": False,
            "task_results_recorded_for_diagnostics_only": True,
            "first_attempt_dynamic_margins_only": bool(
                args.calibration_trace_only
            ),
        },
        "dataset_split": dataset_split,
        "dataset_revision": split_manifest["dataset"]["revision"],
        "selected_sample_count": len(selected),
        "selected_sample_ids_sha256": canonical_json_sha256(selected_ids),
        "slice": {"offset": args.offset, "limit": args.limit},
        "reasoner": reasoner | {"runtime_dtype": args.dtype},
        "prompt_contract": expected_prompt_contract,
        "system_profile": profile.to_dict(),
        "system_profile_sha256": canonical_json_sha256(profile.to_dict()),
        "retrieval_embedding_space": retrieval_space_audit,
        "dual_key_artifact": dual_key_artifact,
        "applicability_calibration": (
            {
                "schema_version": applicability_calibration.get(
                    "schema_version"
                ),
                "status": applicability_calibration.get("status"),
                "artifact_sha256": applicability_calibration.get(
                    "artifact_sha256"
                ),
                "source": applicability_calibration.get("source"),
                "calibration": applicability_calibration.get("calibration"),
                "qualification": applicability_calibration.get(
                    "qualification"
                ),
                "requirements": applicability_calibration.get(
                    "requirements"
                ),
                "partition": applicability_calibration.get("partition"),
                "source_pair_audit": applicability_calibration.get(
                    "source_pair_audit"
                ),
                "task_accuracy_used": applicability_calibration.get(
                    "task_accuracy_used"
                ),
                "answer_or_reward_used": applicability_calibration.get(
                    "answer_or_reward_used"
                ),
            }
            if applicability_calibration is not None
            else None
        ),
        "selector_calibration": (
            {
                "schema_version": selector_calibration.get("schema_version"),
                "status": selector_calibration.get("status"),
                "artifact_sha256": selector_calibration.get(
                    "artifact_sha256"
                ),
                "policy": selector_calibration.get("policy"),
                "source": selector_calibration.get("source"),
                "calibration": selector_calibration.get("calibration"),
                "qualification": selector_calibration.get("qualification"),
                "requirements": selector_calibration.get("requirements"),
                "task_accuracy_used": selector_calibration.get(
                    "task_accuracy_used"
                ),
                "answer_or_reward_used": selector_calibration.get(
                    "answer_or_reward_used"
                ),
                "first_attempt_selection_concentration": (
                    selector_calibration.get(
                        "first_attempt_selection_concentration"
                    )
                ),
            }
            if selector_calibration is not None
            else None
        ),
        "hysteresis_gate": gate.config.to_dict(),
        "hysteresis_threshold_provenance": {
            key: risk_artifact.get("construction", {}).get(key)
            for key in (
                "high_entropy_quantile",
                "high_entropy_threshold",
                "low_entropy_quantile",
                "low_entropy_threshold",
                "sink_token_count",
            )
        },
        "risk_diagnostic_qualification": {
            "heldout_roc_auc": heldout.get("heldout_roc_auc"),
            "minimum_heldout_roc_auc": heldout.get("minimum_heldout_roc_auc"),
            "heldout_balanced_accuracy_at_train_threshold": heldout.get(
                "heldout_balanced_accuracy_at_train_threshold"
            ),
            "train_threshold_calibration": heldout.get(
                "train_threshold_calibration"
            ),
            "online_control_role": profile.risk_role,
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "vanilla": baseline_runtime.native_generation_config_dict,
            "v3": v3_runtime.decoding.config.to_dict() | {
                "use_cache": True,
                "batch_size": 1,
                "implementation": "explicit_live_native_kv_cache",
            },
            "parity_sample_count": min(args.parity_samples, len(selected)),
        },
        "logging": {
            "append_flush_fsync_per_sample": True,
            "resume_requires_profile_hash": True,
            "query_embeddings_sidecar": args.save_query_embeddings,
            "query_embeddings_sidecar_required_for_calibration": bool(
                args.calibration_trace_only
            ),
            "query_embedding_sidecar_representation": (
                V35_QUERY_SIDECAR_REPRESENTATION
                if args.system_version == "v3.5"
                else LEGACY_QUERY_SIDECAR_REPRESENTATION
            ),
            "full_logits_saved": False,
            "full_hidden_states_saved": False,
            "raw_query_token_ids_saved": bool(
                args.calibration_trace_only
            ),
            "calibration_query_hash_reproduction_required": bool(
                args.calibration_trace_only
            ),
        },
        "metric_contract": {
            "strict_accuracy": "official_gsm8k_first_boxed_reward",
            "format_accuracy": "first_boxed_parseable",
            "generated_token_count": "through_first_eos_inclusive_else_full_budget",
            "diagnostic_answer_accuracy_aggregated": False,
            "numeric_correct_but_format_invalid": (
                "diagnostic_count_only_not_a_formal_accuracy"
            ),
            "answer_marker_presence": "paired_diagnostic_only",
            "first_answer_marker_token_index": (
                "zero_based_first_generated_token_whose_decoded_prefix_"
                "matches_the_answer_marker_regex_else_null"
            ),
            "tokens_until_first_answer_marker": (
                "first_answer_marker_token_index_minus_attempt_"
                "affects_generated_token_index_when_marker_is_at_or_after_"
                "the_first_affected_token_else_null"
            ),
            "late_attempt_within_32_tokens_count": (
                "attempt_distances_inclusive_between_zero_and_32"
            ),
        },
        "alignment": alignment,
        "inputs": {
            "split_manifest_sha256": file_sha256(args.split_manifest),
            "memory_records_sha256": file_sha256(args.memory_records),
            "retrieval_key_manifest_sha256": file_sha256(
                args.retrieval_key_manifest
            ),
            "side_kv_manifest_sha256": file_sha256(args.side_kv_manifest),
            "v3_offline_report_sha256": file_sha256(args.v3_offline_report),
            "e0_final_report_sha256": file_sha256(args.e0_final_report),
            "risk_artifact_sha256": file_sha256(args.risk_artifact),
            "selector_calibration_sha256": (
                file_sha256(args.selector_calibration)
                if args.selector_calibration is not None
                else None
            ),
            "dual_key_manifest_sha256": (
                file_sha256(args.dual_key_manifest)
                if args.dual_key_manifest is not None
                else None
            ),
            "applicability_calibration_sha256": (
                file_sha256(args.applicability_calibration)
                if args.applicability_calibration is not None
                else None
            ),
        },
    }
    run_profile["profile_sha256"] = evaluation_profile_sha256(run_profile)
    profile_sha256 = run_profile["profile_sha256"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = args.output_dir / "run_profile.json"
    results_path = args.output_dir / "results.jsonl"
    report_path = args.output_dir / "run_report.json"
    if profile_path.exists():
        existing_profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if (
            existing_profile.get("schema_version") != profile_schema
            or existing_profile.get("system_version") != args.system_version
            or existing_profile.get("profile_sha256")
            != evaluation_profile_sha256(existing_profile)
            or existing_profile.get("profile_sha256") != profile_sha256
        ):
            raise ValueError("Cannot resume V3 evaluation with a different profile")
    else:
        if results_path.exists() and results_path.stat().st_size:
            raise ValueError("V3 results exist without an authenticated run profile")
        write_json_atomic(profile_path, run_profile)

    rows, completed_ids = load_existing_rows(
        path=results_path,
        profile_sha256=profile_sha256,
        selected_ids=set(selected_ids),
        row_schema=row_schema,
        sidecar_root=args.output_dir,
        require_v35_query_sidecars=bool(args.calibration_trace_only),
    )
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split=dataset_split,
        revision=split_manifest["dataset"]["revision"],
    )
    write_json_atomic(report_path, progress_report(
        status="running",
        profile_sha256=profile_sha256,
        selected_count=len(selected),
        rows=rows,
        report_schema=report_schema,
    ))

    try:
        with results_path.open("a", encoding="utf-8") as output_handle:
            for position, entry in enumerate(selected):
                sample_id = str(entry["sample_id"])
                if sample_id in completed_ids:
                    continue
                source = dataset[int(entry["source_index"])]
                question = str(source["question"]).strip()
                answer = str(source["answer"]).strip()
                if (
                    text_sha256(question) != entry.get("question_sha256")
                    or text_sha256(answer) != entry.get("answer_sha256")
                ):
                    raise ValueError(f"Dataset hash drift for {sample_id}")
                prompt_ids = GSM8K_PROMPT_CONTRACT.token_ids(tokenizer, question)
                ground_truth = processed_solution(answer)

                sample_started = time.perf_counter()
                vanilla_started = time.perf_counter()
                vanilla_ids = baseline_runtime.generate_vanilla(prompt_ids)
                vanilla_seconds = time.perf_counter() - vanilla_started
                parity = None
                if position < args.parity_samples:
                    cache_ids = baseline_runtime.generate_cache_greedy(prompt_ids)
                    parity = compare_token_sequences(
                        vanilla_ids, cache_ids
                    ).to_dict()
                    if not parity["exact_match"]:
                        raise RuntimeError(
                            f"Vanilla/cache parity failed for {sample_id}"
                        )

                v3_started = time.perf_counter()
                v3_result = v3_runtime.generate(
                    prompt_token_ids=prompt_ids,
                    question=(question if args.system_version == "v3.5" else None),
                )
                v3_seconds = time.perf_counter() - v3_started
                vanilla_condition = score_condition(
                    tokenizer=tokenizer,
                    completion_token_ids=vanilla_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=vanilla_seconds,
                )
                v3_condition = score_condition(
                    tokenizer=tokenizer,
                    completion_token_ids=v3_result.completion_token_ids,
                    ground_truth=ground_truth,
                    runtime_seconds=v3_seconds,
                )
                runtime_trace = v3_result.to_dict()
                if args.calibration_trace_only:
                    validate_calibration_reproduction_trace(
                        runtime_trace=runtime_trace,
                        tokenizer=tokenizer,
                        question=question,
                        prompt_token_ids=prompt_ids,
                        completion_token_ids=v3_result.completion_token_ids,
                    )
                diagnostics = online_diagnostics(
                    v3_result,
                    system_version=args.system_version,
                    first_answer_marker_token_index=v3_condition[
                        "first_answer_marker_token_index"
                    ],
                )
                vanilla_v3_exact = list(vanilla_ids) == list(
                    v3_result.completion_token_ids
                )
                zero_attempt_path = diagnostics["retrieval_attempt_count"] == 0
                static_unavailable = bool(
                    diagnostics.get("static_selector_unavailable", False)
                )
                diagnostics.update({
                    "zero_attempt_path": zero_attempt_path,
                    "zero_attempt_exact_parity": (
                        not zero_attempt_path or vanilla_v3_exact
                    ),
                    "static_selector_unavailable_exact_parity": (
                        not static_unavailable or vanilla_v3_exact
                    ),
                    "vanilla_v3_exact_completion_match": vanilla_v3_exact,
                    "vanilla_answer_marker_seen": vanilla_condition[
                        "answer_marker_seen"
                    ],
                    "v3_answer_marker_seen": v3_condition[
                        "answer_marker_seen"
                    ],
                    "vanilla_marker_seen_v3_marker_absent": bool(
                        vanilla_condition["answer_marker_seen"]
                        and not v3_condition["answer_marker_seen"]
                    ),
                })
                integrity_keys = [
                    "attempt_budget_respected",
                    "query_context_is_full_prefix",
                    "native_cache_excludes_memory_slots",
                    "memory_attention_mass_finite_and_positive",
                ]
                if args.system_version == "v3.5":
                    integrity_keys.extend([
                        "static_selector_trace_present",
                        "static_shortlist_fixed_for_generation",
                        "both_query_encodings_side_kv_disabled",
                        "dynamic_search_restricted_to_static_shortlist",
                        "selected_memory_belongs_to_static_shortlist",
                        "selected_memory_kv_metadata_aligned",
                        "no_rearm_after_terminal_abstain",
                        "two_low_rearm_respected",
                        "second_low_rearms_without_trigger",
                        "terminal_clear_attention_safe",
                        "terminal_abstain_actual_path_native",
                        "terminal_clear_native_reforward_audited",
                        "static_selector_unavailable_zero_attempt",
                        "zero_attempt_exact_parity",
                        "static_selector_unavailable_exact_parity",
                        "attempt_affects_index_contract_respected",
                    ])
                if not all(diagnostics[key] for key in integrity_keys):
                    raise RuntimeError(f"V3 integrity check failed for {sample_id}")

                query_sidecar = None
                if (
                    args.calibration_trace_only
                    and runtime_trace.get("retrieval_attempts")
                    and not v3_result.query_embeddings
                ):
                    raise RuntimeError(
                        "V3.5 calibration attempts require query embeddings"
                    )
                if args.save_query_embeddings and v3_result.query_embeddings:
                    from safetensors.torch import save_file

                    if len(v3_result.query_embeddings) != len(
                        runtime_trace.get("retrieval_attempts", ())
                    ):
                        raise RuntimeError(
                            "V3 query embedding sidecar/attempt counts differ"
                        )
                    sidecar_embeddings = tuple(v3_result.query_embeddings)
                    sidecar_representation = (
                        LEGACY_QUERY_SIDECAR_REPRESENTATION
                    )
                    if args.system_version == "v3.5":
                        sidecar_embeddings = (
                            prepare_v35_query_sidecar_embeddings(
                                query_embeddings=v3_result.query_embeddings,
                                runtime_trace=runtime_trace,
                            )
                        )
                        sidecar_representation = (
                            V35_QUERY_SIDECAR_REPRESENTATION
                        )

                    safe_sample_id = "".join(
                        character if character.isalnum() or character in "-_" else "_"
                        for character in sample_id
                    )
                    sidecar_path = (
                        args.output_dir
                        / "query_embeddings"
                        / f"{safe_sample_id}.safetensors"
                    )
                    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(
                        {
                            f"attempt_{index:02d}": embedding.contiguous()
                            for index, embedding in enumerate(
                                sidecar_embeddings, start=1
                            )
                        },
                        str(sidecar_path),
                        metadata={
                            "schema_version": (
                                "experience-memory-v3.5-query-embeddings-v1"
                                if args.system_version == "v3.5"
                                else "experience-memory-v3-query-embeddings-v1"
                            ),
                            "sample_id": sample_id,
                            "representation": sidecar_representation,
                        },
                    )
                    query_sidecar = {
                        "path": str(sidecar_path.relative_to(args.output_dir)),
                        "sha256": file_sha256(sidecar_path),
                        "attempt_count": len(sidecar_embeddings),
                        "representation": sidecar_representation,
                    }

                row = {
                    "schema_version": row_schema,
                    "created_at": utc_now(),
                    "profile_sha256": profile_sha256,
                    "sample_id": sample_id,
                    "logical_split": args.logical_split,
                    "dataset_split": dataset_split,
                    "source_index": int(entry["source_index"]),
                    "question_sha256": text_sha256(question),
                    "answer_sha256": text_sha256(answer),
                    "prompt_token_count": len(prompt_ids),
                    "prompt_token_ids_sha256": canonical_json_sha256(prompt_ids),
                    "cache_parity": parity,
                    "task_results_used_for_selector_decision": False,
                    "calibration_trace_only": bool(
                        args.calibration_trace_only
                    ),
                    "conditions": {
                        "vanilla": vanilla_condition,
                        "v3": v3_condition | {
                            "online_diagnostics": diagnostics,
                            "runtime_trace": runtime_trace,
                            "static_selector_trace": (
                                _mapping(runtime_trace.get("static_selector_trace"))
                                if args.system_version == "v3.5"
                                else None
                            ),
                            "query_embedding_sidecar": query_sidecar,
                        },
                    },
                    "paired_generated_token_delta_v3_minus_vanilla": (
                        v3_condition["generated_token_count"]
                        - vanilla_condition["generated_token_count"]
                    ),
                    "sample_runtime_seconds": time.perf_counter() - sample_started,
                }
                row["row_sha256"] = canonical_json_sha256({
                    key: value for key, value in row.items() if key != "created_at"
                })
                output_handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                output_handle.flush()
                os.fsync(output_handle.fileno())
                rows.append(summary_row(row))
                completed_ids.add(sample_id)
                write_json_atomic(report_path, progress_report(
                    status="running",
                    profile_sha256=profile_sha256,
                    selected_count=len(selected),
                    rows=rows,
                    report_schema=report_schema,
                ))
                print(
                    f"[v3-eval] {len(rows)}/{len(selected)} sample={sample_id} "
                    f"strict={int(v3_condition['strict_correct'])} "
                    f"attempts={diagnostics['retrieval_attempt_count']}",
                    flush=True,
                )
    except Exception as error:
        write_json_atomic(report_path, progress_report(
            status="failed_resumable",
            profile_sha256=profile_sha256,
            selected_count=len(selected),
            rows=rows,
            error={"type": type(error).__name__, "message": str(error)},
            report_schema=report_schema,
        ))
        raise
    finally:
        controller.close()

    if len(rows) != len(selected):
        raise RuntimeError("V3 evaluation ended before every selected sample completed")
    final_report = progress_report(
        status="completed",
        profile_sha256=profile_sha256,
        selected_count=len(selected),
        rows=rows,
        report_schema=report_schema,
    )
    final_report["evaluation_interpretation"] = interpretation
    final_report["independent_final_confirmation"] = False
    final_report["report_sha256"] = canonical_json_sha256({
        key: value for key, value in final_report.items() if key != "report_sha256"
    })
    write_json_atomic(report_path, final_report)
    print(
        f"[v3-eval] completed samples={len(rows)} report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
