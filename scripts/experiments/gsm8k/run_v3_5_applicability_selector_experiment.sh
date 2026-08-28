#!/usr/bin/env bash
# V3.5: qualified dual-key selector, 64-sample trace-only calibration, and matched dev.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

V3_BANK_DIR=""
V34_DEV_DIR=""
V31_DEV_DIR=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --v3-bank-dir) V3_BANK_DIR="$2"; shift 2 ;;
    --v34-dev-dir) V34_DEV_DIR="$2"; shift 2 ;;
    --v31-dev-dir) V31_DEV_DIR="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [--v3-bank-dir DIR] [--v34-dev-dir DIR] [--v31-dev-dir DIR] PHASE1_DIR E0_DIR TOKEN_RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
TOKEN_RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"
if [[ -z "$V3_BANK_DIR" ]]; then
  V3_BANK_DIR="$OUTPUT_ROOT/v3_bank"
fi
if [[ -z "$V34_DEV_DIR" ]]; then
  V34_DEV_DIR="$OUTPUT_ROOT/v3_4_continuous_gate/dev_current_token_margin"
fi
if [[ -z "$V31_DEV_DIR" ]]; then
  V31_DEV_DIR="$OUTPUT_ROOT/v3_1_selector/dev_margin"
fi

V35_DIR="$OUTPUT_ROOT/v3_5_applicability_selector"
DUAL_KEY_DIR="$V35_DIR/dual_key_bank"
DUAL_KEY_TENSORS="$DUAL_KEY_DIR/dual_retrieval_key_bank.safetensors"
DUAL_KEY_MANIFEST="$DUAL_KEY_DIR/dual_retrieval_key_manifest.json"
OFFLINE_REPORT="$DUAL_KEY_DIR/offline_report.json"
OFFLINE_REPORT_MD="$DUAL_KEY_DIR/offline_report.md"
APPLICABILITY_CALIBRATION="$V35_DIR/applicability_calibration.json"
APPLICABILITY_CALIBRATION_MD="$V35_DIR/applicability_calibration.md"
CALIBRATION_TRACE_DIR="$V35_DIR/calibration_trace"
SELECTOR_CALIBRATION="$V35_DIR/selector_calibration.json"
DEV_DIR="$V35_DIR/dev"
V34_COMPARISON="$V35_DIR/dev_v35_minus_v34.json"
V31_COMPARISON="$V35_DIR/dev_v35_minus_v31.json"
QUALIFICATION="$V35_DIR/dev_qualification.json"

CALIBRATION_LIMIT=64
DEV_EXPECTED_COUNT=473
PARITY_SAMPLES=8
TARGET_RETAINED_FRACTION=0.5
V35_APPLICABILITY_CALIBRATION_SCHEMA="experience-memory-v3.5-applicability-calibration-v1"
V35_SELECTOR_CALIBRATION_SCHEMA="experience-memory-v3.5-selector-calibration-v1"
V35_DUAL_KEY_BANK_SCHEMA="experience-memory-v3.5-dual-key-bank-v1"
V35_OFFLINE_REPORT_SCHEMA="experience-memory-v3.5-offline-report-v1"
V35_EVALUATION_PROFILE_SCHEMA="experience-memory-v3.5-evaluation-profile-v1"
V35_EVALUATION_REPORT_SCHEMA="experience-memory-v3.5-evaluation-report-v1"
V35_SYSTEM_PROFILE_SCHEMA="experience-memory-system-profile-v3.5"

# A qualified offline artifact is reusable only under the exact compiler code
# identity that produced it. The compiler helper deliberately hashes the same
# 19-file implementation set used by evaluation plus its scoped tracked diff.
OFFLINE_RESUME_VALIDATOR='
import json
import runpy
import subprocess
import sys
from pathlib import Path


def stop(message):
    print(f"V3.5 offline reuse authentication failed: {message}", file=sys.stderr)
    raise SystemExit(1)


try:
    report_path = Path(sys.argv[1])
    calibration_path = Path(sys.argv[2])
    tensor_path = Path(sys.argv[3])
    manifest_path = Path(sys.argv[4])
    split_path = Path(sys.argv[12])
    project_root = Path(sys.argv[16]).resolve()
    evaluator = runpy.run_path(
        str(project_root / "scripts/evaluate_v3_experience_memory.py"),
        run_name="v35_offline_resume_authenticator",
    )
    canonical_json_sha256 = evaluator["canonical_json_sha256"]
    file_sha256 = evaluator["file_sha256"]
    evaluator_repository = evaluator["repository_state"]()
    implementation_files = evaluator_repository[
        "implementation_files_sha256"
    ]
    scoped_diff = subprocess.check_output(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            *implementation_files,
        ],
        cwd=project_root,
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    current_repository = {
        "git_revision": evaluator_repository["git_revision"],
        "tracked_diff_sha256": evaluator["text_sha256"](scoped_diff),
        "implementation_files_sha256": implementation_files,
        "implementation_set_sha256": canonical_json_sha256(
            implementation_files
        ),
    }
    report = json.loads(report_path.read_text(encoding="utf-8"))
    calibration_raw = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report_inputs = report.get("inputs", {})
    calibration_source = calibration_raw.get("source", {})
    manifest_inputs = manifest.get("input_artifacts", {})
    current_compiler_identity = {
        "compiler_git_revision": current_repository["git_revision"],
        "compiler_tracked_diff_sha256": current_repository[
            "tracked_diff_sha256"
        ],
        "compiler_implementation_files_sha256": current_repository[
            "implementation_files_sha256"
        ],
        "compiler_implementation_set_sha256": current_repository[
            "implementation_set_sha256"
        ],
    }
    for owner, provenance in (
        ("offline report", report_inputs),
        ("applicability calibration", calibration_source),
        ("dual-key manifest", manifest_inputs),
    ):
        if not isinstance(provenance, dict) or any(
            provenance.get(key) != value
            for key, value in current_compiler_identity.items()
        ):
            stop(f"{owner} compiler implementation identity differs")
    if report.get("compiler_git_revision") != current_repository["git_revision"]:
        stop("offline report top-level compiler revision differs")

    from memgen.experience.v3_5_selector import (
        load_v35_applicability_calibration,
    )
    split_manifest = evaluator["load_split_manifest"](split_path)
    old_manifest_path = Path(sys.argv[8])
    old_manifest = json.loads(
        old_manifest_path.read_text(encoding="utf-8")
    )
    old_tensor_relative = Path(
        str(old_manifest.get("tensor_artifact", {}).get("path", ""))
    )
    old_tensor_path = (
        old_manifest_path.parent / old_tensor_relative
    ).resolve()
    if (
        not old_tensor_relative.parts
        or old_tensor_relative.is_absolute()
        or ".." in old_tensor_relative.parts
        or old_manifest_path.parent.resolve() not in old_tensor_path.parents
        or not old_tensor_path.is_file()
        or old_manifest.get("manifest_sha256")
        != canonical_json_sha256({
            key: value
            for key, value in old_manifest.items()
            if key != "manifest_sha256"
        })
        or old_manifest.get("tensor_artifact", {}).get("sha256")
        != file_sha256(old_tensor_path)
    ):
        stop("legacy V3 applicability bank identity is invalid")
    expected_inputs = {
        "memory_records_sha256": file_sha256(Path(sys.argv[5])),
        "side_kv_manifest_sha256": file_sha256(Path(sys.argv[6])),
        "e0_final_report_sha256": file_sha256(Path(sys.argv[7])),
        "v3_retrieval_key_manifest_sha256": file_sha256(old_manifest_path),
        "v3_retrieval_key_tensor_sha256": file_sha256(old_tensor_path),
        "v3_offline_report_sha256": file_sha256(Path(sys.argv[9])),
        "phase1_approved_bank_sha256": file_sha256(Path(sys.argv[10])),
        "verified_experiences_sha256": file_sha256(Path(sys.argv[11])),
        "split_manifest_sha256": file_sha256(split_path),
        "split_manifest_logical_sha256": split_manifest["manifest_sha256"],
        "dataset_revision": split_manifest["dataset"]["revision"],
        **current_compiler_identity,
    }
    for owner, provenance in (
        ("offline report", report_inputs),
        ("applicability calibration", calibration_source),
        ("dual-key manifest", manifest_inputs),
    ):
        if any(
            provenance.get(key) != value
            for key, value in expected_inputs.items()
        ):
            stop(f"{owner} input provenance differs")

    manifest_sha256 = canonical_json_sha256({
        key: value
        for key, value in manifest.items()
        if key != "manifest_sha256"
    })
    manifest_tensor = manifest.get("tensor_artifact", {})
    manifest_tensor_relative = Path(str(manifest_tensor.get("path", "")))
    manifest_tensor_path = (
        manifest_path.parent / manifest_tensor_relative
    ).resolve()
    if (
        manifest.get("schema_version") != sys.argv[14]
        or manifest.get("manifest_sha256") != manifest_sha256
        or not manifest_tensor_relative.parts
        or manifest_tensor_relative.is_absolute()
        or ".." in manifest_tensor_relative.parts
        or manifest_tensor_path != tensor_path.resolve()
        or manifest_tensor.get("sha256") != file_sha256(tensor_path)
    ):
        stop("dual-key manifest logical/tensor identity differs")
    calibration = load_v35_applicability_calibration(
        calibration_path,
        expected_input_hashes={
            "dual_key_manifest_sha256": file_sha256(manifest_path)
        },
    )
    if (
        calibration.get("schema_version") != sys.argv[13]
        or calibration.get("status") != "passed"
        or calibration.get("task_accuracy_used") is not False
        or calibration.get("answer_or_reward_used") is not False
        or calibration_source.get("dual_key_manifest_logical_sha256")
        != manifest_sha256
        or calibration_source.get("dual_key_tensor_sha256")
        != file_sha256(tensor_path)
    ):
        stop("applicability calibration is mismatched or not qualified")
    requirements = report.get("requirements")
    artifacts = report.get("artifacts", {})
    report_sha256 = canonical_json_sha256({
        key: value for key, value in report.items() if key != "report_sha256"
    })
    if (
        report.get("schema_version") != sys.argv[15]
        or report.get("status") != "passed"
        or report.get("formal_v3_5_offline_passed") is not True
        or report.get("task_accuracy_used") is not False
        or report.get("answer_or_reward_used") is not False
        or not isinstance(requirements, dict)
        or not requirements
        or not all(value is True for value in requirements.values())
        or report.get("report_sha256") != report_sha256
        or artifacts.get("dual_key_tensor", {}).get("sha256")
        != file_sha256(tensor_path)
        or artifacts.get("dual_key_manifest", {}).get("sha256")
        != file_sha256(manifest_path)
        or artifacts.get("dual_key_manifest", {}).get("logical_sha256")
        != manifest_sha256
        or artifacts.get("applicability_calibration", {}).get("sha256")
        != file_sha256(calibration_path)
    ):
        stop("offline report is mismatched or not qualified")
except SystemExit:
    raise
except Exception as error:
    stop(f"{type(error).__name__}: {error}")
'

# Keep this verifier self-contained and shared by Stage B/C resume checks. It
# imports the evaluator definitions so profile, row, report, and repository
# hashes cannot drift from the writer implementation.
RUN_RESUME_VALIDATOR='
import json
import runpy
import sys
from pathlib import Path


def stop(message):
    print(f"V3.5 resume authentication failed: {message}", file=sys.stderr)
    raise SystemExit(1)


try:
    profile_path = Path(sys.argv[1])
    report_path = Path(sys.argv[2])
    results_path = profile_path.parent / "results.jsonl"
    expected_split = sys.argv[3]
    expected_count = int(sys.argv[4])
    expected_trace = sys.argv[5] == "true"
    project_root = Path(sys.argv[19]).resolve()
    evaluator = runpy.run_path(
        str(project_root / "scripts/evaluate_v3_experience_memory.py"),
        run_name="v35_resume_authenticator",
    )
    canonical_json_sha256 = evaluator["canonical_json_sha256"]
    file_sha256 = evaluator["file_sha256"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_profile_schema, expected_row_schema, expected_report_schema = (
        evaluator["evaluation_schemas"]("v3.5")
    )
    if (
        expected_profile_schema != sys.argv[16]
        or expected_report_schema != sys.argv[17]
    ):
        stop("runner/evaluator schema constants differ")
    actual_profile_sha256 = evaluator["evaluation_profile_sha256"](profile)
    if profile.get("profile_sha256") != actual_profile_sha256:
        stop("run_profile logical profile_sha256 mismatch")
    stored_repository = profile.get("repository", {})
    current_repository = evaluator["repository_state"]()
    repository_identity_fields = (
        "git_revision",
        "tracked_diff_sha256",
        "implementation_files_sha256",
        "missing_implementation_files",
        "implementation_set_sha256",
    )
    if any(
        stored_repository.get(field) != current_repository.get(field)
        for field in repository_identity_fields
    ):
        stop("current repository/code implementation identity differs")
    system = profile.get("system_profile", {})
    if (
        system.get("schema_version") != sys.argv[18]
        or profile.get("system_profile_sha256")
        != canonical_json_sha256(system)
    ):
        stop("system profile schema/hash mismatch")
    trace = profile.get(
        "calibration_trace_only",
        system.get("calibration_trace_only", False),
    )
    expected_limit = expected_count if expected_trace else 0
    logging = profile.get("logging", {})
    decision_contract = profile.get("selector_decision_data_contract", {})
    if (
        profile.get("schema_version") != expected_profile_schema
        or profile.get("system_version") != "v3.5"
        or profile.get("logical_split") != expected_split
        or int(profile.get("selected_sample_count", -1)) != expected_count
        or int(profile.get("slice", {}).get("offset", -1)) != 0
        or int(profile.get("slice", {}).get("limit", -1)) != expected_limit
        or bool(trace) is not expected_trace
        or logging.get("query_embeddings_sidecar") is not expected_trace
        or logging.get("query_embeddings_sidecar_required_for_calibration")
        is not expected_trace
        or logging.get("query_embedding_sidecar_representation")
        != "dynamic_query_l2_normalized_exact_audit"
        or profile.get("reasoner", {}).get("runtime_dtype") != "bfloat16"
        or profile.get("task_results_used_for_selector_decision") is not False
        or decision_contract.get("task_accuracy_used") is not False
        or decision_contract.get("answer_or_reward_used") is not False
        or decision_contract.get("first_attempt_dynamic_margins_only")
        is not expected_trace
    ):
        stop("run profile does not match the frozen Stage B/C contract")
    inputs = profile.get("inputs", {})
    expected_inputs = {
        "split_manifest_sha256": file_sha256(Path(sys.argv[6])),
        "memory_records_sha256": file_sha256(Path(sys.argv[7])),
        "retrieval_key_manifest_sha256": file_sha256(Path(sys.argv[8])),
        "side_kv_manifest_sha256": file_sha256(Path(sys.argv[9])),
        "v3_offline_report_sha256": file_sha256(Path(sys.argv[10])),
        "e0_final_report_sha256": file_sha256(Path(sys.argv[11])),
        "risk_artifact_sha256": file_sha256(Path(sys.argv[12])),
        "dual_key_manifest_sha256": file_sha256(Path(sys.argv[13])),
        "applicability_calibration_sha256": file_sha256(Path(sys.argv[14])),
        "selector_calibration_sha256": (
            None if expected_trace else file_sha256(Path(sys.argv[15]))
        ),
    }
    if any(inputs.get(key) != value for key, value in expected_inputs.items()):
        stop("run profile input artifact hashes differ")
    split_manifest = evaluator["load_split_manifest"](Path(sys.argv[6]))
    expected_samples = [
        item
        for item in split_manifest["samples"]
        if item.get("logical_split") == expected_split
    ]
    if expected_trace:
        expected_samples = expected_samples[:expected_count]
    expected_sample_ids = [
        str(item.get("sample_id", "")) for item in expected_samples
    ]
    expected_sample_by_id = {
        str(item.get("sample_id", "")): item for item in expected_samples
    }
    if (
        len(expected_samples) != expected_count
        or len(expected_sample_by_id) != expected_count
        or canonical_json_sha256(expected_sample_ids)
        != profile.get("selected_sample_ids_sha256")
    ):
        stop("run profile sample selection differs from split manifest")
    rows = []
    compact_rows = []
    sample_ids = []
    seen_sample_ids = set()
    seen_sidecar_paths = set()
    with results_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            expected_sample = expected_sample_by_id.get(sample_id, {})
            actual_row_sha256 = canonical_json_sha256({
                key: value
                for key, value in row.items()
                if key not in {"created_at", "row_sha256"}
            })
            if (
                row.get("schema_version") != expected_row_schema
                or row.get("profile_sha256") != actual_profile_sha256
                or row.get("row_sha256") != actual_row_sha256
                or row.get("logical_split") != expected_split
                or row.get("task_results_used_for_selector_decision") is not False
                or row.get("calibration_trace_only") is not expected_trace
                or row.get("dataset_split")
                != expected_sample.get("dataset_split")
                or int(row.get("source_index", -1))
                != int(expected_sample.get("source_index", -2))
                or row.get("question_sha256")
                != expected_sample.get("question_sha256")
                or row.get("answer_sha256")
                != expected_sample.get("answer_sha256")
                or not sample_id
                or sample_id in seen_sample_ids
            ):
                stop(f"invalid or duplicate results row at line {line_number}")
            for condition_name in ("vanilla", "v3"):
                condition = row.get("conditions", {}).get(condition_name, {})
                token_ids = [
                    int(value)
                    for value in condition.get("completion_token_ids", [])
                ]
                if (
                    int(condition.get("generated_token_count", -1))
                    != len(token_ids)
                    or condition.get("completion_token_ids_sha256")
                    != canonical_json_sha256(token_ids)
                ):
                    stop(
                        f"completion hash/count mismatch at line {line_number}"
                    )
            v3_condition = row.get("conditions", {}).get("v3", {})
            runtime_trace = v3_condition.get("runtime_trace", {})
            attempts = runtime_trace.get("retrieval_attempts")
            descriptor = v3_condition.get("query_embedding_sidecar")
            if not isinstance(attempts, list):
                stop(f"missing retrieval attempt trace at line {line_number}")
            if expected_trace and attempts:
                if not isinstance(descriptor, dict):
                    stop(f"missing query embedding sidecar at line {line_number}")
                relative = Path(str(descriptor.get("path", "")))
                sidecar_path = (results_path.parent / relative).resolve()
                if (
                    not relative.parts
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or results_path.parent.resolve() not in sidecar_path.parents
                    or sidecar_path in seen_sidecar_paths
                    or not sidecar_path.is_file()
                    or descriptor.get("sha256") != file_sha256(sidecar_path)
                    or int(descriptor.get("attempt_count", -1)) != len(attempts)
                    or descriptor.get("representation")
                    != "dynamic_query_l2_normalized_exact_audit"
                ):
                    stop(
                        f"invalid query embedding sidecar at line {line_number}"
                    )
                seen_sidecar_paths.add(sidecar_path)
            elif descriptor is not None:
                stop(f"unexpected query embedding sidecar at line {line_number}")
            seen_sample_ids.add(sample_id)
            sample_ids.append(sample_id)
            rows.append(row)
            compact_rows.append(evaluator["summary_row"](row))
    if (
        len(rows) != expected_count
        or sample_ids != expected_sample_ids
    ):
        stop("results count/order/selected sample IDs differ from profile")
    expected_summary = evaluator["summarize_v3_rows"](compact_rows)
    actual_report_sha256 = canonical_json_sha256({
        key: value
        for key, value in report.items()
        if key != "report_sha256"
    })
    if (
        report.get("schema_version") != expected_report_schema
        or report.get("status") != "completed"
        or report.get("profile_sha256") != actual_profile_sha256
        or int(report.get("selected_sample_count", -1)) != expected_count
        or int(report.get("completed_sample_count", -1)) != len(rows)
        or int(report.get("remaining_sample_count", -1)) != 0
        or report.get("error") is not None
        or report.get("summary") != expected_summary
        or report.get("report_sha256") != actual_report_sha256
    ):
        stop("run_report hash/count/summary does not match authenticated rows")
except SystemExit:
    raise
except Exception as error:
    stop(f"{type(error).__name__}: {error}")
'
DEVICE="${MEMGEN_V35_DEVICE:-cuda}"
DTYPE="${MEMGEN_V35_DTYPE:-bfloat16}"
if [[ "$DTYPE" != "bfloat16" ]]; then
  echo "V3.5 requires MEMGEN_V35_DTYPE=bfloat16" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${MEMGEN_V35_CUDA_VISIBLE_DEVICES:-0}"
V34_COMPARISON_COMPLETED=false
V31_COMPARISON_COMPLETED=false
QUALIFICATION_COMPLETED=false

SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
VERIFIED_EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
E0_FINAL_REPORT="$E0_DIR/e0_final_report.json"
V3_KEY_MANIFEST="$V3_BANK_DIR/retrieval_key_manifest.json"
V3_OFFLINE_REPORT="$V3_BANK_DIR/v3_offline_report.json"

mkdir -p "$DUAL_KEY_DIR"

for REQUIRED in \
  "$SPLIT_MANIFEST" \
  "$APPROVED_BANK" \
  "$VERIFIED_EXPERIENCES" \
  "$MEMORY_RECORDS" \
  "$SIDE_KV_MANIFEST" \
  "$E0_FINAL_REPORT" \
  "$V3_KEY_MANIFEST" \
  "$V3_OFFLINE_REPORT" \
  "$TOKEN_RISK_ARTIFACT"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

offline_files_complete() {
  [[ -s "$DUAL_KEY_TENSORS" && -s "$DUAL_KEY_MANIFEST" && \
     -s "$OFFLINE_REPORT" && -s "$OFFLINE_REPORT_MD" && \
     -s "$APPLICABILITY_CALIBRATION" && \
     -s "$APPLICABILITY_CALIBRATION_MD" ]]
}

offline_is_passed() {
  offline_files_complete && \
    python -c "$OFFLINE_RESUME_VALIDATOR" \
      "$OFFLINE_REPORT" "$APPLICABILITY_CALIBRATION" \
      "$DUAL_KEY_TENSORS" "$DUAL_KEY_MANIFEST" \
      "$MEMORY_RECORDS" "$SIDE_KV_MANIFEST" "$E0_FINAL_REPORT" \
      "$V3_KEY_MANIFEST" "$V3_OFFLINE_REPORT" "$APPROVED_BANK" \
      "$VERIFIED_EXPERIENCES" "$SPLIT_MANIFEST" \
      "$V35_APPLICABILITY_CALIBRATION_SCHEMA" \
      "$V35_DUAL_KEY_BANK_SCHEMA" "$V35_OFFLINE_REPORT_SCHEMA" \
      "$REPO_ROOT"
}

run_is_complete() {
  local run_dir="$1"
  local logical_split="$2"
  local expected_count="$3"
  local trace_only="$4"
  [[ -s "$run_dir/results.jsonl" && -s "$run_dir/run_profile.json" && \
     -s "$run_dir/run_report.json" ]] && \
    python -c "$RUN_RESUME_VALIDATOR" \
      "$run_dir/run_profile.json" "$run_dir/run_report.json" \
      "$logical_split" "$expected_count" "$trace_only" \
      "$SPLIT_MANIFEST" "$MEMORY_RECORDS" "$V3_KEY_MANIFEST" \
      "$SIDE_KV_MANIFEST" "$V3_OFFLINE_REPORT" "$E0_FINAL_REPORT" \
      "$TOKEN_RISK_ARTIFACT" "$DUAL_KEY_MANIFEST" \
      "$APPLICABILITY_CALIBRATION" "$SELECTOR_CALIBRATION" \
      "$V35_EVALUATION_PROFILE_SCHEMA" "$V35_EVALUATION_REPORT_SCHEMA" \
      "$V35_SYSTEM_PROFILE_SCHEMA" "$REPO_ROOT"
}

selector_is_passed() {
  [[ -s "$SELECTOR_CALIBRATION" && \
     -s "${SELECTOR_CALIBRATION%.json}.md" ]] && \
    python -c 'import math,sys; from pathlib import Path; from memgen.experience.phase1 import file_sha256; from memgen.experience.v3_5_selector import load_v35_selector_calibration; value=load_v35_selector_calibration(Path(sys.argv[1]), expected_input_hashes={"dual_key_manifest_sha256":file_sha256(Path(sys.argv[4])),"applicability_calibration_sha256":file_sha256(Path(sys.argv[5]))}); source=value.get("source", {}); calibration=value.get("calibration", {}); ok=value.get("schema_version")==sys.argv[7] and value.get("status")=="passed" and value.get("task_accuracy_used") is False and value.get("answer_or_reward_used") is False and all(value.get("requirements", {}).values()) and source.get("logical_split")=="calibration-val" and source.get("system_version")=="v3.5" and source.get("results_file_sha256")==file_sha256(Path(sys.argv[2])) and source.get("run_profile_file_sha256")==file_sha256(Path(sys.argv[3])) and source.get("risk_artifact_sha256")==file_sha256(Path(sys.argv[6])) and math.isclose(float(calibration.get("target_retained_fraction", -1.0)),float(sys.argv[8]),rel_tol=0.0,abs_tol=1e-12); raise SystemExit(0 if ok else 1)' \
      "$SELECTOR_CALIBRATION" \
      "$CALIBRATION_TRACE_DIR/results.jsonl" \
      "$CALIBRATION_TRACE_DIR/run_profile.json" \
      "$DUAL_KEY_MANIFEST" "$APPLICABILITY_CALIBRATION" \
      "$TOKEN_RISK_ARTIFACT" "$V35_SELECTOR_CALIBRATION_SCHEMA" \
      "$TARGET_RETAINED_FRACTION"
}

if offline_files_complete; then
  if ! offline_is_passed; then
    echo "Existing V3.5 offline artifacts are invalid, mismatched, or not qualified: $DUAL_KEY_DIR" >&2
    exit 3
  fi
  echo "Reusing qualified V3.5 dual-key artifacts: $DUAL_KEY_DIR"
else
  if [[ -e "$OFFLINE_REPORT" || -e "$DUAL_KEY_MANIFEST" || \
        -e "$DUAL_KEY_TENSORS" || -e "$APPLICABILITY_CALIBRATION" ]]; then
    echo "Completing an interrupted V3.5 offline build: $DUAL_KEY_DIR"
  fi
  python scripts/compile_v3_5_dual_selector.py \
    --memory-records "$MEMORY_RECORDS" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --e0-final-report "$E0_FINAL_REPORT" \
    --approved-bank "$APPROVED_BANK" \
    --verified-experiences "$VERIFIED_EXPERIENCES" \
    --v3-retrieval-key-manifest "$V3_KEY_MANIFEST" \
    --v3-offline-report "$V3_OFFLINE_REPORT" \
    --split-manifest "$SPLIT_MANIFEST" \
    --output-dir "$DUAL_KEY_DIR" \
    --applicability-calibration-output "$APPLICABILITY_CALIBRATION" \
    --applicability-calibration-markdown-output "$APPLICABILITY_CALIBRATION_MD" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
  if ! offline_is_passed; then
    echo "V3.5 offline applicability qualification did not pass; online work is blocked." >&2
    exit 3
  fi
fi

if ! run_is_complete \
  "$CALIBRATION_TRACE_DIR" calibration-val "$CALIBRATION_LIMIT" true; then
  python scripts/evaluate_v3_experience_memory.py \
    --system-version v3.5 \
    --split-manifest "$SPLIT_MANIFEST" \
    --logical-split calibration-val \
    --memory-records "$MEMORY_RECORDS" \
    --retrieval-key-manifest "$V3_KEY_MANIFEST" \
    --dual-key-manifest "$DUAL_KEY_MANIFEST" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --v3-offline-report "$V3_OFFLINE_REPORT" \
    --e0-final-report "$E0_FINAL_REPORT" \
    --risk-artifact "$TOKEN_RISK_ARTIFACT" \
    --applicability-calibration "$APPLICABILITY_CALIBRATION" \
    --calibration-trace-only \
    --save-query-embeddings \
    --output-dir "$CALIBRATION_TRACE_DIR" \
    --offset 0 \
    --limit "$CALIBRATION_LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
  if ! run_is_complete \
    "$CALIBRATION_TRACE_DIR" calibration-val "$CALIBRATION_LIMIT" true; then
    echo "V3.5 calibration trace did not satisfy the completed-run authentication contract." >&2
    exit 3
  fi
else
  echo "Reusing complete V3.5 calibration trace: $CALIBRATION_TRACE_DIR"
fi

if [[ -e "$SELECTOR_CALIBRATION" ]]; then
  if ! selector_is_passed; then
    echo "Existing V3.5 selector calibration is invalid, mismatched, or not qualified: $SELECTOR_CALIBRATION" >&2
    exit 3
  fi
  echo "Reusing qualified V3.5 dynamic selector: $SELECTOR_CALIBRATION"
else
  python scripts/calibrate_v3_5_dynamic_selector.py \
    --results "$CALIBRATION_TRACE_DIR/results.jsonl" \
    --run-profile "$CALIBRATION_TRACE_DIR/run_profile.json" \
    --dual-key-manifest "$DUAL_KEY_MANIFEST" \
    --applicability-calibration "$APPLICABILITY_CALIBRATION" \
    --target-retained-fraction "$TARGET_RETAINED_FRACTION" \
    --output "$SELECTOR_CALIBRATION"
  if ! selector_is_passed; then
    echo "V3.5 dynamic selector calibration did not qualify; dev-test is blocked." >&2
    exit 3
  fi
fi

if ! run_is_complete "$DEV_DIR" dev-test "$DEV_EXPECTED_COUNT" false; then
  python scripts/evaluate_v3_experience_memory.py \
    --system-version v3.5 \
    --split-manifest "$SPLIT_MANIFEST" \
    --logical-split dev-test \
    --memory-records "$MEMORY_RECORDS" \
    --retrieval-key-manifest "$V3_KEY_MANIFEST" \
    --dual-key-manifest "$DUAL_KEY_MANIFEST" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --v3-offline-report "$V3_OFFLINE_REPORT" \
    --e0-final-report "$E0_FINAL_REPORT" \
    --risk-artifact "$TOKEN_RISK_ARTIFACT" \
    --applicability-calibration "$APPLICABILITY_CALIBRATION" \
    --selector-calibration "$SELECTOR_CALIBRATION" \
    --output-dir "$DEV_DIR" \
    --offset 0 \
    --limit 0 \
    --parity-samples "$PARITY_SAMPLES" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
  if ! run_is_complete "$DEV_DIR" dev-test "$DEV_EXPECTED_COUNT" false; then
    echo "V3.5 matched dev did not satisfy the completed-run authentication contract." >&2
    exit 3
  fi
else
  echo "Reusing complete V3.5 matched dev: $DEV_DIR"
fi

python scripts/analyze_v3_evaluation.py \
  --results "$DEV_DIR/results.jsonl" \
  --run-profile "$DEV_DIR/run_profile.json" \
  --output "$DEV_DIR/analysis_report.json" \
  --markdown-output "$DEV_DIR/analysis_report.md"

# Baseline comparisons are optional, but each requested/reused baseline is validated
# by the comparison script before it can contribute to qualification.
if [[ -s "$V34_DEV_DIR/results.jsonl" && -s "$V34_DEV_DIR/run_profile.json" ]]; then
  python scripts/compare_v3_5_applicability_selector.py \
    --v35-results "$DEV_DIR/results.jsonl" \
    --v35-profile "$DEV_DIR/run_profile.json" \
    --v35-selector-calibration "$SELECTOR_CALIBRATION" \
    --baseline-results "$V34_DEV_DIR/results.jsonl" \
    --baseline-profile "$V34_DEV_DIR/run_profile.json" \
    --baseline-version v3.4 \
    --output "$V34_COMPARISON"
  python scripts/qualify_v3_5_dev.py \
    --comparison "$V34_COMPARISON" \
    --analysis "$DEV_DIR/analysis_report.json" \
    --selector-calibration "$SELECTOR_CALIBRATION" \
    --output "$QUALIFICATION"
  V34_COMPARISON_COMPLETED=true
  QUALIFICATION_COMPLETED=true
else
  if [[ -e "$V34_COMPARISON" || -e "${V34_COMPARISON%.json}.md" || \
        -e "$QUALIFICATION" || -e "${QUALIFICATION%.json}.md" ]]; then
    echo "Cannot authenticate existing V3.4 comparison/qualification outputs without their baseline: $V34_DEV_DIR" >&2
    exit 1
  fi
  echo "Skipping V3.5-minus-V3.4 comparison and qualification; compatible baseline results/profile not found in $V34_DEV_DIR" >&2
fi

if [[ -s "$V31_DEV_DIR/results.jsonl" && -s "$V31_DEV_DIR/run_profile.json" ]]; then
  python scripts/compare_v3_5_applicability_selector.py \
    --v35-results "$DEV_DIR/results.jsonl" \
    --v35-profile "$DEV_DIR/run_profile.json" \
    --v35-selector-calibration "$SELECTOR_CALIBRATION" \
    --baseline-results "$V31_DEV_DIR/results.jsonl" \
    --baseline-profile "$V31_DEV_DIR/run_profile.json" \
    --baseline-version v3.1 \
    --output "$V31_COMPARISON"
  V31_COMPARISON_COMPLETED=true
else
  if [[ -e "$V31_COMPARISON" || -e "${V31_COMPARISON%.json}.md" ]]; then
    echo "Cannot authenticate existing V3.1 comparison outputs without their baseline: $V31_DEV_DIR" >&2
    exit 1
  fi
  echo "Skipping V3.5-minus-V3.1 comparison; compatible baseline results/profile not found in $V31_DEV_DIR" >&2
fi

echo "V3.5 experiment completed at the exploratory matched-dev stop: $V35_DIR"
echo "Offline dual-key report: $OFFLINE_REPORT_MD"
echo "Applicability calibration: $APPLICABILITY_CALIBRATION_MD"
echo "Dynamic selector calibration: ${SELECTOR_CALIBRATION%.json}.md"
echo "Matched dev analysis: $DEV_DIR/analysis_report.md"
if [[ "$V34_COMPARISON_COMPLETED" == true ]]; then
  echo "V3.5 minus V3.4: ${V34_COMPARISON%.json}.md"
fi
if [[ "$V31_COMPARISON_COMPLETED" == true ]]; then
  echo "V3.5 minus V3.1: ${V31_COMPARISON%.json}.md"
fi
if [[ "$QUALIFICATION_COMPLETED" == true ]]; then
  echo "Exploratory qualification: ${QUALIFICATION%.json}.md"
fi
echo "Final-test remains blocked and is never started by this runner."
