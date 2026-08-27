#!/usr/bin/env bash
# V3.2 retrieval-only experiment: shared-centroid keys/queries plus recalibrated margin.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CALIBRATION_LIMIT=0
DEV_LIMIT=0
PARITY_SAMPLES=8
TARGET_RETAINED_FRACTION=0.5
V31_DEV_DIR=""
V31_SELECTOR_CALIBRATION=""
CENTERED_CALIBRATION_DIR=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --calibration-limit) CALIBRATION_LIMIT="$2"; shift 2 ;;
    --dev-limit) DEV_LIMIT="$2"; shift 2 ;;
    --parity-samples) PARITY_SAMPLES="$2"; shift 2 ;;
    --target-retained-fraction) TARGET_RETAINED_FRACTION="$2"; shift 2 ;;
    --v31-dev-dir) V31_DEV_DIR="$2"; shift 2 ;;
    --v31-selector-calibration) V31_SELECTOR_CALIBRATION="$2"; shift 2 ;;
    --centered-calibration-dir) CENTERED_CALIBRATION_DIR="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [options] PHASE1_DIR E0_DIR RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi
for VALUE in "$CALIBRATION_LIMIT" "$DEV_LIMIT" "$PARITY_SAMPLES"; do
  if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
    echo "calibration-limit, dev-limit, and parity-samples must be non-negative integers" >&2
    exit 2
  fi
done
if ! [[ "$TARGET_RETAINED_FRACTION" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || [[ "$TARGET_RETAINED_FRACTION" =~ ^0(\.0+)?$ ]]; then
  echo "target-retained-fraction must be in (0, 1]" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"
V3_BANK_DIR="$OUTPUT_ROOT/v3_bank"
V31_DIR="$OUTPUT_ROOT/v3_1_selector"
V32_DIR="$OUTPUT_ROOT/v3_2_centered_retrieval"
if [[ -z "$V31_DEV_DIR" ]]; then
  V31_DEV_DIR="$V31_DIR/dev_margin"
fi
if [[ -z "$V31_SELECTOR_CALIBRATION" ]]; then
  V31_SELECTOR_CALIBRATION="$V31_DIR/margin_selector_calibration.json"
fi
if [[ -z "$CENTERED_CALIBRATION_DIR" ]]; then
  CENTERED_CALIBRATION_DIR="$V32_DIR/calibration_centered_disabled"
fi
V32_DEV_DIR="$V32_DIR/dev_centered_margin"
V32_SELECTOR_CALIBRATION="$V32_DIR/centered_margin_selector_calibration.json"
CALIBRATION_QUALIFICATION="$V32_DIR/calibration_geometry_qualification.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
KEY_MANIFEST="$V3_BANK_DIR/retrieval_key_manifest.json"
TRANSFORM="key_bank_centroid_center_l2"

mkdir -p "$V32_DIR"
run_is_complete() {
  local run_dir="$1"
  [[ -s "$run_dir/results.jsonl" && -s "$run_dir/run_profile.json" && -s "$run_dir/run_report.json" ]] && \
    python -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if value.get("status") == "completed" and value.get("remaining_sample_count") == 0 else 1)' "$run_dir/run_report.json"
}

for REQUIRED in \
  "$PHASE1_DIR/split_manifest.json" \
  "$MEMORY_RECORDS" \
  "$E0_DIR/side_kv_manifest.json" \
  "$E0_DIR/e0_final_report.json" \
  "$KEY_MANIFEST" \
  "$V3_BANK_DIR/v3_offline_report.json" \
  "$RISK_ARTIFACT" \
  "$V31_SELECTOR_CALIBRATION"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done
if ! run_is_complete "$V31_DEV_DIR"; then
  echo "A complete V3.1 raw-margin dev run is required: $V31_DEV_DIR" >&2
  echo "Run run_v3_1_selector_experiment.sh first or pass --v31-dev-dir." >&2
  exit 1
fi

python scripts/audit_v3_retrieval_geometry.py \
  --retrieval-key-manifest "$KEY_MANIFEST" \
  --memory-records "$MEMORY_RECORDS" \
  --retrieval-embedding-transform none \
  --output "$V32_DIR/raw_retrieval_geometry_audit.json"

python scripts/audit_v3_retrieval_geometry.py \
  --retrieval-key-manifest "$KEY_MANIFEST" \
  --memory-records "$MEMORY_RECORDS" \
  --retrieval-embedding-transform "$TRANSFORM" \
  --output "$V32_DIR/centered_retrieval_geometry_audit.json"

if ! run_is_complete "$CENTERED_CALIBRATION_DIR"; then
  bash scripts/experiments/gsm8k/run_v3_system.sh \
    --stage eval \
    --logical-split calibration-val \
    --limit "$CALIBRATION_LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --retrieval-embedding-transform "$TRANSFORM" \
    --run-dir "$CENTERED_CALIBRATION_DIR" \
    "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
else
  echo "Reusing centered calibration run: $CENTERED_CALIBRATION_DIR"
fi

if [[ ! -s "$V32_SELECTOR_CALIBRATION" ]]; then
  python scripts/calibrate_v3_margin_selector.py \
    --results "$CENTERED_CALIBRATION_DIR/results.jsonl" \
    --run-profile "$CENTERED_CALIBRATION_DIR/run_profile.json" \
    --retrieval-key-manifest "$KEY_MANIFEST" \
    --target-retained-fraction "$TARGET_RETAINED_FRACTION" \
    --output "$V32_SELECTOR_CALIBRATION"
else
  echo "Reusing centered selector calibration: $V32_SELECTOR_CALIBRATION"
fi

python scripts/qualify_v3_centered_calibration.py \
  --v31-calibration "$V31_SELECTOR_CALIBRATION" \
  --v32-calibration "$V32_SELECTOR_CALIBRATION" \
  --output "$CALIBRATION_QUALIFICATION"
if ! python -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if value.get("qualified_for_dev_test") is True else 1)' "$CALIBRATION_QUALIFICATION"; then
  echo "Centered retrieval did not reduce both calibration top-1 share and Gini." >&2
  echo "Stopping before dev-test; inspect $CALIBRATION_QUALIFICATION" >&2
  exit 3
fi

if ! run_is_complete "$V32_DEV_DIR"; then
  bash scripts/experiments/gsm8k/run_v3_system.sh \
    --stage eval \
    --logical-split dev-test \
    --limit "$DEV_LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --retrieval-embedding-transform "$TRANSFORM" \
    --selector-calibration "$V32_SELECTOR_CALIBRATION" \
    --run-dir "$V32_DEV_DIR" \
    "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
else
  echo "Reusing centered dev run: $V32_DEV_DIR"
fi

for RUN_DIR in "$V31_DEV_DIR" "$V32_DEV_DIR"; do
  python scripts/analyze_v3_evaluation.py \
    --results "$RUN_DIR/results.jsonl" \
    --run-profile "$RUN_DIR/run_profile.json" \
    --output "$RUN_DIR/analysis_report.json" \
    --markdown-output "$RUN_DIR/analysis_report.md"
done

python scripts/compare_v3_retrieval_transforms.py \
  --v31-results "$V31_DEV_DIR/results.jsonl" \
  --v31-profile "$V31_DEV_DIR/run_profile.json" \
  --v31-calibration "$V31_SELECTOR_CALIBRATION" \
  --v32-results "$V32_DEV_DIR/results.jsonl" \
  --v32-profile "$V32_DEV_DIR/run_profile.json" \
  --v32-calibration "$V32_SELECTOR_CALIBRATION" \
  --memory-records "$MEMORY_RECORDS" \
  --output "$V32_DIR/dev_retrieval_transform_comparison.json"

echo "V3.2 experiment completed: $V32_DIR"
echo "Raw geometry: $V32_DIR/raw_retrieval_geometry_audit.md"
echo "Centered geometry: $V32_DIR/centered_retrieval_geometry_audit.md"
echo "Centered calibration: $V32_DIR/centered_margin_selector_calibration.md"
echo "Calibration gate: $V32_DIR/calibration_geometry_qualification.md"
echo "Matched comparison: $V32_DIR/dev_retrieval_transform_comparison.md"
