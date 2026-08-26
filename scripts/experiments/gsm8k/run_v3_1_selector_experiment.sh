#!/usr/bin/env bash
# V3.1 selector-only experiment: audit, answer-blind calibration, matched dev test.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

CALIBRATION_LIMIT=0
DEV_LIMIT=0
PARITY_SAMPLES=8
TARGET_RETAINED_FRACTION=0.5
CALIBRATION_BASELINE_DIR=""
DEV_BASELINE_DIR=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --calibration-limit) CALIBRATION_LIMIT="$2"; shift 2 ;;
    --dev-limit) DEV_LIMIT="$2"; shift 2 ;;
    --parity-samples) PARITY_SAMPLES="$2"; shift 2 ;;
    --target-retained-fraction) TARGET_RETAINED_FRACTION="$2"; shift 2 ;;
    --calibration-baseline-dir) CALIBRATION_BASELINE_DIR="$2"; shift 2 ;;
    --dev-baseline-dir) DEV_BASELINE_DIR="$2"; shift 2 ;;
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
if ! [[ "$TARGET_RETAINED_FRACTION" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]]; then
  echo "target-retained-fraction must be in (0, 1]" >&2
  exit 2
fi
if [[ "$TARGET_RETAINED_FRACTION" =~ ^0(\.0+)?$ ]]; then
  echo "target-retained-fraction must be positive" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"
V3_BANK_DIR="$OUTPUT_ROOT/v3_bank"
V31_DIR="$OUTPUT_ROOT/v3_1_selector"
if [[ -z "$CALIBRATION_BASELINE_DIR" ]]; then
  CALIBRATION_BASELINE_DIR="$V31_DIR/calibration_baseline"
fi
if [[ -z "$DEV_BASELINE_DIR" ]]; then
  DEV_BASELINE_DIR="$V31_DIR/dev_baseline"
fi
DEV_MARGIN_DIR="$V31_DIR/dev_margin"
SELECTOR_ARTIFACT="$V31_DIR/margin_selector_calibration.json"

mkdir -p "$V31_DIR"
run_is_complete() {
  local run_dir="$1"
  [[ -s "$run_dir/results.jsonl" && -s "$run_dir/run_profile.json" && -s "$run_dir/run_report.json" ]] && \
    python -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if value.get("status") == "completed" and value.get("remaining_sample_count") == 0 else 1)' "$run_dir/run_report.json"
}

for REQUIRED in \
  "$PHASE1_DIR/split_manifest.json" \
  "$E0_DIR/memory_records.v2.jsonl" \
  "$E0_DIR/side_kv_manifest.json" \
  "$E0_DIR/e0_final_report.json" \
  "$V3_BANK_DIR/retrieval_key_manifest.json" \
  "$V3_BANK_DIR/v3_offline_report.json" \
  "$RISK_ARTIFACT"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

python scripts/audit_v3_retrieval_geometry.py \
  --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --output "$V31_DIR/retrieval_geometry_audit.json"

if ! run_is_complete "$CALIBRATION_BASELINE_DIR"; then
  bash scripts/experiments/gsm8k/run_v3_system.sh \
    --stage eval \
    --logical-split calibration-val \
    --limit "$CALIBRATION_LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --run-dir "$CALIBRATION_BASELINE_DIR" \
    "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
else
  echo "Reusing calibration baseline: $CALIBRATION_BASELINE_DIR"
fi

python scripts/calibrate_v3_margin_selector.py \
  --results "$CALIBRATION_BASELINE_DIR/results.jsonl" \
  --run-profile "$CALIBRATION_BASELINE_DIR/run_profile.json" \
  --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
  --target-retained-fraction "$TARGET_RETAINED_FRACTION" \
  --output "$SELECTOR_ARTIFACT"

if ! run_is_complete "$DEV_BASELINE_DIR"; then
  bash scripts/experiments/gsm8k/run_v3_system.sh \
    --stage eval \
    --logical-split dev-test \
    --limit "$DEV_LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --run-dir "$DEV_BASELINE_DIR" \
    "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
else
  echo "Reusing dev baseline: $DEV_BASELINE_DIR"
fi

bash scripts/experiments/gsm8k/run_v3_system.sh \
  --stage eval \
  --logical-split dev-test \
  --limit "$DEV_LIMIT" \
  --parity-samples "$PARITY_SAMPLES" \
  --run-dir "$DEV_MARGIN_DIR" \
  --selector-calibration "$SELECTOR_ARTIFACT" \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"

for RUN_DIR in "$DEV_BASELINE_DIR" "$DEV_MARGIN_DIR"; do
  python scripts/analyze_v3_evaluation.py \
    --results "$RUN_DIR/results.jsonl" \
    --run-profile "$RUN_DIR/run_profile.json" \
    --output "$RUN_DIR/analysis_report.json" \
    --markdown-output "$RUN_DIR/analysis_report.md"
done

python scripts/compare_v3_selector_evaluations.py \
  --baseline-results "$DEV_BASELINE_DIR/results.jsonl" \
  --baseline-profile "$DEV_BASELINE_DIR/run_profile.json" \
  --margin-results "$DEV_MARGIN_DIR/results.jsonl" \
  --margin-profile "$DEV_MARGIN_DIR/run_profile.json" \
  --output "$V31_DIR/dev_selector_comparison.json"

echo "V3.1 experiment completed: $V31_DIR"
echo "Key audit: $V31_DIR/retrieval_geometry_audit.md"
echo "Selector calibration: $V31_DIR/margin_selector_calibration.md"
echo "Matched comparison: $V31_DIR/dev_selector_comparison.md"
