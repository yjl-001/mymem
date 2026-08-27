#!/usr/bin/env bash
# V3.3: audit-derived pre-boundary calibration and one matched dev-test run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DEV_LIMIT=0
PARITY_SAMPLES=8
TARGET_RETAINED_FRACTION=0.5
POOLING_AUDIT_DIR=""
V31_SELECTOR_CALIBRATION=""
V31_DEV_DIR=""
V33_DIR=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dev-limit) DEV_LIMIT="$2"; shift 2 ;;
    --parity-samples) PARITY_SAMPLES="$2"; shift 2 ;;
    --target-retained-fraction) TARGET_RETAINED_FRACTION="$2"; shift 2 ;;
    --pooling-audit-dir) POOLING_AUDIT_DIR="$2"; shift 2 ;;
    --v31-selector-calibration) V31_SELECTOR_CALIBRATION="$2"; shift 2 ;;
    --v31-dev-dir) V31_DEV_DIR="$2"; shift 2 ;;
    --v33-dir) V33_DIR="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [options] PHASE1_DIR E0_DIR RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi
for VALUE in "$DEV_LIMIT" "$PARITY_SAMPLES"; do
  if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
    echo "dev-limit and parity-samples must be non-negative integers" >&2
    exit 2
  fi
done
if ! [[ "$TARGET_RETAINED_FRACTION" =~ ^(0(\.[0-9]+)?|1(\.0+)?)$ ]] || \
   [[ "$TARGET_RETAINED_FRACTION" =~ ^0(\.0+)?$ ]]; then
  echo "target-retained-fraction must be in (0, 1]" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"
V3_BANK_DIR="$OUTPUT_ROOT/v3_bank"
V31_DIR="$OUTPUT_ROOT/v3_1_selector"
if [[ -z "$POOLING_AUDIT_DIR" ]]; then
  POOLING_AUDIT_DIR="$OUTPUT_ROOT/v3_3_pooling_audit"
fi
if [[ -z "$V31_SELECTOR_CALIBRATION" ]]; then
  V31_SELECTOR_CALIBRATION="$V31_DIR/margin_selector_calibration.json"
fi
if [[ -z "$V31_DEV_DIR" ]]; then
  V31_DEV_DIR="$V31_DIR/dev_margin"
fi
if [[ -z "$V33_DIR" ]]; then
  V33_DIR="$OUTPUT_ROOT/v3_3_pre_boundary"
fi

POOLING_AUDIT="$POOLING_AUDIT_DIR/pooling_audit.json"
POOLING_SAMPLES="$POOLING_AUDIT_DIR/pooling_audit_samples.jsonl"
V33_SELECTOR_CALIBRATION="$V33_DIR/pre_boundary_margin_selector_calibration.json"
V33_DEV_DIR="$V33_DIR/dev_pre_boundary_margin"
V31_FALLBACK_DEV_DIR="$V33_DIR/dev_boundary_last_margin_baseline"
COMPARISON="$V33_DIR/dev_query_pooling_comparison.json"
mkdir -p "$V33_DIR"

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
  "$RISK_ARTIFACT" \
  "$POOLING_AUDIT" \
  "$POOLING_SAMPLES" \
  "$V31_SELECTOR_CALIBRATION"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

if [[ -s "$V33_SELECTOR_CALIBRATION" ]]; then
  python -c 'from pathlib import Path; import math,sys; from memgen.experience.phase1 import file_sha256; from memgen.experience.v3 import V3_QUERY_POOLING_PRE_BOUNDARY; from memgen.experience.v3_selector import load_margin_selector_calibration,selector_calibration_query_pooling; value=load_margin_selector_calibration(Path(sys.argv[1])); source=value["source"]; ok=selector_calibration_query_pooling(value)==V3_QUERY_POOLING_PRE_BOUNDARY and source.get("pooling_audit_report_sha256")==file_sha256(Path(sys.argv[2])) and source.get("pooling_sample_traces_sha256")==file_sha256(Path(sys.argv[3])) and source.get("retrieval_key_manifest_sha256")==file_sha256(Path(sys.argv[4])) and math.isclose(float(value["calibration"]["target_retained_fraction"]), float(sys.argv[5]), rel_tol=0.0, abs_tol=1e-12); raise SystemExit(0 if ok else 1)' \
    "$V33_SELECTOR_CALIBRATION" "$POOLING_AUDIT" "$POOLING_SAMPLES" \
    "$V3_BANK_DIR/retrieval_key_manifest.json" "$TARGET_RETAINED_FRACTION"
  echo "Reusing V3.3 selector calibration: $V33_SELECTOR_CALIBRATION"
else
  python scripts/calibrate_v3_pre_boundary_selector.py \
    --pooling-audit "$POOLING_AUDIT" \
    --pooling-samples "$POOLING_SAMPLES" \
    --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
    --target-retained-fraction "$TARGET_RETAINED_FRACTION" \
    --output "$V33_SELECTOR_CALIBRATION"
fi

if run_is_complete "$V31_DEV_DIR"; then
  echo "Reusing V3.1 boundary-last dev run: $V31_DEV_DIR"
else
  V31_DEV_DIR="$V31_FALLBACK_DEV_DIR"
  if ! run_is_complete "$V31_DEV_DIR"; then
    bash scripts/experiments/gsm8k/run_v3_system.sh \
      --stage eval \
      --logical-split dev-test \
      --limit "$DEV_LIMIT" \
      --parity-samples "$PARITY_SAMPLES" \
      --run-dir "$V31_DEV_DIR" \
      --query-pooling last_valid_token \
      --retrieval-embedding-transform none \
      --selector-calibration "$V31_SELECTOR_CALIBRATION" \
      "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
  else
    echo "Reusing fallback V3.1 dev run: $V31_DEV_DIR"
  fi
fi

if ! run_is_complete "$V33_DEV_DIR"; then
  bash scripts/experiments/gsm8k/run_v3_system.sh \
    --stage eval \
    --logical-split dev-test \
    --limit "$DEV_LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --run-dir "$V33_DEV_DIR" \
    --query-pooling last_token_before_trigger_boundary \
    --retrieval-embedding-transform none \
    --selector-calibration "$V33_SELECTOR_CALIBRATION" \
    "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT" "$OUTPUT_ROOT"
else
  echo "Reusing V3.3 pre-boundary dev run: $V33_DEV_DIR"
fi

for RUN_DIR in "$V31_DEV_DIR" "$V33_DEV_DIR"; do
  python scripts/analyze_v3_evaluation.py \
    --results "$RUN_DIR/results.jsonl" \
    --run-profile "$RUN_DIR/run_profile.json" \
    --output "$RUN_DIR/analysis_report.json" \
    --markdown-output "$RUN_DIR/analysis_report.md"
done

python scripts/compare_v3_query_pooling.py \
  --v31-results "$V31_DEV_DIR/results.jsonl" \
  --v31-profile "$V31_DEV_DIR/run_profile.json" \
  --v31-calibration "$V31_SELECTOR_CALIBRATION" \
  --v33-results "$V33_DEV_DIR/results.jsonl" \
  --v33-profile "$V33_DEV_DIR/run_profile.json" \
  --v33-calibration "$V33_SELECTOR_CALIBRATION" \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --output "$COMPARISON"

echo "V3.3 pre-boundary experiment completed: $V33_DIR"
echo "Selector calibration: $V33_DIR/pre_boundary_margin_selector_calibration.md"
echo "V3.3 analysis: $V33_DEV_DIR/analysis_report.md"
echo "Matched comparison: $V33_DIR/dev_query_pooling_comparison.md"
echo "No final-test was run."
