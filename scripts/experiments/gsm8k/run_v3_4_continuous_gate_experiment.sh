#!/usr/bin/env bash
# V3.4 online stage: one raw smoke/calibration, one matched dev, optional final.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

SMOKE_LIMIT=64
DEV_LIMIT=0
PARITY_SAMPLES=8
MINIMUM_FIRST_ATTEMPTS=8
TARGET_RETAINED_FRACTION=0.5
RUN_FINAL=false
V31_DEV_DIR=""
V31_SELECTOR_CALIBRATION=""
V34_DIR=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --smoke-limit) SMOKE_LIMIT="$2"; shift 2 ;;
    --dev-limit) DEV_LIMIT="$2"; shift 2 ;;
    --parity-samples) PARITY_SAMPLES="$2"; shift 2 ;;
    --minimum-first-attempts) MINIMUM_FIRST_ATTEMPTS="$2"; shift 2 ;;
    --target-retained-fraction) TARGET_RETAINED_FRACTION="$2"; shift 2 ;;
    --v31-dev-dir) V31_DEV_DIR="$2"; shift 2 ;;
    --v31-selector-calibration) V31_SELECTOR_CALIBRATION="$2"; shift 2 ;;
    --v34-dir) V34_DIR="$2"; shift 2 ;;
    --run-final) RUN_FINAL=true; shift ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [options] PHASE1_DIR E0_DIR TOKEN_RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi
for VALUE in "$SMOKE_LIMIT" "$DEV_LIMIT" "$PARITY_SAMPLES" "$MINIMUM_FIRST_ATTEMPTS"; do
  if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
    echo "Limits must be non-negative integers" >&2
    exit 2
  fi
done
if [[ "$SMOKE_LIMIT" -le 0 || "$MINIMUM_FIRST_ATTEMPTS" -le 0 ]]; then
  echo "smoke-limit and minimum-first-attempts must be positive" >&2
  exit 2
fi
if ! [[ "$TARGET_RETAINED_FRACTION" =~ ^(0\.[0-9]+|1(\.0+)?)$ ]]; then
  echo "target-retained-fraction must be in (0, 1]" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
TOKEN_RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"
V3_BANK_DIR="$OUTPUT_ROOT/v3_bank"
if [[ -z "$V31_DEV_DIR" ]]; then
  V31_DEV_DIR="$OUTPUT_ROOT/v3_1_selector/dev_margin"
fi
if [[ -z "$V31_SELECTOR_CALIBRATION" ]]; then
  V31_SELECTOR_CALIBRATION="$OUTPUT_ROOT/v3_1_selector/margin_selector_calibration.json"
fi
if [[ -z "$V34_DIR" ]]; then
  V34_DIR="$OUTPUT_ROOT/v3_4_continuous_gate"
fi

RAW_SMOKE_DIR="$V34_DIR/smoke_current_token_raw"
GEOMETRY_AUDIT="$V34_DIR/current_token_geometry_audit.json"
SELECTOR_CALIBRATION="$V34_DIR/current_token_margin_selector.json"
DEV_DIR="$V34_DIR/dev_current_token_margin"
COMPARISON="$V34_DIR/dev_v34_minus_v31.json"
QUALIFICATION="$V34_DIR/dev_qualification.json"
FINAL_DIR="$V34_DIR/final_test_1319"
mkdir -p "$V34_DIR"

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
  "$TOKEN_RISK_ARTIFACT" \
  "$V31_DEV_DIR/results.jsonl" \
  "$V31_DEV_DIR/run_profile.json" \
  "$V31_SELECTOR_CALIBRATION"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

if ! run_is_complete "$RAW_SMOKE_DIR"; then
  bash scripts/experiments/gsm8k/run_v3_system.sh \
    --stage eval \
    --system-version v3.4 \
    --logical-split calibration-val \
    --limit "$SMOKE_LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --run-dir "$RAW_SMOKE_DIR" \
    "$PHASE1_DIR" "$E0_DIR" "$TOKEN_RISK_ARTIFACT" "$OUTPUT_ROOT"
else
  echo "Reusing V3.4 raw smoke: $RAW_SMOKE_DIR"
fi

python scripts/audit_v3_4_current_token_geometry.py \
  --results "$RAW_SMOKE_DIR/results.jsonl" \
  --run-profile "$RAW_SMOKE_DIR/run_profile.json" \
  --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
  --minimum-first-attempts "$MINIMUM_FIRST_ATTEMPTS" \
  --output "$GEOMETRY_AUDIT"

if [[ -s "$SELECTOR_CALIBRATION" ]]; then
  python -c 'import math,sys; from pathlib import Path; from memgen.experience.phase1 import file_sha256; from memgen.experience.v3_selector import load_margin_selector_calibration; value=load_margin_selector_calibration(Path(sys.argv[1])); source=value["source"]; calibration=value["calibration"]; ok=source.get("system_version")=="v3.4" and source.get("results_file_sha256")==file_sha256(Path(sys.argv[2])) and source.get("run_profile_file_sha256")==file_sha256(Path(sys.argv[3])) and source.get("retrieval_key_manifest_sha256")==file_sha256(Path(sys.argv[4])) and source.get("risk_artifact_sha256")==file_sha256(Path(sys.argv[5])) and math.isclose(float(calibration["target_retained_fraction"]),float(sys.argv[6]),rel_tol=0.0,abs_tol=1e-12); raise SystemExit(0 if ok else 1)' \
    "$SELECTOR_CALIBRATION" \
    "$RAW_SMOKE_DIR/results.jsonl" \
    "$RAW_SMOKE_DIR/run_profile.json" \
    "$V3_BANK_DIR/retrieval_key_manifest.json" \
    "$TOKEN_RISK_ARTIFACT" \
    "$TARGET_RETAINED_FRACTION"
  echo "Reusing V3.4 selector calibration: $SELECTOR_CALIBRATION"
else
  python scripts/calibrate_v3_margin_selector.py \
    --results "$RAW_SMOKE_DIR/results.jsonl" \
    --run-profile "$RAW_SMOKE_DIR/run_profile.json" \
    --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
    --target-retained-fraction "$TARGET_RETAINED_FRACTION" \
    --minimum-triggered-samples "$MINIMUM_FIRST_ATTEMPTS" \
    --output "$SELECTOR_CALIBRATION"
fi

if ! run_is_complete "$DEV_DIR"; then
  bash scripts/experiments/gsm8k/run_v3_system.sh \
    --stage eval \
    --system-version v3.4 \
    --logical-split dev-test \
    --limit "$DEV_LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --run-dir "$DEV_DIR" \
    --selector-calibration "$SELECTOR_CALIBRATION" \
    "$PHASE1_DIR" "$E0_DIR" "$TOKEN_RISK_ARTIFACT" "$OUTPUT_ROOT"
else
  echo "Reusing V3.4 matched dev: $DEV_DIR"
fi

python scripts/analyze_v3_evaluation.py \
  --results "$DEV_DIR/results.jsonl" \
  --run-profile "$DEV_DIR/run_profile.json" \
  --output "$DEV_DIR/analysis_report.json" \
  --markdown-output "$DEV_DIR/analysis_report.md"

python scripts/compare_v3_4_continuous_gate.py \
  --v31-results "$V31_DEV_DIR/results.jsonl" \
  --v31-profile "$V31_DEV_DIR/run_profile.json" \
  --v31-calibration "$V31_SELECTOR_CALIBRATION" \
  --v34-results "$DEV_DIR/results.jsonl" \
  --v34-profile "$DEV_DIR/run_profile.json" \
  --v34-calibration "$SELECTOR_CALIBRATION" \
  --output "$COMPARISON"

python scripts/qualify_v3_4_dev.py \
  --comparison "$COMPARISON" \
  --output "$QUALIFICATION"

if [[ "$RUN_FINAL" == true ]]; then
  QUALIFIED="$(python -c 'import json,sys; print("true" if json.load(open(sys.argv[1], encoding="utf-8")).get("qualified_for_final_test") is True else "false")' "$QUALIFICATION")"
  if [[ "$QUALIFIED" != true ]]; then
    echo "V3.4 dev did not qualify; final-test is blocked." >&2
    exit 1
  fi
  if ! run_is_complete "$FINAL_DIR"; then
    bash scripts/experiments/gsm8k/run_v3_system.sh \
      --stage eval \
      --system-version v3.4 \
      --logical-split final-test \
      --limit 0 \
      --parity-samples "$PARITY_SAMPLES" \
      --run-dir "$FINAL_DIR" \
      --selector-calibration "$SELECTOR_CALIBRATION" \
      "$PHASE1_DIR" "$E0_DIR" "$TOKEN_RISK_ARTIFACT" "$OUTPUT_ROOT"
  fi
  python scripts/analyze_v3_evaluation.py \
    --results "$FINAL_DIR/results.jsonl" \
    --run-profile "$FINAL_DIR/run_profile.json" \
    --output "$FINAL_DIR/analysis_report.json" \
    --markdown-output "$FINAL_DIR/analysis_report.md"
fi

echo "V3.4 experiment completed: $V34_DIR"
echo "Geometry audit: $V34_DIR/current_token_geometry_audit.md"
echo "Selector calibration: $V34_DIR/current_token_margin_selector.md"
echo "Matched dev comparison: $V34_DIR/dev_v34_minus_v31.md"
echo "Dev qualification: $V34_DIR/dev_qualification.md"
if [[ "$RUN_FINAL" == true ]]; then
  echo "Final analysis: $FINAL_DIR/analysis_report.md"
else
  echo "Final-test was not requested; rerun with --run-final after inspecting dev."
fi
