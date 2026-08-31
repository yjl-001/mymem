#!/usr/bin/env bash
# P0-P2: exhaustive failure-only causal matrix and bottleneck attribution.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

V37_RUN_DIR=""
DIAGNOSIS_K="4"
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --v37-run-dir) V37_RUN_DIR="$2"; shift 2 ;;
    --diagnosis-k) DIAGNOSIS_K="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [--v37-run-dir DIR] [--diagnosis-k K] PHASE1_DIR E0_DIR TOKEN_RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
TOKEN_RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"

if [[ -z "$V37_RUN_DIR" ]]; then
  V37_RUN_DIR="$OUTPUT_ROOT/v3_7_cross_problem_causal/dev_offset0_limit64_k4_random4_seed3617"
fi
RUN_NAME="$(basename "$V37_RUN_DIR")_diagnosis_k${DIAGNOSIS_K}"
AUDIT_DIR="$OUTPUT_ROOT/v3_8_failure_full_bank_causal/$RUN_NAME"

SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
E0_FINAL_REPORT="$E0_DIR/e0_final_report.json"
DUAL_KEY_MANIFEST="$OUTPUT_ROOT/v3_5_applicability_selector/dual_key_bank/dual_retrieval_key_manifest.json"
SOURCE_EVIDENCE="$OUTPUT_ROOT/v3_5_dynamic_source_alignment/source_state_evidence.jsonl"
V36_DIR="$OUTPUT_ROOT/v3_6_source_state_keys"
V36_REPORT="$V36_DIR/state_key_report.json"
STATE_KEY_MANIFEST="$V36_DIR/reference_state_key_manifest.json"

for REQUIRED in \
  "$SPLIT_MANIFEST" \
  "$MEMORY_RECORDS" \
  "$SIDE_KV_MANIFEST" \
  "$E0_FINAL_REPORT" \
  "$TOKEN_RISK_ARTIFACT" \
  "$DUAL_KEY_MANIFEST" \
  "$SOURCE_EVIDENCE" \
  "$V36_REPORT" \
  "$STATE_KEY_MANIFEST" \
  "$V37_RUN_DIR/causal_profile.json" \
  "$V37_RUN_DIR/causal_queries.jsonl" \
  "$V37_RUN_DIR/causal_treatments.jsonl" \
  "$V37_RUN_DIR/causal_report.json"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

python scripts/audit_v3_8_failure_full_bank_causal.py \
  --split-manifest "$SPLIT_MANIFEST" \
  --memory-records "$MEMORY_RECORDS" \
  --side-kv-manifest "$SIDE_KV_MANIFEST" \
  --e0-final-report "$E0_FINAL_REPORT" \
  --token-risk-artifact "$TOKEN_RISK_ARTIFACT" \
  --dual-key-manifest "$DUAL_KEY_MANIFEST" \
  --source-alignment-evidence "$SOURCE_EVIDENCE" \
  --v36-report "$V36_REPORT" \
  --state-key-manifest "$STATE_KEY_MANIFEST" \
  --v37-dir "$V37_RUN_DIR" \
  --output-dir "$AUDIT_DIR" \
  --diagnosis-k "$DIAGNOSIS_K"

echo "V3.8 P0-P2 full-bank causal audit complete: $AUDIT_DIR/full_bank_report.json"
