#!/usr/bin/env bash
# E1C-S: one-shot +log(10) persistent side-KV strength diagnostic.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E1_SERVER_ENV="${MEMGEN_E1_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e1.server.env}"
cd "$REPO_ROOT"
if [[ ! -f "$E1_SERVER_ENV" ]]; then
  echo "Missing $E1_SERVER_ENV; copy e1.server.env.example first." >&2
  exit 1
fi
source "$E1_SERVER_ENV"
if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 PHASE1_DIR E0_DIR E1B_RUN_DIR E1C_V3_RUN_DIR E1CT_RUN_DIR" >&2
  exit 2
fi
: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .e1.server.env}"
export CUDA_VISIBLE_DEVICES="${MEMGEN_E1_CUDA_VISIBLE_DEVICES:-0}"

PHASE1_DIR="$1"
E0_DIR="$2"
E1B_DIR="$3"
E1C_DIR="$4"
E1CT_DIR="$5"
ASSIGNMENT_MANIFEST="$E1B_DIR/assignment_manifest.json"
E1C_RESULTS="$E1C_DIR/evaluation/results.jsonl"
E1C_RUN_REPORT="$E1C_DIR/evaluation/run_report.json"
E1C_SUMMARY="$E1C_DIR/evaluation/e1c_summary.json"
E1CT_RESULTS="$E1CT_DIR/evaluation/results.jsonl"
E1CT_RUN_REPORT="$E1CT_DIR/evaluation/run_report.json"
E1CT_SUMMARY="$E1CT_DIR/evaluation/e1ct_summary.json"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
for FILE in "$ASSIGNMENT_MANIFEST" "$E1C_RESULTS" "$E1C_RUN_REPORT" "$E1C_SUMMARY" "$E1CT_RESULTS" "$E1CT_RUN_REPORT" "$E1CT_SUMMARY" "$SIDE_KV_MANIFEST" "$SPLIT_MANIFEST"; do
  [[ -s "$FILE" ]] || { echo "Missing frozen input: $FILE" >&2; exit 1; }
done
if [[ "$(jq -r '.component_diagnostic.fixed_strength_test_allowed' "$E1CT_SUMMARY")" != "true" ]]; then
  echo "E1C-S stopped because E1C-T did not authorize the fixed-strength test." >&2
  exit 3
fi
if [[ "$(jq -r '.decision.next_step' "$E1CT_SUMMARY")" != "e1cs_fixed_log10_memory_odds_test" ]]; then
  echo "E1C-S stopped because the E1C-T decision route differs." >&2
  exit 3
fi

LOGICAL_SPLIT="$(jq -r '.logical_split' "$ASSIGNMENT_MANIFEST")"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1cs_fixed-strength_${LOGICAL_SPLIT}_${RUN_TAG}"
mkdir -p "$RUN_DIR"

python scripts/evaluate_e1cs_fixed_strength.py \
  --assignment-manifest "$ASSIGNMENT_MANIFEST" \
  --e1c-results "$E1C_RESULTS" \
  --e1c-run-report "$E1C_RUN_REPORT" \
  --e1c-summary "$E1C_SUMMARY" \
  --e1ct-results "$E1CT_RESULTS" \
  --e1ct-run-report "$E1CT_RUN_REPORT" \
  --e1ct-summary "$E1CT_SUMMARY" \
  --side-kv-manifest "$SIDE_KV_MANIFEST" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$RUN_DIR/evaluation" \
  --device cuda \
  --dtype bfloat16

echo "E1C-S artifacts: $RUN_DIR"
echo "E1C-S summary: $RUN_DIR/evaluation/e1cs_summary.json"
