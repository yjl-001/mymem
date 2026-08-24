#!/usr/bin/env bash
# E1-C: persistent side-KV using the exact E1-B matched/shuffled assignments.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E1_SERVER_ENV="${MEMGEN_E1_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e1.server.env}"
cd "$REPO_ROOT"
if [[ ! -f "$E1_SERVER_ENV" ]]; then
  echo "Missing $E1_SERVER_ENV; copy e1.server.env.example first." >&2
  exit 1
fi
source "$E1_SERVER_ENV"
if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 PHASE1_DIR E0_DIR E1B_RUN_DIR" >&2
  exit 2
fi
: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .e1.server.env}"
export CUDA_VISIBLE_DEVICES="${MEMGEN_E1_CUDA_VISIBLE_DEVICES:-0}"

PHASE1_DIR="$1"
E0_DIR="$2"
E1B_DIR="$3"
ASSIGNMENT_MANIFEST="$E1B_DIR/assignment_manifest.json"
E1B_RESULTS="$E1B_DIR/evaluation/results.jsonl"
E1B_RUN_REPORT="$E1B_DIR/evaluation/run_report.json"
E1B_SUMMARY="$E1B_DIR/evaluation/e1b_summary.json"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
for FILE in "$ASSIGNMENT_MANIFEST" "$E1B_RESULTS" "$E1B_RUN_REPORT" "$E1B_SUMMARY" "$MEMORY_RECORDS" "$SIDE_KV_MANIFEST" "$SPLIT_MANIFEST"; do
  [[ -s "$FILE" ]] || { echo "Missing frozen input: $FILE" >&2; exit 1; }
done
if [[ "$(jq -r '.component_diagnostic.e1c_component_diagnostic_allowed' "$E1B_SUMMARY")" != "true" ]]; then
  echo "E1-C stopped because the E1-B component handoff is not valid." >&2
  exit 3
fi
if [[ "$(jq -r '.formal_e1b_passed' "$E1B_SUMMARY")" != "true" ]]; then
  echo "E1-B did not formally pass task accuracy; running E1-C in component-diagnostic mode." >&2
fi

LOGICAL_SPLIT="$(jq -r '.logical_split' "$ASSIGNMENT_MANIFEST")"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1c_side-kv_${LOGICAL_SPLIT}_${RUN_TAG}"
mkdir -p "$RUN_DIR"

python scripts/evaluate_e1c_side_kv_channel.py \
  --assignment-manifest "$ASSIGNMENT_MANIFEST" \
  --e1b-results "$E1B_RESULTS" \
  --e1b-run-report "$E1B_RUN_REPORT" \
  --e1b-summary "$E1B_SUMMARY" \
  --memory-records "$MEMORY_RECORDS" \
  --side-kv-manifest "$SIDE_KV_MANIFEST" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$RUN_DIR/evaluation" \
  --device cuda \
  --dtype bfloat16

echo "E1-C artifacts: $RUN_DIR"
echo "E1-C summary: $RUN_DIR/evaluation/e1c_summary.json"
