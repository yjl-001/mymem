#!/usr/bin/env bash
# E1C-T: decompose the frozen E1-C text effect into wrapper and payload sources.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E1_SERVER_ENV="${MEMGEN_E1_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e1.server.env}"
cd "$REPO_ROOT"
if [[ ! -f "$E1_SERVER_ENV" ]]; then
  echo "Missing $E1_SERVER_ENV; copy e1.server.env.example first." >&2
  exit 1
fi
source "$E1_SERVER_ENV"
if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 PHASE1_DIR E0_DIR E1B_RUN_DIR E1C_V3_RUN_DIR" >&2
  exit 2
fi
: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .e1.server.env}"
export CUDA_VISIBLE_DEVICES="${MEMGEN_E1_CUDA_VISIBLE_DEVICES:-0}"

PHASE1_DIR="$1"
E0_DIR="$2"
E1B_DIR="$3"
E1C_DIR="$4"
ASSIGNMENT_MANIFEST="$E1B_DIR/assignment_manifest.json"
E1C_RESULTS="$E1C_DIR/evaluation/results.jsonl"
E1C_RUN_REPORT="$E1C_DIR/evaluation/run_report.json"
E1C_SUMMARY="$E1C_DIR/evaluation/e1c_summary.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
for FILE in "$ASSIGNMENT_MANIFEST" "$E1C_RESULTS" "$E1C_RUN_REPORT" "$E1C_SUMMARY" "$MEMORY_RECORDS" "$SPLIT_MANIFEST"; do
  [[ -s "$FILE" ]] || { echo "Missing frozen input: $FILE" >&2; exit 1; }
done
if [[ "$(jq -r '.schema_version' "$E1C_SUMMARY")" != "experience-memory-e1c-summary-v3" ]]; then
  echo "E1C-T requires an E1-C v3 summary." >&2
  exit 3
fi
if [[ "$(jq -r '.component_diagnostic.status' "$E1C_SUMMARY")" != "passed" ]]; then
  echo "E1C-T requires a mechanism-valid E1-C v3 source run." >&2
  exit 3
fi

LOGICAL_SPLIT="$(jq -r '.logical_split' "$ASSIGNMENT_MANIFEST")"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1ct_text-source_${LOGICAL_SPLIT}_${RUN_TAG}"
mkdir -p "$RUN_DIR"

python scripts/evaluate_e1ct_text_source.py \
  --assignment-manifest "$ASSIGNMENT_MANIFEST" \
  --e1c-results "$E1C_RESULTS" \
  --e1c-run-report "$E1C_RUN_REPORT" \
  --e1c-summary "$E1C_SUMMARY" \
  --memory-records "$MEMORY_RECORDS" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$RUN_DIR/evaluation" \
  --device cuda \
  --dtype bfloat16

echo "E1C-T artifacts: $RUN_DIR"
echo "E1C-T summary: $RUN_DIR/evaluation/e1ct_summary.json"
