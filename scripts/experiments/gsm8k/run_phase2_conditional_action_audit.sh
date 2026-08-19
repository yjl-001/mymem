#!/usr/bin/env bash
# Read-only H3 feasibility audit. No online generation, injection, or AI calls.
# Usage:
#   bash scripts/experiments/gsm8k/run_phase2_conditional_action_audit.sh \
#     /absolute/path/to/gsm8k_phase1_verified-student-contrast_<tag>
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/phase1-run-directory" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_ENV="$REPO_ROOT/scripts/experiments/.server.env"
PHASE1_DIR="$1"
cd "$REPO_ROOT"

if [[ ! -f "$SERVER_ENV" ]]; then
  echo "Missing $SERVER_ENV. Copy scripts/experiments/server.env.example and fill it in." >&2
  exit 1
fi
if [[ ! -d "$PHASE1_DIR" ]]; then
  echo "Phase 1 run directory does not exist: $PHASE1_DIR" >&2
  exit 1
fi
source "$SERVER_ENV"

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .server.env}"
: "${MEMGEN_DEVICES:?MEMGEN_DEVICES must be set in .server.env}"
: "${MEMGEN_GSM8K_STUDENT_MODEL:?MEMGEN_GSM8K_STUDENT_MODEL must be set in .server.env}"

APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
for REQUIRED_FILE in "$APPROVED_BANK" "$EXPERIENCES"; do
  if [[ ! -s "$REQUIRED_FILE" ]]; then
    echo "Missing or empty Phase 1 artifact: $REQUIRED_FILE" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$MEMGEN_DEVICES"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/experiments/phase2/gsm8k/gsm8k_phase2_conditional_action_audit_${RUN_TAG}"

python scripts/audit_phase2_conditional_actions.py \
  --approved-bank "$APPROVED_BANK" \
  --experiences "$EXPERIENCES" \
  --output-dir "$RUN_DIR" \
  --model "$MEMGEN_GSM8K_STUDENT_MODEL" \
  --model-revision "${MEMGEN_GSM8K_STUDENT_REVISION:-main}" \
  --device cuda \
  --dtype "${MEMGEN_PHASE2_DTYPE:-bfloat16}" \
  --attn-implementation eager \
  --layer "${MEMGEN_PHASE2_RISK_LAYER:-24}" \
  --batch-size "${MEMGEN_PHASE2_COMPILER_BATCH_SIZE:-2}" \
  --sink-token-count "${MEMGEN_PHASE2_SINK_TOKEN_COUNT:-4}" \
  --high-entropy-quantile "${MEMGEN_PHASE2_RISK_HIGH_ENTROPY_QUANTILE:-0.85}" \
  --low-entropy-quantile "${MEMGEN_PHASE2_RISK_LOW_ENTROPY_QUANTILE:-0.50}" \
  --risk-train-fraction "${MEMGEN_PHASE2_RISK_TRAIN_FRACTION:-0.50}" \
  --risk-split-seed "${MEMGEN_PHASE2_RISK_SPLIT_SEED:-42}" \
  --limit "${MEMGEN_PHASE2_H3_AUDIT_LIMIT:-0}"

echo "Phase 2 conditional-action audit: $RUN_DIR"
echo "Report: $RUN_DIR/conditional_action_feasibility_report.json"
