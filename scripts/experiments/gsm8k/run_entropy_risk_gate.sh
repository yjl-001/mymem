#!/usr/bin/env bash
# Compile the frozen layer-24 entropy-risk gate artifact. No residual injection.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 PHASE1_DIR" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_ENV="$REPO_ROOT/scripts/experiments/.server.env"
PHASE1_DIR="$1"
cd "$REPO_ROOT"

if [[ ! -f "$SERVER_ENV" ]]; then
  echo "Missing $SERVER_ENV" >&2
  exit 1
fi
source "$SERVER_ENV"

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set}"
: "${MEMGEN_DEVICES:?MEMGEN_DEVICES must be set}"
: "${MEMGEN_GSM8K_STUDENT_MODEL:?MEMGEN_GSM8K_STUDENT_MODEL must be set}"

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
RUN_DIR="$MEMGEN_OUTPUT_ROOT/risk/gsm8k/gsm8k_entropy-risk-layer24_${RUN_TAG}"
mkdir -p "$RUN_DIR"

python scripts/compile_entropy_risk_gate.py \
  --approved-bank "$APPROVED_BANK" \
  --experiences "$EXPERIENCES" \
  --output-dir "$RUN_DIR" \
  --model "$MEMGEN_GSM8K_STUDENT_MODEL" \
  --model-revision "${MEMGEN_GSM8K_STUDENT_REVISION:-main}" \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation eager \
  --layer 24 \
  --batch-size 2 \
  --sink-token-count 4 \
  --high-entropy-quantile 0.85 \
  --low-entropy-quantile 0.50 \
  --risk-train-fraction 0.50 \
  --risk-split-seed 42 \
  --min-events-per-label 50 \
  --min-heldout-roc-auc 0.60

ARTIFACT="$RUN_DIR/entropy-risk-gate-answer_correctness.pt"
echo "Risk artifact: $ARTIFACT"
