#!/usr/bin/env bash
# Compile the frozen layer-24 entropy-risk gate artifact. No residual injection.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 PHASE1_DIR" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E0_SERVER_ENV="${MEMGEN_E0_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e0.server.env}"
PHASE1_DIR="$1"
cd "$REPO_ROOT"

if [[ ! -f "$E0_SERVER_ENV" ]]; then
  echo "Missing $E0_SERVER_ENV" >&2
  exit 1
fi
source "$E0_SERVER_ENV"

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set}"

APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
for REQUIRED_FILE in "$APPROVED_BANK" "$EXPERIENCES"; do
  if [[ ! -s "$REQUIRED_FILE" ]]; then
    echo "Missing or empty Phase 1 artifact: $REQUIRED_FILE" >&2
    exit 1
  fi
done

PHASE1_MODEL_METADATA="$(
  python - "$APPROVED_BANK" <<'PY'
import json
from pathlib import Path
import sys

with Path(sys.argv[1]).open("r", encoding="utf-8") as handle:
    record = next((json.loads(line) for line in handle if line.strip()), None)
student = record.get("student") if isinstance(record, dict) else None
if not isinstance(student, dict):
    raise SystemExit("Phase-1 approved bank has no frozen student metadata")
values = (
    student.get("model_name"),
    student.get("model_revision"),
    student.get("tokenizer_revision"),
)
if any(not isinstance(value, str) or not value for value in values):
    raise SystemExit("Phase-1 student metadata is incomplete")
print("\t".join(values))
PY
)"
IFS=$'\t' read -r MODEL MODEL_REVISION TOKENIZER_REVISION <<< "$PHASE1_MODEL_METADATA"

export CUDA_VISIBLE_DEVICES="${MEMGEN_E0_CUDA_VISIBLE_DEVICES:-0}"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/risk/gsm8k/gsm8k_entropy-risk-layer24_sdpa_${RUN_TAG}"
mkdir -p "$RUN_DIR"

python scripts/compile_entropy_risk_gate.py \
  --approved-bank "$APPROVED_BANK" \
  --experiences "$EXPERIENCES" \
  --output-dir "$RUN_DIR" \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --device cuda \
  --dtype bfloat16 \
  --attn-implementation sdpa \
  --batch-size 1 \
  --layer 24 \
  --sink-token-count 4 \
  --high-entropy-quantile 0.85 \
  --low-entropy-quantile 0.50 \
  --risk-train-fraction 0.50 \
  --risk-split-seed 42 \
  --min-events-per-label 50 \
  --min-heldout-roc-auc 0.60

ARTIFACT="$RUN_DIR/entropy-risk-gate-answer_correctness.pt"
echo "Risk artifact: $ARTIFACT"
