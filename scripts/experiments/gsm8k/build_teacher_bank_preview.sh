#!/usr/bin/env bash
# Build a small, inspectable DeepSeek-reflected GSM8K target/reference bank.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_ENV="$REPO_ROOT/scripts/experiments/.server.env"
cd "$REPO_ROOT"

if [[ ! -f "$SERVER_ENV" ]]; then
  echo "Missing $SERVER_ENV. Copy scripts/experiments/server.env.example and fill it in." >&2
  exit 1
fi
source "$SERVER_ENV"

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .server.env}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY must be set in .server.env}"

RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="gsm8k_teacher-reflection_flash_target-reference_preview_${RUN_TAG}"
OUTPUT_DIR="$MEMGEN_OUTPUT_ROOT/banks/gsm8k/$RUN_ID"
OUTPUT_PATH="$OUTPUT_DIR/bank_records.jsonl"

python scripts/build_teacher_bank.py \
  --dataset gsm8k \
  --split train \
  --limit 5 \
  --model "${DEEPSEEK_TEACHER_MODEL:-deepseek-v4-flash}" \
  --thinking "${DEEPSEEK_THINKING:-disabled}" \
  --output "$OUTPUT_PATH"

echo "Preview bank: $OUTPUT_PATH"
echo "Inspect it with: python -m json.tool <(head -n 1 \"$OUTPUT_PATH\")"
