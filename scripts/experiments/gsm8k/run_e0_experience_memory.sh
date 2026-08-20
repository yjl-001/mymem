#!/usr/bin/env bash
# E0: compile reviewed experience text into a leak-audited BM25 + side-KV bank.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_ENV="$REPO_ROOT/scripts/experiments/.server.env"
cd "$REPO_ROOT"

if [[ ! -f "$SERVER_ENV" ]]; then
  echo "Missing $SERVER_ENV. Copy scripts/experiments/server.env.example and fill it in." >&2
  exit 1
fi
source "$SERVER_ENV"

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 /absolute/path/to/phase1-run-dir [/absolute/path/to/audit-cases.jsonl]" >&2
  exit 1
fi

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .server.env}"
: "${MEMGEN_E0_MAX_PAYLOAD_TOKENS:?Freeze MEMGEN_E0_MAX_PAYLOAD_TOKENS before a formal E0 run}"

PHASE1_DIR="$1"
AUDIT_CASES="${2:-}"
APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
VERIFIED_EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
if [[ ! -s "$APPROVED_BANK" || ! -s "$VERIFIED_EXPERIENCES" || ! -s "$SPLIT_MANIFEST" ]]; then
  echo "Phase-1 approved bank, verified experiences, or split manifest is missing under $PHASE1_DIR" >&2
  exit 1
fi

MODEL="${MEMGEN_GSM8K_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
MODEL_REVISION="${MEMGEN_GSM8K_STUDENT_REVISION:-main}"
DTYPE="${MEMGEN_E0_DTYPE:-bfloat16}"
DEVICE="${MEMGEN_E0_DEVICE:-cuda}"
TEXT_ONLY="${MEMGEN_E0_TEXT_ONLY:-0}"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="gsm8k_e0_experience-memory_layer24_budget${MEMGEN_E0_MAX_PAYLOAD_TOKENS}_${RUN_TAG}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e0/gsm8k/$RUN_ID"
mkdir -p "$RUN_DIR"

COMPILE_ARGS=(
  --approved-bank "$APPROVED_BANK"
  --verified-experiences "$VERIFIED_EXPERIENCES"
  --output-dir "$RUN_DIR"
  --model "$MODEL"
  --model-revision "$MODEL_REVISION"
  --max-payload-tokens "$MEMGEN_E0_MAX_PAYLOAD_TOKENS"
  --layer 24
  --dtype "$DTYPE"
  --device "$DEVICE"
)
if [[ "$TEXT_ONLY" == "1" ]]; then
  COMPILE_ARGS+=(--text-only)
fi
python scripts/compile_experience_memory_bank.py "${COMPILE_ARGS[@]}"

if [[ "$TEXT_ONLY" != "1" && -z "$AUDIT_CASES" ]]; then
  AUDIT_CASES="$RUN_DIR/calibration_audit_cases.jsonl"
  python scripts/build_side_kv_audit_cases.py \
    --split-manifest "$SPLIT_MANIFEST" \
    --memory-records "$RUN_DIR/memory_records.v1.jsonl" \
    --output "$AUDIT_CASES" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --dataset-revision "${MEMGEN_GSM8K_DATASET_REVISION:-main}" \
    --logical-split calibration-val \
    --case-count "${MEMGEN_E0_AUDIT_CASE_COUNT:-8}" \
    --max-new-tokens "${MEMGEN_E0_AUDIT_MAX_NEW_TOKENS:-128}" \
    --dtype "$DTYPE" \
    --device "$DEVICE"
fi

if [[ "$TEXT_ONLY" != "1" ]]; then
  if [[ ! -s "$AUDIT_CASES" ]]; then
    echo "Side-KV audit cases file is missing or empty: $AUDIT_CASES" >&2
    exit 1
  fi
  python scripts/audit_side_kv_mechanism.py \
    --side-kv-manifest "$RUN_DIR/side_kv_manifest.json" \
    --cases "$AUDIT_CASES" \
    --output-dir "$RUN_DIR" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --layer 24 \
    --dtype "$DTYPE" \
    --device "$DEVICE"
fi

echo "E0 artifacts: $RUN_DIR"
