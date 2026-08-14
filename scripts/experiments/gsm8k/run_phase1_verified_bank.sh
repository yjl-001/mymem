#!/usr/bin/env bash
# Phase 1: collect verifier-backed student rollouts and construct an auditable bank.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_ENV="$REPO_ROOT/scripts/experiments/.server.env"
cd "$REPO_ROOT"

# Preserve explicit one-off CLI overrides; .server.env supplies defaults for
# normal runs but must not turn a bounded retry into an accidental full build.
CALLER_RUN_TAG_SET="${MEMGEN_RUN_TAG+x}"
CALLER_RUN_TAG="${MEMGEN_RUN_TAG:-}"
CALLER_TEACHER_LIMIT_SET="${MEMGEN_PHASE1_TEACHER_LIMIT+x}"
CALLER_TEACHER_LIMIT="${MEMGEN_PHASE1_TEACHER_LIMIT:-}"

if [[ ! -f "$SERVER_ENV" ]]; then
  echo "Missing $SERVER_ENV. Copy scripts/experiments/server.env.example and fill it in." >&2
  exit 1
fi
source "$SERVER_ENV"

if [[ -n "$CALLER_RUN_TAG_SET" ]]; then
  export MEMGEN_RUN_TAG="$CALLER_RUN_TAG"
fi
if [[ -n "$CALLER_TEACHER_LIMIT_SET" ]]; then
  export MEMGEN_PHASE1_TEACHER_LIMIT="$CALLER_TEACHER_LIMIT"
fi

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .server.env}"
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY must be set in .server.env}"

export CUDA_VISIBLE_DEVICES="${MEMGEN_DEVICES:-0}"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="gsm8k_phase1_verified-student-contrast_${RUN_TAG}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/banks/gsm8k/$RUN_ID"

STUDENT_MODEL="${MEMGEN_GSM8K_STUDENT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
STUDENT_REVISION="${MEMGEN_GSM8K_STUDENT_REVISION:-main}"
DATASET_REVISION="${MEMGEN_GSM8K_DATASET_REVISION:-main}"
BANK_SOURCE_SIZE="${MEMGEN_GSM8K_BANK_SOURCE_SIZE:-6000}"
CALIBRATION_VAL_SIZE="${MEMGEN_GSM8K_CALIBRATION_VAL_SIZE:-1000}"
ROLLOUT_SAMPLE_LIMIT="${MEMGEN_PHASE1_ROLLOUT_SAMPLE_LIMIT:-0}"
ROLLOUTS_PER_SAMPLE="${MEMGEN_PHASE1_ROLLOUTS_PER_SAMPLE:-8}"
ROLLOUT_BATCH_SIZE="${MEMGEN_PHASE1_ROLLOUT_BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MEMGEN_PHASE1_MAX_NEW_TOKENS:-768}"
TEMPERATURE="${MEMGEN_PHASE1_TEMPERATURE:-0.8}"
TOP_P="${MEMGEN_PHASE1_TOP_P:-0.95}"
SEED="${MEMGEN_PHASE1_SEED:-42}"
TEACHER_LIMIT="${MEMGEN_PHASE1_TEACHER_LIMIT:-0}"
ATTN_IMPLEMENTATION="${MEMGEN_PHASE1_ATTN_IMPLEMENTATION:-flash_attention_2}"
TEACHER_PROXY_RETRIES="${MEMGEN_TEACHER_PROXY_RETRIES:-20}"
TEACHER_PROXY_RETRY_INITIAL_SECONDS="${MEMGEN_TEACHER_PROXY_RETRY_INITIAL_SECONDS:-30}"
TEACHER_PROXY_RETRY_MAX_SECONDS="${MEMGEN_TEACHER_PROXY_RETRY_MAX_SECONDS:-300}"
TEACHER_CONNECT_TIMEOUT_SECONDS="${MEMGEN_TEACHER_CONNECT_TIMEOUT_SECONDS:-30}"
TEACHER_READ_TIMEOUT_SECONDS="${MEMGEN_TEACHER_READ_TIMEOUT_SECONDS:-180}"
TEACHER_MODEL="${DEEPSEEK_TEACHER_MODEL:-deepseek-v4-flash}"
REVIEW_MODEL="${DEEPSEEK_REVIEW_MODEL:-deepseek-v4-pro}"

mkdir -p "$RUN_DIR"

SPLIT_MANIFEST="$RUN_DIR/split_manifest.json"
ROLLOUTS="$RUN_DIR/student_rollouts.jsonl"
ROLLOUT_SUMMARY="$RUN_DIR/rollout_summary.json"
EXPERIENCES="$RUN_DIR/verified_experiences.jsonl"
EXPERIENCE_REPORT="$RUN_DIR/experience_build_report.json"
TEACHER_RECORDS="$RUN_DIR/teacher_reflections.jsonl"
AI_REVIEW_RECORDS="$RUN_DIR/ai_review_records.jsonl"
AI_APPROVED_BANK="$RUN_DIR/ai_approved_bank_records.jsonl"
AI_REJECTED_BANK="$RUN_DIR/ai_rejected_bank_records.jsonl"
DEFERRED_BANK="$RUN_DIR/deferred_bank_records.jsonl"
QUARANTINED_BANK="$RUN_DIR/quarantined_bank_records.jsonl"
AI_REVIEW_REPORT="$RUN_DIR/ai_review_report.json"

if [[ ! -s "$SPLIT_MANIFEST" ]]; then
  python scripts/build_gsm8k_split_manifest.py \
    --output "$SPLIT_MANIFEST" \
    --bank-source-size "$BANK_SOURCE_SIZE" \
    --calibration-val-size "$CALIBRATION_VAL_SIZE" \
    --seed "$SEED" \
    --dataset-revision "$DATASET_REVISION"
else
  echo "[phase1] reuse completed split manifest: $SPLIT_MANIFEST"
fi

if [[ ! -s "$ROLLOUTS" || ! -s "$ROLLOUT_SUMMARY" ]]; then
  python scripts/collect_gsm8k_rollouts.py \
    --split-manifest "$SPLIT_MANIFEST" \
    --output "$ROLLOUTS" \
    --summary-output "$ROLLOUT_SUMMARY" \
    --model "$STUDENT_MODEL" \
    --model-revision "$STUDENT_REVISION" \
    --dataset-revision "$DATASET_REVISION" \
    --limit "$ROLLOUT_SAMPLE_LIMIT" \
    --rollouts-per-sample "$ROLLOUTS_PER_SAMPLE" \
    --batch-size "$ROLLOUT_BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --seed "$SEED" \
    --attn-implementation "$ATTN_IMPLEMENTATION"
else
  echo "[phase1] reuse completed rollout artifact: $ROLLOUTS"
fi

# This step is deterministic and cheap. Always rebuild it so legacy v1 rollout
# verifier records are re-diagnosed under the current strict+diagnostic policy.
python scripts/build_verified_experiences.py \
  --rollouts "$ROLLOUTS" \
  --output "$EXPERIENCES" \
  --report-output "$EXPERIENCE_REPORT" \
  --max-pairs-per-sample 1

PAIR_COUNT="$(wc -l < "$EXPERIENCES" | tr -d ' ')"
if [[ "$TEACHER_LIMIT" -eq 0 || "$TEACHER_LIMIT" -gt "$PAIR_COUNT" ]]; then
  TEACHER_LIMIT="$PAIR_COUNT"
fi

python scripts/build_teacher_bank.py \
  --input-jsonl "$EXPERIENCES" \
  --limit "$TEACHER_LIMIT" \
  --model "$TEACHER_MODEL" \
  --base-url "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}" \
  --thinking "${DEEPSEEK_THINKING:-disabled}" \
  --proxy-retries "$TEACHER_PROXY_RETRIES" \
  --proxy-retry-initial-seconds "$TEACHER_PROXY_RETRY_INITIAL_SECONDS" \
  --proxy-retry-max-seconds "$TEACHER_PROXY_RETRY_MAX_SECONDS" \
  --connect-timeout-seconds "$TEACHER_CONNECT_TIMEOUT_SECONDS" \
  --read-timeout-seconds "$TEACHER_READ_TIMEOUT_SECONDS" \
  --output "$TEACHER_RECORDS" \
  --resume

python scripts/review_experience_bank.py \
  --experiences "$EXPERIENCES" \
  --teacher-records "$TEACHER_RECORDS" \
  --review-records-output "$AI_REVIEW_RECORDS" \
  --approved-output "$AI_APPROVED_BANK" \
  --rejected-output "$AI_REJECTED_BANK" \
  --deferred-output "$DEFERRED_BANK" \
  --quarantined-output "$QUARANTINED_BANK" \
  --report-output "$AI_REVIEW_REPORT" \
  --model "$REVIEW_MODEL" \
  --base-url "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}" \
  --thinking "${DEEPSEEK_REVIEW_THINKING:-disabled}" \
  --proxy-retries "$TEACHER_PROXY_RETRIES" \
  --proxy-retry-initial-seconds "$TEACHER_PROXY_RETRY_INITIAL_SECONDS" \
  --proxy-retry-max-seconds "$TEACHER_PROXY_RETRY_MAX_SECONDS" \
  --connect-timeout-seconds "$TEACHER_CONNECT_TIMEOUT_SECONDS" \
  --read-timeout-seconds "$TEACHER_READ_TIMEOUT_SECONDS" \
  --resume

echo "Phase 1 artifacts: $RUN_DIR"
echo "AI-approved bank: $AI_APPROVED_BANK"
echo "Deferred records: $DEFERRED_BANK"
echo "Integrity quarantine: $QUARANTINED_BANK"
echo "AI review report: $AI_REVIEW_REPORT"
