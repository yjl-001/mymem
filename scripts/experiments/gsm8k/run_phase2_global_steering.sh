#!/usr/bin/env bash
# Phase 2: compile audited global vectors and run calibration-val controls.
# Usage:
#   bash scripts/experiments/gsm8k/run_phase2_global_steering.sh \
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
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY must be set in .server.env}"

APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
for REQUIRED_FILE in "$APPROVED_BANK" "$EXPERIENCES" "$SPLIT_MANIFEST"; do
  if [[ ! -s "$REQUIRED_FILE" ]]; then
    echo "Missing or empty Phase 1 artifact: $REQUIRED_FILE" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$MEMGEN_DEVICES"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="gsm8k_phase2_global_steering_${RUN_TAG}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/experiments/phase2/gsm8k/$RUN_ID"
VECTOR_DIR="$RUN_DIR/vector_artifacts"
CALIBRATION_DIR="$RUN_DIR/calibration"
EVIDENCE_ANCHORS="$RUN_DIR/evidence_anchors.jsonl"
EVIDENCE_ANCHOR_REPORT="$RUN_DIR/evidence_anchor_report.json"
mkdir -p "$RUN_DIR"

STUDENT_MODEL="$MEMGEN_GSM8K_STUDENT_MODEL"
STUDENT_REVISION="${MEMGEN_GSM8K_STUDENT_REVISION:-main}"
DATASET_REVISION="${MEMGEN_GSM8K_DATASET_REVISION:-main}"
COMPILER_ATTN="${MEMGEN_PHASE2_COMPILER_ATTN_IMPLEMENTATION:-flash_attention_2}"
EVAL_ATTN="${MEMGEN_PHASE2_EVAL_ATTN_IMPLEMENTATION:-eager}"

python scripts/anchor_phase2_evidence.py \
  --approved-bank "$APPROVED_BANK" \
  --experiences "$EXPERIENCES" \
  --output "$EVIDENCE_ANCHORS" \
  --report-output "$EVIDENCE_ANCHOR_REPORT" \
  --model "${DEEPSEEK_REVIEW_MODEL:-deepseek-v4-pro}" \
  --base-url "${DEEPSEEK_BASE_URL:-https://api.deepseek.com}" \
  --thinking "${DEEPSEEK_REVIEW_THINKING:-disabled}" \
  --proxy-retries "${MEMGEN_TEACHER_PROXY_RETRIES:-20}" \
  --proxy-retry-initial-seconds "${MEMGEN_TEACHER_PROXY_RETRY_INITIAL_SECONDS:-30}" \
  --proxy-retry-max-seconds "${MEMGEN_TEACHER_PROXY_RETRY_MAX_SECONDS:-300}" \
  --connect-timeout-seconds "${MEMGEN_TEACHER_CONNECT_TIMEOUT_SECONDS:-30}" \
  --read-timeout-seconds "${MEMGEN_TEACHER_READ_TIMEOUT_SECONDS:-180}" \
  --limit "${MEMGEN_PHASE2_ANCHOR_LIMIT:-0}" \
  --resume

python scripts/compile_steering_vectors.py \
  --approved-bank "$APPROVED_BANK" \
  --experiences "$EXPERIENCES" \
  --evidence-anchors "$EVIDENCE_ANCHORS" \
  --output-dir "$VECTOR_DIR" \
  --model "$STUDENT_MODEL" \
  --model-revision "$STUDENT_REVISION" \
  --device cuda \
  --dtype "${MEMGEN_PHASE2_DTYPE:-bfloat16}" \
  --attn-implementation "$COMPILER_ATTN" \
  --layers "${MEMGEN_PHASE2_LAYERS:-8,16,24}" \
  --experience-types "${MEMGEN_PHASE2_EXPERIENCE_TYPES:-answer_correctness}" \
  --batch-size "${MEMGEN_PHASE2_COMPILER_BATCH_SIZE:-2}" \
  --limit "${MEMGEN_PHASE2_COMPILER_LIMIT:-0}"

# The initial global stay-on-track hypothesis is answer correctness.  Format
# evidence is compiled separately above and remains available for a later
# type-conditioned Phase 3 experiment instead of diluting this primary vector.
PRIMARY_ARTIFACT="$VECTOR_DIR/global-stay-on-track-answer_correctness.pt"
if [[ ! -s "$PRIMARY_ARTIFACT" ]]; then
  echo "Primary answer_correctness vector was not produced: $PRIMARY_ARTIFACT" >&2
  exit 1
fi

python scripts/calibrate_phase2_steering.py \
  --artifact "$PRIMARY_ARTIFACT" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$CALIBRATION_DIR" \
  --model "$STUDENT_MODEL" \
  --model-revision "$STUDENT_REVISION" \
  --dataset-revision "$DATASET_REVISION" \
  --layers "${MEMGEN_PHASE2_LAYERS:-8,16,24}" \
  --alphas "${MEMGEN_PHASE2_ALPHAS:-0.02,0.05}" \
  --gate-slopes "${MEMGEN_PHASE2_GATE_SLOPES:-0.10,0.25}" \
  --max-injections-grid "${MEMGEN_PHASE2_MAX_INJECTIONS_GRID:-2}" \
  --entropy-quantile "${MEMGEN_PHASE2_ENTROPY_QUANTILE:-0.85}" \
  --tune-size "${MEMGEN_PHASE2_TUNE_SIZE:-100}" \
  --confirm-size "${MEMGEN_PHASE2_CONFIRM_SIZE:-100}" \
  --max-new-tokens "${MEMGEN_PHASE2_MAX_NEW_TOKENS:-768}" \
  --r-max "${MEMGEN_PHASE2_R_MAX:-0.10}" \
  --sink-token-count "${MEMGEN_PHASE2_SINK_TOKEN_COUNT:-4}" \
  --seed "${MEMGEN_PHASE2_SEED:-42}" \
  --device cuda \
  --dtype "${MEMGEN_PHASE2_DTYPE:-bfloat16}" \
  --attn-implementation "$EVAL_ATTN"

echo "Phase 2 artifacts: $RUN_DIR"
echo "Evidence-anchor report: $EVIDENCE_ANCHOR_REPORT"
echo "Vector report: $VECTOR_DIR/vector_compilation_report.json"
echo "Calibration report: $CALIBRATION_DIR/phase2_calibration_report.json"
