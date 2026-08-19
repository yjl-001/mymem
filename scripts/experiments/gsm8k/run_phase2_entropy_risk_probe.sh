#!/usr/bin/env bash
# Fixed, one-shot entropy-risk steering probe with no Phase-2 AI calls.
# Usage:
#   bash scripts/experiments/gsm8k/run_phase2_entropy_risk_probe.sh \
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
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
for REQUIRED_FILE in "$APPROVED_BANK" "$EXPERIENCES" "$SPLIT_MANIFEST"; do
  if [[ ! -s "$REQUIRED_FILE" ]]; then
    echo "Missing or empty Phase 1 artifact: $REQUIRED_FILE" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="$MEMGEN_DEVICES"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/experiments/phase2/gsm8k/gsm8k_phase2_entropy_risk_probe_${RUN_TAG}"
VECTOR_DIR="$RUN_DIR/vector_artifacts"
EVAL_DIR="$RUN_DIR/evaluation"
mkdir -p "$RUN_DIR"

STUDENT_MODEL="$MEMGEN_GSM8K_STUDENT_MODEL"
STUDENT_REVISION="${MEMGEN_GSM8K_STUDENT_REVISION:-main}"
DATASET_REVISION="${MEMGEN_GSM8K_DATASET_REVISION:-main}"
EVAL_ATTN="${MEMGEN_PHASE2_EVAL_ATTN_IMPLEMENTATION:-eager}"
EVAL_OFFSET="${MEMGEN_PHASE2_RISK_EVAL_OFFSET:-0}"

python scripts/compile_phase2_entropy_risk_vector.py \
  --approved-bank "$APPROVED_BANK" \
  --experiences "$EXPERIENCES" \
  --output-dir "$VECTOR_DIR" \
  --model "$STUDENT_MODEL" \
  --model-revision "$STUDENT_REVISION" \
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
  --min-events-per-label "${MEMGEN_PHASE2_RISK_MIN_EVENTS:-50}" \
  --min-heldout-roc-auc "${MEMGEN_PHASE2_RISK_MIN_AUC:-0.60}" \
  --limit "${MEMGEN_PHASE2_COMPILER_LIMIT:-0}"

ARTIFACT="$VECTOR_DIR/phase2-entropy-risk-state-delta-answer_correctness.pt"
for CONDITION in vanilla entropy_only real_vector random_vector reversed_vector; do
  EXTRA_ARGS=(--use-artifact-entropy-threshold)
  if [[ "$CONDITION" != "vanilla" ]]; then
    EXTRA_ARGS+=(--risk-gate --first-high-entropy-only)
  fi
  python scripts/evaluate_steering_vector.py \
    --artifact "$ARTIFACT" \
    --split-manifest "$SPLIT_MANIFEST" \
    --output-dir "$EVAL_DIR/$CONDITION" \
    --condition "$CONDITION" \
    --model "$STUDENT_MODEL" \
    --model-revision "$STUDENT_REVISION" \
    --dataset-revision "$DATASET_REVISION" \
    --logical-split "${MEMGEN_PHASE2_RISK_EVAL_SPLIT:-dev-test}" \
    --layer "${MEMGEN_PHASE2_RISK_LAYER:-24}" \
    --alpha "${MEMGEN_PHASE2_RISK_ALPHA:-0.05}" \
    --gate-slope "${MEMGEN_PHASE2_RISK_GATE_SLOPE:-0.10}" \
    --max-injections 1 \
    --r-max "${MEMGEN_PHASE2_R_MAX:-0.10}" \
    --sink-token-count "${MEMGEN_PHASE2_SINK_TOKEN_COUNT:-4}" \
    --max-new-tokens "${MEMGEN_PHASE2_MAX_NEW_TOKENS:-768}" \
    --offset "$EVAL_OFFSET" \
    --limit "${MEMGEN_PHASE2_RISK_EVAL_LIMIT:-100}" \
    --seed "${MEMGEN_PHASE2_SEED:-42}" \
    --device cuda \
    --dtype "${MEMGEN_PHASE2_DTYPE:-bfloat16}" \
    --attn-implementation "$EVAL_ATTN" \
    "${EXTRA_ARGS[@]}"
done

python scripts/summarize_phase2_entropy_risk_probe.py \
  --diagnostic-report "$VECTOR_DIR/entropy_risk_diagnostic_report.json" \
  --evaluation-root "$EVAL_DIR" \
  --bootstrap-resamples "${MEMGEN_PHASE2_RISK_BOOTSTRAP_RESAMPLES:-10000}" \
  --min-paired-events "${MEMGEN_PHASE2_RISK_MIN_PAIRED_EVENTS:-50}" \
  --output "$RUN_DIR/entropy_risk_probe_summary.json"

echo "Phase 2 entropy-risk probe: $RUN_DIR"
echo "Summary: $RUN_DIR/entropy_risk_probe_summary.json"
