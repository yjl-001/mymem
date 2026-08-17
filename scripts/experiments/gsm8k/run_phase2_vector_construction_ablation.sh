#!/usr/bin/env bash
# Compare no-new-AI steering-vector constructions on the frozen Phase 1 bank.
# Usage:
#   bash scripts/experiments/gsm8k/run_phase2_vector_construction_ablation.sh \
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
RUN_ID="gsm8k_phase2_vector_construction_ablation_${RUN_TAG}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/experiments/phase2/gsm8k/$RUN_ID"
VECTOR_DIR="$RUN_DIR/vector_artifacts"
CALIBRATION_DIR="$RUN_DIR/calibration"
mkdir -p "$RUN_DIR"

STUDENT_MODEL="$MEMGEN_GSM8K_STUDENT_MODEL"
STUDENT_REVISION="${MEMGEN_GSM8K_STUDENT_REVISION:-main}"
DATASET_REVISION="${MEMGEN_GSM8K_DATASET_REVISION:-main}"
COMPILER_ATTN="${MEMGEN_PHASE2_COMPILER_ATTN_IMPLEMENTATION:-flash_attention_2}"
EVAL_ATTN="${MEMGEN_PHASE2_EVAL_ATTN_IMPLEMENTATION:-eager}"
METHODS="${MEMGEN_PHASE2_ABLATION_METHODS:-pair_normalized_delta,global_centroid_delta,mechanism_balanced_centroid_delta,seal_style_execution_minus_nonexecution}"

python scripts/compile_phase2_vector_ablation.py \
  --approved-bank "$APPROVED_BANK" \
  --experiences "$EXPERIENCES" \
  --output-dir "$VECTOR_DIR" \
  --model "$STUDENT_MODEL" \
  --model-revision "$STUDENT_REVISION" \
  --device cuda \
  --dtype "${MEMGEN_PHASE2_DTYPE:-bfloat16}" \
  --attn-implementation "$COMPILER_ATTN" \
  --layers "${MEMGEN_PHASE2_LAYERS:-8,16,24}" \
  --experience-types "${MEMGEN_PHASE2_EXPERIENCE_TYPES:-answer_correctness}" \
  --methods "$METHODS" \
  --batch-size "${MEMGEN_PHASE2_COMPILER_BATCH_SIZE:-2}" \
  --min-evidence-count "${MEMGEN_PHASE2_MIN_EVIDENCE_COUNT:-50}" \
  --min-cluster-evidence-count "${MEMGEN_PHASE2_MIN_CLUSTER_EVIDENCE_COUNT:-25}" \
  --limit "${MEMGEN_PHASE2_COMPILER_LIMIT:-0}"

IFS=',' read -r -a METHOD_ARRAY <<< "$METHODS"
for METHOD in "${METHOD_ARRAY[@]}"; do
  ARTIFACT="$VECTOR_DIR/phase2-${METHOD}-answer_correctness.pt"
  if [[ ! -s "$ARTIFACT" ]]; then
    echo "[phase2-ablation] skipped unavailable method: $METHOD"
    continue
  fi
  python scripts/calibrate_phase2_steering.py \
    --artifact "$ARTIFACT" \
    --split-manifest "$SPLIT_MANIFEST" \
    --output-dir "$CALIBRATION_DIR/$METHOD" \
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
done

python scripts/summarize_phase2_vector_ablation.py \
  --compiler-report "$VECTOR_DIR/vector_construction_ablation_report.json" \
  --calibration-root "$CALIBRATION_DIR" \
  --output "$RUN_DIR/vector_construction_comparison.json"

echo "Phase 2 no-new-AI ablation: $RUN_DIR"
echo "Comparison: $RUN_DIR/vector_construction_comparison.json"
