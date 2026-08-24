#!/usr/bin/env bash
# Complete system: gate + semantic retrieval + persistent side-KV with controls.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E1_SERVER_ENV="${MEMGEN_E1_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e1.server.env}"
cd "$REPO_ROOT"

if [[ ! -f "$E1_SERVER_ENV" ]]; then
  echo "Missing $E1_SERVER_ENV." >&2
  echo "Copy scripts/experiments/gsm8k/e1.server.env.example to .e1.server.env." >&2
  exit 1
fi
source "$E1_SERVER_ENV"

LIMIT=100
OFFSET=0
LOGICAL_SPLIT="calibration-val"
PRINT_CONFIG=0
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --offset)
      OFFSET="$2"
      shift 2
      ;;
    --logical-split)
      LOGICAL_SPLIT="$2"
      shift 2
      ;;
    --print-config)
      PRINT_CONFIG=1
      shift
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 3 ]]; then
  echo "Usage: $0 [--limit N] [--offset N] [--logical-split calibration-val|dev-test] PHASE1_DIR E0_DIR RISK_ARTIFACT" >&2
  exit 2
fi
if [[ "$LOGICAL_SPLIT" != "dev-test" && "$LOGICAL_SPLIT" != "calibration-val" ]]; then
  echo "E1 development run only supports dev-test or calibration-val" >&2
  exit 2
fi
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || [[ "$LIMIT" -le 1 ]]; then
  echo "--limit must be an integer greater than one" >&2
  exit 2
fi
if ! [[ "$OFFSET" =~ ^[0-9]+$ ]]; then
  echo "--offset must be a non-negative integer" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
RISK_ARTIFACT="${POSITIONAL[2]}"
: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .e1.server.env}"

SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
BM25_INDEX="$E0_DIR/bm25_index.v1.json"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
E0_FINAL_REPORT="$E0_DIR/e0_final_report.json"
for REQUIRED_FILE in \
  "$SPLIT_MANIFEST" \
  "$MEMORY_RECORDS" \
  "$BM25_INDEX" \
  "$SIDE_KV_MANIFEST" \
  "$E0_FINAL_REPORT" \
  "$RISK_ARTIFACT"; do
  if [[ ! -s "$REQUIRED_FILE" ]]; then
    echo "Missing or empty frozen input: $REQUIRED_FILE" >&2
    exit 1
  fi
done

MODEL="$(jq -r '.reasoner.model_name' "$SIDE_KV_MANIFEST")"
MODEL_REVISION="$(jq -r '.reasoner.model_revision' "$SIDE_KV_MANIFEST")"
TOKENIZER_REVISION="$(jq -r '.reasoner.tokenizer_revision' "$SIDE_KV_MANIFEST")"
DATASET_REVISION="$(jq -r '.dataset.revision' "$SPLIT_MANIFEST")"
export CUDA_VISIBLE_DEVICES="${MEMGEN_E1_CUDA_VISIBLE_DEVICES:-0}"

if [[ "$PRINT_CONFIG" == "1" ]]; then
  printf 'phase1_dir=%s\n' "$PHASE1_DIR"
  printf 'e0_dir=%s\n' "$E0_DIR"
  printf 'risk_artifact=%s\n' "$RISK_ARTIFACT"
  printf 'output_root=%s\n' "$MEMGEN_OUTPUT_ROOT"
  printf 'model=%s\n' "$MODEL"
  printf 'model_revision=%s\n' "$MODEL_REVISION"
  printf 'tokenizer_revision=%s\n' "$TOKENIZER_REVISION"
  printf 'dataset_revision=%s\n' "$DATASET_REVISION"
  printf 'logical_split=%s\n' "$LOGICAL_SPLIT"
  printf 'offset=%s\n' "$OFFSET"
  printf 'limit=%s\n' "$LIMIT"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'system_profile=%s\n' 'layer24-bm25-top1-gate-persistent-logslots-log10-v1'
  exit 0
fi

RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="gsm8k_e1d_full-system_layer24_${LOGICAL_SPLIT}_${RUN_TAG}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/$RUN_ID"
ASSIGNMENT_MANIFEST="$RUN_DIR/assignment_manifest.json"
EVALUATION_DIR="$RUN_DIR/evaluation"
mkdir -p "$RUN_DIR"

python scripts/build_e1_assignment_manifest.py \
  --split-manifest "$SPLIT_MANIFEST" \
  --memory-records "$MEMORY_RECORDS" \
  --bm25-index "$BM25_INDEX" \
  --side-kv-manifest "$SIDE_KV_MANIFEST" \
  --e0-final-report "$E0_FINAL_REPORT" \
  --risk-artifact "$RISK_ARTIFACT" \
  --output "$ASSIGNMENT_MANIFEST" \
  --logical-split "$LOGICAL_SPLIT" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --max-new-tokens 768 \
  --shuffle-seed 42 \
  --device cuda \
  --dtype bfloat16

python scripts/evaluate_e1_experience_memory.py \
  --assignment-manifest "$ASSIGNMENT_MANIFEST" \
  --side-kv-manifest "$SIDE_KV_MANIFEST" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$EVALUATION_DIR" \
  --device cuda \
  --dtype bfloat16

python scripts/summarize_e1_experience_memory.py \
  --assignment-manifest "$ASSIGNMENT_MANIFEST" \
  --results "$EVALUATION_DIR/results.jsonl" \
  --run-report "$EVALUATION_DIR/run_report.json" \
  --output "$RUN_DIR/e1d_summary.json" \
  --bootstrap-resamples 10000 \
  --min-primary-pairs 20 \
  --seed 42

echo "Full-system artifacts: $RUN_DIR"
echo "Full-system summary: $RUN_DIR/e1d_summary.json"
