#!/usr/bin/env bash
# V4: MI-style offline bank construction followed by one-stage online routing.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STAGE="all"
LOGICAL_SPLIT="dev-test"
OFFSET=0
LIMIT=8
MAX_NEW_TOKENS=0
RUN_DIR=""
RESUME=0
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --logical-split) LOGICAL_SPLIT="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--stage construct-cluster|construct-cards|construct|side-kv|selector|offline|eval|all] [--logical-split calibration-val|dev-test] [--offset N] [--limit N] [--max-new-tokens N] [--resume] PHASE1_DIR E0_DIR RISK_ARTIFACT OUTPUT_ROOT"
      exit 0
      ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [--stage construct-cluster|construct-cards|construct|side-kv|selector|offline|eval|all] [--logical-split calibration-val|dev-test] [--offset N] [--limit N] [--max-new-tokens N] [--resume] PHASE1_DIR E0_DIR RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi
case "$STAGE" in
  construct-cluster|construct-cards|construct|side-kv|selector|offline|eval|all) ;;
  *) echo "Unexpected --stage: $STAGE" >&2; exit 2 ;;
esac
if [[ "$LOGICAL_SPLIT" != "calibration-val" && "$LOGICAL_SPLIT" != "dev-test" ]]; then
  echo "V4 initial evaluation permits calibration-val or dev-test only" >&2
  exit 2
fi
for VALUE in "$OFFSET" "$LIMIT" "$MAX_NEW_TOKENS"; do
  if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
    echo "offset, limit, and max-new-tokens must be non-negative integers" >&2
    exit 2
  fi
done

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"
DEVICE="${MEMGEN_V4_DEVICE:-cuda}"
DTYPE="${MEMGEN_V4_DTYPE:-bfloat16}"
CANONICAL_BATCH_SIZE="${MEMGEN_V4_1_CANONICAL_BATCH_SIZE:-24}"
PAIR_BATCH_SIZE="${MEMGEN_V4_1_PAIR_BATCH_SIZE:-24}"
NEIGHBOR_COUNT="${MEMGEN_V4_1_NEIGHBOR_COUNT:-12}"
EMBEDDING_DEVICE="${MEMGEN_V4_1_EMBEDDING_DEVICE:-cpu}"
export CUDA_VISIBLE_DEVICES="${MEMGEN_V4_CUDA_VISIBLE_DEVICES:-0}"

for VALUE in "$CANONICAL_BATCH_SIZE" "$PAIR_BATCH_SIZE" "$NEIGHBOR_COUNT"; do
  if ! [[ "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "V4.1 construction batch sizes and neighbor count must be positive integers" >&2
    exit 2
  fi
done

SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
REASONER_MANIFEST="$E0_DIR/side_kv_manifest.json"
SOURCE_CONSTRUCTION_DIR="${MEMGEN_V4_SOURCE_CONSTRUCTION_DIR:-$OUTPUT_ROOT/offline/construction}"
CONSTRUCTION_DIR="${MEMGEN_V4_1_CONSTRUCTION_DIR:-$OUTPUT_ROOT/offline/construction_v4_1}"
SIDE_KV_DIR="$OUTPUT_ROOT/offline/side_kv"
SELECTOR_DIR="$OUTPUT_ROOT/offline/selector"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$OUTPUT_ROOT/evaluation/$LOGICAL_SPLIT"
fi

for REQUIRED in "$SPLIT_MANIFEST" "$EXPERIENCES" "$REASONER_MANIFEST" "$RISK_ARTIFACT"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty V4 input: $REQUIRED" >&2
    exit 1
  fi
done

RESUME_ARGS=()
if [[ "$RESUME" -eq 1 ]]; then
  RESUME_ARGS=(--resume)
fi

if [[ "$STAGE" == "construct-cluster" || "$STAGE" == "construct-cards" || "$STAGE" == "construct" || "$STAGE" == "offline" || "$STAGE" == "all" ]]; then
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "DEEPSEEK_API_KEY is required for V4 bank construction" >&2
    exit 1
  fi
  for REQUIRED in \
    "$SOURCE_CONSTRUCTION_DIR/repair_signatures.jsonl" \
    "$SOURCE_CONSTRUCTION_DIR/construction_profile.json"; do
    if [[ ! -s "$REQUIRED" ]]; then
      echo "Missing V4 source-signature artifact: $REQUIRED" >&2
      exit 1
    fi
  done
  CONSTRUCTION_STAGE="all"
  if [[ "$STAGE" == "construct-cluster" ]]; then
    CONSTRUCTION_STAGE="cluster"
  elif [[ "$STAGE" == "construct-cards" ]]; then
    CONSTRUCTION_STAGE="cards"
  fi
  python scripts/build_v4_1_repair_bank.py \
    --experiences "$EXPERIENCES" \
    --split-manifest "$SPLIT_MANIFEST" \
    --source-signatures "$SOURCE_CONSTRUCTION_DIR/repair_signatures.jsonl" \
    --source-construction-profile "$SOURCE_CONSTRUCTION_DIR/construction_profile.json" \
    --output-dir "$CONSTRUCTION_DIR" \
    --dataset-revision main \
    --stage "$CONSTRUCTION_STAGE" \
    --canonical-batch-size "$CANONICAL_BATCH_SIZE" \
    --pair-batch-size "$PAIR_BATCH_SIZE" \
    --neighbor-count "$NEIGHBOR_COUNT" \
    --embedding-device "$EMBEDDING_DEVICE" \
    "${RESUME_ARGS[@]}"
fi

if [[ "$STAGE" == "side-kv" || "$STAGE" == "offline" || "$STAGE" == "all" ]]; then
  for REQUIRED in "$CONSTRUCTION_DIR/bank_records.jsonl" "$CONSTRUCTION_DIR/bank_manifest.json"; do
    if [[ ! -s "$REQUIRED" ]]; then
      echo "Missing V4 construction artifact: $REQUIRED" >&2
      exit 1
    fi
  done
  python scripts/compile_v4_side_kv.py \
    --bank-records "$CONSTRUCTION_DIR/bank_records.jsonl" \
    --bank-manifest "$CONSTRUCTION_DIR/bank_manifest.json" \
    --reasoner-manifest "$REASONER_MANIFEST" \
    --output-dir "$SIDE_KV_DIR" \
    --layer 24 \
    --device "$DEVICE" \
    --dtype "$DTYPE"
fi

if [[ "$STAGE" == "selector" || "$STAGE" == "offline" || "$STAGE" == "all" ]]; then
  for REQUIRED in \
    "$CONSTRUCTION_DIR/bank_records.jsonl" \
    "$CONSTRUCTION_DIR/bank_manifest.json" \
    "$SIDE_KV_DIR/v4_side_kv_manifest.json"; do
    if [[ ! -s "$REQUIRED" ]]; then
      echo "Missing V4 selector input: $REQUIRED" >&2
      exit 1
    fi
  done
  python scripts/compile_v4_selector_anchors.py \
    --experiences "$EXPERIENCES" \
    --split-manifest "$SPLIT_MANIFEST" \
    --bank-records "$CONSTRUCTION_DIR/bank_records.jsonl" \
    --bank-manifest "$CONSTRUCTION_DIR/bank_manifest.json" \
    --side-kv-manifest "$SIDE_KV_DIR/v4_side_kv_manifest.json" \
    --token-risk-artifact "$RISK_ARTIFACT" \
    --output-dir "$SELECTOR_DIR" \
    --dataset-revision main \
    --device "$DEVICE" \
    --dtype "$DTYPE"
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  for REQUIRED in \
    "$SIDE_KV_DIR/v4_side_kv_manifest.json" \
    "$SELECTOR_DIR/v4_selector_anchor_manifest.json"; do
    if [[ ! -s "$REQUIRED" ]]; then
      echo "Missing V4 online artifact: $REQUIRED" >&2
      exit 1
    fi
  done
  python scripts/evaluate_v4_experience_memory.py \
    --split-manifest "$SPLIT_MANIFEST" \
    --side-kv-manifest "$SIDE_KV_DIR/v4_side_kv_manifest.json" \
    --selector-anchor-manifest "$SELECTOR_DIR/v4_selector_anchor_manifest.json" \
    --token-risk-artifact "$RISK_ARTIFACT" \
    --output-dir "$RUN_DIR" \
    --logical-split "$LOGICAL_SPLIT" \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "${RESUME_ARGS[@]}"
fi

echo "V4 source signatures: $SOURCE_CONSTRUCTION_DIR"
echo "V4.1 construction: $CONSTRUCTION_DIR"
echo "V4 side-KV: $SIDE_KV_DIR"
echo "V4 selector: $SELECTOR_DIR"
echo "V4 evaluation: $RUN_DIR"
