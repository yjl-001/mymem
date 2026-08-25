#!/usr/bin/env bash
# Canonical GSM8K base-reasoner accuracy and native-vs-live-cache parity.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E1_ENV="${MEMGEN_E1_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e1.server.env}"
cd "$REPO_ROOT"

if [[ ! -f "$E1_ENV" ]]; then
  echo "Missing $E1_ENV." >&2
  echo "Copy e1.server.env.example to .e1.server.env." >&2
  exit 1
fi
source "$E1_ENV"

LIMIT=32
OFFSET=0
LOGICAL_SPLIT="calibration-val"
ATTENTION_IMPLEMENTATION="sdpa"
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
    --attention-implementation)
      ATTENTION_IMPLEMENTATION="$2"
      shift 2
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

if [[ "${#POSITIONAL[@]}" -ne 2 ]]; then
  echo "Usage: $0 [--limit N (0=all)] [--offset N] [--logical-split calibration-val|dev-test|final-test] [--attention-implementation eager|sdpa|flash_attention_2] PHASE1_DIR E0_DIR" >&2
  exit 2
fi
if [[ "$LOGICAL_SPLIT" != "calibration-val" && "$LOGICAL_SPLIT" != "dev-test" && "$LOGICAL_SPLIT" != "final-test" ]]; then
  echo "Unsupported logical split" >&2
  exit 2
fi
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || ! [[ "$OFFSET" =~ ^[0-9]+$ ]]; then
  echo "--limit and --offset must be non-negative integers" >&2
  exit 2
fi
if [[ "$ATTENTION_IMPLEMENTATION" != "eager" && "$ATTENTION_IMPLEMENTATION" != "sdpa" && "$ATTENTION_IMPLEMENTATION" != "flash_attention_2" ]]; then
  echo "Unsupported attention implementation: $ATTENTION_IMPLEMENTATION" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .e1.server.env}"
for REQUIRED_FILE in "$SPLIT_MANIFEST" "$SIDE_KV_MANIFEST"; do
  if [[ ! -s "$REQUIRED_FILE" ]]; then
    echo "Missing or empty frozen input: $REQUIRED_FILE" >&2
    exit 1
  fi
done

export CUDA_VISIBLE_DEVICES="${MEMGEN_E1_CUDA_VISIBLE_DEVICES:-0}"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_base-parity_${LOGICAL_SPLIT}_${ATTENTION_IMPLEMENTATION}_${RUN_TAG}"

python scripts/evaluate_gsm8k_base_parity.py \
  --split-manifest "$SPLIT_MANIFEST" \
  --side-kv-manifest "$SIDE_KV_MANIFEST" \
  --output-dir "$RUN_DIR" \
  --logical-split "$LOGICAL_SPLIT" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --device cuda \
  --dtype bfloat16 \
  --attention-implementation "$ATTENTION_IMPLEMENTATION"

echo "Base parity artifacts: $RUN_DIR"
echo "Base parity summary: $RUN_DIR/base_parity_summary.json"
