#!/usr/bin/env bash
# Compare eager and FlashAttention2 under one batch-size-1 GSM8K contract.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E1_ENV="${MEMGEN_E1_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e1.server.env}"
PARITY_RUNNER="$REPO_ROOT/scripts/experiments/gsm8k/run_base_reasoner_parity.sh"
cd "$REPO_ROOT"

if [[ ! -f "$E1_ENV" ]]; then
  echo "Missing $E1_ENV." >&2
  echo "Copy e1.server.env.example to .e1.server.env." >&2
  exit 1
fi
source "$E1_ENV"

LIMIT=32
OFFSET=0
LOGICAL_SPLIT="final-test"
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
  echo "Usage: $0 [--limit N] [--offset N] [--logical-split calibration-val|dev-test|final-test] PHASE1_DIR E0_DIR" >&2
  exit 2
fi
if [[ "$LOGICAL_SPLIT" != "calibration-val" && "$LOGICAL_SPLIT" != "dev-test" && "$LOGICAL_SPLIT" != "final-test" ]]; then
  echo "Unsupported logical split: $LOGICAL_SPLIT" >&2
  exit 2
fi
if ! [[ "$LIMIT" =~ ^[0-9]+$ ]] || ! [[ "$OFFSET" =~ ^[0-9]+$ ]]; then
  echo "--limit and --offset must be non-negative integers" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .e1.server.env}"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"

for BACKEND in eager flash_attention_2; do
  MEMGEN_RUN_TAG="$RUN_TAG" \
  bash "$PARITY_RUNNER" \
    --logical-split "$LOGICAL_SPLIT" \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --attention-implementation "$BACKEND" \
    "$PHASE1_DIR" "$E0_DIR"
done

EAGER_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_base-parity_${LOGICAL_SPLIT}_eager_${RUN_TAG}"
FLASH_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_base-parity_${LOGICAL_SPLIT}_flash_attention_2_${RUN_TAG}"
COMPARISON_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_base-attention-comparison_${LOGICAL_SPLIT}_${RUN_TAG}"

python scripts/compare_gsm8k_attention_backends.py \
  --eager-dir "$EAGER_DIR" \
  --flash-dir "$FLASH_DIR" \
  --output "$COMPARISON_DIR/comparison_summary.json"

echo "Attention backend comparison: $COMPARISON_DIR/comparison_summary.json"
