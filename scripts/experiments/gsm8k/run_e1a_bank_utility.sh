#!/usr/bin/env bash
# E1-A: fixed representative Phase 1 text bank versus random banks and no memory.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E1_SERVER_ENV="${MEMGEN_E1_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e1.server.env}"
cd "$REPO_ROOT"
if [[ ! -f "$E1_SERVER_ENV" ]]; then
  echo "Missing $E1_SERVER_ENV; copy e1.server.env.example first." >&2
  exit 1
fi
source "$E1_SERVER_ENV"

LIMIT=100
OFFSET=0
LOGICAL_SPLIT="calibration-val"
CATALOG_MANIFEST=""
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --logical-split) LOGICAL_SPLIT="$2"; shift 2 ;;
    --catalog-manifest) CATALOG_MANIFEST="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
if [[ "${#POSITIONAL[@]}" -ne 2 ]]; then
  echo "Usage: $0 [--limit N] [--offset N] [--logical-split calibration-val|dev-test] [--catalog-manifest PATH] PHASE1_DIR E0_DIR" >&2
  exit 2
fi
: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .e1.server.env}"
export CUDA_VISIBLE_DEVICES="${MEMGEN_E1_CUDA_VISIBLE_DEVICES:-0}"

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
BM25_INDEX="$E0_DIR/bm25_index.v1.json"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
E0_FINAL_REPORT="$E0_DIR/e0_final_report.json"
for FILE in "$SPLIT_MANIFEST" "$MEMORY_RECORDS" "$BM25_INDEX" "$SIDE_KV_MANIFEST" "$E0_FINAL_REPORT"; do
  [[ -s "$FILE" ]] || { echo "Missing frozen input: $FILE" >&2; exit 1; }
done

RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1a_bank-utility_${LOGICAL_SPLIT}_${RUN_TAG}"
mkdir -p "$RUN_DIR"

if [[ -z "$CATALOG_MANIFEST" ]]; then
  CATALOG_MANIFEST="$RUN_DIR/catalog_manifest.json"
  python scripts/build_e1a_catalog_manifest.py \
    --memory-records "$MEMORY_RECORDS" \
    --bm25-index "$BM25_INDEX" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --e0-final-report "$E0_FINAL_REPORT" \
    --output "$CATALOG_MANIFEST"
elif [[ ! -s "$CATALOG_MANIFEST" ]]; then
  echo "Missing frozen catalog manifest: $CATALOG_MANIFEST" >&2
  exit 1
fi

python scripts/evaluate_e1a_bank_utility.py \
  --catalog-manifest "$CATALOG_MANIFEST" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$RUN_DIR/evaluation" \
  --logical-split "$LOGICAL_SPLIT" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --max-new-tokens 768 \
  --device cuda \
  --dtype bfloat16

echo "E1-A artifacts: $RUN_DIR"
echo "E1-A summary: $RUN_DIR/evaluation/e1a_summary.json"
