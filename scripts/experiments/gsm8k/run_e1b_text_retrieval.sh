#!/usr/bin/env bash
# E1-B: completion-aware BM25 matched text versus shuffled text and no memory.
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
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --limit) LIMIT="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --logical-split) LOGICAL_SPLIT="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done
if [[ "${#POSITIONAL[@]}" -ne 2 ]]; then
  echo "Usage: $0 [--limit N] [--offset N] [--logical-split calibration-val|dev-test] PHASE1_DIR E0_DIR" >&2
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
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1b_text-retrieval_${LOGICAL_SPLIT}_${RUN_TAG}"
mkdir -p "$RUN_DIR"

python scripts/build_e1b_retrieval_manifest.py \
  --split-manifest "$SPLIT_MANIFEST" \
  --memory-records "$MEMORY_RECORDS" \
  --bm25-index "$BM25_INDEX" \
  --side-kv-manifest "$SIDE_KV_MANIFEST" \
  --e0-final-report "$E0_FINAL_REPORT" \
  --output "$RUN_DIR/assignment_manifest.json" \
  --logical-split "$LOGICAL_SPLIT" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --max-new-tokens 768 \
  --device cuda \
  --dtype bfloat16

python scripts/evaluate_e1b_text_retrieval.py \
  --assignment-manifest "$RUN_DIR/assignment_manifest.json" \
  --memory-records "$MEMORY_RECORDS" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$RUN_DIR/evaluation" \
  --device cuda \
  --dtype bfloat16

echo "E1-B artifacts: $RUN_DIR"
echo "E1-B summary: $RUN_DIR/evaluation/e1b_summary.json"
