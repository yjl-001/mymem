#!/usr/bin/env bash
# V3: compile embedding keys once, then run a resumable vanilla-vs-V3 evaluation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

STAGE="all"
LOGICAL_SPLIT="calibration-val"
OFFSET=0
LIMIT=8
PARITY_SAMPLES=8
RUN_DIR=""
SELECTOR_CALIBRATION=""
RETRIEVAL_EMBEDDING_TRANSFORM="none"
QUERY_POOLING="last_valid_token"
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --logical-split) LOGICAL_SPLIT="$2"; shift 2 ;;
    --offset) OFFSET="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --parity-samples) PARITY_SAMPLES="$2"; shift 2 ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    --selector-calibration) SELECTOR_CALIBRATION="$2"; shift 2 ;;
    --retrieval-embedding-transform) RETRIEVAL_EMBEDDING_TRANSFORM="$2"; shift 2 ;;
    --query-pooling) QUERY_POOLING="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [--stage offline|eval|all] [--logical-split SPLIT] [--limit N] [--run-dir DIR] [--selector-calibration FILE] [--retrieval-embedding-transform TRANSFORM] [--query-pooling POOLING] PHASE1_DIR E0_DIR RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi
if [[ "$STAGE" != "offline" && "$STAGE" != "eval" && "$STAGE" != "all" ]]; then
  echo "--stage must be offline, eval, or all" >&2
  exit 2
fi
if [[ "$LOGICAL_SPLIT" != "calibration-val" && "$LOGICAL_SPLIT" != "dev-test" && "$LOGICAL_SPLIT" != "final-test" ]]; then
  echo "Unexpected --logical-split" >&2
  exit 2
fi
if [[ "$RETRIEVAL_EMBEDDING_TRANSFORM" != "none" && "$RETRIEVAL_EMBEDDING_TRANSFORM" != "key_bank_centroid_center_l2" ]]; then
  echo "Unexpected --retrieval-embedding-transform" >&2
  exit 2
fi
if [[ "$QUERY_POOLING" != "last_valid_token" && "$QUERY_POOLING" != "last_token_before_trigger_boundary" ]]; then
  echo "Unexpected --query-pooling" >&2
  exit 2
fi
for VALUE in "$OFFSET" "$LIMIT" "$PARITY_SAMPLES"; do
  if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
    echo "offset, limit, and parity-samples must be non-negative integers" >&2
    exit 2
  fi
done

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"
V3_BANK_DIR="$OUTPUT_ROOT/v3_bank"
if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$OUTPUT_ROOT/evaluation/$LOGICAL_SPLIT"
fi
DEVICE="${MEMGEN_V3_DEVICE:-cuda}"
DTYPE="${MEMGEN_V3_DTYPE:-bfloat16}"
export CUDA_VISIBLE_DEVICES="${MEMGEN_V3_CUDA_VISIBLE_DEVICES:-0}"

SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
E0_FINAL_REPORT="$E0_DIR/e0_final_report.json"
for REQUIRED in "$SPLIT_MANIFEST" "$MEMORY_RECORDS" "$SIDE_KV_MANIFEST" "$E0_FINAL_REPORT" "$RISK_ARTIFACT"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

if [[ "$STAGE" == "offline" || "$STAGE" == "all" ]]; then
  python scripts/compile_v3_retrieval_keys.py \
    --memory-records "$MEMORY_RECORDS" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --e0-final-report "$E0_FINAL_REPORT" \
    --output-dir "$V3_BANK_DIR" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  for REQUIRED in "$V3_BANK_DIR/retrieval_key_manifest.json" "$V3_BANK_DIR/v3_offline_report.json"; do
    if [[ ! -s "$REQUIRED" ]]; then
      echo "Missing V3 offline artifact: $REQUIRED" >&2
      exit 1
    fi
  done
  SELECTOR_ARGS=()
  if [[ -n "$SELECTOR_CALIBRATION" ]]; then
    if [[ ! -s "$SELECTOR_CALIBRATION" ]]; then
      echo "Missing selector calibration: $SELECTOR_CALIBRATION" >&2
      exit 1
    fi
    SELECTOR_ARGS=(--selector-calibration "$SELECTOR_CALIBRATION")
  fi
  python scripts/evaluate_v3_experience_memory.py \
    --split-manifest "$SPLIT_MANIFEST" \
    --logical-split "$LOGICAL_SPLIT" \
    --memory-records "$MEMORY_RECORDS" \
    --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --v3-offline-report "$V3_BANK_DIR/v3_offline_report.json" \
    --e0-final-report "$E0_FINAL_REPORT" \
    --risk-artifact "$RISK_ARTIFACT" \
    --retrieval-embedding-transform "$RETRIEVAL_EMBEDDING_TRANSFORM" \
    --query-pooling "$QUERY_POOLING" \
    "${SELECTOR_ARGS[@]}" \
    --output-dir "$RUN_DIR" \
    --offset "$OFFSET" \
    --limit "$LIMIT" \
    --parity-samples "$PARITY_SAMPLES" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
fi

echo "V3 bank: $V3_BANK_DIR"
echo "V3 evaluation: $RUN_DIR"
