#!/usr/bin/env bash
# V3.3 answer-blind offline pooling audit; never runs dev-test or final-test.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

V31_CALIBRATION_DIR=""
V31_SELECTOR_CALIBRATION=""
OUTPUT_DIR=""
PROGRESS_EVERY=25
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --v31-calibration-dir) V31_CALIBRATION_DIR="$2"; shift 2 ;;
    --v31-selector-calibration) V31_SELECTOR_CALIBRATION="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --progress-every) PROGRESS_EVERY="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 3 ]]; then
  echo "Usage: $0 [options] PHASE1_DIR E0_DIR OUTPUT_ROOT" >&2
  exit 2
fi
if ! [[ "$PROGRESS_EVERY" =~ ^[1-9][0-9]*$ ]]; then
  echo "progress-every must be a positive integer" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
OUTPUT_ROOT="${POSITIONAL[2]}"
V3_BANK_DIR="$OUTPUT_ROOT/v3_bank"
V31_DIR="$OUTPUT_ROOT/v3_1_selector"
if [[ -z "$V31_CALIBRATION_DIR" ]]; then
  V31_CALIBRATION_DIR="$V31_DIR/calibration_baseline"
fi
if [[ -z "$V31_SELECTOR_CALIBRATION" ]]; then
  V31_SELECTOR_CALIBRATION="$V31_DIR/margin_selector_calibration.json"
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$OUTPUT_ROOT/v3_3_pooling_audit"
fi
DEVICE="${MEMGEN_V3_DEVICE:-cuda}"
DTYPE="${MEMGEN_V3_DTYPE:-bfloat16}"
export CUDA_VISIBLE_DEVICES="${MEMGEN_V3_CUDA_VISIBLE_DEVICES:-0}"

for REQUIRED in \
  "$PHASE1_DIR/split_manifest.json" \
  "$E0_DIR/memory_records.v2.jsonl" \
  "$V3_BANK_DIR/retrieval_key_manifest.json" \
  "$V31_CALIBRATION_DIR/results.jsonl" \
  "$V31_CALIBRATION_DIR/run_profile.json" \
  "$V31_CALIBRATION_DIR/run_report.json" \
  "$V31_SELECTOR_CALIBRATION"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done
python -c 'import json,sys; value=json.load(open(sys.argv[1], encoding="utf-8")); raise SystemExit(0 if value.get("status") == "completed" and value.get("remaining_sample_count") == 0 else 1)' "$V31_CALIBRATION_DIR/run_report.json"

python scripts/audit_v3_pooling_candidates.py \
  --results "$V31_CALIBRATION_DIR/results.jsonl" \
  --run-profile "$V31_CALIBRATION_DIR/run_profile.json" \
  --selector-calibration "$V31_SELECTOR_CALIBRATION" \
  --split-manifest "$PHASE1_DIR/split_manifest.json" \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --retrieval-key-manifest "$V3_BANK_DIR/retrieval_key_manifest.json" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --progress-every "$PROGRESS_EVERY"

echo "V3.3 pooling audit completed: $OUTPUT_DIR"
echo "Summary: $OUTPUT_DIR/pooling_audit.md"
echo "Full report: $OUTPUT_DIR/pooling_audit.json"
echo "Per-sample top-2 traces: $OUTPUT_DIR/pooling_audit_samples.jsonl"
echo "Reusable embeddings: $OUTPUT_DIR/pooling_embeddings.safetensors"
