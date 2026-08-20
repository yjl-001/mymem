#!/usr/bin/env bash
# E0: compile reviewed experience text into a leak-audited BM25 + side-KV bank.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
E0_SERVER_ENV="${MEMGEN_E0_ENV_FILE:-$REPO_ROOT/scripts/experiments/gsm8k/.e0.server.env}"
cd "$REPO_ROOT"

if [[ ! -f "$E0_SERVER_ENV" ]]; then
  echo "Missing $E0_SERVER_ENV." >&2
  echo "Copy scripts/experiments/gsm8k/e0.server.env.example to .e0.server.env." >&2
  exit 1
fi
source "$E0_SERVER_ENV"

PHASE1_DIR=""
AUDIT_CASES=""
TEXT_ONLY=0
PRINT_CONFIG=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --text-only)
      TEXT_ONLY=1
      shift
      ;;
    --print-config)
      PRINT_CONFIG=1
      shift
      ;;
    --audit-cases)
      if [[ "$#" -lt 2 ]]; then
        echo "--audit-cases requires a JSONL path" >&2
        exit 1
      fi
      AUDIT_CASES="$2"
      shift 2
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -z "$PHASE1_DIR" ]]; then
        PHASE1_DIR="$1"
      elif [[ -z "$AUDIT_CASES" ]]; then
        # Preserve the original optional positional audit-case argument.
        AUDIT_CASES="$1"
      else
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$PHASE1_DIR" ]]; then
  echo "Usage: $0 [--text-only] [--print-config] [--audit-cases PATH] PHASE1_DIR" >&2
  exit 1
fi

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .e0.server.env}"
: "${MEMGEN_E0_MAX_PAYLOAD_TOKENS:?Freeze MEMGEN_E0_MAX_PAYLOAD_TOKENS in .e0.server.env}"
if [[ ! "$MEMGEN_E0_MAX_PAYLOAD_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MEMGEN_E0_MAX_PAYLOAD_TOKENS must be a positive integer" >&2
  exit 1
fi

APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
VERIFIED_EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
if [[ ! -s "$APPROVED_BANK" || ! -s "$VERIFIED_EXPERIENCES" || ! -s "$SPLIT_MANIFEST" ]]; then
  echo "Phase-1 approved bank, verified experiences, or split manifest is missing under $PHASE1_DIR" >&2
  exit 1
fi

PHASE1_METADATA="$(
  python - "$APPROVED_BANK" "$SPLIT_MANIFEST" <<'PY'
import json
from pathlib import Path
import sys

approved_path = Path(sys.argv[1])
split_manifest_path = Path(sys.argv[2])
with approved_path.open("r", encoding="utf-8") as handle:
    first_record = next(
        (json.loads(line) for line in handle if line.strip()),
        None,
    )
if not isinstance(first_record, dict):
    raise SystemExit("Approved Phase-1 bank is empty")
student = first_record.get("student")
if not isinstance(student, dict):
    raise SystemExit("Approved Phase-1 bank has no frozen student metadata")
manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
dataset = manifest.get("dataset")
if not isinstance(dataset, dict):
    raise SystemExit("Phase-1 split manifest has no dataset metadata")
values = (
    student.get("model_name"),
    student.get("model_revision"),
    student.get("tokenizer_revision"),
    dataset.get("revision"),
)
if any(not isinstance(value, str) or not value for value in values):
    raise SystemExit("Phase-1 model/tokenizer/dataset revision metadata is incomplete")
print("\t".join(values))
PY
)"
IFS=$'\t' read -r MODEL MODEL_REVISION TOKENIZER_REVISION DATASET_REVISION <<< "$PHASE1_METADATA"
if [[ -z "$MODEL" || -z "$MODEL_REVISION" || -z "$TOKENIZER_REVISION" || -z "$DATASET_REVISION" ]]; then
  echo "Failed to resolve frozen Phase-1 model/tokenizer/dataset metadata" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${MEMGEN_E0_CUDA_VISIBLE_DEVICES:-0}"
DTYPE="bfloat16"
DEVICE="cuda"
AUDIT_CASE_COUNT=8
AUDIT_MAX_NEW_TOKENS=128

if [[ "$PRINT_CONFIG" == "1" ]]; then
  printf 'phase1_dir=%s\n' "$PHASE1_DIR"
  printf 'output_root=%s\n' "$MEMGEN_OUTPUT_ROOT"
  printf 'model=%s\n' "$MODEL"
  printf 'model_revision=%s\n' "$MODEL_REVISION"
  printf 'tokenizer_revision=%s\n' "$TOKENIZER_REVISION"
  printf 'dataset_revision=%s\n' "$DATASET_REVISION"
  printf 'max_payload_tokens=%s\n' "$MEMGEN_E0_MAX_PAYLOAD_TOKENS"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'dtype=%s\n' "$DTYPE"
  printf 'text_only=%s\n' "$TEXT_ONLY"
  exit 0
fi

RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="gsm8k_e0_experience-memory_layer24_budget${MEMGEN_E0_MAX_PAYLOAD_TOKENS}_${RUN_TAG}"
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e0/gsm8k/$RUN_ID"
mkdir -p "$RUN_DIR"

COMPILE_ARGS=(
  --approved-bank "$APPROVED_BANK"
  --verified-experiences "$VERIFIED_EXPERIENCES"
  --output-dir "$RUN_DIR"
  --model "$MODEL"
  --model-revision "$MODEL_REVISION"
  --tokenizer-revision "$TOKENIZER_REVISION"
  --max-payload-tokens "$MEMGEN_E0_MAX_PAYLOAD_TOKENS"
  --layer 24
  --dtype "$DTYPE"
  --device "$DEVICE"
)
if [[ "$TEXT_ONLY" == "1" ]]; then
  COMPILE_ARGS+=(--text-only)
fi
python scripts/compile_experience_memory_bank.py "${COMPILE_ARGS[@]}"

if [[ "$TEXT_ONLY" != "1" && -z "$AUDIT_CASES" ]]; then
  AUDIT_CASES="$RUN_DIR/calibration_audit_cases.jsonl"
  python scripts/build_side_kv_audit_cases.py \
    --split-manifest "$SPLIT_MANIFEST" \
    --memory-records "$RUN_DIR/memory_records.v1.jsonl" \
    --output "$AUDIT_CASES" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --tokenizer-revision "$TOKENIZER_REVISION" \
    --dataset-revision "$DATASET_REVISION" \
    --logical-split calibration-val \
    --case-count "$AUDIT_CASE_COUNT" \
    --max-new-tokens "$AUDIT_MAX_NEW_TOKENS" \
    --dtype "$DTYPE" \
    --device "$DEVICE"
fi

if [[ "$TEXT_ONLY" != "1" ]]; then
  if [[ ! -s "$AUDIT_CASES" ]]; then
    echo "Side-KV audit cases file is missing or empty: $AUDIT_CASES" >&2
    exit 1
  fi
  python scripts/audit_side_kv_mechanism.py \
    --side-kv-manifest "$RUN_DIR/side_kv_manifest.json" \
    --cases "$AUDIT_CASES" \
    --output-dir "$RUN_DIR" \
    --model "$MODEL" \
    --model-revision "$MODEL_REVISION" \
    --tokenizer-revision "$TOKENIZER_REVISION" \
    --layer 24 \
    --dtype "$DTYPE" \
    --device "$DEVICE"
fi

echo "E0 artifacts: $RUN_DIR"
