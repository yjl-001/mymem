#!/usr/bin/env bash
# Collect GSM8K validation entropies and save the q=0.85 threshold.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_ENV="$REPO_ROOT/scripts/experiments/.server.env"
cd "$REPO_ROOT"

if [[ ! -f "$SERVER_ENV" ]]; then
  echo "Missing $SERVER_ENV. Copy scripts/experiments/server.env.example and fill it in." >&2
  exit 1
fi
source "$SERVER_ENV"

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .server.env}"
: "${MEMGEN_DEVICES:?MEMGEN_DEVICES must be set in .server.env}"
: "${MEMGEN_GSM8K_WEAVER_CKPT:?MEMGEN_GSM8K_WEAVER_CKPT must be set in .server.env}"

RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="gsm8k_entropy_gate_lastlayer_sink4_q85_calibration_${RUN_TAG}"

python scripts/launch_experiment.py eval \
  configs/experiments/gsm8k/entropy_calibration.yaml \
  --devices "$MEMGEN_DEVICES" \
  --run-id "$RUN_ID" \
  --set model.load_model_path="$MEMGEN_GSM8K_WEAVER_CKPT"

TRACE_PATH="$(find "$MEMGEN_OUTPUT_ROOT/evaluate/gsm8k/Qwen2.5-1.5B-Instruct" \
  -type f \
  -path "*/${RUN_ID}__*/evaluate/entropy_gate_trace.csv" \
  -print -quit)"
if [[ -z "$TRACE_PATH" ]]; then
  echo "Could not locate entropy_gate_trace.csv for run $RUN_ID." >&2
  exit 1
fi

python scripts/calibrate_entropy_threshold.py "$TRACE_PATH" --quantile 0.85
echo "Calibration complete. Use this file for the formal test:"
echo "  ${TRACE_PATH%/entropy_gate_trace.csv}/entropy_threshold.json"
