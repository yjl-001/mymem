#!/usr/bin/env bash
# Evaluate the GSM8K q=0.85 entropy gate on the test split.
# Usage: bash scripts/experiments/gsm8k/run_entropy_eval_sink4_q85.sh \
#   /absolute/path/to/entropy_threshold.json
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/entropy_threshold.json" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_ENV="$REPO_ROOT/scripts/experiments/.server.env"
CALIBRATION_JSON="$1"
cd "$REPO_ROOT"

if [[ ! -f "$SERVER_ENV" ]]; then
  echo "Missing $SERVER_ENV. Copy scripts/experiments/server.env.example and fill it in." >&2
  exit 1
fi
if [[ ! -f "$CALIBRATION_JSON" ]]; then
  echo "Calibration result not found: $CALIBRATION_JSON" >&2
  exit 1
fi
source "$SERVER_ENV"

: "${MEMGEN_OUTPUT_ROOT:?MEMGEN_OUTPUT_ROOT must be set in .server.env}"
: "${MEMGEN_DEVICES:?MEMGEN_DEVICES must be set in .server.env}"
: "${MEMGEN_GSM8K_WEAVER_CKPT:?MEMGEN_GSM8K_WEAVER_CKPT must be set in .server.env}"

ENTROPY_THRESHOLD="$(python -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["entropy_threshold"])' "$CALIBRATION_JSON")"
RUN_TAG="${MEMGEN_RUN_TAG:-$(date +%Y%m%d-%H%M%S)}"
RUN_ID="gsm8k_entropy_gate_lastlayer_sink4_q85_test_${RUN_TAG}"

python scripts/launch_experiment.py eval \
  configs/experiments/gsm8k/entropy_gate_eval.yaml \
  --devices "$MEMGEN_DEVICES" \
  --run-id "$RUN_ID" \
  --set model.load_model_path="$MEMGEN_GSM8K_WEAVER_CKPT" \
  --set model.weaver.insertion_strategy.entropy_threshold="$ENTROPY_THRESHOLD"
