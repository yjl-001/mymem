#!/usr/bin/env bash
# V4.3 is the default complete experiment; legacy recovery is explicit.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset GLM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY || true
if [[ "${1:-}" == "legacy" ]]; then
  unset DEEPSEEK_API_KEY || true
  shift
  exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_2_recovered_legacy.sh" "$@"
fi
exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_3_unified_bank_experiment.sh" "$@"
