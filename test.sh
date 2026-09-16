#!/usr/bin/env bash
# V4.3 is the default complete experiment; legacy recovery is explicit.
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
unset GLM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY || true
if [[ "${1:-}" == "similarity-study" ]]; then
  shift
  exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_3_similarity_study.sh" "$@"
fi
if [[ "${1:-}" == "final-test" ]]; then
  shift
  exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_3_final_test.sh" "$@"
fi
if [[ "${1:-}" == "gated-prefix" ]]; then
  shift
  exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_3_gated_prefix.sh" "$@"
fi
if [[ "${1:-}" == "timing" ]]; then
  shift
  exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_3_memory_timing.sh" "$@"
fi
if [[ "${1:-}" == "selector" ]]; then
  shift
  exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_3_question_selector.sh" "$@"
fi
if [[ "${1:-}" == "equivalence" ]]; then
  shift
  exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_3_prefix_equivalence.sh" "$@"
fi
if [[ "${1:-}" == "legacy" ]]; then
  unset DEEPSEEK_API_KEY || true
  shift
  exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_2_recovered_legacy.sh" "$@"
fi
exec bash "$REPO_ROOT/scripts/experiments/gsm8k/run_v4_3_unified_bank_experiment.sh" "$@"
