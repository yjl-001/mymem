#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
unset DEEPSEEK_API_KEY GLM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY || true
OUTPUT_ROOT="${MEMGEN_OUTPUT_ROOT:-/data/memgen-runs}"
V4_ROOT="${MEMGEN_V4_OUTPUT_ROOT:-$OUTPUT_ROOT/v4}"
if [[ "${MEMGEN_V43_VALIDATE_ONLY:-0}" == "1" ]]; then set -- --validate-only "$@"; fi
exec "${MEMGEN_PYTHON_BIN:-python}" scripts/run_v4_3_retrieval_coverage.py \
  --selector-dir "${MEMGEN_V43_SELECTOR_ROOT:-$V4_ROOT/offline/v4_3_question_selector}" \
  --study-dir "${MEMGEN_V43_SIMILARITY_ROOT:-$V4_ROOT/offline/v4_3_similarity_study}" \
  --output-dir "${MEMGEN_V43_COVERAGE_ROOT:-$V4_ROOT/offline/v4_3_retrieval_coverage}" \
  "$@"
