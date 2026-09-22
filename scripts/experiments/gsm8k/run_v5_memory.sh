#!/usr/bin/env bash
# V5 two-phase entry point: --phase rollouts, then --phase bank --rollout-source ...
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
exec "${MEMGEN_PYTHON_BIN:-python}" scripts/build_v5_memory.py "$@"
