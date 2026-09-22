#!/usr/bin/env bash
# Two-phase local Bank entry point; pass --phase rollouts or --phase bank.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
exec "${MEMGEN_PYTHON_BIN:-python}" scripts/build_local_memory_bank.py "$@"
