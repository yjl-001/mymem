#!/usr/bin/env bash
# One complete run, with an explicit config and immutable resumable output.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
exec "${MEMGEN_PYTHON_BIN:-python}" scripts/build_local_memory_bank.py "$@"
