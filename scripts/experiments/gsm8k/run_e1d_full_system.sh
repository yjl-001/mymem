#!/usr/bin/env bash
# Stable user-facing entry point for the complete experience-memory system.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_e1_experience_memory.sh" "$@"
