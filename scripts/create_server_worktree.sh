#!/usr/bin/env bash
set -euo pipefail

# Create an immutable server-side checkout for one Git commit.  Run this from
# a clean clone of MemGen, never from a worktree that is currently training.
#
# Usage:
#   bash scripts/create_server_worktree.sh /mnt/worktrees/memgen origin/feature/my-method

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <worktree-root> <git-ref>" >&2
  exit 2
fi

WORKTREE_ROOT=$1
GIT_REF=$2
REPO_ROOT=$(git rev-parse --show-toplevel)

git -C "$REPO_ROOT" fetch origin --tags --prune
COMMIT=$(git -C "$REPO_ROOT" rev-parse "${GIT_REF}^{commit}")
SHORT_SHA=$(git -C "$REPO_ROOT" rev-parse --short "$COMMIT")
STAMP=$(date +%Y%m%d-%H%M%S)
TARGET_DIR="${WORKTREE_ROOT}/memgen-${STAMP}-${SHORT_SHA}"

if [ -e "$TARGET_DIR" ]; then
  echo "Refusing to overwrite existing directory: $TARGET_DIR" >&2
  exit 1
fi

mkdir -p "$WORKTREE_ROOT"
git -C "$REPO_ROOT" worktree add --detach "$TARGET_DIR" "$COMMIT"
echo "Created worktree: $TARGET_DIR"
echo "Commit: $COMMIT"
