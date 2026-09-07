#!/usr/bin/env bash
# Rebuild and seal one explicitly named GSM8K Phase-1 + V3.4 risk lineage.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${MEMGEN_PYTHON_BIN:-python}"
STAGE="all"
ALLOW_PAID_PHASE1="false"
BANK_MANIFEST=""
SIDE_KV_MANIFEST=""
POSITIONAL=()

usage() {
  cat <<'EOF'
Usage:
  run_phase1_risk_lineage.sh [options] LINEAGE_ID OUTPUT_ROOT

Options:
  --stage phase1|risk|manifest|all
  --allow-paid-phase1
  --bank-manifest PATH       Optional current V4 bank compatibility check.
  --side-kv-manifest PATH    Required together with --bank-manifest.

Canonical outputs:
  OUTPUT_ROOT/lineages/gsm8k/LINEAGE_ID/phase1/
  OUTPUT_ROOT/lineages/gsm8k/LINEAGE_ID/risk_v3_4/
  OUTPUT_ROOT/lineages/gsm8k/LINEAGE_ID/phase1_risk_lineage_manifest.json
  OUTPUT_ROOT/lineages/gsm8k/LINEAGE_ID/USE_THIS_LINEAGE.env

Phase-1 uses the existing verifier-backed rollout + DeepSeek teacher/reviewer
pipeline.  The paid stages are never entered unless --allow-paid-phase1 is
explicitly present.  This script never starts V4 bank construction, selector,
dev-test, final-test, source-state extraction, or oracle evaluation.

A newly generated Phase-1 will normally be incompatible with an older V4 bank
because that bank binds exact Phase-1 file hashes.  Supply both V4 manifests to
record an explicit compatibility result in the sealed lineage manifest.
EOF
}

fail() {
  echo "[phase1-risk-lineage] FAIL: $*" >&2
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --stage)
      [[ "$#" -ge 2 ]] || fail "--stage requires a value"
      STAGE="$2"
      shift 2
      ;;
    --allow-paid-phase1)
      ALLOW_PAID_PHASE1="true"
      shift
      ;;
    --bank-manifest)
      [[ "$#" -ge 2 ]] || fail "--bank-manifest requires a path"
      BANK_MANIFEST="$2"
      shift 2
      ;;
    --side-kv-manifest)
      [[ "$#" -ge 2 ]] || fail "--side-kv-manifest requires a path"
      SIDE_KV_MANIFEST="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      fail "unknown option: $1"
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

case "$STAGE" in
  phase1|risk|manifest|all) ;;
  *) fail "--stage must be phase1, risk, manifest, or all" ;;
esac
[[ "${#POSITIONAL[@]}" -eq 2 ]] || {
  usage >&2
  exit 2
}
if [[ -n "$BANK_MANIFEST" || -n "$SIDE_KV_MANIFEST" ]]; then
  [[ -n "$BANK_MANIFEST" && -n "$SIDE_KV_MANIFEST" ]] \
    || fail "--bank-manifest and --side-kv-manifest must be supplied together"
fi

LINEAGE_ID="${POSITIONAL[0]}"
OUTPUT_ROOT="${POSITIONAL[1]%/}"
[[ "$LINEAGE_ID" =~ ^[a-z0-9][a-z0-9._-]{2,127}$ ]] \
  || fail "LINEAGE_ID must be 3-128 lowercase letters, digits, '.', '_' or '-'"
[[ -n "$OUTPUT_ROOT" && "$OUTPUT_ROOT" != "/" ]] \
  || fail "OUTPUT_ROOT must be a specific non-root directory"

LINEAGE_ROOT="$OUTPUT_ROOT/lineages/gsm8k/$LINEAGE_ID"
PHASE1_DIR="$LINEAGE_ROOT/phase1"
RISK_DIR="$LINEAGE_ROOT/risk_v3_4"
LINEAGE_MANIFEST="$LINEAGE_ROOT/phase1_risk_lineage_manifest.json"
LINEAGE_ENV="$LINEAGE_ROOT/USE_THIS_LINEAGE.env"

command -v "$PYTHON_BIN" >/dev/null 2>&1 \
  || fail "Python executable not found: $PYTHON_BIN"
if [[ -s "$LINEAGE_MANIFEST" && "$STAGE" != "manifest" ]]; then
  fail "lineage is already sealed and immutable: $LINEAGE_MANIFEST"
fi

echo "[phase1-risk-lineage] repo_revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[phase1-risk-lineage] stage=$STAGE lineage_id=$LINEAGE_ID"
echo "[phase1-risk-lineage] phase1_dir=$PHASE1_DIR"
echo "[phase1-risk-lineage] risk_dir=$RISK_DIR"
echo "[phase1-risk-lineage] manifest=$LINEAGE_MANIFEST"

if [[ "$STAGE" == "phase1" || "$STAGE" == "all" ]]; then
  [[ "$ALLOW_PAID_PHASE1" == "true" ]] \
    || fail "Phase-1 includes paid teacher/reviewer calls; rerun with --allow-paid-phase1"
  mkdir -p "$PHASE1_DIR"
  MEMGEN_PHASE1_OUTPUT_DIR="$PHASE1_DIR" \
  MEMGEN_RUN_TAG="$LINEAGE_ID" \
    bash scripts/experiments/gsm8k/run_phase1_verified_bank.sh
fi

if [[ "$STAGE" == "risk" || "$STAGE" == "all" ]]; then
  [[ -s "$PHASE1_DIR/verified_experiences.jsonl" ]] \
    || fail "missing Phase-1 experiences: $PHASE1_DIR"
  [[ -s "$PHASE1_DIR/ai_approved_bank_records.jsonl" ]] \
    || fail "missing Phase-1 approved bank: $PHASE1_DIR"
  mkdir -p "$RISK_DIR"
  bash scripts/experiments/gsm8k/run_v3_4_token_risk_gate.sh \
    "$PHASE1_DIR" "$RISK_DIR"
fi

if [[ "$STAGE" == "manifest" || "$STAGE" == "all" ]]; then
  MANIFEST_ARGS=()
  if [[ -n "$BANK_MANIFEST" ]]; then
    MANIFEST_ARGS+=(
      --bank-manifest "$BANK_MANIFEST"
      --side-kv-manifest "$SIDE_KV_MANIFEST"
    )
  fi
  "$PYTHON_BIN" scripts/build_phase1_risk_lineage_manifest.py \
    --lineage-id "$LINEAGE_ID" \
    --lineage-root "$LINEAGE_ROOT" \
    --phase1-dir "$PHASE1_DIR" \
    --risk-dir "$RISK_DIR" \
    --output "$LINEAGE_MANIFEST" \
    --environment-output "$LINEAGE_ENV" \
    "${MANIFEST_ARGS[@]}"
fi

echo "[phase1-risk-lineage] PASS stage=$STAGE"
echo "[phase1-risk-lineage] canonical_phase1=$PHASE1_DIR"
echo "[phase1-risk-lineage] canonical_risk=$RISK_DIR/token-entropy-risk-gate-v3.4.pt"
if [[ -s "$LINEAGE_ENV" ]]; then
  echo "[phase1-risk-lineage] next: source $LINEAGE_ENV"
fi
