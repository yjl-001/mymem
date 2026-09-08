#!/usr/bin/env bash
# Complete V4.3 construction, compilation and four-layer mechanism audits.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
unset DEEPSEEK_API_KEY GLM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY || true

PYTHON_BIN="${MEMGEN_PYTHON_BIN:-python}"
OUTPUT_ROOT="${MEMGEN_OUTPUT_ROOT:-/data/memgen-runs}"
V4_ROOT="${MEMGEN_V4_OUTPUT_ROOT:-$OUTPUT_ROOT/v4}"
RECOVERY_ID="${MEMGEN_V4_RECOVERY_ID:-gsm8k-v4-packet-replay-20260907-r1}"
LINEAGE_ROOT="${MEMGEN_V4_RECOVERY_LINEAGE_ROOT:-$OUTPUT_ROOT/lineages/gsm8k-recovery/$RECOVERY_ID}"
PACKETS="${MEMGEN_V4_SEMANTIC_PACKETS:-$V4_ROOT/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl}"
CURATED="${MEMGEN_V4_CURATED_BANK_DIR:-$V4_ROOT/offline/construction_v4_2_local_curated}"
LEGACY_SIDE_KV="${MEMGEN_V4_SIDE_KV_DIR:-$V4_ROOT/offline/side_kv_v4_2_local_curated}"
BANK_DIR="${MEMGEN_V43_BANK_DIR:-$V4_ROOT/offline/construction_v4_3_unified}"
SIDE_DIR="${MEMGEN_V43_SIDE_KV_DIR:-$V4_ROOT/offline/side_kv_v4_3_unified}"
CACHE_MANIFEST="${MEMGEN_V43_CACHE_MANIFEST:-$LINEAGE_ROOT/v4_oracle_full/source_state_cache/v4_source_state_manifest.json}"
RISK_ARTIFACT="${MEMGEN_V43_RISK_ARTIFACT:-$LINEAGE_ROOT/risk_v3_4/token-entropy-risk-gate-v3.4.pt}"
AUDIT_ROOT="${MEMGEN_V43_AUDIT_ROOT:-$V4_ROOT/offline/v4_3_unified_audit}"
POLICY="${MEMGEN_V43_CURATION_POLICY:-$REPO_ROOT/configs/experiments/gsm8k/v4_2_local_curation_policy.json}"
DEVICE="${MEMGEN_V43_DEVICE:-cuda}"
MODE="${1:-all}"
STAGE="preflight"

usage() {
  cat <<'EOF'
Usage: ./test.sh [smoke|full|all]

V4.3 unified Bank experiment. Default: all (smoke, then full).
  smoke  Construct/authenticate Banks; compile/reuse both tiers; run four-layer smoke.
  full   Run the complete audit using a passed, authenticated smoke from this experiment.
  all    Construct/compile prerequisites, run smoke, then full only if smoke passes.

Always reuses the complete 116-sample V4.2 source-state cache, even for smoke.
Never invokes Phase-1 recovery, paid providers, selector, dev-test or final-test.
Missing source cache/risk fails with its exact path; no automatic regeneration.
Construction qualification failures remain explicit exclusions; thresholds stay fixed.

Overrides:
  MEMGEN_OUTPUT_ROOT, MEMGEN_V4_OUTPUT_ROOT, MEMGEN_V4_RECOVERY_LINEAGE_ROOT
  MEMGEN_V4_SEMANTIC_PACKETS, MEMGEN_V4_CURATED_BANK_DIR, MEMGEN_V4_SIDE_KV_DIR
  MEMGEN_V43_BANK_DIR, MEMGEN_V43_SIDE_KV_DIR, MEMGEN_V43_AUDIT_ROOT
  MEMGEN_V43_CACHE_MANIFEST, MEMGEN_V43_RISK_ARTIFACT, MEMGEN_V43_CURATION_POLICY
  MEMGEN_V43_DEVICE=cuda, MEMGEN_V43_CUDA_VISIBLE_DEVICES=0
  MEMGEN_V43_ALL_BANK_SWEEP=1    Append the optional outcome-informed primary-Bank sweep.
  MEMGEN_V43_VALIDATE_ONLY=1    Authenticate existing complete outputs without inference.
  MEMGEN_PYTHON_BIN=python

Resume is always enabled and authenticates inputs, code and existing per-case files.
Changed inputs/implementation require a new output directory, never a forced overwrite.
EOF
}

fail() { echo "[v4.3] FAIL stage=$STAGE: $*" >&2; exit 1; }
trap 'status=$?; echo "[v4.3] FAIL stage=$STAGE line=$LINENO status=$status; valid artifacts preserved" >&2; exit "$status"' ERR
case "$MODE" in
  -h|--help) usage; exit 0 ;;
  smoke|full|all) ;;
  *) usage >&2; fail "expected smoke, full, or all" ;;
esac
[[ "$#" -le 1 ]] || fail "only one mode argument is supported"
command -v "$PYTHON_BIN" >/dev/null || fail "Python executable missing: $PYTHON_BIN"
for path in "$PACKETS" "$CURATED/bank_records.jsonl" "$CURATED/bank_manifest.json" \
  "$LEGACY_SIDE_KV/v4_side_kv_manifest.json" "$CACHE_MANIFEST" "$RISK_ARTIFACT" "$POLICY"; do
  [[ -s "$path" ]] || fail "missing required existing input: $path"
done
if [[ "$MODE" == "full" ]]; then
  [[ -s "$AUDIT_ROOT/smoke/v4_3_audit_report.json" ]] || fail "run ./test.sh smoke or ./test.sh all before full"
fi
export CUDA_VISIBLE_DEVICES="${MEMGEN_V43_CUDA_VISIBLE_DEVICES:-${MEMGEN_V4_CUDA_VISIBLE_DEVICES:-0}}"
RESUME=(--resume)
if [[ "${MEMGEN_V43_VALIDATE_ONLY:-0}" == "1" ]]; then RESUME=(--validate-only); fi

echo "[v4.3] repo_revision=$(git rev-parse HEAD) mode=$MODE device=$DEVICE"
echo "[v4.3] packets=$PACKETS curated=$CURATED policy=$POLICY"
echo "[v4.3] source_cache=$CACHE_MANIFEST risk=$RISK_ARTIFACT"
echo "[v4.3] bank_output=$BANK_DIR side_kv_output=$SIDE_DIR audit_output=$AUDIT_ROOT"
echo "[v4.3] one_bank_one_memory=true active_steps=32 completion_tokens=1024 selector=false"

STAGE="construction"
echo "[v4.3] stage=$STAGE"
"$PYTHON_BIN" scripts/build_v4_3_unified_bank.py \
  --source-dir "$CURATED" --semantic-packets "$PACKETS" --curation-policy "$POLICY" \
  --output-dir "$BANK_DIR" "${RESUME[@]}"

STAGE="compilation"
echo "[v4.3] stage=$STAGE"
"$PYTHON_BIN" scripts/compile_v4_3_side_kv.py \
  --bank-dir "$BANK_DIR" --reasoner-manifest "$LEGACY_SIDE_KV/v4_side_kv_manifest.json" \
  --output-dir "$SIDE_DIR" --device "$DEVICE" "${RESUME[@]}"

run_mode() {
  local selection="$1"
  STAGE="audit-$selection"
  echo "[v4.3] stage=$STAGE"
  local audit_command=("$PYTHON_BIN" scripts/audit_v4_3_unified_memory.py \
    --mode "$selection" --bank-dir "$BANK_DIR" --side-kv-dir "$SIDE_DIR" \
    --semantic-packets "$PACKETS" --cache-manifest "$CACHE_MANIFEST" \
    --token-risk-artifact "$RISK_ARTIFACT" --output-dir "$AUDIT_ROOT/$selection" \
    --device "$DEVICE" "${RESUME[@]}")
  if [[ "${MEMGEN_V43_ALL_BANK_SWEEP:-0}" == "1" ]]; then audit_command+=(--all-bank-sweep); fi
  if [[ "$selection" == "full" ]]; then audit_command+=(--smoke-report "$AUDIT_ROOT/smoke/v4_3_audit_report.json"); fi
  "${audit_command[@]}"
  "$PYTHON_BIN" -m json.tool "$AUDIT_ROOT/$selection/v4_3_core_summary.json"
  echo "[v4.3] PASS stage=$STAGE report=$AUDIT_ROOT/$selection/v4_3_audit_report.json"
}
case "$MODE" in
  smoke) run_mode smoke ;;
  full) run_mode full ;;
  all) run_mode smoke; run_mode full ;;
esac
STAGE="complete"
echo "[v4.3] PASS mode=$MODE"
