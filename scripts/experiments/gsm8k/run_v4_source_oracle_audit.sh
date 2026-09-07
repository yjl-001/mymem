#!/usr/bin/env bash
# V4.2 offline source-state cache, CPU state audit, and oracle causal audit.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${MEMGEN_PYTHON_BIN:-python}"
DEVICE="${MEMGEN_V4_DEVICE:-cuda}"
DTYPE="${MEMGEN_V4_DTYPE:-bfloat16}"
MODE="smoke"
STAGE="all"
SMOKE_LIMIT="${MEMGEN_V4_ORACLE_SMOKE_LIMIT:-8}"
POSITIONAL=()

usage() {
  cat <<'EOF'
Usage:
  run_v4_source_oracle_audit.sh [--mode smoke|full] [--stage cache|state-audit|oracle|all] \
    PHASE1_DIR CURATED_BANK_DIR SIDE_KV_DIR TOKEN_RISK_ARTIFACT OUTPUT_ROOT

Inputs:
  PHASE1_DIR         verified_experiences.jsonl + split_manifest.json
  CURATED_BANK_DIR  bank_records.jsonl + bank_manifest.json (17 curated banks)
  SIDE_KV_DIR       v4_side_kv_manifest.json (17 target/reference pairs)
  TOKEN_RISK        qualified token-entropy-risk-gate-v3.4.pt

Outputs under OUTPUT_ROOT/offline/v4_oracle_<mode>/:
  source_state_cache/   safetensors + event JSONL + reachability manifest/report
  source_state_audit/   CPU-only window/normalization/LOO/hubness diagnostics
  oracle_audit/         exact-prefix baseline/target/reference cases and report

Smoke mode deterministically caps cache samples and oracle cases. Full mode
extracts all curated construction samples and all gate attempts. This runner
never starts selector compilation, dev-test, final-test, or online evaluation.
EOF
}

fail() {
  echo "[v4-source-oracle] FAIL: $*" >&2
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --mode)
      [[ "$#" -ge 2 ]] || fail "--mode requires a value"
      MODE="$2"
      shift 2
      ;;
    --stage)
      [[ "$#" -ge 2 ]] || fail "--stage requires a value"
      STAGE="$2"
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

case "$MODE" in
  smoke|full) ;;
  *) fail "--mode must be smoke or full" ;;
esac
case "$STAGE" in
  cache|state-audit|oracle|all) ;;
  *) fail "--stage must be cache, state-audit, oracle, or all" ;;
esac
[[ "${#POSITIONAL[@]}" -eq 5 ]] || {
  usage >&2
  exit 2
}

PHASE1_DIR="${POSITIONAL[0]}"
CURATED_BANK_DIR="${POSITIONAL[1]}"
SIDE_KV_DIR="${POSITIONAL[2]}"
TOKEN_RISK_ARTIFACT="${POSITIONAL[3]}"
OUTPUT_ROOT="${POSITIONAL[4]}"
RUN_ROOT="${MEMGEN_V4_ORACLE_RUN_ROOT:-$OUTPUT_ROOT/offline/v4_oracle_${MODE}}"
SOURCE_STATE_DIR="${MEMGEN_V4_SOURCE_STATE_DIR:-$RUN_ROOT/source_state_cache}"
STATE_AUDIT_DIR="${MEMGEN_V4_SOURCE_STATE_AUDIT_DIR:-$RUN_ROOT/source_state_audit}"
ORACLE_AUDIT_DIR="${MEMGEN_V4_ORACLE_AUDIT_DIR:-$RUN_ROOT/oracle_audit}"

EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
BANK_RECORDS="$CURATED_BANK_DIR/bank_records.jsonl"
BANK_MANIFEST="$CURATED_BANK_DIR/bank_manifest.json"
SIDE_KV_MANIFEST="$SIDE_KV_DIR/v4_side_kv_manifest.json"
CACHE_MANIFEST="$SOURCE_STATE_DIR/v4_source_state_manifest.json"

command -v "$PYTHON_BIN" >/dev/null 2>&1 \
  || fail "Python executable not found: $PYTHON_BIN"
command -v jq >/dev/null 2>&1 || fail "jq is required"
if ! [[ "$SMOKE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  fail "MEMGEN_V4_ORACLE_SMOKE_LIMIT must be a positive integer"
fi

for required in \
  "$EXPERIENCES" \
  "$SPLIT_MANIFEST" \
  "$BANK_RECORDS" \
  "$BANK_MANIFEST" \
  "$SIDE_KV_MANIFEST" \
  "$TOKEN_RISK_ARTIFACT"; do
  [[ -s "$required" ]] || fail "missing required input: $required"
done

export CUDA_VISIBLE_DEVICES="${MEMGEN_V4_CUDA_VISIBLE_DEVICES:-0}"
# This route is local/model-inference only. No paid provider key is inherited.
unset DEEPSEEK_API_KEY GLM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY || true

mkdir -p "$RUN_ROOT"
echo "[v4-source-oracle] repo_revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[v4-source-oracle] mode=$MODE stage=$STAGE device=$DEVICE dtype=$DTYPE"
echo "[v4-source-oracle] phase1_dir=$PHASE1_DIR"
echo "[v4-source-oracle] curated_bank_dir=$CURATED_BANK_DIR"
echo "[v4-source-oracle] side_kv_dir=$SIDE_KV_DIR"
echo "[v4-source-oracle] token_risk_artifact=$TOKEN_RISK_ARTIFACT"
echo "[v4-source-oracle] source_state_dir=$SOURCE_STATE_DIR"
echo "[v4-source-oracle] state_audit_dir=$STATE_AUDIT_DIR"
echo "[v4-source-oracle] oracle_audit_dir=$ORACLE_AUDIT_DIR"

if [[ "$STAGE" == "cache" || "$STAGE" == "all" ]]; then
  mkdir -p "$SOURCE_STATE_DIR"
  CACHE_LIMIT_ARGS=()
  if [[ "$MODE" == "smoke" ]]; then
    CACHE_LIMIT_ARGS=(--limit "$SMOKE_LIMIT")
  fi
  "$PYTHON_BIN" scripts/extract_v4_source_state_cache.py \
    --experiences "$EXPERIENCES" \
    --split-manifest "$SPLIT_MANIFEST" \
    --bank-records "$BANK_RECORDS" \
    --bank-manifest "$BANK_MANIFEST" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --token-risk-artifact "$TOKEN_RISK_ARTIFACT" \
    --output-dir "$SOURCE_STATE_DIR" \
    --dataset-revision main \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    "${CACHE_LIMIT_ARGS[@]}" \
    2>&1 | tee "$SOURCE_STATE_DIR/v4_source_state_extract.log"
  jq -e '
    .status == "source_state_cache_built"
    and .offline_only == true
    and .qualified_for_online_use == false
    and .contains_reward_or_answer_signal == false
    and .configuration.layer_number == 24
    and .configuration.maximum_gate_attempts == 3
    and .configuration.maximum_hidden_window == 32
    and .configuration.support_unit == "independent_sample"
    and .artifacts.tensors.path == "v4_source_states.safetensors"
  ' "$CACHE_MANIFEST" >/dev/null \
    || fail "source-state cache manifest validation failed"
  if [[ "$MODE" == "full" ]]; then
    jq -e '
      .configuration.extraction_scope == "full_curated_construction"
      and .counts.bank_count == 17
      and .counts.independent_sample_count == 116
      and .counts.failure_gate_unreachable_independent_sample_count == 50
      and .configuration.expected_full_construction_count == 116
      and .configuration.extracted_construction_count == 116
      and .configuration.extracted_construction_count
        == .configuration.expected_full_construction_count
    ' "$CACHE_MANIFEST" >/dev/null \
      || fail "full source-state extraction coverage failed"
  fi
fi

if [[ "$STAGE" == "state-audit" || "$STAGE" == "all" ]]; then
  [[ -s "$CACHE_MANIFEST" ]] || fail "missing source-state cache: $CACHE_MANIFEST"
  mkdir -p "$STATE_AUDIT_DIR"
  "$PYTHON_BIN" scripts/audit_v4_source_state_cache.py \
    --cache-manifest "$CACHE_MANIFEST" \
    --output-dir "$STATE_AUDIT_DIR" \
    --windows 1,4,8,16,32 \
    --alphas 0,0.25,0.5,0.75,1 \
    2>&1 | tee "$STATE_AUDIT_DIR/v4_source_state_cpu_audit.log"
  jq -e '
    .status == "completed_offline_diagnostic"
    and .offline_only == true
    and .qualified_for_online_use == false
    and .online_artifacts_generated == false
    and .reasoner_loaded == false
    and .multiple_attempts_are_observations_not_support == true
    and .artifacts.online_selector_tensor == null
    and .artifacts.online_selector_manifest == null
  ' "$STATE_AUDIT_DIR/v4_source_state_cpu_audit_report.json" >/dev/null \
    || fail "CPU source-state audit validation failed"
fi

if [[ "$STAGE" == "oracle" || "$STAGE" == "all" ]]; then
  [[ -s "$CACHE_MANIFEST" ]] || fail "missing source-state cache: $CACHE_MANIFEST"
  mkdir -p "$ORACLE_AUDIT_DIR"
  ORACLE_LIMIT_ARGS=()
  if [[ "$MODE" == "smoke" ]]; then
    ORACLE_LIMIT_ARGS=(--limit-per-kind "$SMOKE_LIMIT")
  fi
  "$PYTHON_BIN" scripts/audit_v4_oracle_causal_utility.py \
    --cache-manifest "$CACHE_MANIFEST" \
    --experiences "$EXPERIENCES" \
    --split-manifest "$SPLIT_MANIFEST" \
    --bank-records "$BANK_RECORDS" \
    --bank-manifest "$BANK_MANIFEST" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --token-risk-artifact "$TOKEN_RISK_ARTIFACT" \
    --output-dir "$ORACLE_AUDIT_DIR" \
    --dataset-revision main \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --attempt-policy all \
    --resume \
    "${ORACLE_LIMIT_ARGS[@]}" \
    2>&1 | tee "$ORACLE_AUDIT_DIR/v4_oracle_audit.log"
  jq -e '
    .status == "completed_mechanism_diagnostic"
    and .complete == true
    and .offline_only == true
    and .qualified_for_online_use == false
    and .online_artifacts_generated == false
    and .held_out_generalization_claim == false
    and .gate_unreachable_counted_as_memory_ineffective == false
    and .artifacts.online_selector_tensor == null
    and .artifacts.online_selector_manifest == null
  ' "$ORACLE_AUDIT_DIR/v4_oracle_report.json" >/dev/null \
    || fail "oracle causal audit validation failed"
fi

echo "[v4-source-oracle] PASS mode=$MODE stage=$STAGE"
echo "[v4-source-oracle] source_state_cache=$SOURCE_STATE_DIR"
echo "[v4-source-oracle] source_state_audit=$STATE_AUDIT_DIR"
echo "[v4-source-oracle] oracle_audit=$ORACLE_AUDIT_DIR"
