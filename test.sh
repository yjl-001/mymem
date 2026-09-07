#!/usr/bin/env bash
# Run the current V4 recovered-source oracle workflow on the server.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${MEMGEN_PYTHON_BIN:-python}"
OUTPUT_ROOT="${MEMGEN_OUTPUT_ROOT:-/data/memgen-runs}"
V4_OUTPUT_ROOT="${MEMGEN_V4_OUTPUT_ROOT:-$OUTPUT_ROOT/v4}"
RECOVERY_ID="${MEMGEN_V4_RECOVERY_ID:-gsm8k-v4-packet-replay-20260907-r1}"
SEMANTIC_PACKETS="${MEMGEN_V4_SEMANTIC_PACKETS:-$V4_OUTPUT_ROOT/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl}"
CURATED_BANK_DIR="${MEMGEN_V4_CURATED_BANK_DIR:-$V4_OUTPUT_ROOT/offline/construction_v4_2_local_curated}"
SIDE_KV_DIR="${MEMGEN_V4_SIDE_KV_DIR:-$V4_OUTPUT_ROOT/offline/side_kv_v4_2_local_curated}"
STAGE_POLICY="${MEMGEN_V4_TEST_STAGE:-auto}"
RUN_SELECTION="${1:-all}"

RUNNER="$REPO_ROOT/scripts/experiments/gsm8k/run_v4_question_recovery.sh"
LINEAGE_ROOT="${MEMGEN_V4_RECOVERY_LINEAGE_ROOT:-$OUTPUT_ROOT/lineages/gsm8k-recovery/$RECOVERY_ID}"
RECOVERY_MANIFEST="$LINEAGE_ROOT/recovery/v4_question_recovery_manifest.json"
RISK_ARTIFACT="$LINEAGE_ROOT/risk_v3_4/token-entropy-risk-gate-v3.4.pt"
RISK_REPORT="$LINEAGE_ROOT/risk_v3_4/token_entropy_risk_report.json"
RISK_EVIDENCE="$LINEAGE_ROOT/risk_v3_4/token_entropy_risk_evidence.jsonl"
CURRENT_MODE="preflight"

usage() {
  cat <<'EOF'
Usage:
  ./test.sh [all|smoke|full]

With no arguments, the script runs smoke first and then full. It uses the
recovered GSM8K lineage and the current curated 17-bank Side-KV artifacts.

Default server locations:
  MEMGEN_OUTPUT_ROOT=/data/memgen-runs
  MEMGEN_V4_OUTPUT_ROOT=$MEMGEN_OUTPUT_ROOT/v4
  MEMGEN_V4_RECOVERY_ID=gsm8k-v4-packet-replay-20260907-r1

Optional overrides:
  MEMGEN_V4_SEMANTIC_PACKETS=/path/to/semantic_evidence_packets.jsonl
  MEMGEN_V4_CURATED_BANK_DIR=/path/to/construction_v4_2_local_curated
  MEMGEN_V4_SIDE_KV_DIR=/path/to/side_kv_v4_2_local_curated
  MEMGEN_V4_TEST_STAGE=auto|oracle|all|recover|risk|cache|state-audit
  MEMGEN_V4_DEVICE=cuda
  MEMGEN_V4_DTYPE=bfloat16
  MEMGEN_V4_CUDA_VISIBLE_DEVICES=0
  MEMGEN_V4_ORACLE_SMOKE_LIMIT=8
  MEMGEN_PYTHON_BIN=python

The default stage policy is auto. For each mode, it reuses authenticated
recovery/risk/cache/state-audit artifacts when they exist and runs only the
fixed full-answer oracle. If a prerequisite is absent, that mode runs
stage=all to build it. Set MEMGEN_V4_TEST_STAGE explicitly to override this.

No selector, dev-test, final-test, or paid external API is invoked.
EOF
}

fail() {
  echo "[v4-test] FAIL: $*" >&2
  exit 1
}

on_error() {
  local status="$?"
  echo "[v4-test] FAIL mode=$CURRENT_MODE line=$1 status=$status" >&2
  exit "$status"
}
trap 'on_error "$LINENO"' ERR

case "$RUN_SELECTION" in
  -h|--help)
    usage
    exit 0
    ;;
  all|smoke|full) ;;
  *)
    usage >&2
    fail "argument must be all, smoke, or full"
    ;;
esac

case "$STAGE_POLICY" in
  auto|oracle|all|recover|risk|cache|state-audit) ;;
  *) fail "MEMGEN_V4_TEST_STAGE must be auto, oracle, all, recover, risk, cache, or state-audit" ;;
esac

command -v "$PYTHON_BIN" >/dev/null 2>&1 \
  || fail "Python executable not found: $PYTHON_BIN"
command -v jq >/dev/null 2>&1 || fail "jq is required"
[[ -x "$RUNNER" ]] || fail "missing executable recovery runner: $RUNNER"

for required in \
  "$SEMANTIC_PACKETS" \
  "$CURATED_BANK_DIR/bank_records.jsonl" \
  "$CURATED_BANK_DIR/bank_manifest.json" \
  "$SIDE_KV_DIR/v4_side_kv_manifest.json"; do
  [[ -s "$required" ]] || fail "missing required input: $required"
done

export MEMGEN_PYTHON_BIN="$PYTHON_BIN"
export MEMGEN_V4_DEVICE="${MEMGEN_V4_DEVICE:-cuda}"
export MEMGEN_V4_DTYPE="${MEMGEN_V4_DTYPE:-bfloat16}"
export MEMGEN_V4_CUDA_VISIBLE_DEVICES="${MEMGEN_V4_CUDA_VISIBLE_DEVICES:-0}"
export MEMGEN_V4_ORACLE_SMOKE_LIMIT="${MEMGEN_V4_ORACLE_SMOKE_LIMIT:-8}"

# This workflow is local/model-inference only. Never let a child process inherit
# a paid-provider credential accidentally.
unset DEEPSEEK_API_KEY GLM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY || true

echo "[v4-test] repo_revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[v4-test] run_selection=$RUN_SELECTION stage_policy=$STAGE_POLICY"
echo "[v4-test] recovery_id=$RECOVERY_ID"
echo "[v4-test] output_root=$OUTPUT_ROOT"
echo "[v4-test] semantic_packets=$SEMANTIC_PACKETS"
echo "[v4-test] curated_bank_dir=$CURATED_BANK_DIR"
echo "[v4-test] side_kv_dir=$SIDE_KV_DIR"
echo "[v4-test] oracle_profile=memory_active_steps=32 completion_tokens=1024"

prerequisites_exist() {
  local mode="$1"
  local run_root="$LINEAGE_ROOT/v4_oracle_${mode}"
  local cache_manifest="$run_root/source_state_cache/v4_source_state_manifest.json"
  local state_audit_report="$run_root/source_state_audit/v4_source_state_cpu_audit_report.json"
  local required
  for required in \
    "$RECOVERY_MANIFEST" \
    "$RISK_ARTIFACT" \
    "$RISK_REPORT" \
    "$RISK_EVIDENCE" \
    "$cache_manifest" \
    "$state_audit_report"; do
    [[ -s "$required" ]] || return 1
  done
}

stage_for_mode() {
  local mode="$1"
  if [[ "$STAGE_POLICY" != "auto" ]]; then
    printf '%s\n' "$STAGE_POLICY"
  elif prerequisites_exist "$mode"; then
    printf '%s\n' "oracle"
  else
    printf '%s\n' "all"
  fi
}

write_core_summary() {
  local mode="$1"
  local report="$2"
  local summary="$3"
  local temporary
  temporary="$(mktemp "$(dirname -- "$summary")/.v4_oracle_core_summary.XXXXXX")"
  jq '
    def core_metrics:
      {
        case_count,
        independent_sample_count,
        baseline_accuracy,
        target_accuracy,
        reference_accuracy,
        baseline_independent_sample_macro_accuracy,
        target_independent_sample_macro_accuracy,
        reference_independent_sample_macro_accuracy,
        baseline_wrong_to_target_correct_count,
        baseline_wrong_to_target_correct_independent_sample_count,
        baseline_wrong_to_reference_correct_count,
        baseline_wrong_to_reference_correct_independent_sample_count,
        target_harm_count,
        target_harm_independent_sample_count,
        target_better_than_reference_count,
        target_better_than_reference_independent_sample_count,
        mean_target_minus_reference_reward,
        independent_sample_macro_target_minus_reference_reward,
        local_intervention_observability,
        final_outcome_observability,
        local_trajectory_divergence,
        final_trajectory_divergence,
        intervention_diagnostics
      };
    {
      schema_version: "memgen-v4-oracle-core-summary-v2",
      status,
      complete,
      expected_case_count,
      completed_case_count,
      gate_unreachable_failure_count,
      gate_unreachable_counted_as_memory_ineffective,
      configuration,
      failure_oracle: (.by_dimension.case_kind.failure_oracle | core_metrics),
      success_safety: (.by_dimension.case_kind.success_safety | core_metrics),
      primary: (.by_dimension.curation_tier.primary | core_metrics),
      conditional: (.by_dimension.curation_tier.conditional | core_metrics)
    }
  ' "$report" > "$temporary"
  mv "$temporary" "$summary"
  echo "[v4-test] $mode core summary:"
  jq . "$summary"
}

run_mode() {
  local mode="$1"
  local stage
  local run_root="$LINEAGE_ROOT/v4_oracle_${mode}"
  local oracle_dir="$run_root/oracle_audit_full_answer"
  local report="$oracle_dir/v4_oracle_report.json"
  local summary="$oracle_dir/v4_oracle_core_summary.json"

  CURRENT_MODE="$mode"
  stage="$(stage_for_mode "$mode")"
  echo "[v4-test] START mode=$mode stage=$stage"
  bash "$RUNNER" \
    --mode "$mode" \
    --stage "$stage" \
    "$RECOVERY_ID" \
    "$SEMANTIC_PACKETS" \
    "$CURATED_BANK_DIR" \
    "$SIDE_KV_DIR" \
    "$OUTPUT_ROOT"

  if [[ "$stage" == "oracle" || "$stage" == "all" ]]; then
    [[ -s "$report" ]] || fail "missing oracle report: $report"
    jq -e '
      .status == "completed_mechanism_diagnostic"
      and .complete == true
      and .completed_case_count == .expected_case_count
      and .offline_only == true
      and .qualified_for_online_use == false
      and .online_artifacts_generated == false
      and .held_out_generalization_claim == false
      and .gate_unreachable_counted_as_memory_ineffective == false
      and .configuration.maximum_completion_tokens == 1024
      and .configuration.local_intervention_observation_tokens == 32
      and .configuration.maximum_active_steps == 32
      and .configuration.post_memory_native_continuation == true
      and .configuration.generation_stop_policy
        == "completed_boxed_answer_or_eos_or_completion_budget"
      and .local_and_final_metrics_separated == true
      and .question_recovery.external_api_calls_made == 0
      and .artifacts.online_selector_tensor == null
      and .artifacts.online_selector_manifest == null
    ' "$report" >/dev/null \
      || fail "full-answer oracle report validation failed: $report"
    write_core_summary "$mode" "$report" "$summary"
    echo "[v4-test] report=$report"
    echo "[v4-test] summary=$summary"
  fi
  echo "[v4-test] PASS mode=$mode stage=$stage"
}

case "$RUN_SELECTION" in
  smoke)
    run_mode smoke
    ;;
  full)
    run_mode full
    ;;
  all)
    run_mode smoke
    run_mode full
    ;;
esac

CURRENT_MODE="complete"
echo "[v4-test] PASS run_selection=$RUN_SELECTION"
