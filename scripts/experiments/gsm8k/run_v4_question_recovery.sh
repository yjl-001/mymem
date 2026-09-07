#!/usr/bin/env bash
# Recover current V4 source evidence and run the zero-paid-API risk/oracle audit.
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
  run_v4_question_recovery.sh [--mode smoke|full] \
    [--stage recover|risk|cache|state-audit|oracle|all] \
    RECOVERY_ID SEMANTIC_EVIDENCE_PACKETS CURATED_BANK_DIR SIDE_KV_DIR OUTPUT_ROOT

Inputs:
  RECOVERY_ID                 New immutable lineage name; never reuse for changed inputs
  SEMANTIC_EVIDENCE_PACKETS   Surviving semantic_evidence_packets.jsonl
  CURATED_BANK_DIR            Current 17-bank curated directory
  SIDE_KV_DIR                 Current compiled target/reference Side-KV directory
  OUTPUT_ROOT                 Persistent MemGen output root

Outputs:
  OUTPUT_ROOT/lineages/gsm8k-recovery/RECOVERY_ID/
    recovery/                 rebuilt split, exact packet trajectories, sealed manifest
    risk_v3_4/                newly fitted no-AI V3.4 risk artifact
    v4_oracle_smoke|full/     source-state cache, CPU audit, oracle audit
    logs/                     stage logs

The recovery stage downloads/reads only the public GSM8K dataset. It does not
read API keys or invoke DeepSeek, GLM, OpenAI, Anthropic, or another teacher.
The new risk artifact is fitted from all 116 recovered trajectories in both
smoke and full modes; smoke only caps cache/oracle work. This runner never
starts selector compilation, dev-test, final-test, or online evaluation.
EOF
}

fail() {
  echo "[v4-question-recovery] FAIL: $*" >&2
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
  recover|risk|cache|state-audit|oracle|all) ;;
  *) fail "--stage must be recover, risk, cache, state-audit, oracle, or all" ;;
esac
[[ "${#POSITIONAL[@]}" -eq 5 ]] || {
  usage >&2
  exit 2
}

RECOVERY_ID="${POSITIONAL[0]}"
SEMANTIC_PACKETS="${POSITIONAL[1]}"
CURATED_BANK_DIR="${POSITIONAL[2]}"
SIDE_KV_DIR="${POSITIONAL[3]}"
OUTPUT_ROOT="${POSITIONAL[4]}"
LINEAGE_ROOT="${MEMGEN_V4_RECOVERY_LINEAGE_ROOT:-$OUTPUT_ROOT/lineages/gsm8k-recovery/$RECOVERY_ID}"
RECOVERY_DIR="$LINEAGE_ROOT/recovery"
RISK_DIR="$LINEAGE_ROOT/risk_v3_4"
RUN_ROOT="$LINEAGE_ROOT/v4_oracle_${MODE}"
SOURCE_STATE_DIR="$RUN_ROOT/source_state_cache"
STATE_AUDIT_DIR="$RUN_ROOT/source_state_audit"
ORACLE_AUDIT_DIR="$RUN_ROOT/oracle_audit"
LOG_DIR="$LINEAGE_ROOT/logs"

BANK_RECORDS="$CURATED_BANK_DIR/bank_records.jsonl"
BANK_MANIFEST="$CURATED_BANK_DIR/bank_manifest.json"
SIDE_KV_MANIFEST="$SIDE_KV_DIR/v4_side_kv_manifest.json"
RECOVERY_MANIFEST="$RECOVERY_DIR/v4_question_recovery_manifest.json"
EXPERIENCES="$RECOVERY_DIR/recovered_source_experiences.jsonl"
SPLIT_MANIFEST="$RECOVERY_DIR/split_manifest.json"
TOKEN_RISK_ARTIFACT="$RISK_DIR/token-entropy-risk-gate-v3.4.pt"
RISK_REPORT="$RISK_DIR/token_entropy_risk_report.json"
RISK_EVIDENCE="$RISK_DIR/token_entropy_risk_evidence.jsonl"
CACHE_MANIFEST="$SOURCE_STATE_DIR/v4_source_state_manifest.json"

command -v "$PYTHON_BIN" >/dev/null 2>&1 \
  || fail "Python executable not found: $PYTHON_BIN"
command -v jq >/dev/null 2>&1 || fail "jq is required"
if ! [[ "$SMOKE_LIMIT" =~ ^[1-9][0-9]*$ ]]; then
  fail "MEMGEN_V4_ORACLE_SMOKE_LIMIT must be a positive integer"
fi
if ! [[ "$RECOVERY_ID" =~ ^[a-z0-9][a-z0-9._-]{2,127}$ ]]; then
  fail "RECOVERY_ID must use 3-128 lowercase letters, digits, '.', '_' or '-'"
fi
for required in \
  "$SEMANTIC_PACKETS" \
  "$BANK_RECORDS" \
  "$BANK_MANIFEST" \
  "$SIDE_KV_MANIFEST"; do
  [[ -s "$required" ]] || fail "missing required input: $required"
done

export CUDA_VISIBLE_DEVICES="${MEMGEN_V4_CUDA_VISIBLE_DEVICES:-0}"
unset DEEPSEEK_API_KEY GLM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY || true
mkdir -p "$LINEAGE_ROOT" "$LOG_DIR"

echo "[v4-question-recovery] repo_revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[v4-question-recovery] mode=$MODE stage=$STAGE device=$DEVICE dtype=$DTYPE"
echo "[v4-question-recovery] recovery_id=$RECOVERY_ID"
echo "[v4-question-recovery] semantic_packets=$SEMANTIC_PACKETS"
echo "[v4-question-recovery] curated_bank_dir=$CURATED_BANK_DIR"
echo "[v4-question-recovery] side_kv_dir=$SIDE_KV_DIR"
echo "[v4-question-recovery] lineage_root=$LINEAGE_ROOT"

if [[ "$STAGE" == "recover" || "$STAGE" == "all" ]]; then
  "$PYTHON_BIN" scripts/recover_v4_source_evidence.py \
    --recovery-id "$RECOVERY_ID" \
    --semantic-evidence-packets "$SEMANTIC_PACKETS" \
    --bank-records "$BANK_RECORDS" \
    --bank-manifest "$BANK_MANIFEST" \
    --side-kv-manifest "$SIDE_KV_MANIFEST" \
    --output-dir "$RECOVERY_DIR" \
    --dataset-revision main \
    --bank-source-size 6000 \
    --calibration-val-size 1000 \
    --split-seed 42 \
    --resume \
    2>&1 | tee "$LOG_DIR/recover.log"
  jq -e '
    .status == "sealed_semantic_packet_source_replay"
    and .offline_only == true
    and .qualified_for_online_use == false
    and .counts.bank_count == 17
    and .counts.evidence_count == 116
    and .claims.original_phase1_file_recovery_claim == false
    and .claims.original_risk_artifact_recovery_claim == false
    and .claims.same_source_question == true
    and .claims.same_source_success_trajectory == true
    and .claims.same_source_failure_trajectory == true
    and .claims.held_out_generalization_claim == false
    and .external_api_calls_made == 0
  ' "$RECOVERY_MANIFEST" >/dev/null \
    || fail "question-recovery manifest validation failed"
fi

for recovered in "$RECOVERY_MANIFEST" "$EXPERIENCES" "$SPLIT_MANIFEST"; do
  [[ -s "$recovered" ]] || fail "missing recovery artifact: $recovered"
done
if [[ "$STAGE" == "recover" ]]; then
  echo "[v4-question-recovery] PASS mode=$MODE stage=$STAGE"
  echo "[v4-question-recovery] recovery_manifest=$RECOVERY_MANIFEST"
  exit 0
fi

if [[ "$STAGE" == "risk" || "$STAGE" == "all" ]]; then
  mkdir -p "$RISK_DIR"
  RISK_REUSABLE=0
  if [[ -s "$TOKEN_RISK_ARTIFACT" && -s "$RISK_REPORT" && -s "$RISK_EVIDENCE" ]]; then
    if "$PYTHON_BIN" -c 'import json,sys,torch; from pathlib import Path; from memgen.experience.phase1 import file_sha256; artifact_path,report_path,evidence_path,recovery_path,experiences_path=map(Path,sys.argv[1:]); artifact=torch.load(artifact_path,map_location="cpu",weights_only=False); report=json.load(report_path.open(encoding="utf-8")); inputs=artifact.get("inputs",{}); ok=artifact.get("status")=="passed" and artifact.get("qualification",{}).get("passed") is True and artifact.get("construction",{}).get("source")=="semantic_packet_replay_strict_verifier_no_ai" and artifact.get("construction",{}).get("ai_review_approval_claim") is False and inputs.get("question_recovery_manifest_sha256")==file_sha256(recovery_path) and inputs.get("recovered_experiences_sha256")==file_sha256(experiences_path) and report.get("status")=="passed" and report.get("artifact",{}).get("sha256")==file_sha256(artifact_path) and report.get("evidence_trace",{}).get("sha256")==file_sha256(evidence_path); raise SystemExit(0 if ok else 1)' \
      "$TOKEN_RISK_ARTIFACT" "$RISK_REPORT" "$RISK_EVIDENCE" \
      "$RECOVERY_MANIFEST" "$EXPERIENCES"; then
      RISK_REUSABLE=1
      echo "[v4-question-recovery] reusing qualified recovery risk: $TOKEN_RISK_ARTIFACT"
    fi
  fi
  if [[ "$RISK_REUSABLE" -eq 0 ]]; then
    if [[ -e "$TOKEN_RISK_ARTIFACT" || -e "$RISK_REPORT" || -e "$RISK_EVIDENCE" ]]; then
      fail "risk output is stale/incomplete; use a new RECOVERY_ID"
    fi
    MODEL_METADATA="$("$PYTHON_BIN" -c 'import json,sys; value=json.load(open(sys.argv[1],encoding="utf-8")); reasoner=value["reasoner"]; print("\t".join(str(reasoner[key]) for key in ("model_name","model_revision","tokenizer_revision")))' "$RECOVERY_MANIFEST")"
    IFS=$'\t' read -r MODEL MODEL_REVISION TOKENIZER_REVISION <<< "$MODEL_METADATA"
    "$PYTHON_BIN" scripts/compile_token_entropy_risk_gate.py \
      --question-recovery-manifest "$RECOVERY_MANIFEST" \
      --experiences "$EXPERIENCES" \
      --output-dir "$RISK_DIR" \
      --model "$MODEL" \
      --model-revision "$MODEL_REVISION" \
      --tokenizer-revision "$TOKENIZER_REVISION" \
      --device "$DEVICE" \
      --dtype "$DTYPE" \
      --attn-implementation sdpa \
      --layer 24 \
      --experience-types answer_correctness \
      --batch-size "${MEMGEN_V4_RECOVERY_RISK_BATCH_SIZE:-1}" \
      --sink-token-count 4 \
      --high-entropy-quantile 0.85 \
      --low-entropy-quantile 0.50 \
      --risk-train-fraction 0.5 \
      --risk-split-seed 42 \
      --stable-low-token-count 2 \
      --horizon-quantile 0.75 \
      --maximum-recovery-horizon 32 \
      --min-events-per-label 40 \
      --min-heldout-roc-auc 0.60 \
      2>&1 | tee "$LOG_DIR/risk.log"
  fi
fi

for risk_file in "$TOKEN_RISK_ARTIFACT" "$RISK_REPORT" "$RISK_EVIDENCE"; do
  [[ -s "$risk_file" ]] || fail "missing qualified recovery risk artifact: $risk_file"
done
if [[ "$STAGE" == "risk" ]]; then
  echo "[v4-question-recovery] PASS mode=$MODE stage=$STAGE"
  echo "[v4-question-recovery] token_risk_artifact=$TOKEN_RISK_ARTIFACT"
  exit 0
fi

if [[ "$STAGE" == "cache" || "$STAGE" == "all" ]]; then
  mkdir -p "$SOURCE_STATE_DIR"
  CACHE_LIMIT_ARGS=()
  if [[ "$MODE" == "smoke" ]]; then
    CACHE_LIMIT_ARGS=(--limit "$SMOKE_LIMIT")
  fi
  "$PYTHON_BIN" scripts/extract_v4_source_state_cache.py \
    --question-recovery-manifest "$RECOVERY_MANIFEST" \
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
    2>&1 | tee "$LOG_DIR/source-state-${MODE}.log"
  jq -e '
    .status == "source_state_cache_built"
    and .offline_only == true
    and .qualified_for_online_use == false
    and .contains_reward_or_answer_signal == false
    and .configuration.layer_number == 24
    and .configuration.maximum_gate_attempts == 3
    and .configuration.maximum_hidden_window == 32
    and .configuration.support_unit == "independent_sample"
    and .provenance.question_recovery.original_phase1_file_recovery_claim == false
    and .provenance.question_recovery.original_risk_artifact_recovery_claim == false
    and .provenance.question_recovery.same_source_question == true
  ' "$CACHE_MANIFEST" >/dev/null \
    || fail "recovery source-state cache validation failed"
  if [[ "$MODE" == "full" ]]; then
    jq -e '
      .configuration.extraction_scope == "question_recovery_full_curated_construction"
      and .counts.bank_count == 17
      and .counts.independent_sample_count == 116
      and .configuration.expected_full_construction_count == 116
      and .configuration.extracted_construction_count == 116
    ' "$CACHE_MANIFEST" >/dev/null \
      || fail "full recovery source-state coverage failed"
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
    2>&1 | tee "$LOG_DIR/state-audit-${MODE}.log"
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
    || fail "CPU recovery source-state audit validation failed"
fi

if [[ "$STAGE" == "oracle" || "$STAGE" == "all" ]]; then
  [[ -s "$CACHE_MANIFEST" ]] || fail "missing source-state cache: $CACHE_MANIFEST"
  mkdir -p "$ORACLE_AUDIT_DIR"
  ORACLE_LIMIT_ARGS=()
  if [[ "$MODE" == "smoke" ]]; then
    ORACLE_LIMIT_ARGS=(--limit-per-kind "$SMOKE_LIMIT")
  fi
  "$PYTHON_BIN" scripts/audit_v4_oracle_causal_utility.py \
    --question-recovery-manifest "$RECOVERY_MANIFEST" \
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
    2>&1 | tee "$LOG_DIR/oracle-${MODE}.log"
  jq -e '
    .status == "completed_mechanism_diagnostic"
    and .complete == true
    and .offline_only == true
    and .qualified_for_online_use == false
    and .online_artifacts_generated == false
    and .held_out_generalization_claim == false
    and .gate_unreachable_counted_as_memory_ineffective == false
    and .question_recovery.original_phase1_file_recovery_claim == false
    and .question_recovery.original_risk_artifact_recovery_claim == false
    and .question_recovery.same_source_question == true
    and .question_recovery.same_source_success_trajectory == true
    and .question_recovery.same_source_failure_trajectory == true
    and .question_recovery.external_api_calls_made == 0
    and .artifacts.online_selector_tensor == null
    and .artifacts.online_selector_manifest == null
  ' "$ORACLE_AUDIT_DIR/v4_oracle_report.json" >/dev/null \
    || fail "recovery oracle causal audit validation failed"
fi

echo "[v4-question-recovery] PASS mode=$MODE stage=$STAGE"
echo "[v4-question-recovery] recovery_manifest=$RECOVERY_MANIFEST"
echo "[v4-question-recovery] token_risk_artifact=$TOKEN_RISK_ARTIFACT"
echo "[v4-question-recovery] source_state_cache=$SOURCE_STATE_DIR"
echo "[v4-question-recovery] source_state_audit=$STATE_AUDIT_DIR"
echo "[v4-question-recovery] oracle_audit=$ORACLE_AUDIT_DIR"
