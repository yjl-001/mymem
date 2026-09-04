#!/usr/bin/env bash
# Authenticate V4.2 inputs and build the zero-API provisional local-direct bank.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${MEMGEN_PYTHON_BIN:-python}"
OUTPUT_ROOT="${MEMGEN_V4_OUTPUT_ROOT:-$REPO_ROOT/output/experiments/v4}"
LOCAL_DIR="${MEMGEN_V4_2_LOCAL_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_local}"
SHORTLIST_DIR="${MEMGEN_V4_2_SHORTLIST_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_shortlist}"
AUDIT_DIR="${MEMGEN_V4_2_AUDIT_DIR:-$SHORTLIST_DIR/audit}"
SOURCE_DIR="${MEMGEN_V4_SOURCE_DIR:-$OUTPUT_ROOT/offline/construction}"
SEMANTIC_DIR="${MEMGEN_V4_2_SEMANTIC_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_semantic}"
LOCAL_DIRECT_DIR="${MEMGEN_V4_2_LOCAL_DIRECT_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_local_direct}"
SEMANTIC_POLICY="${MEMGEN_V4_2_SEMANTIC_POLICY:-$REPO_ROOT/configs/experiments/gsm8k/v4_2_semantic_policy.json}"
SEMANTIC_STAGE="${MEMGEN_V4_2_STAGE:-local-direct}"
SKIP_SEMANTIC="${MEMGEN_V4_2_SKIP_SEMANTIC:-0}"
REPORT_PATH="$AUDIT_DIR/v4_2_shortlist_test_report.json"
LOG_PATH="$AUDIT_DIR/v4_2_shortlist_test.log"
SEMANTIC_LOG_PATH="$SEMANTIC_DIR/v4_2_semantic_${SEMANTIC_STAGE}.log"
LOCAL_DIRECT_LOG_PATH="$LOCAL_DIRECT_DIR/v4_2_local_direct.log"

mkdir -p "$AUDIT_DIR"
: > "$LOG_PATH"

fail() {
  echo "[v4.2-test] FAIL: $*" >&2
  exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 \
  || fail "Python executable not found: $PYTHON_BIN"
command -v jq >/dev/null 2>&1 || fail "jq is required"
case "$SEMANTIC_STAGE" in
  preflight | local-direct) ;;
  paid)
    [[ "${MEMGEN_V4_2_APPROVE_PAID_STAGE:-0}" == "1" ]] \
      || fail "paid stage requires MEMGEN_V4_2_APPROVE_PAID_STAGE=1"
    [[ -n "${DEEPSEEK_API_KEY:-}" ]] \
      || fail "paid stage requires DEEPSEEK_API_KEY"
    ;;
  *) fail "MEMGEN_V4_2_STAGE must be preflight, local-direct, or paid" ;;
esac

for REQUIRED in \
  construction_profile.json \
  local_atoms.jsonl \
  multiview_embeddings_manifest.json \
  mechanism_embeddings.npy \
  repair_embeddings.npy \
  applicability_embeddings.npy \
  local_clusters.jsonl \
  cluster_review_packets.jsonl \
  local_cluster_plan.json; do
  [[ -s "$LOCAL_DIR/$REQUIRED" ]] \
    || fail "missing or empty local construction artifact: $LOCAL_DIR/$REQUIRED"
done

SHORTLIST_ENTRYPOINT="$REPO_ROOT/scripts/select_v4_2_bank_candidates.py"
[[ -s "$SHORTLIST_ENTRYPOINT" ]] || fail "missing shortlist entrypoint"
if grep -Eq 'TeacherClient|DEEPSEEK_API_KEY|os\.environ|urllib' \
  "$SHORTLIST_ENTRYPOINT"; then
  fail "shortlist entrypoint contains a forbidden paid/network dependency"
fi

REPO_REVISION="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[v4.2-test] repo_revision=$REPO_REVISION"
echo "[v4.2-test] local_dir=$LOCAL_DIR"
echo "[v4.2-test] shortlist_dir=$SHORTLIST_DIR"
echo "[v4.2-test] audit_dir=$AUDIT_DIR"

# The local shortlist child is deliberately unable to inherit a paid-stage key.
(
  unset DEEPSEEK_API_KEY || true
  "$PYTHON_BIN" "$SHORTLIST_ENTRYPOINT" \
    --local-construction-dir "$LOCAL_DIR" \
    --output-dir "$SHORTLIST_DIR" \
    --preferred-support 6 \
    --minimum-support-cohesion-quantile 0.50 \
    --redundancy-mechanism-threshold 0.92 \
    --redundancy-repair-threshold 0.92 \
    --redundancy-applicability-threshold 0.85 \
    --max-synthesis-candidates 48 \
    --target-runtime-bank-cap 32 \
    --synthesis-batch-size 4 \
    --review-batch-size 8 \
    --resume
) 2>&1 | tee "$LOG_PATH"

PREFLIGHT_PATH="$SHORTLIST_DIR/api_preflight_report.json"
QUALITY_PATH="$SHORTLIST_DIR/candidate_quality_report.json"
SELECTED_PATH="$SHORTLIST_DIR/selected_synthesis_candidates.jsonl"
REJECTED_PATH="$SHORTLIST_DIR/rejected_or_redundant_candidates.jsonl"
EDGES_PATH="$SHORTLIST_DIR/candidate_redundancy_edges.jsonl"
MANIFEST_PATH="$SHORTLIST_DIR/synthesis_shortlist_manifest.json"

for REQUIRED in \
  "$PREFLIGHT_PATH" \
  "$QUALITY_PATH" \
  "$SELECTED_PATH" \
  "$REJECTED_PATH" \
  "$EDGES_PATH" \
  "$MANIFEST_PATH"; do
  [[ -f "$REQUIRED" ]] || fail "missing shortlist artifact: $REQUIRED"
done

jq -e '
  .status == "synthesis_shortlist_complete_api_not_started"
  and .external_api_calls_made == 0
  and .api_key_read == false
  and .automatic_paid_stage_transition == false
  and .qualified_for_online_use == false
  and .source_candidate_count
      == (.selected_synthesis_candidate_count + .rejected_candidate_count)
  and .selected_synthesis_candidate_count > 0
  and .selected_synthesis_candidate_count <= .max_synthesis_candidates
  and (([.decision_counts[]] | add) == .source_candidate_count)
  and .within_synthesis_candidate_guardrail == true
  and .synthesis_blocked_reason == null
' "$PREFLIGHT_PATH" >/dev/null \
  || fail "preflight invariants did not pass"

SOURCE_COUNT="$(jq -r '.source_candidate_count' "$PREFLIGHT_PATH")"
SELECTED_COUNT="$(jq -r '.selected_synthesis_candidate_count' "$PREFLIGHT_PATH")"
REJECTED_COUNT="$(jq -r '.rejected_candidate_count' "$PREFLIGHT_PATH")"
REDUNDANCY_COUNT="$(jq -r '.redundancy_edge_count' "$PREFLIGHT_PATH")"

jq -e \
  --argjson source_count "$SOURCE_COUNT" \
  --argjson selected_count "$SELECTED_COUNT" \
  --argjson rejected_count "$REJECTED_COUNT" '
  .source_candidate_count == $source_count
  and .selected_candidate_count == $selected_count
  and .rejected_candidate_count == $rejected_count
  and (([.candidates[] | select(.decision == "selected")] | length)
       == $selected_count)
  and (([.candidates[] | select(.decision == "rejected")] | length)
       == $rejected_count)
' "$QUALITY_PATH" >/dev/null \
  || fail "candidate-quality terminal coverage did not pass"

REPORT_TEMP="$(mktemp "$AUDIT_DIR/.v4_2_shortlist_test_report.XXXXXX")"
trap 'rm -f "$REPORT_TEMP"' EXIT
jq -n \
  --arg repo_revision "$REPO_REVISION" \
  --arg local_dir "$LOCAL_DIR" \
  --arg shortlist_dir "$SHORTLIST_DIR" \
  --slurpfile preflight "$PREFLIGHT_PATH" \
  --slurpfile quality "$QUALITY_PATH" \
  --slurpfile selected "$SELECTED_PATH" \
  --slurpfile edges "$EDGES_PATH" '
  ($preflight[0]) as $p
  | ($quality[0]) as $q
  | ([
      $edges[]
      | . + {
          weakest_normalized_redundancy_margin: (
            [
              ((.mechanism_similarity - 0.92) / 0.08),
              ((.repair_similarity - 0.92) / 0.08),
              ((.applicability_similarity - 0.85) / 0.15)
            ]
            | min
          )
        }
    ] | sort_by(.weakest_normalized_redundancy_margin) | .[:20])
    as $boundary_edges
  | {
      schema_version: "memgen-v4.2-shortlist-test-report-v1",
      status: "PASS",
      repo_revision: $repo_revision,
      local_construction_dir: $local_dir,
      shortlist_dir: $shortlist_dir,
      assertions: {
        authenticated_resume_passed: true,
        external_api_calls_made: $p.external_api_calls_made,
        api_key_read: $p.api_key_read,
        automatic_paid_stage_transition: $p.automatic_paid_stage_transition,
        qualified_for_online_use: $p.qualified_for_online_use,
        terminal_candidate_coverage_passed: true,
        within_synthesis_candidate_guardrail:
          $p.within_synthesis_candidate_guardrail,
        within_runtime_bank_cap_without_review:
          ($p.selected_synthesis_candidate_count <= $p.target_runtime_bank_cap)
      },
      summary: {
        source_candidate_count: $p.source_candidate_count,
        selected_synthesis_candidate_count:
          $p.selected_synthesis_candidate_count,
        rejected_candidate_count: $p.rejected_candidate_count,
        decision_counts: $p.decision_counts,
        minimum_support_cohesion_threshold:
          $p.minimum_support_cohesion_threshold,
        redundancy_edge_count: $p.redundancy_edge_count,
        max_synthesis_candidates: $p.max_synthesis_candidates,
        target_runtime_bank_cap: $p.target_runtime_bank_cap,
        planned_initial_synthesis_requests:
          $p.planned_initial_synthesis_requests,
        maximum_followup_review_requests:
          $p.maximum_followup_review_requests,
        maximum_total_paid_requests: $p.maximum_total_paid_requests,
        semantic_evidence_characters: $p.semantic_evidence_characters,
        estimated_semantic_evidence_tokens_at_three_chars_per_token:
          $p.estimated_semantic_evidence_tokens_at_three_chars_per_token,
        profile_sha256: $p.profile_sha256,
        shortlist_manifest_sha256: $p.shortlist_manifest_sha256,
        report_sha256: $p.report_sha256
      },
      selected_quality: [
        $q.candidates[]
        | select(.decision == "selected")
        | {
            rank: .selection_rank,
            candidate_id,
            support: .distinct_sample_count,
            support_tier,
            weakest_margin: .weakest_normalized_minimum_margin,
            mechanism_min: .mechanism_similarity_min,
            repair_min: .repair_similarity_min,
            applicability_min: .applicability_similarity_min,
            joint_min: .joint_similarity_min,
            joint_mean: .joint_similarity_mean,
            selection_reason: .reason
          }
      ] | sort_by(.rank),
      high_risk_semantic_audit: [
        $selected[]
        | select(
            .quality.weakest_normalized_minimum_margin < 0.12
            or .quality.joint_similarity_min < 0.86
          )
        | {
            rank: .selection_rank,
            candidate_id: .candidate.candidate_id,
            support: .candidate.distinct_sample_count,
            weakest_margin: .quality.weakest_normalized_minimum_margin,
            mechanism_min: .quality.mechanism_similarity_min,
            repair_min: .quality.repair_similarity_min,
            applicability_min: .quality.applicability_similarity_min,
            joint_min: .quality.joint_similarity_min,
            joint_mean: .quality.joint_similarity_mean,
            semantic_evidence: [
              .semantic_evidence[]
              | {
                  problem_structure,
                  decision_point,
                  failure_mechanism,
                  repair_operator,
                  verification_operator
                }
            ]
          }
      ] | sort_by(.rank),
      boundary_redundancy_edges: $boundary_edges
    }
' > "$REPORT_TEMP"
mv "$REPORT_TEMP" "$REPORT_PATH"
trap - EXIT

HIGH_RISK_COUNT="$(jq -r '.high_risk_semantic_audit | length' "$REPORT_PATH")"
REPORT_BYTES="$(wc -c < "$REPORT_PATH" | tr -d ' ')"

echo "[v4.2-test] PASS source=$SOURCE_COUNT selected=$SELECTED_COUNT rejected=$REJECTED_COUNT redundancy_edges=$REDUNDANCY_COUNT high_risk=$HIGH_RISK_COUNT api_calls=0"
echo "[v4.2-test] report=$REPORT_PATH bytes=$REPORT_BYTES"
echo "[v4.2-test] log=$LOG_PATH"

if [[ "$SKIP_SEMANTIC" == "1" ]]; then
  echo "[v4.2-test] semantic stage explicitly skipped"
  exit 0
fi

for REQUIRED in \
  "$SOURCE_DIR/repair_signatures.jsonl" \
  "$SOURCE_DIR/construction_profile.json" \
  "$SEMANTIC_POLICY"; do
  [[ -s "$REQUIRED" ]] || fail "missing semantic-bank input: $REQUIRED"
done

PHASE1_DIR="${MEMGEN_PHASE1_DIR:-}"
if [[ -z "$PHASE1_DIR" ]]; then
  PHASE1_CANDIDATES=()
  while IFS= read -r CANDIDATE; do
    [[ -s "$CANDIDATE/verified_experiences.jsonl" ]] \
      && PHASE1_CANDIDATES+=("$CANDIDATE")
  done < <(find "$REPO_ROOT/output/experiments" -type f -name split_manifest.json -print 2>/dev/null | sed 's#/split_manifest.json$##' | sort -u)
  if [[ "${#PHASE1_CANDIDATES[@]}" -ne 1 ]]; then
    fail "set MEMGEN_PHASE1_DIR; automatic discovery found ${#PHASE1_CANDIDATES[@]} matching directories"
  fi
  PHASE1_DIR="${PHASE1_CANDIDATES[0]}"
fi
for REQUIRED in \
  "$PHASE1_DIR/verified_experiences.jsonl" \
  "$PHASE1_DIR/split_manifest.json"; do
  [[ -s "$REQUIRED" ]] || fail "missing Phase-1 input: $REQUIRED"
done

SEMANTIC_ENTRYPOINT="$REPO_ROOT/scripts/build_v4_2_semantic_bank.py"
[[ -s "$SEMANTIC_ENTRYPOINT" ]] || fail "missing semantic-bank entrypoint"
mkdir -p "$SEMANTIC_DIR"
SEMANTIC_BUILD_STAGE="$SEMANTIC_STAGE"
if [[ "$SEMANTIC_BUILD_STAGE" == "local-direct" ]]; then
  SEMANTIC_BUILD_STAGE="preflight"
fi
SEMANTIC_COMMAND=(
  "$PYTHON_BIN" "$SEMANTIC_ENTRYPOINT"
  --experiences "$PHASE1_DIR/verified_experiences.jsonl"
  --split-manifest "$PHASE1_DIR/split_manifest.json"
  --source-signatures "$SOURCE_DIR/repair_signatures.jsonl"
  --source-construction-profile "$SOURCE_DIR/construction_profile.json"
  --local-construction-dir "$LOCAL_DIR"
  --shortlist-dir "$SHORTLIST_DIR"
  --semantic-policy "$SEMANTIC_POLICY"
  --output-dir "$SEMANTIC_DIR"
  --dataset-revision main
  --stage "$SEMANTIC_BUILD_STAGE"
  --resume
)
if [[ "$SEMANTIC_BUILD_STAGE" == "paid" ]]; then
  SEMANTIC_COMMAND+=(--approve-paid-stage)
  "${SEMANTIC_COMMAND[@]}" 2>&1 | tee "$SEMANTIC_LOG_PATH"
else
  (
    unset DEEPSEEK_API_KEY || true
    "${SEMANTIC_COMMAND[@]}"
  ) 2>&1 | tee "$SEMANTIC_LOG_PATH"
fi

SEMANTIC_PREFLIGHT="$SEMANTIC_DIR/api_preflight_report.json"
[[ -s "$SEMANTIC_PREFLIGHT" ]] || fail "missing semantic preflight report"
jq -e '
  .status == "semantic_evidence_ready_api_not_started"
  and .external_api_calls_made == 0
  and .api_key_read == false
  and .automatic_paid_stage_transition == false
  and .qualified_for_online_use == false
  and .planned_candidate_count > 0
  and (.planned_candidate_count + .preflight_excluded_candidate_count
       == .source_selected_candidate_count)
  and .policy_excluded_evidence_count == 2
  and .planned_combined_request_count > 0
  and (.nominal_total_paid_request_count_if_all_coherent
       <= .maximum_total_request_units_after_recursive_split)
' "$SEMANTIC_PREFLIGHT" >/dev/null \
  || fail "semantic preflight invariants did not pass"

if [[ "$SEMANTIC_STAGE" == "paid" ]]; then
  PAID_REPORT="$SEMANTIC_DIR/paid_stage_report.json"
  BANK_MANIFEST="$SEMANTIC_DIR/bank_manifest.json"
  BANK_RECORDS="$SEMANTIC_DIR/bank_records.jsonl"
  for REQUIRED in "$PAID_REPORT" "$BANK_MANIFEST" "$BANK_RECORDS"; do
    [[ -s "$REQUIRED" ]] || fail "missing paid semantic-bank artifact: $REQUIRED"
  done
  jq -e '
    .status == "semantic_bank_constructed_not_tensor_compiled"
    and .qualified_for_online_use == false
    and .bank_record_count > 0
    and .bank_record_count <= 32
    and .bank_manifest_sha256 != null
  ' "$PAID_REPORT" >/dev/null || fail "paid semantic-bank report did not pass"
  jq -e '
    .schema_version == "memgen-v4.2-bank-manifest-v1"
    and .status == "constructed_not_tensor_compiled"
    and .qualified_for_online_use == false
    and .record_count > 0
    and .record_count <= .profile.target_runtime_bank_cap
    and (.record_count == (.bank_ids | length))
  ' "$BANK_MANIFEST" >/dev/null || fail "semantic bank manifest did not pass"
  BANK_RECORD_COUNT="$(wc -l < "$BANK_RECORDS" | tr -d ' ')"
  [[ "$BANK_RECORD_COUNT" == "$(jq -r '.record_count' "$BANK_MANIFEST")" ]] \
    || fail "semantic bank record count differs from manifest"
  echo "[v4.2-test] PAID PASS bank_records=$BANK_RECORD_COUNT"
  echo "[v4.2-test] paid_report=$PAID_REPORT"
  echo "[v4.2-test] bank_manifest=$BANK_MANIFEST"
elif [[ "$SEMANTIC_STAGE" == "local-direct" ]]; then
  LOCAL_DIRECT_ENTRYPOINT="$REPO_ROOT/scripts/build_v4_2_local_direct_bank.py"
  [[ -s "$LOCAL_DIRECT_ENTRYPOINT" ]] \
    || fail "missing V4.2 local-direct bank entrypoint"
  mkdir -p "$LOCAL_DIRECT_DIR"
  (
    unset DEEPSEEK_API_KEY || true
    "$PYTHON_BIN" "$LOCAL_DIRECT_ENTRYPOINT" \
      --experiences "$PHASE1_DIR/verified_experiences.jsonl" \
      --split-manifest "$PHASE1_DIR/split_manifest.json" \
      --source-signatures "$SOURCE_DIR/repair_signatures.jsonl" \
      --source-construction-profile "$SOURCE_DIR/construction_profile.json" \
      --local-construction-dir "$LOCAL_DIR" \
      --shortlist-dir "$SHORTLIST_DIR" \
      --semantic-preflight-dir "$SEMANTIC_DIR" \
      --semantic-policy "$SEMANTIC_POLICY" \
      --output-dir "$LOCAL_DIRECT_DIR" \
      --dataset-revision main \
      --resume
  ) 2>&1 | tee "$LOCAL_DIRECT_LOG_PATH"

  LOCAL_DIRECT_REPORT="$LOCAL_DIRECT_DIR/local_direct_report.json"
  BANK_MANIFEST="$LOCAL_DIRECT_DIR/bank_manifest.json"
  BANK_RECORDS="$LOCAL_DIRECT_DIR/bank_records.jsonl"
  MEDOID_SELECTIONS="$LOCAL_DIRECT_DIR/medoid_selections.jsonl"
  for REQUIRED in \
    "$LOCAL_DIRECT_REPORT" \
    "$BANK_MANIFEST" \
    "$BANK_RECORDS" \
    "$MEDOID_SELECTIONS"; do
    [[ -s "$REQUIRED" ]] \
      || fail "missing local-direct bank artifact: $REQUIRED"
  done
  jq -e --slurpfile preflight "$SEMANTIC_PREFLIGHT" '
    .status == "local_direct_bank_constructed_not_tensor_compiled"
    and .quality_tier == "provisional_local_direct"
    and .qualified_for_online_use == false
    and .admission_basis == "authenticated_local_shortlist"
    and .semantic_audit_performed == false
    and .independent_review_performed == false
    and .api_key_read == false
    and .external_api_calls_made == 0
    and .bank_record_count == $preflight[0].planned_candidate_count
    and .source_candidate_count == .bank_record_count
    and .joint_medoid_count == .bank_record_count
    and .evidence_count == $preflight[0].evidence_count
    and .bank_record_count > 0
    and .bank_record_count <= 32
  ' "$LOCAL_DIRECT_REPORT" >/dev/null \
    || fail "V4.2 local-direct report did not pass"
  jq -e '
    .schema_version == "memgen-v4.2-local-direct-bank-manifest-v1"
    and .construction_version == "v4.2-local-direct"
    and .status == "constructed_not_tensor_compiled"
    and .quality_tier == "provisional_local_direct"
    and .qualified_for_online_use == false
    and .semantic_review.performed == false
    and .semantic_review.reviewer == null
    and .semantic_review.external_api_calls_made == 0
    and .api_key_read == false
    and .external_api_calls_made == 0
    and .profile.injection_layer == 24
    and .profile.relative_phase_delta == 0
    and .profile.target_online_only == true
    and .auxiliary_banks_materialized == false
    and .record_count == (.bank_ids | length)
  ' "$BANK_MANIFEST" >/dev/null \
    || fail "V4.2 local-direct manifest did not pass"
  BANK_RECORD_COUNT="$(wc -l < "$BANK_RECORDS" | tr -d ' ')"
  MEDOID_COUNT="$(wc -l < "$MEDOID_SELECTIONS" | tr -d ' ')"
  [[ "$BANK_RECORD_COUNT" == "$(jq -r '.record_count' "$BANK_MANIFEST")" ]] \
    || fail "local-direct bank record count differs from manifest"
  [[ "$MEDOID_COUNT" == "$BANK_RECORD_COUNT" ]] \
    || fail "local-direct medoid count differs from bank count"
  jq -e -s '
    all(.[];
      .construction.distinct_sample_count >= 5
      and .roles == {
        "target_online_injectable": true,
        "reference_online_injectable": false,
        "auxiliary": null
      }
      and .local_direct_admission.semantic_audit_performed == false
      and .local_direct_admission.independent_review_performed == false
      and .compiler_contract.layer_number == 24
    )
  ' "$BANK_RECORDS" >/dev/null \
    || fail "V4.2 local-direct bank records did not pass"
  echo "[v4.2-test] LOCAL-DIRECT PASS bank_records=$BANK_RECORD_COUNT api_calls=0"
  echo "[v4.2-test] local_direct_report=$LOCAL_DIRECT_REPORT"
  echo "[v4.2-test] bank_manifest=$BANK_MANIFEST"
  echo "[v4.2-test] local_direct_log=$LOCAL_DIRECT_LOG_PATH"
else
  echo "[v4.2-test] PREFLIGHT PASS api_key_read=false api_calls=0"
fi
echo "[v4.2-test] semantic_preflight=$SEMANTIC_PREFLIGHT"
echo "[v4.2-test] semantic_log=$SEMANTIC_LOG_PATH"
echo "[v4.2-test] send the local-direct report or, after paid mode, the paid report"
