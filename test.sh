#!/usr/bin/env bash
# Authenticate and inspect the API-free V4.2 high-quality bank shortlist.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${MEMGEN_PYTHON_BIN:-python}"
OUTPUT_ROOT="${MEMGEN_V4_OUTPUT_ROOT:-$REPO_ROOT/output/experiments/v4}"
LOCAL_DIR="${MEMGEN_V4_2_LOCAL_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_local}"
SHORTLIST_DIR="${MEMGEN_V4_2_SHORTLIST_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_shortlist}"
AUDIT_DIR="${MEMGEN_V4_2_AUDIT_DIR:-$SHORTLIST_DIR/audit}"
REPORT_PATH="$AUDIT_DIR/v4_2_shortlist_test_report.json"
LOG_PATH="$AUDIT_DIR/v4_2_shortlist_test.log"

mkdir -p "$AUDIT_DIR"
: > "$LOG_PATH"

fail() {
  echo "[v4.2-test] FAIL: $*" >&2
  exit 1
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 \
  || fail "Python executable not found: $PYTHON_BIN"
command -v jq >/dev/null 2>&1 || fail "jq is required"

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

# This stage is deliberately unable to inherit the paid-stage credential.
unset DEEPSEEK_API_KEY || true

REPO_REVISION="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[v4.2-test] repo_revision=$REPO_REVISION"
echo "[v4.2-test] local_dir=$LOCAL_DIR"
echo "[v4.2-test] shortlist_dir=$SHORTLIST_DIR"
echo "[v4.2-test] audit_dir=$AUDIT_DIR"

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
  --resume \
  2>&1 | tee "$LOG_PATH"

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
echo "[v4.2-test] send the single report file above for the next analysis"
