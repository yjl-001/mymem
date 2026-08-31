#!/usr/bin/env bash
# Held-out cross-problem retrieval plus frozen side-KV causal treatment audit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

OFFSET="0"
LIMIT="64"
CANDIDATE_TOP_K="4"
RANDOM_CANDIDATES="4"
RRF_RANK_CONSTANT="60"
SEED="3617"
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --offset) OFFSET="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --candidate-top-k) CANDIDATE_TOP_K="$2"; shift 2 ;;
    --random-candidates) RANDOM_CANDIDATES="$2"; shift 2 ;;
    --rrf-rank-constant) RRF_RANK_CONSTANT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [--offset N] [--limit N] [--candidate-top-k K] [--random-candidates N] [--rrf-rank-constant N] [--seed N] PHASE1_DIR E0_DIR TOKEN_RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
TOKEN_RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"

SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
SIDE_KV_MANIFEST="$E0_DIR/side_kv_manifest.json"
E0_FINAL_REPORT="$E0_DIR/e0_final_report.json"
DUAL_KEY_MANIFEST="$OUTPUT_ROOT/v3_5_applicability_selector/dual_key_bank/dual_retrieval_key_manifest.json"
SOURCE_EVIDENCE="$OUTPUT_ROOT/v3_5_dynamic_source_alignment/source_state_evidence.jsonl"
V36_DIR="$OUTPUT_ROOT/v3_6_source_state_keys"
V36_REPORT="$V36_DIR/state_key_report.json"
STATE_KEY_MANIFEST="$V36_DIR/reference_state_key_manifest.json"
RUN_NAME="dev_offset${OFFSET}_limit${LIMIT}_k${CANDIDATE_TOP_K}_random${RANDOM_CANDIDATES}_seed${SEED}"
AUDIT_DIR="$OUTPUT_ROOT/v3_7_cross_problem_causal/$RUN_NAME"
REPORT="$AUDIT_DIR/causal_report.json"

for REQUIRED in \
  "$SPLIT_MANIFEST" \
  "$MEMORY_RECORDS" \
  "$SIDE_KV_MANIFEST" \
  "$E0_FINAL_REPORT" \
  "$TOKEN_RISK_ARTIFACT" \
  "$DUAL_KEY_MANIFEST" \
  "$SOURCE_EVIDENCE" \
  "$V36_REPORT" \
  "$STATE_KEY_MANIFEST"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

audit_is_complete() {
  [[ -s "$REPORT" && \
     -s "$AUDIT_DIR/causal_profile.json" && \
     -s "$AUDIT_DIR/causal_queries.jsonl" && \
     -f "$AUDIT_DIR/causal_treatments.jsonl" && \
     -s "$AUDIT_DIR/causal_report.md" ]] && \
    python -c '
import json,sys
from pathlib import Path
from memgen.experience.phase1 import canonical_json_sha256,file_sha256

root=Path(sys.argv[1])
report=json.loads((root/"causal_report.json").read_text(encoding="utf-8"))
stored=report.get("report_sha256")
logical=dict(report)
logical.pop("report_sha256",None)
artifacts=report.get("artifacts",{})
requirements=report.get("requirements",{})
inputs=report.get("inputs",{})
implementation=inputs.get("implementation_files_sha256",{})
implementation_ok=(
    isinstance(implementation,dict)
    and bool(implementation)
    and all(
        not Path(path).is_absolute()
        and ".." not in Path(path).parts
        and Path(path).is_file()
        and file_sha256(Path(path))==digest
        for path,digest in implementation.items()
    )
    and inputs.get("implementation_set_sha256")
    ==canonical_json_sha256(implementation)
)
ok=(
    report.get("schema_version")
    =="experience-memory-v3.7-cross-problem-causal-applicability-report-v1"
    and report.get("status")=="completed_diagnostic"
    and report.get("qualified_for_online_use") is False
    and report.get("formal_v3_5_qualification_changed") is False
    and report.get("same_question_memory_permitted") is False
    and report.get("cross_problem_enforced") is True
    and report.get("task_accuracy_used") is True
    and report.get("answer_or_reward_used") is True
    and report.get("variant_selected") is False
    and report.get("threshold_fitted") is False
    and stored==canonical_json_sha256(logical)
    and isinstance(requirements,dict)
    and bool(requirements)
    and all(requirements.values())
    and implementation_ok
    and inputs.get("split_manifest_sha256")==file_sha256(Path(sys.argv[2]))
    and inputs.get("memory_records_sha256")==file_sha256(Path(sys.argv[3]))
    and inputs.get("side_kv_manifest_sha256")==file_sha256(Path(sys.argv[4]))
    and inputs.get("e0_final_report_sha256")==file_sha256(Path(sys.argv[5]))
    and inputs.get("token_risk_artifact_sha256")==file_sha256(Path(sys.argv[6]))
    and inputs.get("dual_key_manifest_sha256")==file_sha256(Path(sys.argv[7]))
    and inputs.get("source_alignment_evidence_sha256")==file_sha256(Path(sys.argv[8]))
    and inputs.get("v36_report_sha256")==file_sha256(Path(sys.argv[9]))
    and inputs.get("state_key_manifest_sha256")==file_sha256(Path(sys.argv[10]))
    and artifacts.get("profile",{}).get("sha256")
    ==file_sha256(root/"causal_profile.json")
    and artifacts.get("queries",{}).get("sha256")
    ==file_sha256(root/"causal_queries.jsonl")
    and artifacts.get("treatments",{}).get("sha256")
    ==file_sha256(root/"causal_treatments.jsonl")
)
raise SystemExit(0 if ok else 1)
' \
      "$AUDIT_DIR" \
      "$SPLIT_MANIFEST" "$MEMORY_RECORDS" "$SIDE_KV_MANIFEST" \
      "$E0_FINAL_REPORT" "$TOKEN_RISK_ARTIFACT" "$DUAL_KEY_MANIFEST" \
      "$SOURCE_EVIDENCE" "$V36_REPORT" "$STATE_KEY_MANIFEST"
}

if audit_is_complete; then
  echo "Reusing authenticated V3.7 causal audit: $AUDIT_DIR"
  exit 0
fi

python scripts/audit_v3_7_cross_problem_causal_applicability.py \
  --split-manifest "$SPLIT_MANIFEST" \
  --memory-records "$MEMORY_RECORDS" \
  --side-kv-manifest "$SIDE_KV_MANIFEST" \
  --e0-final-report "$E0_FINAL_REPORT" \
  --token-risk-artifact "$TOKEN_RISK_ARTIFACT" \
  --dual-key-manifest "$DUAL_KEY_MANIFEST" \
  --source-alignment-evidence "$SOURCE_EVIDENCE" \
  --v36-report "$V36_REPORT" \
  --state-key-manifest "$STATE_KEY_MANIFEST" \
  --output-dir "$AUDIT_DIR" \
  --offset "$OFFSET" \
  --limit "$LIMIT" \
  --candidate-top-k "$CANDIDATE_TOP_K" \
  --random-candidates "$RANDOM_CANDIDATES" \
  --rrf-rank-constant "$RRF_RANK_CONSTANT" \
  --seed "$SEED"

if ! audit_is_complete; then
  echo "V3.7 causal audit failed authentication: $AUDIT_DIR" >&2
  exit 3
fi

echo "V3.7 cross-problem causal audit complete: $REPORT"
