#!/usr/bin/env bash
# Cross-trajectory V3.6 source-state key audit; no model forward or generation.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PERMUTATION_COUNT="10000"
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --permutation-count) PERMUTATION_COUNT="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [--permutation-count N] PHASE1_DIR E0_DIR TOKEN_RISK_ARTIFACT OUTPUT_ROOT" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
TOKEN_RISK_ARTIFACT="${POSITIONAL[2]}"
OUTPUT_ROOT="${POSITIONAL[3]}"

APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
VERIFIED_EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
SPLIT_MANIFEST="$PHASE1_DIR/split_manifest.json"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
V35_BANK_DIR="$OUTPUT_ROOT/v3_5_applicability_selector/dual_key_bank"
DUAL_KEY_MANIFEST="$V35_BANK_DIR/dual_retrieval_key_manifest.json"
OFFLINE_REPORT="$V35_BANK_DIR/offline_report.json"
SOURCE_DIR="$OUTPUT_ROOT/v3_5_dynamic_source_alignment"
SOURCE_REPORT="$SOURCE_DIR/alignment_report.json"
SOURCE_EVIDENCE="$SOURCE_DIR/source_state_evidence.jsonl"
FIRST_GATE_QUERIES="$SOURCE_DIR/first_gate_query_embeddings.safetensors"
KEY_COMPONENT_REPORT="$OUTPUT_ROOT/v3_5_dynamic_key_components/key_component_report.json"
QUERY_STATE_DIR="$OUTPUT_ROOT/v3_5_dynamic_query_state"
QUERY_STATE_REPORT="$QUERY_STATE_DIR/query_state_report.json"
QUERY_STATE_EMBEDDINGS="$QUERY_STATE_DIR/query_state_embeddings.safetensors"
AUDIT_DIR="$OUTPUT_ROOT/v3_6_source_state_keys"
REPORT="$AUDIT_DIR/state_key_report.json"

for REQUIRED in \
  "$APPROVED_BANK" \
  "$VERIFIED_EXPERIENCES" \
  "$SPLIT_MANIFEST" \
  "$MEMORY_RECORDS" \
  "$DUAL_KEY_MANIFEST" \
  "$OFFLINE_REPORT" \
  "$TOKEN_RISK_ARTIFACT" \
  "$SOURCE_REPORT" \
  "$SOURCE_EVIDENCE" \
  "$FIRST_GATE_QUERIES" \
  "$KEY_COMPONENT_REPORT" \
  "$QUERY_STATE_REPORT" \
  "$QUERY_STATE_EMBEDDINGS"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

audit_is_complete() {
  [[ -s "$REPORT" && \
     -s "$AUDIT_DIR/state_key_evidence.jsonl" && \
     -s "$AUDIT_DIR/reference_state_key_bank.safetensors" && \
     -s "$AUDIT_DIR/reference_state_key_manifest.json" && \
     -s "$AUDIT_DIR/state_key_hub_text_audit.jsonl" ]] && \
    python -c '
import json,sys
from pathlib import Path
from memgen.experience.phase1 import canonical_json_sha256,file_sha256

report_path=Path(sys.argv[1])
value=json.loads(report_path.read_text(encoding="utf-8"))
stored=value.get("report_sha256")
logical=dict(value)
logical.pop("report_sha256",None)
inputs=value.get("inputs",{})
artifacts=value.get("artifacts",{})
requirements=value.get("requirements",{})
configuration=value.get("configuration",{})
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
state_artifact=artifacts.get("state_key_bank",{})
evidence=artifacts.get("evidence",{})
hub=artifacts.get("hub_text_audit",{})
key_count=int(value.get("reference_key_count",-1))
query_count=int(value.get("paired_target_query_count",-1))
manifest_path=Path(sys.argv[16])
manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
manifest_stored=manifest.get("manifest_sha256")
manifest_logical=dict(manifest)
manifest_logical.pop("manifest_sha256",None)
manifest_ok=(
    manifest.get("schema_version")
    =="experience-memory-v3.6-source-state-retrieval-key-bank-v1"
    and manifest.get("status")=="completed_diagnostic"
    and manifest.get("diagnostic_only") is True
    and manifest.get("qualified_for_online_use") is False
    and manifest.get("memory_count")==key_count
    and manifest.get("value_source")
    =="existing_full_when_facing_prefer_avoid_side_kv_unchanged"
    and manifest.get("tensor_artifact",{}).get("sha256")
    ==file_sha256(Path(sys.argv[15]))
    and manifest.get("tensor_artifact",{}).get("tensor_count")==4*key_count
    and manifest_stored==canonical_json_sha256(manifest_logical)
)
ok=(
    value.get("schema_version")
    =="experience-memory-v3.6-source-state-retrieval-key-report-v1"
    and value.get("status")=="completed_diagnostic"
    and value.get("diagnostic_only") is True
    and value.get("qualified_for_online_use") is False
    and value.get("formal_v3_5_qualification_changed") is False
    and value.get("reasoner_forward_or_generation_run") is False
    and value.get("side_kv_used") is False
    and value.get("side_kv_payload_changed") is False
    and value.get("task_accuracy_used") is False
    and value.get("answer_or_reward_used") is False
    and value.get("variant_selected") is False
    and value.get("threshold_fitted") is False
    and configuration.get("fixed_variants")==[
        "text_applicability__target_current_control",
        "state_prompt__target_prompt_identity_control",
        "state_current__target_current",
        "state_delta__target_delta",
        "state_local16__target_local16"
    ]
    and stored==canonical_json_sha256(logical)
    and isinstance(requirements,dict)
    and bool(requirements)
    and all(requirements.values())
    and implementation_ok
    and inputs.get("approved_bank_sha256")==file_sha256(Path(sys.argv[2]))
    and inputs.get("verified_experiences_sha256")==file_sha256(Path(sys.argv[3]))
    and inputs.get("split_manifest_sha256")==file_sha256(Path(sys.argv[4]))
    and inputs.get("memory_records_sha256")==file_sha256(Path(sys.argv[5]))
    and inputs.get("dual_key_manifest_sha256")==file_sha256(Path(sys.argv[6]))
    and inputs.get("v35_offline_report_sha256")==file_sha256(Path(sys.argv[7]))
    and inputs.get("token_risk_artifact_sha256")==file_sha256(Path(sys.argv[8]))
    and inputs.get("source_alignment_report_sha256")==file_sha256(Path(sys.argv[9]))
    and inputs.get("source_alignment_evidence_sha256")==file_sha256(Path(sys.argv[10]))
    and inputs.get("first_gate_queries_sha256")==file_sha256(Path(sys.argv[11]))
    and inputs.get("key_component_report_sha256")==file_sha256(Path(sys.argv[12]))
    and inputs.get("query_state_report_sha256")==file_sha256(Path(sys.argv[13]))
    and inputs.get("query_state_embeddings_sha256")==file_sha256(Path(sys.argv[14]))
    and state_artifact.get("tensor_sha256")==file_sha256(Path(sys.argv[15]))
    and state_artifact.get("manifest_sha256")==file_sha256(Path(sys.argv[16]))
    and state_artifact.get("manifest_logical_sha256")==manifest_stored
    and state_artifact.get("memory_count")==key_count
    and state_artifact.get("tensor_count")==4*key_count
    and manifest_ok
    and evidence.get("sha256")==file_sha256(Path(sys.argv[17]))
    and evidence.get("row_count")==5*query_count
    and hub.get("sha256")==file_sha256(Path(sys.argv[18]))
)
raise SystemExit(0 if ok else 1)
' \
      "$REPORT" \
      "$APPROVED_BANK" "$VERIFIED_EXPERIENCES" "$SPLIT_MANIFEST" \
      "$MEMORY_RECORDS" "$DUAL_KEY_MANIFEST" "$OFFLINE_REPORT" \
      "$TOKEN_RISK_ARTIFACT" "$SOURCE_REPORT" "$SOURCE_EVIDENCE" \
      "$FIRST_GATE_QUERIES" "$KEY_COMPONENT_REPORT" \
      "$QUERY_STATE_REPORT" "$QUERY_STATE_EMBEDDINGS" \
      "$AUDIT_DIR/reference_state_key_bank.safetensors" \
      "$AUDIT_DIR/reference_state_key_manifest.json" \
      "$AUDIT_DIR/state_key_evidence.jsonl" \
      "$AUDIT_DIR/state_key_hub_text_audit.jsonl"
}

if audit_is_complete; then
  echo "Reusing authenticated V3.6 source-state key audit: $AUDIT_DIR"
  exit 0
fi

python scripts/audit_v3_6_source_state_keys.py \
  --approved-bank "$APPROVED_BANK" \
  --verified-experiences "$VERIFIED_EXPERIENCES" \
  --split-manifest "$SPLIT_MANIFEST" \
  --memory-records "$MEMORY_RECORDS" \
  --dual-key-manifest "$DUAL_KEY_MANIFEST" \
  --v35-offline-report "$OFFLINE_REPORT" \
  --token-risk-artifact "$TOKEN_RISK_ARTIFACT" \
  --source-alignment-report "$SOURCE_REPORT" \
  --source-alignment-evidence "$SOURCE_EVIDENCE" \
  --first-gate-queries "$FIRST_GATE_QUERIES" \
  --key-component-report "$KEY_COMPONENT_REPORT" \
  --query-state-report "$QUERY_STATE_REPORT" \
  --query-state-embeddings "$QUERY_STATE_EMBEDDINGS" \
  --output-dir "$AUDIT_DIR" \
  --permutation-count "$PERMUTATION_COUNT"

if ! audit_is_complete; then
  echo "V3.6 source-state key audit failed authentication: $AUDIT_DIR" >&2
  exit 3
fi

echo "V3.6 source-state key audit complete: $REPORT"
