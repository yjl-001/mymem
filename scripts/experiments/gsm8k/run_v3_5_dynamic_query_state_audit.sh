#!/usr/bin/env bash
# Fixed prompt/current/delta/local-window query audit over two V3.5 key banks.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

DEVICE="cuda"
PERMUTATION_COUNT="10000"
POSITIONAL=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --permutation-count) PERMUTATION_COUNT="$2"; shift 2 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *) POSITIONAL+=("$1"); shift ;;
  esac
done

if [[ "${#POSITIONAL[@]}" -ne 4 ]]; then
  echo "Usage: $0 [--device DEVICE] [--permutation-count N] PHASE1_DIR E0_DIR TOKEN_RISK_ARTIFACT OUTPUT_ROOT" >&2
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
AUDIT_DIR="$OUTPUT_ROOT/v3_5_dynamic_query_state"
REPORT="$AUDIT_DIR/query_state_report.json"

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
  "$KEY_COMPONENT_REPORT"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

audit_is_complete() {
  [[ -s "$REPORT" && \
     -s "$AUDIT_DIR/query_state_evidence.jsonl" && \
     -s "$AUDIT_DIR/query_state_embeddings.safetensors" && \
     -s "$AUDIT_DIR/query_state_geometry.jsonl" && \
     -s "$AUDIT_DIR/query_state_hub_text_audit.jsonl" ]] && \
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
query_count=int(value.get("query_count",-1))
ok=(
    value.get("schema_version")
    =="experience-memory-v3.5-dynamic-query-state-decomposition-report-v1"
    and value.get("status")=="completed_diagnostic"
    and value.get("diagnostic_only") is True
    and value.get("formal_v3_5_qualification_changed") is False
    and value.get("reasoner_forward_run") is True
    and value.get("generation_run") is False
    and value.get("side_kv_used") is False
    and value.get("task_accuracy_used") is False
    and value.get("answer_or_reward_used") is False
    and value.get("query_variant_selected") is False
    and value.get("key_variant_selected") is False
    and configuration.get("fixed_query_variants")==[
        "prompt_boundary","current_token","prompt_subtracted_delta",
        "local_reasoning_window_16"
    ]
    and configuration.get("fixed_key_variants")==[
        "applicability_key","dynamic_key"
    ]
    and configuration.get("local_reasoning_window_size")==16
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
    and artifacts.get("grid_evidence",{}).get("sha256")
    ==file_sha256(Path(sys.argv[13]))
    and artifacts.get("grid_evidence",{}).get("row_count")==8*query_count
    and artifacts.get("query_state_embeddings",{}).get("sha256")
    ==file_sha256(Path(sys.argv[14]))
    and artifacts.get("query_state_embeddings",{}).get("tensor_count")
    ==4*query_count
    and artifacts.get("query_geometry",{}).get("sha256")
    ==file_sha256(Path(sys.argv[15]))
    and artifacts.get("query_geometry",{}).get("row_count")==query_count
    and artifacts.get("hub_key_text_audit",{}).get("sha256")
    ==file_sha256(Path(sys.argv[16]))
)
raise SystemExit(0 if ok else 1)
' \
      "$REPORT" \
      "$APPROVED_BANK" "$VERIFIED_EXPERIENCES" "$SPLIT_MANIFEST" \
      "$MEMORY_RECORDS" "$DUAL_KEY_MANIFEST" "$OFFLINE_REPORT" \
      "$TOKEN_RISK_ARTIFACT" "$SOURCE_REPORT" "$SOURCE_EVIDENCE" \
      "$FIRST_GATE_QUERIES" "$KEY_COMPONENT_REPORT" \
      "$AUDIT_DIR/query_state_evidence.jsonl" \
      "$AUDIT_DIR/query_state_embeddings.safetensors" \
      "$AUDIT_DIR/query_state_geometry.jsonl" \
      "$AUDIT_DIR/query_state_hub_text_audit.jsonl"
}

if audit_is_complete; then
  echo "Reusing authenticated V3.5 dynamic query-state audit: $AUDIT_DIR"
  exit 0
fi

python scripts/audit_v3_5_dynamic_query_state.py \
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
  --output-dir "$AUDIT_DIR" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --permutation-count "$PERMUTATION_COUNT"

if ! audit_is_complete; then
  echo "V3.5 dynamic query-state audit failed authentication: $AUDIT_DIR" >&2
  exit 3
fi

echo "V3.5 dynamic query-state audit complete: $REPORT"
