#!/usr/bin/env bash
# Answer-blind source-trajectory audit for the V3.5 dynamic retrieval space.
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
V35_DIR="$OUTPUT_ROOT/v3_5_applicability_selector"
DUAL_KEY_MANIFEST="$V35_DIR/dual_key_bank/dual_retrieval_key_manifest.json"
OFFLINE_REPORT="$V35_DIR/dual_key_bank/offline_report.json"
AUDIT_DIR="$OUTPUT_ROOT/v3_5_dynamic_source_alignment"
REPORT="$AUDIT_DIR/alignment_report.json"

for REQUIRED in \
  "$APPROVED_BANK" \
  "$VERIFIED_EXPERIENCES" \
  "$SPLIT_MANIFEST" \
  "$MEMORY_RECORDS" \
  "$DUAL_KEY_MANIFEST" \
  "$OFFLINE_REPORT" \
  "$TOKEN_RISK_ARTIFACT"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

audit_is_complete() {
  [[ -s "$REPORT" && \
     -s "$AUDIT_DIR/source_state_evidence.jsonl" && \
     -s "$AUDIT_DIR/first_gate_query_embeddings.safetensors" ]] && \
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
implementation=inputs.get("implementation_files_sha256",{})
implementation_ok=(
    isinstance(implementation,dict)
    and bool(implementation)
    and all(
        Path(path).is_file()
        and file_sha256(Path(path))==digest
        for path,digest in implementation.items()
    )
    and inputs.get("implementation_set_sha256")
    ==canonical_json_sha256(implementation)
)
ok=(
    value.get("schema_version")=="experience-memory-v3.5-dynamic-source-alignment-report-v1"
    and value.get("status")=="completed_diagnostic"
    and value.get("diagnostic_only") is True
    and value.get("formal_v3_5_qualification_changed") is False
    and value.get("task_accuracy_used") is False
    and value.get("answer_or_reward_used") is False
    and stored==canonical_json_sha256(logical)
    and all(requirements.values())
    and implementation_ok
    and inputs.get("approved_bank_sha256")==file_sha256(Path(sys.argv[2]))
    and inputs.get("verified_experiences_sha256")==file_sha256(Path(sys.argv[3]))
    and inputs.get("split_manifest_sha256")==file_sha256(Path(sys.argv[4]))
    and inputs.get("memory_records_sha256")==file_sha256(Path(sys.argv[5]))
    and inputs.get("dual_key_manifest_sha256")==file_sha256(Path(sys.argv[6]))
    and inputs.get("v35_offline_report_sha256")==file_sha256(Path(sys.argv[7]))
    and inputs.get("token_risk_artifact_sha256")==file_sha256(Path(sys.argv[8]))
    and artifacts.get("evidence",{}).get("sha256")==file_sha256(Path(sys.argv[9]))
    and artifacts.get("first_gate_query_embeddings",{}).get("sha256")==file_sha256(Path(sys.argv[10]))
)
raise SystemExit(0 if ok else 1)
' \
      "$REPORT" \
      "$APPROVED_BANK" "$VERIFIED_EXPERIENCES" "$SPLIT_MANIFEST" \
      "$MEMORY_RECORDS" "$DUAL_KEY_MANIFEST" "$OFFLINE_REPORT" \
      "$TOKEN_RISK_ARTIFACT" \
      "$AUDIT_DIR/source_state_evidence.jsonl" \
      "$AUDIT_DIR/first_gate_query_embeddings.safetensors"
}

if audit_is_complete; then
  echo "Reusing authenticated V3.5 dynamic source-state audit: $AUDIT_DIR"
  exit 0
fi

python scripts/audit_v3_5_dynamic_source_alignment.py \
  --approved-bank "$APPROVED_BANK" \
  --verified-experiences "$VERIFIED_EXPERIENCES" \
  --split-manifest "$SPLIT_MANIFEST" \
  --memory-records "$MEMORY_RECORDS" \
  --dual-key-manifest "$DUAL_KEY_MANIFEST" \
  --v35-offline-report "$OFFLINE_REPORT" \
  --token-risk-artifact "$TOKEN_RISK_ARTIFACT" \
  --output-dir "$AUDIT_DIR" \
  --device "$DEVICE" \
  --dtype bfloat16 \
  --permutation-count "$PERMUTATION_COUNT"

if ! audit_is_complete; then
  echo "V3.5 dynamic source-state audit failed authentication: $AUDIT_DIR" >&2
  exit 3
fi

echo "V3.5 dynamic source-state audit complete: $REPORT"
