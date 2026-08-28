#!/usr/bin/env bash
# Fixed raw/centered/PC1 audit over authenticated V3.5 first-gate queries.
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

if [[ "${#POSITIONAL[@]}" -ne 3 ]]; then
  echo "Usage: $0 [--permutation-count N] PHASE1_DIR E0_DIR OUTPUT_ROOT" >&2
  exit 2
fi

PHASE1_DIR="${POSITIONAL[0]}"
E0_DIR="${POSITIONAL[1]}"
OUTPUT_ROOT="${POSITIONAL[2]}"

APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
VERIFIED_EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
MEMORY_RECORDS="$E0_DIR/memory_records.v2.jsonl"
V35_BANK_DIR="$OUTPUT_ROOT/v3_5_applicability_selector/dual_key_bank"
DUAL_KEY_MANIFEST="$V35_BANK_DIR/dual_retrieval_key_manifest.json"
SOURCE_DIR="$OUTPUT_ROOT/v3_5_dynamic_source_alignment"
SOURCE_REPORT="$SOURCE_DIR/alignment_report.json"
FIRST_GATE_QUERIES="$SOURCE_DIR/first_gate_query_embeddings.safetensors"
AUDIT_DIR="$OUTPUT_ROOT/v3_5_dynamic_hubness"
REPORT="$AUDIT_DIR/hubness_decomposition_report.json"

for REQUIRED in \
  "$APPROVED_BANK" \
  "$VERIFIED_EXPERIENCES" \
  "$MEMORY_RECORDS" \
  "$DUAL_KEY_MANIFEST" \
  "$SOURCE_REPORT" \
  "$FIRST_GATE_QUERIES"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty input: $REQUIRED" >&2
    exit 1
  fi
done

audit_is_complete() {
  [[ -s "$REPORT" && \
     -s "$AUDIT_DIR/hubness_variant_evidence.jsonl" && \
     -s "$AUDIT_DIR/hubness_transforms.safetensors" && \
     -s "$AUDIT_DIR/hub_key_text_audit.jsonl" ]] && \
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
    value.get("schema_version")
    =="experience-memory-v3.5-dynamic-hubness-decomposition-report-v1"
    and value.get("status")=="completed_diagnostic"
    and value.get("diagnostic_only") is True
    and value.get("formal_v3_5_qualification_changed") is False
    and value.get("reasoner_forward_or_generation_run") is False
    and value.get("task_accuracy_used") is False
    and value.get("answer_or_reward_used") is False
    and value.get("variant_selected") is False
    and stored==canonical_json_sha256(logical)
    and all(requirements.values())
    and implementation_ok
    and inputs.get("approved_bank_sha256")==file_sha256(Path(sys.argv[2]))
    and inputs.get("verified_experiences_sha256")==file_sha256(Path(sys.argv[3]))
    and inputs.get("memory_records_sha256")==file_sha256(Path(sys.argv[4]))
    and inputs.get("dual_key_manifest_sha256")==file_sha256(Path(sys.argv[5]))
    and inputs.get("source_alignment_report_sha256")==file_sha256(Path(sys.argv[6]))
    and inputs.get("first_gate_queries_sha256")==file_sha256(Path(sys.argv[7]))
    and artifacts.get("variant_evidence",{}).get("sha256")
    ==file_sha256(Path(sys.argv[8]))
    and artifacts.get("transforms",{}).get("sha256")
    ==file_sha256(Path(sys.argv[9]))
    and artifacts.get("hub_key_text_audit",{}).get("sha256")
    ==file_sha256(Path(sys.argv[10]))
)
raise SystemExit(0 if ok else 1)
' \
      "$REPORT" \
      "$APPROVED_BANK" "$VERIFIED_EXPERIENCES" "$MEMORY_RECORDS" \
      "$DUAL_KEY_MANIFEST" "$SOURCE_REPORT" "$FIRST_GATE_QUERIES" \
      "$AUDIT_DIR/hubness_variant_evidence.jsonl" \
      "$AUDIT_DIR/hubness_transforms.safetensors" \
      "$AUDIT_DIR/hub_key_text_audit.jsonl"
}

if audit_is_complete; then
  echo "Reusing authenticated V3.5 dynamic hubness audit: $AUDIT_DIR"
  exit 0
fi

python scripts/audit_v3_5_dynamic_hubness.py \
  --approved-bank "$APPROVED_BANK" \
  --verified-experiences "$VERIFIED_EXPERIENCES" \
  --memory-records "$MEMORY_RECORDS" \
  --dual-key-manifest "$DUAL_KEY_MANIFEST" \
  --source-alignment-report "$SOURCE_REPORT" \
  --first-gate-queries "$FIRST_GATE_QUERIES" \
  --output-dir "$AUDIT_DIR" \
  --permutation-count "$PERMUTATION_COUNT"

if ! audit_is_complete; then
  echo "V3.5 dynamic hubness audit failed authentication: $AUDIT_DIR" >&2
  exit 3
fi

echo "V3.5 dynamic hubness audit complete: $REPORT"
