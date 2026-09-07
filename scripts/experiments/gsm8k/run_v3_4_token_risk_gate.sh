#!/usr/bin/env bash
# V3.4 offline stage: compile and qualify the every-token layer-24 risk gate.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 PHASE1_DIR OUTPUT_DIR" >&2
  exit 2
fi

PHASE1_DIR="$1"
OUTPUT_DIR="$2"
PYTHON_BIN="${MEMGEN_PYTHON_BIN:-python}"
ARTIFACT="$OUTPUT_DIR/token-entropy-risk-gate-v3.4.pt"
REPORT="$OUTPUT_DIR/token_entropy_risk_report.json"
EVIDENCE="$OUTPUT_DIR/token_entropy_risk_evidence.jsonl"
APPROVED_BANK="$PHASE1_DIR/ai_approved_bank_records.jsonl"
EXPERIENCES="$PHASE1_DIR/verified_experiences.jsonl"
for REQUIRED in "$APPROVED_BANK" "$EXPERIENCES"; do
  if [[ ! -s "$REQUIRED" ]]; then
    echo "Missing or empty Phase-1 input: $REQUIRED" >&2
    exit 1
  fi
done

if [[ -s "$ARTIFACT" && -s "$REPORT" && -s "$EVIDENCE" ]] && \
  "$PYTHON_BIN" -c 'import json,sys,torch; from pathlib import Path; from memgen.experience.phase1 import file_sha256; artifact_path,report_path,evidence,approved,experiences=map(Path,sys.argv[1:]); artifact=torch.load(artifact_path,map_location="cpu",weights_only=False); report=json.load(report_path.open(encoding="utf-8")); approved_sha=file_sha256(approved); experiences_sha=file_sha256(experiences); ok=artifact.get("schema_version")=="token-entropy-risk-gate-artifact-v3.4" and artifact.get("status")=="passed" and artifact.get("qualification",{}).get("passed") is True and artifact.get("inputs",{}).get("approved_bank_sha256")==approved_sha and artifact.get("inputs",{}).get("verified_experiences_sha256")==experiences_sha and report.get("status")=="passed" and report.get("qualification",{}).get("passed") is True and report.get("inputs",{}).get("approved_bank_sha256")==approved_sha and report.get("inputs",{}).get("verified_experiences_sha256")==experiences_sha and report.get("artifact",{}).get("sha256")==file_sha256(artifact_path) and report.get("evidence_trace",{}).get("sha256")==file_sha256(evidence); raise SystemExit(0 if ok else 1)' "$ARTIFACT" "$REPORT" "$EVIDENCE" "$APPROVED_BANK" "$EXPERIENCES"; then
  echo "Reusing qualified V3.4 token-risk artifact: $ARTIFACT"
  exit 0
fi

if [[ -e "$ARTIFACT" || -e "$REPORT" ]]; then
  echo "Refusing to overwrite a stale or incomplete V3.4 risk output: $OUTPUT_DIR" >&2
  echo "Use a new lineage/risk directory so incompatible artifacts remain distinguishable." >&2
  exit 1
fi

MODEL_METADATA="$("$PYTHON_BIN" -c 'import json,sys; record=next(json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()); student=record.get("student", {}); values=(student.get("model_name"),student.get("model_revision"),student.get("tokenizer_revision")); assert all(isinstance(value,str) and value for value in values), "incomplete student metadata"; print("\t".join(values))' "$APPROVED_BANK")"
IFS=$'\t' read -r MODEL MODEL_REVISION TOKENIZER_REVISION <<< "$MODEL_METADATA"

export CUDA_VISIBLE_DEVICES="${MEMGEN_V3_CUDA_VISIBLE_DEVICES:-0}"
DEVICE="${MEMGEN_V3_DEVICE:-cuda}"
DTYPE="${MEMGEN_V3_DTYPE:-bfloat16}"

"$PYTHON_BIN" scripts/compile_token_entropy_risk_gate.py \
  --approved-bank "$APPROVED_BANK" \
  --experiences "$EXPERIENCES" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL" \
  --model-revision "$MODEL_REVISION" \
  --tokenizer-revision "$TOKENIZER_REVISION" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --attn-implementation sdpa \
  --batch-size 1 \
  --layer 24 \
  --sink-token-count 4 \
  --high-entropy-quantile 0.85 \
  --low-entropy-quantile 0.50 \
  --risk-train-fraction 0.50 \
  --risk-split-seed 42 \
  --stable-low-token-count 2 \
  --horizon-quantile 0.75 \
  --maximum-recovery-horizon 32 \
  --min-events-per-label 40 \
  --min-heldout-roc-auc 0.60

echo "V3.4 token-risk artifact: $ARTIFACT"
echo "V3.4 token-risk report: $REPORT"
