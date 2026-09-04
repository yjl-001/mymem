#!/usr/bin/env bash
# V4.2: curate 17 records, compile layer-24 side-KV, then selector anchors.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${MEMGEN_PYTHON_BIN:-python}"
OUTPUT_ROOT="${MEMGEN_V4_OUTPUT_ROOT:-$REPO_ROOT/output/experiments/v4}"
SOURCE_DIR="${MEMGEN_V4_2_LOCAL_DIRECT_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_local_direct}"
CURATED_DIR="${MEMGEN_V4_2_CURATED_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_local_curated}"
SIDE_KV_DIR="${MEMGEN_V4_2_CURATED_SIDE_KV_DIR:-$OUTPUT_ROOT/offline/side_kv_v4_2_local_curated}"
SELECTOR_DIR="${MEMGEN_V4_2_CURATED_SELECTOR_DIR:-$OUTPUT_ROOT/offline/selector_v4_2_local_curated}"
POLICY="${MEMGEN_V4_2_CURATION_POLICY:-$REPO_ROOT/configs/experiments/gsm8k/v4_2_local_curation_policy.json}"
DEVICE="${MEMGEN_V4_DEVICE:-cuda}"
DTYPE="${MEMGEN_V4_DTYPE:-bfloat16}"
STAGE="all"
POSITIONAL=()

usage() {
  cat <<'EOF'
Usage:
  run_v4_2_curated_offline.sh [--stage curate|side-kv|selector|all]
  run_v4_2_curated_offline.sh [--stage ...] PHASE1_DIR E0_DIR TOKEN_RISK_ARTIFACT [V4_OUTPUT_ROOT]

With no positional arguments, set MEMGEN_PHASE1_DIR, MEMGEN_E0_DIR, and
MEMGEN_TOKEN_RISK_ARTIFACT, or let the script use a unique local match.
The curate-only stage needs none of those three inputs.
EOF
}

fail() {
  echo "[v4.2-curated-offline] FAIL: $*" >&2
  exit 1
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
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

case "$STAGE" in
  curate|side-kv|selector|all) ;;
  *) fail "--stage must be curate, side-kv, selector, or all" ;;
esac
if [[ "${#POSITIONAL[@]}" -ne 0 && "${#POSITIONAL[@]}" -ne 3 && "${#POSITIONAL[@]}" -ne 4 ]]; then
  usage >&2
  exit 2
fi
if [[ "${#POSITIONAL[@]}" -ge 3 ]]; then
  PHASE1_DIR="${POSITIONAL[0]}"
  E0_DIR="${POSITIONAL[1]}"
  TOKEN_RISK_ARTIFACT="${POSITIONAL[2]}"
  if [[ "${#POSITIONAL[@]}" -eq 4 ]]; then
    OUTPUT_ROOT="${POSITIONAL[3]}"
    SOURCE_DIR="${MEMGEN_V4_2_LOCAL_DIRECT_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_local_direct}"
    CURATED_DIR="${MEMGEN_V4_2_CURATED_DIR:-$OUTPUT_ROOT/offline/construction_v4_2_local_curated}"
    SIDE_KV_DIR="${MEMGEN_V4_2_CURATED_SIDE_KV_DIR:-$OUTPUT_ROOT/offline/side_kv_v4_2_local_curated}"
    SELECTOR_DIR="${MEMGEN_V4_2_CURATED_SELECTOR_DIR:-$OUTPUT_ROOT/offline/selector_v4_2_local_curated}"
  fi
else
  PHASE1_DIR="${MEMGEN_PHASE1_DIR:-}"
  E0_DIR="${MEMGEN_E0_DIR:-}"
  TOKEN_RISK_ARTIFACT="${MEMGEN_TOKEN_RISK_ARTIFACT:-}"
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 \
  || fail "Python executable not found: $PYTHON_BIN"
command -v jq >/dev/null 2>&1 || fail "jq is required"
export CUDA_VISIBLE_DEVICES="${MEMGEN_V4_CUDA_VISIBLE_DEVICES:-0}"
# The complete route is local/model-inference only; a paid-stage key is never inherited.
unset DEEPSEEK_API_KEY || true

discover_phase1() {
  local candidates=()
  local candidate
  while IFS= read -r candidate; do
    candidate="${candidate%/verified_experiences.jsonl}"
    if [[ -s "$candidate/split_manifest.json" ]]; then
      candidates+=("$candidate")
    fi
  done < <(find "$REPO_ROOT/output/experiments" -type f -name verified_experiences.jsonl -print 2>/dev/null | sort -u)
  [[ "${#candidates[@]}" -eq 1 ]] \
    || fail "set MEMGEN_PHASE1_DIR; automatic discovery found ${#candidates[@]} Phase-1 directories"
  PHASE1_DIR="${candidates[0]}"
}

discover_e0() {
  local candidates=()
  local candidate
  while IFS= read -r candidate; do
    candidate="${candidate%/side_kv_manifest.json}"
    if [[ -s "$candidate/e0_final_report.json" ]]; then
      candidates+=("$candidate")
    fi
  done < <(find "$REPO_ROOT/output/experiments" -type f -name side_kv_manifest.json -print 2>/dev/null | sort -u)
  [[ "${#candidates[@]}" -eq 1 ]] \
    || fail "set MEMGEN_E0_DIR; automatic discovery found ${#candidates[@]} qualified E0 directories"
  E0_DIR="${candidates[0]}"
}

discover_risk_artifact() {
  local candidates=()
  local candidate
  while IFS= read -r candidate; do
    if [[ -s "$(dirname "$candidate")/token_entropy_risk_report.json" ]]; then
      candidates+=("$candidate")
    fi
  done < <(find "$REPO_ROOT/output/experiments" -type f -name token-entropy-risk-gate-v3.4.pt -print 2>/dev/null | sort -u)
  [[ "${#candidates[@]}" -eq 1 ]] \
    || fail "set MEMGEN_TOKEN_RISK_ARTIFACT; automatic discovery found ${#candidates[@]} V3.4 risk artifacts"
  TOKEN_RISK_ARTIFACT="${candidates[0]}"
}

if [[ "$STAGE" == "side-kv" || "$STAGE" == "all" ]]; then
  [[ -n "$E0_DIR" ]] || discover_e0
fi
if [[ "$STAGE" == "selector" || "$STAGE" == "all" ]]; then
  [[ -n "$PHASE1_DIR" ]] || discover_phase1
  [[ -n "$TOKEN_RISK_ARTIFACT" ]] || discover_risk_artifact
fi

mkdir -p "$OUTPUT_ROOT/offline"
echo "[v4.2-curated-offline] repo_revision=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[v4.2-curated-offline] stage=$STAGE device=$DEVICE dtype=$DTYPE cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "[v4.2-curated-offline] source_dir=$SOURCE_DIR"

if [[ "$STAGE" == "curate" || "$STAGE" == "all" ]]; then
  for required in \
    "$SOURCE_DIR/construction_profile.json" \
    "$SOURCE_DIR/bank_records.jsonl" \
    "$SOURCE_DIR/bank_manifest.json" \
    "$SOURCE_DIR/local_direct_report.json" \
    "$POLICY"; do
    [[ -s "$required" ]] || fail "missing curation input: $required"
  done
  mkdir -p "$CURATED_DIR"
  "$PYTHON_BIN" scripts/curate_v4_2_local_direct_bank.py \
    --source-dir "$SOURCE_DIR" \
    --policy "$POLICY" \
    --output-dir "$CURATED_DIR" \
    --resume \
    2>&1 | tee "$CURATED_DIR/v4_2_curation.log"
  jq -e '
    .status == "curated_bank_constructed_not_tensor_compiled"
    and .source_record_count == 24
    and .source_evidence_count == 167
    and .retained_record_count == 17
    and .retained_evidence_count == 116
    and .excluded_record_count == 7
    and .decision_counts == {
      "conditional": 6,
      "hard_reject": 4,
      "primary": 11,
      "quarantine": 3
    }
    and .api_key_read == false
    and .external_api_calls_made == 0
  ' "$CURATED_DIR/curation_report.json" >/dev/null \
    || fail "curated report validation failed"
fi

if [[ "$STAGE" == "side-kv" || "$STAGE" == "all" ]]; then
  for required in \
    "$CURATED_DIR/bank_records.jsonl" \
    "$CURATED_DIR/bank_manifest.json" \
    "$E0_DIR/side_kv_manifest.json"; do
    [[ -s "$required" ]] || fail "missing side-KV input: $required"
  done
  mkdir -p "$SIDE_KV_DIR"
  "$PYTHON_BIN" scripts/compile_v4_side_kv.py \
    --bank-records "$CURATED_DIR/bank_records.jsonl" \
    --bank-manifest "$CURATED_DIR/bank_manifest.json" \
    --reasoner-manifest "$E0_DIR/side_kv_manifest.json" \
    --output-dir "$SIDE_KV_DIR" \
    --layer 24 \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    2>&1 | tee "$SIDE_KV_DIR/v4_2_side_kv_compile.log"
  jq -e '
    .status == "side_kv_compilation_passed"
    and .qualified_for_online_target_loading == true
    and .configuration.layer_number == 24
    and .configuration.relative_phase_delta == 0
    and .configuration.target_online_only == true
    and .counts.bank_count == 17
    and .counts.target_count == 17
    and .counts.reference_count == 17
  ' "$SIDE_KV_DIR/v4_side_kv_compile_report.json" >/dev/null \
    || fail "side-KV compile report validation failed"
fi

if [[ "$STAGE" == "selector" || "$STAGE" == "all" ]]; then
  for required in \
    "$PHASE1_DIR/verified_experiences.jsonl" \
    "$PHASE1_DIR/split_manifest.json" \
    "$CURATED_DIR/bank_records.jsonl" \
    "$CURATED_DIR/bank_manifest.json" \
    "$SIDE_KV_DIR/v4_side_kv_manifest.json" \
    "$TOKEN_RISK_ARTIFACT"; do
    [[ -s "$required" ]] || fail "missing selector-anchor input: $required"
  done
  mkdir -p "$SELECTOR_DIR"
  "$PYTHON_BIN" scripts/compile_v4_selector_anchors.py \
    --experiences "$PHASE1_DIR/verified_experiences.jsonl" \
    --split-manifest "$PHASE1_DIR/split_manifest.json" \
    --bank-records "$CURATED_DIR/bank_records.jsonl" \
    --bank-manifest "$CURATED_DIR/bank_manifest.json" \
    --side-kv-manifest "$SIDE_KV_DIR/v4_side_kv_manifest.json" \
    --token-risk-artifact "$TOKEN_RISK_ARTIFACT" \
    --output-dir "$SELECTOR_DIR" \
    --dataset-revision main \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    2>&1 | tee "$SELECTOR_DIR/v4_2_selector_anchor_compile.log"
  jq -e '
    .status == "selector_anchor_compilation_passed"
    and .qualified_for_online_use == true
    and .source_bank_count == 17
    and .qualified_bank_count > 0
    and .qualified_bank_count <= .source_bank_count
    and .calibration.qualified == true
  ' "$SELECTOR_DIR/v4_selector_anchor_compile_report.json" >/dev/null \
    || fail "selector-anchor compile report validation failed"
fi

echo "[v4.2-curated-offline] PASS stage=$STAGE"
echo "[v4.2-curated-offline] curated_bank=$CURATED_DIR"
echo "[v4.2-curated-offline] side_kv=$SIDE_KV_DIR"
echo "[v4.2-curated-offline] selector_anchors=$SELECTOR_DIR"
