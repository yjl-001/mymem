#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
unset DEEPSEEK_API_KEY GLM_API_KEY OPENAI_API_KEY ANTHROPIC_API_KEY || true
OUTPUT_ROOT="${MEMGEN_OUTPUT_ROOT:-/data/memgen-runs}"
V4_ROOT="${MEMGEN_V4_OUTPUT_ROOT:-$OUTPUT_ROOT/v4}"
RECOVERY_ID="${MEMGEN_V4_RECOVERY_ID:-gsm8k-v4-packet-replay-20260907-r1}"
LINEAGE_ROOT="${MEMGEN_V4_RECOVERY_LINEAGE_ROOT:-$OUTPUT_ROOT/lineages/gsm8k-recovery/$RECOVERY_ID}"
export CUDA_VISIBLE_DEVICES="${MEMGEN_V43_CUDA_VISIBLE_DEVICES:-${MEMGEN_V4_CUDA_VISIBLE_DEVICES:-0}}"
FLAGS=(--resume)
if [[ "${MEMGEN_V43_VALIDATE_ONLY:-0}" == "1" ]]; then FLAGS=(--validate-only); fi
exec "${MEMGEN_PYTHON_BIN:-python}" scripts/run_v4_3_question_selector.py \
  --bank-dir "${MEMGEN_V43_BANK_DIR:-$V4_ROOT/offline/construction_v4_3_deepseek_prompt_v3}" \
  --side-kv-dir "${MEMGEN_V43_SIDE_KV_DIR:-$V4_ROOT/offline/side_kv_v4_3_deepseek_prompt_v3}" \
  --semantic-packets "${MEMGEN_V4_SEMANTIC_PACKETS:-$V4_ROOT/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl}" \
  --cache-manifest "${MEMGEN_V43_CACHE_MANIFEST:-$LINEAGE_ROOT/v4_oracle_full/source_state_cache/v4_source_state_manifest.json}" \
  --token-risk-artifact "${MEMGEN_V43_RISK_ARTIFACT:-$LINEAGE_ROOT/risk_v3_4/token-entropy-risk-gate-v3.4.pt}" \
  --equivalence-dir "${MEMGEN_V43_EQUIVALENCE_ROOT:-$V4_ROOT/offline/v4_3_prefix_equivalence}" \
  --split-manifest "${MEMGEN_V43_SELECTOR_SPLIT_MANIFEST:-$LINEAGE_ROOT/recovery/split_manifest.json}" \
  --output-dir "${MEMGEN_V43_SELECTOR_ROOT:-$V4_ROOT/offline/v4_3_question_selector}" \
  --train-size "${MEMGEN_V43_SELECTOR_TRAIN_SIZE:-200}" \
  --tune-size "${MEMGEN_V43_SELECTOR_TUNE_SIZE:-100}" \
  --eval-size "${MEMGEN_V43_SELECTOR_EVAL_SIZE:-100}" \
  --device "${MEMGEN_V43_DEVICE:-cuda}" "${FLAGS[@]}" "$@"
