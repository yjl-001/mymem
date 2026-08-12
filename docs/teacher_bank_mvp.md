# Teacher-constructed bank MVP

`scripts/build_teacher_bank.py` is an offline utility for inspecting what a
strong-teacher-reflected target/reference bank looks like. It does not modify
the MemGen reasoner, Weaver, Trigger, or inference path.

## GSM8K preview

Configure `DEEPSEEK_API_KEY` in the untracked
`scripts/experiments/.server.env`, then run:

```bash
bash scripts/experiments/gsm8k/build_teacher_bank_preview.sh
```

Set `DEEPSEEK_BASE_URL="https://api.deepseek.com"`. The preview defaults to
`DEEPSEEK_THINKING="disabled"` so that the API returns a reliable final JSON
payload; this bank-building task does not need a visible chain-of-thought.

The script reads five **GSM8K train** examples and writes one JSON object per
line under `MEMGEN_OUTPUT_ROOT/banks/gsm8k/`. Each record contains a
teacher-generated target and reference experience record.

The source demonstrations are verified successes. GSM8K does not provide a
failed rollout in this preview, so every generated reference is labelled
`"reference_evidence": "teacher_inferred"`. These records are for schema and
quality inspection only; they must not be used as the reference bank for a
formal contrastive evaluation.

Teacher output now also includes a strict `quality` object. The preview still
does not become formal evidence merely because the teacher self-check passes.

## Generic episode input

For a verified rollout dataset, pass `--input-jsonl`. Each input line must
contain `context` and `trajectory`; optional fields are `id`, `source`,
`outcome`, `reward`, `feedback`, and `reference_trajectory`. If
`reference_trajectory` is supplied, output records are labelled
`"reference_evidence": "verified_failure"`.

This allows GSM8K, code, retrieval, and interactive-agent episodes to share a
single bank-building interface. The formal Phase 1 workflow supplies
verifier-backed student rollouts so that both target and reference records derive
from observed outcomes.

The formal GSM8K implementation is now available through
`scripts/experiments/gsm8k/run_phase1_verified_bank.sh`; see
`docs/gsm8k_phase1_verified_bank.md`. For any record claiming
`reference_evidence=verified_failure`, the builder rejects missing success/failure
verifier records, source episode IDs, `bank-source` provenance, or provenance hash.
