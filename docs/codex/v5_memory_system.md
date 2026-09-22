# MemGen V5: applicability-keyed native-prefix memory

V5 fixes the online consumer to one complete, all-layer native prefix KV Bank selected before
reasoning. It separates the question-only selector key from Transformer attention K/V:

```text
question -> positive applicability Top-K -> exclusion-aware reranker
         -> calibrated expected utility / no_memory
         -> one complete Memory Card native prefix KV -> frozen reasoner
```

The construction flow is dataset-independent after a task emits an Episode with `input`, `output`
and `outcome`. Task code still owns prompt construction and result verification. Memory code owns
contrasts, Experience Atoms, semantic grouping, cards, qualification, compilation and selection.

## Frozen contracts

- train builds Banks, valid calibrates the selector, official test is evaluation only;
- one greedy plus seven temperature 0.8/top-p 0.95 rollouts, maximum 1,024 completion tokens;
- truncation is neither success nor failure;
- success requires verifier success, valid format, and a correct process review;
- up to one answer, process and format failure contrast per input;
- Primary requires at least three distinct train inputs, a semantic critic pass, and no protocol fallback;
- embedding retrieval uses only positive input-observable applicability;
- exclusions are separate reranker metadata and are never concatenated into the embedding key;
- the entire card, rather than selector fields alone, is compiled into native attention K/V;
- one Primary Bank or `no_memory` is selected for each input.

## Server workflow

Collect Episodes without a teacher:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/experiments/gsm8k/run_v5_memory.sh \
  --phase rollouts --config configs/experiments/gsm8k/v5.json \
  --output-dir output/experiments/v5/rollouts/gsm8k-r1 --rollout-batch-size 32
```

Start the local Qwen3-32B teacher on a selectable four-GPU set:

```bash
PATH="$PWD/.venv-vllm/bin:$PATH" .venv-vllm/bin/python scripts/serve_v5_teacher.py \
  --config configs/experiments/gsm8k/v5.json --gpus 0,1,2,3 \
  --max-num-seqs 32 --gpu-memory-utilization 0.8
```

Construct and calibrate a separate immutable Bank run:

```bash
CUDA_VISIBLE_DEVICES=4 bash scripts/experiments/gsm8k/run_v5_memory.sh \
  --phase bank --config configs/experiments/gsm8k/v5.json \
  --rollout-source output/experiments/v5/rollouts/gsm8k-r1 \
  --output-dir output/experiments/v5/banks/gsm8k-r1 --teacher-concurrency 16
```

Run the frozen official test:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_v5_memory.py \
  --bank-dir output/experiments/v5/banks/gsm8k-r1 \
  --output-dir output/experiments/v5/evaluation/gsm8k-test-r1 --split test
```

Answer one input:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/use_v5_memory.py \
  --bank-dir output/experiments/v5/banks/gsm8k-r1 --input "A new problem..."
```

Every stage is immutable and resumable. Use `--resume` only with identical code, configuration and
runtime versions. Use a new output directory after any prompt, schema, model or implementation
change. `--stage` is reserved for recovery; normal runs should use the two phases above.
