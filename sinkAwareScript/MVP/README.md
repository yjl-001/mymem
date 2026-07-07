# Sink-aware MVP for KodCode

This directory contains a narrow validation script for the question:

```text
Among delimiter candidate positions, does SinkMassZ select more useful latent-memory
insertion points than first-K or random delimiter selection?
```

The script is intentionally standalone. It does not modify the MemGen training or
generation code paths.

## Module Layout

- `run_kodcode_sink_mvp.py`: experiment orchestration.
- `cli.py`: command-line arguments, seed setup, config overrides.
- `model_setup.py`: KodCode dataset, MemGen loading, prompt encoding, reward.
- `generation.py`: batch-size-1 forced-step generation and sink/entropy capture.
- `sink_metrics.py`: SinkMass, entropy, candidate z-score normalization.
- `strategies.py`: first-K, random, same-bucket random, sink-based selectors.
- `candidate_selectors.py`: candidate-level random and same-bucket baselines.
- `records.py`: output dataclasses.
- `outputs.py`: JSONL, CSV, and summary writers.
- `visualize_results.py`: plotting entry point for a completed output directory.
- `plot_utils.py` / `viz_io.py`: plotting and result-loading helpers.
- `repo_paths.py`: repo-root import setup for standalone execution.

## What It Runs

For each KodCode sample and each reference mode, the script:

1. Loads a MemGen model and a Weaver checkpoint.
2. Runs a reference generation:
   - `no_memory`: no prompt latent and no inference latent.
   - `prompt_only`: prompt latent only, no inference latent.
3. Records delimiter candidate positions during generation.
4. Computes `SinkMassZ` and entropy at those candidate positions.
5. Optionally runs single-insertion counterfactual branches:

```text
U(j) = reward(force one inference latent at delimiter candidate j) - reward(baseline)
```

6. Optionally runs exploratory budget-matched strategy rollouts with
   `--run-strategy-rollouts`:

```text
first_k
random
same_bucket_random
sink_top_b
sink_entropy_top_b
```

## Example

```bash
conda activate memgen
python sinkAwareScript/MVP/run_kodcode_sink_mvp.py \
  --cfg-path configs/latent_memory/kodcode.yaml \
  --load-model-path MemGen/Qwen2.5-1.5B-Instruct/kodcode/weaver-sft/pn=1_pl=4_in=5_il=4/model \
  --output-dir output/sink_aware_mvp/kodcode_debug \
  --limit 20 \
  --max-new-tokens 256 \
  --budget 5 \
  --max-candidates-per-sample 20 \
  --attn-implementation eager \
  --reference-modes no_memory prompt_only \
  --overwrite
```

Use a small `--limit` first. The counterfactual branches are expensive because
each delimiter candidate can trigger an additional generation.

## Outputs

The output directory contains:

- `candidate_rows.jsonl`: one row per delimiter candidate, including `sink_mass`,
  `sink_mass_z`, entropy, optional single-insertion reward, and `U(j)`.
- `strategy_rows.jsonl`: optional, one row per sample and strategy. `planned_steps`
  are selected from baseline candidates; `inserted_steps` are the positions
  that were actually inserted during rollout.
- `summary.json`: aggregate reward and counterfactual ranking metrics.
  Candidate metrics are split by reference mode under `candidate_by_reference`.

## Interpretation

The narrow question is supported if one of these holds under the same memory
budget:

```text
Avg U@B(sink_top_b) > Avg U@B(first_k)
Avg U@B(sink_top_b) > Avg U@B(random)
On effective samples, Avg U@B(sink_top_b) remains higher than baselines
```

If `sink_top_b` only beats global random but not `same_bucket_random`, the signal
is probably mostly position bias.

Strategy reward is exploratory in this MVP. Do not use it as the main claim
until true online policies are implemented.

`rel_pos` and `pos_bucket` are computed against the actual generated length, not
`max_new_tokens`, so same-bucket random controls the observed generation region.

## Visualization

After an experiment finishes:

```bash
python sinkAwareScript/MVP/visualize_results.py \
  --output-dir output/sink_aware_mvp/qwen2_5_kodcode_sft_posfix_limit20
```

This writes PNG figures and `report.md` under `<output-dir>/figures/`.
