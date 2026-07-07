#!/usr/bin/env python3
"""Run the KodCode sink-aware latent-memory MVP."""

from pathlib import Path
import logging
import shutil

from omegaconf import OmegaConf

import repo_paths  # noqa: F401
from candidate_experiment import run_reference, run_single_insertion_branches, run_strategy_rollouts
from cli import apply_overrides, parse_args, set_seed
from model_setup import build_dataset, encode_prompt, load_model
from outputs import append_strategy_csv, write_dataclass_jsonl, write_summary
from strategies import summarize


LOGGER = logging.getLogger("sink-aware-mvp")


def prepare_output_paths(output_dir: Path) -> dict[str, Path]:
    """集中管理输出路径，避免主流程里散落字符串。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "candidate": output_dir / "candidate_rows.jsonl",
        "strategy": output_dir / "strategy_rows.jsonl",
        "summary": output_dir / "summary.json",
        "strategy_csv": output_dir / "strategy_rows.csv",
    }


def maybe_clear_outputs(paths: dict[str, Path], fig_dir: Path, overwrite: bool) -> None:
    """避免复用 output-dir 时 JSONL/CSV 追加污染新实验。"""

    if not overwrite:
        return
    for path in paths.values():
        if path.exists():
            path.unlink()
    if fig_dir.exists():
        shutil.rmtree(fig_dir)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)

    paths = prepare_output_paths(Path(args.output_dir))
    maybe_clear_outputs(paths, Path(args.output_dir) / "figures", args.overwrite)
    config = apply_overrides(OmegaConf.load(args.cfg_path), args.options)
    config.dataset.mode = "sft"
    config.model.max_inference_aug_num = args.budget

    LOGGER.info("Loading KodCode dataset...")
    dataset = build_dataset(config, args.split)
    end = min(args.start_index + args.limit, len(dataset))
    sample_indices = list(range(args.start_index, end))

    LOGGER.info("Loading MemGen model...")
    model = load_model(config, args)
    all_candidate_rows = []
    all_strategy_rows = []

    for local_idx, sample_idx in enumerate(sample_indices, start=1):
        sample = dataset[sample_idx]
        LOGGER.info("Sample %s/%s: dataset index %s", local_idx, len(sample_indices), sample_idx)
        prompt_ids, prompt_mask = encode_prompt(model, sample, args.device)

        for reference_mode in args.reference_modes:
            baseline, reference_reward, candidates = run_reference(
                model, prompt_ids, prompt_mask, sample, sample_idx, reference_mode, args
            )
            if not candidates:
                LOGGER.info("No delimiter candidates found for sample=%s ref=%s", sample_idx, reference_mode)

            run_single_insertion_branches(
                model, prompt_ids, prompt_mask, sample, sample_idx, reference_mode, candidates, reference_reward, args
            )
            strategy_rows = []
            if args.run_strategy_rollouts:
                strategy_rows = run_strategy_rollouts(
                    model, prompt_ids, prompt_mask, sample, sample_idx,
                    reference_mode, baseline, reference_reward, candidates, args
                )

            all_candidate_rows.extend(candidates)
            all_strategy_rows.extend(strategy_rows)
            write_dataclass_jsonl(paths["candidate"], candidates)
            if strategy_rows:
                write_dataclass_jsonl(paths["strategy"], strategy_rows)
                append_strategy_csv(paths["strategy_csv"], strategy_rows)

        summary = summarize(all_candidate_rows, all_strategy_rows, args.budget, args.random_trials)
        write_summary(paths["summary"], summary)
        LOGGER.info("Running summary: %s", summary)

    LOGGER.info("Done. Outputs written to %s", args.output_dir)


if __name__ == "__main__":
    main()
