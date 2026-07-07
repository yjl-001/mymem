import logging
import random

from generation import generate_with_forced_steps
from model_setup import reward_completion
from records import StrategyRecord
from strategies import build_strategy_steps


LOGGER = logging.getLogger("sink-aware-mvp")


def prompt_augment_for(reference_mode: str) -> bool:
    """reference_mode -> 是否在 prompt 后插入 prompt latent。"""

    return reference_mode == "prompt_only"


def run_reference(model, prompt_ids, prompt_mask, sample, sample_idx, reference_mode: str, args):
    """生成一条 reference 轨迹并记录 delimiter candidates。"""

    args.current_reference_mode = reference_mode
    baseline = generate_with_forced_steps(
        model,
        prompt_ids,
        prompt_mask,
        sample_idx=sample_idx,
        forced_steps=set(),
        args=args,
        prompt_augment=prompt_augment_for(reference_mode),
        collect_candidates=True,
    )
    reference_reward = reward_completion(baseline.completion, sample)
    candidates = baseline.candidates[: args.max_candidates_per_sample]
    for candidate in candidates:
        candidate.reference_reward = reference_reward
    return baseline, reference_reward, candidates


def run_single_insertion_branches(model, prompt_ids, prompt_mask, sample, sample_idx, reference_mode,
                                  candidates, reference_reward, args):
    """对每个候选点单独做一次反事实插入。"""

    if args.skip_single_insertion_branches:
        return

    for candidate in candidates:
        branch = generate_with_forced_steps(
            model,
            prompt_ids,
            prompt_mask,
            sample_idx=sample_idx,
            forced_steps={candidate.step},
            args=args,
            prompt_augment=prompt_augment_for(reference_mode),
            collect_candidates=False,
        )
        branch_reward = reward_completion(branch.completion, sample)
        candidate.branch_reward = branch_reward
        candidate.utility = branch_reward - reference_reward
        candidate.branch_forced_steps_used = branch.forced_steps_used
        candidate.branch_inserted = candidate.step in branch.forced_steps_used
        if not candidate.branch_inserted:
            LOGGER.warning(
                "Candidate branch did not insert at sample=%s ref=%s step=%s, actual=%s",
                sample_idx,
                reference_mode,
                candidate.step,
                branch.forced_steps_used,
            )


def run_strategy_rollouts(model, prompt_ids, prompt_mask, sample, sample_idx, reference_mode,
                          baseline, reference_reward, candidates, args):
    """探索性多点策略 rollout；主结论不要依赖这个指标。"""

    rng = random.Random(args.seed + sample_idx)
    strategy_to_steps = build_strategy_steps(candidates, args.budget, args.random_trials, rng)
    rows = [
        StrategyRecord(
            sample_idx=sample_idx,
            reference_mode=reference_mode,
            strategy="reference",
            reward=reference_reward,
            planned_steps=[],
            inserted_steps=[],
            completion=baseline.completion,
        )
    ]

    for strategy, steps in strategy_to_steps.items():
        trace = generate_with_forced_steps(
            model,
            prompt_ids,
            prompt_mask,
            sample_idx=sample_idx,
            forced_steps=set(steps),
            args=args,
            prompt_augment=prompt_augment_for(reference_mode),
            collect_candidates=False,
        )
        rows.append(
            StrategyRecord(
                sample_idx=sample_idx,
                reference_mode=reference_mode,
                strategy=strategy,
                reward=reward_completion(trace.completion, sample),
                planned_steps=steps,
                inserted_steps=trace.forced_steps_used,
                completion=trace.completion,
            )
        )
    return rows
