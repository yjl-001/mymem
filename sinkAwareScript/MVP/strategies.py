import random
from collections.abc import Iterable

from candidate_selectors import select_random_candidates, select_same_bucket_random_candidates
from records import CandidateRecord, StrategyRecord


def top_sink_steps(candidates: list[CandidateRecord], budget: int, *, use_entropy: bool = False) -> list[int]:
    """选择 SinkMassZ 最高的 B 个候选点。

    use_entropy=True 时使用 SinkMassZ + EntropyZ，验证“不确定性”是否能提升
    sink signal 的排序能力。
    """

    if use_entropy:
        scored = sorted(candidates, key=lambda c: c.sink_mass_z + c.entropy_z, reverse=True)
    else:
        scored = sorted(candidates, key=lambda c: c.sink_mass_z, reverse=True)
    return [c.step for c in scored[:budget]]


def first_k_steps(candidates: list[CandidateRecord], budget: int) -> list[int]:
    """原始 MemGen 风格近似：取前 B 个 delimiter 候选点。"""

    return [c.step for c in candidates[:budget]]


def random_steps(candidates: list[CandidateRecord], budget: int, rng: random.Random) -> list[int]:
    """全局 random delimiter baseline。"""

    if len(candidates) <= budget:
        return [c.step for c in candidates]
    return [c.step for c in rng.sample(candidates, budget)]


def same_bucket_random_steps(candidates: list[CandidateRecord], reference_steps: Iterable[int],
                             rng: random.Random) -> list[int]:
    """位置分桶 random baseline。

    如果 sink_top_b 只赢全局 random，不赢 same-bucket random，说明它可能只是
    借了“更偏前/更偏后”的位置分布，而不是 attention signal 真有用。
    """

    by_step = {c.step: c for c in candidates}
    bucket_counts: dict[int, int] = {}
    for step in reference_steps:
        if step in by_step:
            bucket = by_step[step].pos_bucket
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    selected: list[int] = []
    for bucket, count in bucket_counts.items():
        pool = [c for c in candidates if c.pos_bucket == bucket]
        if len(pool) <= count:
            selected.extend(c.step for c in pool)
        else:
            selected.extend(c.step for c in rng.sample(pool, count))
    return selected


def build_strategy_steps(candidates: list[CandidateRecord], budget: int, random_trials: int,
                         rng: random.Random) -> dict[str, list[int]]:
    """为单个样本构造所有待评测策略的 forced step 集合。"""

    budget = min(budget, len(candidates))
    strategy_to_steps: dict[str, list[int]] = {
        "first_k": first_k_steps(candidates, budget),
        "sink_top_b": top_sink_steps(candidates, budget, use_entropy=False),
        "sink_entropy_top_b": top_sink_steps(candidates, budget, use_entropy=True),
    }
    for trial_idx in range(random_trials):
        strategy_to_steps[f"random_{trial_idx}"] = random_steps(candidates, budget, rng)

    sink_ref_steps = strategy_to_steps["sink_top_b"]
    for trial_idx in range(random_trials):
        strategy_to_steps[f"same_bucket_random_{trial_idx}"] = same_bucket_random_steps(candidates, sink_ref_steps, rng)
    return strategy_to_steps


def summarize(candidate_rows: list[CandidateRecord], strategy_rows: list[StrategyRecord],
              budget: int, random_trials: int = 0) -> dict:
    """汇总两类指标。

    strategy_reward_mean 看完整 rollout 的最终 reward；
    counterfactual_avg_utility_at_budget 看“单点插入收益 U(j)”的排序质量。
    """

    summary: dict = {
        "num_candidate_rows": len(candidate_rows),
        "num_strategy_rows": len(strategy_rows),
        "budget": budget,
        "strategy_reward_mean": {},
        "strategy_actual_insertions_mean": {},
        "candidate_by_reference": {},
    }

    for name in sorted({row.strategy for row in strategy_rows}):
        for ref in sorted({row.reference_mode for row in strategy_rows if row.strategy == name}):
            key = f"{ref}/{name}"
            rows = [row for row in strategy_rows if row.strategy == name and row.reference_mode == ref]
            rewards = [row.reward for row in rows]
            insertions = [len(row.inserted_steps) for row in rows]
            summary["strategy_reward_mean"][key] = sum(rewards) / len(rewards) if rewards else None
            summary["strategy_actual_insertions_mean"][key] = sum(insertions) / len(insertions) if insertions else None

    rows_by_ref_sample: dict[tuple[str, int], list[CandidateRecord]] = {}
    for row in candidate_rows:
        if row.utility is not None:
            rows_by_ref_sample.setdefault((row.reference_mode, row.sample_idx), []).append(row)

    selectors = {
        "first_k": lambda rows: rows[:budget],
        "sink_top_b": lambda rows: sorted(rows, key=lambda r: r.sink_mass_z, reverse=True)[:budget],
        "sink_entropy_top_b": lambda rows: sorted(rows, key=lambda r: r.sink_mass_z + r.entropy_z, reverse=True)[:budget],
    }
    for ref in sorted({key[0] for key in rows_by_ref_sample}):
        ref_groups = [rows for (mode, _), rows in rows_by_ref_sample.items() if mode == ref]
        ref_summary = {
            "num_samples": len(ref_groups),
            "num_effective_samples": sum(any(float(row.utility) != 0.0 for row in rows) for rows in ref_groups),
            "reference_reward_mean": None,
            "counterfactual_avg_utility_at_budget": {},
            "counterfactual_avg_utility_at_budget_effective": {},
            "positive_precision_at_budget": {},
        }
        ref_rewards = [rows[0].reference_reward for rows in ref_groups if rows]
        ref_summary["reference_reward_mean"] = sum(ref_rewards) / len(ref_rewards) if ref_rewards else None

        for key, selector in selectors.items():
            all_values, eff_values, precisions = [], [], []
            for rows in ref_groups:
                selected = selector(rows)
                if not selected:
                    continue
                avg_utility = sum(float(row.utility) for row in selected) / len(selected)
                precision = sum(float(row.utility) > 0.0 for row in selected) / len(selected)
                all_values.append(avg_utility)
                precisions.append(precision)
                if any(float(row.utility) != 0.0 for row in rows):
                    eff_values.append(avg_utility)
            ref_summary["counterfactual_avg_utility_at_budget"][key] = (
                sum(all_values) / len(all_values) if all_values else None
            )
            ref_summary["counterfactual_avg_utility_at_budget_effective"][key] = (
                sum(eff_values) / len(eff_values) if eff_values else None
            )
            ref_summary["positive_precision_at_budget"][key] = (
                sum(precisions) / len(precisions) if precisions else None
            )
        add_random_candidate_metrics(ref_summary, ref_groups, ref, budget, random_trials)
        summary["candidate_by_reference"][ref] = ref_summary
    return summary


def add_random_candidate_metrics(ref_summary: dict, ref_groups: list[list[CandidateRecord]], ref: str,
                                 budget: int, random_trials: int) -> None:
    """给 candidate-level summary 增加 random / same-bucket random 均值。"""

    if random_trials <= 0:
        return
    for name, selector in [
        ("random", select_random_candidates),
        ("same_bucket_random", select_same_bucket_random_candidates),
    ]:
        all_values, eff_values, precisions = [], [], []
        for rows in ref_groups:
            for trial_idx in range(random_trials):
                selected = selector(rows, ref, budget, trial_idx)
                if not selected:
                    continue
                avg_utility = sum(float(row.utility) for row in selected) / len(selected)
                precision = sum(float(row.utility) > 0.0 for row in selected) / len(selected)
                all_values.append(avg_utility)
                precisions.append(precision)
                if any(float(row.utility) != 0.0 for row in rows):
                    eff_values.append(avg_utility)
        ref_summary["counterfactual_avg_utility_at_budget"][name] = (
            sum(all_values) / len(all_values) if all_values else None
        )
        ref_summary["counterfactual_avg_utility_at_budget_effective"][name] = (
            sum(eff_values) / len(eff_values) if eff_values else None
        )
        ref_summary["positive_precision_at_budget"][name] = (
            sum(precisions) / len(precisions) if precisions else None
        )

