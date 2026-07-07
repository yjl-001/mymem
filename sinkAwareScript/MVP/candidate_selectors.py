import random

from records import CandidateRecord


def stable_seed(ref: str, sample_idx: int, trial_idx: int) -> int:
    """稳定随机种子，避免 Python hash 随进程变化。"""

    return sum(ord(ch) for ch in ref) * 1_000_003 + sample_idx * 9176 + trial_idx


def select_random_candidates(rows: list[CandidateRecord], ref: str, budget: int, trial_idx: int):
    """candidate-level random baseline。"""

    rng = random.Random(stable_seed(ref, rows[0].sample_idx, trial_idx))
    if len(rows) <= budget:
        return rows
    return rng.sample(rows, budget)


def select_same_bucket_random_candidates(rows: list[CandidateRecord], ref: str, budget: int, trial_idx: int):
    """匹配 sink_top_b 位置分桶分布的 candidate-level random baseline。"""

    sink_rows = sorted(rows, key=lambda r: r.sink_mass_z, reverse=True)[:budget]
    bucket_counts: dict[int, int] = {}
    for row in sink_rows:
        bucket_counts[row.pos_bucket] = bucket_counts.get(row.pos_bucket, 0) + 1

    rng = random.Random(stable_seed(ref, rows[0].sample_idx, trial_idx) + 31)
    selected = []
    for bucket, count in bucket_counts.items():
        pool = [row for row in rows if row.pos_bucket == bucket]
        selected.extend(pool if len(pool) <= count else rng.sample(pool, count))
    return selected
