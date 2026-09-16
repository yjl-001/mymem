"""Outcome-based diagnostics of frozen rankings, never a deployable selector."""
from collections import Counter
from math import comb

import numpy as np

from memgen.experience.v4_3_question_selector import NO_MEMORY
from memgen.experience.v4_3_similarity_study import KEYS, metrics, score_features, validate_rows

POLICY = {
    "version": "v43-retrieval-coverage-v1",
    "ranking": "cosine_desc_bank_id_ascending_no_abstention",
    "repairable": "baseline_wrong_and_at_least_one_bank_correct",
    "recall_denominator": "repairable_questions_only",
    "random_reference": "exact_uniform_k_distinct_banks_without_replacement",
    "oracle": "best_of_no_memory_and_top_k_banks_after_observing_rewards",
    "gold_used_for_diagnostics_only": True, "deployable_oracle": False,
    "new_answer_generations": 0, "new_key_encodings": 0,
    "official_test_used": False, "parameter_fitting": False,
}


def coverage(rows, bank_ids, vectors, split):
    if split not in ("train", "tune"):
        raise ValueError("Coverage diagnostics allow train/tune only")
    validate_rows(rows, bank_ids, split)
    if set(vectors) != set(KEYS):
        raise ValueError("Expected all frozen key variants")
    n, b = len(rows), len(bank_ids)
    base_correct = sum(int(r["rewards"][NO_MEMORY]) for r in rows)
    eligible = [r for r in rows if not r["rewards"][NO_MEMORY] and any(r["rewards"][bid] for bid in bank_ids)]
    repairable = len(eligible)
    good_counts = [sum(int(r["rewards"][bid]) for bid in bank_ids) for r in eligible]
    random_recall = {
        k: (sum(1 - (comb(b-m, k) / comb(b, k) if b-m >= k else 0.) for m in good_counts) / repairable
            if repairable else None)
        for k in range(1, b+1)
    }
    result = {
        "count": n, "baseline": metrics(rows, [NO_MEMORY]*n),
        "baseline_wrong_count": n-base_correct, "repairable_count": repairable,
        "unrepairable_baseline_wrong_count": n-base_correct-repairable,
        "all_bank_oracle_accuracy": (base_correct+repairable)/n,
        "correct_bank_count_among_repairable": {str(k): v for k, v in sorted(Counter(good_counts).items())},
        "keys": {},
    }
    for key in KEYS:
        scores = score_features([r["feature"] for r in rows], bank_ids, vectors[key])
        # Stable sort preserves ascending bank_id for exact score ties, matching old argmax.
        order = np.argsort(-scores, axis=1, kind="stable")
        cases, repair_ranks, harm_cases = [], [], []
        for r, indices, row_scores in zip(rows, order, scores):
            ranked = [bank_ids[int(j)] for j in indices]
            good = [rank for rank, bid in enumerate(ranked, 1) if r["rewards"][bid]]
            first_good = min(good) if good else None
            baseline = int(r["rewards"][NO_MEMORY])
            if not baseline and first_good is not None:
                repair_ranks.append(first_good)
            harm = bool(baseline and not r["rewards"][ranked[0]])
            if harm:
                harm_cases.append(first_good)
            cases.append({"sample_id": r["sample_id"], "baseline_correct": baseline,
                          "ranked_bank_ids": ranked, "ranked_scores": [float(row_scores[j]) for j in indices],
                          "correct_bank_ranks": good, "first_correct_bank_rank": first_good,
                          "repairable": not baseline and first_good is not None, "top1_harm": harm})
        top1_hits = sum(rank == 1 for rank in repair_ranks)
        curve = {}
        for k in range(1, b+1):
            hit = sum(rank <= k for rank in repair_ranks)
            curve[str(k)] = {
                "repair_hit_count": hit, "repair_recall": hit/repairable if repairable else None,
                "additional_repairs_over_top1": hit-top1_hits,
                "fraction_of_top1_missed_repairs_recovered": ((hit-top1_hits)/(repairable-top1_hits)
                                                             if repairable > top1_hits else None),
                "uniform_random_expected_recall": random_recall[k],
                "oracle_accuracy_including_no_memory": (base_correct+hit)/n,
                "top1_harm_with_safe_bank_in_top_k": sum(rank is not None and rank <= k for rank in harm_cases),
                "top_k_boundary_tie_count": int(sum(
                    row_scores[indices[k-1]] == row_scores[indices[k]]
                    for row_scores, indices in zip(scores, order)
                )) if k < b else 0,
            }
        result["keys"][key] = {
            "top1": metrics(rows, [c["ranked_bank_ids"][0] for c in cases]),
            "repair_first_success_rank_histogram": {str(k): repair_ranks.count(k) for k in range(1, b+1)},
            "repair_mean_reciprocal_rank": sum(1/rank for rank in repair_ranks)/repairable if repairable else None,
            "top1_harm_count": len(harm_cases),
            "top1_harm_all_banks_wrong_count": sum(rank is None for rank in harm_cases),
            "top_k": curve, "cases": cases,
        }
    return result


def compact(split_result):
    output = {k: split_result[k] for k in ("count", "baseline_wrong_count", "repairable_count",
              "unrepairable_baseline_wrong_count", "all_bank_oracle_accuracy")}
    output["baseline_accuracy"] = split_result["baseline"]["accuracy"]
    output["keys"] = {}
    for key, data in split_result["keys"].items():
        curve = data["top_k"]
        selected_k = sorted({1, min(3, len(curve)), min(5, len(curve)), len(curve)})
        output["keys"][key] = {
            "top1_accuracy": data["top1"]["accuracy"],
            "first_success_rank_counts": data["repair_first_success_rank_histogram"],
            "repair_hits_at_k": {str(k): curve[str(k)]["repair_hit_count"] for k in selected_k},
            "repair_recall_at_k": {str(k): curve[str(k)]["repair_recall"] for k in selected_k},
            "random_expected_recall_at_k": {str(k): curve[str(k)]["uniform_random_expected_recall"] for k in selected_k},
            "top1_harm": data["top1_harm_count"],
            "top1_harm_safe_alternative_in_top3": curve[str(min(3, len(curve)))]["top1_harm_with_safe_bank_in_top_k"],
            "top1_harm_all_banks_wrong": data["top1_harm_all_banks_wrong_count"],
        }
    return output
