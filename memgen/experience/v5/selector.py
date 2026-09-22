"""Question-only V5 candidate retrieval, applicability scoring and utility policy."""
from __future__ import annotations

import math

FEATURE_NAMES = ("bias", "semantic", "reranker", "semantic_margin", "log_support",
                 "semantic_x_reranker")
NO_MEMORY = "no_memory"


def cosine(left, right):
    if len(left) != len(right) or not left or not all(math.isfinite(x) for x in [*left, *right]):
        raise ValueError("Invalid V5 selector vector")
    return sum(a*b for a, b in zip(left, right))


def retrieve(query_vector, entries, top_k):
    ranked = sorted(((cosine(query_vector, entry["key_vector"]), entry["bank_id"], entry)
                     for entry in entries), key=lambda item: (-item[0], item[1]))
    return ranked[:min(top_k, len(ranked))]


def reranker_document(entry):
    exclusions = entry["selector_key"].get("exclusions_from_input", [])
    return entry["positive_selector_text"] + "\nExcluded when: " + ("; ".join(exclusions) or "none specified")


def candidate_features(ranked, reranker_scores):
    if not ranked:
        return []
    margin = ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 0.)
    result = []
    for semantic, bank_id, entry in ranked:
        rerank = reranker_scores[bank_id]["score"] if reranker_scores else semantic
        values = [1., semantic, rerank, margin, math.log1p(entry["distinct_input_count"]), semantic*rerank]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nonfinite V5 selector feature")
        result.append({"bank_id": bank_id, "semantic": semantic, "reranker": rerank,
                       "features": values})
    return result


def utility_score(features, weights):
    if len(features) != len(weights):
        raise ValueError("V5 utility feature/weight mismatch")
    value = sum(left*right for left, right in zip(features, weights))
    if not math.isfinite(value):
        raise ValueError("Nonfinite V5 utility score")
    return value


def select(candidates, policy):
    if not candidates:
        return {"bank_id": NO_MEMORY, "utility_score": None, "reason": "empty_bank"}
    scored = sorted(((utility_score(row["features"], policy["weights"]), row["bank_id"], row)
                     for row in candidates), key=lambda item: (-item[0], item[1]))
    score, bank_id, row = scored[0]
    if score <= policy["threshold"]:
        return {"bank_id": NO_MEMORY, "utility_score": score, "reason": "utility_abstention",
                "best_candidate": bank_id, "semantic": row["semantic"], "reranker": row["reranker"]}
    return {"bank_id": bank_id, "utility_score": score, "reason": "positive_expected_utility",
            "semantic": row["semantic"], "reranker": row["reranker"]}


def fit_ridge(rows, ridge):
    import numpy as np
    pairs = [(candidate["features"], candidate["utility"])
             for row in rows for candidate in row["candidates"]]
    if not pairs:
        return [0.] * len(FEATURE_NAMES)
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.
    weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    if not np.isfinite(weights).all():
        raise ValueError("V5 utility fit produced nonfinite weights")
    return weights.tolist()


def choose_threshold(rows, weights):
    best_per_row = [(row, sorted(((utility_score(c["features"], weights), c) for c in row["candidates"]),
                                  key=lambda item: (-item[0], item[1]["bank_id"]))[0]
                     if row["candidates"] else None) for row in rows]
    observed = sorted({score for _, best in best_per_row if best for score in [best[0]]})
    if observed:
        span = max(1., observed[-1] - observed[0])
        thresholds = [observed[0] - span, *observed, observed[-1] + span]
    else:
        thresholds = [0.]
    choices = []
    for threshold in thresholds:
        correct = gain = harm = used = 0
        for row, best in best_per_row:
            utility = 0
            if best is not None and best[0] > threshold:
                utility, used = best[1]["utility"], used + 1
            correct += row["baseline_reward"] + utility
            gain += utility == 1
            harm += utility == -1
        choices.append(((correct, gain-harm, -harm, -used, threshold), threshold))
    return max(choices, key=lambda item: item[0])[1]


def metrics(rows, predictions):
    correct = gain = harm = used = 0
    selections = {}
    for row, decision in zip(rows, predictions):
        candidate = next((c for c in row["candidates"] if c["bank_id"] == decision["bank_id"]), None)
        utility = candidate["utility"] if candidate else 0
        correct += row["baseline_reward"] + utility
        gain += utility == 1
        harm += utility == -1
        used += candidate is not None
        selections[decision["bank_id"]] = selections.get(decision["bank_id"], 0) + 1
    count = len(rows)
    return {"count": count, "correct": int(correct), "accuracy": correct/count if count else None,
            "gain": gain, "harm": harm, "net_gain": gain-harm, "memory_use_count": used,
            "selection_counts": selections}


def fit_policy(rows, ridge, folds, profile_sha256, *, top_k=None, reranker_enabled=None):
    if not rows:
        raise ValueError("V5 utility calibration requires valid rows")
    fold_count = min(folds, len(rows))
    oof = [None] * len(rows)
    for fold in range(fold_count):
        held = [index for index in range(len(rows)) if index % fold_count == fold]
        train = [row for index, row in enumerate(rows) if index not in held]
        if not train:
            train = rows
        weights = fit_ridge(train, ridge)
        threshold = choose_threshold(train, weights)
        fold_policy = {"weights": weights, "threshold": threshold}
        for index in held:
            oof[index] = select(rows[index]["candidates"], fold_policy)
    weights = fit_ridge(rows, ridge)
    threshold = choose_threshold(rows, weights)
    from memgen.experience.bank_construction.artifacts import digest
    body = {"schema_version": "memgen-v5-selector-policy-v1", "profile_sha256": profile_sha256,
        "query_access": "input-only", "candidate_retrieval": "positive-selector-key-cosine-top-k",
        "exclusions_embedded": False, "applicability_reranker": "qwen3-reranker" if any(
            candidate["reranker"] != candidate["semantic"] for row in rows for candidate in row["candidates"]) else "disabled",
        "utility_model": "ridge-net-gain", "feature_names": list(FEATURE_NAMES),
        "selector_top_k": top_k, "reranker_enabled": reranker_enabled,
        "weights": weights, "threshold": threshold, "ridge": ridge, "folds": fold_count,
        "oof_metrics": metrics(rows, oof)}
    return {**body, "policy_sha256": digest(body)}
