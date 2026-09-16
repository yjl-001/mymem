"""Train-only selection of similarity keys and abstention rules; frozen KV values."""
from collections import Counter

import numpy as np

from memgen.experience.v4_3_bank import authenticate, canonical_hash, seal
from memgen.experience.v4_3_question_selector import NO_MEMORY, matrix, paired_metrics

KEYS = ("full_card", "applicability", "problem_structure")
RULES = ("threshold", "threshold_margin")
POLICY = {
    "version": "v43-similarity-study-v1", "keys": list(KEYS), "rules": list(RULES),
    "quantiles": [.25, .5, .75, .9], "fold_count": 5, "fold_seed": 43,
    "threshold_source": "fit_partition_only", "family_selection": "train_out_of_fold",
    "rank": ["correct_desc", "harm_asc", "completion_tokens_asc", "memory_use_asc", "candidate_order"],
    "similarity_comparison": "top1_strictly_above_threshold",
    "margin_comparison": "top1_minus_top2_at_least_margin", "bank_tie_break": "bank_id_ascending",
    "abstention": "no_memory", "refit_on_tune": False, "value_and_consumer": "unchanged_native_prefix_kv",
}


def key_texts(records):
    """Only existing abstract card clauses; no evidence questions/answers in retrieval keys."""
    output = {key: {} for key in KEYS}
    for r in records:
        bid = r["bank_id"]
        if r["quality_tier"] != "primary" or bid in output["full_card"]:
            raise ValueError("Expected distinct primary Banks")
        values = (r["descriptor"], r["unified_process_card"]["applies_when"],
                  r["clause_support"]["problem_structure"]["text"])
        for key, value in zip(KEYS, values):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("Empty retrieval key")
            # Preserve the old descriptor exactly. encode_text strips whitespace itself.
            output[key][bid] = value
    return output


def score_features(features, bank_ids, vectors):
    if len(bank_ids) < 2 or bank_ids != sorted(set(bank_ids)) or NO_MEMORY in bank_ids or set(vectors) != set(bank_ids):
        raise ValueError("Invalid Bank namespace")
    x = np.asarray(features, dtype=np.float64)
    keys = np.asarray([vectors[b] for b in bank_ids], dtype=np.float64)
    if (x.ndim != 2 or keys.ndim != 2 or x.shape[1] != keys.shape[1]
            or not np.isfinite(x).all() or not np.isfinite(keys).all()
            or not np.allclose(np.linalg.norm(x, axis=1), 1., atol=1e-5)
            or not np.allclose(np.linalg.norm(keys, axis=1), 1., atol=1e-5)):
        raise ValueError("Similarity requires matching finite unit feature vectors")
    return x @ keys.T


def top_scores(scores):
    top = np.argmax(scores, axis=1)
    maximum = np.max(scores, axis=1)
    runner_up = np.partition(scores, -2, axis=1)[:, -2]
    return top, maximum, maximum - runner_up


def route(scores, bank_ids, policy):
    top, maximum, margin = top_scores(scores)
    threshold = policy["threshold"]
    if threshold is None:
        return [NO_MEMORY] * len(scores)
    return [bank_ids[int(j)] if s > threshold and gap >= policy["margin"] else NO_MEMORY
            for j, s, gap in zip(top, maximum, margin)]


def validate_rows(rows, bank_ids, split):
    matrix(rows, bank_ids, split)
    for r in rows:
        lengths = r["token_counts"]
        if set(lengths) != {NO_MEMORY, *bank_ids} or any(type(n) is not int or not 0 < n <= 1024 for n in lengths.values()):
            raise ValueError("Incomplete or invalid generated-token counts")


def rank(rows, decisions):
    actual = [r["rewards"][d] for r, d in zip(rows, decisions)]
    return (-sum(actual), sum(a < r["rewards"][NO_MEMORY] for a, r in zip(actual, rows)),
            sum(r["token_counts"][d] for r, d in zip(rows, decisions)), sum(d != NO_MEMORY for d in decisions))


def metrics(rows, decisions):
    result = paired_metrics(rows, decisions)
    counts = np.asarray([r["token_counts"][d] for r, d in zip(rows, decisions)])
    base = sum(r["token_counts"][NO_MEMORY] for r in rows)
    capped = counts == 1024
    result["generated_tokens"] = {
        "total": int(counts.sum()), "mean": float(counts.mean()), "median": float(np.median(counts)),
        "p90": float(np.quantile(counts, .9)), "min": int(counts.min()), "max": int(counts.max()),
        "delta_total": int(counts.sum()) - base, "relative_delta": float(counts.sum() / base - 1),
        "at_1024_token_limit_count": int(capped.sum()),
        "correct_at_limit_count": sum(int(r["rewards"][d]) for r, d, cap in zip(rows, decisions, capped) if cap),
    }
    return result


def fit_rule(rows, scores, bank_ids, rule):
    """Thresholds/quantiles see only this fit partition, including inside CV."""
    _, maximum, gap = top_scores(scores)
    thresholds = [-1.000001, *sorted(set(map(float, np.quantile(maximum, POLICY["quantiles"]))))]
    margins = [0.]
    if rule == "threshold_margin":
        margins = sorted({0., *map(float, np.quantile(gap, POLICY["quantiles"]))})
    elif rule != "threshold":
        raise ValueError("Unknown abstention rule")
    candidates = [{"threshold": None, "margin": 0.}]
    candidates.extend({"threshold": t, "margin": m} for t in thresholds for m in margins)
    selected = min(enumerate(candidates), key=lambda item: (*rank(rows, route(scores, bank_ids, item[1])), item[0]))[1]
    return {**selected, "rule": rule, "candidate_count": len(candidates)}


def fit_study(train_rows, bank_ids, vectors, profile_sha256):
    """No tune/eval/final-test argument: choose family by train CV, refit on all train."""
    bank_ids = sorted(bank_ids)
    validate_rows(train_rows, bank_ids, "train")
    if set(vectors) != set(KEYS) or len(train_rows) < POLICY["fold_count"]:
        raise ValueError("Need all key variants and at least five training samples")
    order = sorted(range(len(train_rows)), key=lambda i: canonical_hash({
        "seed": POLICY["fold_seed"], "sample_id": train_rows[i]["sample_id"]}))
    folds = {i: n % POLICY["fold_count"] for n, i in enumerate(order)}
    features = [r["feature"] for r in train_rows]
    candidates, oof_predictions = {}, {}
    for key in KEYS:
        scores = score_features(features, bank_ids, vectors[key])
        for rule in RULES:
            name = key + "/" + rule
            held_out = [None] * len(train_rows)
            fold_policies = []
            for fold in range(POLICY["fold_count"]):
                fit_ids = [i for i in range(len(train_rows)) if folds[i] != fold]
                val_ids = [i for i in range(len(train_rows)) if folds[i] == fold]
                fitted = fit_rule([train_rows[i] for i in fit_ids], scores[fit_ids], bank_ids, rule)
                decisions = route(scores[val_ids], bank_ids, fitted)
                for i, decision in zip(val_ids, decisions):
                    held_out[i] = decision
                fold_policies.append({"fold": fold, **fitted})
            fitted = fit_rule(train_rows, scores, bank_ids, rule)
            candidates[name] = {"key": key, **fitted, "train": metrics(train_rows, route(scores, bank_ids, fitted)),
                                "train_oof": metrics(train_rows, held_out), "fold_policies": fold_policies}
            oof_predictions[name] = held_out
    names = list(candidates)
    recommended = min(enumerate(names), key=lambda item: (*rank(train_rows, oof_predictions[item[1]]), item[0]))[1]
    return seal({"schema_version": POLICY["version"], "policy": POLICY, "profile_sha256": profile_sha256,
                 "bank_ids": bank_ids, "key_features": vectors, "candidates": candidates,
                 "recommended": recommended,
                 "train_data_sha256": canonical_hash(train_rows),
                 "train_sample_ids": [r["sample_id"] for r in train_rows],
                 "fold_assignments": {train_rows[i]["sample_id"]: folds[i] for i in range(len(train_rows))},
                 "oof_predictions": {name: dict(zip([r["sample_id"] for r in train_rows], ds)) for name, ds in oof_predictions.items()},
                 "parameter_selection_data": "train_only", "uses_gold_at_inference": False}, "selector_sha256")


def predict(study, feature, candidate=None):
    """Deployable question-feature-only decision; memory value addressed by bank_id."""
    authenticate(study, "selector_sha256", "similarity study selector")
    if study["policy"] != POLICY:
        raise ValueError("Unknown similarity study policy")
    name = candidate or study["recommended"]
    fitted = study["candidates"][name]
    scores = score_features([feature], study["bank_ids"], study["key_features"][fitted["key"]])
    top, maximum, gap = top_scores(scores)
    return {"selected_bank": route(scores, study["bank_ids"], fitted)[0], "candidate": name,
            "top1_bank": study["bank_ids"][int(top[0])], "similarity": float(maximum[0]), "margin": float(gap[0])}


def evaluate(study, rows, split, legacy_threshold, legacy_fixed):
    authenticate(study, "selector_sha256", "similarity study selector")
    ids = study["bank_ids"]
    validate_rows(rows, ids, split)
    if split not in ("train", "tune"):
        raise ValueError("Study evaluates train/tune only")
    if split == "tune" and set(study["train_sample_ids"]) & {r["sample_id"] for r in rows}:
        raise ValueError("Train/tune sample overlap")
    scores = {k: score_features([r["feature"] for r in rows], ids, v) for k, v in study["key_features"].items()}
    decisions = {"no_memory": [NO_MEMORY] * len(rows), "fixed_from_train": [legacy_fixed] * len(rows),
                 "legacy_semantic": route(scores["full_card"], ids, {"threshold": legacy_threshold, "margin": 0.})}
    for key in KEYS:
        decisions[key + "/always"] = route(scores[key], ids, {"threshold": -1.000001, "margin": 0.})
    for name, fitted in study["candidates"].items():
        decisions[name] = route(scores[fitted["key"]], ids, fitted)
    decisions["recommended"] = decisions[study["recommended"]]
    return {"methods": {name: metrics(rows, ds) for name, ds in decisions.items()},
            "predictions": {name: dict(zip([r["sample_id"] for r in rows], ds)) for name, ds in decisions.items()}}


def _buckets(values, outcomes, edges):
    bucket_ids = np.searchsorted(edges, values, side="right")
    result = []
    for i in range(len(edges) + 1):
        selected = [o for n, o in zip(bucket_ids, outcomes) if n == i]
        gain = sum(o[0] > 0 for o in selected)
        harm = sum(o[0] < 0 for o in selected)
        result.append({"lower_inclusive": None if i == 0 else edges[i-1],
                       "upper_exclusive": None if i == len(edges) else edges[i], "count": len(selected),
                       "gain": gain, "harm": harm, "net_gain": gain-harm,
                       "net_gain_per_case": (gain-harm)/len(selected) if selected else None,
                       "mean_token_delta": sum(o[1] for o in selected)/len(selected) if selected else None})
    return result


def diagnostics(study, train_rows, rows):
    """Descriptive association only; all bin edges come from train question scores."""
    ids = study["bank_ids"]
    result = {}
    for key in KEYS:
        fit_scores = score_features([r["feature"] for r in train_rows], ids, study["key_features"][key])
        scores = score_features([r["feature"] for r in rows], ids, study["key_features"][key])
        _, fit_max, fit_gap = top_scores(fit_scores)
        top, maximum, gap = top_scores(scores)
        outcomes = [(r["rewards"][ids[int(j)]] - r["rewards"][NO_MEMORY],
                     r["token_counts"][ids[int(j)]] - r["token_counts"][NO_MEMORY]) for r, j in zip(rows, top)]
        pair_outcomes = [(r["rewards"][b] - r["rewards"][NO_MEMORY], r["token_counts"][b] - r["token_counts"][NO_MEMORY])
                         for r in rows for b in ids]
        bins = lambda values: sorted(set(map(float, np.quantile(values, [.25, .5, .75]))))
        result[key] = {
            "top1_similarity": _buckets(maximum, outcomes, bins(fit_max)),
            "top1_margin": _buckets(gap, outcomes, bins(fit_gap)),
            "all_question_bank_pairs_similarity": _buckets(scores.ravel(), pair_outcomes, bins(fit_scores.ravel())),
            "top1_selection_counts": dict(sorted(Counter(ids[int(j)] for j in top).items())),
            "by_bank": {},
        }
        for j, bid in enumerate(ids):
            selected_rows = [r for r, chosen in zip(rows, top) if chosen == j]
            result[key]["by_bank"][bid] = {
                "fixed_on_all_questions": metrics(rows, [bid] * len(rows)),
                "when_top1": metrics(selected_rows, [bid] * len(selected_rows)) if selected_rows else None,
            }
    return result
