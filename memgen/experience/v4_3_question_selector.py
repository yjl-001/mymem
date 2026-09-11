"""Question-only routing learned from native-prefix-KV counterfactual utility.

Training, threshold tuning and evaluation have separate sample namespaces.
No source-Bank membership is used as a selection target.
"""
from collections import Counter
import math
import numpy as np

from memgen.experience.phase1 import SPLIT_MANIFEST_SCHEMA, canonical_json_sha256
from memgen.experience.v4_3_bank import authenticate, canonical_hash, seal, text_hash

NO_MEMORY = "no_memory"
SPLITS = ("train", "tune", "eval")
RIDGES = (0.01, 0.1, 1., 10., 100.)
THRESHOLDS = (None, -0.25, 0., 0.05, 0.1, 0.2, 0.5)
POLICY = {"version": "v43-question-utility-ridge-v1", "features": "frozen_reasoner_final_norm_mean_raw_question_l2",
          "target": "bank_strict_reward_minus_no_memory_strict_reward", "ridge_grid": list(RIDGES),
          "gain_threshold_grid": list(THRESHOLDS), "threshold_none_means": "always_abstain",
          "tie_break": "tune_correct_desc_harm_asc_usage_asc_grid_order",
          "refit_on_tune": False, "question_only": True, "gate_enabled": False}


def partition_samples(manifest, packets, *, counts=(200, 100, 100), seed=43):
    logical = {k: v for k, v in manifest.items() if k not in {"created_at", "manifest_sha256"}}
    if manifest.get("schema_version") != SPLIT_MANIFEST_SCHEMA or canonical_json_sha256(logical) != manifest.get("manifest_sha256"):
        raise ValueError("Invalid frozen GSM8K split manifest")
    if manifest["dataset"]["name"] != "openai/gsm8k" or manifest["dataset"]["configuration"] != "main":
        raise ValueError("Unexpected selector dataset")
    if len(counts) != 3 or any(type(c) is not int or c <= 0 for c in counts):
        raise ValueError("Positive train/tune/eval counts required")
    entries = manifest["samples"]
    if len({r["sample_id"] for r in entries}) != len(entries):
        raise ValueError("Duplicate split sample identity")
    if Counter(r["logical_split"] for r in entries) != Counter({k: v for k, v in manifest["counts"].items() if v}):
        raise ValueError("Split counts mismatch")
    by_id = {r["sample_id"]: r for r in entries}
    hash_splits = {}
    for r in entries:
        h = r["question_sha256"]
        if h in hash_splits and hash_splits[h] != r["logical_split"]:
            raise ValueError("Question content crosses logical splits")
        hash_splits[h] = r["logical_split"]
    construction_hashes = set()
    for p in packets:
        for e in p["evidence"]:
            entry = by_id.get(e["sample_id"])
            if entry is None or entry["logical_split"] != "bank-source" or entry["question_sha256"] != text_hash(e["question"].strip()):
                raise ValueError("Construction sample differs from frozen split membership")
            construction_hashes.add(entry["question_sha256"])
    pool = [r for r in entries if r["logical_split"] == "calibration-val"]
    if any(r["dataset_split"] != "train" or r["question_sha256"] in construction_hashes for r in pool):
        raise ValueError("Selector pool overlaps construction or official test")
    if len({r["question_sha256"] for r in pool}) != len(pool):
        raise ValueError("Duplicate selector question content")
    if sum(counts) > len(pool):
        raise ValueError(f"Requested {sum(counts)} samples but calibration-val has {len(pool)}")
    ordered = sorted(pool, key=lambda r: canonical_hash({"seed": seed, "sample_id": r["sample_id"]}))
    selected, offset = [], 0
    for name, count in zip(SPLITS, counts):
        selected.extend({**r, "selector_split": name} for r in ordered[offset:offset+count])
        offset += count
    return selected


def checked_dataset_rows(dataset, entries):
    result = {}
    for e in entries:
        row = dataset[e["source_index"]]
        q, a = row["question"].strip(), row["answer"].strip()
        if text_hash(q) != e["question_sha256"] or text_hash(a) != e["answer_sha256"]:
            raise ValueError(f"Dataset question/answer revision drift: {e['sample_id']}")
        result[e["sample_id"]] = {"question": q, "answer": a}
    return result


def matrix(rows, bank_ids, expected_split):
    if not rows or any(r["selector_split"] != expected_split for r in rows):
        raise ValueError("Selector fit/tune/eval split isolation violated")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate selector sample")
    for r in rows:
        if set(r["rewards"]) != {NO_MEMORY, *bank_ids} or any(type(x) not in (int, float) or x not in (0, 1) for x in r["rewards"].values()):
            raise ValueError("Incomplete or nonbinary counterfactual rewards")
    x = np.asarray([r["feature"] for r in rows], dtype=np.float64)
    rewards = np.asarray([[r["rewards"][b] for b in [NO_MEMORY, *bank_ids]] for r in rows], dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5):
        raise ValueError("Question features must be finite unit vectors")
    return x, rewards


def ridge_fit(x, y, regularization):
    mean_x, mean_y = x.mean(0), y.mean(0)
    xc = x - mean_x
    dual = np.linalg.solve(xc @ xc.T + regularization*np.eye(len(x)), y-mean_y)
    return {"mean_x": mean_x.tolist(), "mean_y": mean_y.tolist(), "weights": (xc.T @ dual).tolist()}


def ridge_scores(model, features):
    return (np.asarray(features)-np.asarray(model["mean_x"])) @ np.asarray(model["weights"]) + np.asarray(model["mean_y"])


def choices(scores, bank_ids, threshold):
    s = np.asarray(scores)
    if s.ndim != 2 or s.shape[1] != len(bank_ids) or not np.isfinite(s).all():
        raise ValueError("Invalid selector scores")
    return [NO_MEMORY if threshold is None or float(row.max()) <= threshold else bank_ids[int(row.argmax())] for row in s]


def paired_metrics(rows, decisions):
    if len(rows) != len(decisions) or not rows:
        raise ValueError("Decision coverage mismatch")
    base = np.asarray([r["rewards"][NO_MEMORY] for r in rows])
    actual = np.asarray([r["rewards"][d] for r, d in zip(rows, decisions)])
    gain, harm = int((actual > base).sum()), int((actual < base).sum())
    discordant = gain+harm
    p = min(1., 2*sum(math.comb(discordant, i) for i in range(min(gain, harm)+1))/2**discordant) if discordant else 1.
    return {"count": len(rows), "correct": int(actual.sum()), "accuracy": float(actual.mean()),
            "gain": gain, "harm": harm, "net_gain": gain-harm, "accuracy_delta": float((actual-base).mean()),
            "memory_use_count": sum(d != NO_MEMORY for d in decisions), "paired_exact_p": p,
            "selection_counts": dict(sorted(Counter(decisions).items()))}


def rank_decisions(rows, decisions):
    m = paired_metrics(rows, decisions)
    return (-m["correct"], m["harm"], m["memory_use_count"])


def fit_selector(train_rows, tune_rows, bank_ids, card_features, profile_sha256):
    if set(r["sample_id"] for r in train_rows) & set(r["sample_id"] for r in tune_rows):
        raise ValueError("Train/tune overlap")
    bank_ids = sorted(bank_ids)
    if not bank_ids or len(set(bank_ids)) != len(bank_ids) or NO_MEMORY in bank_ids or set(card_features) != set(bank_ids):
        raise ValueError("Invalid selector Bank/feature namespace")
    x, rewards = matrix(train_rows, bank_ids, "train")
    tx, _ = matrix(tune_rows, bank_ids, "tune")
    cards = np.asarray([card_features[b] for b in bank_ids], dtype=np.float64)
    if cards.shape != (len(bank_ids), x.shape[1]) or not np.isfinite(cards).all() or not np.allclose(np.linalg.norm(cards, axis=1), 1, atol=1e-5):
        raise ValueError("Card/query feature geometry mismatch")
    trials, candidates = [], []
    for regularization in RIDGES:
        model = ridge_fit(x, rewards[:, 1:]-rewards[:, :1], regularization)
        scores = ridge_scores(model, tx)
        for threshold in THRESHOLDS:
            decisions = choices(scores, bank_ids, threshold)
            metrics = paired_metrics(tune_rows, decisions)
            trials.append({"ridge": regularization, "threshold": threshold, "tune": metrics})
            candidates.append((rank_decisions(tune_rows, decisions), len(candidates), regularization, threshold, model))
    _, _, reg, threshold, model = min(candidates, key=lambda t: t[:2])
    semantic_scores = tx @ cards.T
    semantic_trials = [(rank_decisions(tune_rows, choices(semantic_scores, bank_ids, t)), i, t)
                       for i, t in enumerate((None, -1., 0., .25, .5, .6, .7, .8, .9))]
    semantic_threshold = min(semantic_trials)[2]
    actions = [NO_MEMORY, *bank_ids]
    fixed = min(actions, key=lambda b: rank_decisions(train_rows, [b]*len(train_rows)))
    selected_tune = choices(ridge_scores(model, tx), bank_ids, threshold)
    return seal({"schema_version": "v43-question-selector-v1", "profile_sha256": profile_sha256,
                 "policy": POLICY, "bank_ids": bank_ids, "ridge": reg, "threshold": threshold, "model": model,
                 "card_features": {b: card_features[b] for b in bank_ids}, "semantic_threshold": semantic_threshold,
                 "fixed_bank_from_train": fixed, "train_sample_ids": [r["sample_id"] for r in train_rows],
                 "tune_sample_ids": [r["sample_id"] for r in tune_rows],
                 "train_data_sha256": canonical_hash(train_rows), "tune_data_sha256": canonical_hash(tune_rows),
                 "tuning_trials": trials, "selected_tune_metrics": paired_metrics(tune_rows, selected_tune),
                 "inference_input": "question_feature_only", "uses_gold_at_selection": False,
                 "no_memory_threshold_learned_on": "tune_only"}, "selector_sha256")


def predict(selector, feature):
    """No question ID, answer, reward, trajectory or source membership accepted."""
    authenticate(selector, "selector_sha256", "question selector")
    if selector["policy"] != POLICY:
        raise ValueError("Unknown selector policy")
    x = np.asarray([feature], dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5):
        raise ValueError("Question-only inference requires a finite unit feature")
    ids = selector["bank_ids"]
    scores = ridge_scores(selector["model"], x)
    selected = choices(scores, ids, selector["threshold"])[0]
    sim = x @ np.asarray([selector["card_features"][b] for b in ids]).T
    return {"selected_bank": selected, "predicted_net_gain": dict(zip(ids, map(float, scores[0]))),
            "semantic_bank": choices(sim, ids, selector["semantic_threshold"])[0],
            "fixed_bank": selector["fixed_bank_from_train"]}


def evaluate(selector, rows, predictions):
    ids = selector["bank_ids"]
    matrix(rows, ids, "eval")
    forbidden = set(selector["train_sample_ids"] + selector["tune_sample_ids"])
    if forbidden & {r["sample_id"] for r in rows}:
        raise ValueError("Evaluation overlaps training or tuning")
    decisions = []
    for r in rows:
        p = predictions[r["sample_id"]]
        expected = predict(selector, r["feature"])
        if p["selector_sha256"] != selector["selector_sha256"] or p["decision"] != expected:
            raise ValueError("Evaluation decision differs from frozen question-only selector")
        decisions.append(expected)
    methods = {"no_memory": [NO_MEMORY]*len(rows),
               "selector": [p["selected_bank"] for p in decisions],
               "semantic": [p["semantic_bank"] for p in decisions],
               "fixed_from_train": [p["fixed_bank"] for p in decisions]}
    report = {name: paired_metrics(rows, selected) for name, selected in methods.items()}
    oracle = sum(max(r["rewards"].values()) for r in rows)
    base = report["no_memory"]["correct"]
    return {"methods": report, "oracle_best_correct": oracle, "oracle_best_accuracy": oracle/len(rows),
            "oracle_is_deployable": False, "available_net_gain": oracle-base,
            "selector_gain_capture": (report["selector"]["correct"]-base)/(oracle-base) if oracle > base else None,
            "selector_regret_count": oracle-report["selector"]["correct"],
            "all_bank_accuracy": {b: sum(r["rewards"][b] for r in rows)/len(rows) for b in ids},
            "held_out_from_current_card_construction": True, "official_test_used": False,
            "prior_use_of_calibration_pool": "historical_calibration_pool_not_a_pristine_final_test"}
