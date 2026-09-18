"""Separate task reward, process validity, and construction outcome."""
from __future__ import annotations

from collections import Counter
from .artifacts import digest
from .schemas import validate_review
from .parallel import ordered_map, teacher_workers


def classify(generation, verifier, review):
    if generation["truncated"]:
        return {"outcome": "truncated", "failure_types": [], "reasoning_verdict": "not_reviewed"}
    if review is None:
        raise ValueError("Completed trajectories require process review")
    validate_review(review)
    failures = []
    if not verifier["format_valid"]:
        failures.append("format_error")
    # Never use the last arbitrary number as an unboxed final-answer oracle.
    if verifier["format_valid"]:
        answer_verdict = "correct" if verifier["reward"] == 1. else "incorrect"
    else:
        answer_verdict = review["final_answer_verdict"]
    if answer_verdict == "incorrect":
        failures.append("answer_error")
    if review["reasoning_verdict"] == "incorrect":
        failures.append("reasoning_error")
    contradiction = verifier["format_valid"] and review["final_answer_verdict"] not in {answer_verdict, "uncertain"}
    if contradiction:
        outcome = "uncertain"
    elif failures:
        outcome = "failure"
    elif verifier["reward"] == 1. and review["reasoning_verdict"] == "correct":
        outcome = "success"
    else:
        outcome = "uncertain"
    return {"outcome": outcome, "failure_types": failures, "reasoning_verdict": review["reasoning_verdict"],
            "answer_verdict": answer_verdict, "verifier_teacher_conflict": contradiction}


def run_review(store, teacher):
    split = store.require("split")
    samples = {r["sample_id"]: r for r in split["splits"]["train"]}
    index = store.require("stages/rollouts")
    keys, counts = [], Counter()
    def process(rollout_key):
        rollout = store.require(rollout_key)
        row = samples[rollout["sample_id"]]
        inputs = {"rollout": rollout, "sample": row}
        key = "reviews/" + rollout["rollout_id"]
        result = store.get(key, inputs)
        if result is None:
            audit = None
            if not rollout["generation"]["truncated"]:
                audit = teacher.ask("review", {"question": row["question"],
                    "official_solution": row["official_solution"], "trajectory": rollout["generation"]["text"],
                    "answer_verifier": rollout["verifier"]}, validate_review)
            result = store.put(key, {"rollout_key": rollout_key, "sample_id": row["sample_id"],
                **classify(rollout["generation"], rollout["verifier"], audit), "teacher_review": audit}, inputs)
        return key, result

    for i, (key, result) in enumerate(ordered_map(process, index["keys"], teacher_workers(teacher))):
        keys.append(key)
        counts[result["outcome"]] += 1
        print(f"[local-bank] review={i + 1}/{len(index['keys'])} outcome={result['outcome']}", flush=True)
    return store.put("stages/review", {"keys": keys, "outcomes": dict(counts)}, {"rollouts": digest(index)})
