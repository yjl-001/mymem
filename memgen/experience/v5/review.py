"""Separate deterministic task outcome, process validity, and truncation."""
from __future__ import annotations

from collections import Counter

from memgen.experience.bank_construction.artifacts import digest
from memgen.experience.bank_construction.parallel import ordered_map, teacher_workers
from .schemas import validate_review


def classify(episode, review):
    outcome = episode["outcome"]
    if outcome["truncated"]:
        return {"status": "truncated", "failure_types": [], "process_verdict": "not_reviewed"}
    if review is None:
        raise ValueError("Nontruncated Episode requires process review")
    validate_review(review)
    failures = []
    if not outcome["format_correct"]:
        failures.append("format_error")
    deterministic_answer = "correct" if outcome["answer_correct"] else "incorrect"
    answer = deterministic_answer if outcome["format_correct"] else review["answer_verdict"]
    if answer == "incorrect":
        failures.append("answer_error")
    if review["process_verdict"] == "incorrect":
        failures.append("process_error")
    conflict = outcome["format_correct"] and review["answer_verdict"] not in {deterministic_answer, "uncertain"}
    if conflict or review["process_verdict"] == "uncertain" or answer == "uncertain":
        status = "uncertain"
    elif failures:
        status = "failure"
    elif outcome["answer_correct"] and outcome["format_correct"] and review["process_verdict"] == "correct":
        status = "success"
    else:
        status = "uncertain"
    return {"status": status, "failure_types": failures, "process_verdict": review["process_verdict"],
            "answer_verdict": answer, "verifier_teacher_conflict": conflict}


def run_review(store, teacher):
    split = store.require("split")
    rows = {row["input_id"]: row for row in split["splits"]["train"]}
    episodes = store.require("stages/episodes")
    counts = Counter()

    def process(episode_key):
        episode = store.require(episode_key)
        row = rows[episode["input_id"]]
        inputs = {"episode": episode, "input": row}
        key = "reviews/" + episode["episode_id"]
        record = store.get(key, inputs)
        if record is None:
            semantic = None
            if not episode["outcome"]["truncated"]:
                semantic = teacher.ask("review", {"input": row["input"], "reference": row["reference"],
                    "output": episode["output"], "verifier_outcome": episode["outcome"]}, validate_review)
            record = store.put(key, {"episode_key": episode_key, "episode_id": episode["episode_id"],
                "input_id": episode["input_id"], "semantic_review": semantic,
                **classify(episode, semantic)}, inputs)
        return key, record

    completed = []
    for index, result in enumerate(ordered_map(process, episodes["keys"], teacher_workers(teacher)), start=1):
        completed.append(result)
        counts[result[1]["status"]] += 1
        print(f"[v5] review={index}/{len(episodes['keys'])} status={result[1]['status']}", flush=True)
    return store.put("stages/review", {"keys": [key for key, _ in completed],
        "status_counts": dict(counts)}, {"episodes": digest(episodes)})
