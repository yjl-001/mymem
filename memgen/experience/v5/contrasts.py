"""Build bounded, same-input better/worse comparisons without treating truncation as failure."""
from __future__ import annotations

from collections import Counter, defaultdict

from memgen.experience.bank_construction.artifacts import digest

FAILURE_ORDER = ("answer_error", "process_error", "format_error")


def choose_contrasts(reviews, store, limit):
    episodes = {review["episode_id"]: store.require(review["episode_key"]) for review in reviews}
    successes = [review for review in reviews if review["status"] == "success"]
    failures = [review for review in reviews if review["status"] == "failure"]
    if not successes or not failures:
        return []
    successes.sort(key=lambda review: (episodes[review["episode_id"]]["index"] != 0,
                                       episodes[review["episode_id"]]["generation"]["token_count"],
                                       review["episode_id"]))
    success = successes[0]
    selected, used = [], set()
    for failure_type in FAILURE_ORDER:
        candidates = [review for review in failures if failure_type in review["failure_types"]
                      and review["episode_id"] not in used]
        candidates.sort(key=lambda review: (episodes[review["episode_id"]]["index"] != 0,
                                             episodes[review["episode_id"]]["generation"]["token_count"],
                                             review["episode_id"]))
        if candidates:
            selected.append((success, candidates[0], failure_type))
            used.add(candidates[0]["episode_id"])
        if len(selected) == limit:
            break
    if not selected:
        failure = sorted(failures, key=lambda review: (episodes[review["episode_id"]]["index"] != 0,
                                                        review["episode_id"]))[0]
        selected.append((success, failure, failure["failure_types"][0]))
    return selected


def run_contrasts(store, config):
    reviews = store.require("stages/review")
    grouped = defaultdict(list)
    for key in reviews["keys"]:
        record = store.require(key)
        grouped[record["input_id"]].append(record)
    keys, deferred, counts = [], [], Counter()
    for index, input_id in enumerate(sorted(grouped), start=1):
        pairs = choose_contrasts(grouped[input_id], store, config.contrast_limit_per_input)
        if not pairs:
            states = Counter(item["status"] for item in grouped[input_id])
            deferred.append({"input_id": input_id, "reason": "no_verified_success_failure_contrast",
                             "review_status_counts": dict(states)})
            counts["deferred_inputs"] += 1
        for success, failure, primary_failure_type in pairs:
            good = store.require(success["episode_key"])
            bad = store.require(failure["episode_key"])
            contrast_id = "contrast-" + digest([input_id, good["episode_id"], bad["episode_id"], primary_failure_type])
            key = "contrasts/" + contrast_id
            inputs = {"success": success, "failure": failure, "primary_failure_type": primary_failure_type}
            record = {"contrast_id": contrast_id, "input_id": input_id,
                "success_episode_id": good["episode_id"], "failure_episode_id": bad["episode_id"],
                "success_episode_key": success["episode_key"], "failure_episode_key": failure["episode_key"],
                "success_review": success["semantic_review"],
                "failure_review": failure["semantic_review"],
                "failure_types": failure["failure_types"], "primary_failure_type": primary_failure_type,
                "priority": "greedy_failure_sampled_success" if bad["index"] == 0 and good["index"] > 0
                            else "sampled_contrast" if bad["index"] > 0 else "other_contrast"}
            store.put(key, record, inputs)
            keys.append(key)
            counts[primary_failure_type] += 1
        print(f"[v5] contrasts input={index}/{len(grouped)} total={len(keys)}", flush=True)
    return store.put("stages/contrasts", {"keys": keys, "contrast_count": len(keys),
        "deferred": deferred, "counts": dict(counts)}, {"review": digest(reviews),
        "limit_per_input": config.contrast_limit_per_input})
