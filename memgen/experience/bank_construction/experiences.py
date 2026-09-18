"""Reviewed same-question contrasts and grounded, transferable repair signatures."""
from __future__ import annotations

from collections import defaultdict, Counter
from .artifacts import digest
from .schemas import validate_evidence


def choose_pair(reviews, store):
    """One contrast per question; prefer greedy observations within each outcome."""
    ordered = sorted(reviews, key=lambda r: store.require(r["rollout_key"])["index"])
    successes = [r for r in ordered if r["outcome"] == "success"]
    failures = [r for r in ordered if r["outcome"] == "failure"]
    return (successes[0], failures[0]) if successes and failures else None


def run_experiences(store, teacher):
    split = store.require("split")
    samples = {r["sample_id"]: r for r in split["splits"]["train"]}
    reviews = store.require("stages/review")
    grouped = defaultdict(list)
    for key in reviews["keys"]:
        value = store.require(key)
        grouped[value["sample_id"]].append(value)
    accepted, deferred, counts = [], [], Counter()
    for i, (sid, items) in enumerate(sorted(grouped.items())):
        pair = choose_pair(items, store)
        if pair is None:
            counts["no_reviewed_contrast"] += 1
            deferred.append({"sample_id": sid, "reason": "no_reviewed_success_failure_pair"})
            continue
        success, failure = pair
        good, bad = (store.require(r["rollout_key"]) for r in pair)
        sample = samples[sid]
        payload = {"question": sample["question"], "official_solution": sample["official_solution"],
                   "success": {"trajectory": good["generation"]["text"], "review": success},
                   "failure": {"trajectory": bad["generation"]["text"], "review": failure}}
        eid = "evidence-" + digest([sid, good["rollout_id"], bad["rollout_id"]])
        key = "evidence/" + eid
        record = store.get(key, payload)
        if record is None:
            result = teacher.ask("extract", payload, validate_evidence)
            record = store.put(key, {"evidence_id": eid, "sample_id": sid,
                "source_split": "train", "success_rollout": success["rollout_key"],
                "failure_rollout": failure["rollout_key"], "failure_types": failure["failure_types"],
                "signature": result}, payload)
        if record["signature"]["transferable"]:
            accepted.append(key)
            counts["accepted"] += 1
        else:
            deferred.append({"sample_id": sid, "evidence_key": key, "reason": record["signature"]["reason"]})
            counts["not_transferable"] += 1
        print(f"[local-bank] evidence question={i + 1}/{len(grouped)} accepted={len(accepted)}", flush=True)
    return store.put("stages/evidence", {"keys": accepted, "deferred": deferred, "counts": dict(counts)},
                     {"review": digest(reviews)})
