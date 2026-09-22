"""Extract universal Experience Atoms from authenticated same-input contrasts."""
from __future__ import annotations

from collections import Counter

from memgen.experience.bank_construction.artifacts import digest
from memgen.experience.bank_construction.parallel import ordered_map, teacher_workers
from .schemas import validate_atom


def run_atoms(store, teacher):
    split = store.require("split")
    inputs_by_id = {row["input_id"]: row for row in split["splits"]["train"]}
    contrasts = store.require("stages/contrasts")

    def process(contrast_key):
        contrast = store.require(contrast_key)
        row = inputs_by_id[contrast["input_id"]]
        success = store.require(contrast["success_episode_key"])
        failure = store.require(contrast["failure_episode_key"])
        payload = {"input": row["input"],
            "better": {"output": success["output"], "outcome": success["outcome"],
                       "process_review": contrast["success_review"]},
            "worse": {"output": failure["output"], "outcome": failure["outcome"],
                      "process_review": contrast["failure_review"],
                      "failure_types": contrast["failure_types"]},
            "contrast_priority": contrast["priority"]}
        atom_id = "atom-" + digest([contrast["contrast_id"], payload])
        key = "atoms/" + atom_id
        record = store.get(key, payload)
        if record is None:
            answer = teacher.ask("atom", payload, validate_atom)
            record = store.put(key, {"atom_id": atom_id, "input_id": contrast["input_id"],
                "contrast_id": contrast["contrast_id"], "success_episode_id": contrast["success_episode_id"],
                "failure_episode_id": contrast["failure_episode_id"],
                "failure_types": contrast["failure_types"], "teacher_assessment": {
                    "transferable": answer["transferable"], "confidence": answer["confidence"],
                    "reason": answer["reason"]},
                **({key: answer[key] for key in ("applicability", "experience",
                    "exclusions_from_input", "failure_mechanism")} if answer["transferable"] else {})}, payload)
        return key, record

    completed, counts = [], Counter()
    for index, item in enumerate(ordered_map(process, contrasts["keys"], teacher_workers(teacher)), start=1):
        completed.append(item)
        counts["accepted" if item[1]["teacher_assessment"]["transferable"] else "not_transferable"] += 1
        print(f"[v5] atoms={index}/{len(contrasts['keys'])} accepted={counts['accepted']}", flush=True)
    accepted = [key for key, record in completed if record["teacher_assessment"]["transferable"]]
    deferred = [{"atom_key": key, "reason": record["teacher_assessment"]["reason"]}
                for key, record in completed if not record["teacher_assessment"]["transferable"]]
    return store.put("stages/atoms", {"keys": accepted, "deferred": deferred,
        "counts": dict(counts)}, {"contrasts": digest(contrasts)})
