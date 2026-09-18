"""Teacher-only semantic grouping with bounded candidate windows and full-member reconciliation.

Every existing group is presented to the teacher; there is no vector retrieval,
distance threshold, keyword rule, minimum-size merge, or fixed target bank count.
"""
from __future__ import annotations

from .artifacts import digest
from .schemas import validate_partition, validate_match, validate_merge, validate_membership, exact_ids


def chunks(values, size):
    for i in range(0, len(values), size):
        yield values[i:i + size]


def group_record(group):
    group = {**group, "members": sorted(group["members"])}
    return {**group, "group_id": "group-" + digest(group)}


def group_summary(group):
    return {k: group[k] for k in ("group_id", "method", "applies_when", "exclusions")}


def partition(teacher, records, context):
    ids = [r["evidence_id"] for r in records]
    response = teacher.ask("partition", {"context": context, "evidence": records},
                           lambda r: validate_partition(r, ids))
    return [group_record(g) for g in response["groups"]]


def run_grouping(store, teacher, config):
    index = store.require("stages/evidence")
    source = [store.require(key) for key in index["keys"]]
    records = [{"evidence_id": r["evidence_id"], "signature": r["signature"]} for r in source]
    records.sort(key=lambda r: r["evidence_id"])
    by_id = {r["evidence_id"]: r for r in records}
    inputs = {"evidence": digest(records), "config": config.to_dict()}
    cached = store.get("stages/groups", inputs)
    if cached is not None:
        return cached
    local_groups = []
    for batch in chunks(records, config.group_batch_size):
        local_groups.extend(partition(teacher, batch, "Initial within-batch method grouping"))
    pool = []
    for i, incoming in enumerate(local_groups):
        candidates = []
        # All prior groups, including singletons, are examined across deterministic windows.
        for window in chunks(pool, config.candidate_batch_size):
            match = teacher.ask("match", {"incoming": group_summary(incoming),
                "candidates": [group_summary(g) for g in window]},
                lambda r: validate_match(r, [g["group_id"] for g in window]))
            candidates.extend(match["candidate_ids"])
        if candidates:
            selected = [g for g in pool if g["group_id"] in candidates]
            for candidate in selected:
                proposal = teacher.ask("merge", {"left": group_summary(incoming), "right": group_summary(candidate)}, validate_merge)
                if not proposal["merge"]:
                    continue
                members = sorted(incoming["members"] + candidate["members"])
                coherent = True
                # Re-read ALL original member signatures in bounded batches, including old members.
                for batch in chunks([by_id[m] for m in members], config.group_batch_size):
                    check = teacher.ask("membership", {"proposed_method": proposal, "evidence": batch},
                        lambda r: validate_membership(r, [e["evidence_id"] for e in batch]))
                    if check["incompatible_ids"]:
                        coherent = False
                        break
                if coherent:
                    incoming = group_record({k: proposal[k] for k in ("method", "applies_when", "exclusions", "rationale")} |
                                            {"members": members})
                    pool = [g for g in pool if g["group_id"] != candidate["group_id"]]
            pool.append(incoming)
        else:
            pool.append(incoming)
        pool.sort(key=lambda g: g["group_id"])
        print(f"[local-bank] grouping={i + 1}/{len(local_groups)} groups={len(pool)}", flush=True)
    exact_ids([m for g in pool for m in g["members"]], list(by_id))
    samples = {r["evidence_id"]: r["sample_id"] for r in source}
    return store.put("stages/groups", {"groups": [dict(g, distinct_sample_count=len({samples[m] for m in g["members"]}))
                     for g in pool], "evidence_count": len(records), "semantic_decisions": "local_teacher",
                     "minimum_group_size_enforced": False}, inputs)
