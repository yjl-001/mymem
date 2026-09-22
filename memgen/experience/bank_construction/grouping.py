"""Alias-safe semantic grouping with bounded retrieval and full-member reconciliation."""
from __future__ import annotations

import math

from .artifacts import digest
from .identifiers import alias_records, identifier_protocol
from .parallel import ordered_map, teacher_workers
from .schemas import exact_ids, validate_match, validate_membership, validate_merge, validate_partition

STRATEGY = "alias-embedding-candidates-qwen-v1"


def chunks(values, size):
    for index in range(0, len(values), size):
        yield values[index:index + size]


def group_record(group):
    value = {**group, "members": sorted(group["members"])}
    return {**value, "group_id": "group-" + digest(value)}


def group_summary(group):
    return {key: group[key] for key in ("group_id", "method", "applies_when", "exclusions")}


def save_fallback(store, task, payload, error, fallback):
    inputs = {"strategy": STRATEGY, "task": task, "payload": payload}
    return store.put("grouping/fallback/" + task + "/" + digest(inputs),
                     {"error": str(error), "fallback": fallback}, inputs)


def initial_partition(store, teacher, records, batch_index):
    inputs = {"strategy": STRATEGY, "batch_index": batch_index, "evidence": records}
    key = f"grouping/initial/{batch_index:06d}"
    cached = store.get(key, inputs)
    if cached is not None:
        return cached
    short, aliases, mapping = alias_records(records)
    payload = {"context": "Initial within-batch method grouping with short aliases",
               "evidence": short,
               "identifier_protocol": identifier_protocol(aliases, response_field="groups[].members")}
    fallback = None
    try:
        answer = teacher.ask("partition", payload, lambda value: validate_partition(value, aliases))
        mapped = {"groups": []}
        for value in answer["groups"]:
            group = dict(value)
            group["members"] = [mapping[item] for item in value["members"]]
            mapped["groups"].append(group)
    except RuntimeError as exc:
        fallback = str(exc)
        mapped = {"groups": [{"members": [mapping[alias]],
            "method": record["signature"]["repair_operator"],
            "applies_when": record["signature"]["problem_structure"],
            "exclusions": record["signature"].get("exclusion_conditions", []),
            "rationale": "Conservative singleton after bounded partition retries."}
            for alias, record in zip(aliases, records)]}
        save_fallback(store, "partition", payload, exc, mapped)
    validate_partition(mapped, [record["evidence_id"] for record in records])
    result = {"groups": [group_record(group) for group in mapped["groups"]],
              "fallback": fallback, "evidence_count": len(records)}
    return store.put(key, result, inputs)


def group_text(group):
    return "\n".join(("Method: " + group["method"], "Applies when: " + group["applies_when"],
                      "Exclusions: " + "; ".join(group["exclusions"])))


def group_vector(store, group, encoder, encoder_identity):
    inputs = {"strategy": STRATEGY, "group": group_summary(group), "encoder": encoder_identity}
    key = "grouping/vectors/" + group["group_id"]
    cached = store.get(key, inputs)
    if cached is None:
        cached = store.put(key, {"vector": encoder(group_text(group))}, inputs)
    vector = cached["vector"]
    if not isinstance(vector, list) or not vector or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in vector):
        raise ValueError("Invalid grouping candidate vector")
    return vector


def nearest_groups(pool, query, get_vector, count):
    vectors = [get_vector(group) for group in pool]
    try:
        import numpy as np
        matrix = np.asarray(vectors, dtype=np.float32)
        vector = np.asarray(query, dtype=np.float32)
        if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.shape[0]:
            raise ValueError("Grouping vector dimensions differ")
        scores = (matrix @ vector).tolist()
    except ImportError:
        if any(len(value) != len(query) for value in vectors):
            raise ValueError("Grouping vector dimensions differ")
        scores = [sum(left * right for left, right in zip(query, value)) for value in vectors]
    ranked = sorted(zip(scores, (group["group_id"] for group in pool), pool),
                    key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked[:count]]


def match_candidates(store, teacher, incoming, ranked, window_size):
    windows = list(enumerate(chunks(ranked, window_size)))

    def process(item):
        window_index, window = item
        aliases = [f"C{index:02d}" for index in range(len(window))]
        mapping = dict(zip(aliases, (group["group_id"] for group in window)))
        payload = {"incoming": {**group_summary(incoming), "group_id": "INCOMING"},
                   "candidates": [{**group_summary(group), "group_id": alias}
                                  for alias, group in zip(aliases, window)]}
        try:
            answer = teacher.ask("match", payload, lambda value: validate_match(value, aliases))
            selected = [mapping[value] for value in answer["candidate_ids"]]
        except RuntimeError as exc:
            selected = []
            save_fallback(store, "match", payload, exc, {"candidate_ids": []})
        return window_index, selected

    results = list(ordered_map(process, windows, teacher_workers(teacher)))
    return [group_id for _, selected in sorted(results) for group_id in selected]


def merge_candidate(store, teacher, incoming, candidate, by_id, batch_size):
    payload = {"left": {**group_summary(incoming), "group_id": "LEFT"},
               "right": {**group_summary(candidate), "group_id": "RIGHT"}}
    try:
        proposal = teacher.ask("merge", payload, validate_merge)
    except RuntimeError as exc:
        save_fallback(store, "merge", payload, exc, {"merge": False})
        return None
    if not proposal["merge"]:
        return None
    members = sorted(incoming["members"] + candidate["members"])
    if len(members) != len(set(members)):
        raise ValueError("Candidate groups overlap before merge")
    for batch in chunks([by_id[value] for value in members], batch_size):
        short, aliases, mapping = alias_records(batch)
        check_payload = {"proposed_method": proposal, "evidence": short,
                         "identifier_protocol": {"response_field": "incompatible_ids",
                                                 "allowed_ids": aliases,
                                                 "allow_subset_without_duplicates": True}}
        try:
            answer = teacher.ask("membership", check_payload,
                                 lambda value: validate_membership(value, aliases))
            incompatible = [mapping[value] for value in answer["incompatible_ids"]]
        except RuntimeError as exc:
            incompatible = [value["evidence_id"] for value in batch]
            save_fallback(store, "membership", check_payload, exc,
                          {"incompatible_ids": incompatible, "mode": "reject_merge"})
        if incompatible:
            return None
    return group_record({key: proposal[key] for key in ("method", "applies_when", "exclusions", "rationale")} |
                        {"members": members})


def consolidate_round(store, teacher, groups, by_id, config, encoder, encoder_identity,
                      round_index, vector_cache):
    pool, merge_count = [], 0
    for position, original in enumerate(sorted(groups, key=lambda value: value["group_id"])):
        inputs = {"strategy": STRATEGY, "round": round_index, "position": position,
                  "incoming": original, "pool_before_sha256": digest(pool),
                  "candidate_top_k": config.group_candidate_top_k}
        key = f"grouping/steps/round-{round_index:02d}/{position:06d}"
        cached = store.get(key, inputs)
        if cached is not None:
            removed = set(cached["removed_group_ids"])
            if not removed <= {group["group_id"] for group in pool}:
                raise ValueError("Cached grouping step removes an unavailable group")
            pool = [group for group in pool if group["group_id"] not in removed]
            pool.append(cached["incoming"])
            pool.sort(key=lambda value: value["group_id"])
            merge_count += cached["merge_count"]
            continue
        incoming, removed = original, []
        if pool:
            def vector_for(group):
                if group["group_id"] not in vector_cache:
                    vector_cache[group["group_id"]] = group_vector(
                        store, group, encoder, encoder_identity)
                return vector_cache[group["group_id"]]
            ranked = nearest_groups(pool, vector_for(incoming), vector_for,
                                    config.group_candidate_top_k)
            candidate_ids = match_candidates(store, teacher, incoming, ranked,
                                             config.candidate_batch_size)
            for candidate_id in candidate_ids:
                candidate = next((value for value in pool if value["group_id"] == candidate_id), None)
                if candidate is None:
                    continue
                merged = merge_candidate(store, teacher, incoming, candidate, by_id,
                                         config.group_batch_size)
                if merged is not None:
                    incoming = merged
                    removed.append(candidate_id)
                    pool = [value for value in pool if value["group_id"] != candidate_id]
                    merge_count += 1
        pool.append(incoming)
        pool.sort(key=lambda value: value["group_id"])
        store.put(key, {"incoming": incoming, "removed_group_ids": removed,
                       "merge_count": len(removed), "pool_after_sha256": digest(pool)}, inputs)
        print(f"[local-bank] grouping round={round_index + 1} item={position + 1}/{len(groups)} "
              f"groups={len(pool)} merges={merge_count}", flush=True)
    return pool, merge_count


def run_grouping(store, teacher, config, encoder, encoder_identity):
    index = store.require("stages/evidence")
    source = [store.require(key) for key in index["keys"]]
    records = [{"evidence_id": value["evidence_id"], "signature": value["signature"]}
               for value in source]
    records.sort(key=lambda value: value["evidence_id"])
    inputs = {"evidence": digest(records), "config": config.to_dict(), "strategy": STRATEGY,
              "encoder": encoder_identity}
    cached = store.get("stages/groups", inputs)
    if cached is not None:
        return cached
    by_id = {value["evidence_id"]: value for value in records}
    batches = list(enumerate(chunks(records, config.group_batch_size)))

    def process(item):
        batch_index, batch = item
        return batch_index, initial_partition(store, teacher, batch, batch_index)

    initial = []
    for completed, item in enumerate(ordered_map(process, batches, teacher_workers(teacher)), start=1):
        initial.append(item)
        print(f"[local-bank] partition={completed}/{len(batches)} groups={len(item[1]['groups'])}", flush=True)
    initial.sort(key=lambda item: item[0])
    groups = [group for _, result in initial for group in result["groups"]]
    rounds, vector_cache = [], {}
    for round_index in range(config.group_consolidation_rounds):
        before = len(groups)
        groups, merges = consolidate_round(store, teacher, groups, by_id, config, encoder,
                                           encoder_identity, round_index, vector_cache)
        rounds.append({"round": round_index + 1, "before": before,
                       "after": len(groups), "merges": merges})
        if merges == 0:
            break
    exact_ids([member for group in groups for member in group["members"]], list(by_id))
    samples = {value["evidence_id"]: value["sample_id"] for value in source}
    result = {"groups": [dict(group, distinct_sample_count=len({samples[item] for item in group["members"]}))
                         for group in sorted(groups, key=lambda value: value["group_id"])],
              "evidence_count": len(records), "semantic_decisions": "local_qwen_teacher",
              "minimum_group_size_enforced": False,
              "grouping_strategy": {"name": STRATEGY, "candidate_encoder": encoder_identity,
                  "candidate_top_k": config.group_candidate_top_k,
                  "candidate_generation_is_final_decision": False,
                  "teacher_decisions": ["partition", "match", "merge", "membership"],
                  "initial_batch_size": config.group_batch_size,
                  "candidate_window_size": config.candidate_batch_size,
                  "initial_partition_fallback_count": sum(value["fallback"] is not None for _, value in initial),
                  "rounds": rounds}}
    return store.put("stages/groups", result, inputs)
