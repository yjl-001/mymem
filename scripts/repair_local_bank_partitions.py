#!/usr/bin/env python3
"""Recover a large Bank grouping run with alias-safe, bounded candidate search.

This is deliberately an out-of-profile recovery tool: it does not change the
frozen construction modules recorded by an existing run.  Every alternate
prompt, raw response, alias mapping and accepted mapped answer is stored in the
same authenticated Store before the original request is satisfied.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.bank_construction.artifacts import Store, digest, file_digest, read_json, run_lock
from memgen.experience.bank_construction.config import ConstructionConfig, ModelConfig
from memgen.experience.bank_construction.grouping import chunks, group_record, group_summary
from memgen.experience.bank_construction.parallel import ordered_map, teacher_workers
from memgen.experience.bank_construction.prompts import VERSION, messages
from memgen.experience.bank_construction.schemas import (
    exact_ids, validate_match, validate_membership, validate_merge, validate_partition,
)
from memgen.experience.bank_construction.sources import environment, implementation_hashes
from memgen.experience.bank_construction.teacher import Teacher, parse_object
from memgen.experience.bank_construction.vllm_teacher import VLLMTeacher

STRATEGY = "alias-embedding-candidates-qwen-audit-v1"


def config_from_profile(profile):
    raw = deepcopy(profile["configuration"])
    raw["reasoner"] = ModelConfig(**raw["reasoner"])
    raw["teacher"] = ModelConfig(**raw["teacher"])
    return ConstructionConfig(**raw)


def unresolved_requests(store):
    folder = store.root / "teacher" / "partition"
    if not folder.is_dir():
        return []
    pending = []
    for path in sorted(folder.glob("*-request.json")):
        request_key = str(path.relative_to(store.root))[:-5]
        original_key = request_key.removesuffix("-request")
        request = store.require(request_key)
        if store.get(original_key, request) is None:
            pending.append((original_key, request))
    return pending


def alias_request(original_key, request):
    if request.get("task") != "partition" or not request.get("messages"):
        raise ValueError(f"Not a partition request: {original_key}")
    payload = json.loads(request["messages"][-1]["content"])
    records = payload.get("evidence")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Partition request has no evidence: {original_key}")
    ids = [record.get("evidence_id") for record in records]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Partition request has invalid source IDs: {original_key}")
    aliases = [f"E{i:02d}" for i in range(len(ids))]
    alias_payload = deepcopy(payload)
    for record, alias in zip(alias_payload["evidence"], aliases):
        record["evidence_id"] = alias
    conversation = messages("partition", alias_payload)
    conversation[0]["content"] += (
        "\nMechanical identifier protocol for this recovery request: evidence IDs are short aliases. "
        f"Return each of these aliases exactly once across groups: {json.dumps(aliases)}. "
        "Do not copy or reconstruct the original long identifiers."
    )
    repair = {
        "schema_version": "memgen-partition-alias-repair-v1",
        "original_key": original_key,
        "original_request_sha256": digest(request),
        "repair_script_sha256": file_digest(Path(__file__).resolve()),
        "aliases": [{"alias": alias, "evidence_id": evidence_id}
                    for alias, evidence_id in zip(aliases, ids)],
        "messages": conversation,
    }
    return repair, aliases, ids


def coverage_error(answer, aliases):
    members = []
    if isinstance(answer, dict) and isinstance(answer.get("groups"), list):
        for group in answer["groups"]:
            if isinstance(group, dict) and isinstance(group.get("members"), list):
                members.extend(value for value in group["members"] if isinstance(value, str))
    counts = Counter(members)
    missing = [value for value in aliases if counts[value] == 0]
    duplicate = [value for value in aliases if counts[value] > 1]
    unexpected = sorted(set(members) - set(aliases))
    return f"missing={missing}; duplicate={duplicate}; unexpected={unexpected}; expected={aliases}"


def mapped_answer(answer, aliases, ids):
    validate_partition(answer, aliases)
    mapping = dict(zip(aliases, ids))
    mapped = deepcopy(answer)
    for group in mapped["groups"]:
        group["members"] = [mapping[value] for value in group["members"]]
    validate_partition(mapped, ids)
    return mapped


def singleton_answer(repair):
    """Conservative total-coverage fallback; later semantic merge still runs."""
    groups = []
    for alias, record in zip((row["alias"] for row in repair["aliases"]),
                             json.loads(repair["messages"][-1]["content"])["evidence"]):
        signature = record["signature"]
        groups.append({"members": [alias], "method": signature["repair_operator"],
            "applies_when": signature["problem_structure"],
            "exclusions": signature.get("exclusion_conditions", []),
            "rationale": "Conservative singleton retained after the teacher could not satisfy the identifier protocol."})
    return {"groups": groups}


def repair_one(store, config, adapter, original_key, request):
    repair, aliases, ids = alias_request(original_key, request)
    repair_key = original_key + "-alias-" + digest(repair)[:16]
    store.put(repair_key + "-request", repair, repair)
    accepted = store.get(repair_key, repair)
    if accepted is not None:
        mapped = mapped_answer(accepted["answer"], aliases, ids)
        store.put(original_key, {"answer": mapped, "accepted_attempt": accepted["accepted_attempt"],
            "raw_key": accepted["raw_key"], "repair_key": repair_key,
            "acceptance_mode": "short_alias_partition_repair"}, request)
        return

    error, attempt, new_attempts = None, 0, 0
    while True:
        conversation = list(repair["messages"])
        if error:
            conversation.append({"role": "user", "content":
                "The previous response failed mechanical validation. " + error +
                ". Return a complete corrected JSON object for the original partition task."})
        inputs = {"repair": repair, "attempt": attempt, "messages": conversation}
        raw_key = repair_key + f"-attempt-{attempt}"
        generated = store.get(raw_key, inputs)
        if generated is None:
            if new_attempts >= config.teacher_retries + 1:
                break
            generated = adapter.chat(conversation, seed=int(digest(inputs)[:8], 16))
            store.put(raw_key, generated, inputs)
            new_attempts += 1
        answer = None
        try:
            if generated["truncated"]:
                raise ValueError("Teacher response was truncated")
            answer = parse_object(generated["text"])
            validate_partition(answer, aliases)
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            detail = coverage_error(answer, aliases)
            error = f"{exc}; {detail}"
            print(f"[partition-repair] request={original_key} attempt={attempt + 1} invalid={error}", flush=True)
            attempt += 1
            continue
        store.put(repair_key, {"answer": answer, "accepted_attempt": attempt, "raw_key": raw_key}, repair)
        mapped = mapped_answer(answer, aliases, ids)
        store.put(original_key, {"answer": mapped, "accepted_attempt": attempt,
            "raw_key": raw_key, "repair_key": repair_key,
            "acceptance_mode": "short_alias_partition_repair"}, request)
        print(f"[partition-repair] repaired={original_key} evidence_count={len(ids)}", flush=True)
        return
    # Do not discard or guess membership. A singleton partition is the only
    # lossless fallback and the normal cross-batch match/merge/member audit can
    # still combine it semantically later.
    answer = singleton_answer(repair)
    validate_partition(answer, aliases)
    fallback_inputs = {"repair": repair, "reason": error, "mode": "singleton_fallback"}
    fallback_key = repair_key + "-singleton-fallback"
    store.put(fallback_key, {"answer": answer, "failed_attempt_count": attempt,
        "reason": error}, fallback_inputs)
    store.put(repair_key, {"answer": answer, "accepted_attempt": attempt,
        "raw_key": fallback_key, "acceptance_mode": "singleton_fallback"}, repair)
    mapped = mapped_answer(answer, aliases, ids)
    store.put(original_key, {"answer": mapped, "accepted_attempt": attempt,
        "raw_key": fallback_key, "repair_key": repair_key,
        "acceptance_mode": "singleton_fallback"}, request)
    print(f"[partition-repair] singleton_fallback={original_key} evidence_count={len(ids)}", flush=True)


def repair_pending(store, config, adapter):
    pending = unresolved_requests(store)
    for original_key, request in pending:
        repair_one(store, config, adapter, original_key, request)
    return len(pending)


def alias_records(records, prefix):
    aliases = [f"{prefix}{i:02d}" for i in range(len(records))]
    values = deepcopy(records)
    for value, alias in zip(values, aliases):
        value["evidence_id"] = alias
    return values, aliases, dict(zip(aliases, (value["evidence_id"] for value in records)))


def initial_partition(store, teacher, records, batch_index):
    inputs = {"strategy": STRATEGY, "batch_index": batch_index, "evidence": records}
    key = f"grouping_v2/initial/{batch_index:06d}"
    cached = store.get(key, inputs)
    if cached is not None:
        return cached
    original_payload = {"context": "Initial within-batch method grouping", "evidence": records}
    original_request = {"prompt_version": VERSION, "task": "partition",
                        "messages": messages("partition", original_payload)}
    original_key = "teacher/partition/" + digest(original_request)
    legacy = store.get(original_key, original_request)
    if legacy is not None:
        validate_partition(legacy["answer"], [value["evidence_id"] for value in records])
        result = {"groups": [group_record(group) for group in legacy["answer"]["groups"]],
                  "fallback": None, "evidence_count": len(records), "source": "accepted_legacy_partition"}
        return store.put(key, result, inputs)
    short, aliases, mapping = alias_records(records, "E")
    payload = {"context": "Initial within-batch method grouping with short aliases", "evidence": short}
    fallback = None
    try:
        answer = teacher.ask("partition", payload, lambda value: validate_partition(value, aliases))
        mapped = deepcopy(answer)
        for group in mapped["groups"]:
            group["members"] = [mapping[value] for value in group["members"]]
    except RuntimeError as exc:
        fallback = str(exc)
        mapped = {"groups": [{"members": [mapping[alias]],
            "method": record["signature"]["repair_operator"],
            "applies_when": record["signature"]["problem_structure"],
            "exclusions": record["signature"].get("exclusion_conditions", []),
            "rationale": "Conservative singleton after alias partition exhausted retries."}
            for alias, record in zip(aliases, records)]}
    validate_partition(mapped, [value["evidence_id"] for value in records])
    result = {"groups": [group_record(group) for group in mapped["groups"]],
              "fallback": fallback, "evidence_count": len(records),
              "source": "short_alias_partition" if fallback is None else "singleton_fallback"}
    return store.put(key, result, inputs)


def group_text(group):
    return "\n".join(("Method: " + group["method"], "Applies when: " + group["applies_when"],
                      "Exclusions: " + "; ".join(group["exclusions"])))


def group_vector(store, group, encoder, encoder_identity):
    inputs = {"strategy": STRATEGY, "group": group_summary(group), "encoder": encoder_identity}
    key = "grouping_v2/vectors/" + group["group_id"]
    cached = store.get(key, inputs)
    if cached is None:
        cached = store.put(key, {"vector": encoder(group_text(group))}, inputs)
    vector = cached["vector"]
    if not isinstance(vector, list) or not vector or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
        raise ValueError("Invalid grouping candidate vector")
    return vector


def dot(left, right):
    if len(left) != len(right):
        raise ValueError("Grouping vector dimensions differ")
    return sum(a * b for a, b in zip(left, right))


def nearest_groups(pool, incoming_vector, get_vector, count):
    values = [get_vector(group) for group in pool]
    try:
        import numpy as np
        matrix = np.asarray(values, dtype=np.float32)
        query = np.asarray(incoming_vector, dtype=np.float32)
        if matrix.ndim != 2 or query.ndim != 1 or matrix.shape[1] != query.shape[0]:
            raise ValueError("Grouping vector dimensions differ")
        scores = (matrix @ query).tolist()
    except ImportError:
        scores = [dot(incoming_vector, value) for value in values]
    ranked = sorted(zip(scores, (group["group_id"] for group in pool), pool),
                    key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked[:count]]


def save_fallback(store, task, payload, error, fallback):
    inputs = {"strategy": STRATEGY, "task": task, "payload": payload}
    key = "grouping_v2/fallback/" + task + "/" + digest(inputs)
    return store.put(key, {"error": str(error), "fallback": fallback}, inputs)


def match_candidates(store, teacher, incoming, ranked, window_size):
    windows = list(enumerate(chunks(ranked, window_size)))

    def process(item):
        window_index, window = item
        aliases = [f"C{i:02d}" for i in range(len(window))]
        mapping = dict(zip(aliases, (group["group_id"] for group in window)))
        incoming_summary = {**group_summary(incoming), "group_id": "INCOMING"}
        candidates = [{**group_summary(group), "group_id": alias}
                      for alias, group in zip(aliases, window)]
        payload = {"incoming": incoming_summary, "candidates": candidates}
        try:
            answer = teacher.ask("match", payload, lambda value: validate_match(value, aliases))
            selected = [mapping[value] for value in answer["candidate_ids"]]
        except RuntimeError as exc:
            save_fallback(store, "match", payload, exc, {"candidate_ids": []})
            selected = []
        return window_index, len(window), selected

    results = list(ordered_map(process, windows, teacher_workers(teacher)))
    results.sort(key=lambda item: item[0])
    selected = []
    for window_index, size, values in results:
        selected.extend(values)
        print(f"[grouping-v2] match_window={window_index + 1}/{len(windows)} "
              f"candidates={size} selected_total={len(selected)}", flush=True)
    return selected


def merge_candidate(store, teacher, incoming, candidate, by_id, group_batch_size):
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
    for batch_index, batch in enumerate(chunks([by_id[value] for value in members], group_batch_size)):
        short, aliases, mapping = alias_records(batch, "E")
        check_payload = {"proposed_method": proposal, "evidence": short}
        try:
            check = teacher.ask("membership", check_payload,
                                lambda value: validate_membership(value, aliases))
            incompatible = [mapping[value] for value in check["incompatible_ids"]]
        except RuntimeError as exc:
            incompatible = [value["evidence_id"] for value in batch]
            save_fallback(store, "membership", check_payload, exc,
                          {"incompatible_ids": incompatible, "mode": "reject_merge"})
        if incompatible:
            return None
        print(f"[grouping-v2] membership_batch={batch_index + 1} checked={len(batch)}", flush=True)
    return group_record({k: proposal[k] for k in ("method", "applies_when", "exclusions", "rationale")} |
                        {"members": members})


def consolidate_round(store, teacher, groups, by_id, config, encoder, encoder_identity,
                      candidate_top_k, round_index, vector_cache):
    pool, merge_count = [], 0
    for position, original in enumerate(sorted(groups, key=lambda value: value["group_id"])):
        before = digest(pool)
        inputs = {"strategy": STRATEGY, "round": round_index, "position": position,
                  "incoming": original, "pool_before_sha256": before,
                  "candidate_top_k": candidate_top_k}
        key = f"grouping_v2/steps/round-{round_index:02d}/{position:06d}"
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

        incoming = original
        removed = []
        if pool:
            def vector_for(group):
                if group["group_id"] not in vector_cache:
                    vector_cache[group["group_id"]] = group_vector(store, group, encoder, encoder_identity)
                return vector_cache[group["group_id"]]
            incoming_vector = vector_for(incoming)
            ranked = nearest_groups(pool, incoming_vector, vector_for, candidate_top_k)
            candidate_ids = match_candidates(store, teacher, incoming, ranked, config.candidate_batch_size)
            for candidate_id in candidate_ids:
                candidate = next((value for value in pool if value["group_id"] == candidate_id), None)
                if candidate is None:
                    continue
                merged = merge_candidate(store, teacher, incoming, candidate, by_id, config.group_batch_size)
                if merged is not None:
                    incoming = merged
                    removed.append(candidate_id)
                    pool = [value for value in pool if value["group_id"] != candidate_id]
                    merge_count += 1
        step = {"incoming": incoming, "removed_group_ids": removed,
                "merge_count": len(removed), "pool_after_sha256": None}
        pool.append(incoming)
        pool.sort(key=lambda value: value["group_id"])
        step["pool_after_sha256"] = digest(pool)
        store.put(key, step, inputs)
        print(f"[grouping-v2] round={round_index} group={position + 1}/{len(groups)} "
              f"pool={len(pool)} merges={merge_count}", flush=True)
    return pool, merge_count


def run_scalable_grouping(store, teacher, config, encoder, encoder_identity,
                          *, candidate_top_k=64, consolidation_rounds=3):
    index = store.require("stages/evidence")
    source = [store.require(key) for key in index["keys"]]
    records = [{"evidence_id": value["evidence_id"], "signature": value["signature"]} for value in source]
    records.sort(key=lambda value: value["evidence_id"])
    stage_inputs = {"evidence": digest(records), "config": config.to_dict()}
    cached = store.get("stages/groups", stage_inputs)
    if cached is not None:
        return cached
    if candidate_top_k < 1 or consolidation_rounds < 1:
        raise ValueError("Grouping candidate count and rounds must be positive")
    by_id = {value["evidence_id"]: value for value in records}
    batches = list(enumerate(chunks(records, config.group_batch_size)))

    def process(item):
        batch_index, batch = item
        return batch_index, initial_partition(store, teacher, batch, batch_index)

    initial = []
    for completed, item in enumerate(ordered_map(process, batches, teacher_workers(teacher)), start=1):
        initial.append(item)
        print(f"[grouping-v2] initial_partition={completed}/{len(batches)} source={item[1]['source']} "
              f"groups={len(item[1]['groups'])}", flush=True)
    initial.sort(key=lambda item: item[0])
    groups = [group for _, result in initial for group in result["groups"]]
    initial_fallbacks = sum(result["fallback"] is not None for _, result in initial)
    initial_legacy = sum(result["source"] == "accepted_legacy_partition" for _, result in initial)
    print(f"[grouping-v2] initial_groups={len(groups)} batches={len(batches)} "
          f"legacy_reused={initial_legacy} fallbacks={initial_fallbacks}", flush=True)
    rounds = []
    vector_cache = {}
    for round_index in range(consolidation_rounds):
        before = len(groups)
        groups, merges = consolidate_round(store, teacher, groups, by_id, config, encoder,
                                           encoder_identity, candidate_top_k, round_index, vector_cache)
        rounds.append({"round": round_index, "before": before, "after": len(groups), "merges": merges})
        if merges == 0:
            break
    exact_ids([member for group in groups for member in group["members"]], list(by_id))
    samples = {value["evidence_id"]: value["sample_id"] for value in source}
    result = {"groups": [dict(group, distinct_sample_count=len({samples[m] for m in group["members"]}))
                         for group in sorted(groups, key=lambda value: value["group_id"])],
              "evidence_count": len(records), "semantic_decisions": "local_qwen_teacher",
              "minimum_group_size_enforced": False,
              "grouping_strategy": {"name": STRATEGY, "candidate_encoder": encoder_identity,
                  "candidate_top_k": candidate_top_k, "candidate_generation_is_final_decision": False,
                  "teacher_decisions": ["partition", "match", "merge", "membership"],
                  "initial_batch_size": config.group_batch_size,
                  "candidate_window_size": config.candidate_batch_size,
                  "initial_partition_legacy_reuse_count": initial_legacy,
                  "initial_partition_fallback_count": initial_fallbacks,
                  "rounds": rounds}}
    return store.put("stages/groups", result, stage_inputs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repair-only", action="store_true",
                        help="Repair currently unresolved receipts without running the grouping stage")
    parser.add_argument("--max-cycles", type=int, default=64,
                        help="Deprecated compatibility option; ignored by scalable grouping")
    parser.add_argument("--candidate-top-k", type=int, default=64,
                        help="Embedding-retrieved groups considered per incoming group")
    parser.add_argument("--consolidation-rounds", type=int, default=3)
    args = parser.parse_args(argv)
    if args.max_cycles < 1:
        parser.error("--max-cycles must be positive")
    with run_lock(args.output_dir):
        profile = read_json(args.output_dir / "profile.json")
        if implementation_hashes() != profile["implementation"]:
            raise ValueError("Frozen core implementation differs from this run; use the exact checkout that created it")
        if environment() != profile["environment"]:
            raise ValueError("Runtime environment differs from this run")
        config = config_from_profile(profile)
        if config.teacher_backend != "vllm":
            raise ValueError("Partition alias recovery currently requires the recorded local vLLM teacher")
        store = Store(args.output_dir, profile)
        adapter = VLLMTeacher(store, config, profile["teacher"])
        try:
            repaired = repair_pending(store, config, adapter)
            print(f"[partition-repair] initially_repaired={repaired}", flush=True)
            if args.repair_only:
                return
            teacher = Teacher(store, config, lambda: adapter)
            from memgen.model.local_bank import LocalModel
            from memgen.model.v4_3_question_selector import encode_text
            reasoner = LocalModel(profile["reasoner"], config.reasoner, reasoner=True)
            try:
                result = run_scalable_grouping(store, teacher, config,
                    lambda text: encode_text(reasoner.runtime, text), profile["reasoner"],
                    candidate_top_k=args.candidate_top_k,
                    consolidation_rounds=args.consolidation_rounds)
            finally:
                reasoner.close()
            print(f"[grouping-v2] grouping_complete groups={len(result['groups'])} "
                  f"evidence_count={result['evidence_count']}", flush=True)
        finally:
            adapter.close()


if __name__ == "__main__":
    main()
