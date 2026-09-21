#!/usr/bin/env python3
"""Repair unresolved partition receipts with short aliases, then finish grouping.

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
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.bank_construction.artifacts import Store, digest, file_digest, read_json, run_lock
from memgen.experience.bank_construction.config import ConstructionConfig, ModelConfig
from memgen.experience.bank_construction.grouping import run_grouping
from memgen.experience.bank_construction.prompts import messages
from memgen.experience.bank_construction.schemas import validate_partition
from memgen.experience.bank_construction.sources import environment, implementation_hashes
from memgen.experience.bank_construction.teacher import Teacher, parse_object
from memgen.experience.bank_construction.vllm_teacher import VLLMTeacher


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repair-only", action="store_true",
                        help="Repair currently unresolved receipts without running the grouping stage")
    parser.add_argument("--max-cycles", type=int, default=64,
                        help="Maximum repair/restart cycles while new bad partition requests are discovered")
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
            for cycle in range(1, args.max_cycles + 1):
                try:
                    result = run_grouping(store, teacher, config)
                    print(f"[partition-repair] grouping_complete groups={len(result['groups'])} "
                          f"evidence_count={result['evidence_count']}", flush=True)
                    return
                except RuntimeError as exc:
                    if "Teacher task partition exhausted retries" not in str(exc):
                        raise
                    repaired = repair_pending(store, config, adapter)
                    if not repaired:
                        raise
                    print(f"[partition-repair] cycle={cycle} newly_repaired={repaired}; restarting grouping", flush=True)
            raise RuntimeError("Partition repair exceeded --max-cycles")
        finally:
            adapter.close()


if __name__ == "__main__":
    main()
