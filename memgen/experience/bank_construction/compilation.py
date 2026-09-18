"""Compile primary cards into the frozen native prefix KV and retrieval index."""
from __future__ import annotations

import math
from .artifacts import digest, file_digest, read_json


def key_text(record, kind):
    if kind == "full_card":
        return record["descriptor"]
    if kind == "applicability":
        return record["card"]["applies_when"]
    return record["card"]["problem_structure"]


def select_bank(vector, entries):
    if not entries:
        return None, None
    def score(entry):
        key = entry["key_vector"]
        if len(key) != len(vector) or not all(math.isfinite(v) for v in [*key, *vector]):
            raise ValueError("Invalid retrieval vector")
        return sum(x * y for x, y in zip(vector, key))
    ranked = sorted(((score(e), e["bank_id"]) for e in entries), key=lambda p: (-p[0], p[1]))
    return ranked[0][1], ranked[0][0]


def primary_records(store):
    records = [store.require(k) for k in store.require("stages/cards")["keys"]]
    return sorted((r for r in records if r["quality_tier"] == "primary"), key=lambda r: r["bank_id"])


def validate_compiled(store, bundle):
    for entry in bundle["entries"]:
        path = store.root / "prefix_kv" / (entry["bank_id"] + ".json")
        if file_digest(path) != entry["prefix_manifest_file_sha256"]:
            raise ValueError("Prefix manifest drift")
        manifest = read_json(path)
        tensor = path.with_suffix(".safetensors")
        if file_digest(tensor) != manifest["tensor_sha256"]:
            raise ValueError("Prefix tensor drift")
        if manifest["profile_sha256"] != store.profile_hash or manifest["source_record_sha256"] != entry["record_sha256"]:
            raise ValueError("Prefix source binding drift")
    return bundle


def run_compile(store, config, model_factory):
    records = primary_records(store)
    inputs = {"records": records, "retrieval_key": config.retrieval_key}
    cached = store.get("stages/compile", inputs)
    if cached is not None:
        return validate_compiled(store, cached)
    if not records:
        # Preserve a complete, explicit empty-bank report, never pretend it is usable.
        return store.put("stages/compile", {"entries": [], "bank_count": 0, "status": "no_primary_cards",
                         "consumer": "native_prefix_kv", "all_layers": True}, inputs)
    from memgen.model.v4_3_prefix_equivalence import prefix_bank
    from memgen.model.v4_3_question_selector import encode_text
    model, entries = model_factory(), []
    try:
        for i, record in enumerate(records):
            bid = record["bank_id"]
            ids, tensors = prefix_bank(store.root / "prefix_kv", record, model.runtime, store.profile_hash)
            del tensors
            text = key_text(record, config.retrieval_key)
            key = "retrieval/" + bid
            feature = store.get(key, {"record": record, "text": text})
            if feature is None:
                feature = store.put(key, {"key_text": text, "key_vector": encode_text(model.runtime, text)},
                                    {"record": record, "text": text})
            entries.append({"bank_id": bid, "descriptor": record["descriptor"], "record_sha256": record["record_sha256"],
                            **feature, "prefix_token_count": len(ids),
                            "prefix_manifest_file_sha256": file_digest(store.root / "prefix_kv" / (bid + ".json"))})
            print(f"[local-bank] compile={i + 1}/{len(records)} prefix_tokens={len(ids)}", flush=True)
    finally:
        model.close()
    return store.put("stages/compile", {"entries": entries, "bank_count": len(entries), "status": "compiled",
        "consumer": "native_prefix_kv", "all_layers": True, "retrieval_key": config.retrieval_key,
        "selector": "semantic_top1_no_abstention_reference", "empirically_qualified": False}, inputs)
