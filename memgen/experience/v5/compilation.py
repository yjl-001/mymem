"""Compile Primary V5 cards into complete native prefix KV values and positive retrieval keys."""
from __future__ import annotations

from memgen.experience.bank_construction.artifacts import file_digest, read_json


def primary_records(store):
    cards = [store.require(key) for key in store.require("stages/cards")["keys"]]
    return sorted((card for card in cards if card["quality_tier"] == "primary"),
                  key=lambda card: card["bank_id"])


def validate_compiled(store, bundle):
    if bundle.get("selector_key_policy") != "positive-input-observable-only":
        raise ValueError("Unexpected V5 selector-key policy")
    for entry in bundle["entries"]:
        path = store.root / "prefix_kv" / (entry["bank_id"] + ".json")
        if file_digest(path) != entry["prefix_manifest_file_sha256"]:
            raise ValueError("V5 prefix manifest drift")
        manifest = read_json(path)
        tensor = path.with_suffix(".safetensors")
        if file_digest(tensor) != manifest["tensor_sha256"]:
            raise ValueError("V5 prefix tensor drift")
        if (manifest["profile_sha256"] != store.profile_hash
                or manifest["source_record_sha256"] != entry["record_sha256"]):
            raise ValueError("V5 prefix source binding drift")
    return bundle


def run_compile(store, model_factory):
    records = primary_records(store)
    inputs = {"records": records, "selector_key_policy": "positive-input-observable-only",
              "memory_value": "complete-card-native-prefix-kv"}
    cached = store.get("stages/compile", inputs)
    if cached is not None:
        return validate_compiled(store, cached)
    if not records:
        return store.put("stages/compile", {"entries": [], "bank_count": 0,
            "status": "no_primary_cards", "consumer": "native_prefix_kv", "all_layers": True,
            "selector_key_policy": "positive-input-observable-only"}, inputs)
    from .prefix import prefix_bank
    from memgen.model.v4_3_question_selector import encode_text
    model, entries = model_factory(), []
    try:
        for index, record in enumerate(records, start=1):
            ids, tensors = prefix_bank(store.root / "prefix_kv", record, model.runtime, store.profile_hash)
            del tensors
            vector_inputs = {"record_sha256": record["record_sha256"],
                             "positive_selector_text": record["selector_text"]}
            feature = store.get("retrieval/" + record["bank_id"], vector_inputs)
            if feature is None:
                feature = store.put("retrieval/" + record["bank_id"], {
                    "positive_selector_text": record["selector_text"],
                    "key_vector": encode_text(model.runtime, record["selector_text"])}, vector_inputs)
            entries.append({"bank_id": record["bank_id"], "record_sha256": record["record_sha256"],
                "descriptor": record["descriptor"], "selector_key": record["selector_key"],
                "distinct_input_count": record["distinct_input_count"], **feature,
                "prefix_token_count": len(ids), "prefix_manifest_file_sha256": file_digest(
                    store.root / "prefix_kv" / (record["bank_id"] + ".json"))})
            print(f"[v5] compile={index}/{len(records)} prefix_tokens={len(ids)}", flush=True)
    finally:
        model.close()
    return store.put("stages/compile", {"entries": entries, "bank_count": len(entries),
        "status": "compiled", "consumer": "native_prefix_kv", "all_layers": True,
        "one_bank_per_input": True, "selector_key_policy": "positive-input-observable-only",
        "exclusions_are_not_embedded": True, "memory_value": "complete-card-native-prefix-kv"}, inputs)
