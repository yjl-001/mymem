"""Valid-only utility table; keeps card construction separate from empirical quality."""
from __future__ import annotations

from collections import Counter
import math
import statistics

from data.utils.math_utils import diagnose_gsm8k_completion
from .artifacts import digest
from .compilation import primary_records, select_bank, validate_compiled
from .dataset import selected_rows


def token_statistics(values):
    ordered = sorted(values)
    if not values:
        return {"total": 0, "mean": None, "median": None, "min": None, "max": None, "p90": None, "at_limit": 0}
    return {"total": sum(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "min": ordered[0], "max": ordered[-1], "p90": ordered[math.ceil(.9 * len(values)) - 1],
            "at_limit": sum(v == 1024 for v in values)}


def metrics(rows, baseline):
    count = len(rows)
    correct = sum(r["reward"] for r in rows)
    gain = sum(r["reward"] == 1 and b["reward"] == 0 for r, b in zip(rows, baseline))
    harm = sum(r["reward"] == 0 and b["reward"] == 1 for r, b in zip(rows, baseline))
    return {"count": count, "correct": correct, "accuracy": correct / count if count else None,
            "gain": gain, "harm": harm, "net_gain": gain - harm,
            "generated_tokens": token_statistics([r["generated_token_count"] for r in rows])}


def run_evaluate(store, config, model_factory):
    split = store.require("split")
    bundle = validate_compiled(store, store.require("stages/compile"))
    rows = selected_rows(split, "valid", config)
    inputs = {"split": digest(split), "bundle": digest(bundle), "samples": [r["sample_id"] for r in rows],
              "evaluation_mode": config.evaluation_mode}
    cached = store.get("stages/evaluate", inputs)
    if cached is not None:
        return cached
    from memgen.model.v4_3_question_selector import generate, encode_text
    from memgen.model.v4_3_prefix_equivalence import prefix_bank
    records = primary_records(store)
    model = None
    def get_model():
        nonlocal model
        if model is None:
            model = model_factory()
        return model
    records_by_id = {record["bank_id"]: record for record in records}
    results = {}
    try:
        # Freeze all question-only choices before creating any answer outcomes.
        for row in rows:
            key = "valid_choices/" + row["sample_id"]
            binding = {"question": row["question"], "bundle": digest(bundle)}
            choice = store.get(key, binding)
            if choice is None:
                vector = encode_text(get_model().runtime, row["question"])
                bid, score = select_bank(vector, bundle["entries"])
                store.put(key, {"bank_id": bid or "no_memory", "cosine": score, "query_vector": vector}, binding)
        def run_action(record, indices):
            action = "no_memory" if record is None else record["bank_id"]
            memory = None
            branch = {}
            for completed, i in enumerate(indices, start=1):
                row = rows[i]
                key = "valid_results/" + row["sample_id"] + "/" + action
                binding = {"sample": row, "record": record, "bundle": digest(bundle)}
                result = store.get(key, binding)
                if result is None:
                    runtime = get_model().runtime
                    if record is not None and memory is None:
                        memory = prefix_bank(store.root / "prefix_kv", record, runtime, store.profile_hash, validate_only=True)
                    _, generated = generate(runtime, row["question"], None if record is None else record["descriptor"], memory)
                    text = runtime.tokenizer.decode(generated["continuation_token_ids"], skip_special_tokens=True)
                    verifier = diagnose_gsm8k_completion(text, row["scoring_solution"])
                    result = store.put(key, {"sample_id": row["sample_id"], "action": action,
                        "text": text, "token_ids": generated["continuation_token_ids"],
                        "generated_token_count": len(generated["continuation_token_ids"]),
                        "stop_reason": generated["stop_reason"], "reward": verifier["reward"], "verifier": verifier}, binding)
                branch[i] = result
                print(f"[local-bank] valid action={action} question={completed}/{len(indices)}", flush=True)
            return branch

        baseline_by_index = run_action(None, range(len(rows)))
        results["no_memory"] = [baseline_by_index[i] for i in range(len(rows))]
        choices = [store.require("valid_choices/" + row["sample_id"])["bank_id"] for row in rows]
        if config.evaluation_mode == "exhaustive":
            for record in records:
                branch = run_action(record, range(len(rows)))
                results[record["bank_id"]] = [branch[i] for i in range(len(rows))]
            chosen = [results[action][i] if action != "no_memory" else results["no_memory"][i]
                      for i, action in enumerate(choices)]
        else:
            chosen = [None] * len(rows)
            by_action = {}
            for index, action in enumerate(choices):
                if action == "no_memory":
                    chosen[index] = results["no_memory"][index]
                else:
                    by_action.setdefault(action, []).append(index)
            for action in sorted(by_action):
                if action not in records_by_id:
                    raise ValueError("Selector chose an unavailable Bank")
                chosen_by_index = run_action(records_by_id[action], by_action[action])
                for index, result in chosen_by_index.items():
                    chosen[index] = result
            if any(result is None for result in chosen):
                raise RuntimeError("Selector-only evaluation left a sample unresolved")
    finally:
        if model is not None:
            model.close()
    base = results["no_memory"]
    report = {"evaluation_role": "builder_valid_development_not_final_test", "sample_count": len(rows),
        "bank_count": len(records), "official_test_used": False, "construction_samples_used": False,
        "evaluation_mode": config.evaluation_mode,
        "baseline": metrics(base, base), "semantic_top1": {**metrics(chosen, base), "selection_counts": dict(Counter(choices))},
        "per_bank": ({record["bank_id"]: metrics(results[record["bank_id"]], base) for record in records}
                     if config.evaluation_mode == "exhaustive" else {}),
        "per_bank_complete": config.evaluation_mode == "exhaustive",
        "generation_count": len(rows) * (len(records) + 1) if config.evaluation_mode == "exhaustive"
                            else len(rows) + sum(choice != "no_memory" for choice in choices),
        "generated_token_policy": "completion_including_emitted_eos_excluding_question_and_memory",
        "memory_cost_excluded_from_completion_tokens": True, "automatic_empirical_promotion": False}
    return store.put("stages/evaluate", report, inputs)
