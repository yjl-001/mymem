"""Frozen-policy valid/test evaluation with accuracy, paired effects, and token cost."""
from __future__ import annotations

from collections import Counter
import math

from memgen.experience.bank_construction.artifacts import digest
from .calibration import token_statistics
from .online import choose_memory, prepare_bank
from .selector import NO_MEMORY


def paired_metrics(baseline, memory):
    if len(baseline) != len(memory) or not baseline:
        raise ValueError("V5 paired evaluation coverage mismatch")
    gain = sum(a["reward"] == 0 and b["reward"] == 1 for a, b in zip(baseline, memory))
    harm = sum(a["reward"] == 1 and b["reward"] == 0 for a, b in zip(baseline, memory))
    discordant = gain + harm
    p = min(1., 2 * sum(math.comb(discordant, i) for i in range(min(gain, harm) + 1)) /
            2**discordant) if discordant else 1.
    correct = sum(row["reward"] for row in memory)
    base_correct = sum(row["reward"] for row in baseline)
    return {"count": len(memory), "correct": correct, "accuracy": correct/len(memory),
        "accuracy_delta": (correct-base_correct)/len(memory), "gain": gain, "harm": harm,
        "net_gain": gain-harm, "paired_exact_p": p,
        "generated_tokens": token_statistics([row["generated_token_count"] for row in memory])}


def run_evaluation(output_store, bank_store, config, task, split_name, reasoner_factory,
                   reranker_factory=None):
    bundle, policy, records = prepare_bank(bank_store)
    split = bank_store.require("split")
    rows = split["splits"][split_name]
    if config.valid_limit and split_name == "valid":
        rows = rows[:config.valid_limit]
    inputs = {"bank_profile_sha256": bank_store.profile_hash, "bundle_sha256": digest(bundle),
              "policy_sha256": policy["policy_sha256"], "split": split_name,
              "input_ids": [row["input_id"] for row in rows]}
    cached = output_store.get("evaluation", inputs)
    if cached is not None:
        return cached
    from .prefix import prefix_bank
    from memgen.model.v4_3_question_selector import generate
    reasoner, reranker = reasoner_factory(), reranker_factory() if config.reranker_enabled else None
    try:
        choices = []
        for index, row in enumerate(rows, start=1):
            key = "choices/" + row["input_id"]
            binding = {"input": row["input"], "bundle_sha256": digest(bundle),
                       "policy_sha256": policy["policy_sha256"]}
            choice = output_store.get(key, binding)
            if choice is None:
                decision, candidates, scores = choose_memory(row["input"], reasoner.runtime, reranker,
                                                              bundle, policy, config.selector_top_k)
                choice = output_store.put(key, {"decision": decision, "candidates": candidates,
                                                "reranker_scores": scores}, binding)
            choices.append(choice)
            print(f"[v5-eval] choices={index}/{len(rows)}", flush=True)

        def run_action(row, bank_id):
            action = bank_id or NO_MEMORY
            record = records.get(bank_id)
            binding = {"input": row, "action": action,
                       "record_sha256": record["record_sha256"] if record else None,
                       "bundle_sha256": digest(bundle)}
            key = "results/" + row["input_id"] + "/" + action
            result = output_store.get(key, binding)
            if result is None:
                memory = (prefix_bank(bank_store.root / "prefix_kv", record, reasoner.runtime,
                                      bank_store.profile_hash, validate_only=True) if record else None)
                _, generated = generate(reasoner.runtime, row["input"],
                                        record["descriptor"] if record else None, memory)
                text = reasoner.tokenizer.decode(generated["continuation_token_ids"], skip_special_tokens=True)
                verified = task.verify(text, row)
                result = output_store.put(key, {"input_id": row["input_id"], "action": action,
                    "output": text, "reward": int(verified["reward"]), "verifier": verified,
                    "generated_token_count": len(generated["continuation_token_ids"]),
                    "stop_reason": generated["stop_reason"]}, binding)
            return result

        baseline, memory = [], []
        for index, (row, choice) in enumerate(zip(rows, choices), start=1):
            base = run_action(row, None)
            decision = choice["decision"]["bank_id"]
            selected = base if decision == NO_MEMORY else run_action(row, decision)
            baseline.append(base); memory.append(selected)
            print(f"[v5-eval] generation={index}/{len(rows)} action={decision}", flush=True)
    finally:
        reasoner.close()
        if reranker is not None:
            reranker.model = None; reranker.tokenizer = None
            try:
                import gc, torch
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except ImportError:
                pass
    base_metrics = paired_metrics(baseline, baseline)
    v5_metrics = paired_metrics(baseline, memory)
    selection_counts = dict(Counter(choice["decision"]["bank_id"] for choice in choices))
    v5_metrics.update({"memory_use_count": sum(count for key, count in selection_counts.items() if key != NO_MEMORY),
                       "selection_counts": selection_counts})
    report = {"schema_version": "memgen-v5-evaluation-v1", "complete": True,
        "evaluation_role": "official_final_test_frozen_policy" if split_name == "test"
                           else "valid_frozen_policy_diagnostic",
        "split": split_name, "sample_count": len(rows), "bank_count": len(records),
        "official_test_used": split_name == "test", "final_test_tuning": False,
        "selector_query": "input-only", "one_bank_per_input": True,
        "native_prefix_kv_frozen": True, "baseline": base_metrics, "v5": v5_metrics,
        "policy_sha256": policy["policy_sha256"],
        "generated_token_policy": "completion_including_emitted_eos_excluding_question_and_memory"}
    return output_store.put("evaluation", report, inputs)
