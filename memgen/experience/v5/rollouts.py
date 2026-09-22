"""Collect generic Episodes: one greedy and seven sampled outputs per train input."""
from __future__ import annotations

from time import perf_counter

from memgen.experience.bank_construction.artifacts import digest
from .task import selected_rows


def episode_plan(row, config):
    return [{"episode_id": "episode-" + digest([row["input_id"], index]),
             "input_id": row["input_id"], "index": index, "sampling": index > 0,
             "seed": int(digest([config.sampling_seed, row["input_id"], index])[:8], 16)}
            for index in range(config.greedy_rollouts + config.sampled_rollouts)]


def validate_episode(value, row, plan, config, task):
    if any(value.get(name) != plan[name] for name in plan):
        raise ValueError("Episode plan mismatch")
    generation = value.get("generation", {})
    if (generation.get("seed") != plan["seed"]
            or generation.get("token_count") != len(generation.get("token_ids", []))
            or not 0 < generation.get("token_count", 0) <= config.max_new_tokens
            or generation.get("stop_reason") not in {"length", "eos", "completed_boxed_answer"}
            or generation.get("truncated") != (generation.get("stop_reason") == "length")):
        raise ValueError("Episode generation contract mismatch")
    verified = task.verify(generation["text"], row)
    if value.get("outcome") != {**verified, "truncated": generation["truncated"]}:
        raise ValueError("Episode outcome drift")
    return value


def run_rollouts(store, config, task, model_factory):
    split = store.require("split")
    rows = selected_rows(split, "train", config)
    index, pending, model = [], [], None
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Only train inputs may produce V5 construction Episodes")
        for plan in episode_plan(row, config):
            key = "episodes/" + plan["episode_id"]
            inputs = {"input": row, "plan": plan}
            index.append(key)
            if store.get(key, inputs) is None:
                pending.append((row, plan, key, inputs))
    completed = len(index) - len(pending)
    print(f"[v5] episodes cached={completed}/{len(index)} batch_size={config.rollout_batch_size}", flush=True)
    try:
        for offset in range(0, len(pending), config.rollout_batch_size):
            batch = pending[offset:offset + config.rollout_batch_size]
            if model is None:
                model = model_factory()
            requests = [{"prompt": task.prompt(model.tokenizer, row), "seed": plan["seed"],
                         "sampling": plan["sampling"]} for row, plan, _, _ in batch]
            kwargs = {"max_new_tokens": config.max_new_tokens, "temperature": config.temperature,
                      "top_p": config.top_p, "top_k": config.top_k, "stop_on_box": task.stop_on_box}
            started = perf_counter()
            results = (model.generate_batch(requests, **kwargs) if hasattr(model, "generate_batch")
                       else [model.generate(**request, **kwargs) for request in requests])
            if len(results) != len(batch):
                raise ValueError("Reasoner returned the wrong Episode batch size")
            elapsed = perf_counter() - started
            tokens = sum(result["token_count"] for result in results)
            for (row, plan, key, inputs), generation in zip(batch, results):
                verifier = task.verify(generation["text"], row)
                record = {**plan, "input": row["input"], "output": generation["text"],
                    "generation": generation, "outcome": {**verifier, "truncated": generation["truncated"]},
                    "provenance": {"source_split": "train", "input_sha256": row["input_sha256"]}}
                validate_episode(record, row, plan, config, task)
                store.put(key, record, inputs)
            completed += len(batch)
            print(f"[v5] episodes completed={completed}/{len(index)} batch={len(batch)} "
                  f"seconds={elapsed:.2f} tokens_per_second={tokens / max(elapsed, 1e-9):.2f}", flush=True)
    finally:
        if model is not None:
            model.close()
    return store.put("stages/episodes", {"keys": index, "input_count": len(rows),
        "episode_count": len(index), "episodes_per_input": config.greedy_rollouts + config.sampled_rollouts,
        "teacher_inference_used": False}, {"split": digest(split), "configuration": config.to_dict()})
