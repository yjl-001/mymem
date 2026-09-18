"""Explicit, provenance-preserving migration of completed rollout checkpoints only."""
from pathlib import Path

from data.utils.math_utils import diagnose_gsm8k_completion
from .artifacts import Store, digest, read_json, run_lock
from .dataset import selected_rows
from .rollouts import rollout_plan

CONTRACT_FIELDS = ("dataset", "val_ratio", "split_seed", "sampling_seed", "greedy_rollouts",
                   "sampled_rollouts", "temperature", "top_p", "top_k", "max_new_tokens")
PROMPT_FILES = ("data/gsm8k/prompt.py", "data/gsm8k/splits.py", "data/utils/math_utils.py", "memgen/chat_templates.py")


def reuse_rollouts(store, config, profile, source_dir):
    source_dir = Path(source_dir).resolve()
    if source_dir == store.root or not (source_dir / "profile.json").is_file():
        raise ValueError("Rollout reuse requires a different existing run directory")
    # The old writer must be stopped. Never copy a live, growing rollout run.
    with run_lock(source_dir):
        source_profile = read_json(source_dir / "profile.json")
        for role in ("reasoner", "dataset"):
            if source_profile[role] != profile[role]:
                raise ValueError(f"Cannot reuse rollouts: {role} identity differs")
        previous = source_profile["configuration"]
        if any(previous.get(k) != config.to_dict()[k] for k in CONTRACT_FIELDS):
            raise ValueError("Cannot reuse rollouts: sampling/split contract differs")
        if previous["reasoner"] != config.to_dict()["reasoner"]:
            raise ValueError("Cannot reuse rollouts: reasoner settings differ")
        if any(source_profile["implementation"].get(k) != profile["implementation"].get(k) for k in PROMPT_FILES):
            raise ValueError("Cannot reuse rollouts: prompt/verifier implementation differs")
        source = Store(source_dir, source_profile)
        split = store.require("split")
        if source.require("split") != split:
            raise ValueError("Cannot reuse rollouts: actual dataset split differs")
        manifest = []
        for row in selected_rows(split, "train", config):
            for plan in rollout_plan(row, config):
                key = "rollouts/" + plan["rollout_id"]
                inputs = {"sample": row, "plan": plan}
                value = source.get(key, inputs)
                if value is None:
                    continue
                if any(value.get(k) != v for k, v in plan.items()):
                    raise ValueError("Reused rollout plan mismatch")
                generation = value["generation"]
                if (generation["seed"] != plan["seed"] or generation["token_count"] != len(generation["token_ids"])
                        or not 0 < generation["token_count"] <= config.max_new_tokens
                        or generation["stop_reason"] not in {"length", "eos", "completed_boxed_answer"}
                        or generation["truncated"] != (generation["stop_reason"] == "length")
                        or (generation["truncated"] and generation["token_count"] != config.max_new_tokens)):
                    raise ValueError("Reused rollout generation contract mismatch")
                if value["verifier"] != diagnose_gsm8k_completion(generation["text"], row["scoring_solution"]):
                    raise ValueError("Reused rollout verifier mismatch")
                provenance = {"source_profile_sha256": source.profile_hash, "source_key": key,
                              "source_payload_sha256": digest(value)}
                store.put(key, {**value, "reused_from": provenance}, inputs)
                manifest.append(provenance)
        store.put("imports/source_profile", source_profile, {"source_profile_sha256": source.profile_hash})
        store.put("imports/rollouts", {"source_dir": str(source_dir), "rollout_count": len(manifest),
            "records": manifest, "mixed_generation_backends": True}, {"source_profile_sha256": source.profile_hash})
        print(f"[local-bank] reused_rollouts={len(manifest)} source={source_dir}", flush=True)
