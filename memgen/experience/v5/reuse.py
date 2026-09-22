"""Authenticate and import a completed V5 Episode phase into a separate Bank run."""
from __future__ import annotations

from pathlib import Path

from memgen.experience.bank_construction.artifacts import Store, digest, read_json, run_lock
from .rollouts import episode_plan, validate_episode
from .task import selected_rows

CONTRACT_FIELDS = ("task", "dataset", "val_ratio", "split_seed", "sampling_seed",
                   "greedy_rollouts", "sampled_rollouts", "temperature", "top_p", "top_k",
                   "max_new_tokens")


def reuse_episodes(store, config, profile, task, source_dir):
    source_dir = Path(source_dir).resolve()
    if source_dir == store.root or not (source_dir / "profile.json").is_file():
        raise ValueError("V5 Episode reuse requires a different completed run directory")
    with run_lock(source_dir):
        source_profile = read_json(source_dir / "profile.json")
        if source_profile.get("implementation_scope") != "rollouts":
            raise ValueError("V5 Episode source must be a rollout-phase run")
        for role in ("reasoner", "dataset"):
            if source_profile[role] != profile[role]:
                raise ValueError(f"Cannot reuse V5 Episodes: {role} identity differs")
        previous, current = source_profile["configuration"], config.to_dict()
        if any(previous.get(field) != current.get(field) for field in CONTRACT_FIELDS):
            raise ValueError("Cannot reuse V5 Episodes: sampling/split contract differs")
        if previous["reasoner"] != current["reasoner"]:
            raise ValueError("Cannot reuse V5 Episodes: reasoner settings differ")
        source = Store(source_dir, source_profile)
        split = store.require("split")
        if source.require("split") != split:
            raise ValueError("Cannot reuse V5 Episodes: actual split differs")
        planned = [(row, plan, "episodes/" + plan["episode_id"])
                   for row in selected_rows(split, "train", config)
                   for plan in episode_plan(row, config)]
        index = source.require("stages/episodes")
        expected = [key for _, _, key in planned]
        if index.get("keys") != expected or index.get("episode_count") != len(expected):
            raise ValueError("V5 Episode source is incomplete")
        authenticated = []
        for row, plan, key in planned:
            value = source.get(key, {"input": row, "plan": plan})
            if value is None:
                raise ValueError("V5 Episode source is incomplete")
            validate_episode(value, row, plan, config, task)
            authenticated.append((row, plan, key, value))
        manifest = []
        for row, plan, key, value in authenticated:
            provenance = {"source_profile_sha256": source.profile_hash, "source_key": key,
                          "source_payload_sha256": digest(value)}
            store.put(key, {**value, "reused_from": provenance}, {"input": row, "plan": plan})
            manifest.append(provenance)
        imported = {**index, "reused_from_profile_sha256": source.profile_hash}
        store.put("stages/episodes", imported,
                  {"split": digest(split), "configuration": config.to_dict()})
        store.put("imports/source_profile", source_profile,
                  {"source_profile_sha256": source.profile_hash})
        store.put("imports/episodes", {"source_dir": str(source_dir), "episode_count": len(manifest),
                  "records": manifest}, {"source_profile_sha256": source.profile_hash})
        print(f"[v5] reused_episodes={len(manifest)} source={source_dir}", flush=True)
        return imported
