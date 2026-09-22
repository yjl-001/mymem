#!/usr/bin/env python3
"""Build MemGen V5 Episodes, universal Memory Banks, native KV, and selector policy."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memgen.experience.bank_construction.artifacts import Store, read_json, run_lock
from memgen.experience.v5.config import V5Config
from memgen.experience.v5.pipeline import PHASES, STAGES, run
from memgen.experience.v5.sources import make_profile
from memgen.experience.v5.task import task_for


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/gsm8k/v5.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASES), default="all")
    parser.add_argument("--stage", choices=("all", *STAGES))
    parser.add_argument("--rollout-source", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rollout-batch-size", type=int)
    parser.add_argument("--teacher-concurrency", type=int)
    parser.add_argument("--selector-top-k", type=int)
    parser.add_argument("--no-reranker", action="store_true")
    args = parser.parse_args(argv)
    if args.stage not in (None, "all") and args.phase != "all":
        parser.error("choose --phase or --stage, not both")
    if args.rollout_source and args.phase == "rollouts":
        parser.error("--rollout-source belongs to the bank/all phase")
    config = V5Config.load(args.config)
    overrides = {name: getattr(args, name) for name in
                 ("rollout_batch_size", "teacher_concurrency", "selector_top_k")
                 if getattr(args, name) is not None}
    if args.no_reranker:
        overrides["reranker_enabled"] = False
    if overrides:
        config = replace(config, **overrides)
    if args.plan_only:
        print(json.dumps({"schema_version": "memgen-v5-plan-v1", "configuration": config.to_dict(),
            "phases": PHASES, "recovery_stages": STAGES,
            "invariants": {"selector_query": "input-only", "selector_key": "positive-applicability",
                "exclusions_embedded": False, "memory_value": "complete-card-native-prefix-kv",
                "online_tier": "primary-only", "one_bank_per_input": True}}, indent=2))
        return
    scope = "rollouts" if (args.phase == "rollouts" or args.stage == "episodes") else "bank"
    with run_lock(args.output_dir):
        profile_path = args.output_dir / "profile.json"
        if profile_path.exists() and not (args.resume or args.validate_only):
            raise ValueError("V5 run exists; use --resume for identical configuration/code")
        previous = read_json(profile_path) if profile_path.exists() else None
        if args.validate_only and previous is None:
            raise ValueError("--validate-only requires an existing V5 run")
        if previous is None and any(path.name != ".writer.lock" for path in args.output_dir.iterdir()):
            raise ValueError("Nonempty V5 output has no profile; use a new directory")
        if args.validate_only:
            scope = previous["implementation_scope"]
        profile = make_profile(config, previous, scope=scope)
        store = Store(args.output_dir, profile)
        task = task_for(config)
        if args.validate_only:
            from memgen.experience.v5.audit import audit_complete, audit_rollouts
            audit = audit_rollouts if scope == "rollouts" else audit_complete
            print(json.dumps(audit(store, config), indent=2))
        else:
            run(store, config, profile, task, phase=args.phase, stage=args.stage,
                rollout_source=args.rollout_source)


if __name__ == "__main__":
    main()
