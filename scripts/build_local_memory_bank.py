#!/usr/bin/env python3
"""Construct and validate a local-Qwen memory bank, from builder train to native prefix KV."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memgen.experience.bank_construction.artifacts import Store, read_json, run_lock
from memgen.experience.bank_construction.config import ConstructionConfig
from memgen.experience.bank_construction.pipeline import PHASES, STAGES, run
from memgen.experience.bank_construction.sources import make_profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/gsm8k/local_bank.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=tuple(PHASES), default="all",
                        help="Normal execution boundary: rollouts, bank, or both")
    parser.add_argument("--stage", choices=("all", *STAGES),
                        help="Advanced recovery entry point for one internal stage")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rollout-batch-size", type=int, help="Override simultaneous reasoner trajectories (default 32)")
    parser.add_argument("--teacher-concurrency", type=int, help="Override simultaneous vLLM teacher requests (default 16)")
    parser.add_argument("--group-candidate-top-k", type=int, help="Embedding candidates inspected by the teacher")
    parser.add_argument("--group-consolidation-rounds", type=int, help="Maximum semantic consolidation rounds")
    parser.add_argument("--evaluation-mode", choices=("selector_only", "exhaustive"))
    parser.add_argument("--rollout-source", "--reuse-rollouts-from", dest="rollout_source", type=Path,
                        help="Import a completed rollout-phase artifact into a separate Bank run")
    parser.add_argument("--plan-only", action="store_true", help="Validate config without downloads or inference")
    parser.add_argument("--validate-only", action="store_true", help="Audit a completed run without model inference")
    args = parser.parse_args(argv)
    if args.stage not in (None, "all") and args.phase != "all":
        parser.error("choose --phase or --stage, not both")
    if args.rollout_source and (args.phase == "rollouts" or args.validate_only
                                or args.stage not in (None, "all", "rollouts")):
        parser.error("--rollout-source is for a Bank/all phase or rollouts recovery stage")
    config = ConstructionConfig.load(args.config)
    overrides = {name: getattr(args, name) for name in (
                    "rollout_batch_size", "teacher_concurrency", "group_candidate_top_k",
                    "group_consolidation_rounds", "evaluation_mode")
                 if getattr(args, name) is not None}
    if overrides:
        config = replace(config, **overrides)
    if args.plan_only:
        print(json.dumps({"configuration": config.to_dict(), "phases": PHASES,
            "internal_recovery_stages": STAGES,
            "note": "rollouts performs no teacher inference; bank consumes completed rollouts and uses valid only"}, indent=2))
        return
    with run_lock(args.output_dir):
        path = args.output_dir / "profile.json"
        if path.exists() and not (args.resume or args.validate_only):
            raise ValueError("Run exists; use --resume for identical configuration/code")
        if args.validate_only and not path.exists():
            raise ValueError("--validate-only requires an existing completed run")
        if not path.exists() and any(p.name != ".writer.lock" for p in args.output_dir.iterdir()):
            raise ValueError("Nonempty output without a profile; use a new directory")
        scope = "rollouts" if (args.phase == "rollouts" or args.stage == "rollouts") else "bank"
        previous = read_json(path) if path.exists() else None
        if args.validate_only and previous is not None:
            scope = previous.get("implementation_scope", "bank")
        profile = make_profile(config, previous, scope=scope)
        store = Store(args.output_dir, profile)
        if args.validate_only:
            from memgen.experience.bank_construction.audit import audit_complete, audit_rollouts
            audit = audit_rollouts if scope == "rollouts" else audit_complete
            print(json.dumps(audit(store, config), indent=2))
        else:
            run(store, config, profile, phase=args.phase, stage=args.stage,
                rollout_source=args.rollout_source)


if __name__ == "__main__":
    main()
