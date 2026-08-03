#!/usr/bin/env python3
"""Versioned launcher for MemGen training and evaluation.

Example:
    python scripts/launch_experiment.py train configs/experiments/kodcode/weaver_sft.yaml
    python scripts/launch_experiment.py eval configs/experiments/kodcode/eval.yaml \
        --set model.load_model_path=/mnt/memgen/checkpoints/foo

The experiment file contains only overrides of its base config.  The launcher
adds a unique run name, launches Accelerate, and lets ``main.py`` snapshot the
fully resolved configuration and Git metadata into the output directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a versioned MemGen experiment.")
    parser.add_argument("mode", choices=("train", "eval"))
    parser.add_argument("experiment_cfg", help="versioned experiment YAML")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="highest-priority OmegaConf override; repeat as needed",
    )
    parser.add_argument(
        "--run-id",
        help="optional stable run identifier; default includes config stem, time and Git SHA",
    )
    parser.add_argument(
        "--devices",
        help="CUDA device list for this server run, e.g. 0 or 0,1; defaults to CUDA_VISIBLE_DEVICES",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        help="Accelerate process count; defaults to the number of selected CUDA devices",
    )
    return parser.parse_args()


def checkpoint_path(config) -> Path | None:
    value = config.get("model", {}).get("load_model_path")
    if value in (None, "", "null", "None"):
        return None
    return Path(str(value)).expanduser()


def validate_resolved_config(mode: str, config) -> None:
    """Reject common stage wiring mistakes before reserving GPUs."""
    run = config.get("run", {})
    train_weaver = bool(run.get("train_weaver", False))
    train_trigger = bool(run.get("train_trigger", False))
    checkpoint = checkpoint_path(config)

    if mode == "train":
        if train_weaver == train_trigger:
            raise ValueError(
                "A training experiment must enable exactly one of "
                "run.train_weaver and run.train_trigger."
            )
        is_weaver_grpo = train_weaver and run.get("train_weaver_method") == "grpo"
        if (is_weaver_grpo or train_trigger) and checkpoint is None:
            raise ValueError(
                "Weaver GRPO and Trigger GRPO must set model.load_model_path "
                "to the preceding stage's checkpoint."
            )
    elif mode == "eval":
        if train_weaver or train_trigger:
            raise ValueError("Evaluation configs must set train_weaver=false and train_trigger=false.")
        if checkpoint is None:
            raise ValueError("Evaluation requires --set model.load_model_path=<absolute checkpoint path>.")

    if checkpoint is not None:
        required = ("config.json", "projs.bin", "weaver.bin", "trigger.bin", "weaver", "trigger")
        missing = [name for name in required if not (checkpoint / name).exists()]
        if missing:
            raise FileNotFoundError(
                f"Invalid MemGen checkpoint {checkpoint}; missing: {', '.join(missing)}"
            )


def main() -> None:
    args = parse_args()
    experiment_path = (PROJECT_ROOT / args.experiment_cfg).resolve()
    if not experiment_path.is_file():
        raise FileNotFoundError(f"Experiment config not found: {experiment_path}")

    experiment_cfg = OmegaConf.load(experiment_path)
    base_cfg = experiment_cfg.get("base_cfg_path")
    if not base_cfg:
        raise ValueError(f"{experiment_path} must define base_cfg_path")
    base_path = (PROJECT_ROOT / str(base_cfg)).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    launcher = experiment_cfg.get("launcher", {})
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    num_processes = args.num_processes or len(devices.split(","))
    accelerate_cfg = (PROJECT_ROOT / str(launcher.get("accelerate_config", "configs/zero2.yaml"))).resolve()
    if not accelerate_cfg.is_file():
        raise FileNotFoundError(f"Accelerate config not found: {accelerate_cfg}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = args.run_id or f"{experiment_path.stem}-{timestamp}-{git_short_sha()}"
    options = [
        f"run.experiment_name={run_id}",
        *args.set,
        f"run.mode={ 'evaluate' if args.mode == 'eval' else 'train' }",
    ]

    experiment_overrides = OmegaConf.create(experiment_cfg)
    experiment_overrides.pop("base_cfg_path", None)
    experiment_overrides.pop("launcher", None)
    resolved_config = OmegaConf.merge(
        OmegaConf.load(base_path),
        experiment_overrides,
        OmegaConf.from_dotlist(options),
    )
    validate_resolved_config(args.mode, resolved_config)

    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(accelerate_cfg),
        "--num_processes",
        str(num_processes),
        "main.py",
        "--cfg-path",
        str(base_path),
        "--experiment-cfg",
        str(experiment_path),
        "--options",
        *options,
    ]
    print("[launch] CUDA_VISIBLE_DEVICES=" + os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    print("[launch] run_id=" + run_id)
    print("[launch] " + " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
