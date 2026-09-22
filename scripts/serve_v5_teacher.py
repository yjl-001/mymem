#!/usr/bin/env python3
"""Launch the pinned V5 Qwen3-32B teacher with vLLM."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from memgen.experience.v5.config import V5Config
from memgen.experience.bank_construction.sources import resolve_model
from scripts.serve_local_bank_teacher import command


def main():
    import json, os, shlex, shutil
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/gsm8k/v5.json")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.8)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or any(not item.isdigit() for item in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpus must contain distinct nonnegative indices")
    if len(gpu_ids) not in {1, 2, 4, 8}:
        parser.error("tensor parallelism must use 1, 2, 4, or 8 GPUs")
    config = V5Config.load(args.config)
    identity = resolve_model(config.teacher)
    argv = command(config, identity, gpu_count=len(gpu_ids), port=args.port,
        max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization)
    print(json.dumps({"teacher": identity, "gpus": gpu_ids, "command": shlex.join(argv)}), flush=True)
    if args.print_only:
        return
    executable = shutil.which("vllm")
    if executable is None:
        parser.error("Install vLLM in the serving environment")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.execv(executable, argv)


if __name__ == "__main__":
    main()
