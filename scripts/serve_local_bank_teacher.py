#!/usr/bin/env python3
"""Launch a pinned Qwen teacher in a separate vLLM environment on chosen GPUs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from memgen.experience.bank_construction.config import ConstructionConfig
from memgen.experience.bank_construction.sources import resolve_model
from memgen.experience.bank_construction.vllm_teacher import served_name


def command(config, identity, *, gpu_count, port, max_num_seqs, max_model_len, gpu_memory_utilization):
    args = ["vllm", "serve", identity["source"], "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", served_name(identity, config.teacher.dtype),
        "--dtype", config.teacher.dtype, "--tensor-parallel-size", str(gpu_count),
        "--max-num-seqs", str(max_num_seqs), "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--enable-prefix-caching", "--generation-config", "vllm"]
    if identity["revision"]:
        args += ["--revision", identity["revision"], "--tokenizer-revision", identity["revision"]]
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/gsm8k/local_bank.json")
    parser.add_argument("--gpus", default="0,1,2,3", help="Physical GPU indices; their count sets tensor parallelism")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.8)
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()
    gpu_ids = args.gpus.split(",")
    if not gpu_ids or any(not x.isdigit() for x in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        parser.error("--gpus must contain distinct nonnegative GPU indices")
    if len(gpu_ids) not in {1, 2, 4, 8}:
        parser.error("Qwen3-32B tensor parallelism supports GPU counts 1, 2, 4 or 8 here")
    if not (0 < args.gpu_memory_utilization < 1 and args.max_num_seqs > 0 and args.max_model_len > 0
            and 0 < args.port < 65536):
        parser.error("Invalid serving capacity or port")
    cfg = ConstructionConfig.load(args.config)
    identity = resolve_model(cfg.teacher)
    argv = command(cfg, identity, gpu_count=len(gpu_ids), port=args.port, max_num_seqs=args.max_num_seqs,
                   max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    print(json.dumps({"teacher": identity, "gpus": gpu_ids, "command": shlex.join(argv)}, ensure_ascii=False), flush=True)
    if args.print_only:
        return
    executable = shutil.which("vllm")
    if executable is None:
        parser.error("Install vLLM in this separate serving environment first")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.execv(executable, argv)


if __name__ == "__main__":
    main()
