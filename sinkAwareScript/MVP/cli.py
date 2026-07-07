import argparse
import random

import torch
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    """只负责实验层参数，不复刻训练脚本里的全部 run config。

    这个 MVP 是小样本、离线验证脚本，因此默认值偏保守：greedy decode、
    eager attention、较短 max_new_tokens。正式跑大样本前建议先 limit=3。
    """

    parser = argparse.ArgumentParser(
        description=(
            "KodCode MVP for testing whether SinkMassZ ranks delimiter "
            "latent-memory insertion points better than first-K/random."
        )
    )
    parser.add_argument("--cfg-path", default="configs/latent_memory/kodcode.yaml")
    parser.add_argument("--load-model-path", default=None, help="MemGen Weaver checkpoint directory.")
    parser.add_argument("--output-dir", default="output/sink_aware_mvp/kodcode")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    # eager 最稳妥：能拿 attention weights。flash_attention_2 更快，但通常不适合
    # 直接读取完整 attentions，本实验默认不用它。
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--max-candidates-per-sample", type=int, default=20)
    # MVP 先用“前若干有效 key position”作为 sink key 近似；后续可以替换成
    # 离线校准出的 sink token/head 配置。
    parser.add_argument("--sink-key-count", type=int, default=8)
    parser.add_argument("--sink-layer-window", type=int, default=0, help="0 means all layers; N means last N layers.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--skip-single-insertion-branches", action="store_true")
    parser.add_argument("--reference-modes", nargs="+", default=["no_memory", "prompt_only"],
                        choices=["no_memory", "prompt_only"])
    parser.add_argument("--run-strategy-rollouts", action="store_true",
                        help="Exploratory only: run multi-insertion strategy rollouts.")
    parser.add_argument("--random-trials", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true", help="Remove old result files in output-dir before running.")
    parser.add_argument("--options", nargs="+", default=None, help="Optional OmegaConf dotlist overrides.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机源，让 random/sampling 策略可复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    """把命令行字符串转成 torch dtype。"""

    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def apply_overrides(config, options: list[str] | None):
    """支持和 main.py 一样的 OmegaConf dotlist 覆盖风格。"""

    if not options:
        return config
    if "=" in options[0]:
        override = OmegaConf.from_dotlist(options)
    else:
        pairs = [f"{key}={value}" for key, value in zip(options[0::2], options[1::2])]
        override = OmegaConf.from_dotlist(pairs)
    return OmegaConf.merge(config, override)


def to_plain_dict(node) -> dict:
    """把 OmegaConf 节点转为普通 dict，方便传给项目已有 builder/model。"""

    return OmegaConf.to_container(node, resolve=True)
