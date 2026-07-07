import logging

from transformers import AutoModelForCausalLM, AutoTokenizer

import repo_paths  # noqa: F401
from cli import torch_dtype, to_plain_dict
from data import get_data_builder
from data.kodcode.env import KodCodeEnv
from memgen.model.configuration_memgen import MemGenConfig
from memgen.model.modeling_memgen import MemGenModel


LOGGER = logging.getLogger("sink-aware-mvp")


def load_model(config, args) -> MemGenModel:
    """加载 MemGen 模型，但不走 MemGenModel.from_config。

    原项目的 from_config 会硬编码 flash_attention_2；本 MVP 需要读取
    attention weights，因此这里手动加载 reasoner/weaver/trigger，并把
    attn_implementation 暴露给命令行。
    """

    model_cfg = to_plain_dict(config.model)
    if args.load_model_path:
        model_cfg["load_model_path"] = args.load_model_path

    model_name = model_cfg["model_name"]
    weaver_cfg = model_cfg.get("weaver", {})
    trigger_cfg = model_cfg.get("trigger", {})
    dtype = torch_dtype(args.torch_dtype)

    # MemGenConfig 继承底座模型 config，同时额外挂上 latent 长度、
    # LoRA 配置和 augmentation budget。
    memgen_config = MemGenConfig.from_pretrained(
        model_name,
        max_prompt_aug_num=model_cfg.get("max_prompt_aug_num", 1),
        max_inference_aug_num=model_cfg.get("max_inference_aug_num", args.budget),
        prompt_latents_len=weaver_cfg.get("prompt_latents_len", 8),
        inference_latents_len=weaver_cfg.get("inference_latents_len", 8),
        weaver_lora_config=weaver_cfg.get("lora_config"),
        trigger_active=False,
        trigger_lora_config=trigger_cfg.get("lora_config"),
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 三份 base model 物理独立，和主代码保持一致；区别只是 attention 实现可切换。
    load_kwargs = {
        "torch_dtype": dtype,
        "attn_implementation": args.attn_implementation,
    }
    reasoner = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    weaver = AutoModelForCausalLM.from_pretrained(weaver_cfg.get("model_name", model_name), **load_kwargs)
    trigger = AutoModelForCausalLM.from_pretrained(trigger_cfg.get("model_name", model_name), **load_kwargs)

    load_model_path = model_cfg.get("load_model_path")
    # load_model_path 存在时恢复训练好的 projection、latent 参数和 LoRA adapter。
    if load_model_path:
        model = MemGenModel.from_pretrained(
            load_model_path,
            config=memgen_config,
            base_tokenizer=tokenizer,
            reasoner_base_model=reasoner,
            weaver_base_model=weaver,
            trigger_base_model=trigger,
        )
    else:
        LOGGER.warning("No --load-model-path was provided; using an untrained Weaver shell.")
        model = MemGenModel(
            config=memgen_config,
            base_tokenizer=tokenizer,
            reasoner_base_model=reasoner,
            weaver_base_model=weaver,
            trigger_base_model=trigger,
        )

    # 原始 base model 按 torch_dtype 加载，但 MemGen 自己新增的 projection、
    # latent 参数和 trigger head 默认可能仍是 float32。这里统一 cast，避免
    # bf16 embedding 进入 float32 Linear 时触发 dtype mismatch。
    model.to(device=args.device, dtype=dtype)
    model.eval()
    return model


def build_dataset(config, split: str):
    """构造 KodCode split。

    这里强制 name=kodcode，是为了让该脚本专注第一个 MVP 数据集；
    后续扩展到 GSM8K/GPQA 时再抽象 dataset_name。
    """

    dataset_cfg = to_plain_dict(config.dataset)
    dataset_cfg["name"] = "kodcode"
    builder = get_data_builder(dataset_cfg)
    return builder.get_dataset_dict()[split]


def encode_prompt(model: MemGenModel, sample: dict, device: str):
    """把 KodCode 的 chat-format prompt 编成左 padding token。

    和 runner._static_evaluate 保持同样的 apply_chat_template 入口，避免
    prompt 模板漂移影响 delimiter 与 reward。
    """

    tokenizer = model.tokenizer
    tokenizer.padding_side = "left"
    prompt_inputs = tokenizer.apply_chat_template(
        [sample["prompt"]],
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=True,
        return_dict=True,
    )
    # input_ids/attention_mask: [B=1, L_prompt]
    return prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)


def decode_completion(model: MemGenModel, completion_ids: list[int]) -> str:
    """把 completion token ids 解码成 reward 函数消费的文本。"""

    # completion_ids: Python list，长度为 L_generated。
    return model.tokenizer.decode(completion_ids, skip_special_tokens=True)


def reward_completion(completion: str, sample: dict) -> float:
    """复用 KodCodeEnv 的单样本代码执行 reward。"""

    reward = KodCodeEnv.compute_reward([completion], [sample["test"]], [sample["test_info"]])[0]
    return float(reward)
