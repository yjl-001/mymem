# MemGen 配置说明

## 配置系统概览

MemGen 使用 OmegaConf 管理配置。YAML 文件定义默认参数，命令行通过 `--options key.path value` 覆盖。

```
python main.py --cfg-path configs/latent_memory/kodcode.yaml --options model.model_name Qwen/Qwen2.5-1.5B-Instruct
```

`Config` 类（[common/config.py](../common/config.py)）将 YAML 拆为三个顶层模块，分别由不同组件消费：

| 模块 | 键名 | 消费者 |
|------|------|--------|
| 模型配置 | `model` | `MemGenModel.from_config()` |
| 数据配置 | `dataset` | `get_data_builder()` |
| 运行配置 | `run` | `MemGenRunner` |

---

## 配置文件清单

| 文件 | 用途 |
|------|------|
| `configs/latent_memory/*.yaml` | 各数据集的模型+数据+训练配置（主配置） |
| `configs/zero2.yaml` | DeepSpeed ZeRO-2 分布式配置，供 `accelerate launch` 读取 |

---

## 一、`model` — 模型配置

### 1.1 顶层字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | `str` | — | 基础 LLM 的 HuggingFace 模型名，如 `Qwen/Qwen2.5-1.5B-Instruct`。Reasoner 和 Trigger 也以此为默认值 |
| `load_model_path` | `str` / `null` | `null` | 已训练的 MemGen checkpoint 路径，评测时使用。`null` 表示从头初始化 |
| `max_prompt_aug_num` | `int` | `1` | Prompt 阶段最多插入多少次 latent memory（插入次数上限） |
| `max_inference_aug_num` | `int` | `5` | 推理生成阶段最多插入多少次 latent memory（插入次数上限） |

> `max_prompt_aug_num` 和 `max_inference_aug_num` 控制**插入次数**，不是每次插入的 token 数量。每次插入多少个 token 由下文的 `prompt_latents_len` / `inference_latents_len` 决定。

### 1.2 `model.weaver` — Weaver 配置

Weaver 是负责生成 latent memory 的 LoRA 模型。它包含两套独立的可学习 query latent 参数，分别用于 prompt 增强和推理增强。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | `str` | — | Weaver 的底座模型 HuggingFace 名，与 `model.model_name` 通常一致 |
| `prompt_latents_len` | `int` | `8` | 每次 prompt 增强时插入的 latent token 数量（即文档中的 `k`） |
| `inference_latents_len` | `int` | `8` | 每次推理增强时插入的 latent token 数量（即文档中的 `k`） |

#### `model.weaver.lora_config` — Weaver LoRA 配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | `int` | `16` | LoRA 秩 |
| `lora_alpha` | `int` | `32` | LoRA 缩放系数，实际缩放比为 `alpha/r` |
| `target_modules` | `list[str]` | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` | 应用 LoRA 的线性层模块名 |
| `lora_dropout` | `float` | `0.1` | LoRA dropout 概率 |
| `bias` | `str` | `"none"` | 偏置项训练策略：`"none"` 不训练偏置 |
| `task_type` | `str` | `"CAUSAL_LM"` | PEFT task type |

### 1.3 `model.trigger` — Trigger 配置

Trigger 是一个二分类器，在推理时判断每个分隔符位置是否需要插入 latent memory。Trigger 训练阶段才激活。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | `str` | — | Trigger 的底座模型 HuggingFace 名 |
| `active` | `bool` | `False` | 是否启用 Trigger。`False` 时默认始终输出「增强」信号 |

#### `model.trigger.lora_config` — Trigger LoRA 配置

与 Weaver 的 LoRA 配置结构相同，见 1.2 节。

---

## 二、`dataset` — 数据集配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | `str` | — | 数据集名，对应 `data/__init__.py` 中的 `_DATA_BUILDER_MAP` 注册名。可选：`gsm8k`, `kodcode`, `gpqa`, `triviaqa` |
| `mode` | `str` | `"sft"` | 数据构造模式。`"sft"` 构造监督微调数据，`"grpo"` 构造 GRPO 偏好数据 |

### 2.1 `dataset.sft` — SFT 数据划分

| 字段 | 类型 | 说明 |
|------|------|------|
| `train_ratio` | `float` | 训练集比例（仅 kodcode 使用三比例切分） |
| `valid_ratio` / `val_ratio` | `float` | 验证集比例 |
| `test_ratio` | `float` | 测试集比例（仅 kodcode） |

> 注：不同数据集 YAML 中键名略有不同（`valid_ratio` vs `val_ratio`），实际由各数据集的 Builder 解析。

### 2.2 `dataset.grpo` — GRPO 数据划分

结构同 `dataset.sft`。

---

## 三、`run` — 运行 / 训练配置

### 3.1 顶层字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | `int` | `42` | 随机种子 |
| `mode` | `str` | `"train"` | 运行模式。`"train"` 训练，`"evaluate"` 评测 |
| `train_weaver` | `bool` | `True` | 是否训练 Weaver |
| `train_weaver_method` | `str` | `"sft"` | Weaver 训练方式。`"sft"` 或 `"grpo"` |
| `train_trigger` | `bool` | `False` | 是否训练 Trigger |
| `train_trigger_method` | `str` | `"grpo"` | Trigger 训练方式，仅支持 `"grpo"` |

### 3.2 `run.weaver.sft` — Weaver SFT 训练参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_train_epochs` | `int` | `2` | 训练轮数 |
| `per_device_train_batch_size` | `int` | `4` | 单卡训练 batch size |
| `per_device_eval_batch_size` | `int` | `4` | 单卡验证 batch size |
| `gradient_accumulation_steps` | `int` | `1` | 梯度累积步数 |
| `optim` | `str` | `"adamw_torch"` | 优化器 |
| `lr_scheduler_type` | `str` | `"cosine"` | 学习率调度策略 |
| `warmup_ratio` | `float` | `0.1` | 学习率预热比例 |
| `learning_rate` | `float` | `1e-5` | 学习率 |
| `logging_strategy` | `str` | `"steps"` | 日志记录策略 |
| `logging_steps` | `int` | `1` | 每隔多少步记录一次日志 |
| `eval_strategy` | `str` | `"epoch"` | 评估策略 |
| `eval_steps` | `int` | `100` | 每隔多少步评估（当前用 epoch 策略时此字段未生效） |
| `save_strategy` | `str` | `"epoch"` | 模型保存策略 |
| `save_steps` | `int` | `100` | 每隔多少步保存（当前用 epoch 策略时此字段未生效） |
| `assistant_only_loss` | `bool` | 看数据集 | 是否仅在 assistant 回复 token 上计算 loss。对话式数据集（gsm8k, triviaqa）为 `True`，其他为 `False` |
| `max_length` | `int` | `1024` | 最大序列长度 |
| `remove_unused_columns` | `bool` | `False` | 是否移除数据集中未使用的列 |
| `load_best_model_at_end` | `bool` | `True` | 训练结束后是否加载最优模型 |
| `bf16` | `bool` | `True` | 是否使用 bfloat16 混合精度 |
| `report_to` | `list[str]` | `["tensorboard"]` | 日志报告后端 |

### 3.3 `run.weaver.grpo` — Weaver GRPO 训练参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_generations` | `int` | `8` | 每个 prompt 生成多少个回复（GRPO 采样数） |
| `num_iterations` | `int` | `1` | GRPO 迭代次数 |
| `beta` | `float` | `0.0` | KL 惩罚系数，`0.0` 表示不使用 KL 惩罚 |
| `loss_type` | `str` | 看数据集 | 损失函数类型。`"bnpo"` (Batch Normalized Preference Optimization) 或 `"grpo"` |
| `max_prompt_length` | `int` | `1024` | prompt 最大长度 |
| `max_completion_length` | `int` | 看数据集 | 模型生成回复的最大长度 |
| `temperature` | `float` | `1.0` | 生成时的温度参数 |

其余字段（optimizer、schedule、logging/save/eval 策略、bf16 等）与 SFT 相同，见 3.2 节。

### 3.4 `run.trigger.grpo` — Trigger GRPO 训练参数

字段与 Weaver GRPO (3.3) 结构相同，以下列出 Trigger 特定的默认值差异：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | `1e-5` (gsm8k: `1e-6`) | Trigger 训练学习率，部分数据集使用更小的值 |

### 3.5 `run.interaction` — 交互 / 评测配置

评测时控制模型与环境交互的行为（多轮数据集如 TriviaQA 中意义更明显）。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_turns` | `int` | `1`（triviaqa: `5`） | 最大交互轮数。单轮数据集为 1，多轮数据集 > 1 |
| `max_start_length` | `int` | `1024` | 初始 prompt 的最大长度 |
| `max_prompt_length` | `int` | `4096` | 多轮交互中 prompt 的最大长度（含所有历史） |
| `max_response_length` | `int` | `1024` | 模型单次回复的最大长度 |
| `max_obs_length` | `int` | `512` | 环境反馈（observation）的最大长度 |
| `temperature` | `float` | `0.0`（gpqa: `1.0`） | 生成温度。`0.0` 即贪心解码 |
| `batch_size` | `int` | `8` | 评测 batch size |
| `weaver_do_sample` | `bool` | `False` | Weaver 生成时是否随机采样 |
| `trigger_do_sample` | `bool` | `False` | Trigger 判断时是否随机采样 |

---

## 四、`configs/zero2.yaml` — DeepSpeed ZeRO-2 配置

由 `accelerate launch --config_file=configs/zero2.yaml` 读取，控制分布式训练行为。

| 字段 | 值 | 说明 |
|------|-----|------|
| `compute_environment` | `LOCAL_MACHINE` | 计算环境类型 |
| `distributed_type` | `DEEPSPEED` | 分布式后端 |
| `deepspeed_config.zero_stage` | `2` | ZeRO 优化阶段（2：分片优化器状态 + 梯度） |
| `deepspeed_config.zero3_init_flag` | `false` | 不使用 ZeRO-3 参数初始化 |
| `deepspeed_config.offload_optimizer_device` | `none` | 不卸载优化器状态 |
| `deepspeed_config.offload_param_device` | `none` | 不卸载参数 |
| `mixed_precision` | `no` | DeepSpeed 侧不管理混合精度（由 HF Trainer 的 `bf16: True` 控制） |
| `num_processes` | `1` | GPU 数量，由启动脚本自动覆盖 |
| `main_process_port` | `44326` | 主进程通信端口 |

---

## 五、各数据集配置差异

| 参数 | kodcode | gsm8k | gpqa | triviaqa |
|------|---------|-------|------|----------|
| `max_prompt_aug_num` | `1` | `1` | `1` | `8` |
| `max_inference_aug_num` | `5` | `5` | `5` | `0` |
| `max_completion_length` (weaver GRPO) | `512` | `1024` | `512` | `512` |
| `max_completion_length` (trigger GRPO) | `512` | `1024` | `512` | `512` |
| `assistant_only_loss` | `False` | `True` | `False` | `True` |
| `loss_type` (weaver GRPO) | `bnpo` | `bnpo` | `grpo` | `grpo` |
| `loss_type` (trigger GRPO) | `bnpo` | `bnpo` | `bnpo` | `bnpo` |
| `interaction.temperature` | `0.0` | `0.0` | `1.0` | `0.0` |
| `interaction.max_turns` | `1` | `1` | `1` | `5` |
| 数据类型 | 代码 (StaticEnv) | 数学 (StaticEnv) | 科学QA (StaticEnv) | 知识QA (DynamicEnv) |

### kodcode 特有的数据划分字段

`dataset.sft` 和 `dataset.grpo` 下使用三比例切分：

| 字段 | 默认值 |
|------|--------|
| `train_ratio` | `0.7` |
| `valid_ratio` | `0.1` |
| `test_ratio` | `0.2` |

其他数据集仅使用 `valid_ratio`/`val_ratio`（训练集 = 1 - 验证集比例）。

---

## 六、命令行覆盖

通过 `--options` 可以覆盖 YAML 中的任意字段，使用 OmegaConf dotlist 格式：

```bash
# 覆盖模型名
--options model.model_name Qwen/Qwen2.5-1.5B-Instruct

# 覆盖 Weaver 训练方式
--options run.train_weaver_method grpo

# 覆盖多个字段
--options run.train_weaver True run.train_trigger False model.max_prompt_aug_num 2
```

这会在 YAML 之上合并用户指定的值，最终传给 `Config` 的三个子模块。

---

## 七、MemGenConfig 代码层映射

YAML 配置中的 `model.*` 字段最终会传入 `MemGenConfig`（[configuration_memgen.py](../memgen/model/configuration_memgen.py)），对应关系：

| YAML 路径 | `MemGenConfig` 属性 |
|-----------|---------------------|
| `model.weaver.lora_config` | `weaver_lora_config` |
| `model.weaver.prompt_latents_len` | `prompt_latents_len` |
| `model.weaver.inference_latents_len` | `inference_latents_len` |
| `model.trigger.active` | `trigger_active` |
| `model.trigger.lora_config` | `trigger_lora_config` |
| `model.max_prompt_aug_num` | `max_prompt_aug_num` |
| `model.max_inference_aug_num` | `max_inference_aug_num` |
