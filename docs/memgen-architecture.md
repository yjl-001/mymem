# MemGen 代码架构详解

> 结合源码，深入理解 MemGen 的 Weaver、Trigger 算法及整体流程。

---

## 目录

1. [整体架构](#1-整体架构)
2. [Weaver 算法](#2-weaver-算法)
3. [Trigger 算法](#3-trigger-算法)
4. [训练流程](#4-训练流程)
5. [推理生成流程](#5-推理生成流程)
6. [关键设计细节](#6-关键设计细节)

---

## 1. 整体架构

### 1.1 三个模型

MemGen 在 frozen base model（Reasoner）的推理流中插入**潜在记忆 token**，由 Weaver 和 Trigger 两个模块协作完成。

```
MemGenModel  (memgen/model/modeling_memgen.py:31)
│
├── Reasoner   Qwen2.5-1.5B (始终 frozen)
│   ├── Embedding Table   token ID → 向量  (frozen)
│   └── Transformer Blocks               (frozen)
│
├── Weaver     Qwen2.5-1.5B + LoRA (可训练)
│   ├── prompt_query_latents      可学习参数
│   ├── inference_query_latents   可学习参数
│   ├── prompt/inference_latent_ln + scale
│   └── LoRA adapter
│
├── Trigger    Qwen2.5-1.5B + LoRA (可训练)
│   ├── LoRA adapter
│   └── output_layer  Linear(1536, 2)  二分类头
│
├── reasoner_to_weaver  Linear(D_r, D_w)  可训练
└── weaver_to_reasoner  Linear(D_w, D_r)  可训练
```

**Weaver** — 生成 latent memory tokens（生成记忆内容）
**Trigger** — 决定何时插入 latent memory（门控开关）
**Reasoner** — 在 latent memory 增强下进行推理（推理引擎）

### 1.2 时序关系：串行，非并行

```
推理时每一步:
  ┌─────────┐     ┌─────────┐     ┌──────────┐
  │ Trigger │ ──► │ Weaver  │ ──► │ Reasoner │
  │ 决策    │     │ 生成记忆 │     │ 生成token │
  └─────────┘     └─────────┘     └──────────┘
      ↑               ↑               ↑
   只用 input_ids   用 embedding   用 embedding
                   序列作为输入    序列(含latent)
                                   作为输入
```

Trigger 和 Weaver 不总执行——Trigger 只在分隔符处评估，只有决策为 1 时才走 Weaver。

### 1.3 项目结构速览

| 文件 | 作用 |
|------|------|
| `main.py` | 入口，config 解析，组件构建 |
| `memgen/runner.py` | `MemGenRunner` — 训练/评估编排 |
| `memgen/model/modeling_memgen.py` | `MemGenModel` — 核心：forward / generate / save / load |
| `memgen/model/modeling_utils.py` | Mixin：LoRA切换、生成辅助函数、Trigger调用入口 |
| `memgen/model/weaver.py` | `MemGenWeaver` — latent augmentation 模块 |
| `memgen/model/trigger.py` | `MemGenTrigger` — 二分类门控 |
| `memgen/model/configuration_memgen.py` | `MemGenConfig` |
| `memgen/trainer/weaver_grpo_trainer.py` | `WeaverGRPOTrainer` — Weaver GRPO |
| `memgen/trainer/trigger_grpo_trainer.py` | `TriggerGRPOTrainer` — Trigger GRPO |
| `memgen/utils.py` | Chat template、eval recorder、参数工具 |
| `common/config.py` | OmegaConf 配置系统 |
| `data/base_env.py` | `BaseEnv / StaticEnv / DynamicEnv` — 环境抽象 |
| `interactions/base_interaction.py` | `InteractionManager` — 生成循环抽象 |

---

## 2. Weaver 算法

### 2.1 模型结构

```python
# memgen/model/weaver.py
class MemGenWeaver(nn.Module):
    adapter_name = "weaver"

    def __init__(self, model: PeftModel, prompt_latents_len, inference_latents_len):
        self.model = model   # PeftModel(LoRA) 包裹的完整 CausalLM

        # 两套可学习的 query latents
        self.prompt_query_latents    = nn.Parameter(randn(prompt_latents_len, hidden_size))
        self.inference_query_latents = nn.Parameter(randn(inference_latents_len, hidden_size))

        # 每套的归一化层
        self.prompt_latent_ln    = nn.LayerNorm(hidden_size)
        self.inference_latent_ln = nn.LayerNorm(hidden_size)
        self.prompt_latent_scale    = nn.Parameter(torch.ones(1))
        self.inference_latent_scale = nn.Parameter(torch.ones(1))
```

### 2.2 输入输出

```python
# Weaver 被调用时（generate() line 521-529）:
weaver_inputs_embeds = self.reasoner_to_weaver(current_inputs_embeds)
# ↑ 将 Reasoner embedding 空间的序列投影到 Weaver 空间

weaver_hidden_states, attn_mask, pos_ids = weaver.augment_prompt(  # 或 augment_inference
    weaver_inputs_embeds,       # [B, L, D_weaver]  当前序列在 Weaver 空间的表示
    candidate_attention_mask,   # [B, L]            attention mask
    candidate_position_ids      # [B, L]            位置索引
)

# Weaver 返回:
latent_hidden_states   # [B, k, D_weaver]  — 生成的潜在记忆
latent_attention_mask  # [B, k]            — 全 1
latent_position_ids    # [B, k]            — 递增的位置索引
```

**输入是什么？**
**Reasoner Embedding Table 查表得到的原始 token embedding**（不是 Reasoner Transformer 的 hidden states），加上之前 Weaver 产出的 latent memory embedding（经过 `weaver_to_reasoner` 投影到 Reasoner 空间）。

```python
# current_inputs_embeds 的组成:
[prompt_emb | gen_0_emb | gen_1_emb | latent_emb_from_previous | gen_2_emb | ...]
 ↑查表       ↑查表       ↑查表        ↑Weaver 上一次的 hidden    ↑查表
                                       states 投影到 Reasoner 空间
```

### 2.3 核心方法 `_augment`

```python
# memgen/model/weaver.py:52-94
def _augment(self, latents, latent_ln, latent_scale,
             inputs_embeds, attention_mask, position_ids):

    # 1. 归一化 + 缩放 query latents，扩展到 batch
    latents = latent_ln(latents) * latent_scale                 # [k, D]
    latents = latents.unsqueeze(0).repeat(batch_size, 1, 1)     # [B, k, D]

    # 2. 拼接到序列末尾
    inputs_embeds = cat([inputs_embeds, latents], dim=1)        # [B, L+k, D]

    # 3. 扩展 attention_mask（latent 全可见）
    latents_mask = ones([B, k])
    attention_mask = cat([attention_mask, latents_mask])        # [B, L+k]

    # 4. 扩展 position_ids（从当前最大位置递增）
    last_pos = position_ids.max(dim=1)[0]                       # [B]
    latents_pos = last_pos.unsqueeze(1) + arange(k) + 1          # [B, k]
    position_ids = cat([position_ids, latents_pos])             # [B, L+k]

    # 5. 喂给 Weaver LoRA 模型
    outputs = self.model(
        inputs_embeds=inputs_embeds,             # [B, L+k, D]
        attention_mask=attention_mask,           # [B, L+k]
        position_ids=position_ids,               # [B, L+k]
        output_hidden_states=True,
    )

    # 6. 取 latent 位置的 hidden states → 这就是"潜在记忆"
    hidden_states = outputs.hidden_states[-1]                    # [B, L+k, D]
    latents_hidden_states = hidden_states[:, -k:, :]             # [B, k, D]

    return latents_hidden_states, latents_mask, latents_position_ids
```

### 2.4 直觉理解

Query latents 是**可学习的"问题槽位"**。Weaver LoRA 模型读完整个序列后，在这些槽位位置产出 hidden states——这些 hidden states 就是"读完上下文的记忆总结"。

```
类比:
  原文: "写一个排序函数\ndef sort(arr):\n    for i in range(len(arr)):\n"
  问题: [____] [____] [____] [____]        ← query latents
  回答: "注意边界" "用索引比较" "考虑空数组"  ← latent hidden states (记忆)
```

两套 query latents 对应两种场景：
- **prompt** — "读完问题后要记住什么？"
- **inference** — "推理到一半要记住什么？"

---

## 3. Trigger 算法

### 3.1 模型结构

```python
# memgen/model/trigger.py
class MemGenTrigger(nn.Module):
    adapter_name = "trigger"

    def __init__(self, model: PeftModel, active: bool):
        self.active = active
        self.model = model    # PeftModel(LoRA) 包裹的完整 CausalLM
        self.output_layer = nn.Linear(hidden_size, 2)
        # dim 0 = "不插入", dim 1 = "插入"
```

**`active` 开关**：Weaver 训练时 `active=False`，Trigger forward 直接返回 class=1（始终 augment）。Trigger 训练时 `active=True`，正常走模型 + 分类头。

```python
# trigger.py:20-42
def forward(self, input_ids, attention_mask, position_ids):
    if self.active:
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]       # [B, L, D]
        logits = self.output_layer(hidden_states)        # [B, L, 2]
    else:
        logits = torch.zeros(batch_size, seq_len, 2)
        logits[..., 1] = 1.0   # 强制 class=1 (始终 augment)
    return logits
```

### 3.2 触发的时机

在 `generate()` 循环的每一步调用 `_should_augment()`：

```python
# modeling_utils.py:246-295
def _should_augment(self, input_ids, sentence_augment_count,
                    do_sample, temperature, is_prompt):

    if is_prompt:   # i == 0，生成第一个 token 前
        aug_vector = zeros([B])        # 所有序列强制评估
    else:           # i > 0
        aug_vector = full([B], -100)   # 默认不评估
        ends_with_delimiter → aug_vector = 0   # 只在分隔符后标记候选
        over_limit → aug_vector = -100         # 超限的排除

    # 跑 Trigger 模型
    trigger_indices = (aug_vector != -100).nonzero()
    if trigger_indices.numel() > 0:
        trigger_logits = trigger(input_ids[...], attn_mask[...], pos_ids[...])
        last_token_logits = trigger_logits[:, -1, :]  # [N, 2]
        next_tokens = argmax/sample(last_token_logits)  # 0 or 1
        aug_vector[trigger_indices] = next_tokens

    return aug_vector  # [B], 每个值: -100(不评估) / 0(不插入) / 1(插入)
```

| 阶段 | 时机 | 触发条件 |
|------|------|---------|
| Prompt (i=0) | 生成第一个 token 前 | **无条件**，所有序列都评估 |
| Inference (i>0) | 每步生成后 | ① 最后 token 是 `,` `.` `\n` ② 未达 `max_inference_aug_num` |

### 3.3 Trigger 的输入

```
Trigger 接收:
  - input_ids:      当前完整序列的 token ID（prompt + 全部已生成token）
  - attention_mask: 从 input_ids 构造的 mask
  - position_ids:   从 attention_mask cumsum 生成

它只取最后一个位置的 logits 做决策:
  trigger_logits[:, -1, :]  →  [logit_no_augment, logit_augment]
  argmax → 0 或 1
```

Trigger 本质上是一个**轻量级门控网络**：读完整段序列，只做一个二分类——"此时此地，是否需要在生成下一个 token 之前插入一段潜在记忆？"

### 3.4 Trigger 的训练

```
TriggerGRPOTrainer (memgen/trainer/trigger_grpo_trainer.py):
  - Weaver frozen, Reasoner frozen
  - generate(return_augmentation_mask=True) → 收集 trigger 在每个位置的决策轨迹
  - augmentation_mask: [B, completion_len]
      -100 → 未评估（非分隔符处）
      0   → 决策为"不插入"
      1   → 决策为"插入"
  - Reward 来自 env.compute_reward
  - GRPO loss 只算 augmentation 决策有效的位置（!= -100）
```

---

## 4. 训练流程

### 4.1 两阶段训练

```python
# memgen/runner.py:190-200
def train(self):
    if self.train_weaver:
        trainer = self._create_weaver_trainer()
        self.model.fix_component('trigger')    # 冻结 Trigger
    if self.train_trigger:
        trainer = self._create_trigger_trainer()
        self.model.fix_component('weaver')     # 冻结 Weaver
```

### 4.2 Weaver SFT

```python
# runner.py:127-134
weaver_trainer = SFTTrainer(
    model=self.model,
    args=self.weaver_sft_training_args,   # lr=1e-5, epochs=2, max_length=1024
    train_dataset=self.weaver_train_dataset,
    processing_class=self.processing_class,
)
```

**可训练参数：**
| 组件 | 状态 |
|------|------|
| Reasoner（含 Embedding Table） | ❌ frozen |
| Trigger | ❌ frozen |
| Weaver 基座模型 | ❌ frozen（只训练 LoRA） |
| Weaver LoRA (lora_A, lora_B) | ✅ 训练 |
| Weaver query_latents | ✅ 训练 |
| Weaver LayerNorm + scale | ✅ 训练 |
| reasoner_to_weaver | ✅ 训练 |
| weaver_to_reasoner | ✅ 训练 |

**Loss 计算 (`_forward` 中)：**

```python
# modeling_memgen.py:367-372
shift_logits = all_logits[..., :-1, :]
shift_labels = all_labels[..., 1:]
loss = CrossEntropy(shift_logits, shift_labels, ignore_index=-100)
```

**Latent mask 的作用**：在 `_forward` 中逐段构建，标记哪些位置是 latent token。最终 shifted 后，**预测 "latent token 本身" 的 logits 被排除出 loss**。但 latent token 通过 attention 影响后续真实 token 的预测——这种影响是训练信号的一部分。

### 4.3 Weaver GRPO

```python
# runner.py:147-158
weaver_trainer = WeaverGRPOTrainer(
    model=self.model,
    reward_funcs=[self.env_cls.compute_reward],
    args=self.weaver_grpo_training_args,   # num_generations=8
    train_dataset=self.weaver_train_dataset,
    generation_manager=self.generation_manager,  # trigger_do_sample=False
)
```

**一个 training step 分两阶段：**

**阶段 1 — `_generate_and_score_completions`（no_grad）：**
```
对每个 prompt 生成 num_generations=8 个 completion（weaver_do_sample=True）
  → env.compute_reward(completions) → reward
  → Group-relative advantage: (reward - group_mean) / group_std
  → 同一 prompt 的 completion 相互比较，好的正 advantage，差的负 advantage
```

**阶段 2 — `_compute_loss`（有梯度）：**
```python
# weaver_grpo_trainer.py:375-407
ratio = exp(per_token_logps - old_per_token_logps)
ratio_clipped = clamp(ratio, 1-ε, 1+ε)
per_token_loss = -min(ratio * advantage, ratio_clipped * advantage)
loss = mean(per_token_loss * supervised_mask)
```

标准 PPO clipping：限制策略更新幅度，稳定训练。

### 4.4 Trigger GRPO

同 Weaver GRPO 架构，但：
- `generate()` 时 `return_augmentation_mask=True`，记录 trigger 在每个位置的决策轨迹
- Loss 只算 trigger 做决策的位置（`augmentation_mask != -100`）
- 只更新 Trigger 的 LoRA + output_layer

---

## 5. 推理生成流程

### 5.1 `generate()` 主循环

```python
# modeling_memgen.py:452-629
@torch.no_grad()
def generate(self, input_ids, attention_mask, generation_config, ...):

    # ──── 初始化 ────
    current_inputs_embeds = reasoner.get_input_embeddings()(input_ids)
    current_input_ids = input_ids
    current_cache = None
    sentence_augment_count = zeros([B])

    # ──── 主循环 ────
    for i in range(max_new_tokens):

        # Step A: Trigger 决策
        augment_decision = _should_augment(
            current_input_ids, sentence_augment_count,
            do_sample=trigger_do_sample, is_prompt=(i==0)
        )   # [B]: -100/0/1

        # Step B: 如果决定 augment → Weaver 运行
        if augment_decision == 1:
            weaver_emb = reasoner_to_weaver(current_inputs_embeds)
            latent = weaver.augment_prompt/inference(weaver_emb, ...)
            latent_emb = weaver_to_reasoner(latent)
            current_inputs_embeds = cat([current_inputs_embeds, latent_emb])
            current_cache = None              # KV cache 失效
            sentence_augment_count++

        # Step C: 全部达到上限 → 一次性生成剩余
        if all(sentence_augment_count >= max_augment_num):
            generated = reasoner.generate(current_inputs_embeds, ...)
            current_input_ids = cat([current_input_ids, generated])
            break

        # Step D: 单步生成
        if current_cache is not None:
            reasoner_input = current_inputs_embeds[:, -1:]  # 只用最后一个 token
        else:
            reasoner_input = current_inputs_embeds           # 完整 prefill

        outputs = reasoner(reasoner_input, past_key_values=current_cache, ...)
        next_token = sample(outputs.logits[:, -1])
        追加到 current_input_ids 和 current_inputs_embeds
        更新 current_cache

        if all(next_token == eos): break

    return current_input_ids  # 纯 token ID 序列（不含 latent）
```

### 5.2 一条序列的完整旅程

```
Prompt: "写一个排序函数"
══════════════════════════════════════════════════════

Embedding 查表: prompt token IDs → prompt embeddings [B, L_prompt, 1536]

i=0 (prompt aug):
  Trigger: 评估 → 决定 1 (插入)
  Weaver:  读 prompt embeddings
           追加 prompt_query_latents [Q0 Q1 Q2 Q3]
           LoRA 前向 → 取 Q 位置的 hidden
           投影回 Reasoner 空间
           拼入: [prompt_emb | P_L0 P_L1 P_L2 P_L3]
  cache = None

Reasoner prefill: [prompt_emb | P_L0 P_L1 P_L2 P_L3] → logits
  → 生成 "def" → 追加到序列

i=1: 最后 token "f"，不是分隔符 → Trigger 跳过
  Reasoner decode: 只处理 "f" 的 emb → 生成 " "

i=2: 最后 token " "，不是分隔符 → Trigger 跳过
  ...

生成了 "def sort(arr):\n"
  → 最后 token "\n" 是分隔符！

i=k: Trigger 评估 → 决定 1 (插入)
  Weaver: 读 [全部 emb 序列]，推理 query latents 拼末尾
          LoRA 前向 → hidden → 投影
          拼入: [... | I_L0 I_L1 I_L2 I_L3]
  cache = None

Reasoner prefill: 完整序列 + latent → 生成新 token（被记忆增强）

... 循环直到 EOS 或 max_new_tokens
```

---

## 6. 关键设计细节

### 6.1 Latent Memory 为什么能直接插入？

**Transformer 的输入是连续向量，不依赖 token ID。** 无论向量来自 Embedding Table 查表还是 Weaver 生成，只要维度一致，Transformer 一视同仁。

```
current_inputs_embeds:
  [emb("def")] [emb("sort")] [emb("(")] [LATENT_0] [LATENT_1] [emb("arr")] ...
       ↑            ↑           ↑          ↑           ↑           ↑
    查表得到    查表得到    查表得到   Weaver生成  Weaver生成   查表得到
```

两个投影层负责空间转换：
- `reasoner_to_weaver`：Reasoner embedding 空间 → Weaver 空间
- `weaver_to_reasoner`：Weaver hidden states → Reasoner embedding 空间

### 6.2 位置编码处理

每次 augmentation 后，通过 `_generate_position_ids` 重新计算整个序列的位置：

```python
# modeling_utils.py:91-93
def _generate_position_ids(self, attention_mask):
    position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids
```

Latent token 占据真实位置槽位，后续真实 token 的位置被推移。RoPE 天然支持任意位置距离，且训练时用了同样的机制，所以不会出问题。

### 6.3 KV Cache 管理

每次 augmentation 后 `current_cache = None`（line 561），下次 forward 完整 prefill。

原因：
- Latent token 需要新的 KV 条目
- Batch 内非 augment 序列左填充后位置变了
- 简单安全 > 复杂优化

### 6.4 Batch 内的动态对齐

当 batch 中部分序列 augment、部分不 augment 时：

```python
# modeling_memgen.py:544-556
non_augment_indices = torch.where(augment_decision != 1)[0]
if len(non_augment_indices) > 0:
    pad_len = weaver.prompt_latents_num if i == 0 else weaver.inference_latents_num
    左补 pad_len 个零 → 维度与 augment 序列对齐
    补零的 attention_mask=0 → 不参与 attention 计算
```

### 6.5 `current_input_ids` vs `current_inputs_embeds`

```
current_input_ids:        [1533, 8394, 7, 12005, ...]
                          离散整数 token ID 序列
                          latent memory 没有 ID，不在这里
                          给 Trigger 用（检测分隔符）

current_inputs_embeds:    [e0, e1, e2, L0, L1, e3, ...]
                          连续向量序列（embedding 空间）
                          包含 latent memory embedding
                          给 Reasoner 和 Weaver 做前向计算
```

两者长度不同：`len(current_inputs_embeds) = len(current_input_ids) + latent_token_count`

### 6.6 训练配置速查

| 参数 | SFT | GRPO (Weaver) | GRPO (Trigger) |
|------|-----|---------------|----------------|
| `train_weaver` | True | True | False |
| `train_trigger` | False | False | True |
| `trigger.active` | False | False | True |
| `weaver_do_sample` | - | True | False |
| `trigger_do_sample` | - | False | True |
| `num_generations` | - | 8 | 8 |
| `learning_rate` | 1e-5 | 1e-5 | 1e-5 |
| `loss_type` | CrossEntropy | bnpo | bnpo |

---

## 附录：关键代码索引

| 想了解... | 看这里 |
|-----------|--------|
| 整体模型结构 | `memgen/model/modeling_memgen.py:31-75` `MemGenModel.__init__` |
| Weaver 内部 | `memgen/model/weaver.py:52-94` `_augment` |
| Trigger 内部 | `memgen/model/trigger.py:20-42` `forward` |
| Trigger 调用入口 | `memgen/model/modeling_utils.py:246-295` `_should_augment` |
| 训练前向 `_forward` | `memgen/model/modeling_memgen.py:98-198` |
| 推理生成 `generate` | `memgen/model/modeling_memgen.py:452-629` |
| Latent mask 构造 | `memgen/model/modeling_memgen.py:126-195` |
| 位置编码 | `memgen/model/modeling_utils.py:91-93` `_generate_position_ids` |
| KV cache 管理 | `memgen/model/modeling_memgen.py:558-561, 582-607` |
| Batch 对齐填充 | `memgen/model/modeling_memgen.py:544-556` |
| Weaver GRPO | `memgen/trainer/weaver_grpo_trainer.py:143-433` |
| Trigger GRPO | `memgen/trainer/trigger_grpo_trainer.py:109-390` |
| Embedding 投影 | `memgen/model/modeling_memgen.py:64-65, 146, 162` |
| Chat template | `memgen/utils.py:17-35` `CONVERSATION_TEMPLATE` |
