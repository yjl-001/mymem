# GRPO（Group Relative Policy Optimization）训练机制详解

> 本文档基于 TRL 库中 GRPOTrainer 的标准实现，梳理 GRPO 训练的完整流程与核心机制。
> 不涉及任何特定项目的定制逻辑，仅关注通用 GRPO 算法。

---

## 一、GRPO 概述

GRPO（Group Relative Policy Optimization）是一种在线强化学习算法，用于微调大语言模型。它的核心思想是：

- 对每个 prompt，用当前策略**采样多条候选回复**（一组 rollout）
- 用 reward 函数对每条回复打分
- **组内标准化**得到每条回复的 advantage（相对优劣）
- 用 PPO 风格的 clipped objective 更新策略

### 与标准 PPO 的差异

| | PPO | GRPO |
|---|---|---|
| advantage 来源 | 需要一个独立的 value model 估计 | 组内相对 reward 标准化 |
| 内存开销 | 高（需要额外训练 value model） | 低（无需 value model） |
| 适用场景 | 通用 RL | LLM 微调 |

---

## 二、核心流程：两阶段交替

GRPO 的每个 training step 包含两个交替的阶段：

```
┌──────────────────────────────────────────────────────┐
│               一个 Training Step                      │
│                                                      │
│  阶段一：采样（Rollout + Advantage 计算）              │
│    1. 当前策略 → 每个 prompt 生成多条候选回复          │
│    2. reward 函数打分                                │
│    3. 组内标准化 → advantage                          │
│    4. 缓存 old_logprob                               │
│                                                      │
│  阶段二：优化（Loss 计算 + 参数更新）                  │
│    1. 重新 forward → new_logprob                     │
│    2. 计算 PPO-clipped loss                          │
│    3. backward → optimizer.step()                    │
└──────────────────────────────────────────────────────┘
```

---

## 三、关键概念辨别："Step" 的含义

GRPO 中有两个层面会用到"step"，含义完全不同：

| | Token Step | Training Step |
|---|---|---|
| **粒度** | 生成一个 token | 一次梯度更新 |
| **位置** | `model.generate()` 内部循环 | `Trainer` 主循环 |
| **包含内容** | 一次 reasoner 前向 → 选一个 token | 多轮完整 rollout + loss + backward |
| **是否有梯度** | 无（`@torch.no_grad()`） | 有（`loss.backward()`） |
| **数量关系** | 一个 training step 可能包含数千个 token step | 1 个 training step 只做 1 次参数更新 |

---

## 四、采样阶段详解

### 4.1 输入准备

对于每个 prompt，将其 tokenize 为 `prompt_ids: (1, prompt_len)`。GRPO 使用的是**只有 prompt 没有 completion** 的数据（completion 由采样生成）。

### 4.2 Rollout 生成

对同一个 prompt 重复调用 `model.generate()` `num_generations` 次（例如 8 次），产生多条候选回复。

### 4.3 多样性的来源：温度采样

8 次生成能产生不同回复，**唯一随机源**是生成时的温度采样：

```python
def _get_next_token(logits, do_sample, temperature):
    if do_sample and temperature != 0:
        probs = softmax(logits / temperature, dim=-1)
        return multinomial(probs, num_samples=1)    # 从概率分布中随机抽取
    else:
        return argmax(logits, dim=-1)                # greedy，无随机性
```

8 次之间没有显式 seed 控制 → 每次 `multinomial` 抽到不同 token → 后续自回归全部连锁不同 → 产生 8 条不同轨迹。

### 4.4 Reward 计算

对每条生成的 completion 调用 reward 函数，得到标量 reward：

```
rewards = [r₁, r₂, r₃, ..., r₈]    # 例如 [1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0]
```

### 4.5 Advantage 计算（组内标准化）

将同一 prompt 的 `num_generations` 条候选视为一组，组内标准化：

```python
# 重组: (8,) → 按 prompt 分组的视图
# 假设 batch 中有多个 prompt，每个有 num_generations 条候选
# 实际 batch_size = num_prompts × num_generations

mean_grouped = rewards.view(-1, num_generations).mean(dim=1)   # 每组均值
std_grouped  = rewards.view(-1, num_generations).std(dim=1)    # 每组标准差

# 每条 completion 的 advantage
advantage = (reward - mean_grouped) / (std_grouped + 1e-4)
```

**为什么组内标准化？**

- 消除 prompt 固有难度的差异：难题可能全体低 reward，但优秀的回复仍有正 advantage
- 相对优势比绝对 reward 更稳定
- advantage 的符号告诉策略"这条回复比组内均值好还是差"

**示例：**

```
rewards:    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]
mean:       0.75
std:        ~0.46

advantages: [+0.54, +0.54, +0.54, +0.54, +0.54, -1.62, -1.62, +0.54]
              ↑ 优秀回复，鼓励           ↑ 差回复，惩罚          ↑ 优秀
```

### 4.6 Old Logprob 缓存

采样阶段还会对每条 completion 调用 `model.forward()`，计算每条 completion 上每个 token 的 log probability：

```python
old_per_token_logps: (batch_size, completion_len)  # 每个 token 一个 logprob
```

同时计算 `supervise_mask`，标记哪些 token 真正应该参与监督（排除 padding、tool info 等）。

---

## 五、优化阶段详解

### 5.1 重新前向计算

优化阶段对同样的 (prompt, completion) 数据重新调用 `model.forward()`：

```python
per_token_logps, supervise_mask = model.forward(input_ids, labels)
# per_token_logps: (batch_size, completion_len)
```

### 5.2 PPO-Style Clipped Objective

GRPO 使用和 PPO 一致的 clipped surrogate objective：

```python
# 1. 计算重要性采样比率
ratio = exp(new_logprob - old_logprob)          # coef_1: (B, completion_len)

# 2. 裁剪比率 (防止策略变化过大)
ratio_clipped = clamp(ratio, 1 - ε_low, 1 + ε_high)  # coef_2

# 3. 两段式 loss（PPO-clip）
per_token_loss1 = ratio * advantage.unsqueeze(1)      # 广播到每个 token
per_token_loss2 = ratio_clipped * advantage.unsqueeze(1)
per_token_loss = -min(per_token_loss1, per_token_loss2)
```

**Clipping 的意义：**

- 当 advantage > 0：ratio 不能 > 1+ε，防止过度增大好的回复的 logprob
- 当 advantage < 0：ratio 不能 < 1-ε，防止过度减小差的回复的 logprob
- `min()` 取两段中的保守值

### 5.3 Loss Mask

并非 completion 中所有 token 都参与 loss。只有同时满足多个条件的 token 才被计算：

```python
supervised_mask = completion_mask * supervise_mask * old_supervise_mask * ref_supervise_mask
```

被 mask 掉的内容通常包括：

- padding token
- tool info / observation（非 agent 自身回复）
- chat 模板标记（如 `<|im_start|>assistant\n`）

### 5.4 Loss 聚合

```python
# 每条轨迹内 token 维度的平均
loss_per_trajectory = (per_token_loss * supervised_mask).sum(-1) / supervised_mask.sum(-1)

# 所有轨迹平均
loss = loss_per_trajectory.mean()

# 一次 backward
loss.backward()
```

### 5.5 多条轨迹全部参与更新

GRPO **不会只选最好的那条**，而是**所有 `num_generations` 条全部参与梯度更新**：

```
轨迹₁: adv=+0.54 → loss=-0.54 → 正向梯度：增大这些 token 的 logprob
轨迹₂: adv=+0.54 → loss=-0.54 → 正向梯度
轨迹₃: adv=+0.54 → loss=-0.54 → 正向梯度
轨迹₄: adv=-1.62 → loss=+1.62 → 反向梯度：减小这些 token 的 logprob
...

total_loss = mean(loss₁, loss₂, ..., loss₈)
```

核心优势：

- 充分利用所有采样数据
- 连续梯度信号（不是二值的"好/坏"）
- 组内标准化 + 加权，减少方差

---

## 六、核心参数：num_iterations / gradient_accumulation_steps

这两个参数决定了 "old logprob" 和 "new logprob" 是否一致。

### 6.1 条件判断

```python
if num_iterations > 1 or steps_per_generation > gradient_accumulation_steps:
    old_per_token_logps = 真的计算并缓存   # 之后参数会变 → old ≠ new
else:
    old_per_token_logps = None              # 参数不会变 → old == new
```

### 6.2 num_iterations

**含义**：在同一批 rollouts 上做几轮优化。

```
num_iterations = 1:
  采样 → forward → loss → backward → step
  old == new（因为参数从采样到优化没变过）

num_iterations = 4:
  采样 → 缓存 old_logprob(θ₀)
  iter1: forward(θ₀) → old==new → loss₁ → backward → step → θ₁
  iter2: forward(θ₁) → old≠new → loss₂ → backward → step → θ₂
  iter3: forward(θ₂) → old≠new → loss₃ → backward → step → θ₃
  iter4: forward(θ₃) → old≠new → loss₄ → backward → step → θ₄
```

### 6.3 gradient_accumulation_steps

**含义**：几次 forward+backward 后才执行一次 `optimizer.step()`。

```
gradient_accumulation_steps = 4:
  iter1: forward → loss₁ → backward（梯度累加，不 step）
  iter2: forward → loss₂ → backward（梯度累加，不 step）
  iter3: forward → loss₃ → backward（梯度累加，不 step）
  iter4: forward → loss₄ → backward（梯度累加）
         optimizer.step()  ← 4 次后才执行一次
```

这 4 次 iteration 之间参数没变，所以 old 和 new 相同。

### 6.4 steps_per_generation

**含义**：过多少个 training step 重新生成 rollouts。

```
steps_per_generation = 2:
  gen1: θ₀ → rollouts → 缓存 old_logprob(θ₀)
    step1: forward(θ₀) → loss → backward → step → θ₁  ← old==new
    step2: forward(θ₁) → loss → backward → step → θ₂  ← old≠new（模型变了但用旧 rollout）
  gen2: θ₂ → 新 rollouts → 缓存 old_logprob(θ₂)
    step3: forward(θ₂) → loss → backward → step → θ₃  ← old==new
    ...
```

### 6.5 四种场景总结

| 场景 | num_iter | grad_accum | old vs new | 本质 |
|---|---|---|---|---|
| A | 1 | 1 | 永远相等 | REINFORCE w/ baseline |
| B | 4 | 1 | 第1次相等，之后不等 | 标准 PPO multi-epoch |
| C | 4 | 2 | 前2次相等，后2次不等 | 梯度累积 + PPO multi-epoch |
| D | 1 | 4 | 每次重新生成，永远相等 | 大等效 batch 的 REINFORCE |

---

## 七、REINFORCE vs PPO 的退化关系

### num_iterations=1 时（REINFORCE with baseline）

```python
old == new  →  ratio = exp(new - old) = 1
             →  ratio_clipped = clamp(1, 1-ε, 1+ε) = 1
             →  per_token_loss = -min(1×A, 1×A) = -A
```

学习信号**完全来自 advantage**：

- 正 advantage → loss 为负 → backward 增大这些 token 的 logprob
- 负 advantage → loss 为正 → backward 减小这些 token 的 logprob

### num_iterations > 1 时（PPO）

```python
old ≠ new  →  ratio = exp(new - old) ≠ 1
           →  ratio_clipped 可能生效
           →  学习信号 = advantage × ratio
```

多一重保护：策略变化过大时 clipping 生效，防止单步更新过猛。

---

## 八、完整 Training Step 流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                     ONE TRAINING STEP                            │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 采样阶段 (torch.no_grad())                                 │  │
│  │                                                           │  │
│  │ 对每个 prompt:                                            │  │
│  │   model.generate() × num_generations 次                   │  │
│  │   → 产生多条候选 completion                                │  │
│  │      多样性来源: temperature → multinomial sampling        │  │
│  │                                                           │  │
│  │ compute_reward(completions, solutions)                    │  │
│  │   → 每条 completion 一个标量 reward                       │  │
│  │                                                           │  │
│  │ advantage = (reward - group_mean) / group_std             │  │
│  │   → 组内标准化                                             │  │
│  │                                                           │  │
│  │ old_logprob = model.forward(full_sequences).logprob       │  │
│  │   → 缓存每个 token 的 log probability                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                             ↓                                    │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 优化阶段 (有梯度)                                           │  │
│  │                                                           │  │
│  │ new_logprob = model.forward(full_sequences).logprob       │  │
│  │                                                           │  │
│  │ ratio = exp(new_logprob - old_logprob)                    │  │
│  │ ratio_clipped = clamp(ratio, 1-ε, 1+ε)                    │  │
│  │                                                           │  │
│  │ per_token_loss = -min(ratio × A, ratio_clipped × A)       │  │
│  │                                                           │  │
│  │ mask = completion_mask × supervise_mask                   │  │
│  │       × old_supervise_mask × ref_supervise_mask           │  │
│  │                                                           │  │
│  │ loss = mean(per_token_loss × mask)                        │  │
│  │                                                           │  │
│  │ loss.backward() → optimizer.step()                        │  │
│  │                                                           │  │
│  │ 所有 num_generations 条轨迹全部参与梯度更新                 │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 九、关键 FAQ

### Q1: 优化阶段和采样阶段的 logprob 会一样吗？

当 `num_iterations=1` 且 `gradient_accumulation_steps` 与 `steps_per_generation` 相同时，**同一个 step 内 old == new**。因为输入相同、参数相同、前向逻辑确定。此时 GRPO 退化为 REINFORCE with baseline，学习信号完全来自 advantage。

当 `num_iterations > 1` 时，第 2+ 轮迭代的参数已经变了，old ≠ new，PPO clipping 开始生效。

### Q2: 多条轨迹中选哪条更新？

**全部参与更新**。每条用自己的 advantage 加权。正 advantage 的轨迹被鼓励，负 advantage 的轨迹被压制。不是"best-of-N"式的二选一。

### Q3: Advantage 只在采样阶段计算吗？

是的。采样阶段算出 reward → advantage 后缓存下来。优化阶段直接使用缓存的 advantage，**不再重新调用 reward 函数**。

### Q4: 同一个 completion 内不同 token 的 advantage 一样吗？

同一个 completion 的所有 token 共享同一个 sequence-level advantage 标量。它被广播（broadcast）到该 completion 的每个 token 位置。

### Q5: 如何保证生成出不同的轨迹？

唯一随机源是 **reasoner 的温度采样**（`torch.multinomial`）。每次从中随机抽取下一个 token。一旦某步抽到不同 token，后续所有自回归都会进入不同的分支。
