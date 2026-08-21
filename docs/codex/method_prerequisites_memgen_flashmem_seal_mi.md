# 前置方法：MemGen、FlashMem、SEAL 与 Memory Inception

本文件是 [Experience-Calibrated Entropy-Gated Steering 执行计划](experience_calibrated_steering_plan.md)
的技术前置知识。它区分四篇论文各自解决的子问题，避免把“训练无关”“熵门控”或
“latent memory”混为同一个机制。

## 1. 一页对照

| 方法 | memory/content 从何而来 | 何时介入 | 如何与原状态交互 | 是否训练 | 本项目借用的部分 |
|---|---|---|---|---|---|
| [MemGen](https://arxiv.org/abs/2509.24704) | 训练后的 Weaver 依据当前上下文在线生成 latent embeddings | Trigger 在 reasoning boundary 决策 | latent embedding 插入 token embedding sequence | Weaver 与 Trigger 均需训练，reasoner 冻结 | 生成循环、boundary、预算、评测框架 |
| [FlashMem](https://arxiv.org/abs/2601.05505) | Shared-KV Consolidator 从当前 hidden state 与 backbone cache 生成 latent memory | 最后一层、去 sink 的 attention entropy 超阈值 | 生成的 embedding soft-inject 回 input stream | Consolidator 需 SFT；Cognitive Monitor 无参数 | sink-masked entropy 作为候选触发信号 |
| [SEAL](https://arxiv.org/abs/2504.07986) | 从目标模型 rollout 的不同 reasoning pattern hidden states 求均值差 | thought boundary（论文用 `\n\n`） | 选定层做 residual update `H ← H + αS` | 无反传/权重更新；需离线 rollout 和校准 | Phase 2 的 vector compiler/integrator |
| [Memory Inception (MI)](https://arxiv.org/abs/2605.06225) | 指导文本经冻结模型前向编译为 K/V slots | selected layers/heads/KV groups，query 自适应读 bank | side KV bank 与原 prompt/history KV 共同 attention | 无参数训练；需离线 selector calibration | 后续的 compiler/integrator 升级接口 |

本项目的组合不是“把四篇论文直接拼接”：

```text
MemGen：在线控制框架
FlashMem：何时值得考虑干预
SEAL：已完成并关闭的 residual-vector 假设
MI：当前用于验证经验内容传输的 query-dependent side-KV memory
```

## 2. MemGen：在线生成式 latent token

### 2.1 原论文方法

MemGen 有三个角色：冻结的 `reasoner`、可训练的 Memory Weaver，以及可训练的
Memory Trigger。每当生成处到达 delimiter 等 augmentation point，Trigger 先判断是否
调用；若调用，Weaver 读取当前上下文并产出一段连续 latent embeddings。经投影回
reasoner embedding space 后，这些 embedding 被拼到当前 token embedding sequence 中，
reasoner 再继续生成。

```text
current prompt / generated tokens
          │
     Trigger: augment?
          │ yes
          ▼
Weaver(current context) → latent embeddings
          │
concat into input-embedding sequence → frozen reasoner continues decoding
```

它的 memory 是**在线、上下文特定、由训练过的模块生成**的，不是一个离线可复用的
experience bank。因为 latent embedding 变成了序列的一部分，后续 token 的位置会被推后；
若在 decoding 中插入，也必须以新的序列状态继续生成，造成 cache/计算开销。

本仓库的实现细节与对照实验见 [MemGen 代码架构](../memgen-architecture.md)。

### 2.2 对本项目的意义

保留：delimiter/action boundary、注入预算、批处理生成循环、已有评测环境，以及
MemGen Weaver 作为训练式对照。

不沿用：每次触发都在线调用 Weaver 来生成内容。我们的目标是把“内容构造”离线化，
在线只做 `gate → select → integrate`。

## 3. FlashMem：熵触发与计算复用式 memory consolidator

### 3.1 Cognitive Monitor（本项目直接借用）

FlashMem 在当前 token 的**最后一层 attention**上，对每个 head 移除 attention-sink
集合后重新归一化，并计算平均 Shannon entropy：

\[
\tilde A_{t,h}[j] =
\frac{A_{t,h}[j]\mathbb{1}[j\notin\mathcal S_{sink}]}
{\sum_k A_{t,h}[k]\mathbb{1}[k\notin\mathcal S_{sink}]},
\qquad
\mathcal H_t=\frac1H\sum_h -\sum_j\tilde A_{t,h}[j]\log\tilde A_{t,h}[j].
\]

当 \(\mathcal H_t>\tau\) 时，论文将其解释为当前 attention 缺少聚焦、可能有较高
epistemic uncertainty，进而触发 memory consolidation。去掉 sink 很关键：BOS/初始 token
常获得与语义无关的大 attention mass；若直接计算熵，会人为制造低熵假象。

我们的 GSM8K 实验采用其思想，但作了清晰的工程化近似：在 delimiter 上，移除前
`attention_sink_token_count` 个有效 token（当前首版为 4），然后计算最后层逐 head 熵的
平均。FlashMem 没有给出通用动态 sink 识别器，因此这应称为“FlashMem 风格”，而非论文的
逐字复现。当前实现和阈值校准见 [GSM8K entropy gate](../gsm8k_entropy_gate.md)。

**重要限制**：高熵只表示“可能值得帮助”的候选信号，不等价于模型已经错误，也不等价于
任何 memory 都会有益。因此它必须与边界、预算、检索置信度和对照实验共同使用。

### 3.2 Shared-KV Consolidator（本项目当前不复用）

FlashMem 被称为轻量，但不是整个方法都 training-free。其 Cognitive Monitor 无参数；
Memory Consolidator 则是一个需 SFT 的轻量 decoder。它以 backbone 最后一层 hidden state
为种子，只学习 query projection，并 cross-attend 到 backbone 已有 KV cache；随后自回归
产生若干 latent embeddings，再 soft-inject 回 backbone input stream。

```text
last hidden state h_t → MLP seed → trained consolidator query
                                      │
                         cross-attend to frozen backbone KV cache
                                      │
                         latent embeddings → soft injection
```

它的主要创新是避免为历史上下文做额外完整编码，而非免训练地产生 memory。当前方案只借用
其 sink-masked entropy gate；我们不把 FlashMem consolidator 误称为 training-free baseline。

## 4. SEAL：离线方向、在线 residual steering

### 4.1 离线 vector 提取

SEAL 让**目标模型本身**在一小部分训练数据上生成完整 CoT，按 `\n\n` 切为 thought
blocks，并用关键词规则标为：

- `Execution`：持续执行解题；
- `Reflection`：检查、重算、质疑先前过程；
- `Transition`：转向另一条思路。

在每个 thought 的分界 token 处，提取某个 transformer layer 的 hidden state。然后计算：

\[
S = \bar H_{Execution} - \bar H_{Reflection \cup Transition}.
\]

它没有优化器、反向传播或权重更新；但它**并非没有离线成本**：需要 rollout、thought
标注、layer/strength calibration，且原论文使用训练数据来提取向量。

### 4.2 在线交互

每到 thought boundary，SEAL 在指定层直接加向量：

\[
\tilde H = H + \alpha S.
\]

这影响后续层和 next-token logits。它不改变 token 序列，不额外生成 memory token，也不对
KV cache 加 slot；因此在线代价低，但方向是全局的、对当前内容不够条件化。

### 4.3 本项目如何安全地泛化 SEAL

我们不直接复制“Execution 减 Reflection/Transition”的任务定义，而是用已验证经验构造：

\[
S_{k,l}=\operatorname{mean}(h^{+}_{k,l})-
\operatorname{mean}(h^{-}_{k,l}),
\]

其中 \(h^+\) 来自同一经验簇的 `verified_success` evidence，\(h^-\) 来自
`verified_failure` evidence。强教师可抽象、归类和质检轨迹，但只能提供文本/标签建议；
向量必须由**冻结 student 在自己的 representation space**中计算。不能把教师模型的
hidden-state vector 直接加到 student。

裸加均值差会造成 distribution shift 风险，因此执行计划使用 current-state RMS 归一化、
soft entropy gate、最大扰动比约束，以及随机向量/随机触发/反转方向对照。

## 5. Memory Inception：文本编译出的 side KV bank

### 5.1 无训练的 bank 构造

MI 的“training-free”是指没有训练参数，而不是记忆凭空出现。输入可以是 persona
descriptor、对话总结、检索事实或 reasoning heuristic。MI 将文本置于 template 中，用
冻结 base model forward，保留 guidance token 的 hidden states，并用该模型本身的 key/value
projection 形成每层每头的 K/V slots：

\[
k^{(b)}_{l,u,m}=W^{(l)}_{K,u}\operatorname{Norm}_l(h^{(b)}_{l,m}),\qquad
v^{(b)}_{l,u,m}=W^{(l)}_{V,u}\operatorname{Norm}_l(h^{(b)}_{l,m}).
\]

这些 slots 是离线 compiled artifact，可重复挂载；与 MemGen 的在线 Weaver generation
不同。

### 5.2 与原始状态交互

仅在 selector 选中的 layer/head/KV group 中，当前 query 同时 attend 原始 prompt/history
与 side bank：

\[
K_t^*=\operatorname{concat}(K^x_{\le t},K^{(1)},\ldots,K^{(B)}),\qquad
V_t^*=\operatorname{concat}(V^x_{\le t},V^{(1)},\ldots,V^{(B)}).
\]

原生 prompt/history cache 不被改写；bank 是 selected attention site 消费的 side path。当前
query 决定读取多少 bank，因此相比 SEAL 的全局 residual direction 更具条件性。

MI 可同时持有：

- `target bank`：期望靠近的策略、行为或 heuristic；
- `reference bank`：相反/不期望的策略；
- `auxiliary bank`：额外事实或启发式；
- `prompt bank`：普通 prompt/history。

它基于 target-reference evidence gap 做 query-dependent gain，并以 size-normalized bank
evidence 避免 slot 多的 bank 自然胜出。

### 5.3 选择器与 RoPE

MI 仍需 calibration：在 candidate prompts 上测量每个 layer/head（GQA 时为 KV group）对
target 相比 reference 的 query-key alignment，逐层保留 top-k unit，再选择 top-m layers，
并冻结 selector artifact。这不是训练，但它是离线模型/任务特定配置。

对于 RoPE 模型，bank key 必须保存为 **canonical pre-RoPE key**。推理时使用当前 query
的反旋转表示参与 score；默认 relative phase \(\delta=0\)，从而避免 bank key 绑定到它在
离线构造文本中曾处的绝对位置。若直接把已旋转 key 当可复用 bank 保存，换一个长度或位置
的上下文可能使 attention score 不稳定。

### 5.4 本仓库当前 side-KV 契约

E0 已将每条审核 payload 编译为 layer-24 canonical pre-RoPE K/V，并验证 GQA head 映射、共享
RoPE phase、native cache prefix/length 与非零 attention mass。memory slot 是一个 payload token
在 layer 24 的一组 K/V 向量；memory slot 永远不作为真实 token 写入 HuggingFace cache。

E1-v1 只在 completion trigger token 上让 memory 可见一次。该机制能改变后续 completion，但
matched 与 shuffled 的任务结果及首步 KL 几乎相同，因此后续 E1-C 改为 prompt-end persistent
side path：从最后一个 prompt token 到 EOS，每一步 query 都 joint-attend native KV 和同一份静态
memory KV。为抵消多 slot bank 自然获得更大总 softmax 质量，固定使用
`memory_scores -= log(valid_slot_count)`；该归一化不是可调 alpha。

## 6. 当前决策：关闭 residual vector，分阶段验证 MI-style side-KV

全局 `recovery - persistence` residual vector 已在独立确认中无效；同题 raw-state local action 的
检索 margin 也不足。该结论关闭了继续搜索 alpha、layer、符号和样本量的路线，不构成阻止文本经验
或 side-KV 的理由，因为 vector 与结构化 payload 的容量、路由和交互机制不同。

当前依次回答四个可归因问题：

| 阶段 | 通道 | 要隔离的问题 |
|---|---|---|
| E1-A | 固定多经验文本目录 | Phase 1 经验集合是否有可利用信息？ |
| E1-B | completion-aware BM25 + 单条文本 | 检索器是否选出比 shuffled 更有用的经验？ |
| E1-C | 复用 E1-B IDs 的 persistent side-KV | latent K/V 通道是否保留文本经验的作用？ |
| E1-D | entropy+risk gate | 在内容、检索和通道成立后，何时开始可见更好？ |

因此，FlashMem 风格 gate 暂时从 A/B/C 移除，避免把失败归因混在一起；只有前三阶段通过后才恢复。
具体冻结协议见 [E1 分阶段设计](e1_experience_memory_design.md)。

## 7. 术语防混淆

- **冻结 reasoner**：其基础权重不更新；不等于整个系统无训练。
- **training-free steering**：不会更新模型/latent 参数；可仍有离线 forward、rollout、
  calibration 和人工/教师生成文本的成本。
- **teacher-constructed bank**：强模型只在离线阶段帮助归纳经验，不能访问 test，也不能
  用自己的 hidden states 直接干预 student。
- **entropy gate**：候选触发器，不是 correctness classifier。
- **vector artifact**：SEAL-style、全局或簇级的 residual direction；不是完整的可检索
  text/KV memory。
- **KV-bank artifact**：MI-style 的 pre-RoPE keys/values；不是在 token sequence 中插入
  的 latent token。
