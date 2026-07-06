# Latent Memory 相关论文整理

本文整理前面讨论中提到的论文，重点说明每篇论文与 “Latent Memory 如何参与 LLM 推理” 的关系。这里的 “Latent Memory” 采用较宽泛定义：非原始文本 token 的记忆信号，以 embedding、K/V、cache、cross-attention、概率分布或 recurrent state 等形式影响推理过程。

## 一、建议阅读顺序

如果目标是快速建立整体图景，建议按以下顺序阅读：

1. **Prefix-Tuning**：理解 continuous / virtual tokens 如何被模型读取。
2. **Recurrent Memory Transformer**：理解 memory tokens 如何跨 segment 传递。
3. **Compressive Transformer**：理解历史记忆压缩与长上下文建模。
4. **RETRO**：理解外部 memory bank 如何通过 cross-attention 进入生成。
5. **kNN-LM**：理解 memory 如何在输出概率层影响下一个 token。
6. **LoMA / IndexMem / Memory Inception**：理解最新 KV-cache 与 latent memory 方向。
7. **One Token per Multimodal Evidence**：理解近期明确以 “Latent Memory” 命名的 QA 范式。

## 二、核心基础论文

### 1. Prefix-Tuning: Optimizing Continuous Prompts for Generation

- 年份：2021
- 作者：Xiang Lisa Li, Percy Liang
- 链接：https://arxiv.org/abs/2101.00190
- 关键词：continuous prompt, virtual tokens, prefix, parameter-efficient tuning

**研究问题**

如何在冻结预训练语言模型参数的情况下，通过很少的可训练参数适配下游生成任务。

**核心方法**

学习一组连续前缀向量，让后续 token 像 attend 到普通 prompt token 一样 attend 到这些 prefix。论文将其描述为类似 “virtual tokens” 的连续提示。

**与 Latent Memory 的关系**

它是 embedding / K-V prefix 类 latent memory 的重要基础。虽然论文目标是参数高效微调，不是长期记忆，但它证明了连续向量可以作为可被 Transformer 注意力读取的“非文本 token”。

**对应接口**

```text
token embeddings + continuous prefix
或
layer-wise K/V prefix
```

**阅读重点**

- prefix 如何作为 virtual tokens 被 attention 读取。
- prefix 与 prompt token 的区别。
- 这种方法为何适合 frozen LM。

## 三、跨段与长上下文记忆

### 2. Compressive Transformers for Long-Range Sequence Modelling

- 年份：2019
- 作者：Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Timothy P. Lillicrap
- 链接：https://arxiv.org/abs/1911.05507
- 关键词：compressed memory, long-range dependency, memory compression

**研究问题**

Transformer 处理长序列时，历史信息保存成本高，如何压缩过去 memory 以建模长距离依赖。

**核心方法**

在普通 memory 之外引入 compressed memory，把较早的 hidden states 压缩后继续提供给后续 attention 使用。

**与 Latent Memory 的关系**

这是 “历史信息压缩成 latent memory” 的早期代表。它不是专门面向现代 LLM 推理 KV cache，但思想上直接支撑 “evicted history -> compressed latent memory -> later attention”。

**对应接口**

```text
past memories -> compression -> compressed memory
later tokens attend to compressed memory
```

**阅读重点**

- memory 与 compressed memory 的层级关系。
- 压缩损失如何训练。
- 与 Transformer-XL 类 recurrent memory 的区别。

### 3. Recurrent Memory Transformer

- 年份：2022
- 作者：Aydar Bulatov, Yuri Kuratov, Mikhail S. Burtsev
- 链接：https://arxiv.org/abs/2207.06881
- 关键词：memory tokens, segment-level recurrence, long-term dependency

**研究问题**

如何让 Transformer 在不大改模型结构的情况下处理超过上下文窗口的长序列。

**核心方法**

在输入或输出序列中加入 special memory tokens，让这些 tokens 在 segment 之间递归传递信息。

**与 Latent Memory 的关系**

它是 “memory token 作为跨段 latent state” 的直接代表。memory token 不是普通自然语言内容，而是模型学习得到的跨段信息载体。

**对应接口**

```text
chunk_i + memory_{i-1} -> chunk_i output + memory_i
```

**阅读重点**

- memory tokens 如何插入输入/输出。
- segment 之间如何传递 memory。
- 这种方式如何避免完整保留长上下文。

### 4. Associative Recurrent Memory Transformer

- 年份：2024
- 作者：Ivan Rodkin, Yuri Kuratov, Aydar Bulatov, Mikhail Burtsev
- 链接：https://arxiv.org/abs/2407.04841
- 关键词：associative retrieval, recurrent memory, very long context

**研究问题**

如何在非常长的序列上，用近似常数时间处理新信息，同时保留可检索的任务相关信息。

**核心方法**

结合局部 self-attention 与 segment-level recurrent memory，在长上下文任务中实现 associative retrieval。

**与 Latent Memory 的关系**

它延续 RMT 思路，重点是超长上下文下的 recurrent memory 与关联检索能力。

**对应接口**

```text
local context attention + recurrent memory state
```

**阅读重点**

- 与 RMT 的改进关系。
- associative retrieval 是如何通过 memory 实现的。
- BABILong 等超长上下文任务设置。

## 四、检索式外部记忆与 cross-attention

### 5. Improving language models by retrieving from trillions of tokens / RETRO

- 年份：2021
- 作者：Sebastian Borgeaud et al.
- 链接：https://arxiv.org/abs/2112.04426
- 关键词：retrieval-enhanced Transformer, external memory, chunked cross-attention

**研究问题**

如何让语言模型利用超大规模外部语料库，而不是只依赖参数记忆。

**核心方法**

根据当前上下文从大规模数据库检索相关 chunks，再通过 chunked cross-attention 让模型读取检索内容。

**与 Latent Memory 的关系**

RETRO 的 memory 多数仍来源于文本 chunk，不一定是压缩 latent token；但它明确展示了 “外部 memory bank 通过 cross-attention 参与推理” 这一接口。

**对应接口**

```text
current hidden states -> cross-attention -> retrieved memory chunks
```

**阅读重点**

- retrieval 与 generation 如何耦合。
- chunked cross-attention 的插入位置。
- 为什么这种方式能减少对模型参数规模的依赖。

## 五、输出层 memory fusion

### 6. Generalization through Memorization: Nearest Neighbor Language Models

- 年份：2019
- 作者：Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, Mike Lewis
- 链接：https://arxiv.org/abs/1911.00172
- 关键词：kNN-LM, datastore, probability interpolation, long-tail knowledge

**研究问题**

如何让语言模型在不继续训练的情况下，利用外部 datastore 改善预测，尤其是长尾事实和稀有模式。

**核心方法**

用当前上下文的 hidden representation 在 datastore 中做 kNN 检索，将近邻对应的下一个 token 构成 kNN 分布，再与原 LM 分布线性插值。

**与 Latent Memory 的关系**

这是输出层 memory fusion 的经典事实依据。memory 没有进入 Transformer 中间层，而是在最后概率分布层影响 next-token prediction。

**对应接口**

```text
p_final = lambda * p_knn + (1 - lambda) * p_lm
```

**阅读重点**

- datastore key/value 如何构建。
- kNN 分布如何转成 token probability。
- 为什么它对 rare patterns 和 domain adaptation 有帮助。

### 7. Memorizing Transformers

- 年份：2022
- 作者：Yuhuai Wu, Markus N. Rabe, DeLesley Hutchins, Christian Szegedy
- 链接：https://arxiv.org/abs/2203.08913
- 关键词：non-differentiable memory, kNN lookup, test-time memorization

**研究问题**

语言模型能否在推理时直接记住新数据，而不需要更新模型权重。

**核心方法**

扩展 Transformer，使其可以记忆过去输入的内部 representations，并通过 approximate kNN 从非可微 memory 中读取 key/value pairs。

**与 Latent Memory 的关系**

它位于 “内部表示 memory + kNN lookup” 方向，比普通文本 RAG 更接近 latent-space memory。它也证明了推理时写入/读取 memory 对代码、数学、定理等任务有价值。

**对应接口**

```text
internal representations -> external memory
current representation -> kNN lookup -> memory value
```

**阅读重点**

- memory 中保存的 key/value 是什么。
- approximate kNN 如何接入 Transformer。
- test-time memorization 与参数更新的区别。

### 8. Get To The Point: Summarization with Pointer-Generator Networks

- 年份：2017
- 作者：Abigail See, Peter J. Liu, Christopher D. Manning
- 链接：https://arxiv.org/abs/1704.04368
- 关键词：pointer-generator, copy mechanism, coverage

**研究问题**

生成式摘要容易事实错误和重复，如何让模型既能生成新词，又能从源文本准确复制信息。

**核心方法**

同时产生 vocabulary distribution 和 copy distribution，再通过门控混合。

**与 Latent Memory 的关系**

它不是 Latent Memory 论文，但它是输出分布融合的经典事实依据：外部源内容可以在最后 token distribution 层直接影响生成。

**对应接口**

```text
p_final = p_gen * p_vocab + (1 - p_gen) * p_copy
```

**阅读重点**

- copy distribution 如何从 attention distribution 得到。
- p_gen 门控如何控制生成与复制。
- 这种 late fusion 为什么适合事实实体保真。

## 六、KV-cache 与推理系统方向

### 9. LoMA: Lossless Compressed Memory Attention

- 年份：2024
- 作者：Yumeng Wang, Zhenyang Xiao
- 链接：https://arxiv.org/abs/2401.09486
- 关键词：KV cache compression, compressed memory attention, autoregressive generation

**研究问题**

LLM 长上下文推理中 KV cache 显存和计算成本很高，如何压缩 KV cache 同时尽量保留生成能力。

**核心方法**

引入 compressed memory attention，并配合专门训练或微调流程，使模型具备压缩上下文下的自回归生成能力。

**与 Latent Memory 的关系**

这是 KV-cache 压缩方向的重要论文。它把 memory compression 与推理时 attention 结合起来，支撑 “compressed latent memory 替代完整 KV” 的路线。

**对应接口**

```text
KV cache -> compressed memory -> autoregressive attention
```

**阅读重点**

- compressed memory attention 的具体形式。
- 压缩比例与生成质量的关系。
- 它和简单 KV eviction 的区别。

### 10. Memory Inception: Latent-Space KV Cache Manipulation for Steering LLMs

- 年份：2026
- 作者：Andy Zeyi Liu, Michael Zhang, Ilana Greenberg, Adam Alnasser, Lucas Baker, John Sous
- 链接：https://arxiv.org/abs/2605.06225
- 关键词：KV cache manipulation, latent attention space, steering, persistent guidance

**研究问题**

长期提示或行为指导如果一直放在 visible prompt 中，会增加 KV cache 开销并污染上下文；如何在 latent attention space 中更高效地注入指导。

**核心方法**

把文本指导转成 text-derived KV banks，并只在选定层插入这些 latent slots，从而实现训练无关的 steering。

**与 Latent Memory 的关系**

这是最直接的 KV-cache latent memory / latent steering 论文之一。memory 不作为可见文本存在，而是以 K/V bank 形式影响注意力。

**对应接口**

```text
selected layers:
  attention reads normal KV + injected guidance KV bank
```

**阅读重点**

- text-derived KV banks 如何生成。
- 为什么只在部分层插入。
- 与 prompting、activation steering 的控制力和成本比较。

### 11. IndexMem: Learned KV-Cache Eviction with Latent Memory for Long-Context LLM Inference

- 年份：2026
- 作者：Xintong Yang et al.
- 链接：https://arxiv.org/abs/2605.25475
- 关键词：learned KV eviction, latent memory module, long-context inference

**研究问题**

长上下文推理中，KV cache 不能无限增长；简单 eviction 会永久丢失信息，如何在有限 KV budget 下减少遗忘。

**核心方法**

使用 learnable indexer 预测 token/KV 重要性，保留关键 KV；同时把被淘汰 token 的信息压缩到轻量 latent memory module，并提供 residual readout 补偿 attention 贡献。

**与 Latent Memory 的关系**

这是 “KV eviction + latent memory compensation” 的直接代表。它非常贴近推理系统中的 memory 管理问题。

**对应接口**

```text
evicted KV -> latent memory module
current attention -> residual readout from latent memory
```

**阅读重点**

- learned indexer 如何判断 KV 重要性。
- latent memory module 如何在线更新。
- residual readout 如何补偿被 evict 的信息。
- RULER、Needle-in-a-Haystack、LongBench 上的评估方式。

## 七、明确命名为 Latent Memory 的近期 QA 论文

### 12. One Token per Multimodal Evidence: Latent Memory for Resource-Constrained QA

- 年份：2026
- 作者：Zhi Zheng, Ziqiao Meng, Hao Luan, Wei Liu, Wee Sun Lee
- 链接：https://arxiv.org/abs/2606.10572
- 关键词：Latent Memory, multimodal QA, evidence compression, latent token

**研究问题**

传统 RAG 将原始文本或图像证据传给生成模型，token 与存储成本高；如何在资源受限场景下用更紧凑的证据表示完成 QA。

**核心方法**

用小型 compressor LLM/VLM 将每条文本或图像证据压缩成一个高维 latent token。检索时不返回原始证据，而是在统一 latent space 中检索相关 latent tokens，并把这些 tokens 直接提示给 LLM/VLM 生成答案。

**与 Latent Memory 的关系**

这是本文讨论中最“正名”的 Latent Memory 论文之一。它明确把 memory item 从 raw text/image 改成 latent token，并训练 compressor 兼顾 reconstruction、contrastive retrieval 和 distillation generation。

**对应接口**

```text
raw evidence -> compressor -> one latent token
query -> latent retrieval -> retrieved latent tokens -> generator
```

**阅读重点**

- 单个 evidence 如何压成一个 latent token。
- reconstruction、contrastive、distillation 三种目标各自作用。
- 与 RAG baseline 的 token 消耗和 QA 效果对比。

## 八、按技术路线归类

| 技术路线 | 代表论文 | 与 LLM 推理的交互位置 | 是否直接属于 Latent Memory |
|---|---|---|---|
| continuous / virtual tokens | Prefix-Tuning | embedding 或 K/V prefix | 间接相关 |
| segment recurrent memory | RMT, ARMT | 跨 segment memory tokens/state | 强相关 |
| compressed history | Compressive Transformer | compressed memory attention | 强相关 |
| retrieval cross-attention | RETRO | cross-attention memory bank | 相关，但多为文本检索 |
| output distribution fusion | kNN-LM, Pointer-generator | logits/probability 层 | 相关，但不一定叫 latent memory |
| internal representation memory | Memorizing Transformers | kNN lookup over hidden states | 强相关 |
| KV-cache compression | LoMA | compressed KV / memory attention | 强相关 |
| KV-cache latent steering | Memory Inception | injected KV banks | 直接相关 |
| KV eviction compensation | IndexMem | evicted KV -> latent memory | 直接相关 |
| evidence latent token | One Token per Multimodal Evidence | latent token retrieval + generation | 直接相关 |

## 九、与前述问题的对应关系

### “Latent memory 和 token embedding 直接 cat” 对应哪些论文？

- Prefix-Tuning
- Recurrent Memory Transformer
- One Token per Multimodal Evidence

### “Latent memory 是否可以作为 K/V 或 cache 进入推理？”

- Prefix-Tuning
- Memory Inception
- IndexMem
- LoMA

### “Latent memory 是否可以与最后 logits 相加？”

严格说，已有事实更多是输出概率层融合，而不是明确的 latent memory logits addition：

- kNN-LM：`p_final = lambda * p_knn + (1 - lambda) * p_lm`
- Pointer-generator：`p_final = p_gen * p_vocab + (1 - p_gen) * p_copy`

因此，`z_final = z_lm + W_m m_latent` 目前仍建议写成合理设计猜想，除非找到直接采用该形式的论文。

### “哪条路线最值得继续跟？”

如果目标是大模型推理系统，优先看：

1. Memory Inception
2. IndexMem
3. LoMA
4. Compressive Transformer

如果目标是 QA/RAG 压缩，优先看：

1. One Token per Multimodal Evidence
2. RETRO
3. kNN-LM

如果目标是可实现的小实验，优先看：

1. Prefix-Tuning
2. kNN-LM
3. Pointer-generator / copy fusion

## 十、近期精读论文的方法流程与接入点对比

本节整理近期重点讨论的几篇论文，关注三个问题：

1. latent memory 从哪里来。
2. latent memory 如何与原本 LLM/VLM 推理过程结合。
3. 如果涉及 token/KV/attention 路径，RoPE 或位置编码需要注意什么。

### 10.1 总览表

| 方法 | latent 的来源 | latent 放在哪里 | 如何与原推理结合 | RoPE / 位置重点 |
|---|---|---|---|---|
| MemGen | Weaver 根据上下文生成 latent memory embeddings | token embedding sequence / reasoning stream | 像额外连续 token 一样拼入输入 embedding，参与后续 self-attention | latent token 会占序列位置，后续 token position 会被推后 |
| One Token per Multimodal Evidence | compressor 将每条 evidence 压成一个 latent token | generator input embedding / prompt latent token | 查询检索 latent token 后投影到 LLM/VLM hidden dim，作为证据 token 被 generator attend | 作为输入 token 使用，需要正常 position id；插入位置会影响相对位置 |
| IndexMem | 被驱逐 KV 通过 fast-weight memory 压缩 | attention output residual path | 不进入 softmax；从 memory 读出 `m(q)`，加到 retained-KV attention output | latent residual 本身不需要 RoPE；保留 KV 不能重新编号 |
| Memory Inception | guidance text 通过 frozen LLM 编码成 latent KV bank | selected attention sites 的 side KV bank | selected layers/heads/KV groups 直接 attend 到 hidden KV slots | 必须存 canonical pre-RoPE key，避免 bank 绑定构造时绝对位置 |
| MLA | hidden state 压缩成低维 latent KV 表示 | KV cache representation | 缓存低维 latent，再还原 K/V 做正常 attention | content latent 与 RoPE positional part 要分开处理 |

可以按 latent memory 的接入位置来理解：

```text
输入 embedding 层:
    MemGen
    One Token per Multimodal Evidence

attention softmax 内部:
    Memory Inception

attention output 之后:
    IndexMem

KV representation 层:
    MLA
```

### 10.2 MemGen：latent memory 作为 reasoning stream token

**方法流程**

MemGen 的 latent memory 最接近“额外思维 token”。它在模型推理过程中动态插入 latent memory embeddings：

```text
原始 prompt / reasoning tokens
        ↓
检测 augmentation point
        ↓
Memory Weaver 根据上下文生成 latent memory embeddings
        ↓
latent embeddings 投影回 reasoner embedding space
        ↓
拼进原 token embedding sequence
        ↓
frozen reasoner 继续 self-attention / generation
```

核心结构包括：

- **Memory Weaver**：生成 latent memory。
- **Memory Trigger**：决定何时插入 latent memory。
- **Frozen reasoner**：不更新参数，消费插入后的 embedding sequence。

**与原推理过程的结合点**

MemGen 的结合点是 **input embedding sequence**：

```text
[token embeddings]
    ->
[token embeddings; latent memory embeddings]
```

因此 latent memory 会像普通 token 一样进入 Transformer 层，参与后续 self-attention。它不是 KV cache 压缩，也不是 attention output residual，而是把 latent memory 直接放入 reasoning stream。

**关键细节**

- Prompt augmentation 与 inference augmentation 使用不同 latent 参数。
- Latent tokens 插入 delimiter 边界，例如逗号、句号、换行或 prompt-label boundary。
- 训练时只在真实 token 上算 loss，latent positions 通过 mask 排除。
- Weaver/Trigger 是 LoRA-tuned copies，reasoner 冻结。

**RoPE / 位置注意点**

因为 latent memory 被当作序列 token 插入，所以它会占据 position。

例如原序列：

```text
A B C D
```

插入 latent 后变成：

```text
A B [LATENT] C D
```

则 `C`、`D` 的 position 会被推后。RoPE 会把 latent token 当作正常序列位置处理。这里没有绕开 RoPE，latent token 的插入会自然改变后续 token 的相对位置结构。

### 10.3 One Token per Multimodal Evidence：latent token RAG

**方法流程**

这篇论文更像 latent RAG。它把每条文本或图像 evidence 压成一个 latent token，查询时检索 latent token，而不是检索原始文本/图像 evidence。

```text
每条 evidence x_i
        ↓
compressor / composer 编码
        ↓
得到 latent token z_i
        ↓
存入 latent memory index
        ↓
查询 Q 也投影到同一检索空间
        ↓
相似度检索相关 latent tokens
        ↓
retrieved z_i 投影到 frozen LLM/VLM hidden dim
        ↓
作为 prompt latent token 参与答案生成
```

核心压缩可以写为：

```text
z_i = compressor(x_i, [MEM])
```

查询阶段：

```text
q = composer(Q)
TopK(sim(q, z_i))
```

然后将检索出的 latent evidence 投影到 generator hidden space：

```text
z_hat_i = W_g z_i
```

**与原推理过程的结合点**

结合点仍然是 **input embedding / prompt latent token**。

Retrieved latent evidence 不是以原始文本 token 放进 prompt，而是以连续 embedding 的形式输入 frozen generator。它会被 generator 后续 self-attention 消费。

**关键细节**

- 每个 evidence 被压成一个 latent token，极大降低 token budget。
- 训练通常包含 reconstruction、contrastive retrieval、distillation generation 等目标。
- 一个 latent token 同时承担检索表示和生成条件表示。
- 和传统 RAG 最大差异是：系统存储与传递的是 latent token，而不是 raw evidence chunk。

**RoPE / 位置注意点**

因为 retrieved latent tokens 作为 generator input token 使用，所以需要 position id。它们的插入位置会影响后续 token 的 RoPE 相对位置。

这篇论文重点不在 RoPE 设计，因此更接近普通 soft prompt / prefix token 的处理：latent evidence token 被当作 prompt 中的一部分进入 Transformer。

### 10.4 IndexMem：KV eviction 后的 latent residual compensation

**方法流程**

IndexMem 处理的是长上下文推理中的 KV cache memory wall。它不把 latent memory 当 token，而是在 KV eviction 后用 latent memory 补偿被删除信息。

```text
完整上下文 KV cache
        ↓
learnable indexer 为 token importance 打分
        ↓
保留重要 KV token
        ↓
驱逐低分 KV token
        ↓
被驱逐 KV 写入 fixed-size latent memory
        ↓
当前 query 从 latent memory 读出补偿向量
        ↓
补到 retained-KV attention output
```

Indexer 输出 query-to-key score matrix：

```text
A = Indexer(X, Q)
```

并用 max aggregation 得到 token importance：

```text
imp_t = max_s A_{s,t}
```

Latent memory 是 fast-weight state：

```text
M in R^{d_mem x d_model}
b in R^{d_mem}
```

写入被驱逐 KV：

```text
M <- lambda M + eta sum_i phi(k_i) v_i^T
b <- lambda b + eta sum_i phi(k_i) * phi(k_i)
```

读取：

```text
m(q) = phi(q)^T M / (phi(q)^2^T b + epsilon)
```

**与原推理过程的结合点**

结合点是 **attention output 之后的 residual addition**：

```text
q 和 K_kept 算 attention score
        ↓
softmax
        ↓
attention weights × V_kept
        ↓
得到 o_attn
        ↓
加 g(q) m(q)
        ↓
得到最终 attention output
```

核心公式：

```text
o = o_attn + g(q) * m(q)
```

因此 IndexMem 不改变 attention weights，也不把 memory token 加进 softmax。它学习补偿：

```text
full attention output - compressed attention output
```

也就是被驱逐 KV 原本对 attention output 的贡献。

**关键细节**

- Indexer 是 learnable 的，不是 SnapKV/PyramidKV 那样的纯 heuristic。
- Indexer 训练时模仿 backbone full attention 的 token importance distribution。
- Latent memory 每层一个，层内所有 heads 共享。
- Memory readout 是 query-conditioned，而不是固定全局向量。
- Gate `g(q)` 控制是否以及多大程度使用 memory。

**RoPE / 位置注意点**

IndexMem 的 RoPE 注意点主要在 retained KV attention path，而不在 latent memory residual。

1. Indexer 使用 pre-RoPE query states。
2. 真实 attention 仍使用 backbone 原本的 RoPE Q/K。
3. KV 被压缩后，保留 token 不能重新编号。
4. latent memory residual 本身不需要 RoPE。

错误示例：

```text
原始保留 positions: [0, 3, 7, 9]
错误重排为:       [0, 1, 2, 3]
```

这会破坏 retained KV attention 的相对位置语义。正确做法是保留原始 position / cache_position。IndexMem 的 residual 加法发生在 `Softmax(QK^T)V` 之后，因此不需要为 latent memory 分配 RoPE position。

### 10.5 Memory Inception：hidden guidance KV bank

**方法流程**

Memory Inception 的目标是 LLM steering。它把 guidance text 编译成 latent KV bank，并只在 selected attention sites 暴露给模型。

```text
guidance text / reasoning heuristic
        ↓
套 template
        ↓
跑 frozen LLM
        ↓
抽取 guidance token hidden states
        ↓
经过原模型自己的 W_K / W_V projection
        ↓
得到 latent KV bank
        ↓
selector 选择 layers / heads / KV groups
        ↓
推理时 selected attention sites attend 到这些 KV slots
```

Bank 构造：

```text
k_{ell,u,m}^{(b)} = W_{K,u}^{(ell)} Norm_ell(h_{ell,m}^{(b)})
v_{ell,u,m}^{(b)} = W_{V,u}^{(ell)} Norm_ell(h_{ell,m}^{(b)})
```

在 selected site 上：

```text
K_t* = concat(K_x<=t, K^(1), ..., K^(B))
V_t* = concat(V_x<=t, V^(1), ..., V^(B))
o_t* = Attn(q_t, K_t*, V_t*)
```

**与原推理过程的结合点**

结合点是 **selected attention site 内部的 KV 扩展**。

普通 attention：

```text
q attend prompt/history KV
```

Memory Inception：

```text
q attend prompt/history KV + latent KV bank
```

它不同于 IndexMem：Memory Inception 的 latent memory 进入 softmax attention 竞争；IndexMem 则绕开 softmax，在 attention output 后 residual addition。

**Target bank / reference bank**

Memory Inception 可以使用多个 bank：

- **target bank**：希望模型靠近的行为、风格、策略或 heuristic。
- **reference bank**：希望模型远离的反向行为或旧行为。
- **auxiliary bank**：额外 facts 或 heuristics。
- **prompt bank**：普通 prompt/history。

Target-reference evidence gap：

```text
Delta_t =
log(1/M+ sum_m exp(s^+_{t,m}))
-
log(1/M- sum_n exp(s^-_{t,n}))
```

Bank-level evidence：

```text
beta_t^(b) =
log(1/M_b sum_m exp(s_{t,m}^{(b)})) + c_t^(b)
```

不同 bank 通过 softmax routing 竞争：

```text
pi_t = softmax([beta_t^(0), ..., beta_t^(B)])
```

**Layer/head 选择**

MI 不训练模型参数，但需要 calibration selector。它在 calibration prompts 上计算每个 layer/head 或 KV group 与 target/reference banks 的 alignment margin：

```text
a_{ell,u} =
max_j q_{ell,u}^T k^+_{ell,u,j} / sqrt(d_h)
-
max_j q_{ell,u}^T k^-_{ell,u,j} / sqrt(d_h)
```

再结合 target-bank mass 与 prompt-bank mass：

```text
U_{ell,u} = a_{ell,u} + xi m^+_{ell,u} - chi m^x_{ell,u}
```

流程：

```text
每层选 top-k heads / KV groups
        ↓
聚合 top-k units 的分数
        ↓
选 top-m layers
        ↓
冻结 selector artifact
        ↓
推理时只在这些 sites 接入 side KV bank
```

这一步不是训练，没有反向传播，不更新模型或 bank 参数。它是 offline calibration / site selection。

**Attention sink 风险**

Memory Inception 确实添加了额外可 attend 的 latent KV slots。如果控制不好，可能退化成 hidden prompt attention sink。论文通过以下机制降低风险：

- 只插入 selected layers / heads / KV groups。
- 用 target/reference contrast 做选择和 routing。
- 使用 size-normalized bank evidence，避免 slots 多的 bank 天然获胜。
- 监控 target mass、reference mass、prompt mass。

理想状态是 query-dependent memory routing，而不是所有 query 都无脑 attend memory bank。

**RoPE / 位置注意点**

Memory Inception 对 RoPE 处理最明确。问题是：memory bank 离线构造，如果直接保存带绝对位置旋转的 key，就会绑定到 bank 构造时的 position。推理时位置变化会导致 score 不稳定。

因此它保存 **canonical pre-RoPE key**。

有 RoPE 时：

```text
q_t = R_t q_bar_t
k_j^x = R_j k_bar_j^x
```

Memory key 存成：

```text
k_tilde_m
```

推理时计算：

```text
s_{t,m}^{mem}
= <R_t^{-1} q_t, R_delta_m k_tilde_m> / sqrt(d_h)
= <q_bar_t, R_delta_m k_tilde_m> / sqrt(d_h)
```

默认：

```text
delta_m = 0
```

这意味着 memory slot 是 position-agnostic reminder slot。核心原则是：

```text
不要把 memory key 绑定到构造 bank 时的绝对位置。
存 pre-RoPE canonical key。
推理时用当前 query 的反旋转形式匹配。
```

### 10.6 MLA：latent KV representation compression 背景

**方法流程**

MLA 不是 memory 方法，但它是理解 latent KV 表示的重要前置知识。普通 MHA 缓存完整 K/V：

```text
K, V in R^{B x S x H_kv x d_head}
```

MLA 不缓存完整 per-head K/V，而是缓存低维 latent：

```text
c_t^KV = W^DKV h_t
```

推理时再还原 content key/value：

```text
k_{t,i}^C = W_i^UK c_t^KV
v_{t,i}^C = W_i^UV c_t^KV
```

同时 RoPE 相关 key 部分单独处理：

```text
k_{t,i} = [k_{t,i}^C; k_t^R]
```

**与原推理过程的结合点**

MLA 的结合点是 **KV cache representation 层**。它不删除 token，也不构造外部 memory，而是让每个 token 的 KV 存储更小。模型仍然做正常 attention，只是 K/V 的缓存方式变成低维 latent + 还原。

**RoPE / 位置注意点**

MLA 的关键是：

```text
content KV 可以压缩成 latent。
RoPE positional part 不能天真压掉。
```

这也是为什么它通常区分 content key/value 与 RoPE key。这个原则与 Memory Inception 的 canonical pre-RoPE key 思路相通：可复用 latent 表示不要随便绑定绝对位置相位。

### 10.7 关键结论

这些论文都可以被宽泛地称为 latent memory，但它们与原模型结合的位置完全不同。

1. **进入输入序列的 latent memory**
   - MemGen
   - One Token per Multimodal Evidence
   - 特点：作为 token embedding / prompt latent token，参与 self-attention。
   - RoPE：需要 position id，会影响后续 token 相对位置。

2. **进入 attention softmax 的 latent memory**
   - Memory Inception
   - 特点：作为 side KV bank 被 selected attention sites attend。
   - RoPE：必须使用 canonical pre-RoPE key，避免绑定构造时绝对位置。

3. **attention output 后相加的 latent memory**
   - IndexMem
   - 特点：不进入 softmax，作为 query-conditioned residual compensation。
   - RoPE：latent residual 本身不需要 RoPE；retained KV 必须保留原始 position。

4. **压缩 KV 表示的 latent**
   - MLA
   - 特点：降低每个 token 的 KV storage，不改变 token 数量。
   - RoPE：content latent 与 positional RoPE part 需要分开处理。

最重要的分辨方式是问：

```text
latent memory 是 token？
是 KV bank？
是 attention output residual？
还是 KV representation compression？
```

对应答案不同，RoPE 和位置处理也完全不同。
