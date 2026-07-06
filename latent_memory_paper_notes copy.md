
| 方法 | latent 的来源 | latent 放在哪里 | 如何与原推理结合 | RoPE / 位置重点 |
|---|---|---|---|---|
| MemGen | Weaver 根据上下文生成 latent memory embeddings | token embedding sequence / reasoning stream | 像额外连续 token 一样拼入输入 embedding，参与后续 self-attention | latent token 会占序列位置，后续 token position 会被推后 |
| One Token per Multimodal Evidence | compressor 将每条 evidence 压成一个 latent token | generator input embedding / prompt latent token | 查询检索 latent token 后投影到 LLM/VLM hidden dim，作为证据 token 被 generator attend | 作为输入 token 使用，需要正常 position id；插入位置会影响相对位置 |
| IndexMem | 被驱逐 KV 通过 fast-weight memory 压缩 | attention output residual path | 不进入 softmax；从 memory 读出 `m(q)`，加到 retained-KV attention output | latent residual 本身不需要 RoPE；保留 KV 不能重新编号 |
| Memory Inception | guidance text 通过 frozen LLM 编码成 latent KV bank | selected attention sites 的 side KV bank | selected layers/heads/KV groups 直接 attend 到 hidden KV slots | 必须存 canonical pre-RoPE key，避免 bank 绑定构造时绝对位置 |


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
