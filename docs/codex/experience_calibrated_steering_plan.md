# Experience-Calibrated Entropy-Gated Memory：研究计划与验收标准

## 1. 当前目标

本项目要验证一个训练无关、可审计的经验记忆机制：当冻结 reasoner 在 reasoning boundary
出现高熵、且当前 hidden state 显示有较高的局部持续发散风险时，系统检索一条与当前题目和
partial CoT 语义匹配的 Phase 1 经验，并以 side-KV memory 的方式供模型 attention 使用。

核心问题是：

> 相比不提供经验或提供同预算的错配经验，匹配的 target/reference 经验内容能否被模型利用，
> 并改善 GSM8K 的任务结果而不损害格式？

高熵和风险分数只用于判断**何时值得请求经验**。它们不代表模型一定出错；下一 boundary 的
熵变化只是诊断信号，低熵既不是正确性的定义，也不是干预成功的必要条件。

## 2. 当前系统契约

```text
bank-source rollout ── verifier ──► verified success / failure
                                      │
                       Flash Teacher + Pro review ──► ai_approved experience abstraction
                                                         │
                              payload sanitization ──► MemoryRecord + retrieval index + side-KV

online prompt + partial CoT ──► reasoning boundary ──► entropy-risk trigger
                                                          │
                                            semantic retrieval of one MemoryRecord
                                                          │
                                                side-KV attention → frozen reasoner
```

接口职责固定如下：

- `EntropyRiskTrigger`：以 sink-masked entropy 和当前 hidden-risk score 决定是否触发；不决定
  memory 内容。
- `SemanticMemoryRetriever`：以问题和固定窗口的 partial CoT 查询经验索引；不能用 raw hidden
  state 的 nearest neighbour 代替语义检索。
- `MemoryRecordCompiler`：将审核后的抽象文本编译为可审计 payload 与 canonical pre-RoPE side KV。
- `SideKVIntegrator`：在固定的 layer 使当前 query 可 attention 到 memory KV；不得覆盖、重建或
  改写原 prompt/history KV cache。

当前首版固定在 layer 24，在线每条样本只允许在**首次**满足 trigger 的 reasoning boundary
附加一条 memory。layer 在 E0 前固定；E1 的容量和检索器在读取 E0 的长度/系统成本诊断后冻结，
此后不得用 E1 或 final-test accuracy 调整。

## 3. 数据、质量与反泄漏约束

| 数据 | 用途 | 不允许用途 |
|---|---|---|
| `bank-source` | rollout、verifier、Teacher/Pro 审核、MemoryRecord 与风险原型 | 线上结果选择、final-test |
| `calibration-val` | cache 可行性、E1 容量与固定检索配置 | 写入 bank、final-test |
| 预注册 evaluation split | E1–E3 的开发性因果实验 | 改写 bank 或 payload 定义 |
| `final-test` | 一次性确认 | 任意设计或配置选择 |

1. 正式 memory pool 只使用 `ai_approved + answer_correctness` 记录；`teacher_inferred`、rejected、
   deferred、quarantined 记录都不能进入 index。
2. `verified_success` / `verified_failure` 只能由确定性 verifier/reward 定义；Teacher 或 Pro 不能
   覆盖该标签。
3. 后续阶段不新增 Teacher、Pro 或人工归因。它们在 Phase 1 的作用是离线抽象与质量审查，
   而不是在线教 student 解当前题。
4. 每条 MemoryRecord 必须保存 source `experience_id`、split、模型/rollout revision、Pro 路由和
   payload hash，供离线审计；这些 provenance 不进入模型输入。
5. runtime payload 严禁包含原题、完整 target/reference trajectory、最终答案、`\\boxed{}`、原始
   evidence quote，或可以复原实例答案的数值 literal。

## 4. MemoryRecord 与风险 artifact

每条可用经验首先保留 target/reference 的审核语义：target 是成功的策略或验证动作；reference
是失败机制、警告信号或不应复用的模式。在线 payload 是经过清理的固定对比形式：

```text
When facing: <reviewed situation / applicability>
Prefer: <target success decision or verification strategy>
Avoid: <reference failure mechanism or warning signal>
```

每个编译 artifact 至少包含：

```text
memory_id, source_experience_id, experience_type, approved_route,
sanitized_retrieval_key, sanitized_contrast_payload, payload_hash, token_count,
kv_layer, canonical_pre_rope_kv, reasoner_revision, tokenizer_revision, compiler_config
```

风险 trigger 复用已验证的 layer-24 原型：

\[
R(h_t)=\cos(h_t,\mu_{\mathrm{persistence}})-\cos(h_t,\mu_{\mathrm{recovery}}).
\]

`recovery` / `persistence` 仅由 bank-train 中高熵 boundary 之后的自然熵转移定义，原型按
`experience_id` 切分后在 held-out bank 验证。在线只读取当前 state；`R(h_t)>0` 表示更像局部
持续风险，而非“确定错误”或“需要施加 residual vector”。

## 5. 已确定的研究证据

这些结果只决定后续路线，不再作为可调参数：

- 风险识别成立：`R(h)` 在按 `experience_id` held-out 的 bank 上得到 ROC-AUC `0.8053`、
  zero-threshold balanced accuracy `0.7158`。它说明高熵时的 hidden state 含有可泛化的局部持续
  风险信息。
- 局部熵转移不等于最终任务结果：成功 target 和失败 reference 中都同时有 recovery 与
  persistence 事件。因此不能把“降熵”写成记忆机制的目标或成功判据。
- 全局 `μ_recovery − μ_persistence` residual action 在独立、按 sample ID 配对的确认实验中未优于
  entropy-only、随机或反向对照，且格式表现低于 vanilla。该分支已关闭。
- 同题 reference-persistence → target-recovery 的 raw-state local action 仅有 23 个 bank-train 和
  9 个 held-out 候选。虽然 action directions 不共线（effective rank `13.75/23`），但 held-out
  top-1/top-2 routing margin 的中位数仅 `0.0059`，不足以稳定选择具体 action。该分支已关闭。
- E0 已正式通过：192 条合格来源中 161 条形成完整 MemoryRecord；8/8 answer-blind runtime cases
  通过，平均 memory attention mass `0.3331`、平均 first-step logits KL `0.00507`，最大 canonical
  RoPE score 相对误差 `0.00109`。native cache prefix/length 与 disabled-path parity 均通过；E0
  未读取任务正确率。

结论：风险 gate 可以作为访问经验的时机；它不能直接提供一个通用 residual 纠偏方向，也不能
承担具体经验检索键的职责。

## 6. 后续实验

### E0：payload 与 side-KV 可行性审计

**问题**：Phase 1 的审核抽象能否被安全地转换为可用 memory，而不是答案泄漏或无效 cache？

**固定输入**：全部 `ai_approved + answer_correctness` records；不受历史 local-action 候选数量
限制，不调用新的 AI。

**工作与输出**：

1. 从 Pro 支持的 `situation/applicability`、target 策略/验证、reference 机制/信号构造 payload；
2. 逐条拒绝显式答案容器、实例 literal、完整原文复制和重复；局部 overlap 仅作诊断；
3. 记录完整 payload 的 token-length 分布，只以模型真实序列上限作为 E0 技术边界；
4. 产生 versioned `MemoryRecord` JSONL、payload audit report、检索 key index 和 layer-24 side-KV bank；
5. 单测 KV shape、canonical RoPE、cache 保持性；在运行时 trace 中证明实际附加的 memory 有非零
   attention mass。

**通过条件**：payload audit 无泄漏；所有 KV/cache 单测通过；每次 side-KV 应用均能记录 memory
attention mass。E0 不得以任务正确率宣称效果。

### E1：匹配经验内容是否具有因果作用

**问题**：模型的变化来自匹配的经验内容，还是任意额外文本/KV 扰动？

**冻结前置项**：根据 E0 的完整 payload 长度、KV footprint 和系统成本冻结 E1 全局容量；在
`calibration-val` 上冻结一个透明的检索 baseline（首版为 BM25 over sanitized retrieval keys）、
top-1、layer 24、首次触发和每样本最多一次。冻结文本
embedding retriever 可在独立后续 ablation 中比较，但不以 E1 或 final-test accuracy 选择。

先运行 `gate-observation-only` prepass，为每题生成不可变 assignment manifest：首次候选 boundary、
entropy、risk score、trigger、retrieval query hash、matched memory id/score、token budget 与 abstain
reason。所有条件都复用此 manifest。

| 条件 | 处理 | 需要回答的问题 |
|---|---|---|
| `vanilla` | 无 gate、无 memory | 原始模型会怎样？ |
| `gate-observation-only` | 相同 gate，但不附加 memory | gate 本身是否改变生成？必须与 vanilla completion 一致。 |
| `matched-memory` | 在冻结 boundary 附加检索到的 top-1 side KV | 匹配经验是否有用？ |
| `shuffled-memory` | 对 triggered sample 的 matched ids 作确定性 derangement | 是否只是任意同预算 memory 都有效？ |

`matched` 和 `shuffled` 必须使用同一题、同一 prefix、同一 trigger 集合、相同 side-KV layer、
相同 memory 条数与 token budget；只允许 memory id 不同。记录 deterministic shuffle seed。

**判定顺序**：

1. 检查 memory attention mass、retrieval score 与 logits KL，确认 payload 真正进入计算；
2. 以 sample ID 配对、bootstrap CI 比较 `matched-memory` 对 `shuffled-memory` 和
   `gate-observation-only` 的 GSM8K accuracy；格式准确率不得低于 vanilla；
3. 报告下一 boundary 熵、生成长度、延迟和失败案例作为诊断，而非进入通过条件。

若 E1 失败，结论只限于当前 payload / BM25 retrieval / side-KV integration 的组合无效；不能把
失败归因于整个 Phase 1 bank 或风险 trigger。

### E2：target/reference 字段的贡献（仅 E1 通过后）

在相同 manifest、retrieval id、token budget 和 side-KV 接口下比较：

1. `target-only`：`Prefer`；
2. `reference-only`：`Avoid`；
3. `contrast`：`When facing + Prefer + Avoid`。

该实验回答失败经验是否为成功策略提供额外 guardrail；它不重新测试 residual vector。

### E3：风险触发时机的贡献（仅 E1 通过后）

同一条 matched memory 分别在风险 gate 的首次触发 boundary 与同生成内的确定性随机 delimiter
附加。两组的 memory、预算和生成条件相同。该实验回答高熵+risk 是否提供了有价值的访问时机。

### Final：一次性确认

只有 E1 及必要 E2/E3 通过后，才冻结 bank、payload compiler、retriever、layer、token budget、
manifest 规则和报告脚本，并在未参与任何选择的 `final-test` 运行一次。正式报告包括 accuracy、
format accuracy、trigger rate、retrieval score、memory attention、latency、KV footprint、置信区间和
失败案例；不重跑已关闭的 residual-vector 分支。

## 7. 运行记录与实验纪律

每个 candidate boundary 至少记录：

```text
sample_id, split, generated_token_index, boundary_type,
entropy_with_sink, entropy_threshold, hidden_risk_score, gate_triggered,
trigger_reason, retrieval_query_hash, retrieval_method, retrieved_memory_id,
retrieval_score, memory_condition, memory_payload_hash, memory_token_count,
memory_kv_layer, memory_attention_mass, logits_kl_baseline_to_memory,
side_kv_applied, generation_length, final_reward, format_reward, output_path
```

`shuffled-memory` 额外记录 `matched_memory_id`、`assigned_memory_id` 与 shuffle seed。每个实验使用
版本化 split manifest、artifact hash、独立输出目录和一键服务器脚本。代码、配置、脚本和最小校验
完成后必须一起 commit 并 push；`.server.env` 只保存服务器本地路径/API key，不进入 Git。

## 8. 当前交接点

E0 已完成并冻结。当前执行 E1：先在 `dev-test` 运行 answer-blind observation-only prepass，冻结
BM25 top-1 与确定性 shuffled assignment，再用同一 manifest 运行四条件配对评测。具体 artifact、
对照和服务器入口见 [E1 设计](e1_experience_memory_design.md)。

在 E1 得到 matched-memory 的独立因果证据前，不扩展 memory 数量、不搜索 layer/注入强度，
不进入 E2/E3 或 final-test。
