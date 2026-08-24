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

layer 24 已由 E0 固定。原先“首次满足 trigger 时单步附加一条 memory”的 E1-v1 已完成诊断但未产生
任务收益；后续不再把经验内容、检索、表示通道和 gate 时机捆绑验证。E1-A/B/C 分别冻结经验目录、
BM25 assignment 和 persistent side-KV 接口，E1-D 才恢复 gate。任何配置都不得用 `final-test`
accuracy 调整。

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

### E1：分离经验内容、检索、side-KV 与 gate

E1-v1 在 `dev-test` 前 100 条上同时测试 BM25、风险 gate 和单步 side-KV。33 条 assigned 样本中，
matched、shuffled、gate-only 均为 `6/33`；matched/shuffled 都改变了 `21/33` 条 completion，但首
token 均未改变，两者之间只有 `6/33` 条 completion 不同。memory attention mass 为正，说明机制
进入计算，但当前组合主要产生内容不敏感的轨迹扰动。该结果不用于配置选择，也不能单独否定 bank、
retriever 或 side-KV。

后续采用四级验证阶梯：

| 阶段 | 固定处理 | 主要问题 |
|---|---|---|
| `E1-A` | 无 retriever/gate；同一份 2048-token k-medoids 代表经验目录追加到所有题 | Phase 1 经验集合整体是否包含模型可利用的信息？ |
| `E1-B` | no-memory 首次完整回答只用于 `question + sanitized preanswer` query；BM25 top-1 文本重答 | 检索到的经验是否优于 shuffled 和 no-memory？ |
| `E1-C` | 原样复用 E1-B memory IDs；prompt-end persistent side-KV 与相同 ID 的文本条件比较 | side-KV 是否保留了经验内容的作用？ |
| `E1-D` | 仅在 A/B/C 通过后恢复 entropy+risk gate | 风险 gate 是否提供更好的开始时机？ |

E1-A 的 medoid 必须是真实 MemoryRecord，不使用生成式聚类总结；随机 bank 使用三个预注册种子且与
代表 bank 等条数、近似等 token budget。E1-B 的第一次回答不进入第二次 prompt，assignment 阶段不
读取 gold answer。E1-C 的 memory K/V 不写入 native cache，从最后一个 prompt token 开始在每个
decode step 持续可见，并固定对 memory logits 使用 `-log(valid_slot_count)` 归一化。E1-C 的正式主
对照统一采用 split-prefill：no-memory、matched/shuffled text 与 matched/shuffled side-KV 均从最后一个
prompt token 的相同 cache segmentation 开始；E1-B full-prefill 结果只作为跨阶段数值 reference。

E1-A 在首轮 100 条 `calibration-val` 上未满足预注册 accuracy 判据：no-memory、representative 和三份
random bank 的 accuracy 分别为 `0.15/0.14/0.17/0.20/0.15`。但经验条件的严格格式准确率分别达到
`0.61/0.67/0.66/0.55`，相对 no-memory 的 `0.36` 均有明显提高。这证明经验文本可被模型消费并产生
稳定行为效应，可作为继续诊断 retriever 和 side-KV 的机制正对照；它不证明数学策略已经提高推理正确率。
逐样本审计显示 random-seed42 的 accuracy 点收益主要来自正确数值的格式修复，经验抽象粒度、审核/格式
语言占比、`k=5` 聚类覆盖和全局目录负迁移均登记为风险。E1-B/E1-C 因此可以继续作为组件诊断，但在
经验内容风险解除且任务收益确认前，不恢复 E1-D gate、不进入 final-test，也不声称 E1-A 正式通过。

E1-B 首轮 100 条 `calibration-val` 中，no-memory/matched/shuffled 的严格 accuracy 为
`0.15/0.26/0.19`，格式准确率为 `0.36/0.58/0.50`。matched 的严格收益主要来自格式修复，matched
相对 shuffled 的 diagnostic answer 只差 `+0.01`，因此 BM25 的数学策略选择能力未确认。旧 E1-C
实现又发现 E1-B full-prefill 与 E1-C split-prefill no-memory 只有 `28/100` 条完整轨迹一致，使跨路径
side-KV 对照带有数值混杂。当前 E1-C v3 原样复用 assignment 和 normalization，只修正对照路径，并把
full-vs-split 首步误差与首次分叉位置作为诊断而非机制硬门槛。

E1-C v3 的同路径机制已经通过：split no-memory 重复一致、side-KV baseline 首 token 与 split baseline
一致、native cache/trace 不变量全部成立。wrapped matched/shuffled text 相对 split no-memory 的格式差为
`+0.26/+0.25`，但 matched side-KV 格式差为 `-0.03`；当前 `log_valid_slots` memory attention mass 约
`0.013`。在调强度前先运行 E1C-T，拆分 wrapper-only 与 payload-only 的格式效应。只有 matched
payload-only 给出严格正格式 CI，才允许一次固定 `+log(10)` memory-odds 诊断；否则先对齐 compiler 文本
契约或停止通道声明。

每个阶段都包含 no-memory 与错配/随机对照并使用 sample-level paired 统计。具体 manifest、条件、
机制不变量和停止规则见 [E1 设计](e1_experience_memory_design.md)。

### E2：target/reference 字段的贡献（仅 E1 通过后）

在相同 manifest、retrieval id、token budget 和 side-KV 接口下比较：

1. `target-only`：`Prefer`；
2. `reference-only`：`Avoid`；
3. `contrast`：`When facing + Prefer + Avoid`。

该实验回答失败经验是否为成功策略提供额外 guardrail；它不重新测试 residual vector。

### E3：风险触发时机的贡献（对应 E1-D，仅 E1-A/B/C 通过后）

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

E0 已完成并冻结；E1-v1 已完成且当前组合未通过。E1-A 和 E1-B 首轮 `calibration-val` 已完成：格式行为
给出经验可消费性的正证据，但经验数学内容与 BM25 选择能力均未正式通过。当前只重跑修正对照路径后的
E1-C v3 已确认机制正确但当前 normalized side-KV 未传递 wrapped text 格式效应。当前只运行 E1C-T
文本来源分解，原样复用 E1-B assignment、E1-C v3 reference 和 E0 MemoryRecord；E1C-T 结果出来前不
实现强度改动，不改 layer 或 gate。在经验库生成策略调整前，不把组件结果外推为完整方法的任务收益。
配置冻结后只在 `dev-test` offset 100 之后做一次独立确认。

在 E1-A/B/C 得到逐组件证据前，不实现 gate timing、不扩展 memory 数量、不搜索 layer/注入强度，
不进入 E2/E3 或 `final-test`。
