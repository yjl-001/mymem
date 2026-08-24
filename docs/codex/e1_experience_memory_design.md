# E1：经验内容、检索、side-KV 与 gate 的分阶段验证

## 1. 为什么拆分 E1

E1-v1 同时使用了 Phase 1 payload、BM25 top-1、风险 gate 和单步 side-KV，因此失败时不能区分是
经验内容、检索、传输通道还是插入时机的问题。100 条 `dev-test` 诊断中有 33 条被分配 memory：

- matched、shuffled 和 gate-only 在 assigned subset 上均为 `6/33`；
- matched 与 shuffled 相对 gate-only 均改变 `21/33` 条 completion，但两者只在 `6/33` 条上不同；
- 首 token 无一改变，matched/shuffled 的首步 KL 均约为 `0.0030`；
- memory attention mass 约为 `0.38`，说明 side path 活跃，但内容区分度不足。

因此，E1-v1 只证明“当前组合未产生准确率收益”，不能直接否定 Phase 1 经验、BM25 或 side-KV。
后续按 E1-A/B/C/D 分阶段检验，每个阶段都设置 matched、shuffled/random 与 no-memory 对照。阶段未满足
预注册任务判据时，不把下游机制结果解释成上游任务收益，也不用下游结果反调上游配置；但如果已经获得
明确的机制正对照，可以继续运行后续组件诊断，以定位经验选择或表示通道的问题。组件诊断不等于正式通过。

## 2. 共享冻结项

- 只使用 E0 中 `ai_approved + answer_correctness` 且通过在线 payload 安全约束的 MemoryRecord；
- payload 原文与 E0 side-KV bank 一一对应，不重新调用 Teacher/Pro；
- 模型、tokenizer、ChatML 模板、greedy decoding、最大生成长度和 GSM8K verifier 固定；
- `calibration-val` 用于开发与机制选择，`dev-test` 的未触碰部分用于一次独立确认；
- 不在 `final-test` 选择聚类数、检索器、side-KV 配置或 gate 阈值；
- 每次实验输出 immutable manifest，后续阶段只消费 manifest，不暗中重算 assignment。

## 3. E1-A：Phase 1 经验集合是否有用

### 3.1 问题

E1-A 不使用检索器、不使用 gate，也不检验某一条经验是否适合某一道题。它检验：把 Phase 1 bank
压缩成一个固定、覆盖多种经验模式的文本目录后，模型整体 GSM8K 表现是否优于 no-memory。

### 3.2 代表经验目录

在全部 MemoryRecord 的 `sanitized_retrieval_key` 上构建 TF-IDF cosine 距离，并运行确定性
constrained k-medoids：

1. medoid 必须是 bank 中真实 MemoryRecord，不生成新的总结；
2. 目录使用完整 `When facing / Prefer / Avoid` payload；
3. 目录总预算固定为 2048 tokenizer tokens，预算包含目录包装文本；
4. `k` 不是按最短 payload 能塞入的最大数量确定，而是按每个目录位置最昂贵真实 entry 的
   additive token upper bound，选择“任意 `k` 条经验都不超过预算”的最大共同可行数量；
5. 代表目录和全部随机目录共享这个 `k`，相同聚类目标值按 `memory_id` 稳定打破平局；
6. 输出容量 upper bound、每个 medoid 的 cluster size、平均/最大半径和被覆盖记录 ID。

随机对照使用种子 `17/42/73`，每个对照与代表目录条数相同、总 token 数不超过 2048，并最小化与
代表目录的总 token 数差；随机集合不得与代表目录完全相同。共同可行容量保证随机目录天然满足
预算，不允许通过减少条数或更换 seed 补救。随机目录同样只含真实 payload。

### 3.3 条件与判据

- `no_memory`
- `representative_bank_text`
- `random_bank_text_seed17`
- `random_bank_text_seed42`
- `random_bank_text_seed73`

所有题看到同一份固定目录，目录位于 user question 之后、assistant generation prompt 之前。主要判据是
representative 相对 no-memory 的 paired GSM8K accuracy 与严格格式差；三份随机目录报告经验集合效应
对抽样的敏感度。代表目录优于 no-memory 且格式不下降，才认为“Phase 1 经验集合有可利用信息”。

### 3.4 calibration-val 结果与当前解释

首轮 100 条 `calibration-val` 结果如下：

| 条件 | GSM8K accuracy | 严格格式准确率 |
|---|---:|---:|
| `no_memory` | 0.15 | 0.36 |
| `representative_bank_text` | 0.14 | 0.61 |
| `random_bank_text_seed17` | 0.17 | 0.67 |
| `random_bank_text_seed42` | 0.20 | 0.66 |
| `random_bank_text_seed73` | 0.15 | 0.55 |

代表目录未满足预注册的 accuracy 判据，因此 E1-A 的正式状态保持 `did_not_pass`。但所有经验目录都明显
提高严格格式准确率，说明文本经验被模型读取，payload 中的操作性指令能够稳定改变生成行为。这是经验
可消费性的正对照，足以支持继续把 E1-B/E1-C 作为检索和表示通道的组件诊断；它不能单独证明经验中的
数学策略改善了推理，也不能据此进入 gate timing 或 final-test。

逐样本审计还显示，`random_bank_text_seed42` 的 14 个 accuracy 收益中，9 个是“原数值正确但未使用
boxed 格式”的格式修复，只有 5 个改变了错误数值；9 个损失中有 7 个把原本正确数值改错。因此其
`+0.05` accuracy 点估计主要由格式合规驱动，且置信区间包含零。后续继续保留 GSM8K accuracy，但在
当前组件开发阶段把它作为风险与诊断指标，不以本轮小样本结果反向修改经验生成规则。

当前登记的经验库风险为：

1. 抽象粒度不稳定：`When facing` 往往保留较窄题型，`Prefer` 又可能过于通用，未形成一致的“适用条件—
   可执行动作—验证规则”层级；问题不只是单向的“不够具体”或“不够抽象”。
2. payload 中反复出现 boxed、verifier、expected answer 等格式/审核语言，容易让最显著的可观测收益集中在
   格式，而稀释数学策略信号。
3. 受 2048-token 预算约束的 `k=5` representative 聚类高度不均衡，最大簇覆盖 `97/161` 条记录且簇内
   距离较大；TF-IDF medoid 的词面中心性不等于经验效用。
4. E1-A 把同一目录用于所有题，必然包含大量不适用经验；正确经验也可能因为没有检索和适用性过滤而产生
   负迁移。
5. 约 1180 个额外 prompt token 可能对小模型造成注意力稀释；这与经验内容质量在当前 no-memory 对照中
   尚未完全分离。

这些风险留待 E1-B 的 matched-vs-shuffled 结果和后续经验库生成迭代共同处理；本轮不新增 Teacher/Pro
调用，也不为了追逐 calibration accuracy 调整经验内容。

## 4. E1-B：BM25 是否能选出更有用的经验

### 4.1 两遍 answer-blind assignment

对每道题先执行一次完整 no-memory greedy generation，得到 `y0`。`y0` 只用于构造检索 query，绝不
放进第二遍回答的 prompt：

```text
query = question + sanitize_preanswer(y0)
```

`sanitize_preanswer` 在 `\boxed{}`、`\fbox{}`、`final answer`、`answer is` 等答案标记前截断，移除
数字/数学 literal 和模板控制符，再经与 BM25 index 相同的 analyzer 处理。assignment 阶段不读取
gold answer、reward 或 verifier 结果。

BM25 top-1 返回 matched `memory_id`。对全部 matched ID 做确定性全局 bijective derangement，保证：

- 每题 shuffled ID 不等于 matched ID；
- matched/shuffled 的 memory-ID 多重集合完全一致；
- payload 数量和包装模板完全一致；
- 若多重集合无法无自配对，manifest 构建明确失败，不换用回退策略。

### 4.2 条件与判据

- `no_memory`：复用第一遍 `y0`；
- `matched_text`：第二遍 prompt 追加 BM25 top-1 payload；
- `shuffled_text`：相同 prompt/token 包装，仅替换 memory ID。

主要检验 matched-text 是否同时优于 shuffled-text 和 no-memory；严格格式不得低于 no-memory。
manifest 冻结 question hash、`y0` token/hash、sanitized query/hash、BM25 score、matched/shuffled ID、
payload hash 与精确 prompt token 数。

正式任务判据仍使用严格 GSM8K reward，不因 calibration 结果改写。组件诊断同时报告
`diagnostic_answer_accuracy`、严格 reward 翻转中的 format-only/answer-content 分解，以及三组条件间的
token-level completion divergence。只要 assignment、无自配对 shuffle、answer-blind query 和 artifact
provenance 完整，即允许 E1-C 原样消费该 manifest；这不要求 E1-B 已在 accuracy 上正式通过。

### 4.3 calibration-val 结果与当前解释

100 条样本中，no-memory、matched-text、shuffled-text 的严格 accuracy 分别为 `0.15/0.26/0.19`，
格式准确率为 `0.36/0.58/0.50`，diagnostic answer accuracy 为 `0.37/0.30/0.29`。matched 相对
no-memory 的严格收益为 `+0.11`，但 15 个严格收益中有 12 个是数值判断未改善、只修复格式；matched
相对 shuffled 的严格差为 `+0.07`，置信区间跨零，diagnostic answer 只差 `+0.01`。因此 E1-B 的
assignment/provenance 组件通过，但 BM25 的数学策略选择能力尚未得到证明。matched 和 shuffled 都能提高
格式，说明单条 Phase 1 payload 的通用指令效应仍然存在，后续 E1-C 可把它作为通道正对照。

matched/shuffled 在全局保持同一 memory-ID 多重集合，但逐题 prompt token 数没有精确相等；这仍是
matched-vs-shuffled 的长度—内容混杂风险。当前不修改冻结 assignment，也不据此调整 BM25。

## 5. E1-C：side-KV 通道是否保留经验内容的作用

### 5.1 固定 assignment

E1-C 必须原样复用 E1-B manifest 中的 matched/shuffled ID，不重新检索。E1-C v3 将条件分成两类。
跨阶段 reference 只用于观察 full/split 数值路径差异：

- `e1b_full_no_memory`
- `e1b_full_matched_text`
- `e1b_full_shuffled_text`

正式 E1-C 对照全部重新走相同的 split-prefill 路径：

- `split_no_memory`
- `split_matched_text`
- `split_shuffled_text`
- `matched_persistent_side_kv`
- `shuffled_persistent_side_kv`

文本 prompt 从冻结 E0 MemoryRecord 重建，并必须与 E1-B 保存的 prompt-token hash 一致。这使主对照不仅
使用完全相同的题目、经验 ID 和配对关系，也使用相同的 cache segmentation；E1-B full-prefill 结果不再
直接充当 side-KV 的 no-memory/text control。

### 5.2 prompt-end persistent side path

side-KV 不写入 HuggingFace native cache。运行时先对 `prompt_tokens[:-1]` 做普通 prefill；从最后一个
prompt token 的 layer-24 attention 开始激活同一份静态 memory K/V，并在每个 decode step 持续可见
到 EOS：

```text
joint_attention(q_t, native_KV_t, memory_KV)
native cache update: only real token K/V
memory side path: static, visible at every step, never cached as real tokens
```

它在语义上类似 cross-attention，但复用 layer-24 原生 self-attention 的 Q/K/V/O、head layout 和 joint
softmax，不新增可训练 cross-attention 模块。

为避免 memory token 数量本身扩大总先验质量，对每个 query/head 的全部有效 memory logits 固定减去
`log(M)`，其中 `M` 是该 MemoryRecord 的有效 slot 数：

```text
memory_scores = memory_scores - log(valid_slot_count)
```

slot 是 payload 中一个 tokenizer token 在 layer 24 得到的一组 canonical pre-RoPE K/V 向量；它不是
真实上下文 token、词表 token 或可训练 latent。一个含 `M` 个有效 payload token 的 MemoryRecord 就有
`M` 个 slot。

### 5.3 机制与任务判据

每个 side-KV 样本必须满足：

- `split_no_memory` 重复运行 token-level parity；
- 两个 side-KV 分支在启用 memory 前的 baseline 首 token 都与 `split_no_memory` 一致；
- 每个生成 token 恰有一条 layer-24 memory trace；
- 每一步 memory attention mass 有限且为正；
- native cache 长度只按真实 prompt/completion token 增长；
- memory ID、slot count、normalization mode 在整个 completion 中不变；
- 所有主条件明确记录同一 `split-before-final-prompt-token` 路径。

另行记录 full-prefill 与 split-prefill 的首步 logits KL、最大绝对误差、top-1 是否变化、完整 completion
是否一致及首次分叉位置。这组数值只作诊断，不再作为 side-KV 机制硬门槛：只要同一路径重复确定、native
cache/trace 不变量成立，就可以把主对照解释为 side-KV 的因果差异。

任务层主要检验 matched persistent side-KV 是否优于 shuffled persistent side-KV 和
`split_no_memory`；同时以 `split_matched_text` 效果作为同路径文本上界。若同路径文本有效而 side-KV
无效，结论应归因于表示/传输通道，不能归因于经验内容或检索器。

由于 E1-A 已观察到稳定格式效应，E1-C 额外把格式作为内容传递正对照：分别计算 matched-text 和
matched-side-KV 相对 no-memory 的格式差。只有文本差为正时，才判断 side-KV 是否复现同方向格式效应；
若文本条件在该批 top-1 assignment 上没有正格式效应，则报告 `no_positive_text_control`，不把它误判为
side-KV 传输失败。机制完整性、格式正对照传递和正式任务收益分别报告，互不替代。

### 5.4 E1-C v2 结果为何需要同路径重跑

旧版 100 条结果中，matched/shuffled side-KV 的 memory attention mass 约为 `0.0134/0.0132`，首步
KL 约为 `0.0010/0.0008`，说明 persistent side path 活跃但很弱；matched side-KV 的严格/格式准确率为
`0.10/0.29`，低于当时直接引用的 matched-text `0.26/0.58`。然而 E1-B full-prefill no-memory 与 E1-C
split-prefill no-memory 只有 `28/100` 条完整 completion 一致。BF16 下微小 shape-dependent logits 差异
可能被 greedy decoding 放大，因此旧版跨路径的 side-KV-vs-no-memory/text 数值不能作为干净的通道
结论。E1-C v3 随后确认 split 重复 `100/100` 一致、两个 side-KV baseline 首 token 均 `100/100`
一致，cache/trace 机制通过。split matched/shuffled text 相对 split no-memory 的格式差为 `+0.26/+0.25`，
而 matched side-KV 为 `-0.03`；matched/shuffled memory attention mass 约为 `0.0133/0.0132`。因此当前
`layer-24 + log_valid_slots` 通道没有传递 wrapped text 的格式正效应，也没有 matched 内容收益。该结果
触发 E1C-T 文本来源分解，不直接调整 `log_valid_slots`、layer 或 gate。

### 5.5 E1C-T：文本效应来源分解

E1-C v3 已确认同路径机制完整，但 matched side-KV 没有复现 wrapped text 的格式正效应。由于文本条件
使用固定 `General experience guidance` wrapper，而 E0 side-KV 用另一段 compiler prefix 编译并只保留
`When facing / Prefer / Avoid` payload slots，下一步先分解文本效应，不直接调 side-KV 强度。

E1C-T 原样复用 E1-C v3 的：

- `split_no_memory`
- `split_matched_text`
- `split_shuffled_text`

只新增三个相同 split-prefill 路径的条件：

- `split_wrapper_only`：只追加 single-experience guidance wrapper；
- `split_payload_only_matched`：只追加 matched MemoryRecord payload；
- `split_payload_only_shuffled`：只追加 shuffled MemoryRecord payload。

E1C-T 不重新检索、不重算 assignment、不调用 Teacher/Pro，也不改变 E0 side-KV。输入必须通过 E1-C v3
results/run-report hash、E1-C summary v3、E0 MemoryRecord hash 和 split manifest hash 校验。主要诊断
使用 paired 格式差；严格 accuracy 与 diagnostic answer 作为风险指标，不作为文本来源定义。

预注册路由为：

1. matched payload-only 相对 no-memory 的格式 bootstrap 95% CI 下界大于零：允许进入一次固定的
   `+log(10)` memory-odds 强度测试；
2. payload-only 不为正，但 wrapper-only 或 wrapped matched 为正：先对齐 side-KV compiler 与在线文本
   契约，不做强度测试；
3. wrapped/payload/wrapper 均无正对照：停止当前 side-KV 内容传递声明。

shuffled payload-only 是否也为正用于判断 payload 行为效应能否跨 memory ID 复现；matched payload-only
若显著损害 diagnostic answer，报告会单独登记，但不会把格式正对照误写成数学推理收益。E1C-T 本身不
实现或运行强度实验。

E1C-T 的 100 条 `calibration-val` 结果已经完成：wrapper-only 相对 no-memory 没有格式收益；
payload-only matched/shuffled 的格式准确率分别为 `0.63/0.61`，相对 no-memory 的 `0.32` 均给出严格
正的 paired bootstrap 区间。因此可观测文本效应来自 MemoryRecord payload，而不是固定 wrapper。
matched payload-only 相对 shuffled 的格式差只有 `+0.02` 且不显著，仍不能证明 BM25 语义选择能力。
该结果按预注册路由只授权 E1C-S 一次固定强度诊断。

### 5.6 E1C-S：一次固定 side-KV 强度诊断

E1C-S 不搜索强度。它保持 E1-C v3 的 layer 24、canonical pre-RoPE bank、persistent side path、
`log_valid_slots` normalization 和 E1-B matched/shuffled assignment，只在每个有效 memory score 上增加
唯一预注册常数：

```text
memory_scores = raw_memory_scores - log(valid_slot_count) + log(10)
```

这等价于把 memory 相对 native token 的总 attention odds 乘以 `10`，不是把归一化后的 attention mass
直接乘以 `10`。以 E1-C v3 约 `0.013` 的平均 mass 估算，预期新 mass 约为 `0.12`；E1C-S 预注册的
matched/shuffled 聚合均值区间均为 `[0.05, 0.25]`。超出区间不再改常数补跑。

E1C-S 只新增两个生成条件：

- `fixed_log10_matched_persistent_side_kv`；
- `fixed_log10_shuffled_persistent_side_kv`。

以下五个条件直接从已认证结果复制，不重新生成：`split_no_memory`、payload-only matched/shuffled，
以及 normalized side-KV matched/shuffled。所有条件保持同一个 split-prefill 路径。报告必须逐 token
验证 cache 长度、memory ID、slot count、normalization、精确 score bias、attention mass 与 baseline
首 token；并再次从 E1-C result rows 验证旧机制证据，而不是只信 summary。

E1C-S 的主要问题是“当前 canonical layer-24 side-KV 通道在合理的更强 attention 下，能否传递已经由
payload-only text 证明存在的格式行为效应”。`fixed matched - no-memory` 的格式 bootstrap 95% CI 下界
严格大于零，且 matched/shuffled mean attention mass 均在预注册区间，才记为 channel-capacity evidence。
shuffled 是否同样传递用于判断该效应是通用 payload 行为还是 matched 内容特异效应；matched-vs-shuffled
仍是次要语义诊断。strict accuracy 与 diagnostic answer 只报告风险：它们不把本组件诊断升级为任务收益。

停止规则如下：

1. 机制不完整、attention mass 越界或 matched 格式效应不传递：停止当前
   `layer-24 + canonical payload KV` 通道，不搜索 layer/强度；
2. 格式效应传递但 diagnostic answer 显著受损：只记录 channel capacity，拒绝该强度作为候选配置；
3. 格式效应传递且无显著 answer harm：记录 channel capacity，下一步回到经验抽象与检索设计；在数学
   内容和 matched 选择收益成立前仍不恢复 gate。

E1C-S 的 100 条 `calibration-val` 结果已经完成。matched/shuffled mean attention mass 为
`0.0841/0.0832`，均在目标区间，cache/prefill/trace 不变量全部通过；但格式准确率只有 `0.30/0.28`，
低于 no-memory 的 `0.32`，更显著低于 payload-only text 的 `0.63/0.61`。matched 相对 no-memory 的
格式差为 `-0.02`，CI `[-0.12, 0.08]`；matched 相对 shuffled 的格式差 `+0.02`，CI 跨零。系统因此
输出 `no_payload_effect_transfer_at_fixed_strength` 和 `stop_current_layer24_side_kv_channel`。这关闭当前
表示通道的效果调参路线，但不妨碍把它保留为完整系统骨架中的已知弱基线。

## 6. E1-D：完整系统骨架与 gate 时机

为支持后续逐模块优化，E1-D 的工程链路已经完整实现：在线 entropy+risk gate、`question + partial CoT`
BM25、top-1 MemoryRecord、触发后 persistent side-KV，以及冻结 prefix 的 matched/shuffled/no-memory
因果评测。当前 reference profile 使用 E1C-S 的 `log_valid_slots + log(10)`，确保 channel 具有非平凡
attention；这不推翻 E1C-S 的负结论，也不表示该强度被接受为有效方法配置。

科学上仍不允许用 E1-D 调 gate 阈值或声明 gate 收益：E1-A/B 的数学内容与检索区分度未通过，E1C-S 又
没有证明当前 side-KV 能传递 payload 行为。当前 E1-D 只证明整个系统可以执行并产生完整审计 artifact。
后续必须先逐模块获得正证据，最后才能比较 prompt-end 与风险触发时机。实现接口、在线/冻结执行语义和
服务器命令见 [完整系统文档](experience_memory_full_system.md)。

## 7. 数据使用与停止规则

1. 已运行的 `dev-test` 前 100 条 E1-v1 仅作为诊断，不再用于配置选择或独立确认；
2. E1-A/B/C 首先在 `calibration-val` 完成机制与方向验证；
3. 配置冻结后，只在未触碰的 `dev-test` offset 100 之后做一次确认；
4. 任一阶段未通过预注册任务判据，记录正式状态；可以执行 E1-D 工程完整性诊断，但不得据此优化 gate、
   进入 final-test 或声称完整方法获得任务收益；
5. 全过程不运行 `final-test`，不新增 Teacher/Pro 调用，不恢复 residual-vector 路线。

## 8. 服务器执行顺序

三个阶段使用同一份精简环境文件：

```bash
cp scripts/experiments/gsm8k/e1.server.env.example \
  scripts/experiments/gsm8k/.e1.server.env
```

其中只需设置 `MEMGEN_OUTPUT_ROOT`；`MEMGEN_E1_CUDA_VISIBLE_DEVICES` 可选。先运行 E1-A：

```bash
MEMGEN_RUN_TAG=e1a-calibration-v1 \
bash scripts/experiments/gsm8k/run_e1a_bank_utility.sh \
  --limit 100 \
  "$PHASE1_DIR" "$E0_DIR"
```

检查 `evaluation/e1a_summary.json`。E1-A 的格式正对照已经允许 E1-B 继续作为组件诊断：

```bash
MEMGEN_RUN_TAG=e1b-calibration-v1 \
bash scripts/experiments/gsm8k/run_e1b_text_retrieval.sh \
  --limit 100 \
  "$PHASE1_DIR" "$E0_DIR"
```

完成后检查正式状态、组件 handoff 和格式/数值分离诊断：

```bash
E1B_RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/<实际的-e1b-运行目录>"
jq '{
  status,
  formal_e1b_passed,
  component_diagnostic,
  conditions,
  accuracy_effects,
  diagnostic_answer_effects,
  format_effects,
  strict_accuracy_transition_diagnostics,
  completion_difference_diagnostics,
  retrieval_diagnostics,
  pairing_diagnostics
}' "$E1B_RUN_DIR/evaluation/e1b_summary.json"
```

E1-B 会先写入 answer-blind `assignment_manifest.json`，再读取 gold answer 做文本条件评测。运行目录只需
满足 `component_diagnostic.e1c_component_diagnostic_allowed=true` 即可交给 E1-C；若
`formal_e1b_passed=false`，E1-C 会明确标记为 component-diagnostic mode，不会把结果解释成 E1-B 正式
通过：

```bash
E1B_RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/<实际的-e1b-运行目录>"
MEMGEN_RUN_TAG=e1c-calibration-v3 \
bash scripts/experiments/gsm8k/run_e1c_side_kv_channel.sh \
  "$PHASE1_DIR" "$E0_DIR" "$E1B_RUN_DIR"
```

完成后分别检查正式任务判据、side-KV 机制和格式正对照传递：

```bash
E1C_RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/<实际的-e1c-运行目录>"
jq '{
  status,
  formal_e1c_passed,
  source_e1b_formal_passed,
  component_diagnostic,
  condition_roles,
  conditions,
  accuracy_effects,
  diagnostic_answer_effects,
  format_effects,
  format_positive_control_transfer,
  completion_difference_diagnostics,
  exact_slot_count_sensitivity,
  prefill_path_diagnostics,
  mechanism_diagnostics,
  acceptance
}' "$E1C_RUN_DIR/evaluation/e1c_summary.json"
```

E1-C 脚本和 Python runner 都会校验 E1-B component handoff、summary/results hash、assignment hash 和
E0 side-KV manifest；不满足时直接停止，不创建替代 assignment。冻结配置确认时使用
`--logical-split dev-test --offset 100`，不要覆盖已有 calibration 目录。E1-A 确认运行还必须通过
`--catalog-manifest <calibration-run/catalog_manifest.json>` 复用原目录，不重新聚类。

完成 E1-C v3 后运行 E1C-T。这里的 `E1C_V3_RUN_DIR` 必须指向刚才机制通过的 v3 目录，不使用旧 v2：

```bash
E1C_V3_RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1c_side-kv_calibration-val_e1c-calibration-v3"
MEMGEN_RUN_TAG=e1ct-calibration-v1 \
bash scripts/experiments/gsm8k/run_e1ct_text_source.sh \
  "$PHASE1_DIR" "$E0_DIR" "$E1B_RUN_DIR" "$E1C_V3_RUN_DIR"
```

检查文本来源与自动路由：

```bash
E1CT_RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1ct_text-source_calibration-val_e1ct-calibration-v1"
jq '{
  status,
  formal_task_claim,
  component_diagnostic,
  condition_roles,
  conditions,
  format_effects,
  diagnostic_answer_effects,
  strict_accuracy_transition_diagnostics,
  decision
}' "$E1CT_RUN_DIR/evaluation/e1ct_summary.json"
```

只有 `decision.next_step == "e1cs_fixed_log10_memory_odds_test"` 时 E1C-S runner 才接受该 E1C-T
目录；其他结果按 `decision.next_step` 停止或对齐 compiler，不使用 accuracy 反调强度。

当前 E1C-T 已授权 E1C-S。运行时必须同时提供原 E1-C v3 和 E1C-T 目录，以复用并认证全部冻结对照：

```bash
E1CT_RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1ct_text-source_calibration-val_e1ct-calibration-v1"
MEMGEN_RUN_TAG=e1cs-calibration-v1 \
bash scripts/experiments/gsm8k/run_e1cs_fixed_strength.sh \
  "$PHASE1_DIR" "$E0_DIR" "$E1B_RUN_DIR" "$E1C_V3_RUN_DIR" "$E1CT_RUN_DIR"
```

只需要检查一个汇总文件：

```bash
E1CS_RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1cs_fixed-strength_calibration-val_e1cs-calibration-v1"
jq '{
  status,
  formal_task_claim,
  component_diagnostic,
  fixed_strength,
  condition_roles,
  conditions,
  format_effects,
  diagnostic_answer_effects,
  mechanism_diagnostics,
  decision
}' "$E1CS_RUN_DIR/evaluation/e1cs_summary.json"
```

必须按 `decision.next_step` 解释结果；本轮不允许换 bias、layer、memory 数量或样本 split 补跑。
