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
结论。matched-vs-shuffled side-KV 同为 split 路径，仍未显示内容选择收益，但在 E1-C v3 完成前不调整
`log_valid_slots`、layer 或 gate。

## 6. E1-D：gate 时机（暂不实现）

只有 E1-A、E1-B、E1-C 均给出正证据后，才恢复 entropy+risk gate，比较 prompt-end persistent 与
completion 中触发后的 persistent side path。E1-D 只优化“何时开始可见”，不得重新选择 memory ID、
改变 payload 或 side-KV normalization。

## 7. 数据使用与停止规则

1. 已运行的 `dev-test` 前 100 条 E1-v1 仅作为诊断，不再用于配置选择或独立确认；
2. E1-A/B/C 首先在 `calibration-val` 完成机制与方向验证；
3. 配置冻结后，只在未触碰的 `dev-test` offset 100 之后做一次确认；
4. 任一阶段未通过预注册任务判据，记录正式状态；已有有效机制正对照时可以继续下游组件诊断，但不得据此
   进入 E1-D、final-test 或声称完整方法获得任务收益；
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
