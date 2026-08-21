# E1：经验内容、检索、side-KV 与 gate 的分阶段验证

## 1. 为什么拆分 E1

E1-v1 同时使用了 Phase 1 payload、BM25 top-1、风险 gate 和单步 side-KV，因此失败时不能区分是
经验内容、检索、传输通道还是插入时机的问题。100 条 `dev-test` 诊断中有 33 条被分配 memory：

- matched、shuffled 和 gate-only 在 assigned subset 上均为 `6/33`；
- matched 与 shuffled 相对 gate-only 均改变 `21/33` 条 completion，但两者只在 `6/33` 条上不同；
- 首 token 无一改变，matched/shuffled 的首步 KL 均约为 `0.0030`；
- memory attention mass 约为 `0.38`，说明 side path 活跃，但内容区分度不足。

因此，E1-v1 只证明“当前组合未产生准确率收益”，不能直接否定 Phase 1 经验、BM25 或 side-KV。
后续按 E1-A/B/C/D 分阶段检验，每个阶段都设置 matched、shuffled/random 与 no-memory 对照；任何阶段
失败都停止向后解释，不用下游结果反调上游配置。

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
4. 在预算内选择最大的可行 `k`，相同目标值按 `memory_id` 稳定打破平局；
5. 输出每个 medoid 的 cluster size、平均/最大半径和被覆盖记录 ID。

随机对照使用种子 `17/42/73`，每个对照与代表目录条数相同、总 token 数不超过 2048，并最小化与
代表目录的总 token 数差；随机集合不得与代表目录完全相同。随机目录同样只含真实 payload。

### 3.3 条件与判据

- `no_memory`
- `representative_bank_text`
- `random_bank_text_seed17`
- `random_bank_text_seed42`
- `random_bank_text_seed73`

所有题看到同一份固定目录，目录位于 user question 之后、assistant generation prompt 之前。主要判据是
representative 相对 no-memory 的 paired GSM8K accuracy 与严格格式差；三份随机目录报告经验集合效应
对抽样的敏感度。代表目录优于 no-memory 且格式不下降，才认为“Phase 1 经验集合有可利用信息”。

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

## 5. E1-C：side-KV 通道是否保留经验内容的作用

### 5.1 固定 assignment

E1-C 必须原样复用 E1-B manifest 中的 matched/shuffled ID，不重新检索。条件为：

- `no_memory`
- `matched_text`、`shuffled_text`（复用 E1-B 结果）
- `matched_persistent_side_kv`
- `shuffled_persistent_side_kv`

这使“文本通道与 side-KV 通道”使用完全相同的题目、经验 ID 和配对关系。

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

- `prompt[:-1] + prompt[-1]` 的分段 native prefill 与 E1-B no-memory completion token-level parity；
- 每个生成 token 恰有一条 layer-24 memory trace；
- 每一步 memory attention mass 有限且为正；
- native cache 长度只按真实 prompt/completion token 增长；
- memory ID、slot count、normalization mode 在整个 completion 中不变；
- disabled path 与 no-memory logits 保持 parity。

任务层主要检验 matched persistent side-KV 是否优于 shuffled persistent side-KV 和 no-memory；同时以
E1-B matched-text 效果作为可达到的文本上界。若文本有效而 side-KV 无效，结论应归因于表示/传输通道，
不能归因于经验内容或检索器。

## 6. E1-D：gate 时机（暂不实现）

只有 E1-A、E1-B、E1-C 均给出正证据后，才恢复 entropy+risk gate，比较 prompt-end persistent 与
completion 中触发后的 persistent side path。E1-D 只优化“何时开始可见”，不得重新选择 memory ID、
改变 payload 或 side-KV normalization。

## 7. 数据使用与停止规则

1. 已运行的 `dev-test` 前 100 条 E1-v1 仅作为诊断，不再用于配置选择或独立确认；
2. E1-A/B/C 首先在 `calibration-val` 完成机制与方向验证；
3. 配置冻结后，只在未触碰的 `dev-test` offset 100 之后做一次确认；
4. 任一阶段未通过，记录该组件结论并停止进入依赖它的下一阶段；
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

检查 `evaluation/e1a_summary.json`。只有方向和机制符合预注册假设后再运行 E1-B：

```bash
MEMGEN_RUN_TAG=e1b-calibration-v1 \
bash scripts/experiments/gsm8k/run_e1b_text_retrieval.sh \
  --limit 100 \
  "$PHASE1_DIR" "$E0_DIR"
```

E1-B 会先写入 answer-blind `assignment_manifest.json`，再读取 gold answer 做文本条件评测。只有
`formal_e1b_passed=true` 才允许把对应运行目录交给 E1-C：

```bash
E1B_RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/<实际的-e1b-运行目录>"
MEMGEN_RUN_TAG=e1c-calibration-v1 \
bash scripts/experiments/gsm8k/run_e1c_side_kv_channel.sh \
  "$PHASE1_DIR" "$E0_DIR" "$E1B_RUN_DIR"
```

E1-C 脚本和 Python runner 都会校验 E1-B summary、results hash、assignment hash 和 E0 side-KV
manifest；不满足时直接停止，不创建替代 assignment。冻结配置确认时使用
`--logical-split dev-test --offset 100`，不要覆盖已有 calibration 目录。E1-A 确认运行还必须通过
`--catalog-manifest <calibration-run/catalog_manifest.json>` 复用原目录，不重新聚类。
