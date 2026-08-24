# 经验记忆完整系统：实现契约与运行方式

## 1. 实现目标

完整系统已经把此前分阶段验证的组件连接为一条可运行链路：

```text
question + generated partial CoT
        │
        ├─ delimiter boundary → entropy + persistence-risk gate
        │                              │
        │                         first joint trigger
        │                              │
        ├─ answer-blind BM25 query: question + last 96 partial-CoT tokens
        │                              │
        └─ selected MemoryRecord → canonical layer-24 side-KV
                                       │
                         persistent joint attention through EOS
```

实现完整不代表效果成立。E1C-S 已表明当前 layer-24 canonical side-KV 没有传递 payload-only text 的
格式正效应；本实现的用途是形成稳定系统骨架，后续分别替换经验生成、检索、表示通道或 gate，而不再修改
其他模块的隐含行为。

## 2. 当前 reference profile

`ExperienceMemorySystemProfile` 是代码中的唯一 v1 profile，所有参数必须写入 manifest：

| 模块 | v1 固定值 |
|---|---|
| gate | 首个同时满足 entropy 与 persistence-risk 阈值的 pre-answer delimiter |
| query | 原题 + 最近 96 个 partial-CoT token |
| retriever | E0 BM25，选择 top-1，保留 top-2 score/margin 诊断 |
| memory count | 1 |
| layer | 24 |
| KV | E0 canonical pre-RoPE side-KV，relative phase delta 0 |
| normalization | `-log(valid_slot_count)` |
| fixed bias | `+log(10)`，memory odds multiplier 10 |
| lifetime | 从触发 boundary 的 attention 开始，持续到 EOS |
| HF cache | 只保存真实 prompt/completion token；memory slot 永不写入 native cache |

这些值不放入 `.e1.server.env`。后续优化必须产生新的 profile/schema，不能静默修改 v1。

## 3. 面向对象接口

- `ExperienceMemorySystemProfile`：冻结系统配置和序列化契约；
- `SemanticMemoryRetriever`：构造 answer-blind query、执行 BM25、连接 text record 与 KV slot metadata；
- `SemanticRetrievalDecision`：保存 query hash、top-2 hits、选择或拒绝原因；
- `EntropyRiskGate`：从冻结 risk artifact 计算 boundary entropy 与 persistence-risk；
- `SideKVAttentionController`：执行 layer-24 joint attention，保持 native cache 独立；
- `OnlineExperienceMemorySystem`：编排单次在线 gate→retrieval→persistent side-KV generation；
- `GreedyE1Runtime.generate_from_trigger_with_persistent_memory`：从冻结 trigger prefix 回放因果条件。

线上接口在一个 generation 内完成所有操作；冻结评测接口只为了构造 matched/shuffled 因果对照，不是另一套
检索或注入逻辑。

## 4. 在线执行语义

生成开始时 memory 未激活。每个 pre-answer delimiter 最多执行一次 gate probe；第一个 joint trigger 消耗
唯一检索机会：

1. query 只读取 question 和触发位置以前已经生成的 completion；
2. query 出现 final-answer marker、没有有效 term 或 BM25 无正分时 fail closed；
3. top-1 hit 的 payload hash、token count 和 side-KV slot count 必须一致；
4. 同一个 boundary 使用无 memory logits 作为 baseline，再从相同 native cache clone 计算 memory treatment；
5. treatment cache 继续 decode，side-KV controller 在每一步保持激活；
6. 每个 post-trigger token 必须产生一条 layer-24 trace，native cache length 只随真实 token 递增。

如果没有 gate trigger 或 retrieval abstain，系统完成普通 generation，并明确记录未应用 side-KV 的原因。

## 5. 冻结四条件评测

为了避免不同条件在触发位置、partial CoT 或 memory multiset 上产生混杂，评测分两步：

1. observation-only pass：完成无 memory generation，冻结第一个 joint trigger、prefix、query、matched ID；
2. global derangement：保持 matched/shuffled memory-ID 多重集合完全相同，且逐题 ID 不自配对；
3. replay：从相同 prefix 分别执行 matched/shuffled persistent side-KV。

当前 derangement 在保持全局 ID 多重集合的约束下最小化逐题 valid-slot 数差，并通过
`-log(valid_slot_count)` 控制 slot 数造成的总 attention prior；它不能保证每一对 valid-slot 数完全相等。
汇总因此必须报告 slot 差分布和 `exact_slot_count_sensitivity`。这是已登记的剩余混杂，不能把普通
assigned-subset 差异单独解释为纯内容效应。

四个条件为：

- `vanilla`；
- `gate_observation_only`；
- `matched_persistent_memory`；
- `shuffled_persistent_memory`。

未触发或检索 abstain 的样本在 memory 条件中精确复用 gate-only completion。assigned subset 用于主要
matched-vs-shuffled/matched-vs-gate 对照，全样本另报 intention-to-treat。

每条结果审计：question/prompt/prefix hash、gate observation、retrieval decision、memory/payload ID、完整
completion token、每步 attention mass、trace count、native cache length、normalization、score bias、首次
logits KL、baseline token parity、严格 GSM8K reward、diagnostic answer 和格式。

## 6. 服务器运行

完整四条件评测使用现有精简环境文件，只需要输出根目录和可选 GPU：

```bash
source scripts/experiments/gsm8k/.e1.server.env

MEMGEN_RUN_TAG=e1d-full-system-v1 \
bash scripts/experiments/gsm8k/run_e1d_full_system.sh \
  --logical-split calibration-val \
  --offset 0 \
  --limit 100 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT"
```

输出目录：

```text
$MEMGEN_OUTPUT_ROOT/e1/gsm8k/
  gsm8k_e1d_full-system_layer24_calibration-val_e1d-full-system-v1/
```

检查：

```bash
RUN_DIR="$MEMGEN_OUTPUT_ROOT/e1/gsm8k/gsm8k_e1d_full-system_layer24_calibration-val_e1d-full-system-v1"
jq '{
  status,
  formal_e1_passed,
  frozen_effect_criteria_passed,
  formal_task_claim,
  component_diagnostic,
  system_profile,
  triggered_count,
  assigned_count,
  conditions,
  primary_assigned_subset,
  intention_to_treat,
  mechanism_diagnostics,
  pairing_violations,
  acceptance
}' "$RUN_DIR/e1d_summary.json"
```

单题真实在线运行不读取 gold answer：

```bash
python scripts/run_online_experience_memory.py \
  --question 'A concrete GSM8K-style question' \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --bm25-index "$E0_DIR/bm25_index.v1.json" \
  --side-kv-manifest "$E0_DIR/side_kv_manifest.json" \
  --e0-final-report "$E0_DIR/e0_final_report.json" \
  --risk-artifact "$RISK_ARTIFACT" \
  --output /tmp/experience_memory_online.json \
  --device cuda \
  --dtype bfloat16
```

## 7. 后续模块优化边界

推荐依次优化：

1. MemoryRecord 抽象的数学可执行性；
2. matched-vs-shuffled 的语义检索区分度；
3. side-KV 表示/传输通道；
4. 最后才优化 gate 时机。

每次只替换一个接口并冻结新的 profile。当前 E1C-S 负结果仍有效，所以完整系统结果只能报告工程与机制
完整性；在上游组件重新给出正证据前，不进入 final-test，也不把 full-system accuracy 当作方法收益声明。
