# MemGen 经验记忆研究：核心结论与当前系统

本文是经验记忆研究的唯一当前文档。历史实验实现和过程性设计已从工作树移除；需要复现时使用 Git 历史。

## 1. 已确认结论

### 经验来源

- Phase 1 已构造 verifier-backed、Flash Teacher 抽象、Pro Reviewer 审核的 experience bank。
- 在线 memory pool 只使用 `ai_approved + answer_correctness` records。
- E0 从 192 条合格来源中得到 161 条无实例答案 literal 的 `MemoryRecord`。
- payload 固定为 `When facing / Prefer / Avoid`，不包含原题、完整轨迹、最终答案、`\boxed{}` 或原始
  evidence quote。

### 何时需要经验

- 高熵和 layer-24 hidden state 可以区分“高熵后自然恢复”与“持续发散”。
- held-out ROC-AUC 为 `0.8053`，balanced accuracy 为 `0.7158`。
- gate 只负责决定何时访问经验，不负责决定经验内容。

### 已关闭路线

- 全局 `recovery - persistence` residual vector 在独立确认中无效；不再调整 alpha、layer、符号或样本量。
- 同题 local-action raw hidden-state 检索 margin 很低；不再用于具体经验检索。

### E0 side-KV 机制

- 161 条 payload 已编译为 layer-24 canonical pre-RoPE side-KV。
- 8/8 answer-blind runtime audit cases 通过。
- mean memory attention mass 为 `0.3331`，mean first-step logits KL 为 `0.00507`。
- maximum canonical RoPE score relative error 为 `0.00109`。
- memory K/V 不写入 HuggingFace native cache，disabled path 与原始 logits 保持一致。

这些结果只证明机制正确且能够影响 logits，不证明 memory 内容改善任务表现。

### 经验内容、检索与通道诊断

- 固定经验文本目录明显提高严格格式准确率，但没有稳定提高数学答案正确率。
- BM25 top-1 文本经验相对 no-memory 提高严格 accuracy，主要来源仍是格式修复；数学策略选择能力尚未
  得到证明。
- payload-only 文本能稳定传递格式行为，说明模型会读取经验内容。
- 当前 layer-24 canonical side-KV 即使使用 `-log(valid slots) + log(10)`，mean attention mass 达到
  约 `0.084`，仍没有复现 payload-only 文本的格式效应。

因此当前最大风险依次是：经验抽象的数学可执行性、检索质量，以及 side-KV 的内容表示能力。gate timing
不是当前优先优化对象。

## 2. 当前完整系统

```text
question + generated partial CoT
        │
        ├─ pre-answer delimiter
        │        └─ entropy + persistence-risk gate
        │                 └─ first joint trigger
        │
        ├─ BM25 query: question + last 96 partial-CoT tokens
        │        └─ top-1 MemoryRecord
        │
        └─ canonical layer-24 side-KV
                 └─ persistent joint attention from trigger through EOS
```

冻结 reference profile：

| 项目 | 当前值 |
|---|---|
| layer | 24 |
| gate | 首个同时通过 entropy/risk 阈值的答案前 delimiter |
| query | 原题 + 最近 96 个 partial-CoT token |
| retrieval | BM25 top-1，保留 top-2 score/margin 诊断 |
| memory count | 1 |
| score | `raw - log(valid_slot_count) + log(10)` |
| lifetime | 从触发 boundary 到 EOS 持续可见 |
| native cache | 只包含真实 prompt/completion token |

核心对象：

- `ExperienceMemorySystemProfile`：版本化系统配置；
- `SemanticMemoryRetriever`：answer-blind query 和 BM25；
- `EntropyRiskGate`：冻结风险识别；
- `SideKVAttentionController`：canonical side path；
- `OnlineExperienceMemorySystem`：单遍在线编排；
- `GreedyE1Runtime`：冻结 prefix 的因果回放。

## 3. 当前评测条件

只保留三个条件：

- `vanilla`：不运行 gate，不检索，不注入 memory；
- `gate_observation_only`：观察并冻结 gate/retrieval，但不注入 memory；
- `matched_persistent_memory`：在同一冻结 trigger prefix 注入 BM25 top-1 side-KV，并持续到 EOS。

主要比较：

- `vanilla == gate-only` token parity：证明 gate probe 不改变原始生成；
- `matched - gate-only`：估计整个 memory treatment 的增量；
- `matched format - vanilla format`：检查格式安全性。

删除 mismatched/shuffled 对照后，`matched - gate-only` 同时包含检索、经验内容和 side-KV 激活效应，不能
单独识别“BM25 是否选对了经验”。汇总 artifact 会明确记录这一解释限制。

未触发或检索 abstain 的样本在 matched 条件中精确复用 gate-only completion。报告同时给出 assigned
subset 与全样本 intention-to-treat。

## 4. 运行与 artifact

服务器环境文件只保留输出目录和可选 GPU：

```bash
cp scripts/experiments/gsm8k/e1.server.env.example \
  scripts/experiments/gsm8k/.e1.server.env
```

开发诊断：

```bash
source scripts/experiments/gsm8k/.e1.server.env

MEMGEN_RUN_TAG=e1d-full-system-v2 \
bash scripts/experiments/gsm8k/run_e1d_full_system.sh \
  --logical-split calibration-val \
  --offset 0 \
  --limit 100 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT"
```

经明确决策，当前冻结系统允许在官方 GSM8K `final-test` 上进行一次全量评测：

```bash
MEMGEN_RUN_TAG=e1d-final-test-full-v1 \
bash scripts/experiments/gsm8k/run_e1d_full_system.sh \
  --logical-split final-test \
  --offset 0 \
  --limit 0 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT"
```

`--limit 0` 表示选择该 split 的全部样本。assignment 阶段只读取 test question；test answer 只在冻结
assignment 后的 evaluation 阶段用于评分。artifact 会记录 `dataset_split=test` 和
`evaluation_role=final_evaluation`。从本次运行开始，官方 test 不再是未查看的 pristine final set；其结果
不得用于反向调整同一版本后再宣称独立 final-test confirmation。

如需重建风险 artifact，只运行风险分类编译，不执行 residual 干预：

```bash
bash scripts/experiments/gsm8k/run_entropy_risk_gate.sh "$PHASE1_DIR"
```

输出包括：

- `assignment_manifest.json`：answer-blind gate、query 和 matched ID；
- `evaluation/results.jsonl`：三条件逐题结果和逐 token side-KV trace；
- `evaluation/run_report.json`：运行完整性与聚合指标；
- `e1d_summary.json`：paired effects、mechanism diagnostics 和解释边界。

单题在线运行：

```bash
python scripts/run_online_experience_memory.py \
  --question 'A GSM8K-style question' \
  --memory-records "$E0_DIR/memory_records.v2.jsonl" \
  --bm25-index "$E0_DIR/bm25_index.v1.json" \
  --side-kv-manifest "$E0_DIR/side_kv_manifest.json" \
  --e0-final-report "$E0_DIR/e0_final_report.json" \
  --risk-artifact "$RISK_ARTIFACT" \
  --output /tmp/experience_memory_online.json
```

## 5. 后续优化顺序

1. 提高 `MemoryRecord` 中数学策略的具体性、可执行性和验证能力；
2. 改进语义检索 query/index，使检索与当前错误状态更相关；
3. 重新设计 side-KV 内容表示，使其能够传递文本 payload 已证明存在的行为效应；
4. 前三项得到正证据后，再优化 gate 触发位置。

后续仍不新增 Teacher/Pro 调用，不恢复 residual-vector 路线，也不把完整系统的工程可运行性表述为任务
收益。final-test 结果可以报告冻结系统的实际表现，但在没有内容错配对照的当前三条件设计中，不能单独
归因于 BM25 匹配质量。
