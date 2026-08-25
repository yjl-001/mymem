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
- SDPA 重新编译后的 held-out ROC-AUC 为 `0.8026`，balanced accuracy 为 `0.7180`，与 eager
  历史结果 `0.8053/0.7158` 基本一致。
- SDPA holdout 包含 353 个 persistence 和 48 个 recovery 事件；每个 partition/label 的最低事件数
  门槛在看到 48 之后由 50 事后调整为 40，必须作为实验限制披露。该调整不改变 prototype、entropy
  threshold 或固定的 `risk_threshold=0`。
- gate 只负责决定何时访问经验，不负责决定经验内容。

### 已关闭路线

- 全局 `recovery - persistence` residual vector 在独立确认中无效；不再调整 alpha、layer、符号或样本量。
- 同题 local-action raw hidden-state 检索 margin 很低；不再用于具体经验检索。

### E0 side-KV 机制

- 161 条 payload 已在 SDPA 下重新编译为 layer-24 canonical pre-RoPE side-KV。
- 8/8 answer-blind、pre-answer runtime audit cases 通过，`formal_e0_passed=true`。
- canonical shared-phase RoPE、disabled-path logits parity、native cache prefix/length、正 attention mass 和
  active logits effect 均通过审计。
- memory K/V 不写入 HuggingFace native cache，只在 layer-24 SDPA side path 中与 native K/V 联合归一化。

这些结果只证明 SDPA 机制正确且能够影响 logits，不证明 memory 内容改善任务表现。

### Attention backend 基线

- 在同一 32 道 official GSM8K test 题、`batch_size=1` 和相同生成契约下，eager 严格准确率为
  `0.09375`，FlashAttention2 为 `0.65625`。
- 两个 backend 各自的 native generation 与 explicit live-cache generation 都逐 token 完全一致；
  因此低 baseline 不是 KV cache 或手工 decode 导致的。
- 32/32 completion 在 backend 之间最终分叉，但首 token 全部一致，平均共享前缀为 23.125 token。
- 正式 E1 不再允许使用 eager，也不允许 vanilla 与 matched 使用不同 backend。
- 同一批 32 题上，SDPA 严格准确率为 `0.53125`、诊断答案准确率为 `0.6875`、格式准确率为
  `0.65625`，且 SDPA native/cache 逐 token 完全一致。SDPA 因此被选为当前 side-KV 正式 runtime。
- FlashAttention2 仍是质量参考（严格准确率 `0.65625`），SDPA 结果只能表述为“SDPA runtime 下”的
  系统效果，不能表述为 FlashAttention2 等价效果。

当前代码与正式 artifact 已迁移到 SDPA；旧 eager risk artifact、side-KV bank、E0 final report 和
E1 assignment 均由 schema 拒绝复用。

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
| attention backend | SDPA（vanilla、gate observation、matched 完全一致） |
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

GSM8K 输入统一使用 `GSM8KPromptContract(gsm8k-memgen-builder-v1)`，严格复现
原始 dataset builder 的 `\boxed{}.Question:` 拼接；不允许 E1 单独维护 prompt。
生成预算固定为 1024。`vanilla` 精确复现仓库原始 baseline：以 `inputs_embeds`
调用 HuggingFace greedy generation，`use_cache=False`、`repetition_penalty=1.0`；
matched 路径使用 raw-argmax 显式 live KV-cache，并记录首个 token 分叉位置。所有正式评测均固定
`batch_size=1`。

只保留两个测评条件：

- `vanilla`：不运行 gate，不检索，不注入 memory；
- `matched`：运行 gate/retrieval，在冻结 trigger prefix 注入 BM25 top-1 side-KV，并持续到 EOS。

主要比较：

- assigned subset 的 `matched - vanilla`：检查实际收到 memory 的样本；
- 全样本 intention-to-treat 的 `matched - vanilla`：估计完整系统的总体增量；
- `matched format - vanilla format`：检查格式安全性。

gate observation completion 仍在 assignment 阶段生成，用于冻结触发位置和检索输入，但不作为测评条件。
系统内部要求它与 vanilla 逐 token 相同；该 parity 只是一项运行完整性审计。

未触发或检索 abstain 的样本在 matched 条件中精确复用 vanilla completion。`matched - vanilla`
同时包含 gate、检索、经验内容和 side-KV 激活效应，不能单独识别任一组件的贡献。

## 4. SDPA 全量 final-test 结果

冻结系统已在 official GSM8K test 的全部 1319 题上运行；643 题触发并全部完成 BM25 top-1 assignment，
占 `48.75%`。运行完整性通过：assignment/runtime 无违规，vanilla 与内部 observation 逐 token 一致。

| 指标 | vanilla | matched | 全量差值 |
|---|---:|---:|---:|
| 严格准确率 | `0.478393` | `0.478393` | `0.000000` |
| diagnostic answer accuracy | `0.684610` | `0.680819` | `-0.003791` |
| 格式准确率 | `0.669447` | `0.668688` | `-0.000758` |
| 平均生成长度 | `281.875` | `282.576` | `+0.701` token |

在 643 个 assigned samples 上：

- 严格正确性 14 题由错变对、14 题由对变错，净效应为零，95% bootstrap CI 为
  `[-0.01555, 0.01555]`，McNemar `p=1.0`；
- diagnostic answer 8 题改善、13 题退化，差值 `-0.00778`，95% CI 为
  `[-0.02177, 0.00622]`；
- 格式 12 题改善、13 题退化，差值 `-0.00156`，95% CI 为
  `[-0.01711, 0.01400]`。

机制持续生效：mean memory attention mass 为 `0.07765`，mean first-step logits KL 为 `0.001116`，
每个触发样本平均记录约 149 个 persistent-memory decode steps。由此可以排除 memory 未进入 attention、
只使用一次、cache 损坏或 baseline replay 漂移；但当前影响没有正确方向，表现为收益与损失相互抵消。

Gate 选中的题在 vanilla 下 diagnostic answer accuracy 为 `0.6096`，未触发题约为 `0.7559`，说明 gate
确实富集了数学上更困难的样本。相反，触发题格式准确率为 `0.7745`，未触发题约为 `0.5695`；当前
pre-answer delimiter gate 不是格式失败检测器。

正式结论是：当前 `SDPA gate + BM25 top-1 + layer-24 persistent side-KV` 端到端任务收益为零，预设
effect acceptance 未通过；但机制与运行完整性通过。该结果不能单独证明经验库、BM25 或 side-KV 中
任一模块无效。结合既有文本 payload 能传递格式行为而 side-KV 未能复现的证据，当前首要嫌疑仍是
side-KV 内容表示，其次是检索与经验抽象；不应先盲调 memory bias 或 gate timing。

official GSM8K test 已不再是未查看的 pristine final set。后续只能对该结果做描述性/只读误差分析；
所有配置优化必须在 calibration/dev 上进行，不能再次使用同一 test 调参后声称独立 final confirmation。

## 5. 运行与 artifact

先进行 base parity 预检：

```bash
MEMGEN_RUN_TAG=base-parity-v1 \
bash scripts/experiments/gsm8k/run_base_reasoner_parity.sh \
  --logical-split final-test \
  --limit 32 \
  "$PHASE1_DIR" "$E0_DIR"
```

只有 `base_parity_summary.json.status=passed` 且 native baseline 回到已知合理区间，
才继续解释 E1。该预检不加载 gate、BM25 或 side-KV，不产生 memory 效果结论。

已完成的 eager/FlashAttention2 诊断固定 `batch_size=1`：

```bash
MEMGEN_RUN_TAG=base-attention-final32-v1 \
bash scripts/experiments/gsm8k/run_base_attention_backend_comparison.sh \
  --logical-split final-test \
  --limit 32 \
  --reference-backend eager \
  --candidate-backend flash_attention_2 \
  "$PHASE1_DIR" "$E0_DIR"
```

结果已确认 attention backend 是主要差异来源。随后以 FlashAttention2 为参考完成了 SDPA 检查：

```bash
MEMGEN_RUN_TAG=base-sdpa-final32-v1 \
bash scripts/experiments/gsm8k/run_base_attention_backend_comparison.sh \
  --logical-split final-test \
  --limit 32 \
  --reference-backend flash_attention_2 \
  --candidate-backend sdpa \
  "$PHASE1_DIR" "$E0_DIR"
```

该诊断已经满足 SDPA baseline 质量与 native/cache parity 要求；当前 gate 与 side-KV 已固定迁移到
SDPA。

服务器环境文件只保留输出目录和可选 GPU：

```bash
cp scripts/experiments/gsm8k/e1.server.env.example \
  scripts/experiments/gsm8k/.e1.server.env
```

重建 SDPA artifact（旧 eager 目录不可复用）：

```bash
MEMGEN_RUN_TAG=risk-sdpa-v2 \
bash scripts/experiments/gsm8k/run_entropy_risk_gate.sh "$PHASE1_DIR"

MEMGEN_RUN_TAG=e0-sdpa-v1 \
bash scripts/experiments/gsm8k/run_e0_experience_memory.sh "$PHASE1_DIR"
```

risk report 已通过 held-out AUC 门槛，`e0_final_report.json` 已满足
`attention_implementation=sdpa`、`formal_e0_passed=true`。

SDPA risk qualification 固定要求 train/holdout 的 recovery、persistence 各至少 40 个事件；该门槛只判断
诊断样本是否充足，不参与 prototype、entropy threshold 或 risk threshold 的拟合。

```bash
source scripts/experiments/gsm8k/.e1.server.env

MEMGEN_RUN_TAG=e1d-full-system-v2 \
bash scripts/experiments/gsm8k/run_e1d_full_system.sh \
  --logical-split calibration-val \
  --offset 0 \
  --limit 100 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT"
```

已完成的官方 GSM8K `final-test` 全量冻结评测命令：

```bash
MEMGEN_RUN_TAG=e1-sdpa-final-full-v1 \
bash scripts/experiments/gsm8k/run_e1d_full_system.sh \
  --logical-split final-test \
  --offset 0 \
  --limit 0 \
  "$PHASE1_DIR" "$E0_DIR" "$RISK_ARTIFACT"
```

`--limit 0` 表示选择该 split 的全部样本。assignment 阶段只读取 test question；test answer 只在冻结
assignment 后的 evaluation 阶段用于评分。artifact 记录 `dataset_split=test` 和
`evaluation_role=final_evaluation`。

如需重建风险 artifact，只运行风险分类编译，不执行 residual 干预：

```bash
bash scripts/experiments/gsm8k/run_entropy_risk_gate.sh "$PHASE1_DIR"
```

输出包括：

- `assignment_manifest.json`：answer-blind gate、query 和 matched ID；
- `evaluation/results.jsonl`：vanilla/matched 逐题结果和逐 token side-KV trace；
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

## 6. 下一阶段

1. 先对 final-test artifact 做只读 discordant audit：交叉分析 14/14 strict flips、8/13 answer flips、
   12/13 format flips 与 memory ID、payload、BM25 score/margin、trigger 位置、attention mass 和 KL；该分析
   只能用于形成假设，不能用于选择配置。
2. 在 calibration/dev 上冻结同一 assignment 和 memory ID，对比 no-memory、matched payload text 与
   matched side-KV，直接判断瓶颈属于经验/检索还是表示通道。
3. 若 matched text 有效而 side-KV 无效，优先研究 query-conditioned 编译、多层 side-KV 或学习式 memory
   adapter；若 matched text 也无效，再改进经验抽象与检索。
4. 在内容与通道得到正证据前，不调整 `log(10)` bias，不优先优化 gate timing。

后续仍不新增 Teacher/Pro 调用，不恢复 residual-vector 路线，也不把完整系统的工程可运行性表述为任务
收益。final-test 结果可以报告冻结系统的实际表现，但在当前双条件端到端设计中，不能单独
归因于 BM25 匹配质量。
