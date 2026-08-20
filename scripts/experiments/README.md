# 一键实验脚本

每个正式训练或评测都必须同时包含：

1. `configs/experiments/<dataset>/<method>.yaml`：提交到 Git 的参数差异；
2. `scripts/experiments/<dataset>/run_<method>.sh`：提交到 Git 的一键启动脚本。

脚本负责：读取服务器本地环境、设置明确的 `run-id`、调用
`scripts/launch_experiment.py`。`run-id` 使用
`<dataset>_<method>_<key-settings>_<stage>_<timestamp>`，让输出目录能直接说明
实验核心。

## 一次性的服务器设置

```bash
cp scripts/experiments/server.env.example scripts/experiments/.server.env
```

编辑 `.server.env` 中的输出根目录、GPU 与 checkpoint 路径。它只包含服务器本地信息，
已被 Git 忽略。

之后每次运行只需同步代码并执行对应脚本：

```bash
git pull --ff-only origin main
bash scripts/experiments/gsm8k/run_entropy_calibration_sink4_q85.sh
```

需要依赖前一阶段产物时，后续脚本接受该产物的绝对路径作为参数，而不会猜测“最新”运行。

离线 bank 构造同样使用版本化的一键脚本。当前 GSM8K 预览见
[`gsm8k/build_teacher_bank_preview.sh`](gsm8k/build_teacher_bank_preview.sh)；它需要
未提交的 `DEEPSEEK_API_KEY`，且输出仅写入服务器的 `MEMGEN_OUTPUT_ROOT`。

正式的 verifier-backed Phase 1 bank 使用：

```bash
bash scripts/experiments/gsm8k/run_phase1_verified_bank.sh
```

它会依次冻结 split、采样 student rollout、用 GSM8K verifier 标记成功/失败、形成
同题 contrast、调用 Flash teacher、执行确定性审计，再由 Pro reviewer 独立复核。
确定性审计只对 provenance、verifier 绑定和 schema 等数据完整性负责；这类异常不参与
内容优劣判断，而是单独进入 `quarantined_bank_records.jsonl` 等待修复或重建。关键词、
相似度、格式描述和 teacher 自评等都只是语义警告，最终内容通过或拒绝由 Pro 的逐字段
证据审核决定。八个 bank 字段和五个 pair 属性全部 supported 才进入正式 bank；
存在 unsupported/contradicted 时拒绝，其余包含 partially supported 的记录进入
`deferred_bank_records.jsonl`。Reviewer confidence 仅作诊断，不参与路由。报告同时按
`experience_type` 统计各路由数量与通过率，用于检查精度优先筛选是否造成类别覆盖偏差。

Phase 1 verifier 同时保存严格任务奖励和诊断字段。缺少或损坏 `\\boxed{}` 仍然是
正式任务失败；若宽松诊断能确认自然语言中的最终数值正确，该 reference 会被归为
`format_compliance`，而不是被误写为推理失败。答案错误和混合/无法判定失败使用独立
experience type，供后续分簇编译。Teacher prompt 会收到 target/reference 两侧的完整
verifier 记录。确定性审计会检查失败类型一致性，并把 format-specific 描述问题作为
语义警告交给 Pro 判断。

脚本会复用已经完成的 student rollout，但每次都会廉价重建 typed experiences。旧版
Teacher/Reviewer 记录、模型或 provenance 已变化的记录会在 `--resume` 时自动丢弃并重新
生成，因此升级后无需重新执行 GPU rollout。审核路由规则变化时，来源与模型未变化的
首轮 Pro 结果会复用并重新路由。被淘汰的旧 JSONL 会先保存为同目录下带
`.stale-<UTC>.bak` 后缀的备份。

`ai_approved_bank_records.jsonl` 即正式 Phase 1 bank 输入；`ai_rejected`、`deferred`
和 `quarantined` 仅保留用于审计、误差分析或未来扩充稀缺类别。这里的 `AI review` 不得
在论文或实验报告中表述为人工审核。

当前 Phase 2 不再调用新的 AI。它只读取 `ai_approved_bank_records.jsonl` 作为质量闸门，并按
`experience_id` 连接 `verified_experiences.jsonl` 的原始 student 成功/失败轨迹。该阶段已经证明：
第 24 层的 hidden state 能区分高熵后的 `recovery` / `persistence` 风险（held-out ROC-AUC
`0.8053`），但 `recovery − persistence` 的全局 residual vector 不能作为有效动作；独立确认也
否定了它。**不得**继续调整该 vector 的 alpha、layer、符号或样本量。

以下仅为复现实验历史 Phase 2 probe 的命令，**不应用于继续调参或作为当前主线运行**：

```bash
bash scripts/experiments/gsm8k/run_phase2_entropy_risk_probe.sh \
  /absolute/path/to/gsm8k_phase1_verified-student-contrast_<tag>
```

默认使用完整 bank 做离线风险诊断、`dev-test` 的前 100 题做在线 smoke probe。风险诊断不
通过会在生成 vector 前停止；不要通过降低 AUC 门槛来强行进入在线实验。正式 vector 只使用
`answer_correctness` evidence；`format_compliance` 等类型保留给后续条件化/类型化实验，避免
格式监督淹没主推理向量。

确认时不改变任何干预参数：将 `.server.env` 中
`MEMGEN_PHASE2_RISK_EVAL_OFFSET=100`、`MEMGEN_PHASE2_RISK_EVAL_LIMIT=0`，只运行未看过的
`dev-test` 后缀。汇总会按 sample ID 对 real/control 的实际熵转移做 bootstrap CI；每个对照至少
需要 50 个配对事件，且 CI 上界小于 0 才通过。bank-heldout 同时报告 ROC-AUC、PR-AUC 与
persistence 正类比例，避免类别失衡下误读 PR-AUC。

历史 H3 local-action audit 也已经完成：虽然能形成 23 个 bank-train / 9 个 held-out action，且
方向不共线，但 held-out raw hidden-state top-1/top-2 routing margin 极低，不能稳定选择具体 action。
该脚本保留为只读负向诊断，不构成后续线上实验入口。

```bash
bash scripts/experiments/gsm8k/run_phase2_conditional_action_audit.sh \
  /absolute/path/to/gsm8k_phase1_verified-student-contrast_<tag>
```

当前主线改为“风险触发 → 语义检索经验内容 → side-KV integration”。它继续复用 Phase 1 的
`ai_approved + answer_correctness` records，但这一次 Teacher/Pro 已审核的经验抽象会成为**实际
memory payload**，而不是只做 Phase 2 的质量标签。payload 仅允许通用的：适用情境、成功策略/验证、
失败机制/警告信号；必须剔除原题、原始轨迹、`\boxed{}` 答案、原始 evidence quote 与实例特有数值。
不再产生新的 Teacher/Pro 调用。

下一轮尚未实现，因此目前没有新的服务器运行命令。实现顺序与实验门槛已固定：

1. E0：审计 payload 无泄漏，并验证 layer-24 canonical side-KV 能附加而不改写原 cache，且有非零
   memory attention mass；
2. E1：先用 observation-only pass 冻结每题首次风险触发 boundary 和 matched memory id，再比较
   `vanilla`、`gate-observation-only`、`matched-memory` 与同预算 `shuffled-memory`；所有条件使用
   相同 sample、prefix、触发位置和单次预算；
3. 仅当 matched memory 相比 shuffled/gate-only 有配对因果证据且格式不受损时，才做 target/reference
   字段消融和随机位置时机消融，最后才进入 final-test。

下一 boundary 熵、logits KL、memory attention、检索分数和延迟均会记录为诊断；主要问题是模型是否
利用了匹配的经验内容并改善最终任务表现，而不是是否单纯降低熵。完整规范见
[`experience_calibrated_steering_plan.md`](../../docs/codex/experience_calibrated_steering_plan.md)。
