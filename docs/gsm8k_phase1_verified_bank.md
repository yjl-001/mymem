# GSM8K Phase 1：verifier-backed rollout bank

本阶段用冻结的原始 student reasoner 在 `bank-source` 上进行随机采样，并使用项目现有
的 `data.utils.math_utils.compute_score` 确定性 verifier 将每条轨迹标记为
`verified_success` 或 `verified_failure`。只有同一道题同时拥有成功和失败 rollout 时，
才会形成正式 target/reference contrast。

## 一键运行

在服务器的未跟踪 `scripts/experiments/.server.env` 中配置输出目录、GPU、student revision
和 DeepSeek API key，然后运行：

```bash
bash scripts/experiments/gsm8k/run_phase1_verified_bank.sh
```

默认设置把 GSM8K train 固定划分为 6000 条 `bank-source`、1000 条
`calibration-val` 和剩余 `dev-test`；官方 test 全部保留为 `final-test`。manifest 只保存
索引和问题/答案 hash，不复制数据内容，并自动检查四个逻辑 split 的问题 hash 交集为零。

正式运行默认对每条 `bank-source` 采样 8 个 rollout。若只验证服务器环境，可在
`.server.env` 临时设置较小的 `MEMGEN_PHASE1_ROLLOUT_SAMPLE_LIMIT` 和
`MEMGEN_PHASE1_TEACHER_LIMIT`；这些限制会进入产物统计，不能冒充完整实验。

## 产物

每次运行在 `MEMGEN_OUTPUT_ROOT/banks/gsm8k/<run-id>/` 生成：

| 文件 | 内容 |
|---|---|
| `split_manifest.json` | 固定 split、source index、内容 hash、manifest hash |
| `student_rollouts.jsonl` | 完整 CoT、verifier、模型/分词器 revision、采样配置和 seed |
| `rollout_summary.json` | 成功/失败数量及 rollout artifact hash |
| `verified_experiences.jsonl` | 同题 verified success/failure contrast |
| `teacher_reflections.jsonl` | teacher 抽象、teacher self-quality mark 和完整 provenance |
| `approved_bank_records.jsonl` | 通过自动 provenance/quality gate 的正式记录 |
| `rejected_bank_records.jsonl` | 被拒记录及明确原因 |
| `audit_report.json` | 数量、hash、拒绝原因和人工复核状态 |
| `human_review_sample_30.jsonl` | 30 条人工一致性复核清单 |

## 验收边界

自动 gate 会拒绝非 `bank-source`、非真实失败、episode/provenance 不匹配、字段缺失、
teacher 自评不支持、target/reference 高度等价和包含明显实例数字/公式的记录。它不能取代
人工事实检查，因此 `audit_report.json` 初始状态必然是 `pending_manual_review`。人工复核
30 条达到至少 90% 一致率后，Phase 1 的数据验收才算完成。

人工填写四个布尔字段后，运行以下命令生成带 hash 的最终验收结果；未填写完整、少于
30 条或一致率低于 90% 时脚本会失败：

```bash
python scripts/finalize_phase1_human_review.py \
  --review <run-dir>/human_review_sample_30.jsonl \
  --audit-report <run-dir>/audit_report.json \
  --output <run-dir>/human_review_result.json
```

Preview 脚本产生的 `teacher_inferred` reference 仍只用于 schema 检查，自动审计不会把它
纳入正式 approved bank。

## Teacher 代理恢复

Teacher client 在整个 bank 构造期间复用同一个 HTTP session，减少企业代理的 HTTPS
CONNECT 次数。代理隧道或认证返回 407 时，默认按 30、60、120、240 秒退避，随后每次
最多等待 300 秒，共额外重试 20 次（约 90 分钟）。等待日志不会打印原始代理异常、
代理 URL、API key 或可能嵌在 URL 中的代理凭据。

这些参数可在未跟踪的 `.server.env` 中调整：

```bash
export MEMGEN_TEACHER_PROXY_RETRIES="20"
export MEMGEN_TEACHER_PROXY_RETRY_INITIAL_SECONDS="30"
export MEMGEN_TEACHER_PROXY_RETRY_MAX_SECONDS="300"
```

超过长退避窗口后任务仍会安全退出。每条 teacher record 已即时写盘；代理恢复后使用同一
`MEMGEN_RUN_TAG` 重启，一键脚本会跳过已完成的 experience ID 并继续。
