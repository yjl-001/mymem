# V4.3 全量 final-test 三组冻结评估

运行 `bash test.sh final-test`，自动断点续跑。查看完整样本计划用 `--plan-only`；
完成后只读复核用 `MEMGEN_V43_VALIDATE_ONLY=1 bash test.sh final-test`。
输出默认 `$V4_ROOT/final_test/v4_3_frozen_three_branch`，可用
`MEMGEN_V43_FINAL_TEST_ROOT` 指定独立目录。其余路径沿用 selector/gated-prefix 的环境变量。

读取原 recovery split manifest 的全部 `logical_split=final-test` 条目，即 GSM8K
官方 test。验证 dataset revision、题目/答案哈希、test 长度和连续 source_index 全覆盖，
不抽样、不筛题、不排除长题；超出上下文时显式报错。没有 `--limit` 或调参选项。
原 Bank 和 selector 的 train/tune/eval 产物仅用于验证冻结来源，不作为 final-test 结果。

## 三组

1. `baseline`：无记忆。
2. `native_prefix_kv`：原相似度 selector（冻结 card/query 编码和阈值）选择 Bank，
   使用原 system 包装对应的完整全层 native prefix KV，从题目前开始消费。
3. `gated_prefix_kv`：同一题选同一个 Bank，原联合 entropy/risk gate 首次触发后，
   从下一次前向开始读取虚拟负位置的离线原生 KV；全层、无偏置、持续到结束，历史不重放。

不运行学习型 selector 或固定 Bank 对照，不重新生成卡片，不重训/重调任何参数。
原 selector 的 train/tune 拟合可能会由来源验证器重算以校验其冻结身份，不写回或更新模型。
真实 final-test 路由仅使用题目 feature 和原相似度规则。
每题先封存 `decision.json`，再分别保存三组原始 generation；三组都保存后，才用 gold 评分。
弃用记忆时两个记忆策略复用**同一道 final-test 题的新 baseline**，不借用历史诊断题结果。

## 准确率和 token 统计

每组报告：accuracy、correct、count、相对 baseline 的 gain/harm/net_gain 和配对检验。
同时报告生成 token 的 total、mean、median、min、max、p90，以及达到 1024 token 上限的题数。

生成数量严格等于 `len(continuation_token_ids)`：包括实际生成的 EOS；不含题目 token、
system 包装/卡片 prefix token，也不把直接可见的 KV 槽位当成新生成 token。
每组的数量是该策略全体测试题的输出长度总计，不是三组共享计算后的实际 GPU 执行量。
读取离线 KV 仍有 attention 计算开销，不能只用输出 token 数当作总计算成本。
报告另列 native prefix 记忆输入 token 总量和 gated 实际开放的记忆槽位数（每题计一次，
不按层或读取次数重复计数），与生成数量分开。

## 输出和续跑

`profile.json` 绑定完整测试集、冻结协议/模型/selector/Bank 来源与新 runner 代码。
每题目录包含 `decision.json`、三份未评分生成文件和 `scored.json`。
逐分支原子写入；中断时保留完成的生成，续跑跳过它们。
已经存在生成却缺少原选择的情况不能事后补造选择。
完整产物的 `--validate-only` 不加载 reasoner 或 test 数据集。

最终 `brief_summary.json` 即需贴回的结果：三组准确率和生成 token 统计、gate 触发率相关
计数、历史 KV/轨迹检查，以及 gated 相对 native prefix 的配对比较。
`report.json` 额外包含逐 Bank 的选择次数。官方 test 结果用于最终评估，不回流为阈值选择依据。
