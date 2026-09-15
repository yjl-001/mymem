# V4.3：固定相似度 selector，比较记忆消费时机

运行：`bash test.sh timing`。默认自动断点续跑；只查看计划用
`bash test.sh timing --plan-only`，完成后只读复核用
`MEMGEN_V43_VALIDATE_ONLY=1 bash test.sh timing`。

输入沿用 selector 实验的环境变量；`MEMGEN_V43_SELECTOR_ROOT` 指向已完成的
`v4_3_question_selector` 目录，`MEMGEN_V43_TIMING_ROOT` 指定新输出目录，默认
`$MEMGEN_V4_OUTPUT_ROOT/offline/v4_3_memory_timing`。需要原有 Bank、side-KV、
prefix equivalence、split manifest 和 risk artifact，以验证来源；不会重建卡片、训练
selector、重新编译或修改已有缓存。必须在同一 reasoner、tokenizer、运行库版本和设备配置下运行。

## 固定条件与分支

使用原 selector 实验的全部 eval 题及其已保存的 `semantic_bank` 决定；不重调阈值。
这是已观察过的评估题上的诊断实验，不是新的独立测试；不加载 official test。

| 分支 | 行为 |
| --- | --- |
| baseline | 复用原题 no_memory 分支 |
| native_prefix_kv | 复用相似度选中 Bank 的原生全层 prefix 分支；弃用时复用 baseline |
| prompt_end | 编码完整无记忆题目提示后，在 assistant 内容开头输入记忆，再生成 |
| entropy_gate | 无记忆起步，在首个合格 gate 位置输入同一份记忆，再继续生成 |

额外报告训练集选定的固定 Bank 结果，作为参照，不新增生成。
若原 eval 为 100 题、其中 68 题选中记忆，仅需新增 136 次生成。
弃用记忆的题目全部复用 baseline，不探测 gate。

Gate 复用冻结 V3.4 token-risk artifact 的 `EntropyHysteresisGate`：原 observer 的末层注意力熵
达到 high threshold 且 layer 24 隐藏状态计算的 risk 严格高于 threshold。从第一个已生成 token 开始逐 token
观察；出现 boxed/fbox/final answer/answer is 标记后不再触发。没有固定分隔符限制。
每题最多注入一次；不在 memory token 上探测 gate。低熵 rearm/关闭规则在单次、
保留到结束的实验中不参与消费，但 artifact 的原始配置完整保留。

## 延迟消费的准确含义

两个延迟分支共享固定包装：

```text
\n[Reusable reasoning guidance]
Use this reusable reasoning guidance only when applicable:
{原始卡片 descriptor}
[End reusable reasoning guidance]
Continue solving the problem.
```

这段输入追加在当前 assistant 内容中，不插入新的 ChatML role，不重新编码之前的题目或推理。
完整包装 token 经 reasoner 在当前上下文中前向计算，产生全层、原生位置的 K/V。
这不是直接拼接离线 prefix KV。KV 保留到生成停止，无旧 side-KV 的 score bias、
单层限制或 32 步关闭。gate 消费时，先处理触发 token，然后追加记忆，由最后一个
记忆 token 的 logits 预测后续推理 token。

当前 prefix 参照仍采用原来的 system-message 包装，因此 prefix 与延迟分支的比较
同时涉及位置和包装差异，不能声称是严格的单一位置消融。prompt_end 与 entropy_gate
则使用完全相同的延迟包装。保持 prefix 参照不变，是为了保留已验证的运行配置。

最多生成 1024 个真实 completion token（含注入前后的推理），外加记忆输入 token。
所有题目在生成前检查最大上下文，不截断、不按结果排除题目。评分只看生成的推理和
答案，记忆包装和卡片文本不进入 completion；EOS/完整 boxed 答案检测也只看生成文本。

## 产物和解释

`profile.json` 绑定原 selector/profile/report、逐题选择及参照结果哈希、risk 文件、
新代码哈希、包装和实验配置。每题先写 `decision.json`，再写四个分支文件。
生成文件逐分支原子保存，可续跑；发现来源、配置、代码漂移时拒绝混用产物。

`brief_summary.json` 是需要贴回的精简结果：四分支 accuracy/gain/harm、固定 Bank
参照、实际 gate 激活数与平均注入位置，以及触发前轨迹/未触发轨迹一致性诊断。
`report.json` 另含相对 native prefix 和 fixed 的配对比较，以及选中记忆、gate 激活、
选中但未触发三个子集的结果。子集由 gate 轨迹定义，只作诊断，不当成独立测试。

解读时先看全部 eval 的准确率，再看 gate 是否实际激活、触发位置和 gain/harm。
无收益可能来自介入时机、包装、记忆适用性或上下文消费本身；不能直接归因于某一层。
本实验尚不研究离线 KV 直接复用、压缩、减少层数或多次注入。
