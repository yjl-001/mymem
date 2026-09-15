# V4.3：gate 控制离线 native prefix KV 的可见性

运行 `bash test.sh gated-prefix`。沿用已完成 selector 的环境变量，自动断点续跑。
`--plan-only` 查看计划；`MEMGEN_V43_VALIDATE_ONLY=1 bash test.sh gated-prefix` 只读验证完整产物。
输出由 `MEMGEN_V43_GATED_PREFIX_ROOT` 指定，默认 `$V4_ROOT/offline/v4_3_gated_prefix`。
运行不依赖上一轮 timing 的结果目录，也不会改写它。需要原 selector 和 prefix equivalence
产物及其原始 Bank、split、risk 等来源文件；来源、版本或配置漂移时拒绝混用。

## 实验

同一批已观察过的 selector eval 题，同一份已保存的相似度选择和阈值，不重新训练/调参。
三个分支：

- `baseline`：复用原 no_memory 结果。
- `native_prefix_kv`：复用相似度选中 Bank 从题目前开始可见的原生 prefix 结果。
- `gated_prefix_kv`：从无记忆起步，首次合格 gate 后开放离线 prefix KV 读取。

原 fixed-from-train 的结果作为额外参照。弃用记忆的题目完全复用 baseline；原来
100 道 eval 中有 68 道选中记忆，因此只新增 68 次生成。原 reference 准确率不算新的复现。
这是 diagnostic reuse，不是 fresh held-out test，也不加载 official test。

## 记忆和位置

直接读取原来编译好的全层 native prefix KV，包括完整 system 包装 token。
不输入新的记忆文本，不重编译，不做当前上下文中的记忆 prefill。

原记忆位置为 `0..M-1`，将每层 post-RoPE Key 做统一 `R(-M)` 旋转，放在虚拟位置
`-M..-1`。所有 Value 原样复制。仅支持 default 固定频率 RoPE、full-attention Qwen2 SDPA；
动态/缩放 RoPE 或滑动窗口不静默套用这个公式。原记忆文件和 CPU 输入张量不修改。
题目与生成 token 继续使用 `0,1,...` 原生位置；记忆不占原生 position/cache 槽位。
生成前检查 `题目长度 + 记忆长度 + 1024` 的相对位置跨度，不截断或按结果排除题目。

## gate 和时序

原冻结 V3.4 联合判据：末层注意力熵达到 high threshold，且 layer 24 risk 严格高于
risk threshold。从首个已生成 token 开始逐 token 观察，答案标记后不再寻找触发。

gate 的前向计算及其采样结果保持原样；从**下一次尚未执行的前向计算**起，全层 Query
共同读取 `[native K; memory K]` 和 `[native V; memory V]`。两部分同一次 softmax，
原生 causal mask 保留，所有记忆位置可见，无额外 score bias/槽位归一化。
物理拼接只在 attention 读取中进行，`cache.update` 只写新增的真实 token。

若 gate 观察的是 generated token 的索引 `j`（从 0 开始），它同时产生下一个 token：

- 前 `j+2` 个已生成 token 保持 baseline 路径；
- 以 generated token `j+1` 为 Query 的下一次前向才首次读取记忆；
- 该次前向预测的 generated token `j+2` 才可能变化。

这是有意保留的边界，不重算触发 token。若触发前向已经产生 EOS、完整 boxed 答案或
耗尽预算，只报告 joint trigger，不虚报实际 activation。
最多开启一次，开启后不再 probe/rearm，也不在 32 步或低熵时关闭，保持到生成结束。

## 完整性与结果

运行时检查：

- 每次前向原生缓存仅增加一个真实 token；插入 token 数和重放 token 数均为 0。
- 开启前复制历史 KV 到 CPU，仅作审计；首次读取后及生成结束时逐层核对历史内容完全相同。
- 每个 active forward 在全部层各读取一次记忆；记录首次读取的逐层 memory attention mass。
- 未开启前不安装注意力包装，gate observer 使用原来的原生模型路径。
- 三分支的评分只使用生成 token；问题/答案哈希绑定原 dataset revision。
- 每题先保存冻结选择，再原子写入逐分支结果；续跑和 validate-only 验证来源及结果。

`brief_summary.json` 包含准确率/gain/harm、fixed 参照、gate 触发与实际开启次数、
首次读取位置、首次读取注意力质量和历史/轨迹检查。`report.json` 另含相对 prefix/fixed
的配对比较，以及 selected/activated/selected-but-not-activated 子集。

历史 KV 保留不意味着结果与始终可见的 prefix 等价：历史题目和推理没有受记忆影响。
本实验只检验后续读取离线原生记忆能否带来收益，不声称自动继承原 prefix 的准确率。
