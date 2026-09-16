# V4.3 有效记忆的检索排名诊断

研究问题：相似度 top-1 没有修复的题，有效记忆是否已在第二、第三名，还是排得更靠后？

固定既有 11 个 primary Bank、完整卡片 native prefix KV、题目向量和三种 key 向量。
只读已有 train/tune 的逐题 × 全 Bank utility 表，不重新编码、不生成答案、不改 selector，
不读取旧 eval 或 final-test 的结果。本轮使用已有模型环境即可，无需 GPU 或模型权重。

## 执行

沿用之前的路径变量：

```bash
bash test.sh retrieval-coverage
```

输入是 `MEMGEN_V43_SELECTOR_ROOT` 和 `MEMGEN_V43_SIMILARITY_ROOT`。
输出为 `MEMGEN_V43_COVERAGE_ROOT`，默认
`$MEMGEN_V4_OUTPUT_ROOT/offline/v4_3_retrieval_coverage`。
V4 root 未设置时沿用 `$MEMGEN_OUTPUT_ROOT/v4`，最终默认 `/data/memgen-runs/v4`。

重复运行会核对相同结果并保留文件，不覆盖既有产物。只读验证：

```bash
MEMGEN_V43_VALIDATE_ONLY=1 bash test.sh retrieval-coverage
```

脚本认证 source/study profile、selector、report、各 key 向量及逐题 action 的内容哈希，
核对训练/调参 utility 表与此前拟合和报告使用的数据一致。重新计算的三种 key 始终
top-1 结果，必须逐项复现上一轮报告中的准确率、gain/harm、选择分布及 token 统计。
旧消费者、卡片构造和相似度研究文件均不修改，避免破坏已有 artifacts 的实现哈希。

## 定义

**可修复题**：无记忆答错，但至少一个 Bank 能使其答对。

`repairable_count` 是所有覆盖率的分母。无记忆答错且所有 Bank 都答错的题单独记录为
`unrepairable_baseline_wrong_count`，不混入覆盖率分母。

每种 key 对所有 Bank 按余弦相似度降序排列；同分按 bank_id 升序。
不使用弃用阈值，也不把 `no_memory` 加入 Bank 排名。

- **修复覆盖率 @k**：可修复题中，前 k 个候选至少包含一个正确 Bank 的比例。
- **首次成功排名**：每道可修复题中，排得最靠前的正确 Bank 的名次。
- **@k 相对 top-1 的额外修复数**：若能事后挑出正确 Bank，扩大候选可以多修复多少题。
- **随机参考**：从 B 个 Bank 均匀、不放回取 k 个，至少命中一个正确 Bank 的精确期望。
  一道题有 m 个正确 Bank 时，命中概率为 `1 - C(B-m,k)/C(B,k)`；对可修复题取平均。
  这不是随机重跑模型，也不是随机策略的真实准确率。
- **oracle@k**：事后在 `no_memory` 与前 k 个 Bank 中挑最好结果的准确率。
  等于 `(无记忆正确题数 + 前 k 名覆盖的可修复题数) / 总题数`。

分母为零时返回 null，而不是 0。完整报告覆盖 k=1…B，并报告候选边界同分数量；
精简报告保留 @1、@3、@5、@B 和首次成功排名分布。

**Oracle 使用了答题结果，不可部署，也不代表重排器实际能达到的准确率。**
尤其 oracle@1 也不等于实际 top-1 准确率：oracle 可以利用答案信息选择无记忆来避免伤害。
没有为 oracle 输出 token 成本，因为这不是一个实际可执行的选择策略。

## 伤害分解

对无记忆答对但 top-1 Bank 答错的题，分别统计：

- 前 k 个候选中是否存在能保持正确的 Bank。
- 所有 Bank 是否全部答错，此时改善 Bank 排序也无法保护答案，需要弃用。

这只是离线区分排序与弃用的改进空间，不会把答案信号带入在线 selector。

## 如何读结果

优先看 tune，同时对照 train 的趋势；两组均属于已使用过的 calibration 数据，
不是新的独立最终测试，不据此重新报告 final-test 成绩。

1. @1 低而 @3 大幅提高：有效记忆可能已在附近，值得研究 top-3 重排。
2. @3/@5 仍低：现有表示对有用记忆的排序不足，应研究检索表示或 key 的语义匹配。
3. @3 高但与随机候选接近：不能据此证明相似度排序有效，可能许多 Bank 都能修复同一题。
4. 全 Bank 可修复题本来就少：需要同时考虑记忆内容的有效性，单改 selector 的空间有限。

程序不设置任意的“足够高”阈值，也不自动根据 tune 结果拟合新策略。
实际路线结合覆盖率、随机参考、可修复题总量及伤害分解判断。

## 输出

- `brief_summary.json`：train/tune 的可修复题总数、三种 key 的覆盖率、随机参考、
  首次成功排名和伤害分解。只需贴回这一份。
- `report.json`：完整 k 曲线、原 top-1 准确率及 token 统计、每道题的排名和正确 Bank 位置，
  以及来源哈希和诊断口径。

本地测试覆盖明确构造的 rank=1/3/5、无法修复、所有 Bank 有害、零分母、多正确 Bank、
相似度完全同分、随机不放回概率，以及真实磁盘 artifacts 的认证、重复运行和只读验证。
真实 11-Bank 结果需在保存原始 utility 表的服务器上运行命令取得。
