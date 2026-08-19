# Experience-Calibrated Entropy-Gated Steering：实施计划与验收标准

## 1. 目标与边界

本计划验证一个训练无关、可审计的 latent-memory 原型：离线从**已验证**的
历史推理经验中提取 target/reference steering vectors；在线仅在推理边界且
sink-masked attention entropy 较高时，将相关向量以受控 residual intervention
施加给冻结的 reasoner。

它不是原始 MemGen Weaver 的替代结论，而是一个低工程复杂度的实验路线。MemGen
保留生成循环、候选 boundary、预算和评测框架；FlashMem 风格熵负责候选触发；
SEAL-style vector 是第一版的注入接口。后续可在不改变经验生命周期和选择器的
前提下，把 vector integrator 替换为 Memory Inception（MI）side-KV integrator。

本计划的核心可检验问题是：

> 相对于不注入、随机注入和随机向量，来自真实 target/reference 经验的 steering
> vector 能否在高熵 reasoning boundary 带来稳定且非破坏性的收益？

截至 2026-08-19，Phase 2 已以独立确认集否定其中最简单的一种答案：全局
`recovery − persistence` 状态中心差可以识别局部风险，但不是有效的 residual
**动作**。这不否定 Phase 1 bank 或风险识别器；下一步问题收窄为：在 gate 已识别
“可能持续发散”的当前状态后，能否从状态相近、已验证的经验中检索到一个可用的
条件化 action，而不是套用一个全局均值方向。

本计划不宣称 zero-shot 或“无外部监督”：bank 构造使用训练期的、已验证的
rollout；教师模型只在离线阶段参与反思、归类和质检。

本计划中 MemGen、FlashMem、SEAL 和 MI 的精确机制、训练边界与项目映射，统一见
[前置方法：MemGen、FlashMem、SEAL 与 Memory Inception](method_prerequisites_memgen_flashmem_seal_mi.md)。
下文仅引用其接口职责：MemGen 提供控制框架，FlashMem 提供候选熵门控，SEAL 提供
Phase 2 的 residual-vector MVP，MI 是 Phase 5 的 side-KV 升级路径。

## 2. 总体架构

```text
bank-source 训练样本
        │
        ├─ frozen student rollout ── verifier ──► verified success / failure episodes
        │                                             │
        │                                  strong teacher: abstract / cluster / pair
        │                                             │
        ▼                                             ▼
  target/reference memory records ──► student-space compiler ──► latent artifacts
                                                                    │
在线 prompt + 已生成 CoT ──► boundary ─► sink-masked entropy gate ─► retrieve ─► integrate
```

在线控制由三个可独立替换的接口组成：

```python
artifact = compiler.compile(memory_record, frozen_reasoner)
selected = selector.select(runtime_state, artifact_store)
runtime_state = integrator.apply(runtime_state, selected)
```

第一版的具体实现分别为：

- `SteeringVectorCompiler`：target/reference evidence → student hidden-space vector；
- `EntropyRetrieverSelector`：boundary + entropy + relevance retrieval；
- `ResidualVectorIntegrator`：在选定层的 boundary hidden state 加入受控向量。

MI 阶段仅替换为 `KVBankCompiler` 与 `SideKVIntegrator`；经验数据、检索、门控、
日志和实验切分保持不变。

## 3. 数据切分与反泄漏约束

每个数据集必须固定以下互不重叠的 split：

| Split | 用途 | 禁止事项 |
|---|---|---|
| `bank-source` | student rollout、verifier、教师归纳、构造 vector | 不用于阈值/超参选择 |
| `calibration-val` | 选熵阈值、layer、alpha、retrieval 阈值 | 不写入 bank，不供教师查看 |
| `dev-test`（可选） | 完整 pipeline 的小规模调试 | 不用于最终报告 |
| `final-test` | 一次性最终评估 | 不供 bank、教师、任何超参选择使用 |

约束：

1. 教师只读取 `bank-source` 的轨迹及 verifier feedback；绝不读取 test 问题或答案。
2. `verified_success` 与 `verified_failure` 必须由确定性 verifier/reward 标记；教师
   不能覆盖这一标签。
3. 当前 GSM8K preview 的 `teacher_inferred` reference 仅用于 schema 检查，不可用于
   正式 contrastive vector。
4. 每条 memory record 必须保存 source split、episode id、model revision、rollout
   configuration 和生成日期，以支持审计与复现。

## 4. Latent artifact 定义

### 4.1 原始经验记录

```text
context + trajectory/actions + outcome/reward + verifier feedback
  → target: situation / decision / verification / applicability boundary
  → reference: competing pattern / failure signal / failure mechanism / non-reuse boundary
```

target 和 reference 的价值不是“正负文本装饰”，而是将可复用成功行为与已经观察到
的失败模式区分开。reference 必须是 verified failure，或单独标记为不可用于正式
实验的 `teacher_inferred`。

### 4.2 条件匹配的 entropy-risk artifact

最小探针只保存固定 layer `l=24` 的一个 artifact：

```text
memory_id, cluster_id, layer, boundary_definition,
vector S[l], recovery/persistence prototypes, vector_rms, evidence_count,
reasoner_model_revision, tokenizer_revision,
source_episode_ids, target/reference provenance,
construction_config, calibration_statistics
```

向量由**目标 student 模型自身**生成，不能直接使用强教师的 hidden state。离线构造必须
与在线 gate 的状态条件匹配：仅保留 sink-masked attention entropy 高于 bank-source 分位数
阈值 \(\tau_H\) 的 reasoning boundary。强教师/Pro 不参与 Phase 2 的新标注。

高熵后下一 reasoning boundary 下降到低阈值 \(\tau_L\) 的状态记为 `recovery`，否则记为
`persistence`。此标签同时适用于成功 target 和失败 reference，并以四格表报告；不能预设
最终答案结果与局部熵转移一一对应。bank 先按 `experience_id` 切分，原型仅由 bank-train
构造，是否存在可用风险轴由 bank-heldout 的 ROC-AUC 决定。未来 boundary 仅用于离线标签，
在线触发始终只使用当前 state。

\[
S_l=\operatorname{Mean}(h_{l,t}\mid\mathrm{recovery,bank\text{-}train})
-\operatorname{Mean}(h_{l,t}\mid\mathrm{persistence,bank\text{-}train}),
\]

\[
R(h_t)=\cos(h_t,\mu_{\mathrm{persistence}})-\cos(h_t,\mu_{\mathrm{recovery}}).
\]

其中 \(t'\) 是 \(t\) 后的下一 reasoning boundary，\(R(h_t)>0\) 表示当前状态更像持续
发散。该假设不是“高熵等于错误”，而是：当前 state 可以预测局部 persistence 时，它才是
值得考虑干预的候选。风险识别不推导任何全局方向就是有效动作；动作的因果有效性必须单独
测试。若证据数不足或 held-out AUC 不达标，方法必须停止，不得用额外 AI 补标或继续调参。

### 4.3 历史全局注入候选（已被否定）

以下 `S` 是 Phase 2 用于证伪的全局候选，不再允许作为正式在线方法。它保留在此处
仅为了可审计地记录已检验过的假设与接口；否定结果见 Phase 2 结果小节。

使用当前 state 的尺度进行归一化，而非直接使用无约束的均值差：

\[
\hat S_{k,l}=S_{k,l}/\operatorname{RMS}(S_{k,l}),
\]

\[
H'_{l,t}=H_{l,t}+g_E(t)\,g_R(t,k)\,\alpha\,
\operatorname{RMS}(H_{l,t})\hat S_{k,l}.
\]

- \(g_E\)：sink-masked entropy 的软门控；
- \(g_R\)：当前 state 的 persistence-risk 门；本最小探针中为 \(\mathbb{1}[R(h_t)>0]\)，
  而不是检索或另一个训练模型；
- \(\alpha\)：本最小探针预先固定为 `0.05`，不按任务正确率搜索。

注入位置为选定 Transformer layer 的 boundary token hidden state，影响后续层与
next-token logits；不插入 token，也不重建既有 KV cache。

## 5. 阶段计划

### Phase 0：冻结基线、数据分区与观测能力

**目的**：让之后任何收益都可归因，而不是来自 split 泄漏或不一致的生成配置。

**工作项**：

- 固化 GSM8K 的 `bank-source`、`calibration-val`、`final-test` 清单；
- 固化 model checkpoint、tokenizer、generation config、seed；
- 完成原始模型、当前 entropy-only 模型的基线结果；
- 对每个 delimiter 记录 sink-masked entropy、token 位置、生成长度和答案结果。

**产物**：版本化 split manifest、baseline JSONL、entropy calibration artifact、运行脚本。

**验收标准**：

- split 交集为零，且检查脚本自动失败而非仅人工确认；
- 相同 seed/config 的重复运行在答案和 trace 上一致；
- entropy trace 覆盖率 100%，能回溯到每一个候选 boundary。

### Phase 1：构造 verifier-backed rollout bank

**目的**：获得真实成功与失败证据，替代 `teacher_inferred` reference。

**工作项**：

- 在 `bank-source` 上以固定采样配置运行 student rollout；
- 用 GSM8K answer verifier 标记每条 rollout；
- 保留成功、失败、奖励、完整 CoT、随机种子及 generation config；
- 用低成本 Flash teacher 做跨 episode 经验抽象、失败原因分析、经验簇建议和质量标记；
- 用独立 Pro reviewer 对照原始轨迹、verifier 和 teacher bank 做第二票审核；
- provenance/verifier/schema 异常只进入独立 quarantine，不作为内容拒绝；所有语义启发式仅作警告，由 Pro 的结构化证据审核决定内容路由；
- Pro 对八个 bank 字段和五个 pair 属性逐项给出 supported/partially-supported/unsupported 证据；含 partial 的记录进入 deferred，confidence 只作诊断；
- 每个正式 reference 均绑定至少一个 `verified_failure` rollout。

**产物**：原始 rollout JSONL、verified experience JSONL、教师反思 JSONL、审计报告。

**验收标准**：

- 所有正式 vector evidence 都能追溯到 verifier 结果；
- `teacher_inferred` 与 `verified_failure` 在 schema 和下游过滤中严格区分；
- 全部 teacher records 都有独立 Pro review 和可追溯的审核结论；
- 正式 bank 只包含所有关键 assessment 均为 supported 的记录；deferred、rejected 与 quarantined 均保留可追溯证据但不参与 vector 编译；
- 出现自相矛盾、事实错误或 target/reference 等价的记录必须可被 Pro 审核拒绝并保留证据。

### Phase 2：最小化的 entropy-risk probe

**目的**：先区分“高熵但自然恢复”与“高熵后持续发散”，再验证单次定向 residual
intervention；不再同时搜索 vector 类型、layer、强度、门控斜率与注入预算。

**工作项**：

- 冻结 `ai_approved` Phase 1 bank，不新增 teacher、reviewer 或人工归因；正式 compiler 只重放
  对应 `verified_experiences` 的原始成功/失败 student trajectories；
- 在第 24 层、每个 `\boxed` 前 reasoning delimiter 重放 sink-masked attention entropy；高/低
  阈值只由 bank-train 分位数确定；
- 对 target（verified success）与 reference（verified failure）都标注高熵后的
  `recovery/persistence`，报告完整四格表，而不是只保留其中对角线；
- 以 `experience_id` 分割 bank-train / bank-heldout，从 train 构造 recovery/persistence 的状态
  原型，以 held-out ROC-AUC 验证当前 state 是否可分；AUC 不达预设门槛则停止；
- 只有诊断通过，才构造单一 `recovery − persistence` state vector，并以
  `cos(h,persistence)-cos(h,recovery)>0` 作为在线风险门；
- 在线固定第 24 层、`alpha=0.05`、soft-gate slope `0.10`、每轨迹最多一次；只在**第一个**
  高熵边界作是否注入的决定。

**必须对照**：

1. vanilla；
2. entropy + risk gate 但不注入；
3. entropy + risk gate 注入真实 vector；
4. 相同门控下的同范数随机 vector；
5. 相同门控下的反转 vector。

**产物**：四格风险诊断报告、带 risk prototype 的 `SteeringVectorArtifact`、按 sample ID
对齐的单次 intervention trace、`dev-test` probe 汇总。

**验收标准**：

- held-out bank ROC-AUC 必须达到预设门槛（当前 `0.60`），且 train/held-out 两类事件均不少于
  50；同时报告 PR-AUC、persistence 正类比例及其相对比例的提升；否则不进入在线阶段；
- vanilla 与 entropy-only completion 必须完全一致；真实、随机、反向 vector 的首个决策前缀
  必须一致；
- 真实 vector 在实际注入后到下一个 candidate boundary 的 entropy 变化，必须优于 entropy-only、
  同范数随机 vector 与反转 vector；该 entropy 必须从携带先前注入影响的 causal KV cache
  continuation 读取，不能由相同 token 文本重新无干预 forward 得到；该后果指标不参与在线触发；
- 每个 entropy 对照都必须按 `sample_id` 配对，实际配对事件不少于 50，且
  `ΔH_real − ΔH_control` 的 95% bootstrap CI 上界小于 0；只比较均值不构成通过；
- 若先运行了 `dev-test` 的 smoke 前缀，确认实验只能使用未看过的剩余 offset，不能重新将已看过
  的样本混入确认统计；
- 格式正确率不低于 vanilla，且没有 disabled/超限注入；准确率仅在上述机制标准满足后解释；
- 发生 NaN、格式崩坏或超出扰动上限时自动禁用该次注入并留下 trace。

#### 已确认的 Phase 2 负结果（2026-08-19）

确认实验固定使用正式 `answer_correctness` Phase 1 bank、第 24 层、85% 高熵阈值、
风险门 `R(h)>0`、安全归一化后 `alpha=0.05`，并且每个样本只在首个候选边界最多注入一次。
这些配置在确认前已冻结；它们不是按确认集正确率挑选的。

**Stage A：风险识别通过。** bank 按 `experience_id` 切分后，held-out 的风险分数
`cos(h, persistence) - cos(h, recovery)` 得到 ROC-AUC `0.8053`、零阈值 balanced accuracy
`0.7158`、PR-AUC `0.9614`。persistence 正类占比为 `0.8706`，因此 PR-AUC 相对正类比例的
提升只有 `0.0909`，不能将很高的绝对 PR-AUC 误读为同等强度的增益。四格统计也确认了
局部熵转移并不等同于最终答案：reference 有 `542` persistence / `50` recovery 事件，
target 有 `603` persistence / `72` recovery 事件。结论仅是当前 hidden state 携带了
“下一边界更可能持续还是恢复”的可泛化**描述性风险信号**。

**Stage B：全局动作向量失败。** 在未用于前缀 smoke 的确认样本中，真实向量与每个
对照以 `sample_id` 对齐，均有 `140` 个配对触发事件。令
`d = ΔH_real − ΔH_control`，负值才表示真实向量使下一候选边界的熵下降更多：

| 对照 | `mean(d)` | 95% bootstrap CI |
|---|---:|---:|
| entropy-only | `+0.00417` | `[-0.00136, +0.01354]` |
| 同范数随机向量 | `+0.00556` | `[-0.00029, +0.01535]` |
| 反转向量 | `+0.00869` | `[+0.00023, +0.02074]` |

三项均未满足预注册的“CI 上界小于 0”标准；真实向量相对反转向量还出现了方向相反的、
有统计支持的退化。格式正确率也低于 vanilla。因而不能将首轮小样本的均值差解释为效果，
更不能推进到大规模或 final-test。

**因果解释与停止规则。**
`S = μ_recovery − μ_persistence` 是两个观测到的状态分布之间的差异轴；它能够帮助
分类“会恢复 / 会持续”，但没有识别“把一个持续状态变成恢复状态所需的残差动作”。
分类方向与可干预方向并不等价，且该全局差分把不同题型、推理阶段和失败机制平均在了一起。
因此，本项目关闭“全局 centroid state vector”分支：不得通过重试、增大样本、搜索 layer、
alpha 或符号来挽救它；反转向量也不得事后改作 treatment。风险 gate 可以保留，后续动作
必须以新的、预先声明的条件化假设和未看过的评估数据重新检验。

### Phase 3：条件化检索式 recovery action（设计，未实现）

**目的**：检验失败的全局均值方向能否被一个严格条件化的、检索到的经验动作替代。此处
尚是研究设计，不构成实现授权。

**最小假设 H3**：若在线状态 $q$ 同时满足高熵与 $R(q)>0$，则它与某个已验证 failure
persistence state 的 student-space 相似性，可选择与该经验相配套的 successful recovery
displacement；这个局部动作比任意同尺度扰动更能降低下一 candidate boundary 的熵。

对每条候选经验 $i$，仅当同一 `experience_id` 中同时存在可审计的 reference persistence
事件和 target recovery 事件时，构造：

\[
k_i=h^{\mathrm{reference}}_{24,t_i},\qquad
a_i=h^{\mathrm{target}}_{24,u'_i}-h^{\mathrm{target}}_{24,u_i},
\]

其中 $t_i$ 是 reference 的高熵后持续事件，$u_i\rightarrow u'_i$ 是 target 的高熵后
自然恢复事件。在线只做 student hidden-state 的 top-1 cosine 检索：

\[
i^*=\arg\max_i\cos(q,k_i),\qquad
H'_{24}=H_{24}+0.05\operatorname{RMS}(H_{24})\widehat{a}_{i^*}.
\]

这不是把相似性或 observation 当作因果证明：$a_i$ 仍是一个待证伪的 action candidate。
它相对全局 `S` 的唯一新主张是**状态条件化**——用相近的失败状态选择相应成功轨迹中实际发生的
局部状态位移，而不平均不同机制和推理阶段。

**预先固定的最小设计**：

- 仅用正式、`ai_approved` 的 `answer_correctness` records；不引入 Phase 2 teacher、Pro、
  文本 embedding 或人工归因；
- 保持 layer `24`、熵 85% 分位数、风险门 $R(q)>0$、首个候选边界、每样本最多一次、
  安全归一化 `alpha=0.05`，并以真实 causal KV-cache continuation 读取下一个边界熵；
- 检索仅在该 gate 已触发时发生；若候选池为空或 top-1 相似度低于**仅由 bank-train** 确定的
  阈值，则 abstain，不注入；不搜索 layer、alpha、符号、top-k 或阈值；
- 先按 `experience_id` 切分：bank-train 形成 action pool、prototype 与相似度阈值；bank-heldout
  在配置冻结后仅报告候选覆盖率及 top-1 相似度分布，不能反向改变 action 定义或阈值；在线
  开发/确认集均不得与这些构造样本重叠。

**必要对照与判定**：保留同题、同一首次候选 boundary、相同风险 gate、最多一次的四组记录：

1. `vanilla / entropy-only`：只记录触发与风险，不注入；两者的 completion 必须一致；
2. `retrieved action`：使用 $a_{i^*}$；
3. `reversed retrieved action`：使用 $-a_{i^*}$，检验方向依赖；
4. `query-shuffled action`：以另一个已触发样本的 query 选择 action，保留 action 分布和范数、
   破坏“当前状态—经验”的匹配关系，检验检索条件化而非任意扰动。

真实 action 必须在按 `sample_id` 配对、至少 50 个共同事件上，使
`ΔH_retrieved − ΔH_control` 对每一项动作对照的 bootstrap 95% CI 上界都小于 0；同时不能损害
格式。只有这些机制标准通过，才观察 accuracy。若失败，结论只能是“当前的检索键/动作定义无效”，
而非风险 gate 或整个 Phase 1 bank 无效。

**数据纪律与停止条件**：现有 `dev-test` 已用于 Phase 2 smoke 与确认，不能重用来选择 H3。
开始实现前必须审计 `calibration-val` 和剩余 evaluation split，预注册一个未看过的开发集及其
独立确认集；若不足以满足配对事件下限，则先新建 split manifest，而不是在既有结果上调参。
不在 H3 中进行聚类、文本检索、最终测试或 Phase 1 重审。

**设计产物**：经验级 key/action manifest、构造 split manifest、预注册配置、按 sample ID 对齐的
trace schema 和停止规则。只有这些设计审查完成且 split 审计通过后，才进入代码实现。

### Phase 4：冻结配置后的最终评估

**目的**：在从未参与任何构造或调参的 test 集上确认结论。

**工作项**：

- 锁定 split、bank、vector、layer、alpha、entropy threshold、检索配置；
- 只运行一次完整 final-test；
- 与 vanilla/entropy-only、条件化 retrieved action、反转 action、query-shuffled action、
  MemGen Weaver 对照；已被否定的全局 centroid vector 仅作为历史负结果报告，不重新运行；
- 报告 accuracy、token 数、trigger rate、延迟、memory artifact 大小、格式错误率和
  intervention diagnostics。

**验收标准**：

- final-test 不参与任何选择；配置文件和 artifact hash 可完全复现；
- 条件化真实 action 优于预注册的反转与 query-shuffled action 对照，且没有不可接受的
  格式/延迟回归；
- 结论按 effect size、置信区间和失败案例报告，不能只报告最优单次 seed。

### Phase 5：MI side-KV 升级决策

**进入条件**：Phase 4 表明条件化经验内容有真实收益，但 retrieved residual action 出现
明显过干预、条件错配或表征分布破坏。

**工作项**：

- 用同一份 memory records 编译 target/reference canonical pre-RoPE KV banks；
- 做 calibration-based layer/head 或 GQA KV-group selection；
- 实现 side-bank attention，不修改原始 prompt/history cache；
- 复用 Phase 3 selector，替换 `ResidualVectorIntegrator` 为 `SideKVIntegrator`。

**验收标准**：

- 与 vector 版本使用相同 bank、相同 split、相同门控预算；
- RoPE canonical-key 单元测试通过，bank 不绑定构造时绝对位置；
- 保留/改善正确率与稳定性，同时报告真实端到端 latency 和 KV footprint，而非只报告
  理论缓存占用。

## 6. 统一日志与诊断要求

每次 evaluation 的每个样本、每个候选 boundary 至少记录：

```text
sample_id, split, generated_token_index, boundary_type,
entropy_with_sink, entropy_without_sink, entropy_threshold,
gate_value, retrieved_memory_id, retrieval_score,
layer, alpha, delta_norm_ratio, logits_kl,
injection_applied, generation_length, final_reward, output_path
```

这些记录用于区分以下不同失败：熵门控错误、检索错误、经验质量差、向量方向错误、
注入过强、或任务本身不适合该记忆。

## 7. 实验纪律

- 每个训练/评测实验使用一个反映核心变量的 bash 脚本和独立输出目录；
- **每个 Phase 的编码完成后，必须先生成该 Phase 对应的一键服务器运行 `.sh` 脚本**。
  脚本需加载 `scripts/experiments/.server.env`、使用能反映阶段与核心变量的运行目录名、
  输出 artifact/trace 的明确路径，并可在服务器上通过一条 `bash <script>` 命令执行。
  完成最小本地校验后，连同代码、配置和脚本一起 commit + push 到 `origin/main`；只有
  推送完成后，才向用户交付服务器运行命令。不得要求用户手工拼接 Python 参数或修改
  已跟踪脚本来完成某个阶段的正式实验；
- 每次代码修改完成后，运行对应的最小校验，并 commit + push 到 `origin/main`；
- `.server.env` 仅保存本机路径与 API key，不进入 Git；
- 任何 bank artifact、split manifest 和结果必须带版本/hash；
- 高熵仅代表“可能需要帮助”，绝不能在论文或日志中被表述为“模型确定出错”。

## 8. 当前状态（2026-08-19）

已完成：

- GSM8K delimiter 上的 sink-masked entropy tracing、阈值校准和实验脚本；
- DeepSeek 离线 teacher-bank MVP，能产生可审计的 target/reference schema；
- 原始 MemGen 与 entropy gate 的基础实验工作流。
- Phase 1 verifier-backed bank 流水线代码：固定 split manifest、冻结 student rollout、
  success/failure 配对、Flash teacher、确定性审计、Pro reviewer 与人工争议分流；
- Phase 2 entropy-risk probe：按 `experience_id` 的 held-out 风险识别通过（ROC-AUC `0.8053`），
  并完整记录 target/reference 的 recovery/persistence 四格表；
- Phase 2 causal-cache 确认实验：真实全局 `recovery − persistence` vector 在 140 个配对事件上
  未优于 entropy-only、随机或反转对照，且相对反转向量显著更差；全局 centroid action 分支
  已关闭，详见本文件的 Phase 2 负结果；
- 下一版仅保留风险 gate，提出了尚未实现的 H3：student-state top-1 检索式、条件化 recovery
  displacement action，且已规定 split、对照、停止规则与确认门槛；

尚未完成：

- 审计未看过的 `calibration-val` / evaluation 剩余 split，并预注册 H3 的开发集和独立确认集；
- 对 H3 的经验级 key/action 定义做设计审查；通过前不实现、不调参、不重跑全局向量；
- H3 通过机制验证后才进行冻结配置的最终评测；
- MI side-KV bank 实现。
