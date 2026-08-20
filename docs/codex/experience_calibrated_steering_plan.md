# Experience-Calibrated Entropy-Gated Memory：实施计划与验收标准

## 1. 目标与边界

本计划验证一个训练无关、可审计的**风险触发式经验记忆**原型。Phase 1 从已验证的
成功/失败 rollout 中产生经 Pro 审核的经验抽象；在线阶段只在 reasoning boundary 出现高熵、且
hidden state 显示可能持续发散时，检索语义相关的经验内容，并将其作为 side-KV memory 提供给
冻结的 reasoner。

核心问题不再是“能否用一个 vector 降低熵”，而是：

> 当风险 gate 触发时，相比同预算的随机/错配经验内容，语义匹配的 Phase 1 经验能否被模型
> 实际注意和利用，并改善推理结果而不损害格式？

高熵和 risk score 只回答**何时值得请求帮助**，不等价于模型出错；下一 boundary 熵只是辅助
诊断，低熵既不是正确性的定义，也不是本路线的必要中介目标。主结果依次是经验内容的因果
利用、任务正确率及格式安全性。

截至 2026-08-20，Phase 2 已确认风险识别成立，但否定了两个“动作即 hidden residual
direction”的候选：全局 `recovery − persistence` centroid action 在独立确认中无效；同题
local-contrast action 虽有可构造样本，但 raw hidden-state 的 top-1 路由 margin 很小。因此二者
均保留为可审计的负向探索，不再作为主方法。该结果不否定 Phase 1 bank、风险 gate 或“在风险时
调用经验内容”的假设。

本计划不宣称 zero-shot 或“无外部监督”：bank 构造使用训练期已验证 rollout；Flash Teacher 和
Pro Reviewer 仅在 Phase 1 离线归纳/质检，后续 Phase 2/3 不新增 AI 标注。

本计划中 MemGen、FlashMem、SEAL 和 MI 的机制、训练边界与项目映射见
[前置方法：MemGen、FlashMem、SEAL 与 Memory Inception](method_prerequisites_memgen_flashmem_seal_mi.md)。
MemGen 提供生成与预算框架，FlashMem 风格熵提供候选时机，MI side-KV 是当前的内容整合接口；
SEAL-style residual vector 仅作为已完成的历史 ablation。

## 2. 总体架构

```text
bank-source 样本 ── frozen student rollout ── verifier ──► verified success / failure
                                                     │
                              Flash Teacher + Pro ──► ai_approved experience abstraction
                                                     │
                                  payload audit ────► sanitized MemoryRecord + semantic index + side-KV

在线 prompt + 已生成 CoT ──► boundary ─► entropy + hidden-risk gate
                                             │
                                    semantic query / retrieve MemoryRecord
                                             │
                                      SideKVIntegrator ─► frozen reasoner continuation
```

在线控制由三个可独立检验的接口组成：

```python
memory = compiler.compile(sanitized_memory_record, frozen_reasoner)
selected = retriever.select(question_and_partial_cot, memory_index)
runtime_state = integrator.apply_side_kv(runtime_state, selected)
```

- `EntropyRiskTrigger`：只根据当前 boundary 的 sink-masked entropy 与 hidden-risk score 决定
  是否请求经验；它不决定经验内容。
- `SemanticMemoryRetriever`：以题目和固定窗口的 partial CoT 为 query，在 Phase 1 的可用经验中
  检索；不得以 raw risk hidden state 的 top-1 取代语义检索。
- `MemoryRecordCompiler` / `SideKVIntegrator`：将冻结经验文本预编码为 canonical side KV，并在
  固定 layer 让当前 query attention 可见；不改变原 prompt/history cache。

## 3. 数据切分与反泄漏约束

每个数据集必须固定以下互不重叠的 split：

| Split | 用途 | 禁止事项 |
|---|---|---|
| `bank-source` | rollout、verifier、教师归纳、MemoryRecord 与风险原型 | 不用于线上方法选择或最终评估 |
| `calibration-val` | payload token budget、检索器/side-KV 可行性与固定阈值 | 不写入 bank，不供教师查看 |
| `dev-test`（可选） | 只读调试或预注册开发实验 | 不用于最终报告 |
| `final-test` | 一次性最终评估 | 不供 bank、教师、检索器、任何配置选择使用 |

约束：

1. 教师只读取 `bank-source` 的轨迹及 verifier feedback，绝不读取 test 问题或答案。
2. `verified_success` / `verified_failure` 必须由确定性 verifier/reward 标记，教师不能覆盖它。
3. 正式 record 必须来自 `ai_approved`；`teacher_inferred`、rejected、deferred、quarantined 都不进
   memory index。
4. 每条 record 保存 source split、experience id、模型/rollout revision、审核结论和 payload hash。
5. runtime payload 严禁携带原题、完整 target/reference trajectory、`\\boxed{}` 答案、原始 evidence
   quote 或可复原当前实例答案的数值 literal；它只能携带经审核的、可泛化的经验抽象。

## 4. Artifact 定义

### 4.1 Phase 1 原始经验与可用记忆

```text
context + target/reference trajectory + verifier feedback
  → teacher abstraction + Pro evidence review
  → ai_approved MemoryRecord
  → sanitized retrieval key + contrast payload + side-KV
```

target 是已验证的成功策略/验证动作；reference 是已验证失败的机制、信号和避免条件。二者的
价值在于定义一个可复用的决策边界，而非向在线 student 暴露成功答案或失败 CoT。

首版 `MemoryRecord` 的 payload 采用固定对比模板：

```text
When facing: <reviewed situation / applicability>
Prefer: <target success decision or verification strategy>
Avoid: <reference failure mechanism or warning signal>
```

payload audit 必须逐项证明三个字段存在、长度受预算约束、无答案泄漏/原始 quote/实例特有 literal，
并保留 source `experience_id` 与审核证据用于离线追溯，但这些 provenance 不送入模型。

### 4.2 entropy-risk gate artifact（保留）

固定 layer `l=24`，只用 bank-train 的高熵事件形成 recovery/persistence 原型：

\[
R(h_t)=\cos(h_t,\mu_{\mathrm{persistence}})-\cos(h_t,\mu_{\mathrm{recovery}}).
\]

高熵后下一 reasoning boundary 从高阈值自然降到低阈值记为 `recovery`，否则为 `persistence`。
这只用于离线监督风险原型并在 held-out bank 评估可分性；在线始终只读取当前 state。`R(h_t)>0`
表示“更像会持续”的候选，不能被表述为“确定错误”，也不产生一个应该加到 residual stream 的动作。

### 4.3 Side-KV memory artifact（当前主路径）

每个 `MemoryRecord` 编译为下列不可变 artifact：

```text
memory_id, source_experience_id, approved_route, experience_type,
sanitized_retrieval_key, sanitized_contrast_payload, payload_hash, token_count,
kv_layer, canonical_pre_rope_kv, reasoner/tokenizer revision, compiler config
```

side KV 由目标 student reasoner 编码，而不是教师 hidden state；它以 canonical pre-RoPE key/value
形式缓存，在线按当前位置正确施加 RoPE 后追加为 memory-only attention source。原 prompt 与生成
history 的 KV cache 不得被覆盖、重建或作为 memory payload 泄漏。

### 4.4 历史 residual-vector artifacts（已关闭）

`S=μ_recovery-μ_persistence` 的全局 centroid residual action 已在独立确认中失败；同题
reference-persistence → target-recovery local action 的 raw-state top-1 路由也因低 margin 停止。
它们只作为负结果和诊断实现保留，不可通过重试、扩大样本、搜索 layer/alpha/符号来恢复为主实验。

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

- 所有正式 memory evidence 都能追溯到 verifier 结果；
- `teacher_inferred` 与 `verified_failure` 在 schema 和下游过滤中严格区分；
- 全部 teacher records 都有独立 Pro review 和可追溯的审核结论；
- 正式 bank 只包含所有关键 assessment 均为 supported 的记录；deferred、rejected 与 quarantined 均保留可追溯证据但不参与 memory payload 编译；
- 出现自相矛盾、事实错误或 target/reference 等价的记录必须可被 Pro 审核拒绝并保留证据。

### Phase 2：entropy-risk probe 与 residual-vector 证伪（已完成）

**目的**：先区分“高熵但自然恢复”与“高熵后持续发散”，并用严格的单次 residual
intervention 证伪“风险分类方向可直接作为纠偏动作”的最小假设。它是历史 ablation，不是当前
side-KV 主路径的优化入口。

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

### Phase 3：经验内容检索与 side-KV 的最小因果实验（当前设计）

**假设 H3-content**：在高熵且 `R(h)>0` 的首个 reasoning boundary，给 frozen reasoner
提供语义匹配、已审核且无泄漏的 Phase 1 contrast memory，能够产生与同预算错配 memory 不同的
推理行为，并在不损害格式的前提下改善最终任务表现。

这不是“让当前 hidden state 变低熵”的假设；memory 是供 attention 使用的外部内容。若下一
boundary 熵变化存在，只记录为辅助机制指标，不能代替内容利用或 accuracy 的判断。

#### E0：payload 与 side-KV 可行性审计

**目的**：先证明送入模型的是安全、可用的经验内容，而不是答案泄漏或无效 cache。

- 只用 Phase 1 `ai_approved + answer_correctness` records，不受历史 23 个 local-action
  candidate 的限制；冻结已有 Teacher/Pro 输出，不发起新调用；
- 从 Pro 支持的 `situation/applicability`、target 成功策略/验证、reference failure
  mechanism/signal 构建固定模板 payload；逐条剔除原题、轨迹、答案、`\\boxed{}`、原始 evidence
  quote 与实例数值；
- 报告字段覆盖率、长度/重复率、泄漏审计、payload hash 与可追溯 provenance；
- 使用冻结 reasoner 在 layer 24 编译 canonical pre-RoPE KV，验证其可附加、不会改写原 cache，
  且 online query 对 memory 的 attention mass 非零。

**通过条件**：不存在 payload 泄漏；KV shape/RoPE/cache 单测通过；每次实际注入都可报告非零
memory attention mass。E0 不报告“效果提升”。

#### E1：匹配内容与错配内容的最小因果比较

先以 observation-only pass 生成不可变 assignment manifest：每个 sample 记录首个高熵候选
boundary、risk score、是否触发、retrieval query、matched memory id/score、token budget 及 abstain
reason。所有条件必须复用此 manifest，保证相同题、相同 prefix、同一触发位置且每题至多一次。

四个条件：

1. `vanilla`：不计算/不使用 memory；
2. `gate-observation-only`：运行相同 gate、但不附加 memory；其 completion 必须和 vanilla 一致；
3. `matched-memory`：按 question + 固定 partial-CoT 窗口检索的 top-1 MemoryRecord side-KV；
4. `shuffled-memory`：对触发样本按 `sample_id` 做确定性 derangement，使用另一条 matched
   memory；保持完全相同的触发集合、memory 数量、token budget 和 side-KV layer。

检索先在 `calibration-val` 离线比较透明 BM25 与冻结文本 embedding 的 retrieval quality，再固定一种
方法、固定 top-1 和 payload budget；不能用最终 accuracy 选择检索器。在线 query 不得包含未来 token
或最终答案。

**判定顺序**：

1. 触发样本的 matched memory attention mass、retrieval score 和 injection logits-KL 证明内容实际
   进入计算；
2. `matched-memory` 相比 `shuffled-memory` 和 gate-only 的效果差异（paired sample ID，bootstrap CI）
   是主要因果证据；主要任务指标为 GSM8K accuracy，格式正确率不得明显低于 vanilla；
3. 只将下一 boundary 熵、token/延迟、失败案例作为诊断。若 E1 未过，结论是当前 payload、检索或
   side-KV integrator 之一无效，不能归咎于 risk gate 或 Phase 1 bank 整体无效。

#### E2：经验对比字段的消融（仅在 E1 通过后）

在完全相同的 gate、manifest、retrieval id、token budget 和 side-KV 接口下比较：

1. `target-only`（Prefer）；
2. `reference-only`（Avoid）；
3. `contrast`（When facing + Prefer + Avoid）。

它回答 Phase 1 的 reference failure mechanism 是否为 target success strategy 提供额外 guardrail，
而非重新测试任意 residual 方向。没有 E1 的 matched-vs-shuffled 证据，不做此消融。

#### E3：风险触发时机消融（仅在 E1 通过后）

同一条 matched memory 在风险 gate 的首次触发 boundary，与同生成中预算相同的确定性随机 delimiter
比较。它回答高熵+risk 是不是比“随便找一个位置”更好的记忆访问时机。此阶段不增加记忆数量、
不做 alpha/layer 网格、不把熵下降设为通过门槛。

### Phase 4：冻结配置后的最终评估

仅在 E1 及必要的 E2/E3 开发结论通过后，锁定 bank、payload compiler、retriever、layer 24、单次
触发预算和全部 manifest 规则，在从未参与选择的 `final-test` 运行一次。报告 vanilla、gate-only、
matched-memory、shuffled-memory（以及已通过才纳入的字段/时机消融），同时给出 accuracy、格式、
trigger rate、memory attention、检索分数、延迟、KV footprint、置信区间和失败案例。历史 residual
vectors 不重跑，也不作为主对照。

## 6. 统一日志与诊断要求

每次 evaluation 的每个样本、每个候选 boundary 至少记录：

```text
sample_id, split, generated_token_index, boundary_type,
entropy_with_sink, entropy_without_sink, entropy_threshold,
hidden_risk_score, gate_triggered, trigger_reason,
retrieval_query_hash, retrieval_method, retrieved_memory_id, retrieval_score,
memory_condition, memory_payload_hash, memory_token_count, memory_kv_layer,
memory_attention_mass, logits_kl_baseline_to_memory,
side_kv_applied, generation_length, final_reward, format_reward, output_path
```

对 `shuffled-memory` 还必须记录 `matched_memory_id`、`assigned_memory_id` 和确定性 shuffle
seed。它们用于区分触发错误、语义检索错误、payload 质量/泄漏问题、side-KV 没有被注意、内容
错配、或任务本身不适合此类经验；不再将 residual norm/alpha 当作主路径诊断。

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

## 8. 当前状态（2026-08-20）

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
- H3 的只读可行性审计脚本：重放冻结轨迹并统计同一 `experience_id` 内的
  reference-persistence → target-recovery local-contrast candidates、对齐相似度、action RMS 与
  leave-one-out 检索阈值；并检查最大对齐的选择增益、检索 top-1/top-2 margin、action 方向
  effective rank 与检索集中度；结果为 23 个 bank-train / 9 个 held-out 候选，action 方向并不
  共线（effective rank `13.75/23`），但 held-out top-1/top-2 margin 很小（中位数 `0.0059`），
  说明 raw hidden state 可描述风险邻域却不能稳定选择具体 action；该分支已归档；
- 研究主线已更新为风险触发的、语义检索的 Phase 1 经验内容 side-KV，而不是继续优化向量构造或
  以降熵作为目标。

尚未完成：

- 对所有 `ai_approved + answer_correctness` record 构建并审计无泄漏 `MemoryRecord` payload；
- 在不看 final-test 的前提下，冻结 semantic retriever（BM25 baseline / frozen embedding）和
  payload token budget；
- 实现并单测 layer-24 canonical side-KV compiler/integrator，先完成 E0 cache 与 attention 可见性
  审计；
- 在新的、未用于选择的 evaluation split 上完成 E1 matched-memory vs shuffled-memory 的单次、
  同 manifest 因果实验；只有 E1 成立才做字段与时机消融并进入 final-test。
