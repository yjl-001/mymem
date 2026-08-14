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

### 4.2 SEAL-style compiled artifact

每个经验簇 `k`、每个候选 layer `l` 保存：

```text
memory_id, cluster_id, layer, boundary_definition,
vector S[k,l], vector_rms, evidence_count,
reasoner_model_revision, tokenizer_revision,
source_episode_ids, target/reference provenance,
construction_config, calibration_statistics
```

向量由**目标 student 模型自身**生成，不能直接使用强教师的 hidden state：

\[
S_{k,l} = \frac{1}{N}\sum_i \operatorname{Normalize}
\left(h^+_{i,l} - h^-_{i,l}\right).
\]

其中正负 evidence 属于同一经验簇，优先使用相近任务/难度的配对 rollout。第一版可
先使用全局簇 `stay_on_track`；后续才扩展到多个经验簇。

### 4.3 在线注入

使用当前 state 的尺度进行归一化，而非直接使用无约束的均值差：

\[
\hat S_{k,l}=S_{k,l}/\operatorname{RMS}(S_{k,l}),
\]

\[
H'_{l,t}=H_{l,t}+g_E(t)\,g_R(t,k)\,\alpha\,
\operatorname{RMS}(H_{l,t})\hat S_{k,l}.
\]

- \(g_E\)：sink-masked entropy 的软门控；
- \(g_R\)：检索相关度；第一版全局 vector 时取 1；
- \(\alpha\)：在 `calibration-val` 上选择的相对扰动强度。

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

### Phase 2：全局 SEAL-style vector 可行性

**目的**：先验证 residual intervention 是否有意义，不引入动态检索复杂度。

**工作项**：

- 从正确且紧凑的 execution-like boundary 与错误且冗余/转向的 boundary 中提取
  student hidden states；
- 为多个候选层构造全局 `stay_on_track` vector；
- 在 `calibration-val` 上选择 layer、alpha、soft-gate slope 和最大注入次数；
- 实现 boundary layer hook、vector artifact load/save、完整 intervention trace。

**必须对照**：

1. vanilla；
2. entropy gate 但不注入；
3. 每个 boundary 注入真实 vector；
4. 随机 boundary 注入真实 vector；
5. entropy gate 注入同范数随机 vector；
6. entropy gate 注入正负标签反转 vector。

**产物**：`SteeringVectorArtifact`、校准报告、对照实验 JSONL、可一键复现实验脚本。

**验收标准**：

- 真实 vector + entropy gate 在 `calibration-val` 上优于同范数随机向量与反转向量；
- 真实 vector 的结果不能仅等价于“输出更短”：格式正确率不下降，且 accuracy/奖励
  的变化方向优于随机对照；
- 95% 以上的触发满足预设的相对扰动上限 `||ΔH|| / ||H|| <= r_max`；
- 发生 NaN、格式崩坏或超出扰动上限时自动禁用该次注入并留下 trace。

> 数值的绝对通过线不在开发前伪造：`r_max`、alpha、layer 均只可在
> `calibration-val` 确定，随后锁定并进入下一阶段。

### Phase 3：经验簇与轻量检索

**目的**：从单一全局行为先验升级为条件化经验选择。

**工作项**：

- 在 Phase 1 的 verified records 上形成少量、可审计经验簇；
- 每簇构造 `S[k,l]`，保存 evidence 数量与质量；
- 初版使用当前问题与 `situation_signature` 的文本 embedding 做 top-1 检索；
- entropy gate 后才执行检索；低相似度则不注入；
- 记录 memory id、相似度、注入强度和最终结果。

**产物**：cluster manifest、多个 vector artifacts、retrieval trace、per-cluster 评估表。

**验收标准**：

- 检索候选只来自 `bank-source`；
- 人工审阅至少 50 个触发：top-1 经验的适用性准确率至少 80%；
- 条件化 vector 在固定计算预算下不低于 Phase 2 全局 vector，且至少一个经验簇的
  verified-failure subset 有可重复的正向收益；
- 无关或低相似度样本的触发率/干预强度明显低于高相似度样本。

### Phase 4：冻结配置后的最终评估

**目的**：在从未参与任何构造或调参的 test 集上确认结论。

**工作项**：

- 锁定 split、bank、vector、layer、alpha、entropy threshold、检索配置；
- 只运行一次完整 final-test；
- 与 vanilla、entropy-only、随机-vector、全局-vector、条件化-vector、MemGen Weaver
  对照；
- 报告 accuracy、token 数、trigger rate、延迟、memory artifact 大小、格式错误率和
  intervention diagnostics。

**验收标准**：

- final-test 不参与任何选择；配置文件和 artifact hash 可完全复现；
- 条件化真实 vector 优于随机 vector 对照，且没有不可接受的格式/延迟回归；
- 结论按 effect size、置信区间和失败案例报告，不能只报告最优单次 seed。

### Phase 5：MI side-KV 升级决策

**进入条件**：Phase 4 表明经验内容有真实收益，但全局/簇级 residual vector 出现
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

## 8. 当前状态（2026-08-12）

已完成：

- GSM8K delimiter 上的 sink-masked entropy tracing、阈值校准和实验脚本；
- DeepSeek 离线 teacher-bank MVP，能产生可审计的 target/reference schema；
- 原始 MemGen 与 entropy gate 的基础实验工作流。
- Phase 1 verifier-backed bank 流水线代码：固定 split manifest、冻结 student rollout、
  success/failure 配对、Flash teacher、确定性审计、Pro reviewer 与人工争议分流；

尚未完成：

- 在服务器上运行完整 Phase 1 rollout/teacher/reviewer 流水线，并完成争议裁决；
- 经验聚类；
- vector compiler、layer hook、soft-gated residual integrator；
- Phase 2–4 的对照评测；
- MI side-KV bank 实现。
