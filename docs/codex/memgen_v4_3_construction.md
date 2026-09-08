# V4.3 Unified Heuristic Memory：实现与运行合同

本轮续接 V4.3 unified positive heuristic 路线。V4.2 文档、target/reference
工件和旧 oracle 保留为历史来源；不再继续 selector calibration，也不重新调用教师。
当前实现覆盖 tensor-free construction、统一 Side-KV compiler/loader、四层审计、
固定 wrong-Bank 对照、可选 all-bank sweep、认证断点续跑和 smoke/full 总入口。
本地测试使用合成 17/116 数据与随机初始化的小型原生 Qwen；真实服务器数据 qualification
及 CUDA smoke/full 结果需要在已有服务器工件上运行，不能从本地测试推断。

## 已核对的来源与复用边界

- 起始 revision：`f592d66`，本地 `main` 与本地 `origin/main` 指向相同提交。
- `scripts/build_v4_2_semantic_bank.py::build_evidence_packet` 保存完整 evidence、五元签名和逐包哈希。
- `memgen/experience/v4_2_local_direct.py::build_local_direct_bank_record` 保存原 membership、sample 对应关系、packet SHA 和 source-signature SHA map。
- `memgen/experience/v4_2_curated.py::build_curated_manifest` 继承 packet 文件 SHA，并绑定 curation policy、源 manifest 与逐条 Bank record。
- `memgen/experience/v4_question_recovery.py` 展示 surviving packet 的认证与 sample/question hash 检查；V4.3 不调用 recovery 或重新拟合 risk。
- `memgen/model/v4_side_kv.py::_compile_descriptor` 提供内容位置提取与三个 variants 的参考；新 compiler/loader 使用独立 unified schema，旧 target/reference 工件不作为新 memory。
- `memgen/model/v4_oracle.py` 的 cache/decoding 工具由新 runtime 复用；四分支执行与单次激活逻辑在 `v4_3_runtime.py` 中独立实现。

本地未发现真实 `semantic_evidence_packets.jsonl`、curated `bank_records.jsonl` 或
`bank_manifest.json`，没有可认证复用的本地 embedding artifact。默认 Python 没有 Torch；
原生模型测试使用 `/private/tmp` 下独立安装的 Torch/Transformers 环境。
单元测试使用明确标注的合成数据，不能作为真实 116 条 evidence 的构造结果。

## 输入与认证

入口：`scripts/build_v4_3_unified_bank.py`，核心：`memgen/experience/v4_3_bank.py`。
仅依赖 CPU/Python 标准库，不加载 embedding 模型，不读 provider key，不调用远程服务。

输入是 curated 目录中的 `bank_records.jsonl`、`bank_manifest.json`，原始 semantic
packet JSONL，以及既有 `v4_2_local_curation_policy.json`。

认证顺序：

1. 计算四个输入文件 SHA，读取后再次检查文件未变化。
2. 验证 curated manifest 逻辑哈希、逐条 record 哈希和顺序。
3. 用 curated manifest 内保存的文件 SHA 验证完整 packet 文件与 curation policy；验证 policy 逻辑 SHA 和旧 local-direct manifest/profile/order 血缘。
4. 验证每个 packet 的 schema、逻辑 SHA、5～8 evidence、完整五元签名、provenance SHA 格式和 verifier 字段存在。
5. 验证 sample ID 的 GSM8K train 命名与 question SHA 前缀；拒绝重复 evidence/sample，包括跨 packet 的重复。
6. 逐 Bank 精确验证 candidate、packet SHA、全部 evidence→sample 对应、source-signature SHA map；固定 17 个 retained Bank、116 个独立 sample、11 primary / 6 conditional。

不需要已删除的 Phase-1 原件。原 `source_signature_sha256` 覆盖的不是 packet 中的五元
子集，因此不把五元签名重新计算的哈希冒充原 signature 哈希。原 SHA 通过已认证 packet
及 curated map 交叉绑定，五元内容另保存 `semantic_signature_content_sha256`。
这些是 artifact 完整性校验，不是来源签名的独立密码学签署或事实正确性证明。

## 全部 evidence 的确定性消费

保持原有 17 组，不合并同 category Bank，不随机抽样。每组全部 5～8 条 evidence
的五个字段都进入规范化、静态泄漏检查、相似度计算和候选审计；116 条 evidence
对应 580 个字段候选。旧 medoid 没有选择优先权。

本轮是 **extractive lexical consensus**，不声称完成 LLM synthesis 或语义蕴含审核。
固定规则包含在 `CONSTRUCTION_POLICY`，导出到 `construction_policy.json`，并绑定到每条
record、manifest 和 Bank ID。服务器入口没有调低阈值或改变预期数量的参数。

- 规范化：NFKC、空白、标点和排版引号；不删除数字或姓名以洗白泄漏。
- 表示：CPU token sequence。对称 SequenceMatcher 分数至少 `0.80`，否定、操作和顺序相关 protected tokens 必须保持相同顺序。
- 支持：穷举最多八成员中的 complete-link 子集；所有支持成员两两通过，而非经中间成员传递合并。
- 排名：最大独立 sample support、最大有效候选组内 centrality、较短完整条款、experience ID 字典序。支持集合相同时也用 experience IDs 字典序确定。
- 核心字段：problem_structure、decision_point、repair_operator、failure_mechanism、verification_operator，各自至少五个独立 sample 支持。
- 组合字段：applies_when 同时使用 structure 和 decision，因此两者支持集合的交集也必须至少五个独立 sample；各自有五条但共同支持不足仍隔离。

这是一项预先冻结的保守表面共识规则；相近表述不等于逻辑等价，表达差异较大的正确
条款也可能被拒绝。不可根据后续 outcome 放宽阈值。未来若更换表示，需要新构造版本。

每个 clause 保存代表 experience、支持 experience/sample IDs、独立支持数、支持规则、
source signature hashes、两两分数、完整候选排名，以及所有候选的规范化文本 SHA 和
静态审计原因。带泄漏的候选文本本身不写入报告。

## Unified card 与资格

固定字段和渲染顺序：

```text
Use when:      problem_structure + decision_point 的独立共识
Procedure:     repair_operator 共识
Avoid:         “Avoid this failure:” + failure_mechanism 共识
Verify:        verification_operator 共识
Use only when: 继承适用范围的固定限制 + conditional curation scope guard
```

`only_use_when` 是收窄使用范围的约束；其中 conditional guard 来自已有 curation
理由的固定映射。它不冒充五条 evidence 的独立新发现；`boundary_provenance` 明确记录
`independent_support_claim=false`。core 字段仍严格要求五个独立样本。

静态检查包括数字/数字词、姓名线索、答案标记/片段、公式线索、source ID、role/reward
语言、与原问题/解答/轨迹的连续八词重叠，以及 repair/verify 是否有可执行操作词。
这些检查可能保守误报，也不保证排除所有语义泄漏或隐含矛盾；报告明确声明没有完成
独立事实一致性审核。原问题、解答、完整轨迹与 verifier 只在输入审计中使用，不进入
card、descriptor 或导出的 runtime record。

任何核心字段支持不足、无合格 process 候选、缺少 conditional guard 或组装后静态
检查失败，都将整个候选放入 quarantine，并将 card/descriptor/descriptor SHA 设为
`null`。它仍保留原 membership、lineage 和诊断，不出现在可编译 tier manifest。
其余候选只取得 `qualified_for_offline_compilation`，所有 record/manifest 始终
`qualified_for_online_use=false`。11/6 是输入 tier 数，不能预先假定输出仍为 11/6。

新 ID 是 `v43-bank-` 加完整 SHA256，绑定 construction version、旧 Bank ID、candidate、
新 card、完整 construction provenance 和构造规则 SHA。新内容不复用旧 hash。
`source_v42_to_v43_lineage.json` 覆盖全部 17 个候选，包括被隔离候选，并绑定新 record SHA。

## 输出与恢复

```text
candidate_bank_records.jsonl
primary_bank_records.jsonl
primary_bank_manifest.json
conditional_bank_records.jsonl
conditional_bank_manifest.json
quarantined_bank_records.jsonl
source_v42_to_v43_lineage.json
clause_support_report.json
leakage_audit_report.json
construction_report.json
construction_policy.json
construction_bundle_manifest.json
```

record、tier manifest、lineage、report 均有逻辑哈希；最后写入的 bundle manifest 还
绑定所有输出文件的字节 SHA 和内容逻辑 SHA、全部输入及 builder 实现 SHA。
输出不嵌入绝对路径、时间戳或 Git revision，因而相同输入/实现可跨目录 byte-identical；
revision 与路径打印到日志。Git revision 变化但实现不变不破坏逻辑复用。

`--resume` 先核对全部已有文件，再补齐未 seal 的中断输出；发生任何漂移则停止，保留
已有内容。完整 seal 后缺失文件不会被静默重建。未知文件和 symlink 输出也拒绝覆盖。
`--validate-only` 从原输入重新构造并比较所有输出，不写文件。资格失败是可审计结果，
构造命令可以成功结束；下游应检查 tier manifest 的 Bank 数和状态，不能只看退出码。

## 单独构造命令

```bash
python scripts/build_v4_3_unified_bank.py \
  --source-dir /data/memgen-runs/v4/offline/construction_v4_2_local_curated \
  --semantic-packets /data/memgen-runs/v4/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl \
  --curation-policy configs/experiments/gsm8k/v4_2_local_curation_policy.json \
  --output-dir /data/memgen-runs/v4/offline/construction_v4_3_unified \
  --resume
```

将 `--resume` 替换为 `--validate-only` 可只读复核已完成工件。先查看
`construction_report.json` 的 qualified tier counts、quarantined count 和 flagged clause
count，再查逐字段支持与泄漏报告。真实数据未通过时，不根据准确率降低门槛。

## 完整系统入口

```bash
./test.sh                 # 默认 all：构造、编译、smoke、认证通过后 full
./test.sh smoke
./test.sh full            # 要求当前输入与实现对应的 smoke 已通过
MEMGEN_V43_VALIDATE_ONLY=1 ./test.sh all
```

根入口分派到 `scripts/experiments/gsm8k/run_v4_3_unified_bank_experiment.sh`。
旧 V4.2 recovered-source 流程保留为显式 `./test.sh legacy [smoke|full|all]`，
V4.3 不自动调用它，也不重新恢复 Phase-1、拟合 risk、提取 source cache 或生成教师证据。

默认复用 `/data/memgen-runs/lineages/gsm8k-recovery/gsm8k-v4-packet-replay-20260907-r1/`
下面的 `v4_oracle_full/source_state_cache/v4_source_state_manifest.json` 和
`risk_v3_4/token-entropy-risk-gate-v3.4.pt`。smoke 也绑定完整 116-sample cache。
新输出在 `/data/memgen-runs/v4/offline/` 下的 `construction_v4_3_unified`、
`side_kv_v4_3_unified`、`v4_3_unified_audit/{smoke,full}`，不会覆盖旧工件。
可用 `./test.sh --help` 查看全部路径、设备与只读验证覆盖项。
`memgen.model.MemGenModel` 改为按需导入，旧公开 import 保持兼容；离线 side-KV 工具不再
因为 package 初始化而强制加载 PEFT/训练模型/TensorBoard。

## 编译与运行时

`scripts/compile_v4_3_side_kv.py` 分别编译 qualified primary/conditional 清单。
每条 Bank 对应一个 memory ID；三个 descriptor variants 沿 slot dimension 拼接，
wrapper 影响内容隐藏状态但 wrapper slots 不保留。在 Layer 24 输入处捕获 hidden states，
经过原生 input LN/k_proj/v_proj，保留全部 KV groups、canonical pre-RoPE、delta 0。
服务器模型用固定 commit revision、SDPA、bfloat16。编译工件保存 tensor 文件 SHA、
逐 memory K/V SHA、形状、slot mask、RMS、variant spans、record 与实现绑定。
清单还绑定 Torch、Transformers 和 safetensors 版本；版本漂移时不能复用旧实验工件。
loader 校验后返回独立 storage；不接受旧 Bank ID 或 `::reference` ID。

`scripts/audit_v4_3_unified_memory.py` 将旧 source event 映射到新 Bank，验证完整 packet、
question/trajectory hashes、sample membership、旧 record、risk 与 reasoner 身份。
这生成新的 source-state binding，不改旧 cache manifest。运行前用固定 tokenizer 和
GSM8K prompt contract 重建并校验所有选定 prefix。官方答案仅用于四分支生成后的评分。

四层均执行 baseline/matched/near_wrong/far_wrong：

| 审计层 | 注入点与对照 | 诊断目的 |
|---|---|---|
| visible_content | 同一 descriptor 加入 system message，prefix 明确不同 | 文本内容是否有用 |
| prompt_end_latent | 原生 prompt 最后一个 live query，绕过 gate | latent 编码与注入是否可用 |
| exact_gate_failure | 失败轨迹的真实 gate prefix | 实际介入点的修复效用 |
| success_safety | 成功轨迹自身真实 gate prefix | 成功样本上的伤害 |

三个 latent 层先 replay 原生 prefix（保留最后一个 token 为 live query），再为全部分支
克隆精确相等且 storage 独立的 cache；运行中逐步检查 native cache 长度不被 memory 改变。
一次激活一个 memory，最多 active 32 步；连续两个 low-entropy token、答案标记或 EOS
可提前卸载。卸载后继续原生生成，直到完整 boxed answer、EOS 或总 completion 1024 token
（包含已 replay 的 completion prefix）。local-32 指标与最终 strict reward 分开评分。

## 对照、汇总与可选 sweep

wrong-Bank 在生成前冻结，使用 descriptor 的本地 word/bigram TF-IDF cosine，不读答案。
候选不得共享 construction sample。near 优先同 category、再按相似度降序；far 必须不同
category 且不同于 near，再按相似度升序；tier、slot 数差与 ID 作为固定 tie-break。
无合格对照时显式排除并报告，不随机补选、不根据效果更换对照。

每层报告按 tier、Bank、gate attempt 分组，同时给 case accuracy 与 independent-sample
macro accuracy、gain/harm、local/final 格式和正确性、KL、attention mass、top1 与轨迹分歧。
gate-unreachable failure 单列，不进入实际 gate 效用的分母，也不记为 memory ineffective。
真实 cache 原有 99 failure cases/61 samples、107 success cases/65 samples 与 55 unreachable
是待核对来源统计；V4.3 qualification 隔离后实际纳入数由报告明确列出，不硬凑这些数。

`MEMGEN_V43_ALL_BANK_SWEEP=1 ./test.sh all` 在四层之后附加全部 qualified primary memories
的 failure-prefix sweep。启用或关闭 sweep 会改变 plan，已有输出需另设 `MEMGEN_V43_AUDIT_ROOT`。
输出包括 oracle-best（含 baseline）上界、任一 memory 有帮助率、至少帮助两个独立样本的
reusable memory 数、harm、tie-fractional hubness、matched rank 和同类替换成对统计。
这些统计使用 outcome 信息，仅代表 construction 机制诊断，不是 online accuracy。

## 审计工件与恢复

每个运行目录保存 `v4_3_source_state_binding.json`、`v4_3_wrong_bank_controls.json`、
`v4_3_audit_plan.json`、`v4_3_audit_profile.json`、`cases/<case_id>.json`、
`v4_3_audit_report.json` 和 `v4_3_core_summary.json`。
每个 case 原子写入并独立认证；中断后复用已有 case，汇总可重建。
profile 绑定输入、编译清单、prompt contract、配置与相关实现文件 SHA。
漂移时拒绝覆盖，必须使用新的输出目录。

smoke 按固定 Bank ID 规则选择每个 tier 一个有真实 failure/success gate 的 qualified Bank，
每层最多两个 case，并执行完整 1024 completion horizon。smoke pass 要求完整 case、cache/
attention 检查以及每个 tier 至少一个 latent 分支的非零 KL 或轨迹分歧；不要求准确率上升。
full 开始前重新认证 smoke profile、plan 和所有 case；缺失、损坏或不同实验的 smoke 不放行。
所有输出保持 `offline_only=true`、`qualified_for_online_use=false`，没有 selector 工件。

## 验证

```bash
python3 -m unittest discover -s tests -p 'test_v4_3*.py' -v
python3 -m py_compile memgen/experience/v4_3_bank.py scripts/build_v4_3_unified_bank.py \
  tests/test_v4_3_bank.py tests/test_v4_3_pipeline_contract.py
git diff --check
bash -n test.sh
shasum -a 256 memgen/model/e1_runtime.py memgen/model/side_kv.py \
  memgen/model/v3_5_retrieval.py memgen/model/v3_runtime.py
```

测试覆盖完整 17/116 合成流程、哈希与身份、重复与外组证据、独立字段选择、complete-link
不传递、固定 tie-break、process/leakage、资格失败隔离、lineage、内容寻址、descriptor
与支持交叉认证、tier 分离、字节确定性、断点恢复及无模型/网络/环境读取的 CPU 构造。
原生模型测试需安装仓库兼容的 Torch/Transformers/safetensors/numpy；依赖不存在时明确 skip，
不能把 skip 当作 tensor/runtime 已验证。测试还覆盖随机初始化 24 层 Qwen 的原生 LN/K/V
一致性、bf16 编译与加载、真实 Side-KV 四分支和缓存隔离、卸载后原生继续生成、driver
认证恢复与完整 shell 编排。本地测试不下载预训练模型。
未执行真实 116 条构造、GPU compilation、GPU smoke/full、dev-test/final-test 或任何付费 API。

2026-09-08 本地验证：69 项 V4.3 测试通过（包括 10 项原生 tensor/runtime 测试和完整
17/116 工件交接测试，无 skip），43 项旧 side-KV/oracle/recovery/pipeline 回归通过。
原生测试环境为 CPU Torch 2.7.1、Transformers 4.55.4、safetensors 0.7.0，随机初始化
24 层小型 Qwen，未下载预训练模型。py_compile、shell 语法、diff whitespace 和四个
frozen SHA 均通过；这些结果不构成服务器 CUDA 实验的 PASS。
