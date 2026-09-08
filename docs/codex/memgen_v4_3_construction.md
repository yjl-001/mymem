# V4.3 Unified Heuristic Memory：实现与运行合同

本轮续接 V4.3 unified positive heuristic 路线。V4.2 文档、target/reference
工件和旧 oracle 保留为历史来源；不继续 selector calibration。根据用户新授权，
仅记忆卡构造改为调用 DeepSeek，复用既有 evidence，不生成新题目轨迹。
当前实现覆盖 DeepSeek construction、统一 Side-KV compiler/loader、四层审计、
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

默认入口：`scripts/build_v4_3_deepseek_bank.py`，语义构造：
`memgen/experience/v4_3_deepseek.py`；公共 lineage/card schema 在 `v4_3_bank.py`。
只有存在未缓存的 Bank 请求时才读取 `DEEPSEEK_API_KEY`、加载 requests 客户端并调用
`https://api.deepseek.com/chat/completions`。不加载本地模型或 embedding 模型。
旧 `build_v4_3_unified_bank.py` 保留为词面共识实现，不再由默认总入口调用。

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

## 全部 evidence 的语义综合

保持原有 17 组，不合并同 category Bank，不随机抽样。每组全部 5～8 条 evidence
的原题、官方解答、成功/失败轨迹和五元签名一起进入该 Bank 的请求。正常每 Bank 一次，
共 17 次；网络/JSON 格式错误最多尝试两次，proxy 不长时间重试。模型固定
`deepseek-v4-flash`，JSON object 输出，thinking disabled，temperature 0，max_tokens 8192。
API 无不可变模型 commit 承诺，因此保存完整请求和响应，复用以缓存内容哈希为准；
同样输入重新请求不保证生成同样文本。

DeepSeek 综合出五条通用过程描述；每条必须对全部 evidence 给出支持/不支持判断、
来自该 evidence 同名 signature 字段的精确引用和理由。程序验证所有成员恰好出现一次、
引用真实存在、支持 ID 没有重复或跨组、样本映射一致。支持数是模型判断的独立 sample
计数，不再通过 0.80 词面相似度决定，也不把这些检查称为独立语义证明。

每个核心字段仍需至少五个独立 sample 支持；structure 与 decision 支持集合交集也需
至少五个。支持不足会保存原响应和诊断，不通过反复请求迫使模型凑够五条。
原始 evidence 中的数字、姓名和具体计算可以供抽象使用，不因这些内容而拒绝来源。
构造 policy、prompt、请求配置、源文件和实现 SHA 都纳入认证。

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

生成后的卡片检查包括数字字符、姓名线索、答案标记/片段、公式线索、source ID、role/reward
语言、与原问题/解答/轨迹的连续八词重叠，以及 repair/verify 是否有可执行操作词。
这些检查可能保守误报，也不保证排除所有语义泄漏或隐含矛盾；报告明确声明没有完成
独立事实一致性审核。通用的 twice/half 关系、普通 target quantity 表述允许保留。
原问题、解答、完整轨迹与 verifier 会发送给 DeepSeek 并保存在离线来源审计记录中，
不进入 card、descriptor 或 side-KV 编译内容。

任何核心字段支持不足、生成条款不合格、缺少 conditional guard 或组装后静态
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
输出不嵌入绝对路径、时间戳或 Git revision；相同输入、实现和已缓存响应可跨目录
byte-identical。Git revision 变化但实现不变不破坏逻辑复用。

单独的 `construction_v4_3_deepseek_requests/` 保存固定 `profile.json`、进程锁及逐请求
SHA 命名的响应文件。先校验全部已有响应，再读取 key/请求缺失 Bank；每条响应原子
写入，构造中断后只补缺失项。缓存漂移、并发写入、未知条目会明确停止。
响应尚未落盘时进程被杀，重跑可能重复该次请求；调用数只统计已保存响应对应的尝试，
不声称与服务端账单完全一致。缓存不保存 key、HTTP headers 或代理凭据。

`--resume` 先核对全部已有文件，再补齐未 seal 的中断输出；发生任何漂移则停止，保留
已有内容。完整 seal 后缺失文件不会被静默重建。未知文件和 symlink 输出也拒绝覆盖。
`--validate-only` 从原输入及完整缓存重建并比较所有输出，不写文件、不读 key、不请求 API。资格失败是可审计结果，
构造命令可以成功结束；下游应检查 tier manifest 的 Bank 数和状态，不能只看退出码。

## 单独构造命令

```bash
python scripts/build_v4_3_deepseek_bank.py \
  --source-dir /data/memgen-runs/v4/offline/construction_v4_2_local_curated \
  --semantic-packets /data/memgen-runs/v4/offline/construction_v4_2_semantic/semantic_evidence_packets.jsonl \
  --curation-policy configs/experiments/gsm8k/v4_2_local_curation_policy.json \
  --output-dir /data/memgen-runs/v4/offline/construction_v4_3_deepseek \
  --cache-dir /data/memgen-runs/v4/offline/construction_v4_3_deepseek_requests \
  --resume
```

将 `--resume` 替换为 `--validate-only` 可只读复核已完成工件。先查看
`construction_report.json` 的 qualified tier counts、quarantined count 和 flagged clause
count，再查逐字段支持与泄漏报告。真实数据未通过时，不根据准确率降低门槛。

## 完整系统入口

```bash
./test.sh                 # 默认 all：构造、编译、smoke、认证通过后 full
./test.sh construct       # 仅构造卡片，不要求 GPU/source cache/risk
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
新输出在 `/data/memgen-runs/v4/offline/` 下的 `construction_v4_3_deepseek`、
`side_kv_v4_3_deepseek`、`v4_3_deepseek_audit/{smoke,full}`，不会覆盖旧隔离工件。
DeepSeek key 仅传给构造阶段，构造返回后在诊断、编译和审计前清除。
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

## 构造隔离的排查

当 primary/conditional 都为零时，构造认证成功只表示输入和输出完整，并不表示卡片通过资格。
总入口在构造后调用只读诊断，要求两个 tier 各有至少一个 qualified Bank 才进入编译；
直接调用 compiler 也会先检查是否至少有一个 qualified Bank，再读取 reasoner 和加载 Torch。
这样不会由后续 revision 错误掩盖 construction 的首要阻塞。

```bash
python scripts/diagnose_v4_3_construction.py \
  --bank-dir /path/to/construction_v4_3_deepseek \
  --reasoner-manifest /path/to/side_kv_v4_2_local_curated/v4_side_kv_manifest.json
```

诊断输出每个 Bank 的 qualification failures、每字段支持数和候选静态检查原因计数，
以及旧 reasoner 元数据，不输出原题/答案/轨迹，不修改旧 bundle 或构造阈值。
服务器反馈的旧模型 revision 为 `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`，tokenizer
revision 为 `main`。对这种组合，新流程保留 `source_reasoner` 原始声明，并将模型的固定
commit 作为候选 tokenizer revision；远端浮动 `main` 不作为实际加载请求。

编译入口要求提供 `--cache-manifest` 与 `--semantic-packets`，先认证完整旧 cache 与 packet
血缘，再用候选 tokenizer 重建 116 个 prompt 和全部 actual failure/success gate prefix。
所有 token counts/IDs hashes 一致后才加载 reasoner、编译 side-KV。任一不一致都停止，
不会重写旧 cache、risk 或 manifest。固定模型 commit 以外的模型 revision 仍拒绝；
本地目录或其他旧 revision 别名没有被未经验证地放行。

新 Side-KV 清单分别保存 source/effective reasoner 和 `tokenizer_replay_validation`，
绑定 cache manifest、packet 文件、全部 source event 顺序与逐条 event hashes。resume
检查这份绑定，审计启动时再次重建全部缓存 prefix 后才加载模型；risk 身份与原始
`source_reasoner` 比较。验证范围明确限于旧缓存中的原生 prefix，不声称恢复了旧 tokenizer
文件的逐字节身份。两个 tier 和 smoke/full 使用相同的固定 tokenizer 与 replay binding。

这项 tokenizer 兼容修复不改变 construction 资格：服务器第一次运行得到 17 个 quarantine，
多数核心字段词面支持数为 1。新的 DeepSeek 路径已替换该词面共识判定，是否有可复用
语义共识由新响应与引用说明；后续有效性仍须 GPU 审计验证。

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

DeepSeek 新测试覆盖数字/表达不同的 17/116 来源、完整成员引用、重复/伪造引用拒绝、
真实支持不足、生成卡片泄漏、响应篡改、部分缓存恢复、零 API 缓存复用、凭据隔离及
mock 响应经过原生 bf16 编译进入 smoke/full 计划。这里的 provider 响应是明确合成的 mock。
2026-09-08 本地验证：92 项 V4.3 测试通过（无 skip），本次另外执行的 26 项旧
side-KV/oracle-runtime/recovery/pipeline 回归通过。
原生测试环境为 CPU Torch 2.7.1、Transformers 4.55.4、safetensors 0.7.0，随机初始化
24 层小型 Qwen，未下载预训练模型。py_compile、shell 语法、diff whitespace 和四个
frozen SHA 均通过；这些结果不构成服务器 CUDA 实验的 PASS。
