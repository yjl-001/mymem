# GSM8K：FlashMem 风格 sink-masked entropy gate

本实验只验证“何时调用 Weaver”，不涉及 experience bank。Weaver checkpoint 保持固定，
推理时不使用 Trigger；只有在生成前缀以 `,`、`.` 或换行结尾时，才计算门控。

门控使用 reasoner **最后一层**的注意力。对当前 query 的每个 head，先移除 token-only
prefix 中前 `attention_sink_token_count` 个有效 token 的 key，再对剩余 attention mass
归一化并计算 Shannon entropy；最终分数是各 head entropy 的均值。高于阈值才会插入
inference latent。prompt augmentation 设为零，避免其成为混杂因素。

FlashMem 主文将 `S_sink` 描述为 attention-sink 的索引集合，并以初始 token / BOS 为例，
但未给出动态 sink 识别器的精确算法。本实现的首个可复现实验将其操作化为前 4 个有效
token（`attention_sink_token_count: 4`）；因此它是 FlashMem 风格门控，而不是声称完全
复刻该论文。后续应在 `{1, 4, 8}` 上做消融。

## 1. 校准阈值

校准配置将阈值设为 `100.0`，因此不会插入 inference latent，但会记录每一个 delimiter
候选的 entropy。`--devices` 按服务器实际 GPU 指定；checkpoint 使用绝对路径。

```bash
python scripts/launch_experiment.py eval \
  configs/experiments/gsm8k/entropy_calibration.yaml \
  --devices 0 \
  --run-id gsm8k-entropy-calibration \
  --set model.load_model_path=/data/memgen-runs/train/<weaver-sft>/model
```

评测输出的 `evaluate/entropy_gate_trace.csv` 包含 entropy、被掩掉的 sink token 数、
sink attention mass、阈值、预算状态与插入决定。使用验证集候选 entropy 的 85 分位作为
FlashMem 风格初始阈值：

```bash
python scripts/calibrate_entropy_threshold.py \
  /data/memgen-runs/evaluate/gsm8k/.../evaluate/entropy_gate_trace.csv
```

该脚本会产生 `entropy_threshold.json` 和可直接复制的 `--set` 覆盖项。

## 2. 正式评测

```bash
python scripts/launch_experiment.py eval \
  configs/experiments/gsm8k/entropy_gate_eval.yaml \
  --devices 0 \
  --run-id gsm8k-entropy-q85 \
  --set model.load_model_path=/data/memgen-runs/train/<weaver-sft>/model \
  --set model.weaver.insertion_strategy.entropy_threshold=<CALIBRATED_VALUE>
```

初始对照应保持其余条件相同：

1. 无 inference augmentation；
2. `first_k`（前 5 个 delimiter）；
3. `candidate_sink_threshold`（现有首 key attention 分数）；
4. `candidate_entropy_threshold`。

所有设置均应使用相同 checkpoint、seed、解码温度、max response length 和
`max_inference_aug_num`。除 accuracy 外，报告每题的候选数、插入数、插入相对位置与
`entropy_gate_trace.csv` 中的 entropy 分布。
