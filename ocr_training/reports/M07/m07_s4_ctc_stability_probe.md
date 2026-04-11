# M07 S4: CTC Stability-Only Probe

## 1. 背景与目标

S4 的唯一目标是验证：

> 在不改 row-major、不改 decoder 结构、不加 reroute 的前提下，仅做最小 CTC 动力学稳定化，是否能把模型从“全空 / 超长”两端拉回中间可学习区间。

约束：

- 不进入 E2
- 不加 LM / lexicon / beam search
- 不改 tokenizer / charset / split / unified evaluation schema
- 不引入新的 neck / adapter 变体

---

## 2. S4 采用的稳定化手段

本轮只用一个最小组合：

1. `blank bias init = -1.5`
2. `logit_temperature = 1.5`（前向时 `logits / temperature`）

选择理由：

- `blank bias` 直接约束 blank 初始偏向，抑制“全 blank”塌缩。
- `temperature` 仅做数值平滑，避免 logit 过尖导致极端解码。
- 两项都属于动力学层面校准，不增加结构容量，不改变范式。

---

## 3. 代码改动

### 修改文件

- `ocr_training/strhub/models/maevit_ctc_probe/system.py`
  - 新增参数：`blank_bias_init`、`logit_temperature`
  - 支持 blank 类别 bias 初始化
  - 前向支持固定 temperature 缩放

### 新增文件

- `ocr_training/configs/model/maevit_ctc_probe_s4_stability.yaml`
  - 明确 `adapter_mode=none`、`sequence_neck=none`
  - 仅打开 S4 稳定化参数

---

## 4. 实际运行命令

```bash
# train: small budget
python train.py model=maevit_ctc_probe_s4_stability \
  trainer.gpus=1 data.num_workers=8 model.batch_size=32 \
  +trainer.max_steps=200 +trainer.limit_val_batches=20 \
  trainer.val_check_interval=50 run_group=M07 run_tag=s4_stability_probe
```

```bash
# sanity stability metrics
python evaluation/inspect_ctc_probe_stability.py \
  outputs/M07/maevit_ctc_probe/2026-04-11_07-36-40_s4_stability_probe/checkpoints/last.ckpt \
  --device cuda \
  --output results/M07/s4_stability_metrics.json
```

```bash
# unified evaluation
python evaluation/evaluate_unified.py \
  outputs/M07/maevit_ctc_probe/2026-04-11_07-36-40_s4_stability_probe/checkpoints/last.ckpt \
  --device cuda --batch_size 128 \
  --output_dir evaluation/results/M07/s4_stability_eval
```

```bash
# unified breakdown
python evaluation/analyze_m07_ctc_probe.py \
  --baseline evaluation/results/phase1_repro_baseline_eval_v3/samples.csv \
  --candidate evaluation/results/M07/s4_stability_eval/samples.csv \
  --results_dir results/M07/triage_s4 \
  --figures_dir figures/M07/triage_s4 \
  --candidate_tag s4_stability
```

---

## 5. Sanity 结果（动力学）

来源：`results/M07/s4_stability_metrics.json`

- `blank_token_rate = 0.0`
- `empty_prediction_ratio = 0.0`
- `avg_pred_len = 3.898`
- `pred_len` 主峰在长度 `1`（256 样本中 216 个）

结论：

- S4 已经脱离“全空 / 超长”两端极值
- 动力学层面确实被拉回到中间短串区间

---

## 6. Small Validation（unified 口径）

来源：

- `evaluation/results/M07/s4_stability_eval/samples.csv`
- `results/M07/triage_s4/main_results.csv`
- `results/M07/triage_s4/subset_breakdown.csv`
- `results/M07/triage_s4/error_op_breakdown.csv`

### 6.1 Overall

- `overall accuracy = 0.00% (0/6884)`
- `avg_pred_len = 4.784`
- `empty_prediction_ratio = 0.0`

### 6.2 关键子集

- `single = 0.0`
- `hard = 0.0`
- `OOV = 0.0`
- `long_21+ = 0.0`

### 6.3 error-op

- `replace = 8777`
- `insert = 18973`
- `delete = 80855`

---

## 7. 与 S0 对比

来源：`results/M07/triage_s4_vs_s0.csv`

- S0（E1）：
  - `avg_pred_len = 0.0`
  - 全空塌缩（`replace=0, insert=0, delete=94815`）
- S4：
  - `avg_pred_len = 4.784`
  - 变为非空短串，错误从纯 delete 变为 `replace + insert + delete` 混合

### 对比结论

- 是否回到中间区间：**是（动力学层面）**
- 是否仍性能塌缩：**是（accuracy 仍 0）**
- 是否出现非零局部正信号：**否（single/hard/OOV/long_21+ 全为 0）**

---

## 8. 结论与建议

### 已确认结论

S4 证明：

- 仅靠 CTC 稳定化可以显著改善“输出形态失稳”（不再全空/超长）
- 但不能单独带来可用识别能力（整体与关键子集仍全 0）

### 对 M07 的建议

- 建议：**收尾当前 M07（E1 线）并转向 M08**
- 理由：S4 已回答“稳定性是否主导”的问题，结果是“可稳但不可用”，继续在同约束下追加预算的价值较低。

### 一句话硬判断

**S4 证明当前 M07 在稳定性上可修复，但识别能力仍为硬负；可以收尾 M07，不建议继续主线推进。**

