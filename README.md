# SLP34Kv2 Research Workspace

本仓库当前用于 `SLP34K` 上的持续研究迭代（项目代号：`SLP34Kv2`）。

- 数据集：`SLP34K`
- 当前仓库目录：`SLP34K`（目录名保持不变）
- AAAI-25 原始公开说明：[`archive/README_AAAI25_original.md`](archive/README_AAAI25_original.md)

## 当前进度（更新于 2026-04-11）

### 阶段状态

| 阶段 | 主题 | 当前状态 | 结论 |
|---|---|---|---|
| Phase1 (repro v3) | baseline vs `+length-head` | 已完成 | `+length-head` 小幅提升 overall（`82.83% -> 82.87%`），但长度错误未下降（`9.69% -> 9.75%`） |
| M05 | replace 视觉混淆探针 | 已完成 | 主矛盾更接近字符级替换混淆（尤其 `single/hard/OOV/long_21+`） |
| M06 | substitution-aware / joint | 已完成并归档 | 在 `6884 unified` 全量上未形成净收益，`overall/replace/OOV` 退化，不作为主线 |
| M07 | decoder 范式 reroute（CTC probe） | 已完成并收尾 | S0/S1/S3/S4 均未恢复可用识别能力（overall 仍 `0.0%`），建议转入下一阶段 |

### M07 关键信息（收尾结论）

- 最小 CTC probe 与其 triage 变体（S1 adapter、S3 bilstm、S4 stability）均已完成。
- S4 能把“全空/超长”动力学拉回中间区间（`avg_pred_len: 0.0 -> 4.784`，`empty_prediction_ratio: 1.0 -> 0.0`），但准确率仍为 `0.0%`。
- 当前结论：`M07` 不再继续扩展预算，进入收尾状态。

## 当前主线判断

1. baseline 与 `+length-head` 仍是当前可用参考线。
2. M06 不作为主性能杠杆，保留为低优先级备选方向。
3. M07 已完成可行性与稳定性审计，但未形成可用识别能力；主线应转向下一模块（M08）。

## 关键入口

- 训练入口：[`ocr_training/train.py`](ocr_training/train.py)
- 常规评测入口：[`ocr_training/test.py`](ocr_training/test.py)
- 统一导表入口：[`ocr_training/evaluation/evaluate_unified.py`](ocr_training/evaluation/evaluate_unified.py)
- Phase1 报告生成：[`ocr_training/evaluation/generate_phase1_report.py`](ocr_training/evaluation/generate_phase1_report.py)
- M07 对比分析：[`ocr_training/evaluation/analyze_m07_ctc_probe.py`](ocr_training/evaluation/analyze_m07_ctc_probe.py)

## 关键实验产物索引

### Phase1（repro v3）

- baseline 样本：[`ocr_training/evaluation/results/phase1_repro_baseline_eval_v3/samples.csv`](ocr_training/evaluation/results/phase1_repro_baseline_eval_v3/samples.csv)
- `+length-head` 样本：[`ocr_training/evaluation/results/phase1_repro_length_head_eval_v3/samples.csv`](ocr_training/evaluation/results/phase1_repro_length_head_eval_v3/samples.csv)
- 汇总报告：[`ocr_training/evaluation/results/phase1_acceptance_repro_v3/phase1_report.md`](ocr_training/evaluation/results/phase1_acceptance_repro_v3/phase1_report.md)

### M05 / M06 / M07

- M05 报告：[`ocr_training/reports/M05/m05_replace_visual_confusion_probe.md`](ocr_training/reports/M05/m05_replace_visual_confusion_probe.md)
- M06 回执：[`ocr_training/reports/M06/m06_stage_receipt_for_master.md`](ocr_training/reports/M06/m06_stage_receipt_for_master.md)
- M07 审计报告：[`ocr_training/reports/M07/m07_decoder_paradigm_reroute.md`](ocr_training/reports/M07/m07_decoder_paradigm_reroute.md)
- M07 S4 稳定性报告：[`ocr_training/reports/M07/m07_s4_ctc_stability_probe.md`](ocr_training/reports/M07/m07_s4_ctc_stability_probe.md)
- M07 triage 汇总：[`ocr_training/results/M07/triage_summary.csv`](ocr_training/results/M07/triage_summary.csv)

## 当前可复现命令（最小）

### 1) baseline / +length-head 训练

```bash
cd ocr_training
python train.py model=maevit_infonce_plm dataset=SLP34K run_tag=baseline
python train.py model=maevit_infonce_plm dataset=SLP34K run_tag=baseline_length-head model.use_length_head=true
```

### 2) unified 导表

```bash
cd ocr_training
python evaluation/evaluate_unified.py <checkpoint_path> --device cuda --batch_size 128 --output_dir <output_dir>
```

### 3) 生成 Phase1 报告

```bash
cd ocr_training
python evaluation/generate_phase1_report.py \
  --baseline_csv evaluation/results/phase1_repro_baseline_eval_v3/samples.csv \
  --length_head_csv evaluation/results/phase1_repro_length_head_eval_v3/samples.csv \
  --output_dir evaluation/results/phase1_acceptance_repro_v3
```

### 4) M07 CTC probe 对比分析

```bash
cd ocr_training
python evaluation/analyze_m07_ctc_probe.py \
  --baseline evaluation/results/phase1_repro_baseline_eval_v3/samples.csv \
  --candidate evaluation/results/M07/s4_stability_eval/samples.csv \
  --results_dir results/M07/triage_s4 \
  --figures_dir figures/M07/triage_s4 \
  --candidate_tag s4_stability
```

## 目录说明

- [`ocr_training`](ocr_training)：主训练、评测与分析代码
- [`mae`](mae)：MAE 预训练代码
- [`archive`](archive)：历史文档归档
- [`image`](image)：README 图像资源

## 环境与输出约定

- 当前主要运行环境：`slpr_ocr`
- Hydra 输出目录：`ocr_training/outputs/<run_group>/<model>/<timestamp>_<run_tag>`
- 相关配置：[`ocr_training/configs/main.yaml`](ocr_training/configs/main.yaml)
