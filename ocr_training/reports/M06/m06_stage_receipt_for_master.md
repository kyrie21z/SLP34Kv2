# M06 Stage Receipt For Master

## 1. 模块基本信息

- 模块名：`M06 = glyph-aware / substitution-aware`
- 当前回执范围：`P1 inference-only screening` + `P2A train-time substitution-aware / joint`
- 评测口径：统一沿用 baseline / M04-final 既有口径，不改 tokenizer / charset / test split / 正确样本定义
- 本回执用途：供主控做阶段性裁决，不新增实验结论

## 2. 本轮做了什么

- P1 只做了 `105-sample focus slice` 的 inference-only 轻干预筛查。
- P1 结论是：`glyph-only` 没有形成正信号，`substitution-aware / joint` 仅有极弱、局部、pair-level 正信号。
- P2A 从 `M04-final` checkpoint 起跑，做了短程微调版 `m06_sub_train` 与 `m06_joint_train`。
- P2A 同时完成了 `105-sample focus pilot` 和 `6884 unified full` 的同口径评测。
- 本步只把 reference、P1、P2A 证据压缩成可裁决回执，不再补做训练或探索。

## 3. 关键结果表

| 变体 | overall accuracy | replace | insert | delete | OOV | single | hard | long_21+ | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 16.19% | 193 | 18 | 19 | 25.37% | 16.19% | 5.26% | 0.00% | focus pilot |
| M04-final | 0.00% | 198 | 4 | 25 | 0.00% | 0.00% | 0.00% | 0.00% | focus pilot |
| P1 substitution-aware inference-only | 0.95% | 199 | 4 | 25 | 1.49% | 0.95% | 0.00% | 0.00% | focus pilot |
| P1 joint inference-only | 0.95% | 198 | 4 | 25 | 1.49% | 0.95% | 0.00% | 0.00% | focus pilot |
| P2A sub_train | 6.67% | 195 | 20 | 44 | 8.96% | 6.67% | 5.26% | 5.56% | focus pilot |
| P2A joint_train | 8.57% | 192 | 22 | 43 | 10.45% | 8.57% | 7.89% | 5.56% | focus pilot |
| baseline | 82.83% | 2276 | 2240 | 1920 | 61.54% | 81.98% | 62.93% | 78.20% | full unified |
| M04-final | 82.90% | 2347 | 2394 | 1702 | 61.48% | 82.37% | 63.73% | 78.84% | full unified |
| P2A sub_train | 81.94% | 2410 | 2278 | 1938 | 61.29% | 80.31% | 61.45% | 78.52% | full unified |
| P2A joint_train | 81.87% | 2420 | 2349 | 1866 | 60.86% | 80.46% | 60.95% | 78.12% | full unified |

P2A full unified 相对 `M04-final` 的关键退化：

- `overall`: `82.90% -> 81.94% -> 81.87%`
- `replace`: `2347 -> 2410 -> 2420`
- `OOV`: `61.48% -> 61.29% -> 60.86%`
- `single`: `82.37% -> 80.31% -> 80.46%`
- `hard`: `63.73% -> 61.45% -> 60.95%`
- `long_21+`: `78.84% -> 78.52% -> 78.12%`

## 4. 结果解释

### 已确认结论

1. P1 的弱正信号没有在 P2A `full unified` 上泛化。
2. 当前实现下，训练版 `substitution-aware` 与 `joint` 都未降低 `overall`，也未降低整体 `replace`。
3. `OOV` 在 `full unified` 上受伤，且 `single / hard / long_21+` 也同步下滑。
4. 当前 M06 不能按“主性能杠杆成立”汇报。

### 初步观察

1. focus pilot 的局部收益说明 pair-aware 方向不是完全无信息。P1 `sub/joint` 从 `0/105` 到 `1/105`，P2A `sub/joint` 到 `7/105` 和 `9/105`。
2. 但当前实现明显存在 `confusion-list overfit` 与 sequence-level 副作用。full unified 中确实压低了部分 top pairs，例如 `sub_train` 的 9->8 (-15), 6->8 (-7), 1->6 (-7), 8->3 (-5)，`joint_train` 的 9->8 (-10), 8->6 (-5), 1->8 (-5), 2->0 (-4)；同时又恶化了 9->0 (+12), 0->1 (+7), 8->9 (+6) 与 9->0 (+8), 8->9 (+7), 1->0 (+4) 这类关键 pair，导致净效果转负。
3. joint 在 pilot 略强，但在 full unified 更差，因此不能据此说 joint 更优。

### 尚未验证假设

1. 若显著减小 `lambda_sub` 或缩小 confusion top-K，是否能减轻副作用。
2. 是否存在更稳的“只约束极少数高置信高频 pair”的训练版本。
3. 是否值得单独拆出 anti-vocabulary-reliance / anti-overfit 辅助模块。

## 5. 风险与副作用

- 当前最大风险不是“没有 pair 信号”，而是 pair-level 约束在 full unified 上放大成 sequence-level 退化。
- `replace` 没有下降，说明当前收益/退化主轴仍然是字符替换问题没有被真正解决。
- `OOV` 下滑意味着存在 vocabulary reliance 加重的迹象。
- `joint` 比 `sub` 更容易把 pilot 局部收益包装成假正例，但 full unified 已说明它并不更稳。

## 6. 与主控目标的关系

- 主控优先关心 `replace`、top confusion pairs、`single / hard / long_21+ / OOV` 和 overall。
- 当前证据表明：局部 confusion pair 的改善不足以抵消整体 `replace`、overall 与 `OOV` 的退化。
- 因此，M06 在当前实现下没有满足“作为新的主性能杠杆成立”的标准。

## 7. 建议给主控的裁决

1. M06 当前不成立为新的主线模块。
2. 当前更适合把 M06 定义为“仅保留为失败后收缩的备选辅助方向”，而不是已成立的主贡献。
3. `substitution-aware` 若继续，只能作为非常小的收缩验证方向，不能继续按主线推进。
4. `joint` 不建议继续作为优先保留方向；若保留，也只能作为比 `substitution-aware` 更低优先级的备选辅助项。
5. 当前不建议继续深挖 M06 主线实现。
6. 下一步更合理的是归档本轮失败结果；若仍要继续，也只能做非常小的收缩验证，且明确为非主线。

## 8. 必交付文件

- `ocr_training/reports/M06/m06_stage_receipt_for_master.md`
- `ocr_training/results/M06/stage_main_results.csv`
- `ocr_training/results/M06/stage_error_op_breakdown.csv`
- `ocr_training/results/M06/stage_confusion_pairs_delta.csv`
- `ocr_training/results/M06/stage_subset_breakdown.csv`
- `ocr_training/figures/M06/stage_confusion_delta_heatmap.png`
- `ocr_training/figures/M06/stage_error_op_delta.png`
- `ocr_training/figures/M06/stage_subset_gain.png`

## 9. 一句话标准结论

M06 在当前 `substitution-aware / joint` 实现下，虽在 focus slice 暴露局部 pair-level 信号，但在 `6884 unified` 全量上 overall、replace 与 OOV 同步退化，暂不足以支持其升级为主线模块。
