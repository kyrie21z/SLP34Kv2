# M06 P2A: 训练版 substitution-aware 主实现，joint 为辅

## 1. 本步目的

在不改 tokenizer / charset / test split / 评测口径的前提下，从 M04-final checkpoint 出发做短程微调，验证训练期 `substitution-aware` 辅助项是否能把 P1 里的 pair-level 弱信号放大成稳定收益；同时用一个轻量 `joint` 版本做对照。

本步判断优先级仍然是：

- `replace` 是否下降
- top confusion pairs 是否被压低
- `single / hard / long_21+ / OOV` 是否受益
- overall accuracy 是否优于此前 0.0x 级别改进

## 2. 改动总览

修改文件：

- `ocr_training/strhub/models/maevit_infonce_plm/system.py`
  - 新增训练期 `substitution-aware` margin 辅助项
  - 新增轻量 `joint glyph` prototype margin 辅助项
  - 所有新项均由 flag 控制，默认关闭时回到原始行为
- `ocr_training/train.py`
  - 新增 `pretrained` 非严格权重加载路径，用于从 M04-final checkpoint 起跑且不恢复旧 optimizer/trainer 状态
- `ocr_training/configs/model/maevit_infonce_plm.yaml`
  - 增加 M06 训练辅助项默认配置项，默认均为 `false`
- `ocr_training/evaluation/analyze_m06_p2a_results.py`
  - 统一汇总 pilot/full 的 accuracy、replace/insert/delete、subset、confusion pair delta、代表案例

本步新增交付：

- `ocr_training/results/M06/p2a_main_results.csv`
- `ocr_training/results/M06/p2a_error_op_breakdown.csv`
- `ocr_training/results/M06/p2a_confusion_pairs_delta.csv`
- `ocr_training/results/M06/p2a_subset_breakdown.csv`
- `ocr_training/results/M06/p2a_pred_vs_gt_alignment.jsonl`
- `ocr_training/reports/M06/p2a_representative_cases.md`

## 3. 实现的训练版变体

### raw

- 参考模型：`m04_final_raw`
- 结果来源：`ocr_training/evaluation/results/phase2_1b_eval/samples.csv`

### substitution-aware train

- 标记：`m06_sub_train`
- 训练期启用 `use_substitution_train_aux=true`
- 推理期不启用任何 P1 logits heuristic

### joint train

- 标记：`m06_joint_train`
- 在 `m06_sub_train` 基础上额外启用 `use_joint_glyph_train_aux=true`
- glyph 分支只作为局部 prototype margin 辅助，不改推理主路径

## 4. confusion pair 来源

confusion pair 来自：

- `ocr_training/evaluation/results/M05/confusion_pairs_topk.csv`

使用方式：

- 只读取 `subset_name=overall`
- 将 `src_char -> tgt_char` 视为 “GT 为 `src_char` 时容易被错成 `tgt_char`”
- 频次归一化后作为 pair weight

## 5. 辅助项定义

### substitution-aware

放置位置：

- `system.py` 的 `training_step()` 中，`out = self.decode(...)` 与 `step_logits = self.head(out)` 之后、总 loss 聚合之前

等价形式：

\[
L_{sub}=\sum_t\sum_{c\in C(y_t)} w_{y_t,c}\max(0,\ m-(z_{y_t}-z_c))
\]

本实现：

- `C(y_t)` 来自 M05 confusion list
- `m = 0.60`
- `\lambda_sub = 0.15`

原因：

- 直接对 GT 条件的易混淆负类拉开 margin，最接近本轮主假设
- 不改 decoder 结构，不引入推理期规则

### joint glyph

放置位置：

- 同样在 `training_step()`，与 `substitution-aware` 同层聚合

等价实现：

- 用归一化后的 `self.head.weight` 作为字符 prototype
- 在 glyph neighborhood 内约束 `cos(h_t, W_{y_t}) - cos(h_t, W_c)` 的 margin

\[
L_{joint}=\sum_t\sum_{c\in G(y_t)} \alpha_{y_t,c}\max(0,\ m_g-(s(h_t,y_t)-s(h_t,c)))
\]

本实现：

- `m_g = 0.35`
- `\lambda_joint = 0.05`

原因：

- 成本最低，不需要新分支参数
- 只在 glyph-neighbor 局部生效，避免回到 P1 的 fixed prior inference-only 路线

总 loss：

\[
L = L_{ocr} + 0.1L_{cross\_modal} + \lambda_{len}L_{len} + \lambda_{sub}L_{sub} + \lambda_{joint}L_{joint}
\]

## 6. 训练配置

起点 checkpoint：

- `ocr_training/outputs/new_oov/maevit_infonce_plm/2026-04-09_23-36-18_paper_baseline_repro_length-head/checkpoints/last.ckpt`

随机种子：

- `42`

共同训练配置：

- 单卡 GPU
- `batch_size=120`
- `accumulate_grad_batches=5`
- `max_steps=200`
- `val_check_interval=100`
- `use_length_head=true`

关键超参：

- `substitution_train_loss_weight=0.15`
- `substitution_train_margin=0.60`
- `joint_glyph_loss_weight=0.05`
- `joint_glyph_margin=0.35`

最佳 checkpoint：

- `m06_sub_train`
  - `epoch=3-step=178-val_accuracy=81.9436-val_NED=94.2122.ckpt`
- `m06_joint_train`
  - `epoch=3-step=178-val_accuracy=81.8565-val_NED=94.1995.ckpt`

## 7. pilot 结果

`pilot_focus_105`：

| model | acc | correct | replace | insert | delete |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m04_final_raw` | 0.00% | 0/105 | 198 | 4 | 25 |
| `m06_sub_train` | 6.67% | 7/105 | 195 | 20 | 44 |
| `m06_joint_train` | 8.57% | 9/105 | 192 | 22 | 43 |

pilot 观察：

- `joint` 在 focus slice 上优于 `sub`
- 两个训练版都显著高于 `m04_final_raw` 的 `0/105`
- 但 pilot 中 `delete` 明显上升，已经提示副作用

## 8. 6884 unified 全量结果

`full_unified_6884`：

| model | acc | correct | replace | insert | delete |
| --- | ---: | ---: | ---: | ---: | ---: |
| `m04_final_raw` | 82.90% | 5707/6884 | 2347 | 2394 | 1702 |
| `m06_sub_train` | 81.94% | 5641/6884 | 2410 | 2278 | 1938 |
| `m06_joint_train` | 81.87% | 5636/6884 | 2420 | 2349 | 1866 |

相对 `m04_final_raw`：

- `m06_sub_train`
  - overall `-0.96` pct-pt
  - replace `+63`
- `m06_joint_train`
  - overall `-1.03` pct-pt
  - replace `+73`

关键 subset：

- `single`
  - `82.37% -> 80.31% -> 80.46%`
- `hard`
  - `63.73% -> 61.45% -> 60.95%`
- `OOV`
  - `61.48% -> 61.29% -> 60.86%`
- `long_21+`
  - `78.84% -> 78.52% -> 78.12%`
- `9-12`
  - `81.07% -> 80.04% -> 80.04%`

replace 位置：

- `head`: `908 -> 972 -> 981`
- `middle`: `678 -> 665 -> 656`
- `tail`: `761 -> 773 -> 783`

## 9. 结果解释

### 已确认结论

1. 训练版 `substitution-aware` 没有降低全量 `replace`，反而增加了。
2. 它比 P1 inference-only 在 focus slice 上更有信号，但这个信号没有泛化到 6884 全量。
3. `joint` 并不比纯 `substitution-aware` 更稳；pilot 更好，但 full 更差。
4. 确实有部分 top confusion pairs 被压低，但整体 replace 没有因此下降。
5. `single / hard / long_21+ / OOV` 在 full 上都没有真实受益。
6. full 上出现了 OOV 下滑，且存在 confusion-list overfit 风险。

### 初步观察

pilot 与 full 明显反向：

- focus slice 上：
  - `sub`: `0/105 -> 7/105`
  - `joint`: `0/105 -> 9/105`
- full 上：
  - `82.90% -> 81.94% / 81.87%`

说明：

- 训练辅助项确实能在 replace-dominant focus slice 上放大局部信号
- 但同时把错误从局部字符混淆扩展成更广泛的 sequence-level 副作用，尤其是 `delete`、多布局样本和 OOV 样本

被压低的典型 top pairs：

- `sub_train`
  - `9->8: -15`
  - `6->8: -7`
  - `1->6: -7`
  - `5->6: -6`
  - `8->3: -5`
- `joint_train`
  - `9->8: -10`
  - `8->6: -5`
  - `1->8: -5`
  - `2->0: -4`
  - `1->6: -3`

同时恶化的 pairs 也很明显：

- `sub_train`
  - `9->0: +12`
  - `0->1: +7`
  - `8->9: +6`
  - `9->6: +6`
- `joint_train`
  - `9->0: +8`
  - `8->9: +7`
  - `1->0: +4`
  - `9->6: +4`

### 尚未验证

- 当前 full 退化究竟更偏向 `margin` 过强，还是 confusion list 覆盖面过窄
- 如果只保留极小的 top-pair 集，是否能减少副作用
- 更短训练或更小 `\lambda_sub` 是否能保留 pilot 信号而不伤 full

## 10. 风险与副作用

- 训练版辅助项目前存在明显 full-set 退化，不能作为当前主线方法直接推进
- OOV 在 full 上下降：
  - `61.48% -> 61.29% -> 60.86%`
- `single / hard` 同步下降，说明收益并不局限于某个 layout 主线
- representative cases 显示，部分 `multi` 样本出现了明显的长串扩写 / 截断副作用
- confusion list overfit 风险存在：
  - 部分指定 pairs 确实下降
  - 但新 pair 同时被放大，overall replace 反而更高

## 11. 对主控的阶段性建议

建议结论：

- 不建议把当前 P2A 结果整理成“模块成立”
- 更不建议直接把训练版 `substitution-aware` 或 `joint` 升为主线

如果主控只需要裁决证据，本步已经足够：

- `substitution-aware train`：局部有效，full 失败
- `joint train`：pilot 更强，但 full 仍失败且 OOV 更差

若还要继续 M06，应仅作为“失败后收缩”的最小后续：

- 降低 `\lambda_sub`
- 缩小 confusion pair 覆盖面到更保守的 top-K
- 避免当前这种会把局部 pair 信号放大成 sequence-level 副作用的设定

当前阶段更合适的主控结论：

- P2A 证明“训练版 substitution-aware/joint 在当前实现下不支持继续升级为主线”
- 可归档为：`pilot 有信号，但 6884 unified 不成立`
