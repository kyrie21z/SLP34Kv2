# M05 Replace Visual Confusion Probe

## 1. 模块基本信息

- 模块编号：`M05`
- 模块主题：基于统一样本表的字符级替换混淆探针
- 输入文件：
  - `evaluation/results/M05/error_op_breakdown.csv`
  - `evaluation/results/M05/subset_breakdown.csv`
  - `evaluation/results/M05/confusion_pairs_topk.csv`
  - `evaluation/results/M05/case_inventory.csv`
  - `evaluation/results/M05/pred_vs_gt_alignment.jsonl`
  - `figures/M05/op_distribution.png`
  - `figures/M05/error_by_subset.png`
  - `figures/M05/confusion_heatmap.png`
- 比较对象：
  - `baseline`
  - `length_head`
  - `selective_eos_final`

## 2. 本轮做了什么

- 仅使用三份标准化 `samples.csv` 副本进行离线分析，没有重新推理，没有改模型代码。
- 对全部样本重新做了标准 Levenshtein alignment，并保留 `M/R/I/D` 回溯路径。
- 派生了样本级统计：`num_replace / num_insert / num_delete / eos_flag / dominant_error_type / provisional_cause`。
- 输出了总表、子集表、top confusion pairs、代表样本清单和三张图。
- `error_type` 只保留作参考列，没有作为主分析标签。

## 3. 关键结果表

### 3.1 Overall Error Breakdown

| model | accuracy | replace_total | insert_total | delete_total | eos_early | eos_late | replace_share | insert_share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 82.83% | 2276 | 2240 | 1920 | 139 | 165 | 35.36% | 34.80% |
| length_head | 82.87% | 2337 | 2275 | 1861 | 138 | 160 | 36.10% | 35.15% |
| selective_eos_final | 82.90% | 2347 | 2394 | 1702 | 128 | 165 | 36.43% | 37.16% |

### 3.2 Replace-Rich Subsets

| subset | baseline replace_share | length_head replace_share | selective_eos_final replace_share | note |
|---|---:|---:|---:|---|
| layout = single | 48.55% | 51.67% | 52.82% | replace 最集中 |
| quality_group = hard | 36.92% | 37.59% | 38.37% | replace 稳定偏高 |
| long_21p = true | 31.50% | 33.91% | 36.22% | 长序列中 replace 持续上升 |
| vocab = OOV | 32.84% | 33.77% | 34.16% | OOV 中 replace 更稳定 |
| layout = vertical | 16.96% | 22.78% | 22.78% | vertical 不是 replace 富集区 |

### 3.3 Top Confusion Pairs

overall top confusion pairs:

1. `6 -> 8` (`166`)
2. `8 -> 6` (`135`)
3. `9 -> 8` (`117`)
4. `8 -> 9` (`77`)
5. `1 -> 0` (`70`)

这些 top pair 以数字对数字的替换为主，说明当前高频混淆更像字符级视觉/字形可分性问题，而不是全局顺序问题。

## 4. 结果解释（已确认结论 / 初步观察 / 尚未验证假设）

### 4.1 已确认结论

- overall 上，`replace` 不是压倒性高于所有其他误差；它与 `insert` 很接近。
- 在 `baseline` 和 `length_head` 中，`replace` 是最高误差族；在 `selective_eos_final` 中，`insert` 略高于 `replace`。
- 但在高价值子集里，`replace` 更稳定、更集中、更可干预，尤其是：
  - `single`
  - `hard`
  - `long_21+`
  - `OOV`
- `vertical` 不是当前 replace 富集区。它的 accuracy 本身较高，replace_share 也明显低于 `single / hard / OOV / long_21+`。
- top confusion pairs 以数字混淆为主，说明当前更像字符级视觉/字形混淆主导。
- 位置上，replace 在三模型中都更偏 `head`，大约 `38.5% - 38.9%`，`middle` 最少，`tail` 次之。

### 4.2 初步观察

- `selective_eos_final` 相比 `length_head`，`eos_early` 略降，但 `eos_late` 没有同步消失，且 insert 反而更高。
- 这意味着 EOS 线能影响长度型错误，但没有改变当前主误差结构。
- `OOV` 子集里也能看到字母对混淆，例如 `I -> G`，但高频主峰仍然是数字对数字。
- `single` 子集的 replace_share 最高，说明当前主矛盾不在复杂布局，而在局部字符辨识。

### 4.3 尚未验证假设

- `language_prior_suspect` 在部分 OOV 样本上出现，但现有证据不足以将 vocabulary reliance 视为第一主因。
- `layout_or_order_issue` 在部分 `multi` / `vertical` 样本中可见，但频率和集中度都不足以成为当前主线。
- `restoration` 方向可能帮助极端退化样本，但目前没有证据表明它优先级高于字符级替换混淆处理。

### 4.4 Sanity Note

- 原始 `quality` 字段唯一值为：`easy / hard / middle`。
- 本轮主统计中的 `quality_group` 规则是：`quality == "hard"` 记为 `hard`，其余值统一并入 `easy`。
- 因此 `middle` 没有单独作为主汇总组，而是并入了 `easy`；这是一种聚合规则，不是说原字段只有两种唯一值。

## 5. 风险与副作用

- 若继续把主要资源投入 EOS 线，风险是继续改善一小部分长度型错误，但不能改变主误差族结构。
- 若过早把 vertical / complex-layout 设为第一干预点，风险是把资源投入到当前非主矛盾区域。
- 若直接把 OOV 问题解释为 language prior 主导，风险是误判根因，因为当前 top confusion 仍以字符级视觉混淆为主。
- 代表样本中仍有 `severe_degradation` 类极端失败，这类样本不会仅靠简单替换约束完全解决。

## 6. 与主控目标的关系

- 主控当前需要决定下一正式模块的主线干预方向。
- M05 的价值在于先把“主错误族”和“高价值子集”锁定。
- 当前证据表明：下一步更值得投入的是稳定、可重复、字符级的替换混淆，而不是继续围绕 EOS 或布局复杂度展开。

## 7. 建议给主控的裁决

- M05 结果不支持继续把主要资源投入 EOS 线。
- M05 结果也不支持把 vertical / complex-layout 作为当前第一干预点。
- 当前更应优先处理稳定的字符级替换混淆。
- 下一正式模块建议立项为 `glyph-aware / substitution-aware`。
- `anti-vocabulary-reliance` 与 `restoration` 可以作为次级备选，而不是主线首选。

更具体地说：

- overall 上，replace 不是压倒性高于 insert/delete/eos，但它是主导误差族之一。
- 在 `single / hard / long_21+ / OOV` 中，replace 更稳定、更集中、更可干预。
- 当前 replace 更像字符级视觉/字形混淆主导。
- `layout/order` 不是主导。
- `vocabulary reliance` 目前证据不足以成为第一主因。

## 8. 必交付文件

- [error_op_breakdown.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/error_op_breakdown.csv)
- [subset_breakdown.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/subset_breakdown.csv)
- [confusion_pairs_topk.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/confusion_pairs_topk.csv)
- [case_inventory.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/case_inventory.csv)
- [pred_vs_gt_alignment.jsonl](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/pred_vs_gt_alignment.jsonl)
- [op_distribution.png](/mnt/data/zyx/SLP34K/ocr_training/figures/M05/op_distribution.png)
- [error_by_subset.png](/mnt/data/zyx/SLP34K/ocr_training/figures/M05/error_by_subset.png)
- [confusion_heatmap.png](/mnt/data/zyx/SLP34K/ocr_training/figures/M05/confusion_heatmap.png)
- [representative_cases.md](/mnt/data/zyx/SLP34K/ocr_training/reports/M05/representative_cases.md)

## 9. 一句话标准结论

M05 不支持继续以 EOS 或 vertical 复杂布局作为主线，当前更应优先立项 `glyph-aware / substitution-aware`，去处理稳定富集在 `single / hard / long_21+ / OOV` 中的字符级替换混淆。
