# M06 P1 Inference-Only Screening

## 1. 本步目的

本步不是训练版 M06，也不是路线终裁。

目标只有两个：

1. 在推荐插点 `decode() -> self.head(...)` 之间做 inference-only 的最小 logits adjustment；
2. 快速判断 `glyph-aware`、`substitution-aware`、以及联合版，是否在高价值 replace focus slice 上出现值得进入训练版的信号。

本轮筛查样本不是全量 unified test，而是从 M05 中抽出的 105 个高价值 focus ids：

- `length_head` 下的 replace-dominant
- `visual_confusion / structure_confusion`
- `layout == single`
- 且命中 `hard / OOV / long_21+` 至少一项

因此本轮所有“overall”都指 **focus slice overall**，不是 6884 样本全量 overall。

## 2. 代码改动位置

修改文件：

1. [system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py)
   - 在 `tgt_out = self.decode(...)` 与 `self.head(tgt_out)` 之间加入可开关 logits adjustment。
   - 新增四种模式的开关组合：
     - no-op
     - glyph-aware
     - substitution-aware
     - glyph + substitution
   - 保持主前向路径不重构；所有模式都能完全关闭回到 no-op。

2. [evaluate_unified.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/evaluate_unified.py)
   - 新增 `--image_ids_file`，用于 focus slice 最小筛查。
   - 统一评测口径保持不变；只是增加样本过滤能力。

3. [analyze_m06_p1_screening.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/analyze_m06_p1_screening.py)
   - 汇总 reference / screening 结果。
   - 生成：
     - `p1_reference_main_results.csv`
     - `p1_screening_main_results.csv`
     - `p1_error_op_breakdown.csv`
     - `p1_confusion_pairs_delta.csv`
     - `p1_subset_breakdown.csv`
     - `p1_pred_vs_gt_alignment.jsonl`

## 3. 实现的轻干预模式

### no-op

- 所有 M06 bias flag 关闭。
- baseline 与 M04-final reference 直接复用已有统一评测结果文件，不重新跑 no-op。

### glyph-aware

- 使用当前 charset 索引空间构建固定 `glyph prior bank`。
- 该 prior bank 不是训练得到的，而是基于字符级视觉近邻先验构造的固定相似性矩阵。
- 当 top-1 与 top-k 候选之间落入高相似混淆邻域、且 margin 不大时：
  - 轻微压低 top-1 sink logits
  - 轻微提升相似 alternative logits

### substitution-aware

- confusion pair 来源优先读取 [confusion_pairs_topk.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/confusion_pairs_topk.csv) 的 overall pair。
- 若 csv 不可读，再回退到内置的高频 pair 默认表。
- 这是方向性的 pair-aware 调整：
  - 若 `src -> tgt` 是 M05 高频混淆，且 `tgt` 当前是 top-1、`src` 已在 top-k 中且 margin 不大；
  - 则轻微抑制 `tgt`，提升 `src`。

### glyph + substitution 联合版

- 先执行 glyph-aware，再执行 substitution-aware。
- 仍然是 inference-only，不改训练图，不引入额外样本定义。

## 4. confusion pair 来源

主来源：

- [confusion_pairs_topk.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/confusion_pairs_topk.csv)

主要使用的是 M05 已暴露出的高频 overall confusion，对应典型数字和少量字母混淆，例如：

- `8 -> 6`
- `6 -> 8`
- `1 -> 8`
- `1 -> 7`
- `0 -> 1`
- `1 -> 0`

这满足本步约束：pair list 来自已有 M05 分析，不引入词典或语义纠错。

## 5. 运行命令

reference 复用：

- baseline: [phase1_repro_baseline_eval_v3/samples.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase1_repro_baseline_eval_v3/samples.csv)
- M04-final: [phase2_1b_eval/samples.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/phase2_1b_eval/samples.csv)

GPU-only screening 命令核心形态：

```bash
cd /mnt/data/zyx/SLP34K/ocr_training
source /mnt/data/zyx/miniconda3/etc/profile.d/conda.sh
conda activate slpr_ocr

python evaluation/evaluate_unified.py <ckpt> \
  --device cuda \
  --batch_size 64 \
  --image_ids_file evaluation/results/M06/focus_single_hard_oov_long21_ids.txt \
  --output_dir evaluation/results/M06/<variant_dir> \
  use_length_head:bool=true \
  use_selective_eos_aware_decoding:bool=true \
  use_predicted_length_for_eos:bool=true \
  use_length_bucket_gating:bool=true \
  eos_long_seq_threshold:int=17 \
  eos_suppress_margin:int=2 \
  eos_neutral_margin:int=1 \
  eos_boost_margin:int=2 \
  eos_suppress_bias:float=-1.5 \
  eos_boost_bias:float=2.0 \
  use_uncertainty_conditioned_eos:bool=false \
  ...
```

其中 `...` 分别切换：

- glyph-aware
- substitution-aware
- joint

最后再运行：

```bash
python evaluation/analyze_m06_p1_screening.py ...
```

## 6. 关键结果表

reference：

| model | focus accuracy | replace_total | insert_total | delete_total | replace_share |
|---|---:|---:|---:|---:|---:|
| baseline_raw | 16.19% | 193 | 18 | 19 | 83.91% |
| m04_final_raw | 0.00% | 198 | 4 | 25 | 87.22% |

screening：

| model | focus accuracy | delta vs M04-final | replace_total | delta replace | replace_share |
|---|---:|---:|---:|---:|---:|
| m04_glyph | 0.00% | +0.00% | 200 | +2 | 87.34% |
| m04_substitution | 0.95% | +0.95% | 199 | +1 | 87.28% |
| m04_joint | 0.95% | +0.95% | 198 | +0 | 87.22% |

关键子集：

- `OOV`
  - `m04_final_raw`: 0.00%
  - `m04_substitution`: 1.49%
  - `m04_joint`: 1.49%
- `easy`
  - `m04_final_raw`: 0.00%
  - `m04_substitution`: 1.49%
  - `m04_joint`: 1.49%
- `hard`
  - 三个 screening 模式都没有把 `hard` 从 0 拉起来
- `long_21+`
  - 三个 screening 模式都没有把 `long_21+` 从 0 拉起来

top confusion pair 变化：

- `glyph-aware`
  - 改善：`1->7 (-1)`, `6->8 (-1)`, `8->3 (-1)`
  - 恶化：`1->8 (+1)`, `6->0 (+1)`, `7->1 (+1)`
- `substitution-aware`
  - 改善：`8->3 (-2)`, `8->6 (-1)`, `1->7 (-1)`, `6->8 (-1)`
  - 恶化：`0->1 (+1)`, `1->0 (+1)`, `7->1 (+1)`
- `joint`
  - 改善：`8->3 (-2)`, `8->6 (-1)`, `1->8 (-1)`
  - 恶化：`0->1 (+1)`, `1->0 (+1)`, `7->1 (+1)`

唯一明确修正样本：

- `image_id=6646`
  - `gt`: `德盛8868`
  - `m04_final_raw`: `德盛8668`
  - `m04_substitution`: `德盛8868`
  - `m04_joint`: `德盛8868`
  - 标签：`middle / OOV / len=6`

## 7. 结果解释

### 已确认结论

1. inference-only `glyph-aware` 没有形成正信号。
   - accuracy 没有提升
   - replace_total 反而 `198 -> 200`
   - `long_21+` 和 `OOV` 都没有得到实际收益

2. inference-only `substitution-aware` 有非常弱的正信号。
   - focus accuracy 从 `0/105 -> 1/105`
   - 纠正了一个典型数字混淆样本 `8668 -> 8868`
   - 一些 top confusion 被压低，尤其 `8->3`、`8->6`

3. `joint` 比 `glyph-only` 更稳，也比 `substitution-only` 更平衡。
   - accuracy 同样 `+1/105`
   - replace_total 没有继续上涨
   - 额外压低了 `1->8`

### 初步观察

1. 本轮信号几乎全部集中在短 OOV 数字段。
2. `hard`、`long_21+` 没有被拉起来，说明 inference-only 轻干预对真正困难样本的能力非常有限。
3. `glyph-aware` 固定 prior bank 过于粗糙，容易把一种混淆换成另一种混淆，而不是稳定降低 replace。

### 尚未验证

1. 尚未在 unified 全量 6884 样本上做同口径复核。
2. 尚未验证把 substitution-aware 变成训练期辅助损失后，是否能把 “修正 1 个短 OOV 数字段” 扩展成更稳定的 replace 下降。
3. 尚未验证 joint 版在训练期是否仍优于纯 substitution-aware。

## 8. 风险与局限

### inference-only 的局限

- 它只能在现有 top-k logits 上做局部搬移，不能从表征层真正提升易混字符分离度。
- 因此很容易出现：
  - 压下一个 pair
  - 但把错误转移到另一个相邻 pair

### OOV 风险

- 本轮没有看到 OOV 下滑。
- 相反，唯一修正样本来自 `OOV`，且 `OOV` 子集从 `0.00% -> 1.49%`。
- 但这个样本数太小，不能据此宣称 “对 OOV 有稳定帮助”。

### vocabulary reliance 风险

- substitution-aware 使用的是 M05 高频 confusion list。
- 这会带来明显的 `overfit-to-confusion-list` 风险：
  - 对已知 pair 有帮助
  - 对未进入 list 的新 pair 几乎无能为力
- 因此如果继续推进，更合理的方向是训练版辅助约束，而不是继续堆 inference-time pair list。

## 9. 对下一步训练版 M06 的建议

结论不是“glyph-aware 成立”，而是：

1. **不建议**继续推进当前 inference-only `glyph-aware` 固定 prior 方案。
   - 没有带来收益
   - 还放大了 overall replace

2. **可以保留** `substitution-aware`，但只建议把它作为训练版 M06 的辅助方向。
   - 原因不是它已经有效，而是它至少暴露出一点可解释的 pair-level 正信号

3. 若进入训练版，优先级建议：
   - 第一优先：`substitution-aware` 训练辅助项
   - 第二优先：`joint` 作为附带对照
   - 不建议：继续单独深挖当前 inference-only `glyph-aware`

对 5 个硬问题的明确回答：

1. inference-only glyph-aware 是否能压低 replace？
   - 不能。本轮 `replace_total` 从 `198 -> 200`，没有压低。

2. inference-only substitution-aware 是否能压低 top confusion pairs？
   - 能，但幅度很小。`8->3` 降了 `2`，`8->6` 降了 `1`，同时也引入了少量新 pair 恶化。

3. 哪种模式对 `single / hard / long_21+ / OOV` 更有帮助？
   - `single`：只有 `substitution/joint` 有极弱正信号。
   - `hard`：没有模式真正有帮助。
   - `long_21+`：没有模式真正有帮助。
   - `OOV`：`substitution/joint` 略好于 glyph-only。

4. 是否出现 OOV 下滑或 vocabulary reliance 迹象？
   - 没有出现 OOV 下滑。
   - 但存在明显的 confusion-list reliance 风险。

5. 哪种模式最值得进入下一步训练版实现？
   - `joint` 略优先，其次是 `substitution-aware`。
   - 原因：它们有微弱 pair-level 正信号，且 `joint` 没有像 glyph-only 那样放大 replace。
