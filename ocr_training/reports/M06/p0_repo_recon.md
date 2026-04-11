# M06 P0 Repo Recon

## 1. 仓库关键路径表

| 文件路径 | 类/函数名 | 作用 | 与 M06 的相关性 |
|---|---|---|---|
| [train.py](/mnt/data/zyx/SLP34K/ocr_training/train.py#L58) | `main(config)` | Hydra 训练入口，实例化 `config.model` 和 `SceneTextDataModule`，再 `trainer.fit()` | 确认 M06 若走训练线，真实入口就是这里，不需要新建训练主线 |
| [configs/main.yaml](/mnt/data/zyx/SLP34K/ocr_training/configs/main.yaml) | `defaults/model/data/trainer` | 主配置装配点 | M06 若增加最小开关，优先经这里和 model yaml 注入 |
| [configs/model/maevit_infonce_plm.yaml](/mnt/data/zyx/SLP34K/ocr_training/configs/model/maevit_infonce_plm.yaml) | `_target_: ...system.Model` | 当前主模型配置 | 说明当前实际主模型是 `strhub.models.maevit_infonce_plm.system.Model` |
| [strhub/models/utils.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/utils.py#L42) | `_get_model_class`, `load_from_checkpoint`, `parse_model_args` | 训练/测试统一的模型恢复入口 | M06 若加推理开关，可直接走现有 `name:type=value` CLI 覆盖机制 |
| [strhub/models/maevit_infonce_plm/system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L185) | `class Model` | 当前主模型类 | M06 主插点的核心文件 |
| [strhub/models/maevit_infonce_plm/system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L362) | `encode()` | 图像编码，返回 encoder memory；训练时还返回 cross-modal loss | 若 M06 想做 encoder 侧 glyph prototype，不是最小插点 |
| [strhub/models/maevit_infonce_plm/system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L377) | `decode()` | 生成 decoder 输出 `query` hidden states | 这是 decoder hidden states 最直接可得的位置 |
| [strhub/models/maevit_infonce_plm/system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L561) | `forward()` | 测试时 AR decoding + logits 生成 | M06 若先做最小可复现推理验证，最适合在这里插 |
| [strhub/models/maevit_infonce_plm/system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L781) | `training_step()` | 训练时 loss 汇总 | M06 若走训练辅助损失，这里是最小聚合点 |
| [strhub/models/maevit_infonce_plm/modules.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/modules.py#L96) | `DecoderLayer` | decoder layer 定义 | 用于理解 hidden states 来源 |
| [strhub/models/maevit_infonce_plm/modules.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/modules.py#L152) | `Decoder.forward()` | 堆叠 decoder layers 并输出最终 `query` | 说明 final decoder hidden state 已存在，不需要重构 |
| [strhub/data/utils.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/data/utils.py#L25) | `CharsetAdapter` | test charset 过滤/适配 | M06 不应改这里，避免口径漂移 |
| [strhub/data/utils.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/data/utils.py#L102) | `Tokenizer` | train tokenizer，定义 `BOS/EOS/PAD`、encode/decode | M06 不应改 tokenizer |
| [configs/charset/SLP34K_568.yaml](/mnt/data/zyx/SLP34K/ocr_training/configs/charset/SLP34K_568.yaml) | `charset_train`, `charset_test` | 训练/测试字符集配置 | M06 必须复用，不应改 charset |
| [test.py](/mnt/data/zyx/SLP34K/ocr_training/test.py#L65) | `main()` | 标准 benchmark/test 入口 | 可作为通用测试入口参考，但不是当前 unified 对比主入口 |
| [evaluation/evaluate_unified.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/evaluate_unified.py#L190) | `main()` | baseline、length-head、M04-final 当前统一口径评测入口 | M06 若要和 baseline、M04-final 对齐，必须走它 |
| [evaluation/analyze_m05_alignment.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/analyze_m05_alignment.py#L66) | `levenshtein_alignment()` | pred-vs-gt 对齐回溯 | M06 结果分析可直接复用 |
| [evaluation/analyze_m05_alignment.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/analyze_m05_alignment.py#L235) | `analyze_rows()` | replace/insert/delete、cause、confusion 统计 | M06 结果分析可直接复用 |
| [evaluation/results/M05](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05) | `error_op_breakdown.csv` 等 | 已落地的 M05 误差证据链 | M06 可直接复用口径与字段定义 |

## 2. 模型前向路径简述

当前主模型是 [system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L185) 中的 `Model`。

训练链路：

1. [train.py](/mnt/data/zyx/SLP34K/ocr_training/train.py#L91) 通过 Hydra 实例化 `config.model`。
2. `Model.training_step()` 在 [system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L781) 读取 batch。
3. `self.tokenizer.encode(labels, self._device)` 生成 `tgt`。
4. `memory, cross_modal_loss = self.encode(images, labels)`，其中 `memory` 是 encoder output。
5. `out = self.decode(...)` 得到 decoder 最后一层 hidden states。
6. `logits = self.head(out)` 生成 classifier logits。
7. OCR loss 来自 `F.cross_entropy(logits, tgt_out)`；如果启用 length head，则再加 `length_loss`；总 loss 为：
   - `loss = ocr_loss + 0.1 * cross_modal_loss`
   - `if use_length_head: loss += length_loss_weight * length_loss`

测试/推理链路：

1. [evaluation/evaluate_unified.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/evaluate_unified.py#L224) 用 `load_from_checkpoint()` 恢复模型。
2. `memory = model.encode(imgs)`。
3. `logits = model.forward(imgs)`。
4. 在 `Model.forward()` 中，AR decode 循环每一步先 `tgt_out = self.decode(...)`，再 `step_logits = self.head(tgt_out).squeeze(1)`。
5. 最后 `tokenizer.decode(probs)` 生成字符串预测，`CharsetAdapter` 再做 test charset 过滤。

结论：

- decoder 最后一层 hidden state 已经存在，而且在训练和推理两条路径里都能在 `self.head(...)` 之前直接拿到。
- 当前并不缺 hidden states，本质上缺的是“对外暴露/复用该张量的最小接口”。

## 3. M06 最小插点建议

### 推荐插点

推荐插点：`strhub/models/maevit_infonce_plm/system.py` 中 `decode()` 之后、`self.head(...)` 之前。

具体落点有两个等价最小点：

- 推理线：[system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L612) 到 [system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L620)
- 训练线：[system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L797) 到 [system.py](/mnt/data/zyx/SLP34K/ocr_training/strhub/models/maevit_infonce_plm/system.py#L801)

原因：

1. hidden states 已可直接拿到，不需要改 decoder 结构。
2. 距 classifier 最近，最适合做 glyph-aware / substitution-aware 的最小附加头、相似性约束或 confusion-aware logit regularization。
3. 不需要改 tokenizer、charset、dataset、评测口径。
4. 同一逻辑可同时支持：
   - P1 最小推理验证：对 logits 做轻量 reweight / rerank
   - P2 训练验证：基于 hidden states 或 logits 加 substitution-aware 辅助损失

### 备选插点

1. `decode()` 函数本身返回值扩展
   - 现状：`decode()` 已返回 final decoder hidden states。
   - 若上层想显式消费，可最小改法是给 `forward()` 增加 `return_hidden=False`，在推理时一并返回 `logits, hidden_states`。
   - 这是可行备选，但不是必须，因为训练/推理内部已经能拿到 `tgt_out`。

2. `training_step()` 的 loss 汇总处
   - 适合把 `glyph/substitution-aware loss` 作为第三个训练项接入。
   - 最小方式是在 `ocr_loss` 之后、`loss` 汇总之前加一个新项，不动 datamodule 和 tokenizer。
   - 这是训练版最小插点，但不适合作为 P0/P1 的首个实现点，因为需要重新训练。

3. `evaluate_unified.py`
   - 适合作为实验开关和统一评测入口。
   - 但它不应该承载核心算法本体；最好只负责把 M06 开关透传给模型，并维持统一口径输出。

### 不推荐插点及原因

1. Encoder 侧
   - `encode()` 太靠前，改动会把 M06 从“字符级 substitute/glyph 干预”扩大成表征重写。
   - 风险：影响范围过大，不符合最小可复现验证。

2. Tokenizer / charset
   - 会直接破坏与 baseline、M04-final 的统一口径比较。
   - 违反本轮全局约束。

3. `CharsetAdapter` 或字符串后处理层
   - 太靠后，容易退化成规则纠错。
   - 很难证明收益来自 glyph-aware / substitution-aware 主方法，而不是后处理。

4. `evaluation/analyze_m05_alignment.py`
   - 这是分析脚本，不是模型行为发生的位置。
   - 适合复用统计，不适合承载方法实现。

## 4. M05 分析能力复用情况

结论：可直接复用，且复用价值高。

已存在能力：

1. pred-vs-gt 对齐
   - [analyze_m05_alignment.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/analyze_m05_alignment.py#L66)
   - 标准 Levenshtein alignment，保留 `M/R/I/D` 回溯路径。

2. replace/insert/delete 统计
   - [analyze_m05_alignment.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/analyze_m05_alignment.py#L235)
   - 已统计 `num_replace / num_insert / num_delete / eos_flag / dominant_error_type`。

3. confusion pair 统计
   - 同脚本后续 `confusion_pairs()`、`case_inventory()`、`overall_breakdown()`、`subset_breakdown()`。
   - 已经生成：
     - [error_op_breakdown.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/error_op_breakdown.csv)
     - [subset_breakdown.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/subset_breakdown.csv)
     - [confusion_pairs_topk.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/confusion_pairs_topk.csv)
     - [case_inventory.csv](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/case_inventory.csv)
     - [pred_vs_gt_alignment.jsonl](/mnt/data/zyx/SLP34K/ocr_training/evaluation/results/M05/pred_vs_gt_alignment.jsonl)

4. baseline / M04-final / length-head 三方对齐口径
   - M05 脚本默认就读三份 `samples.csv` 进行统一统计。
   - M06 最小实现后，只需要增加一份新的 `samples.csv` 输入，几乎不必重写统计逻辑。

## 5. 当前阻塞点

1. `forward()` 没有公开返回 hidden states
   - 内部其实拿得到 `tgt_out`，但对外 API 只有 logits。
   - 若后续要做更系统的 hidden-state 诊断，最小改法是给 `forward(return_hidden=False)` 增加可选返回。

2. 统一评测脚本对工作目录敏感
   - [evaluate_unified.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/evaluate_unified.py#L224) 里 `mae_pretrained_path='pretrain_model/ship/...'` 是相对路径。
   - 从仓库根目录调用会找不到预训练权重；默认假设 cwd 为 `ocr_training/`。

3. 当前环境是 CPU-only
   - 轻量 smoke run 可以做，但 6884 样本 unified 全量评测耗时很高。
   - 这不影响 P0 仓库定位，但会限制 P1/P2 的验证节奏。

4. 当前工作树本身是脏的
   - `evaluate_unified.py`、`system.py`、model yaml 等文件已存在未提交变更。
   - 后续做 M06 最小实现时，需要严格控制改动范围并避免覆盖现有实验分支。

## 6. 下一步最小实现建议

P1 建议不直接上训练，而是先做“最小推理插点验证”：

1. 在 `system.py` 中，把 M06 插在 `tgt_out = self.decode(...)` 和 `self.head(tgt_out)` 之间。
2. 先实现可开关的 `glyph-aware / substitution-aware logit adjustment`，只改 inference path。
3. 统一评测仍走 [evaluate_unified.py](/mnt/data/zyx/SLP34K/ocr_training/evaluation/evaluate_unified.py)，不改 test split、不改 tokenizer、不改 charset。
4. 结果分析直接复用 M05 对齐脚本，新增一份 M06 `samples.csv` 即可。
5. 如果推理版在 `replace_total / top confusion pairs / single-hard-long_21+-OOV` 上没有明确收益，再决定是否进入训练版辅助损失。

最小 API 改法建议：

- `Model.forward(..., return_hidden=False)` 可选返回 hidden states
- 或更小：只在内部先做 logits 级干预，不改公开接口

推荐优先级：

1. `decode() -> hidden -> head()` 之间做 inference-only 轻干预
2. 若有效，再在 `training_step()` 同位置补训练辅助损失
3. 不要先动 encoder、tokenizer、charset、后处理规则层
