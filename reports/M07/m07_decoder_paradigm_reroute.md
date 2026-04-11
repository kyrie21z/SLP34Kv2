# M07 Decoder Paradigm Reroute: Feasibility Audit

## 1. 工程入口定位

### 1.1 当前 baseline 训练入口

- 主训练入口：`ocr_training/train.py`
  - `main()` 用 Hydra 解析配置并实例化 model/datamodule，再调用 `trainer.fit(...)`。
  - 关键位置：
    - `hydra.utils.instantiate(config.model)`：模型构建入口
    - `hydra.utils.instantiate(config.data)`：数据入口
    - `trainer.fit(model, datamodule=datamodule, ckpt_path=config.ckpt_path)`：训练恢复入口
- 配置总入口：`ocr_training/configs/main.yaml`
  - 默认加载：
    - `model: maevit_infonce_plm`
    - `charset: SLP34K_568`
    - `dataset: SLP34K`
- baseline 模型配置：`ocr_training/configs/model/maevit_infonce_plm.yaml`
  - 当前为 `decode_ar: true`
  - 当前图像尺寸为 `img_size: [224, 224]`
  - 当前 encoder `embed_dim: 768`
- 数据集配置：`ocr_training/configs/dataset/SLP34K.yaml`
  - train: `SLP34K_lmdb_train`
  - val: `SLP34K_lmdb_test`
  - test: `SLP34K_lmdb_benchmark`

### 1.2 当前 baseline 的测试 / 推理 / unified evaluation 入口

- 常规 benchmark 测试入口：`ocr_training/test.py`
  - `load_from_checkpoint(...)` 恢复模型
  - `SceneTextDataModule.test_dataloaders(...)` 生成各 test subset dataloader
  - `model.test_step(...)` 复用 `BaseSystem._eval_step()`
- unified evaluation 入口：`ocr_training/evaluation/evaluate_unified.py`
  - 直接读 `unified_lmdb`
  - 输出统一 `samples.csv`
  - 当前 baseline / length-head / selective-EOS 的统一口径都依赖它

### 1.3 checkpoint 加载位置

- 训练 warm-start / 非严格恢复：
  - `ocr_training/train.py`
  - `load_pretrained_checkpoint_weights()` 读取 checkpoint 并 `strict=False` 加载到当前 model
- 推理/测试 checkpoint 恢复：
  - `ocr_training/strhub/models/utils.py`
  - `load_from_checkpoint()` 根据 checkpoint 路径分派到对应 `ModelClass.load_from_checkpoint(...)`
- encoder 预训练 MAE 权重加载：
  - `ocr_training/strhub/models/maevit_infonce_plm/system.py`
  - `torch.load(mae_pretrained_path)` 后 `mae_model.load_state_dict(checkpoint['model'], strict=False)`

### 1.4 tokenizer / charset 定义位置

- charset 配置：
  - `ocr_training/configs/charset/SLP34K_568.yaml`
  - `charset_train` 与 `charset_test` 当前相同
- tokenizer / charset adapter：
  - `ocr_training/strhub/data/utils.py`
  - `CharsetAdapter`
  - `Tokenizer`：AR / CE 路径
  - `CTCTokenizer`：CTC 路径
- BaseSystem 绑定 tokenizer：
  - `ocr_training/strhub/models/base.py`
  - `CrossEntropySystem` 使用 `Tokenizer`
  - `CTCSystem` 使用 `CTCTokenizer`

### 1.5 模型构建入口

- Hydra 实例化入口：
  - `ocr_training/train.py`
  - `hydra.utils.instantiate(config.model)`
- checkpoint / pretrained 创建入口：
  - `ocr_training/strhub/models/utils.py`
  - `_get_model_class()` 目前只分派到：
    - `maevit_infonce_plm`
    - `maevit_plm`
- baseline 实际模型类：
  - `ocr_training/strhub/models/maevit_infonce_plm/system.py`
  - `_target_: strhub.models.maevit_infonce_plm.system.Model`

### 1.6 当前 PARSeq / AR decoder 的实际实现文件与挂接位置

- 不是外部单独 `parseq.py` 文件。
- 当前 baseline 实际使用的是本仓库本地化的 PARSeq-style permuted AR decoder：
  - decoder layer / stack 定义：
    - `ocr_training/strhub/models/maevit_infonce_plm/modules.py`
    - `DecoderLayer`
    - `Decoder`
  - decoder 挂接：
    - `ocr_training/strhub/models/maevit_infonce_plm/system.py`
    - `self.decoder = Decoder(...)`
    - `self.head = nn.Linear(...)`
    - `self.text_embed = TokenEmbedding(...)`
  - permutation / AR masking：
    - `gen_tgt_perms()`
    - `generate_attn_masks()`
  - 推理时 AR decode loop：
    - `forward()` 中 `if self.decode_ar: ... for i in range(num_steps): ...`

### 1.7 预测字符串生成位置

- 通用字符串生成：
  - `ocr_training/strhub/models/base.py`
  - `_eval_step()` 中：
    - `probs = logits.softmax(-1)`
    - `preds, probs = self.tokenizer.decode(probs)`
    - `pred = self.charset_adapter(pred)`
- unified evaluation 导表链：
  - `ocr_training/evaluation/evaluate_unified.py`
  - `preds, _ = model.tokenizer.decode(probs)`
  - `pred = model.charset_adapter(pred_raw)`
  - 然后写入 `samples.csv`

### 1.8 unified evaluation 中 overall / subset / error-op 的统计入口

- prediction export 链：
  - `ocr_training/evaluation/evaluate_unified.py`
  - `unified_lmdb -> model.forward() -> tokenizer.decode() -> charset_adapter() -> rows -> samples.csv`
- overall / subset / error-op 的现成统计不在 `evaluate_unified.py` 内部完成，而是在导出的 `samples.csv` 之上做二次分析：
  - `ocr_training/evaluation/generate_phase1_report.py`
    - `overall_metrics()`
    - `attr_breakdown()`：quality/layout/vocabulary/resolution
    - `subset_metrics()`：`hard` / `OOV` / `long_21+`
    - `edit_op_stats()`：错误操作汇总
  - `ocr_training/evaluation/fine_grained_error_analysis.py`
    - `analyze_edit_operations()`：replace / insert / delete / eos_early / eos_late
- 结论：
  - overall / subset / error-op 已有现成实现。
  - 若后续上 CTC decoder，只要继续产出同口径 `samples.csv`，现有 unified evaluation 下游统计链可以直接复用。

## 2. decoder 挂接链路

### 2.1 训练链路

1. `SceneTextDataModule` 读 LMDB，返回 `(images, labels)`
2. `Model.training_step()`：
   - `tgt = self.tokenizer.encode(labels, self._device)`
   - `memory, cross_modal_loss = self.encode(images, labels)`
3. `gen_tgt_perms()` 生成 PARSeq permutation
4. `generate_attn_masks()` 生成 query/content mask
5. `out = self.decode(tgt_in, memory, ...)`
6. `step_logits = self.head(out)`
7. `F.cross_entropy(...)` 计算 OCR loss

### 2.2 推理链路

1. `memory = self.encode(images)`
2. `forward()` 进入 `decode_ar=True` 分支
3. 逐步自回归：
   - `tgt_out = self.decode(...)`
   - `step_logits = self.head(tgt_out).squeeze(1)`
   - greedy `argmax` 回填到 `tgt_in`
4. optional refine：
   - `refine_iters: 1`
5. 返回 `logits`
6. `tokenizer.decode(probs)` 生成字符串
7. `charset_adapter()` 过滤到 test charset

### 2.3 对 M07 的直接含义

- 当前 baseline 的 decoder 范式不是“简单 left-to-right AR”，而是：
  - 训练期：permuted AR / PARSeq-style
  - 推理期：canonical AR + refine
- 因此 M07 若要做 decoder paradigm reroute，最小切口不该从 `tokenizer/charset/unified eval` 下手，而该从：
  - `Model.encode()`
  - `forward()`
  - `training_step()`
  - `BaseSystem` 的 `CTCSystem`
  这条链替换 decoder/head/loss 即可。

## 3. unified evaluation 现状

### 3.1 现状总结

- 当前 unified evaluation 已经稳定形成两段式：
  - 第一段：`evaluate_unified.py` 导出统一 `samples.csv`
  - 第二段：`generate_phase1_report.py` / `fine_grained_error_analysis.py` 做 overall / subset / error-op
- 这意味着 M07 不需要改 unified evaluation 口径。
- 新 decoder 只要保证：
  - `model.forward(images)` 返回 `N x L x C`
  - `tokenizer.decode(...)` 能产出 `pred`
  - `evaluate_unified.py` 还能写出同 schema `samples.csv`
  即可完全复用下游分析。

### 3.2 当前 CSV 字段

- `evaluate_unified.py` 当前输出列：
  - `image_id, quality, layout, vocabulary_type, resolution_type`
  - `gt, pred, correct, error_type, note`
  - `gt_len, pred_text_len, eos_type, pred_len_from_head`

### 3.3 对 error-op 的判断

- 当前 error-op 不是在推理时即时统计。
- 当前是从 `samples.csv` 离线重算：
  - `generate_phase1_report.py -> edit_op_stats()`
  - `fine_grained_error_analysis.py -> analyze_edit_operations()`
- 所以 M07 若切 decoder，不需要先实现新的 error-op 逻辑。
- 需要守住的是 prediction export 链不能断。

## 4. feasibility audit 关键结论

### 4.1 A. 测试集标签长度分布

基于实际 LMDB 统计：

| Split | Samples | Mean Len | Median | Max Len | 9-12 bucket | 21+ | 24+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 27501 | 13.86 | 12 | 47 | 6471 (23.53%) | 5212 (18.95%) | 1837 (6.68%) |
| test / unified_test | 6884 | 13.77 | 12 | 47 | 1638 (23.79%) | 1243 (18.06%) | 436 (6.33%) |

补充：

- `test == unified_test` 的长度统计一致。
- train/test 的长度分布非常接近：
  - mean 都约 `13.8`
  - median 都是 `12`
  - max 都是 `47`
  - `21+` 都约 `18%`
  - `24+` 都约 `6%`
- 结论：
  - 长尾不是测试集特有问题，而是 train/test 同时存在的稳定分布。
  - M07 若上任何时序模型，必须优先检查 `T` 是否能覆盖 21+ / 24+ 长尾。

### 4.2 B. 当前 encoder 最终输出的 token / feature shape

代码依据：

- `img_size = [224, 224]`
- `patch_size = 16`
- `num_patches = 14 x 14 = 196`
- encoder 额外拼接 `cls_token`

因此 encoder 输出为：

- 训练时：
  - `memory, cross_modal_loss = self.encode(images, labels)` 注释已写明 `B x 197 x 768`
- 推理时：
  - `memory = self.encode(images)`，同一路径，同 shape

可行的 CTC 序列化长度：

- row-major flatten：
  - 去掉 cls 后 `14 x 14 = 196`
  - `T = 196`
- width-only sequenceization：
  - 沿高度聚合成 14 个列 token
  - `T = 14`

`T < gt length` 样本占比：

| 方案 | T | train 中 `gt_len > T` | test 中 `gt_len > T` |
|---|---:|---:|---:|
| row-major flatten | 196 | 0 / 27501 = 0.00% | 0 / 6884 = 0.00% |
| width-only sequenceization | 14 | 11587 / 27501 = 42.13% | 2888 / 6884 = 41.95% |

判断：

- 从纯长度可达性看：
  - row-major flatten 完全可行
  - width-only sequenceization 首轮不可作为默认主探针
- 宽度-only 会在训练一开始就对约 42% 样本形成硬上限，且长尾恰好是当前数据的重要组成部分。

### 4.3 C. canonical label order 与固定空间扫描顺序的相容性抽样判断

#### single

抽样观察：

- 多数 `single` 样本表现为单条横向序列，视觉顺序大致是“汉字块 + 数字块”。
- 但存在明显例外：
  - `id=453`, label=`浙余杭货01963`
  - 图上更接近“数字在左、汉字在右”的视觉布局
- 结论：
  - `single` 上 fixed-order CTC 不是严格无风险。
  - 但相较 `multi-line / vertical`，它仍是最接近单调顺序假设的子域。
  - 审计判断：`single` 上“roughly holds but noisy”，可作为首轮 probe 的主要可验证对象，但不能把全量 `single` 视为完美单调。

#### multi-line

抽样观察：

- `id=1`, label=`鲁菏泽货0846LUHEZEHUO`
  - 图像同时出现数字块、中文块、拼音块，空间上呈多块拼接，不是单一线性阅读路径。
- `id=2`, label=`万顺668WANSHUN`
  - 中文在左、数字在右、拼音压在中文下方。
  - canonical label 为“中文 -> 数字 -> 拼音”，而 row-major 扫描更可能变成“中文上半/拼音/数字”交织。
- 结论：
  - `multi-line` 存在显著 canonical-order 冲突风险。
  - 不适合作为 fixed-order CTC 首轮“成功与否”的主判断样本。

#### vertical

抽样观察：

- `id=992`, label=`上海港SHANGHAIGANG`
  - 图中汉字纵排，拼音又以更小字符贴在侧边。
  - canonical label 是“汉字串 -> 拼音串”，但固定 row-major 扫描会先遇到上方/侧边拼音，再逐行扫过纵排汉字。
- `id=993`, label=`九江JIUJIANG`
  - 同样出现纵排中文与侧边拼音共存。
- 结论：
  - `vertical` 与固定扫描顺序的冲突最明显。
  - 这里不仅是“左右顺序”问题，而是“阅读方向 + 多轨文本”同时冲突。

### 4.4 总结判断

- fixed-order CTC 在 `single` 上大致成立，但不是无噪声成立。
- `multi-line / vertical` 都有明显 canonical-order 冲突风险。
- 因此首轮 probe 的正确目标不是“立刻替换全域 decoder”，而是：
  - 先验证 encoder feature 在固定顺序时序化后，是否能在 `single` 与 overall 上保留可接受可学性；
  - 再决定是否需要 layout-aware reroute 或更强 2D-to-1D ordering 设计。

## 5. 推荐的 E1 最小实现方案

### 5.1 最小可复现的新范式方案

推荐 E1 方案：

- 保持现有 encoder、数据、charset、tokenizer 口径、unified evaluation 全不变
- 仅新增一个最小 `fixed-order CTC probe`
- encoder 输出采用：
  - 去 cls
  - `14 x 14` patch token 做 row-major flatten
  - 接一个最小线性 CTC head
- 训练基类复用 `CTCSystem`
- 推理使用 greedy CTC decode

一句话定义：

- `MAE encoder + row-major 2D token flatten + linear CTC head + existing unified evaluation`

### 5.2 为什么首轮优先做 fixed-order CTC probe

1. 它回答的是最核心的范式问题，而不是优化问题。
   - 先看“固定顺序时序化”是否有基本可学性。
2. 它改动面最小。
   - 不动 tokenizer/charset/eval schema
   - 不引入 LM / beam / lexicon
   - 不需要多 decoder 并行分叉
3. 它最容易和当前 baseline 做 apples-to-apples 对比。
   - 同 encoder
   - 同 train/test split
   - 同 unified evaluation
4. 它能最快暴露真正瓶颈是在：
   - 时序长度
   - order mismatch
   - 还是 decoder capacity
5. 当前审计已经证明：
   - 宽度-only `T=14` 先天吃亏
   - row-major `T=196` 长度完全够
   - 所以最小 probe 应先让“长度不是瓶颈”

### 5.3 需要复用哪些现有模块

- `SceneTextDataModule`
- `SLP34K_568` charset 配置
- `CTCTokenizer` / `CTCSystem`
- 现有 MAE encoder：
  - `strhub.models.models_mae`
  - `maevit_infonce_plm.system.Model.encode()` 中的 encoder 构建和 MAE 权重加载逻辑
- `load_from_checkpoint()` / Hydra 配置体系
- `evaluate_unified.py`
- `generate_phase1_report.py`
- `fine_grained_error_analysis.py`

### 5.4 需要新增哪些最小模块

- 一个新 model 文件，建议独立，不侵入现有 baseline：
  - 例如 `ocr_training/strhub/models/maevit_ctc_probe/system.py`
- 最小模块内容：
  - encoder 复用 MAE
  - `row_major_sequenceize(memory)`：
    - 去 cls
    - reshape 为 `B x 14 x 14 x 768`
    - flatten 为 `B x 196 x 768`
  - `ctc_head = nn.Linear(768, num_classes)`
  - `forward(images)` 返回 `B x T x C`
- 一个最小 model yaml：
  - 例如 `ocr_training/configs/model/maevit_ctc_probe.yaml`

### 5.5 训练脚本 / 测试脚本 / 导表脚本最小需要改哪些地方

训练脚本：

- `train.py` 本身不一定要改。
- 只要新增 model config，并让 Hydra 能实例化新 `_target_` 即可。

测试脚本：

- `test.py` 原则上不一定要改。
- 前提是新 model 也走 `load_from_checkpoint()`，并且 `BaseSystem.test_step()` + `CTCTokenizer.decode()` 能直接工作。

导表脚本：

- `evaluate_unified.py` 原则上也不一定要改。
- 前提是：
  - `model.forward(imgs)` 仍输出 `N x L x C`
  - `model.tokenizer.decode(probs)` 仍可直接得到字符串
- 可能需要的唯一小改动：
  - 若导表脚本硬依赖 `model.encode()` 后再额外访问 `length_head(memory)`，需要保证无 `length_head` 时仍按当前 `-1` 分支处理。
  - 当前脚本其实已经支持 `has_head = False`，所以大概率无需改。

## 6. 下一步最小代码改动清单

仅列 E1 必需改动，不实现：

1. 新增 `ocr_training/strhub/models/maevit_ctc_probe/system.py`
   - 复用 MAE encoder 初始化与预训练加载
   - 继承 `CTCSystem`
   - 新增 row-major sequenceize
   - 新增 CTC linear head

2. 新增 `ocr_training/configs/model/maevit_ctc_probe.yaml`
   - `img_size=[224,224]`
   - `embed_dim=768`
   - `mae_pretrained_path` 复用当前 baseline

3. 更新 `ocr_training/strhub/models/utils.py`
   - `_get_model_class()` 增加 `maevit_ctc_probe`

4. 不改 `charset`
   - 不改 `SLP34K_568.yaml`

5. 不改数据切分
   - 不改 `SLP34K_lmdb_train / test / unified_lmdb`

6. 不改 unified evaluation schema
   - 继续输出同格式 `samples.csv`
   - 继续复用 `generate_phase1_report.py` 与 `fine_grained_error_analysis.py`

7. E1 验证顺序
   - 先恢复训练可跑
   - 再恢复 `evaluate_unified.py` 可导表
   - 再对 `overall / single / multi / vertical / long_21+` 做统一口径审计

## 最终结论

- 工程入口当前是完整可恢复的，且训练、测试、unified evaluation 入口都已明确。
- M07 首轮不应直接做多 decoder 并行试验。
- 首轮最小可复现替换方案应是：
  - `fixed-order row-major CTC probe`
- 原因不是它一定最好，而是它能以最小工程改动，最快回答：
  - 当前 encoder feature 是否支持“非 AR、固定顺序”的可学解码。
- 风险判断：
  - 纯长度上 row-major 可行，width-only 不可优先
  - 真正风险在 canonical order mismatch
  - 该风险在 `single` 可容忍、在 `multi-line / vertical` 明显
- 因此 E1 的价值是：
  - 用最小改动先测范式上限和失败模式
  - 而不是提前承诺全域替换成功
