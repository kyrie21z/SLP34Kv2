# SLP34K Error Analysis Report

## 1. Overall Performance

| Metric | Value |
|--------|-------|
| Total Samples | 6884 |
| Correct Predictions | 5726 |
| Overall Accuracy | 83.18% |

## 2. Accuracy by Difficulty (Quality)

| Quality | Accuracy | Correct/Total |
|---------|----------|---------------|
| easy     |  94.72% | 1489/1572 |
| middle   |  83.62% | 3598/4303 |
| hard     |  63.33% | 639/1009 |

## 3. Accuracy by Layout (Structure)

| Layout | Accuracy | Correct/Total |
|--------|----------|---------------|
| single   |  82.62% | 1687/2042 |
| multi    |  80.27% | 3088/3847 |
| vertical |  95.58% | 951/995 |

## 4. Accuracy by Vocabulary Type

| Type | Accuracy | Correct/Total |
|------|----------|---------------|
| IV   |  89.92% | 4745/5277 |
| OOV  |  61.05% | 981/1607 |

## 5. Accuracy by Resolution Type

| Resolution | Accuracy | Correct/Total |
|------------|----------|---------------|
| normal   |  94.72% | 1489/1572 |
| low      |  79.76% | 4237/5312 |

## 6. Cross Analysis: Quality x Layout

| Quality | Layout | Accuracy | Correct/Total |
|---------|--------|----------|---------------|
| easy    | single |  94.72% | 628/663 |
| easy    | multi  |  93.23% | 565/606 |
| easy    | vertical |  97.69% | 296/303 |
| middle  | single |  80.17% | 942/1175 |
| middle  | multi  |  82.68% | 2105/2546 |
| middle  | vertical |  94.67% | 551/582 |
| hard    | single |  57.35% | 117/204 |
| hard    | multi  |  60.14% | 418/695 |
| hard    | vertical |  94.55% | 104/110 |

## 7. Error Type Distribution

| Error Type | Count | Percentage |
|------------|-------|------------|
| correct              |  5726 |  83.18% |
| length_error         |   670 |   9.73% |
| single_char_error    |   241 |   3.50% |
| multi_char_error     |   229 |   3.33% |
| order_error          |    18 |   0.26% |

## 8. Key Findings

- **Layout Impact**: Layout 类型对准确率影响显著 (差距 15.3%). Single (82.6%) > Multi (80.3%) > Vertical (95.6%)
- **Difficulty Gap**: Hard 样本比 Easy 样本低 31.4%, 表明模型在处理困难样本时性能下降明显
- **OOV Challenge**: OOV 样本准确率 (61.0%) 低于 IV 样本 (89.9%) 差距 28.9%, 模型对未见过的词汇泛化能力有限
- **Resolution Sensitivity**: 低分辨率样本准确率 (79.8%) 低于正常分辨率 (94.7%) 差距 15.0%
- **Length Errors**: 长度错误占比 9.7%, 建议关注 EOS 预测机制

## 9. Recommendations

基于以上分析，建议优先关注以下方向：

1. **Layout-Aware Enhancement**: 针对 multi-line 和 vertical 布局设计专门的特征提取模块
2. **Degradation Robustness**: 增强模型对 hard 样本（低质量图像）的鲁棒性
3. **OOV Generalization**: 通过数据增强或外部知识提升对罕见词汇的识别能力
4. **Resolution Adaptation**: 设计分辨率自适应机制，提升低分辨率图像识别效果


---
*Report generated automatically from error_analysis.csv*
