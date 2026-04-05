# SLP34K Evaluation Toolkit

SLP34K 测试评估工具集，用于模型性能分析和错误诊断。

## 目录结构

```
evaluation/
├── README.md                 # 本文件
├── csv_to_lmdb.py           # CSV 转 LMDB 工具
├── analyze_errors.py        # 错误分析工具
├── results/                 # 分析结果输出目录
│   └── error_analysis.csv   # 错误分析结果 (生成)
└── unified_lmdb/            # 统一 LMDB 数据库 (生成)
    ├── data.mdb
    └── lock.mdb
```

## 工具说明

### 1. csv_to_lmdb.py - CSV 转 LMDB 工具

将 benchmark CSV 文件转换为统一的 LMDB 数据库，避免重复推理。

**功能：**
- 读取 benchmark.csv (包含 base64 编码图像和元数据)
- 生成统一 LMDB 数据库，包含图像、标签和元数据
- 支持多维度标签：quality, structure, vocabulary_type, resolution_type

**用法：**
```bash
python evaluation/csv_to_lmdb.py \
    /path/to/benchmark.csv \
    --output_dir data/test/SLP34K_lmdb_benchmark/unified_lmdb
```

**输出：**
- `data.mdb`: 数据库文件 (~50MB)
- `lock.mdb`: 锁文件

**LMDB 数据结构：**
- `num-samples`: 样本总数 (6884)
- `image-{index:09d}`: JPEG 图像二进制数据
- `label-{index:09d}`: 文本标签
- `meta-{index:09d}`: JSON 格式元数据

---

### 2. analyze_errors.py - 错误分析工具

对模型预测结果进行多维度错误分析。

**功能：**
- 加载统一 LMDB 数据库，只推理一次
- 按多维度统计准确率：
  - Difficulty (easy/middle/hard)
  - Layout (single/multi/vertical)
  - Vocabulary Type (IV/OOV)
  - Resolution Type (normal/low)
- 错误类型分类：correct, length_error, order_error, single_char_error, multi_char_error
- 输出详细 CSV 报告

**用法：**
```bash
python evaluation/analyze_errors.py \
    checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
    --unified_db data/test/SLP34K_lmdb_benchmark/unified_lmdb \
    --output_csv evaluation/results/error_analysis.csv
```

**默认输出路径**: `evaluation/results/error_analysis.csv`

**参数：**
- `checkpoint`: 模型检查点路径
- `--unified_db`: 统一 LMDB 数据库路径
- `--device`: 计算设备 (默认: cuda)
- `--output_csv`: 输出 CSV 文件路径

**输出报告示例：**
```
============================================================
SLP34K Error Analysis Report
============================================================

Total samples: 6884
Correct: 5695
Overall Accuracy: 83.53%

============================================================
Accuracy by Difficulty (Quality)
============================================================
easy                :  91.34% (1436/1572)
middle              :  82.56% (3553/4303)
hard                :  73.21% ( 706/1009)

============================================================
Accuracy by Layout (Structure)
============================================================
single              :  88.12% (1800/2042)
multi               :  79.23% (3048/3847)
vertical            :  72.34% ( 720/ 995)
...
```

**CSV 输出格式：**
| 列名 | 说明 |
|------|------|
| image_id | 样本 ID |
| quality | 难度 (easy/middle/hard) |
| layout | 布局 (single/multi/vertical) |
| vocabulary_type | 词表类型 (IV/OOV) |
| resolution_type | 分辨率 (normal/low) |
| gt | 真实标签 |
| pred | 预测标签 |
| correct | 是否正确 |
| error_type | 错误类型 |
| note | 备注 |

---

## 完整工作流程

```bash
# 1. 激活环境
cd /mnt/data/zyx/SLP34K/ocr_training
source $(conda info --base)/etc/profile.d/conda.sh
conda activate slpr_slp34k

# 2. 创建统一 LMDB 数据库 (只需执行一次)
python evaluation/csv_to_lmdb.py \
    /mnt/data/zyx/A-Multi-Agent-Collaborative-Framework-for-Robust-Ship-Registration-Plate-Recognition/data/benchmark/benchmark.csv \
    --output_dir data/test/SLP34K_lmdb_benchmark/unified_lmdb

# 3. 运行错误分析
python evaluation/analyze_errors.py \
    checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \
    --unified_db data/test/SLP34K_lmdb_benchmark/unified_lmdb \
    --output_csv evaluation/results/error_analysis.csv
```

---

## 注意事项

1. **环境依赖**：需要激活 `slpr_slp34k` conda 环境
2. **GPU 要求**：默认使用 CUDA，如需 CPU 运行请添加 `--device cpu`
3. **LMDB 路径**：统一数据库默认路径为 `data/test/SLP34K_lmdb_benchmark/unified_lmdb/`
4. **结果解读**：
   - IV (In-Vocabulary): 训练集中出现过的标签
   - OOV (Out-of-Vocabulary): 训练集中未出现的标签
   - 三个维度 (quality/structure/resolution) 是正交划分的

---

## 故障排除

**问题1**: `torch.load` 报错 `weights_only`
- **解决**: 已修复，见 `strhub/models/maevit_infonce_plm/system.py` 第 214 行

**问题2**: 显存不足
- **解决**: 使用 CPU 推理 `--device cpu` 或减小 batch size

**问题3**: LMDB 文件损坏
- **解决**: 删除 `data/test/SLP34K_lmdb_benchmark/unified_lmdb/` 目录并重新运行 `csv_to_lmdb.py`

---

## 输出文件说明

所有分析结果默认保存在 `evaluation/results/` 目录：

| 文件 | 说明 |
|------|------|
| `error_analysis.csv` | 详细错误分析结果，包含每个样本的预测、标签、错误类型等 |

可以通过 `--output_csv` 参数自定义输出路径。
