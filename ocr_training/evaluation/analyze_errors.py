#!/usr/bin/env python3
"""
SLP34K Error Analysis Tool
分析测试集预测结果，按不同维度统计错误分布

支持统一 LMDB 数据库 (unified_lmdb)，包含完整元数据:
- quality: easy/middle/hard
- structure: single/multi/vertical
- vocabulary_type: IV/OOV
- resolution_type: normal/low
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from tqdm import tqdm

import lmdb
from strhub.models.utils import load_from_checkpoint
from torchvision import transforms as T
from PIL import Image
import io


@dataclass
class ErrorRecord:
    """单条错误记录"""
    image_id: str
    quality: str  # easy/middle/hard
    layout: str  # single/multi/vertical
    vocabulary_type: str  # IV/OOV
    resolution_type: str  # normal/low
    gt: str
    pred: str
    correct: bool
    error_type: str
    note: str = ""


def classify_error(gt: str, pred: str) -> str:
    """
    分类错误类型
    - correct: 完全正确
    - length_error: 长度错误（插入/删除）
    - order_error: 字符相同但顺序不同
    - single_char_error: 只有一个字符错误
    - multi_char_error: 多个字符错误
    """
    if gt == pred:
        return "correct"
    
    if len(gt) != len(pred):
        return "length_error"
    
    # 长度相同，检查是否是顺序错误
    if sorted(gt) == sorted(pred):
        return "order_error"
    
    # 统计不同位置
    diff_count = sum(1 for g, p in zip(gt, pred) if g != p)
    
    if diff_count == 1:
        return "single_char_error"
    else:
        return "multi_char_error"


class UnifiedLmdbDataset:
    """统一 LMDB 数据集，支持元数据读取"""
    
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self._env = None
        self.num_samples = self._get_num_samples()
    
    def _create_env(self):
        return lmdb.open(self.root, max_readers=1, readonly=True, create=False,
                         readahead=False, meminit=False, lock=False)
    
    @property
    def env(self):
        if self._env is None:
            self._env = self._create_env()
        return self._env
    
    def _get_num_samples(self):
        with self._create_env() as env:
            with env.begin() as txn:
                return int(txn.get('num-samples'.encode()))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, index):
        idx = index + 1  # LMDB 从 1 开始
        
        img_key = f'image-{idx:09d}'.encode()
        label_key = f'label-{idx:09d}'.encode()
        meta_key = f'meta-{idx:09d}'.encode()
        
        with self.env.begin() as txn:
            img_data = txn.get(img_key)
            label = txn.get(label_key).decode()
            meta = json.loads(txn.get(meta_key).decode())
        
        # 解码图像
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        return img, label, meta


def analyze_predictions(checkpoint_path: str, 
                        unified_db: str = 'data/test/SLP34K_lmdb_benchmark/unified_lmdb',
                        device: str = 'cuda',
                        output_csv: str = 'error_analysis.csv') -> Dict:
    """
    分析模型在SLP34K统一测试集上的预测结果
    """
    # SLP34K字符集
    charset_test = "睢荷焦射渔灌馬轩森引猛球祁智卓禾翼ptihgy松园淠澎宗禹领茹斌潜舜孝感爱船帮玮丽月学燕炉玲事必思屹展長牛双邦霖粮纬亮致圣降语奥树昱配然郭唐uan珠蓝陆邳郡惜泾帅巡卸孟峻澳加涵淳毅神艇刘救助政百劲锋凌硕潮漂葆莱凓沛戈喜忠聚获抚绣意羽微梁久心午鸥甸渡杨韦友电焊勇征如满跃景齐朝子铭复壁庙涟逸欣舸关升经星晟溆浦冠咏多才统烁沙世力渝巢宛宸煜送鹰广之潥仁皋晶昶漕伦梓架普凤能捷濉石屏浚名仕前繁为好定帆喻颖卫袁旭保濠漯弋雅杰柏义俊军福沈津拖散乐蓼环威店枞亨姑塘嵊渚怡含丘飞波元骏青弘V沭雷傲大惠自梧财坤悦锐观音文春钓一沚程健荣荆昭佳众椒耀兆寺藤虎乾博贸工驻益游台得裕日韵茂茗融豪朗商辰靖化雨迎超涡全鄂黄冈湾胜亚鸣高滨成辉闽三浩驳启沪内同正氏民槽京汉荻驰西迁巨申泉庆志锦生王良来康钢舒林锚饶虹鹏君宜伟谐梅兰鞍俞淤固绿洲-溧动邱璧菏九峰泗交浍灵二水红陶四恒诚临怀池张界国邮霍强垛北铜泽洪玉家蒙五腾衢旺润舟姚轮川银瑞舵赣湘宏圆鼎泓凯隆常善清光方联风马创万业镇昌姜和乡宝陵滁寿武丰鸿连郎颍县辛汇柯洋云天振桥扬枣河永明建庐庄宇溪源发油太宿芜翔利宣六信徐PR号龙锡肥无合吉新东中桐德盛平南越祥鑫M华金K阜4口钱长通余蚌埠远达济鲁Q淮海宁顺运亳周嘉虞富上D湖盐集江安豫泰航机城F阳B山萧TLC苏绍诸暨杭W皖兴XS州J7Y35港E2Z浙961货IO08GAUHN"

    # 加载模型
    print(f"Loading model from: {checkpoint_path}")
    kwargs = {'mae_pretrained_path': "pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth",
              'charset_test': charset_test}
    model = load_from_checkpoint(checkpoint_path, **kwargs).eval().to(device)
    hp = model.hparams

    # 准备数据
    img_size = getattr(hp, 'img_size', [224, 224])

    # 图像预处理
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    transform = T.Compose([
        T.Resize(img_size, T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])

    # 创建统一数据集
    print(f"\nLoading unified database from: {unified_db}")
    dataset = UnifiedLmdbDataset(unified_db, transform=transform)
    print(f"Total samples: {len(dataset)}")

    # 收集所有预测结果
    all_records: List[ErrorRecord] = []

    print("\nRunning inference...")
    for idx in tqdm(range(len(dataset)), desc="Inference"):
        img, label, meta = dataset[idx]
        
        # 预测
        img_batch = img.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model.forward(img_batch)
            probs = logits.softmax(-1)
            preds, confs = model.tokenizer.decode(probs)
            pred = preds[0]
            # 应用charset adapter
            pred = model.charset_adapter(pred)

        gt = label
        correct = (pred == gt)
        error_type = classify_error(gt, pred)

        record = ErrorRecord(
            image_id=str(meta['id']),
            quality=meta['quality'],
            layout=meta['structure'],
            vocabulary_type=meta['vocabulary_type'],
            resolution_type=meta['resolution_type'],
            gt=gt,
            pred=pred,
            correct=correct,
            error_type=error_type,
            note=""
        )
        all_records.append(record)

    return analyze_records(all_records, output_csv)


def analyze_records(records: List[ErrorRecord], output_csv: str) -> Dict:
    """
    分析错误记录并生成统计报告
    """
    print(f"\n{'='*60}")
    print("SLP34K Error Analysis Report")
    print(f"{'='*60}")

    total = len(records)
    correct = sum(1 for r in records if r.correct)
    overall_acc = 100.0 * correct / total if total > 0 else 0

    print(f"\nTotal samples: {total}")
    print(f"Correct: {correct}")
    print(f"Overall Accuracy: {overall_acc:.2f}%")

    # 按 difficulty (quality) 统计
    print(f"\n{'='*60}")
    print("Accuracy by Difficulty (Quality)")
    print(f"{'='*60}")

    quality_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in records:
        quality_stats[r.quality]['total'] += 1
        if r.correct:
            quality_stats[r.quality]['correct'] += 1

    for quality in ['easy', 'middle', 'hard']:
        if quality in quality_stats:
            stats = quality_stats[quality]
            acc = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{quality:20s}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")

    # 按 layout (structure) 统计
    print(f"\n{'='*60}")
    print("Accuracy by Layout (Structure)")
    print(f"{'='*60}")

    layout_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in records:
        layout_stats[r.layout]['total'] += 1
        if r.correct:
            layout_stats[r.layout]['correct'] += 1

    for layout in ['single', 'multi', 'vertical']:
        if layout in layout_stats:
            stats = layout_stats[layout]
            acc = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{layout:20s}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")

    # 按 vocabulary_type 统计
    print(f"\n{'='*60}")
    print("Accuracy by Vocabulary Type")
    print(f"{'='*60}")

    vocab_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in records:
        vocab_stats[r.vocabulary_type]['total'] += 1
        if r.correct:
            vocab_stats[r.vocabulary_type]['correct'] += 1

    for vocab in ['IV', 'OOV']:
        if vocab in vocab_stats:
            stats = vocab_stats[vocab]
            acc = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{vocab:20s}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")

    # 按 resolution_type 统计
    print(f"\n{'='*60}")
    print("Accuracy by Resolution Type")
    print(f"{'='*60}")

    res_stats = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in records:
        res_stats[r.resolution_type]['total'] += 1
        if r.correct:
            res_stats[r.resolution_type]['correct'] += 1

    for res in ['normal', 'low']:
        if res in res_stats:
            stats = res_stats[res]
            acc = 100.0 * stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"{res:20s}: {acc:6.2f}% ({stats['correct']}/{stats['total']})")

    # 错误类型分布
    print(f"\n{'='*60}")
    print("Error Type Distribution")
    print(f"{'='*60}")

    error_type_counts = defaultdict(int)
    for r in records:
        error_type_counts[r.error_type] += 1

    for error_type, count in sorted(error_type_counts.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total > 0 else 0
        print(f"{error_type:20s}: {count:5d} ({pct:5.2f}%)")

    # 保存CSV
    print(f"\n{'='*60}")
    print(f"Saving detailed results to: {output_csv}")
    print(f"{'='*60}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'quality', 'layout', 'vocabulary_type', 'resolution_type',
                        'gt', 'pred', 'correct', 'error_type', 'note'])
        for r in records:
            writer.writerow([r.image_id, r.quality, r.layout,
                           r.vocabulary_type, r.resolution_type,
                           r.gt, r.pred, r.correct, r.error_type, r.note])

    # 返回统计结果字典
    return {
        'overall_accuracy': overall_acc,
        'total_samples': total,
        'correct_samples': correct,
        'quality_accuracy': {k: 100.0 * v['correct'] / v['total']
                            for k, v in quality_stats.items()},
        'layout_accuracy': {k: 100.0 * v['correct'] / v['total']
                           for k, v in layout_stats.items()},
        'vocabulary_accuracy': {k: 100.0 * v['correct'] / v['total']
                               for k, v in vocab_stats.items()},
        'resolution_accuracy': {k: 100.0 * v['correct'] / v['total']
                               for k, v in res_stats.items()},
        'error_type_distribution': dict(error_type_counts)
    }


def main():
    parser = argparse.ArgumentParser(description='SLP34K Error Analysis Tool')
    parser.add_argument('checkpoint', help='Model checkpoint path')
    parser.add_argument('--unified_db', default='data/test/SLP34K_lmdb_benchmark/unified_lmdb',
                       help='Unified LMDB database path')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--output_csv', default='evaluation/results/error_analysis.csv',
                       help='Output CSV file path (default: evaluation/results/error_analysis.csv)')
    args = parser.parse_args()

    # 确保输出目录存在
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = analyze_predictions(
        args.checkpoint,
        args.unified_db,
        args.device,
        args.output_csv
    )

    print("\n" + "="*60)
    print("Analysis complete!")
    print(f"Results saved to: {args.output_csv}")
    print("="*60)


if __name__ == '__main__':
    main()
