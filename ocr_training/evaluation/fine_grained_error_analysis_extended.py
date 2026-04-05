#!/usr/bin/env python3
"""
Fine-Grained Error Analysis Extended for SLP34K
细粒度错误分析扩展工具

新增功能：
1. first_wrong_position_analysis - 首个错误位置分析
2. gt_pred_length_heatmap - GT vs Pred 长度热力图
3. segment_cer_analysis - 分段 CER 分析
4. segment_confusion_matrix - 分段混淆矩阵
5. uncertainty_error_relation - 不确定性与错误关系
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# 添加项目根目录到 Python 路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher


def classify_char_type(char: str) -> str:
    """分类字符类型"""
    if '\u4e00' <= char <= '\u9fff':
        return 'chinese'
    elif char.isdigit():
        return 'digit'
    elif char.isalpha() and char.isascii():
        return 'pinyin'
    else:
        return 'other'


def get_segment_type_at_position(text: str, pos: int) -> str:
    """获取指定位置的段类型"""
    if pos >= len(text):
        return 'suffix-quarter'
    char_type = classify_char_type(text[pos])
    if char_type == 'other':
        # 向后查找最近的非other类型
        for i in range(pos + 1, len(text)):
            t = classify_char_type(text[i])
            if t != 'other':
                return t
        return 'suffix-quarter'
    return char_type


def find_first_wrong_position(gt: str, pred: str) -> Tuple[Optional[int], str]:
    """
    找到第一个错误位置
    返回: (位置, 该位置的段类型)
    """
    max_len = max(len(gt), len(pred))
    
    for i in range(max_len):
        gt_char = gt[i] if i < len(gt) else None
        pred_char = pred[i] if i < len(pred) else None
        
        if gt_char != pred_char:
            seg_type = get_segment_type_at_position(gt, i)
            return i, seg_type
    
    return None, 'none'


def analyze_first_wrong_position(rows: List[Dict]) -> Dict:
    """
    分析首个错误位置
    """
    # 按段类型统计
    segment_stats = defaultdict(lambda: {'count': 0, 'positions': []})
    
    for row in rows:
        if row['correct'] != 'True':
            pos, seg_type = find_first_wrong_position(row['gt'], row['pred'])
            if pos is not None:
                segment_stats[seg_type]['count'] += 1
                segment_stats[seg_type]['positions'].append(pos)
    
    # 计算统计信息
    result = {}
    for seg_type, stats in segment_stats.items():
        positions = stats['positions']
        result[seg_type] = {
            'count': stats['count'],
            'mean_position': np.mean(positions) if positions else 0,
            'median_position': np.median(positions) if positions else 0,
            'std_position': np.std(positions) if positions else 0
        }
    
    return result


def analyze_gt_pred_length_heatmap(rows: List[Dict]) -> Dict:
    """
    分析 GT length vs Pred length 热力图
    """
    # 收集所有长度对
    all_pairs = []
    hard_pairs = []
    oov_pairs = []
    multi_pairs = []
    
    for row in rows:
        gt_len = len(row['gt'])
        pred_len = len(row['pred'])
        pair = (gt_len, pred_len)
        
        all_pairs.append(pair)
        
        if row['quality'] == 'hard':
            hard_pairs.append(pair)
        if row['vocabulary_type'] == 'OOV':
            oov_pairs.append(pair)
        if row['layout'] == 'multi':
            multi_pairs.append(pair)
    
    def create_heatmap_data(pairs, max_len=30):
        """创建热力图数据"""
        heatmap = np.zeros((max_len, max_len))
        for gt_len, pred_len in pairs:
            if gt_len < max_len and pred_len < max_len:
                heatmap[gt_len, pred_len] += 1
        return heatmap
    
    return {
        'all': create_heatmap_data(all_pairs),
        'hard': create_heatmap_data(hard_pairs),
        'oov': create_heatmap_data(oov_pairs),
        'multi': create_heatmap_data(multi_pairs),
        'max_gt': max(len(row['gt']) for row in rows),
        'max_pred': max(len(row['pred']) for row in rows)
    }


def calculate_cer(refs: List[str], hyps: List[str]) -> float:
    """
    计算字符错误率 (CER)
    CER = (S + D + I) / N
    """
    total_chars = 0
    total_errors = 0
    
    for ref, hyp in zip(refs, hyps):
        matcher = SequenceMatcher(None, ref, hyp)
        errors = 0
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                errors += max(i2 - i1, j2 - j1)
            elif tag == 'delete':
                errors += i2 - i1
            elif tag == 'insert':
                errors += j2 - j1
        
        total_chars += len(ref)
        total_errors += errors
    
    return 100.0 * total_errors / total_chars if total_chars > 0 else 0


def analyze_segment_cer(rows: List[Dict]) -> Dict:
    """
    分段 CER 分析
    """
    def extract_segments(text: str) -> Dict[str, List[str]]:
        """提取各类型段"""
        segments = {'chinese': [], 'digit': [], 'pinyin': []}
        current_seg = []
        current_type = None
        
        for char in text:
            char_type = classify_char_type(char)
            if char_type == 'other':
                continue
            
            if current_type is None:
                current_type = char_type
                current_seg = [char]
            elif char_type == current_type:
                current_seg.append(char)
            else:
                if current_type in segments:
                    segments[current_type].append(''.join(current_seg))
                current_type = char_type
                current_seg = [char]
        
        if current_seg and current_type in segments:
            segments[current_type].append(''.join(current_seg))
        
        return segments
    
    # 收集各段类型的 refs 和 hyps
    segment_refs = {'chinese': [], 'digit': [], 'pinyin': []}
    segment_hyps = {'chinese': [], 'digit': [], 'pinyin': []}
    
    # Hard 子集
    hard_refs = {'chinese': [], 'digit': [], 'pinyin': []}
    hard_hyps = {'chinese': [], 'digit': [], 'pinyin': []}
    
    # OOV 子集
    oov_refs = {'chinese': [], 'digit': [], 'pinyin': []}
    oov_hyps = {'chinese': [], 'digit': [], 'pinyin': []}
    
    for row in rows:
        gt = row['gt']
        pred = row['pred']
        
        # 提取 GT 中的各段
        gt_segments = extract_segments(gt)
        pred_segments = extract_segments(pred)
        
        # 匹配同类型的段
        for seg_type in ['chinese', 'digit', 'pinyin']:
            gt_segs = gt_segments[seg_type]
            pred_segs = pred_segments[seg_type]
            
            # 简单匹配：按顺序配对
            for i, gt_seg in enumerate(gt_segs):
                pred_seg = pred_segs[i] if i < len(pred_segs) else ""
                segment_refs[seg_type].append(gt_seg)
                segment_hyps[seg_type].append(pred_seg)
                
                # 子集
                if row['quality'] == 'hard':
                    hard_refs[seg_type].append(gt_seg)
                    hard_hyps[seg_type].append(pred_seg)
                if row['vocabulary_type'] == 'OOV':
                    oov_refs[seg_type].append(gt_seg)
                    oov_hyps[seg_type].append(pred_seg)
    
    # 计算 CER
    results = {
        'overall': {},
        'hard': {},
        'oov': {}
    }
    
    for seg_type in ['chinese', 'digit', 'pinyin']:
        results['overall'][seg_type] = calculate_cer(
            segment_refs[seg_type], segment_hyps[seg_type]
        )
        results['hard'][seg_type] = calculate_cer(
            hard_refs[seg_type], hard_hyps[seg_type]
        ) if hard_refs[seg_type] else 0
        results['oov'][seg_type] = calculate_cer(
            oov_refs[seg_type], oov_hyps[seg_type]
        ) if oov_refs[seg_type] else 0
    
    return results


def analyze_segment_confusion(rows: List[Dict], top_k: int = 20) -> Dict:
    """
    分段混淆矩阵分析
    """
    def get_char_confusion(gt: str, pred: str, char_type_func) -> List[Tuple[str, str]]:
        """获取特定类型的字符混淆对"""
        confusions = []
        matcher = SequenceMatcher(None, gt, pred)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                for i, j in zip(range(i1, i2), range(j1, j2)):
                    if i < len(gt) and j < len(pred):
                        gt_char = gt[i]
                        pred_char = pred[j]
                        if char_type_func(gt_char) and char_type_func(pred_char):
                            confusions.append((gt_char, pred_char))
        
        return confusions
    
    # 收集混淆
    digit_confusions = []
    pinyin_confusions = []
    chinese_confusions = []
    
    for row in rows:
        if row['correct'] != 'True':
            gt = row['gt']
            pred = row['pred']
            
            digit_confusions.extend(get_char_confusion(
                gt, pred, lambda c: c.isdigit()
            ))
            pinyin_confusions.extend(get_char_confusion(
                gt, pred, lambda c: c.isalpha() and c.isascii()
            ))
            chinese_confusions.extend(get_char_confusion(
                gt, pred, lambda c: '\u4e00' <= c <= '\u9fff'
            ))
    
    def get_top_confusions(confusions, k):
        """获取 top k 混淆"""
        from collections import Counter
        counter = Counter(confusions)
        return counter.most_common(k)
    
    return {
        'digit': get_top_confusions(digit_confusions, top_k),
        'pinyin': get_top_confusions(pinyin_confusions, top_k),
        'chinese': get_top_confusions(chinese_confusions, top_k)
    }


def analyze_uncertainty_error_relation(rows: List[Dict]) -> Dict:
    """
    不确定性与错误关系分析
    注意：当前 error_analysis.csv 中没有 confidence/logits 数据
    此函数预留接口，如果后续有数据可以扩展
    """
    # 检查是否有 confidence 数据
    has_confidence = 'confidence' in rows[0] if rows else False
    
    if not has_confidence:
        return {
            'available': False,
            'message': 'No confidence/logits data available in error_analysis.csv'
        }
    
    # 如果有数据，进行分析...
    # 这里预留实现
    return {
        'available': True,
        'reliability_diagram': None,
        'uncertainty_histogram': None
    }


def create_extended_visualizations(results: Dict, output_dir: Path):
    """
    创建扩展可视化
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 首个错误位置分布
    if 'first_wrong' in results:
        first_wrong = results['first_wrong']
        if first_wrong:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 按段类型统计数量
            seg_types = list(first_wrong.keys())
            counts = [first_wrong[st]['count'] for st in seg_types]
            mean_pos = [first_wrong[st]['mean_position'] for st in seg_types]
            
            ax = axes[0]
            bars = ax.bar(seg_types, counts, color='steelblue', edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_title('First Wrong Position by Segment Type')
            ax.set_xlabel('Segment Type')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            ax = axes[1]
            bars = ax.bar(seg_types, mean_pos, color='coral', edgecolor='black')
            ax.set_ylabel('Mean Position')
            ax.set_title('Mean First Error Position by Segment Type')
            ax.set_xlabel('Segment Type')
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'first_wrong_position.png', dpi=150)
            plt.close()
    
    # 2. GT vs Pred 长度热力图
    if 'length_heatmap' in results:
        heatmap_data = results['length_heatmap']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        subsets = [
            ('all', 'All Samples'),
            ('hard', 'Hard Quality'),
            ('oov', 'OOV Vocabulary'),
            ('multi', 'Multi Layout')
        ]
        
        for idx, (key, title) in enumerate(subsets):
            ax = axes[idx // 2, idx % 2]
            data = heatmap_data[key][:25, :25]  # 限制到25
            
            sns.heatmap(data, annot=False, cmap='YlOrRd', ax=ax, 
                       cbar_kws={'label': 'Count'})
            ax.set_title(f'GT vs Pred Length - {title}')
            ax.set_xlabel('Predicted Length')
            ax.set_ylabel('Ground Truth Length')
            
            # 添加对角线
            ax.plot([0, 25], [0, 25], 'b--', linewidth=2, alpha=0.5, label='Perfect Match')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'length_heatmap_subsets.png', dpi=150)
        plt.close()
    
    # 3. 分段 CER 对比
    if 'segment_cer' in results:
        cer_data = results['segment_cer']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(3)
        width = 0.25
        
        seg_types = ['chinese', 'digit', 'pinyin']
        overall_cer = [cer_data['overall'].get(st, 0) for st in seg_types]
        hard_cer = [cer_data['hard'].get(st, 0) for st in seg_types]
        oov_cer = [cer_data['oov'].get(st, 0) for st in seg_types]
        
        bars1 = ax.bar(x - width, overall_cer, width, label='Overall', color='steelblue')
        bars2 = ax.bar(x, hard_cer, width, label='Hard', color='coral')
        bars3 = ax.bar(x + width, oov_cer, width, label='OOV', color='lightgreen')
        
        ax.set_ylabel('CER (%)')
        ax.set_title('Character Error Rate by Segment Type')
        ax.set_xticks(x)
        ax.set_xticklabels(seg_types)
        ax.legend()
        ax.set_ylim(0, max(max(overall_cer), max(hard_cer), max(oov_cer)) * 1.2)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'segment_cer_comparison.png', dpi=150)
        plt.close()
    
    # 4. 分段混淆矩阵 (Top 10)
    if 'segment_confusion' in results:
        confusion = results['segment_confusion']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (seg_type, title) in enumerate([('digit', 'Digit'), 
                                                  ('pinyin', 'Pinyin'), 
                                                  ('chinese', 'Chinese')]):
            ax = axes[idx]
            confusions = confusion.get(seg_type, [])
            
            if confusions:
                # 获取 top 10
                top_conf = confusions[:10]
                gt_chars = [c[0][0] for c in top_conf]
                pred_chars = [c[0][1] for c in top_conf]
                counts = [c[1] for c in top_conf]
                
                # 创建简化的混淆矩阵
                unique_chars = list(set(gt_chars + pred_chars))
                if len(unique_chars) > 1:
                    matrix = np.zeros((len(unique_chars), len(unique_chars)))
                    for (gt_c, pred_c), count in confusions[:20]:
                        if gt_c in unique_chars and pred_c in unique_chars:
                            i = unique_chars.index(gt_c)
                            j = unique_chars.index(pred_c)
                            matrix[i, j] += count
                    
                    sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd',
                               xticklabels=unique_chars, yticklabels=unique_chars, ax=ax)
                    ax.set_title(f'{title} Confusion Matrix (Top)')
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Ground Truth')
                else:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
                    ax.set_title(f'{title} Confusion Matrix')
            else:
                ax.text(0.5, 0.5, 'No confusion data', ha='center', va='center')
                ax.set_title(f'{title} Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'segment_confusion_matrices.png', dpi=150)
        plt.close()
    
    print(f"Extended visualizations saved to: {output_dir}")


def save_extended_csv_results(results: Dict, output_dir: Path):
    """
    保存扩展 CSV 结果
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 首个错误位置分析
    if 'first_wrong' in results:
        with open(output_dir / 'first_wrong_position.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['segment_type', 'count', 'mean_position', 'median_position', 'std_position'])
            for seg_type, stats in sorted(results['first_wrong'].items()):
                writer.writerow([
                    seg_type, stats['count'], 
                    f"{stats['mean_position']:.2f}",
                    f"{stats['median_position']:.2f}",
                    f"{stats['std_position']:.2f}"
                ])
    
    # 2. 长度热力图数据
    if 'length_heatmap' in results:
        heatmap_data = results['length_heatmap']
        
        # All samples
        with open(output_dir / 'length_heatmap_all.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['gt_length', 'pred_length', 'count'])
            for i in range(25):
                for j in range(25):
                    if heatmap_data['all'][i, j] > 0:
                        writer.writerow([i, j, int(heatmap_data['all'][i, j])])
    
    # 3. 分段 CER
    if 'segment_cer' in results:
        with open(output_dir / 'segment_cer.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['subset', 'chinese_cer', 'digit_cer', 'pinyin_cer'])
            for subset in ['overall', 'hard', 'oov']:
                cer = results['segment_cer'][subset]
                writer.writerow([
                    subset,
                    f"{cer.get('chinese', 0):.2f}%",
                    f"{cer.get('digit', 0):.2f}%",
                    f"{cer.get('pinyin', 0):.2f}%"
                ])
    
    # 4. 分段混淆矩阵
    if 'segment_confusion' in results:
        for seg_type in ['digit', 'pinyin', 'chinese']:
            confusions = results['segment_confusion'].get(seg_type, [])
            if confusions:
                with open(output_dir / f'{seg_type}_confusion.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['gt_char', 'pred_char', 'count'])
                    for (gt_c, pred_c), count in confusions:
                        writer.writerow([gt_c, pred_c, count])
    
    print(f"Extended CSV results saved to: {output_dir}")


def generate_extended_report(results: Dict, output_path: Path):
    """
    生成扩展 Markdown 报告
    """
    report = """# SLP34K Fine-Grained Error Analysis Report (Extended)

## 1. 首个错误位置分析

"""
    
    if 'first_wrong' in results and results['first_wrong']:
        report += "| Segment Type | Count | Mean Position | Median Position | Std Position |\n"
        report += "|--------------|-------|---------------|-----------------|--------------|\n"
        for seg_type, stats in sorted(results['first_wrong'].items()):
            report += f"| {seg_type:12s} | {stats['count']:5d} | {stats['mean_position']:13.2f} | {stats['median_position']:15.2f} | {stats['std_position']:12.2f} |\n"
        
        # 关键发现
        first_wrong = results['first_wrong']
        if first_wrong:
            most_common = max(first_wrong.items(), key=lambda x: x[1]['count'])
            earliest = min(first_wrong.items(), key=lambda x: x[1]['mean_position'])
            report += f"\n**关键发现**: 首个错误最常发生在 **{most_common[0]}** 段 ({most_common[1]['count']} 次)，"
            report += f"最早出错的段类型是 **{earliest[0]}** (平均位置 {earliest[1]['mean_position']:.1f})\n"
    
    report += """
## 2. 分段 CER 分析

"""
    
    if 'segment_cer' in results:
        report += "| Subset | Chinese CER | Digit CER | Pinyin CER |\n"
        report += "|--------|-------------|-----------|------------|\n"
        for subset in ['overall', 'hard', 'oov']:
            cer = results['segment_cer'][subset]
            report += f"| {subset:6s} | {cer.get('chinese', 0):10.2f}% | {cer.get('digit', 0):9.2f}% | {cer.get('pinyin', 0):10.2f}% |\n"
        
        # 分析
        overall = results['segment_cer']['overall']
        worst_seg = max(overall.items(), key=lambda x: x[1])
        report += f"\n**关键发现**: 整体 CER 最高的是 **{worst_seg[0]}** 段 ({worst_seg[1]:.2f}%)\n"
    
    report += """
## 3. 分段混淆矩阵 (Top 10)

"""
    
    if 'segment_confusion' in results:
        for seg_type in ['digit', 'pinyin', 'chinese']:
            confusions = results['segment_confusion'].get(seg_type, [])
            if confusions:
                report += f"### {seg_type.capitalize()} Confusion (Top 10)\n\n"
                report += "| GT Char | Pred Char | Count |\n"
                report += "|---------|-----------|-------|\n"
                for (gt_c, pred_c), count in confusions[:10]:
                    report += f"| {gt_c:7s} | {pred_c:9s} | {count:5d} |\n"
                report += "\n"
    
    report += """
## 4. 长度热力图分析

GT Length vs Predicted Length 的热力图已生成：
- `length_heatmap_subsets.png` - 包含 All/Hard/OOV/Multi 四个子集

**解读**: 对角线表示长度预测正确，偏离对角线表示长度预测错误。

## 5. 可视化图表

扩展分析生成的图表：

1. `first_wrong_position.png` - 首个错误位置分布
2. `length_heatmap_subsets.png` - 长度热力图（多子集）
3. `segment_cer_comparison.png` - 分段 CER 对比
4. `segment_confusion_matrices.png` - 分段混淆矩阵

## 6. CSV 数据文件

- `first_wrong_position.csv` - 首个错误位置统计
- `length_heatmap_all.csv` - 长度热力图数据
- `segment_cer.csv` - 分段 CER 数据
- `digit_confusion.csv` - 数字混淆矩阵
- `pinyin_confusion.csv` - 拼音混淆矩阵
- `chinese_confusion.csv` - 中文混淆矩阵

---
*Generated by fine_grained_error_analysis_extended.py*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Extended report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-Grained Error Analysis Extended for SLP34K')
    parser.add_argument('error_csv', help='Path to error_analysis.csv')
    parser.add_argument('--output_dir', default='evaluation/results/fine_grained_extended',
                       help='Output directory for results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取数据
    print(f"Reading {args.error_csv}...")
    with open(args.error_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total samples: {len(rows)}")
    
    results = {'rows': rows}
    
    # 1. 首个错误位置分析
    print("\n1. Analyzing first wrong position...")
    results['first_wrong'] = analyze_first_wrong_position(rows)
    print(f"  Found {len(results['first_wrong'])} segment types")
    
    # 2. 长度热力图
    print("\n2. Analyzing GT vs Pred length heatmap...")
    results['length_heatmap'] = analyze_gt_pred_length_heatmap(rows)
    print("  Heatmap data generated")
    
    # 3. 分段 CER
    print("\n3. Analyzing segment CER...")
    results['segment_cer'] = analyze_segment_cer(rows)
    print("  CER calculated for all segments")
    
    # 4. 分段混淆矩阵
    print("\n4. Analyzing segment confusion matrices...")
    results['segment_confusion'] = analyze_segment_confusion(rows, top_k=20)
    print(f"  Found confusion pairs")
    
    # 5. 不确定性分析
    print("\n5. Checking uncertainty data...")
    results['uncertainty'] = analyze_uncertainty_error_relation(rows)
    if not results['uncertainty']['available']:
        print("  Note: No confidence/logits data available")
    
    # 保存 CSV
    print("\nSaving CSV results...")
    save_extended_csv_results(results, output_dir)
    
    # 创建可视化
    print("\nCreating visualizations...")
    create_extended_visualizations(results, output_dir)
    
    # 生成报告
    print("\nGenerating report...")
    generate_extended_report(results, output_dir / 'extended_report.md')
    
    print("\n" + "="*60)
    print("Extended fine-grained analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
