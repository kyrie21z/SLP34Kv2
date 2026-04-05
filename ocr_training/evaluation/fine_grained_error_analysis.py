#!/usr/bin/env python3
"""
Fine-Grained Error Analysis for SLP34K
细粒度错误分析工具

功能：
1. 字符级编辑操作统计 (replace/insert/delete/eos_early/eos_late)
2. 位置级准确率 (按字符位置、中文/数字/拼音段)
3. 长度分析 (GT vs Pred 分布、长度桶 accuracy)
4. 条件错误分析 (difficulty×layout、OOV×difficulty)
5. 可视化输出 (confusion matrix、heatmap、accuracy curve)
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
matplotlib.use('Agg')  # 无头模式
import matplotlib.pyplot as plt
import seaborn as sns
from difflib import SequenceMatcher


@dataclass
class EditOperation:
    """编辑操作"""
    op_type: str  # replace, insert, delete, eos_early, eos_late
    gt_char: Optional[str] = None
    pred_char: Optional[str] = None
    position: int = -1


@dataclass
class SegmentInfo:
    """文本段信息"""
    text: str
    seg_type: str  # chinese, digit, pinyin, mixed
    start_pos: int
    end_pos: int


def analyze_edit_operations(gt: str, pred: str) -> List[EditOperation]:
    """
    分析字符级编辑操作
    返回: 编辑操作列表
    """
    operations = []
    
    # 使用 SequenceMatcher 分析差异
    matcher = SequenceMatcher(None, gt, pred)
    
    gt_len = len(gt)
    pred_len = len(pred)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # 替换操作
            for i, j in zip(range(i1, i2), range(j1, j2)):
                operations.append(EditOperation(
                    op_type='replace',
                    gt_char=gt[i] if i < gt_len else None,
                    pred_char=pred[j] if j < pred_len else None,
                    position=i
                ))
            # 处理长度不等的情况
            if i2 - i1 > j2 - j1:
                # GT 更长，额外的字符被删除
                for i in range(i1 + (j2 - j1), i2):
                    operations.append(EditOperation(
                        op_type='delete',
                        gt_char=gt[i],
                        pred_char=None,
                        position=i
                    ))
            elif j2 - j1 > i2 - i1:
                # Pred 更长，额外的字符是插入
                for j in range(j1 + (i2 - i1), j2):
                    operations.append(EditOperation(
                        op_type='insert',
                        gt_char=None,
                        pred_char=pred[j],
                        position=i2
                    ))
        elif tag == 'delete':
            # 删除操作
            for i in range(i1, i2):
                operations.append(EditOperation(
                    op_type='delete',
                    gt_char=gt[i],
                    pred_char=None,
                    position=i
                ))
        elif tag == 'insert':
            # 插入操作
            for j in range(j1, j2):
                operations.append(EditOperation(
                    op_type='insert',
                    gt_char=None,
                    pred_char=pred[j],
                    position=i1
                ))
    
    # 分析 EOS 问题
    if pred_len < gt_len:
        # 提前结束 (EOS early)
        operations.append(EditOperation(
            op_type='eos_early',
            position=pred_len
        ))
    elif pred_len > gt_len:
        # 延迟结束 (EOS late)
        operations.append(EditOperation(
            op_type='eos_late',
            position=gt_len
        ))
    
    return operations


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


def segment_text(text: str) -> List[SegmentInfo]:
    """
    将文本分段为中文、数字、拼音段
    """
    if not text:
        return []
    
    segments = []
    current_seg = []
    current_type = None
    start_pos = 0
    
    for i, char in enumerate(text):
        char_type = classify_char_type(char)
        
        if char_type == 'other':
            continue
            
        if current_type is None:
            current_type = char_type
            start_pos = i
            current_seg = [char]
        elif char_type == current_type:
            current_seg.append(char)
        else:
            # 保存当前段
            segments.append(SegmentInfo(
                text=''.join(current_seg),
                seg_type=current_type,
                start_pos=start_pos,
                end_pos=i
            ))
            # 开始新段
            current_type = char_type
            start_pos = i
            current_seg = [char]
    
    # 保存最后一段
    if current_seg:
        segments.append(SegmentInfo(
            text=''.join(current_seg),
            seg_type=current_type,
            start_pos=start_pos,
            end_pos=len(text)
        ))
    
    return segments


def analyze_position_accuracy(rows: List[Dict]) -> Dict:
    """
    按字符位置统计准确率
    """
    position_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    segment_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for row in rows:
        gt = row['gt']
        pred = row['pred']
        
        # 逐位置比较
        max_len = max(len(gt), len(pred))
        for i in range(max_len):
            gt_char = gt[i] if i < len(gt) else None
            pred_char = pred[i] if i < len(pred) else None
            
            position_stats[i]['total'] += 1
            if gt_char == pred_char:
                position_stats[i]['correct'] += 1
        
        # 分段统计
        gt_segments = segment_text(gt)
        pred_segments = segment_text(pred)
        
        for gt_seg in gt_segments:
            segment_stats[gt_seg.seg_type]['total'] += len(gt_seg.text)
            
            # 检查该段在 pred 中的匹配情况
            for i in range(gt_seg.start_pos, min(gt_seg.end_pos, len(pred))):
                if i < len(gt) and i < len(pred) and gt[i] == pred[i]:
                    segment_stats[gt_seg.seg_type]['correct'] += 1
    
    return {
        'position': {k: {'accuracy': 100.0 * v['correct'] / v['total'] if v['total'] > 0 else 0, **v} 
                     for k, v in sorted(position_stats.items())},
        'segment': {k: {'accuracy': 100.0 * v['correct'] / v['total'] if v['total'] > 0 else 0, **v} 
                    for k, v in sorted(segment_stats.items())}
    }


def analyze_length_distribution(rows: List[Dict]) -> Dict:
    """
    分析长度分布和不同长度桶的准确率
    """
    length_buckets = {
        '1-5': {'min': 1, 'max': 5, 'correct': 0, 'total': 0},
        '6-10': {'min': 6, 'max': 10, 'correct': 0, 'total': 0},
        '11-15': {'min': 11, 'max': 15, 'correct': 0, 'total': 0},
        '16-20': {'min': 16, 'max': 20, 'correct': 0, 'total': 0},
        '21+': {'min': 21, 'max': 999, 'correct': 0, 'total': 0},
    }
    
    length_diff_dist = defaultdict(int)
    
    for row in rows:
        gt_len = len(row['gt'])
        pred_len = len(row['pred'])
        correct = row['correct'] == 'True'
        
        # 长度桶统计
        for bucket_name, bucket in length_buckets.items():
            if bucket['min'] <= gt_len <= bucket['max']:
                bucket['total'] += 1
                if correct:
                    bucket['correct'] += 1
                break
        
        # 长度差异分布
        diff = pred_len - gt_len
        length_diff_dist[diff] += 1
    
    # 计算准确率
    for bucket in length_buckets.values():
        bucket['accuracy'] = 100.0 * bucket['correct'] / bucket['total'] if bucket['total'] > 0 else 0
    
    return {
        'buckets': length_buckets,
        'diff_distribution': dict(sorted(length_diff_dist.items()))
    }


def analyze_conditional_errors(rows: List[Dict]) -> Dict:
    """
    条件错误分析
    """
    # difficulty × layout
    diff_layout_errors = defaultdict(lambda: defaultdict(int))
    # OOV × difficulty
    oov_diff_errors = defaultdict(lambda: defaultdict(int))
    
    for row in rows:
        difficulty = row['quality']
        layout = row['layout']
        vocab_type = row['vocabulary_type']
        error_type = row['error_type']
        
        # difficulty × layout
        diff_layout_errors[(difficulty, layout)][error_type] += 1
        
        # OOV × difficulty
        oov_diff_errors[(vocab_type, difficulty)][error_type] += 1
    
    return {
        'difficulty_layout': dict(diff_layout_errors),
        'oov_difficulty': dict(oov_diff_errors)
    }


def create_visualizations(results: Dict, output_dir: Path):
    """
    创建可视化图表
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. 字符混淆矩阵 (Top 20 常见字符)
    char_confusion = defaultdict(lambda: defaultdict(int))
    for row in results['rows']:
        if row['error_type'] in ['single_char_error', 'multi_char_error']:
            ops = analyze_edit_operations(row['gt'], row['pred'])
            for op in ops:
                if op.op_type == 'replace' and op.gt_char and op.pred_char:
                    char_confusion[op.gt_char][op.pred_char] += 1
    
    if char_confusion:
        # 获取最常见的字符
        all_chars = set()
        for gt, preds in char_confusion.items():
            all_chars.add(gt)
            all_chars.update(preds.keys())
        
        top_chars = sorted(all_chars, 
                          key=lambda c: sum(char_confusion[c].values()) + sum(char_confusion[x][c] for x in char_confusion), 
                          reverse=True)[:20]
        
        if len(top_chars) > 1:
            matrix = np.zeros((len(top_chars), len(top_chars)))
            for i, gt in enumerate(top_chars):
                for j, pred in enumerate(top_chars):
                    matrix[i, j] = char_confusion[gt].get(pred, 0)
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd', 
                       xticklabels=top_chars, yticklabels=top_chars, ax=ax)
            ax.set_title('Character Confusion Matrix (Top 20)')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Ground Truth')
            plt.tight_layout()
            plt.savefig(output_dir / 'confusion_matrix.png', dpi=150)
            plt.close()
    
    # 2. 长度差异热力图
    length_analysis = results['length_analysis']
    diff_dist = length_analysis['diff_distribution']
    
    if diff_dist:
        diffs = list(range(min(diff_dist.keys()), max(diff_dist.keys()) + 1))
        counts = [diff_dist.get(d, 0) for d in diffs]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(diffs, counts, color='steelblue', edgecolor='black')
        ax.set_xlabel('Length Difference (Pred - GT)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Length Differences')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Match')
        ax.legend()
        
        # 在柱子上添加数值
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'length_diff_distribution.png', dpi=150)
        plt.close()
    
    # 3. 位置准确率曲线
    pos_analysis = results['position_analysis']
    pos_stats = pos_analysis['position']
    
    if pos_stats:
        positions = list(pos_stats.keys())[:30]  # 前30个位置
        accuracies = [pos_stats[p]['accuracy'] for p in positions]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(positions, accuracies, marker='o', linewidth=2, markersize=6, color='steelblue')
        ax.set_xlabel('Character Position')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Position-wise Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'position_accuracy.png', dpi=150)
        plt.close()
    
    # 4. 各维度准确率对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Quality
    quality_data = results['quality_stats']
    if quality_data:
        ax = axes[0, 0]
        categories = list(quality_data.keys())
        accs = [quality_data[c]['accuracy'] for c in categories]
        ax.bar(categories, accs, color=['green', 'orange', 'red'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Difficulty')
        ax.set_ylim(0, 100)
        for i, v in enumerate(accs):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Layout
    layout_data = results['layout_stats']
    if layout_data:
        ax = axes[0, 1]
        categories = list(layout_data.keys())
        accs = [layout_data[c]['accuracy'] for c in categories]
        ax.bar(categories, accs, color='steelblue')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Layout')
        ax.set_ylim(0, 100)
        for i, v in enumerate(accs):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Vocabulary
    vocab_data = results['vocab_stats']
    if vocab_data:
        ax = axes[1, 0]
        categories = list(vocab_data.keys())
        accs = [vocab_data[c]['accuracy'] for c in categories]
        ax.bar(categories, accs, color=['green', 'red'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Vocabulary Type')
        ax.set_ylim(0, 100)
        for i, v in enumerate(accs):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    # Length buckets
    length_data = results['length_analysis']['buckets']
    if length_data:
        ax = axes[1, 1]
        categories = list(length_data.keys())
        accs = [length_data[c]['accuracy'] for c in categories]
        ax.bar(categories, accs, color='purple')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Length Bucket')
        ax.set_ylim(0, 100)
        for i, v in enumerate(accs):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150)
    plt.close()
    
    print(f"Visualizations saved to: {output_dir}")


def save_csv_results(results: Dict, output_dir: Path):
    """
    保存 CSV 结果
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 编辑操作统计
    edit_ops = results.get('edit_operations', {})
    with open(output_dir / 'edit_operations.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['operation_type', 'count', 'percentage'])
        total = sum(edit_ops.values()) if edit_ops else 1
        for op_type, count in sorted(edit_ops.items(), key=lambda x: -x[1]):
            writer.writerow([op_type, count, f"{100.0*count/total:.2f}%"])
    
    # 2. 位置准确率
    pos_analysis = results.get('position_analysis', {})
    with open(output_dir / 'position_accuracy.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['position', 'accuracy', 'correct', 'total'])
        for pos, stats in sorted(pos_analysis.get('position', {}).items()):
            writer.writerow([pos, f"{stats['accuracy']:.2f}%", stats['correct'], stats['total']])
    
    # 3. 分段准确率
    with open(output_dir / 'segment_accuracy.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['segment_type', 'accuracy', 'correct', 'total'])
        for seg_type, stats in sorted(pos_analysis.get('segment', {}).items()):
            writer.writerow([seg_type, f"{stats['accuracy']:.2f}%", stats['correct'], stats['total']])
    
    # 4. 长度分析
    length_analysis = results.get('length_analysis', {})
    with open(output_dir / 'length_analysis.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['length_bucket', 'accuracy', 'correct', 'total'])
        for bucket, stats in sorted(length_analysis.get('buckets', {}).items()):
            writer.writerow([bucket, f"{stats['accuracy']:.2f}%", stats['correct'], stats['total']])
    
    # 5. 长度差异分布
    with open(output_dir / 'length_diff_distribution.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['length_difference', 'count'])
        for diff, count in sorted(length_analysis.get('diff_distribution', {}).items()):
            writer.writerow([diff, count])
    
    # 6. 条件错误分析
    conditional = results.get('conditional_errors', {})
    
    # difficulty × layout
    with open(output_dir / 'conditional_difficulty_layout.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['difficulty', 'layout', 'error_type', 'count'])
        for (diff, layout), errors in sorted(conditional.get('difficulty_layout', {}).items()):
            for error_type, count in sorted(errors.items()):
                writer.writerow([diff, layout, error_type, count])
    
    # OOV × difficulty
    with open(output_dir / 'conditional_oov_difficulty.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['vocabulary_type', 'difficulty', 'error_type', 'count'])
        for (vocab, diff), errors in sorted(conditional.get('oov_difficulty', {}).items()):
            for error_type, count in sorted(errors.items()):
                writer.writerow([vocab, diff, error_type, count])
    
    print(f"CSV results saved to: {output_dir}")


def generate_report(results: Dict, output_path: Path):
    """
    生成 Markdown 报告
    """
    report = """# SLP34K Fine-Grained Error Analysis Report

## 1. 编辑操作分析

"""
    
    edit_ops = results.get('edit_operations', {})
    if edit_ops:
        report += "| Operation Type | Count | Percentage |\n"
        report += "|----------------|-------|------------|\n"
        total = sum(edit_ops.values())
        for op_type, count in sorted(edit_ops.items(), key=lambda x: -x[1]):
            report += f"| {op_type:20s} | {count:5d} | {100.0*count/total:6.2f}% |\n"
    
    report += """
## 2. 位置级准确率

"""
    
    pos_analysis = results.get('position_analysis', {})
    segment_stats = pos_analysis.get('segment', {})
    if segment_stats:
        report += "### 按文本段类型\n\n"
        report += "| Segment Type | Accuracy | Correct/Total |\n"
        report += "|--------------|----------|---------------|\n"
        for seg_type, stats in sorted(segment_stats.items()):
            report += f"| {seg_type:12s} | {stats['accuracy']:6.2f}% | {stats['correct']}/{stats['total']} |\n"
    
    report += """
## 3. 长度分析

### 按长度桶

"""
    
    length_analysis = results.get('length_analysis', {})
    buckets = length_analysis.get('buckets', {})
    if buckets:
        report += "| Length Bucket | Accuracy | Correct/Total |\n"
        report += "|---------------|----------|---------------|\n"
        for bucket, stats in sorted(buckets.items()):
            report += f"| {bucket:13s} | {stats['accuracy']:6.2f}% | {stats['correct']}/{stats['total']} |\n"
    
    report += """
## 4. 关键发现

"""
    
    # 自动分析关键发现
    findings = []
    
    # 编辑操作分析
    if edit_ops:
        eos_early = edit_ops.get('eos_early', 0)
        eos_late = edit_ops.get('eos_late', 0)
        total_errors = sum(edit_ops.values())
        if eos_early + eos_late > total_errors * 0.1:
            findings.append(f"- **EOS 问题**: EOS early ({eos_early}) 和 EOS late ({eos_late}) "
                           f"占总编辑操作的 {100.0*(eos_early+eos_late)/total_errors:.1f}%, "
                           f"建议优化序列结束预测机制")
    
    # 分段准确率差异
    if segment_stats:
        best_seg = max(segment_stats.items(), key=lambda x: x[1]['accuracy'])
        worst_seg = min(segment_stats.items(), key=lambda x: x[1]['accuracy'])
        if best_seg[1]['accuracy'] - worst_seg[1]['accuracy'] > 10:
            findings.append(f"- **段类型差异**: {best_seg[0]} ({best_seg[1]['accuracy']:.1f}%) "
                           f"vs {worst_seg[0]} ({worst_seg[1]['accuracy']:.1f}%), "
                           f"差距 {best_seg[1]['accuracy']-worst_seg[1]['accuracy']:.1f}%")
    
    # 长度桶分析
    if buckets:
        best_bucket = max(buckets.items(), key=lambda x: x[1]['accuracy'])
        worst_bucket = min(buckets.items(), key=lambda x: x[1]['accuracy'])
        if best_bucket[1]['accuracy'] - worst_bucket[1]['accuracy'] > 10:
            findings.append(f"- **长度影响**: {best_bucket[0]} ({best_bucket[1]['accuracy']:.1f}%) "
                           f"vs {worst_bucket[0]} ({worst_bucket[1]['accuracy']:.1f}%), "
                           f"表明模型对特定长度范围的文本识别能力有差异")
    
    if findings:
        report += "\n".join(findings)
    else:
        report += "- 未发现显著的性能差异模式。"
    
    report += """

## 5. 可视化图表

生成的图表：

1. `confusion_matrix.png` - 字符混淆矩阵 (Top 20)
2. `length_diff_distribution.png` - 长度差异分布
3. `position_accuracy.png` - 位置准确率曲线
4. `accuracy_comparison.png` - 各维度准确率对比

## 6. 详细数据

CSV 文件：

- `edit_operations.csv` - 编辑操作统计
- `position_accuracy.csv` - 位置级准确率
- `segment_accuracy.csv` - 分段准确率
- `length_analysis.csv` - 长度分析
- `length_diff_distribution.csv` - 长度差异分布
- `conditional_difficulty_layout.csv` - difficulty × layout 条件错误
- `conditional_oov_difficulty.csv` - OOV × difficulty 条件错误

---
*Generated by fine_grained_error_analysis.py*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Fine-Grained Error Analysis for SLP34K')
    parser.add_argument('error_csv', help='Path to error_analysis.csv')
    parser.add_argument('--output_dir', default='evaluation/results/fine_grained',
                       help='Output directory for results')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取错误分析数据
    print(f"Reading {args.error_csv}...")
    with open(args.error_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total samples: {len(rows)}")
    
    # 收集所有结果
    results = {'rows': rows}
    
    # 1. 编辑操作统计
    print("\nAnalyzing edit operations...")
    edit_operations = defaultdict(int)
    for row in rows:
        if row['error_type'] != 'correct':
            ops = analyze_edit_operations(row['gt'], row['pred'])
            for op in ops:
                edit_operations[op.op_type] += 1
    results['edit_operations'] = dict(edit_operations)
    print(f"  Found {len(edit_operations)} operation types")
    
    # 2. 位置级准确率
    print("\nAnalyzing position accuracy...")
    results['position_analysis'] = analyze_position_accuracy(rows)
    print("  Position analysis completed")
    
    # 3. 长度分析
    print("\nAnalyzing length distribution...")
    results['length_analysis'] = analyze_length_distribution(rows)
    print("  Length analysis completed")
    
    # 4. 条件错误分析
    print("\nAnalyzing conditional errors...")
    results['conditional_errors'] = analyze_conditional_errors(rows)
    print("  Conditional analysis completed")
    
    # 基础统计
    results['quality_stats'] = {}
    results['layout_stats'] = {}
    results['vocab_stats'] = {}
    
    quality_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    layout_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    vocab_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for row in rows:
        quality_stats[row['quality']]['total'] += 1
        layout_stats[row['layout']]['total'] += 1
        vocab_stats[row['vocabulary_type']]['total'] += 1
        if row['correct'] == 'True':
            quality_stats[row['quality']]['correct'] += 1
            layout_stats[row['layout']]['correct'] += 1
            vocab_stats[row['vocabulary_type']]['correct'] += 1
    
    results['quality_stats'] = {k: {'accuracy': 100.0 * v['correct'] / v['total'], **v} 
                                for k, v in quality_stats.items()}
    results['layout_stats'] = {k: {'accuracy': 100.0 * v['correct'] / v['total'], **v} 
                               for k, v in layout_stats.items()}
    results['vocab_stats'] = {k: {'accuracy': 100.0 * v['correct'] / v['total'], **v} 
                              for k, v in vocab_stats.items()}
    
    # 保存 CSV 结果
    print("\nSaving CSV results...")
    save_csv_results(results, output_dir)
    
    # 创建可视化
    print("\nCreating visualizations...")
    create_visualizations(results, output_dir)
    
    # 生成报告
    print("\nGenerating report...")
    generate_report(results, output_dir / 'fine_grained_report.md')
    
    print("\n" + "="*60)
    print("Fine-grained analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
