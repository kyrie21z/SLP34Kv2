#!/usr/bin/env python3
"""
generate_phase1_report.py — Phase 1 analysis: baseline vs +length-head.

Reads one or two samples.csv files produced by evaluate_unified.py and outputs:
  • phase1_report.md       — human-readable markdown report
  • eos_analysis.csv       — per-sample file for EOS-aware decoding
  • plots/*.png            — distribution charts

Reuses analyze_edit_operations() from fine_grained_error_analysis.py.

Usage
-----
  cd ocr_training

  # baseline only
  python evaluation/generate_phase1_report.py \\
      --baseline evaluation/results/baseline_eval/samples.csv \\
      --output_dir evaluation/results/phase1

  # baseline + length-head comparison
  python evaluation/generate_phase1_report.py \\
      --baseline evaluation/results/baseline_eval/samples.csv \\
      --length_head evaluation/results/length_head_eval/samples.csv \\
      --output_dir evaluation/results/phase1
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# reuse edit-op analysis from existing script
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))
from fine_grained_error_analysis import analyze_edit_operations

# ── length buckets (as requested) ────────────────────────────────────────────
LENGTH_BUCKETS = [
    ('0-8',   0,  8),
    ('9-12',  9, 12),
    ('13-16', 13, 16),
    ('17-20', 17, 20),
    ('21+',   21, 9999),
]


def bucket_of(gt_len: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= gt_len <= hi:
            return name
    return '21+'


# ── CSV loading ───────────────────────────────────────────────────────────────
def load_csv(path: str) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['correct']           = row['correct'] in ('True', 'true', '1')
            row['gt_len']            = int(row['gt_len'])
            row['pred_text_len']     = int(row['pred_text_len'])
            row['pred_len_from_head']= int(row['pred_len_from_head'])
            rows.append(row)
    return rows


# ── edit-distance (standalone, for segment CER) ───────────────────────────────
def edit_distance(s1: str, s2: str) -> int:
    if not s1: return len(s2)
    if not s2: return len(s1)
    prev = list(range(len(s2) + 1))
    for c1 in s1:
        curr = [prev[0] + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[-1] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def classify_char(c: str) -> str:
    if '\u4e00' <= c <= '\u9fff':
        return 'chinese'
    if c.isdigit():
        return 'digit'
    if c.isalpha() and c.isascii():
        return 'pinyin'
    return 'other'


def extract_chars(text: str, char_type: str) -> str:
    return ''.join(c for c in text if classify_char(c) == char_type)


# ── metric functions ──────────────────────────────────────────────────────────
def overall_metrics(rows: List[Dict]) -> Dict:
    total   = len(rows)
    correct = sum(1 for r in rows if r['correct'])
    return {'total': total, 'correct': correct,
            'accuracy': 100.0 * correct / total if total else 0.0}


def attr_breakdown(rows: List[Dict], field: str, cats: List[str]) -> Dict:
    stats = {}
    for cat in cats:
        sub = [r for r in rows if r[field] == cat]
        if not sub:
            continue
        c = sum(1 for r in sub if r['correct'])
        stats[cat] = {'total': len(sub), 'correct': c,
                      'accuracy': 100.0 * c / len(sub)}
    return stats


def length_error_rate(rows: List[Dict]) -> Dict:
    total   = len(rows)
    n_err   = sum(1 for r in rows if r['gt_len'] != r['pred_text_len'])
    return {'count': n_err, 'rate': 100.0 * n_err / total if total else 0.0}


def eos_stats(rows: List[Dict]) -> Dict:
    counter = defaultdict(int)
    for r in rows:
        counter[r['eos_type']] += 1
    total = len(rows)
    return {k: {'count': v, 'rate': 100.0 * v / total}
            for k, v in counter.items()}


def bucket_accuracy(rows: List[Dict]) -> Dict:
    stats = {name: {'total': 0, 'correct': 0} for name, _, _ in LENGTH_BUCKETS}
    for r in rows:
        name = bucket_of(r['gt_len'])
        stats[name]['total'] += 1
        if r['correct']:
            stats[name]['correct'] += 1
    for s in stats.values():
        t = s['total']
        s['accuracy'] = 100.0 * s['correct'] / t if t else 0.0
    return stats


def length_head_metrics(rows: List[Dict]) -> Optional[Dict]:
    valid = [r for r in rows if r['pred_len_from_head'] >= 0]
    if not valid:
        return None
    total  = len(valid)
    exact  = sum(1 for r in valid if r['pred_len_from_head'] == r['gt_len'])
    mae    = sum(abs(r['pred_len_from_head'] - r['gt_len']) for r in valid) / total
    bias   = defaultdict(int)
    for r in valid:
        bias[r['pred_len_from_head'] - r['gt_len']] += 1
    # length dist: gt_len vs pred_len_from_head
    gt_dist   = defaultdict(int)
    head_dist = defaultdict(int)
    for r in valid:
        gt_dist[r['gt_len']] += 1
        head_dist[r['pred_len_from_head']] += 1
    return {
        'total': total,
        'accuracy': 100.0 * exact / total,
        'mae': mae,
        'bias_distribution': dict(sorted(bias.items())),
        'gt_len_distribution':   dict(sorted(gt_dist.items())),
        'head_len_distribution': dict(sorted(head_dist.items())),
    }


def pred_text_len_dist(rows: List[Dict]) -> Dict:
    d = defaultdict(int)
    for r in rows:
        d[r['pred_text_len']] += 1
    return dict(sorted(d.items()))


def segment_metrics(rows: List[Dict]) -> Dict:
    result = {}
    for seg in ('chinese', 'digit', 'pinyin'):
        n_samples = ed_sum = exact_count = 0
        for r in rows:
            gt_seg   = extract_chars(r['gt'],   seg)
            pred_seg = extract_chars(r['pred'],  seg)
            if not gt_seg:
                continue
            n_samples += 1
            ed_sum    += edit_distance(gt_seg, pred_seg) / max(len(gt_seg), 1)
            if gt_seg == pred_seg:
                exact_count += 1
        result[seg] = {
            'n_samples':      n_samples,
            'cer':            100.0 * ed_sum / n_samples     if n_samples else 0.0,
            'exact_accuracy': 100.0 * exact_count / n_samples if n_samples else 0.0,
        }
    return result


def subset_metrics(rows: List[Dict]) -> Dict:
    subsets = {
        'hard':     [r for r in rows if r['quality'] == 'hard'],
        'OOV':      [r for r in rows if r['vocabulary_type'] == 'OOV'],
        'long_21+': [r for r in rows if r['gt_len'] >= 21],
    }
    return {name: overall_metrics(sub) for name, sub in subsets.items() if sub}


def edit_op_stats(rows: List[Dict]) -> Dict:
    """Aggregate character-level edit operations for erroneous samples."""
    counter = defaultdict(int)
    for r in rows:
        if not r['correct']:
            ops = analyze_edit_operations(r['gt'], r['pred'])
            for op in ops:
                counter[op.op_type] += 1
    return dict(counter)


# ── length dist frequency table (text) ───────────────────────────────────────
def freq_table_lines(dist: Dict[int, int], label: str, max_len: int = 50) -> List[str]:
    total = sum(dist.values())
    lines = [f'| {label} | Count | % |', '|---|---:|---:|']
    for k in range(0, max_len + 1):
        if k in dist:
            lines.append(f'| {k} | {dist[k]} | {100.0*dist[k]/total:.1f}% |')
    return lines


# ── plots ─────────────────────────────────────────────────────────────────────
def plot_bias(bias_dist: Dict[int, int], title: str, path: Path):
    if not bias_dist:
        return
    lo, hi = min(bias_dist), max(bias_dist)
    xs  = list(range(lo, hi + 1))
    ys  = [bias_dist.get(x, 0) for x in xs]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xs, ys, color='steelblue', edgecolor='black', alpha=0.8)
    ax.axvline(0, color='red', linestyle='--', lw=2, label='No bias')
    ax.set_xlabel('pred_len_from_head − gt_len')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_len_dist_compare(dist_a: Dict, dist_b: Optional[Dict],
                           label_a: str, label_b: str,
                           title: str, path: Path, max_len: int = 50):
    all_keys = list(dist_a.keys()) + (list(dist_b.keys()) if dist_b else [])
    lo = min(all_keys); hi = min(max(all_keys), max_len)
    xs = list(range(lo, hi + 1))
    tot_a = sum(dist_a.values())
    pct_a = [dist_a.get(x, 0) / tot_a * 100 for x in xs]

    fig, ax = plt.subplots(figsize=(13, 5))
    w = 0.4 if dist_b else 0.7
    offset = w / 2 if dist_b else 0
    ax.bar([x - offset for x in xs], pct_a, width=w,
           label=label_a, alpha=0.75, color='steelblue')
    if dist_b:
        tot_b = sum(dist_b.values())
        pct_b = [dist_b.get(x, 0) / tot_b * 100 for x in xs]
        ax.bar([x + offset for x in xs], pct_b, width=w,
               label=label_b, alpha=0.75, color='darkorange')
    ax.set_xlabel('Length')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_bucket_acc(b_buckets: Dict, h_buckets: Optional[Dict],
                    title: str, path: Path):
    names  = [n for n, _, _ in LENGTH_BUCKETS]
    b_accs = [b_buckets.get(n, {}).get('accuracy', 0) for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    w = 0.35 if h_buckets else 0.55
    ax.bar(x - (w / 2 if h_buckets else 0), b_accs, width=w,
           label='Baseline', color='steelblue', alpha=0.8)
    if h_buckets:
        h_accs = [h_buckets.get(n, {}).get('accuracy', 0) for n in names]
        ax.bar(x + w / 2, h_accs, width=w,
               label='+Length Head', color='darkorange', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ── markdown helpers ──────────────────────────────────────────────────────────
def two_col_table(b_dict: Dict, h_dict: Optional[Dict],
                  keys: List[str], field: str = 'accuracy') -> List[str]:
    if h_dict is not None:
        lines = ['| Attribute | Baseline | +LengthHead | Δ |',
                 '|-----------|----------|-------------|---|']
        for k in keys:
            b = b_dict.get(k)
            h = h_dict.get(k)
            if b and h:
                d = h['accuracy'] - b['accuracy']
                lines.append(
                    f'| {k} | {b["accuracy"]:.2f}% ({b["correct"]}/{b["total"]}) '
                    f'| {h["accuracy"]:.2f}% ({h["correct"]}/{h["total"]}) '
                    f'| {d:+.2f}% |'
                )
            elif b:
                lines.append(
                    f'| {k} | {b["accuracy"]:.2f}% ({b["correct"]}/{b["total"]}) | — | — |'
                )
    else:
        lines = ['| Attribute | Accuracy |', '|-----------|----------|']
        for k in keys:
            b = b_dict.get(k)
            if b:
                lines.append(f'| {k} | {b["accuracy"]:.2f}% ({b["correct"]}/{b["total"]}) |')
    return lines


# ── report generator ──────────────────────────────────────────────────────────
def generate_report(b_rows: List[Dict], h_rows: Optional[List[Dict]],
                    output_dir: Path) -> None:
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    has_head = h_rows is not None

    # ── compute all metrics ──
    b_overall  = overall_metrics(b_rows)
    b_quality  = attr_breakdown(b_rows, 'quality',         ['easy', 'middle', 'hard'])
    b_layout   = attr_breakdown(b_rows, 'layout',          ['single', 'multi', 'vertical'])
    b_vocab    = attr_breakdown(b_rows, 'vocabulary_type', ['IV', 'OOV'])
    b_res      = attr_breakdown(b_rows, 'resolution_type', ['normal', 'low'])
    b_len_err  = length_error_rate(b_rows)
    b_eos      = eos_stats(b_rows)
    b_buckets  = bucket_accuracy(b_rows)
    b_subsets  = subset_metrics(b_rows)
    b_segs     = segment_metrics(b_rows)
    b_edit_ops = edit_op_stats(b_rows)
    b_pred_dist = pred_text_len_dist(b_rows)
    b_gt_dist   = defaultdict(int)
    for r in b_rows:
        b_gt_dist[r['gt_len']] += 1

    if has_head:
        h_overall   = overall_metrics(h_rows)
        h_quality   = attr_breakdown(h_rows, 'quality',         ['easy', 'middle', 'hard'])
        h_layout    = attr_breakdown(h_rows, 'layout',          ['single', 'multi', 'vertical'])
        h_vocab     = attr_breakdown(h_rows, 'vocabulary_type', ['IV', 'OOV'])
        h_res       = attr_breakdown(h_rows, 'resolution_type', ['normal', 'low'])
        h_len_err   = length_error_rate(h_rows)
        h_eos       = eos_stats(h_rows)
        h_buckets   = bucket_accuracy(h_rows)
        h_subsets   = subset_metrics(h_rows)
        h_segs      = segment_metrics(h_rows)
        h_edit_ops  = edit_op_stats(h_rows)
        h_head_m    = length_head_metrics(h_rows)
        h_pred_dist = pred_text_len_dist(h_rows)

        # plots
        plot_len_dist_compare(
            dict(b_gt_dist), dict(b_gt_dist),
            'GT length (baseline)', 'GT length (+head)',
            'GT Length Distribution', plots_dir / 'gt_len_dist.png',
        )
        plot_len_dist_compare(
            b_pred_dist, h_pred_dist,
            'Baseline pred_text_len', '+Head pred_text_len',
            'Predicted Text Length Distribution', plots_dir / 'pred_text_len_dist.png',
        )
        if h_head_m:
            plot_bias(
                h_head_m['bias_distribution'],
                'Length Head Bias  (pred_len_from_head − gt_len)',
                plots_dir / 'head_bias.png',
            )
            plot_len_dist_compare(
                h_head_m['gt_len_distribution'],
                h_head_m['head_len_distribution'],
                'GT length', 'pred_len_from_head',
                'GT len vs Head-predicted len (+LengthHead)',
                plots_dir / 'gt_vs_head_len_dist.png',
            )
        plot_bucket_acc(
            b_buckets, h_buckets,
            'Accuracy by Length Bucket',
            plots_dir / 'bucket_accuracy.png',
        )
    else:
        h_quality = h_layout = h_vocab = h_res = {}
        h_buckets = None; h_subsets = {}; h_segs = {}; h_head_m = None
        plot_bucket_acc(b_buckets, None, 'Accuracy by Length Bucket',
                        plots_dir / 'bucket_accuracy.png')

    # ── build markdown ────────────────────────────────────────────────────────
    md = []

    def h(level, text):
        md.append('\n' + '#' * level + ' ' + text + '\n')

    def table(lines):
        md.extend(lines)
        md.append('')

    # Title
    md.append('# Phase 1 Analysis Report: Length Head\n')
    md.append('Generated by `generate_phase1_report.py`.\n')

    # ── 1. Overall accuracy ──
    h(2, '1. Overall Accuracy')
    if has_head:
        acc_delta = h_overall['accuracy'] - b_overall['accuracy']
        table([
            '| Model | Accuracy | Correct | Total |',
            '|-------|----------|---------|-------|',
            f'| Baseline     | {b_overall["accuracy"]:.2f}% | {b_overall["correct"]} | {b_overall["total"]} |',
            f'| +Length Head | {h_overall["accuracy"]:.2f}% | {h_overall["correct"]} | {h_overall["total"]} |',
        ])
        md.append(f'**Δ (accuracy): {acc_delta:+.2f}%**\n')
    else:
        md.append(
            f'- Accuracy: **{b_overall["accuracy"]:.2f}%** '
            f'({b_overall["correct"]}/{b_overall["total"]})\n'
        )

    # ── 2. Attribute breakdown ──
    h(2, '2. Breakdown by Attribute')

    h(3, '2.1 By Difficulty (Quality)')
    table(two_col_table(b_quality, h_quality if has_head else None,
                        ['easy', 'middle', 'hard']))

    h(3, '2.2 By Layout')
    table(two_col_table(b_layout, h_layout if has_head else None,
                        ['single', 'multi', 'vertical']))

    h(3, '2.3 By Vocabulary Type (IV / OOV)')
    table(two_col_table(b_vocab, h_vocab if has_head else None, ['IV', 'OOV']))

    h(3, '2.4 By Resolution')
    table(two_col_table(b_res, h_res if has_head else None, ['normal', 'low']))

    # ── 3. Length-related metrics ──
    h(2, '3. Length-Related Metrics')

    h(3, '3.1 Length Error Rate')
    if has_head:
        d = h_len_err['rate'] - b_len_err['rate']
        table([
            '| Model | Length Errors | Rate |',
            '|-------|---------------|------|',
            f'| Baseline     | {b_len_err["count"]} | {b_len_err["rate"]:.2f}% |',
            f'| +Length Head | {h_len_err["count"]} | {h_len_err["rate"]:.2f}% |',
        ])
        md.append(f'> Δ = {d:+.2f}%  (negative = fewer length errors)\n')
    else:
        md.append(
            f'- Length error rate: **{b_len_err["rate"]:.2f}%**  '
            f'({b_len_err["count"]} / {b_overall["total"]} samples)\n'
        )

    h(3, '3.2 EOS Statistics')
    eos_types = ['eos_early', 'eos_late', 'none']
    if has_head:
        table([
            '| EOS Type  | Baseline | Baseline% | +LengthHead | +Head% |',
            '|-----------|----------|-----------|-------------|--------|',
        ] + [
            f'| {et:9s} | {b_eos.get(et, {"count":0})["count"]:8d} '
            f'| {b_eos.get(et, {"rate":0.0})["rate"]:9.2f}% '
            f'| {h_eos.get(et, {"count":0})["count"]:11d} '
            f'| {h_eos.get(et, {"rate":0.0})["rate"]:6.2f}% |'
            for et in eos_types
        ])
    else:
        table(['| EOS Type  | Count | Rate |', '|-----------|-------|------|'] + [
            f'| {et:9s} | {b_eos.get(et, {"count":0})["count"]:5d} '
            f'| {b_eos.get(et, {"rate":0.0})["rate"]:5.2f}% |'
            for et in eos_types
        ])

    h(3, '3.3 Character-level Edit Operations (errors only)')
    def edit_op_table(ops_dict, label):
        total = sum(ops_dict.values()) or 1
        lines = [f'**{label}**', '',
                 '| Op Type | Count | % of Ops |',
                 '|---------|------:|----------:|']
        for op in sorted(ops_dict, key=lambda x: -ops_dict[x]):
            lines.append(f'| {op} | {ops_dict[op]} | {100.0*ops_dict[op]/total:.1f}% |')
        return lines
    md.extend(edit_op_table(b_edit_ops, 'Baseline'))
    md.append('')
    if has_head:
        md.extend(edit_op_table(h_edit_ops, '+Length Head'))
        md.append('')

    h(3, '3.4 Length Bucket Accuracy')
    if has_head:
        table([
            '| Length | Baseline | +LengthHead | Δ |',
            '|--------|----------|-------------|---|',
        ] + [
            f'| {name:6s} | {b_buckets[name]["accuracy"]:.2f}% ({b_buckets[name]["correct"]}/{b_buckets[name]["total"]}) '
            f'| {h_buckets[name]["accuracy"]:.2f}% ({h_buckets[name]["correct"]}/{h_buckets[name]["total"]}) '
            f'| {h_buckets[name]["accuracy"]-b_buckets[name]["accuracy"]:+.2f}% |'
            for name, _, _ in LENGTH_BUCKETS
        ])
    else:
        table(['| Length | Accuracy |', '|--------|----------|'] + [
            f'| {name:6s} | {b_buckets[name]["accuracy"]:.2f}% ({b_buckets[name]["correct"]}/{b_buckets[name]["total"]}) |'
            for name, _, _ in LENGTH_BUCKETS
        ])
    md.append('See `plots/bucket_accuracy.png` for chart.\n')

    h(3, '3.5 GT Length Distribution vs Predicted Text Length')
    # side-by-side frequency table (top-20 most common lengths)
    b_gt_items = sorted(b_gt_dist.items())
    b_pd_items = sorted(b_pred_dist.items())
    total_b = b_overall['total']
    md.append('Distribution of GT lengths (baseline dataset):')
    md.append('')
    md.append('| GT len | Count | % |')
    md.append('|-------:|------:|--:|')
    for k, v in b_gt_items:
        md.append(f'| {k} | {v} | {100.0*v/total_b:.1f}% |')
    md.append('')
    md.append('`plots/pred_text_len_dist.png` shows pred_text_len distribution for both models.\n')

    # ── 3.6 Length head metrics ──
    if has_head and h_head_m:
        h(3, '3.6 Length Head Metrics (+Length Head model only)')
        md.append(f'- **Head Accuracy** (exact length match): **{h_head_m["accuracy"]:.2f}%**')
        md.append(f'- **Head MAE**: **{h_head_m["mae"]:.3f}** characters')
        md.append('')
        # bias table (compact: ±5)
        bias = h_head_m['bias_distribution']
        total_valid = h_head_m['total']
        md.append('**Bias distribution** (pred_len_from_head − gt_len):')
        md.append('')
        md.append('| Bias | Count | % |')
        md.append('|-----:|------:|--:|')
        for b_val in sorted(bias):
            if -8 <= b_val <= 8:
                md.append(f'| {b_val:+d} | {bias[b_val]} | {100.0*bias[b_val]/total_valid:.1f}% |')
        md.append('')
        md.append(
            'See `plots/head_bias.png` and `plots/gt_vs_head_len_dist.png`.\n'
        )

    # ── 4. Subset results ──
    h(2, '4. Subset Results')
    subset_keys = ['hard', 'OOV', 'long_21+']
    if has_head:
        table([
            '| Subset | Baseline | +LengthHead | Δ |',
            '|--------|----------|-------------|---|',
        ] + [
            f'| {name} | {b_subsets[name]["accuracy"]:.2f}% ({b_subsets[name]["correct"]}/{b_subsets[name]["total"]}) '
            f'| {h_subsets.get(name, {"accuracy":0,"correct":0,"total":0})["accuracy"]:.2f}% '
            f'({h_subsets.get(name, {"correct":0})["correct"]}/{h_subsets.get(name, {"total":0})["total"]}) '
            f'| {h_subsets.get(name,{"accuracy":0})["accuracy"]-b_subsets[name]["accuracy"]:+.2f}% |'
            for name in subset_keys if name in b_subsets
        ])
    else:
        table(['| Subset | Accuracy |', '|--------|----------|'] + [
            f'| {name} | {b_subsets[name]["accuracy"]:.2f}% ({b_subsets[name]["correct"]}/{b_subsets[name]["total"]}) |'
            for name in subset_keys if name in b_subsets
        ])

    # ── 5. Segment analysis ──
    h(2, '5. Segment Analysis (Chinese / Digit / Pinyin)')
    if has_head:
        table([
            '| Segment | Baseline CER↓ | Baseline Acc↑ | +Head CER↓ | +Head Acc↑ |',
            '|---------|-------------|-------------|----------|----------|',
        ] + [
            f'| {seg} '
            f'| {b_segs[seg]["cer"]:.2f}% '
            f'| {b_segs[seg]["exact_accuracy"]:.2f}% '
            f'| {h_segs[seg]["cer"]:.2f}% '
            f'| {h_segs[seg]["exact_accuracy"]:.2f}% |'
            for seg in ('chinese', 'digit', 'pinyin')
        ])
    else:
        table([
            '| Segment | CER↓ | Exact Match Acc↑ | #Samples |',
            '|---------|------|-----------------|----------|',
        ] + [
            f'| {seg} | {b_segs[seg]["cer"]:.2f}% | {b_segs[seg]["exact_accuracy"]:.2f}% | {b_segs[seg]["n_samples"]} |'
            for seg in ('chinese', 'digit', 'pinyin')
        ])
    md.append('> CER = mean(edit_dist(gt_seg, pred_seg) / len(gt_seg)) over samples with that segment type.\n')

    # ── 6. Verdict ──
    h(2, '6. Verdict')

    b_eos_rate = (
        b_eos.get('eos_early', {'rate': 0.0})['rate'] +
        b_eos.get('eos_late',  {'rate': 0.0})['rate']
    )

    if has_head and h_head_m:
        acc_delta   = h_overall['accuracy'] - b_overall['accuracy']
        len_delta   = h_len_err['rate']    - b_len_err['rate']
        head_acc    = h_head_m['accuracy']
        head_mae    = h_head_m['mae']

        h(3, '6.1 Is the Length Head Effective?')
        head_acc_ok = head_acc >= 60.0
        head_mae_ok = head_mae <= 3.0
        md.extend([
            f'| Criterion | Value | Target | Result |',
            f'|-----------|-------|--------|--------|',
            f'| Head accuracy | {head_acc:.2f}% | ≥ 60% | {"✅" if head_acc_ok else "❌"} |',
            f'| Head MAE | {head_mae:.3f} chars | ≤ 3.0 | {"✅" if head_mae_ok else "❌"} |',
            f'| OCR accuracy Δ | {acc_delta:+.2f}% | > 0 | {"✅" if acc_delta > 0 else "⚠️"} |',
            f'| Length error Δ | {len_delta:+.2f}% | < 0 | {"✅" if len_delta < 0 else "⚠️"} |',
        ])
        md.append('')
        if head_acc_ok and head_mae_ok:
            md.append(
                '**Conclusion**: The length head has learned to predict sequence length with '
                f'reasonable accuracy ({head_acc:.1f}%) and low MAE ({head_mae:.2f}). '
                'It is a reliable signal for guiding EOS-aware decoding.\n'
            )
        else:
            md.append(
                '**Conclusion**: Length head accuracy is below target. '
                'Consider training longer or adjusting `length_loss_weight` '
                'before using it to guide decoding.\n'
            )

        h(3, '6.2 Is EOS-aware Decoding Worth Pursuing?')
        eos_ok = b_eos_rate >= 10.0
        md.extend([
            f'| Metric | Value | Threshold | Signal |',
            f'|--------|-------|-----------|--------|',
            f'| EOS early+late (baseline) | {b_eos_rate:.2f}% | ≥ 10% | {"✅ significant" if eos_ok else "⚠️ minor"} |',
            f'| Head accuracy | {head_acc:.2f}% | ≥ 60% | {"✅ usable" if head_acc_ok else "❌ too low"} |',
        ])
        md.append('')
        if eos_ok and head_acc_ok:
            md.append(
                f'**Recommendation**: ✅ **Proceed with EOS-aware decoding.** '
                f'{b_eos_rate:.1f}% of baseline samples have length errors (eos_early + eos_late), '
                f'and the length head ({head_acc:.1f}% accuracy) provides a reliable length prior. '
                'The `eos_analysis.csv` file contains all necessary signals.\n'
            )
        elif not eos_ok:
            md.append(
                '**Recommendation**: ⚠️ EOS-related errors are below 10% — '
                'EOS-aware decoding may have limited impact. '
                'Consider prioritising character-substitution errors instead.\n'
            )
        else:
            md.append(
                '**Recommendation**: ⚠️ EOS errors are significant but the length head '
                'accuracy is too low. Improve the head first, then add EOS-aware decoding.\n'
            )
    else:
        h(3, '6.1 Baseline-only Summary')
        md.append(f'- Length error rate: **{b_len_err["rate"]:.2f}%**')
        md.append(f'- EOS early+late: **{b_eos_rate:.2f}%**')
        md.append('')
        if b_eos_rate >= 10.0:
            md.append(
                '**Recommendation**: EOS-related errors are significant. '
                'Training with `use_length_head=true` and then running this script '
                'with `--length_head` will enable full comparison.\n'
            )

    # ── 7. File index ──
    h(2, '7. Output Files')
    md.extend([
        '| File | Description |',
        '|------|-------------|',
        '| `samples.csv` | Per-sample predictions (all columns) |',
        '| `eos_analysis.csv` | Input for EOS-aware decoding analysis |',
        '| `plots/bucket_accuracy.png` | Accuracy by length bucket |',
        '| `plots/pred_text_len_dist.png` | Predicted text length distributions |',
        '| `plots/head_bias.png` | Head prediction bias (+head model) |',
        '| `plots/gt_vs_head_len_dist.png` | GT vs head-predicted length (+head) |',
        '',
        '---',
        '_Generated by `evaluation/generate_phase1_report.py`_',
    ])

    report_text = '\n'.join(md)
    report_path = output_dir / 'phase1_report.md'
    report_path.write_text(report_text, encoding='utf-8')
    print(f'Report  : {report_path}')


# ── EOS analysis CSV ──────────────────────────────────────────────────────────
def save_eos_analysis(b_rows: List[Dict], h_rows: Optional[List[Dict]],
                      output_dir: Path) -> None:
    """
    Write eos_analysis.csv — one row per image, includes signals for EOS-aware decoding.
    Uses +length-head rows when available, otherwise baseline rows.
    """
    rows_to_use = h_rows if h_rows is not None else b_rows
    fieldnames = [
        'image_id', 'gt_len', 'pred_text_len', 'pred_len_from_head',
        'correct_or_not', 'eos_type', 'len_head_correct',
        'len_diff_text',  # pred_text_len - gt_len
        'len_diff_head',  # pred_len_from_head - gt_len  (-1 if no head)
    ]
    eos_path = output_dir / 'eos_analysis.csv'
    with open(eos_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_to_use:
            plh = r['pred_len_from_head']
            lhc = int(plh == r['gt_len']) if plh >= 0 else -1
            writer.writerow({
                'image_id':          r['image_id'],
                'gt_len':            r['gt_len'],
                'pred_text_len':     r['pred_text_len'],
                'pred_len_from_head': plh,
                'correct_or_not':    int(r['correct']),
                'eos_type':          r['eos_type'],
                'len_head_correct':  lhc,
                'len_diff_text':     r['pred_text_len'] - r['gt_len'],
                'len_diff_head':     plh - r['gt_len'] if plh >= 0 else -999,
            })
    print(f'EOS CSV : {eos_path}')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Phase 1 analysis report generator')
    parser.add_argument('--baseline',    required=True,
                        help='Baseline samples.csv from evaluate_unified.py')
    parser.add_argument('--length_head', default=None,
                        help='+Length head samples.csv (optional, enables comparison)')
    parser.add_argument('--output_dir',  default='evaluation/results/phase1',
                        help='Output directory for report and plots')
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading baseline    : {args.baseline}')
    b_rows = load_csv(args.baseline)
    print(f'  {len(b_rows)} samples')

    h_rows = None
    if args.length_head:
        print(f'Loading +length_head: {args.length_head}')
        h_rows = load_csv(args.length_head)
        print(f'  {len(h_rows)} samples')

    print('\nGenerating report ...')
    generate_report(b_rows, h_rows, out_dir)

    print('Saving EOS analysis ...')
    save_eos_analysis(b_rows, h_rows, out_dir)

    print(f'\nAll outputs → {out_dir}')


if __name__ == '__main__':
    main()
