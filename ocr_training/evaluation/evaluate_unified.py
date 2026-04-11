#!/usr/bin/env python3
"""
evaluate_unified.py — Unified evaluation for SLP34K baseline and +length-head models.

Runs inference on unified_lmdb (which carries per-image metadata) and writes a
comprehensive per-sample CSV that is:
  • A superset of error_analysis.csv  → still compatible with fine_grained_error_analysis.py
  • Ready to be consumed by generate_phase1_report.py

Output columns
--------------
  image_id, quality, layout, vocabulary_type, resolution_type,
  gt, pred, correct, error_type, note,
  gt_len, pred_text_len, eos_type, pred_len_from_head

Usage
-----
  cd ocr_training
  python evaluation/evaluate_unified.py <checkpoint> [options]

  # baseline
  python evaluation/evaluate_unified.py \\
      checkpoint/SLP34K_maevit_infonce_plm_SOTA_83.53%/checkpoints/last.ckpt \\
      --output_dir evaluation/results/baseline_eval

  # +length-head
  python evaluation/evaluate_unified.py \\
      outputs/new_oov/maevit_infonce_plm/2026-04-05_22-13-13/checkpoints/last.ckpt \\
      --output_dir evaluation/results/length_head_eval
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms as T
from PIL import Image
import io
import lmdb

# ── project root ──────────────────────────────────────────────────────────────
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from strhub.models.utils import load_from_checkpoint, parse_model_args
from strhub.data.utils import CharsetAdapter
from strhub.data.module import SceneTextDataModule

# ── constants ─────────────────────────────────────────────────────────────────
CHARSET_SLP34K = (
    "睢荷焦射渔灌馬轩森引猛球祁智卓禾翼ptihgy松园淠澎宗禹领茹斌潜舜孝感爱船帮玮丽月学燕炉玲事必思屹展長牛双邦霖粮纬亮"
    "致圣降语奥树昱配然郭唐uan珠蓝陆邳郡惜泾帅巡卸孟峻澳加涵淳毅神艇刘救助政百劲锋凌硕潮漂葆莱凓沛戈喜忠聚获抚绣意羽微"
    "梁久心午鸥甸渡杨韦友电焊勇征如满跃景齐朝子铭复壁庙涟逸欣舸关升经星晟溆浦冠咏多才统烁沙世力渝巢宛宸煜送鹰广之潥仁皋"
    "晶昶漕伦梓架普凤能捷濉石屏浚名仕前繁为好定帆喻颖卫袁旭保濠漯弋雅杰柏义俊军福沈津拖散乐蓼环威店枞亨姑塘嵊渚怡含丘飞"
    "波元骏青弘V沭雷傲大惠自梧财坤悦锐观音文春钓一沚程健荣荆昭佳众椒耀兆寺藤虎乾博贸工驻益游台得裕日韵茂茗融豪朗商辰靖"
    "化雨迎超涡全鄂黄冈湾胜亚鸣高滨成辉闽三浩驳启沪内同正氏民槽京汉荻驰西迁巨申泉庆志锦生王良来康钢舒林锚饶虹鹏君宜伟谐"
    "梅兰鞍俞淤固绿洲-溧动邱璧菏九峰泗交浍灵二水红陶四恒诚临怀池张界国邮霍强垛北铜泽洪玉家蒙五腾衢旺润舟姚轮川银瑞舵赣"
    "湘宏圆鼎泓凯隆常善清光方联风马创万业镇昌姜和乡宝陵滁寿武丰鸿连郎颍县辛汇柯洋云天振桥扬枣河永明建庐庄宇溪源发油太宿"
    "芜翔利宣六信徐PR号龙锡肥无合吉新东中桐德盛平南越祥鑫M华金K阜4口钱长通余蚌埠远达济鲁Q淮海宁顺运亳周嘉虞富上D湖盐"
    "集江安豫泰航机城F阳B山萧TLC苏绍诸暨杭W皖兴XS州J7Y35港E2Z浙961货IO08GAUHN"
)

CSV_FIELDNAMES = [
    'image_id', 'quality', 'layout', 'vocabulary_type', 'resolution_type',
    'gt', 'pred', 'correct', 'error_type', 'note',
    'gt_len', 'pred_text_len', 'eos_type', 'pred_len_from_head',
]

DIAG_FIELDNAMES = [
    'image_id', 'correct_before', 'pred_before', 'pred_after',
    'final_prediction_changed', 'gt_len', 'pred_text_len_before', 'pred_text_len_after',
    'pred_len_from_head', 'uncertainty_value', 'gating_enabled',
    'passed_uncertainty_gate', 'passed_length_gate', 'eos_bias_applied',
    'num_steps_modified', 'num_steps_argmax_changed',
    'eos_type_before', 'eos_type_after',
]


# ── helpers ───────────────────────────────────────────────────────────────────
def classify_error(gt: str, pred: str) -> str:
    if gt == pred:
        return 'correct'
    if len(gt) != len(pred):
        return 'length_error'
    if sorted(gt) == sorted(pred):
        return 'order_error'
    diff = sum(1 for g, p in zip(gt, pred) if g != p)
    return 'single_char_error' if diff == 1 else 'multi_char_error'


def get_eos_type(gt_len: int, pred_len: int) -> str:
    if pred_len < gt_len:
        return 'eos_early'
    if pred_len > gt_len:
        return 'eos_late'
    return 'none'


# ── dataset ───────────────────────────────────────────────────────────────────
class UnifiedLmdbDataset(Dataset):
    """Reads unified_lmdb which stores image / label / meta (JSON)."""

    def __init__(
        self,
        root: str,
        charset: str,
        max_label_len: int,
        transform=None,
        remove_whitespace: bool = False,
        allowed_ids=None,
    ):
        self.root = root
        self.charset_adapter = CharsetAdapter(charset)
        self.max_label_len = max_label_len
        self.remove_whitespace = remove_whitespace
        self.transform = transform
        self.allowed_ids = None if allowed_ids is None else {str(x) for x in allowed_ids}
        self._env = None
        self.labels = []
        self.metas = []
        self.filtered_index_list = []
        self.num_samples = self._preprocess_labels()

    def _open_env(self):
        return lmdb.open(
            self.root, max_readers=1, readonly=True, create=False,
            readahead=False, meminit=False, lock=False,
        )

    @property
    def env(self):
        if self._env is None:
            self._env = self._open_env()
        return self._env

    def _preprocess_labels(self) -> int:
        with self._open_env() as e:
            with e.begin() as txn:
                num_samples = int(txn.get(b'num-samples'))
                for index in range(num_samples):
                    index += 1  # LMDB keys are 1-based
                    label = txn.get(f'label-{index:09d}'.encode()).decode()
                    if self.remove_whitespace:
                        label = ''.join(label.split())
                    if len(label) > self.max_label_len:
                        continue
                    label = self.charset_adapter(label)
                    if not label:
                        continue
                    meta_raw = txn.get(f'meta-{index:09d}'.encode())
                    meta = json.loads(meta_raw.decode()) if meta_raw else {}
                    if self.allowed_ids is not None and str(meta.get('id', '')) not in self.allowed_ids:
                        continue
                    self.labels.append(label)
                    self.metas.append(meta)
                    self.filtered_index_list.append(index)
                return len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        i = self.filtered_index_list[idx]
        with self.env.begin() as txn:
            img_bytes = txn.get(f'image-{i:09d}'.encode())
        label = self.labels[idx]
        meta = self.metas[idx]
        img  = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # Serialize meta so DataLoader can collate it
        return img, label, json.dumps(meta, ensure_ascii=False)


def collate_fn(batch):
    imgs, labels, metas = zip(*batch)
    return (
        torch.stack(imgs),
        list(labels),
        [json.loads(m) for m in metas],
    )


# ── main ──────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description='Unified SLP34K evaluation')
    parser.add_argument('checkpoint', help='Path to model checkpoint (.ckpt)')
    parser.add_argument(
        '--unified_db',
        default='data/test/SLP34K_lmdb_benchmark/unified_lmdb',
        help='Path to unified_lmdb directory',
    )
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader workers (0 = main process, safest for LMDB)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: <ckpt_parent_parent>/eval_unified)')
    parser.add_argument('--enable_eos_diagnostics', action='store_true',
                        help='Enable selective-EOS diagnostic outputs')
    parser.add_argument('--save_eos_diagnostics_csv', action='store_true',
                        help='Save eos_diagnostics.csv when diagnostics are enabled')
    parser.add_argument('--save_eos_diagnostics_report', action='store_true',
                        help='Save phase2_diagnostic_report.md when diagnostics are enabled')
    parser.add_argument('--image_ids_file', default=None,
                        help='Optional text/csv file with image ids to evaluate, one per line')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    ckpt = Path(args.checkpoint)
    out_dir = (Path(args.output_dir) if args.output_dir
               else ckpt.parent.parent / 'eval_unified')
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    print(f'Loading checkpoint: {ckpt}')
    model = load_from_checkpoint(
        str(ckpt),
        mae_pretrained_path='pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth',
        charset_test=CHARSET_SLP34K,
        **kwargs,
    ).eval().to(args.device)
    hp = model.hparams
    has_head = getattr(model, 'use_length_head', False)
    print(f'Length head: {has_head}')
    diagnostics_enabled = (
        args.enable_eos_diagnostics
        or getattr(model, 'enable_eos_diagnostics', False)
    )
    if diagnostics_enabled and hasattr(model, 'reset_eos_diagnostics'):
        model.reset_eos_diagnostics()

    # ── data ──────────────────────────────────────────────────────────────────
    img_size  = getattr(hp, 'img_size', [224, 224])
    transform = SceneTextDataModule.get_transform(tuple(img_size))
    allowed_ids = None
    if args.image_ids_file:
        allowed_ids = []
        with open(args.image_ids_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                allowed_ids.append(line.split(',')[0].strip())
    dataset = UnifiedLmdbDataset(
        args.unified_db,
        charset=CHARSET_SLP34K,
        max_label_len=getattr(hp, 'max_label_length', 50),
        transform=transform,
        allowed_ids=allowed_ids,
    )
    loader  = DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, collate_fn=collate_fn,
        shuffle=False,
    )
    print(f'Total samples: {len(dataset)}')

    # ── inference ─────────────────────────────────────────────────────────────
    rows = []
    diag_rows = []
    correct_count = 0

    for imgs, labels, metas in tqdm(loader, desc='Inference'):
        imgs   = imgs.to(args.device)
        memory = model.encode(imgs)
        preds_before = [None] * len(labels)
        if diagnostics_enabled and getattr(model, 'use_selective_eos_aware_decoding', False):
            old_use_eos = model.use_selective_eos_aware_decoding
            old_diag = getattr(model, 'enable_eos_diagnostics', False)
            model.use_selective_eos_aware_decoding = False
            model.enable_eos_diagnostics = False
            logits_before = model.forward(imgs)
            probs_before = logits_before.softmax(-1)
            preds_before, _ = model.tokenizer.decode(probs_before)
            model.use_selective_eos_aware_decoding = old_use_eos
            model.enable_eos_diagnostics = old_diag

        logits = model.forward(imgs)
        probs  = logits.softmax(-1)
        preds, _ = model.tokenizer.decode(probs)
        batch_diag = []
        if diagnostics_enabled and hasattr(model, 'consume_last_eos_diagnostics_batch'):
            batch_diag = model.consume_last_eos_diagnostics_batch()
        if not batch_diag:
            batch_diag = [{
                'pred_len_from_head': -1,
                'uncertainty_value': 0.0,
                'gating_enabled': False,
                'passed_uncertainty_gate': False,
                'passed_length_gate': False,
                'eos_bias_applied': False,
                'num_steps_modified': 0,
                'num_steps_argmax_changed': 0,
                'any_token_changed': False,
            } for _ in labels]

        if has_head:
            len_logits  = model.length_head(memory)
            head_preds  = len_logits.argmax(dim=-1).cpu().tolist()
        else:
            head_preds = [-1] * len(labels)

        for pred_before_raw, pred_raw, gt, meta, head_len, diag in zip(preds_before, preds, labels, metas, head_preds, batch_diag):
            pred_before = model.charset_adapter(pred_before_raw) if pred_before_raw is not None else ''
            pred       = model.charset_adapter(pred_raw)
            is_correct = pred == gt
            gt_len     = len(gt)
            pred_len   = len(pred)
            if is_correct:
                correct_count += 1

            rows.append({
                'image_id':          str(meta.get('id', '')),
                'quality':           meta.get('quality', ''),
                'layout':            meta.get('structure', ''),
                'vocabulary_type':   meta.get('vocabulary_type', ''),
                'resolution_type':   meta.get('resolution_type', ''),
                'gt':                gt,
                'pred':              pred,
                'correct':           is_correct,
                'error_type':        classify_error(gt, pred),
                'note':              '',
                'gt_len':            gt_len,
                'pred_text_len':     pred_len,
                'eos_type':          get_eos_type(gt_len, pred_len),
                'pred_len_from_head': int(head_len),
            })
            if diagnostics_enabled:
                pred_before_len = len(pred_before)
                diag_rows.append({
                    'image_id':               str(meta.get('id', '')),
                    'correct_before':         pred_before == gt,
                    'pred_before':            pred_before,
                    'pred_after':             pred,
                    'final_prediction_changed': pred_before != pred,
                    'gt_len':                 gt_len,
                    'pred_text_len_before':   pred_before_len,
                    'pred_text_len_after':    pred_len,
                    'pred_len_from_head':     int(diag['pred_len_from_head']),
                    'uncertainty_value':      float(diag['uncertainty_value']),
                    'gating_enabled':         bool(diag['gating_enabled']),
                    'passed_uncertainty_gate': bool(diag['passed_uncertainty_gate']),
                    'passed_length_gate':     bool(diag['passed_length_gate']),
                    'eos_bias_applied':       bool(diag['eos_bias_applied']),
                    'num_steps_modified':     int(diag['num_steps_modified']),
                    'num_steps_argmax_changed': int(diag['num_steps_argmax_changed']),
                    'eos_type_before':        get_eos_type(gt_len, pred_before_len),
                    'eos_type_after':         get_eos_type(gt_len, pred_len),
                })

    # ── report ────────────────────────────────────────────────────────────────
    total       = len(rows)
    overall_acc = 100.0 * correct_count / total
    print(f'\nOverall Accuracy: {overall_acc:.2f}%  ({correct_count}/{total})')
    if has_head:
        head_correct = sum(
            1 for r in rows
            if r['pred_len_from_head'] >= 0 and r['pred_len_from_head'] == r['gt_len']
        )
        print(f'Length Head Accuracy: {100.0 * head_correct / total:.2f}%')

    # ── save CSV ──────────────────────────────────────────────────────────────
    csv_path = out_dir / 'samples.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f'CSV saved: {csv_path}')

    # ── save summary ──────────────────────────────────────────────────────────
    summary_lines = [
        f'checkpoint       : {ckpt}',
        f'has_length_head  : {has_head}',
        f'total_samples    : {total}',
        f'overall_accuracy : {overall_acc:.4f}%',
    ]
    summary_path = out_dir / 'eval_summary.txt'
    summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print(f'Summary: {summary_path}')

    if diagnostics_enabled:
        diag_summary = model.get_eos_diagnostics_summary() if hasattr(model, 'get_eos_diagnostics_summary') else {}
        diag_summary['num_samples_final_prediction_changed'] = sum(
            1 for r in diag_rows if r['final_prediction_changed']
        )
        if args.save_eos_diagnostics_csv or args.save_eos_diagnostics_report:
            diag_csv_path = out_dir / 'eos_diagnostics.csv'
            with open(diag_csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=DIAG_FIELDNAMES)
                writer.writeheader()
                writer.writerows(diag_rows)
            print(f'Diagnostics CSV: {diag_csv_path}')
        if args.save_eos_diagnostics_report:
            changed_counter = Counter()
            for r in diag_rows:
                if r['final_prediction_changed']:
                    changed_counter[(r['eos_type_before'], r['eos_type_after'])] += 1
            reason = 'gating too strict'
            if diag_summary.get('num_steps_eos_logit_modified', 0) > 0 and diag_summary.get('num_steps_argmax_changed', 0) == 0:
                reason = 'bias too weak'
            elif diag_summary.get('num_steps_argmax_changed', 0) > 0 and diag_summary.get('num_samples_final_prediction_changed', 0) == 0:
                reason = 'token-level change did not propagate to final prediction'
            report_path = out_dir / 'phase2_diagnostic_report.md'
            lines = [
                '# Phase 2 Diagnostic Report',
                '',
                '## Summary',
                f"- total_samples: {diag_summary.get('total_samples', 0)}",
                f"- num_samples_gating_enabled: {diag_summary.get('num_samples_gating_enabled', 0)}",
                f"- num_samples_gating_disabled: {diag_summary.get('num_samples_gating_disabled', 0)}",
                f"- num_samples_failed_uncertainty_gate: {diag_summary.get('num_samples_failed_uncertainty_gate', 0)}",
                f"- num_samples_failed_length_gate: {diag_summary.get('num_samples_failed_length_gate', 0)}",
                f"- num_samples_eos_bias_applied: {diag_summary.get('num_samples_eos_bias_applied', 0)}",
                f"- num_samples_any_token_changed: {diag_summary.get('num_samples_any_token_changed', 0)}",
                f"- num_samples_final_prediction_changed: {diag_summary.get('num_samples_final_prediction_changed', 0)}",
                '',
                '## Step-Level',
                f"- total_decode_steps: {diag_summary.get('total_decode_steps', 0)}",
                f"- num_steps_gating_active: {diag_summary.get('num_steps_gating_active', 0)}",
                f"- num_steps_eos_logit_modified: {diag_summary.get('num_steps_eos_logit_modified', 0)}",
                f"- num_steps_eos_rank_changed: {diag_summary.get('num_steps_eos_rank_changed', 0)}",
                f"- num_steps_argmax_changed: {diag_summary.get('num_steps_argmax_changed', 0)}",
                '',
                '## Interpretation',
                f"- current 0-change diagnosis: {reason}",
                '',
                '## EOS Type Changes',
            ]
            if changed_counter:
                for (before, after), count in changed_counter.items():
                    lines.append(f"- {before} -> {after}: {count}")
            else:
                lines.append('- no final prediction changes')
            lines.extend([
                '',
                '## Next Parameter To Tune',
                '- If almost no samples enter gating: lower `uncertainty_threshold` or relax length gating.',
                '- If EOS logits are modified but argmax never changes: increase `eos_suppress_bias` / `eos_boost_bias`.',
                '- If argmax changes but final prediction does not: inspect when the changed token occurs and whether it is before first EOS.',
            ])
            report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            print(f'Diagnostics Report: {report_path}')

    print('\nNext — generate analysis report:')
    print(f'  python evaluation/generate_phase1_report.py \\')
    print(f'      --baseline {csv_path} \\')
    print(f'      --output_dir evaluation/results/phase1')


if __name__ == '__main__':
    main()
