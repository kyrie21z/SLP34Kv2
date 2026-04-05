#!/usr/bin/env python3
"""
Test script with length prediction analysis.

Outputs:
1. Standard accuracy metrics (same as test.py)
2. Length prediction accuracy
3. CSV file with detailed length analysis:
   - image_id
   - gt_text
   - gt_len
   - pred_len_from_head
   - pred_text
   - pred_text_len
   - length_match (bool)
"""

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import torch
from tqdm import tqdm

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint, parse_model_args


@dataclass
class Result:
    dataset: str
    num_samples: int
    accuracy: float
    ned: float
    confidence: float
    label_length: float
    length_acc: float  # Length prediction accuracy


def print_results_table(results: List[Result], file=None):
    w = max(map(len, map(getattr, results, ['dataset'] * len(results))))
    w = max(w, len('Dataset'), len('Combined'))
    print('| {:<{w}} | # samples | Accuracy | 1 - NED | Confidence | Label Length | Length Acc |'.format('Dataset', w=w), file=file)
    print('|:{:-<{w}}:|----------:|---------:|--------:|-----------:|-------------:|-----------:|'.format('----', w=w), file=file)
    c = Result('Combined', 0, 0, 0, 0, 0, 0)
    for res in results:
        c.num_samples += res.num_samples
        c.accuracy += res.num_samples * res.accuracy
        c.ned += res.num_samples * res.ned
        c.confidence += res.num_samples * res.confidence
        c.label_length += res.num_samples * res.label_length
        c.length_acc += res.num_samples * res.length_acc
        print(f'| {res.dataset:<{w}} | {res.num_samples:>9} | {res.accuracy:>8.2f} | {res.ned:>7.2f} '
              f'| {res.confidence:>10.2f} | {res.label_length:>12.2f} | {res.length_acc:>10.2f} |', file=file)
    c.accuracy /= c.num_samples
    c.ned /= c.num_samples
    c.confidence /= c.num_samples
    c.label_length /= c.num_samples
    c.length_acc /= c.num_samples
    print('|-{:-<{w}}-|-----------|----------|---------|------------|--------------|------------|'.format('----', w=w), file=file)
    print(f'| {c.dataset:<{w}} | {c.num_samples:>9} | {c.accuracy:>8.2f} | {c.ned:>7.2f} '
          f'| {c.confidence:>10.2f} | {c.label_length:>12.2f} | {c.length_acc:>10.2f} |', file=file)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help="Model checkpoint")
    parser.add_argument('--data_root', default='data')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--test_data', default='SLP34K', choices=['SLP34K', 'SLP34K_UNIFIED'])
    parser.add_argument('--test_dir', default='SLP34K_lmdb_benchmark')
    parser.add_argument('--output_csv', default=None, help='Output CSV file for length analysis')
    args, unknown = parser.parse_known_args()
    kwargs = parse_model_args(unknown)

    # SLP34K charset
    if args.test_data in ["SLP34K", "SLP34K_UNIFIED"]:
        charset_test = "睢荷焦射渔灌馬轩森引猛球祁智卓禾翼ptihgy松园淠澎宗禹领茹斌潜舜孝感爱船帮玮丽月学燕炉玲事必思屹展長牛双邦霖粮纬亮致圣降语奥树昱配然郭唐uan珠蓝陆邳郡惜泾帅巡卸孟峻澳加涵淳毅神艇刘救助政百劲锋凌硕潮漂葆莱凓沛戈喜忠聚获抚绣意羽微梁久心午鸥甸渡杨韦友电焊勇征如满跃景齐朝子铭复壁庙涟逸欣舸关升经星晟溆浦冠咏多才统烁沙世力渝巢宛宸煜送鹰广之潥仁皋晶昶漕伦梓架普凤能捷濉石屏浚名仕前繁为好定帆喻颖卫袁旭保濠漯弋雅杰柏义俊军福沈津拖散乐蓼环威店枞亨姑塘嵊渚怡含丘飞波元骏青弘V沭雷傲大惠自梧财坤悦锐观音文春钓一沚程健荣荆昭佳众椒耀兆寺藤虎乾博贸工驻益游台得裕日韵茂茗融豪朗商辰靖化雨迎超涡全鄂黄冈湾胜亚鸣高滨成辉闽三浩驳启沪内同正氏民槽京汉荻驰西迁巨申泉庆志锦生王良来康钢舒林锚饶虹鹏君宜伟谐梅兰鞍俞淤固绿洲-溧动邱璧菏九峰泗交浍灵二水红陶四恒诚临怀池张界国邮霍强垛北铜泽洪玉家蒙五腾衢旺润舟姚轮川银瑞舵赣湘宏圆鼎泓凯隆常善清光方联风马创万业镇昌姜和乡宝陵滁寿武丰鸿连郎颍县辛汇柯洋云天振桥扬枣河永明建庐庄宇溪源发油太宿芜翔利宣六信徐PR号龙锡肥无合吉新东中桐德盛平南越祥鑫M华金K阜4口钱长通余蚌埠远达济鲁Q淮海宁顺运亳周嘉虞富上D湖盐集江安豫泰航机城F阳B山萧TLC苏绍诸暨杭W皖兴XS州J7Y35港E2Z浙961货IO08GAUHN"
        kwargs.update({'mae_pretrained_path': "pretrain_model/ship/224x224_pretrain_ship_vit_checkpoint-1499.pth"})
    else:
        raise ValueError(f"Unsupported test_data: {args.test_data}")

    kwargs.update({'charset_test': charset_test})
    
    print(f'Additional keyword arguments: {kwargs}')

    # Load model
    model = load_from_checkpoint(args.checkpoint, **kwargs).eval().to(args.device)
    hp = model.hparams
    
    # Check if model has length head
    has_length_head = getattr(model, 'use_length_head', False)
    print(f"Model has length head: {has_length_head}")
    
    datamodule = SceneTextDataModule(
        args.data_root, '_unused_', '_unused_', args.test_dir,
        hp.img_size, hp.max_label_length, hp.charset_train,
        charset_test, args.batch_size, args.num_workers, False
    )

    if args.test_data == "SLP34K":
        test_set = SceneTextDataModule.TEST_SLP34K
    elif args.test_data == "SLP34K_UNIFIED":
        # Only test unified_lmdb which contains complete metadata
        test_set = ['unified_lmdb']
    else:
        raise ValueError(f"Unsupported test_data: {args.test_data}")

    # Prepare CSV output
    csv_data = []
    
    results = {}
    max_width = max(map(len, test_set))
    
    for name, dataloader in datamodule.test_dataloaders(test_set).items():
        total = 0
        correct = 0
        ned = 0
        confidence = 0
        label_length = 0
        length_correct = 0
        
        for batch in tqdm(iter(dataloader), desc=f'{name:>{max_width}}'):
            imgs, labels = batch
            imgs = imgs.to(model.device)
            
            # Get encoder features
            memory = model.encode(imgs)
            
            # Get predictions
            logits = model.forward(imgs)
            probs = logits.softmax(-1)
            preds, pred_probs = model.tokenizer.decode(probs)
            
            # Get length predictions if available
            if has_length_head:
                length_logits = model.length_head(memory)
                pred_lengths = length_logits.argmax(dim=-1).cpu().tolist()
            else:
                pred_lengths = [None] * len(labels)
            
            for i, (pred, gt, pred_len) in enumerate(zip(preds, labels, pred_lengths)):
                pred = model.charset_adapter(pred)
                gt_len = len(gt)
                pred_text_len = len(pred)
                
                # Standard metrics
                from nltk import edit_distance
                confidence += pred_probs[i].prod().item()
                ned += edit_distance(pred, gt) / max(len(pred), len(gt))
                if pred == gt:
                    correct += 1
                total += 1
                label_length += len(pred)
                
                # Length accuracy
                if has_length_head and pred_len is not None:
                    if pred_len == gt_len:
                        length_correct += 1
                
                # Collect data for CSV
                csv_data.append({
                    'image_id': f'{name}_{total}',
                    'gt_text': gt,
                    'gt_len': gt_len,
                    'pred_len_from_head': pred_len if has_length_head else -1,
                    'pred_text': pred,
                    'pred_text_len': pred_text_len,
                    'length_match': 1 if (has_length_head and pred_len == gt_len) else 0,
                    'text_match': 1 if pred == gt else 0
                })
        
        accuracy = 100 * correct / total
        mean_ned = 100 * (1 - ned / total)
        mean_conf = 100 * confidence / total
        mean_label_length = label_length / total
        length_acc = 100 * length_correct / total if has_length_head else 0.0
        
        results[name] = Result(name, total, accuracy, mean_ned, mean_conf, mean_label_length, length_acc)

    # Print results
    result_groups = {'Benchmark': test_set}
    log_file = args.checkpoint + '.length_analysis.log.txt'
    
    with open(log_file, 'w') as f:
        for out in [f, sys.stdout]:
            for group, subset in result_groups.items():
                print(f'{group} set:', file=out)
                print_results_table([results[s] for s in subset], out)
                print('\n', file=out)
    
    print(f"\nResults saved to: {log_file}")
    
    # Save CSV
    if args.output_csv is None:
        args.output_csv = args.checkpoint + '.length_analysis.csv'
    
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image_id', 'gt_text', 'gt_len', 'pred_len_from_head',
            'pred_text', 'pred_text_len', 'length_match', 'text_match'
        ])
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"Detailed analysis saved to: {args.output_csv}")
    
    # Print summary statistics
    if has_length_head:
        total_samples = len(csv_data)
        length_matches = sum(1 for d in csv_data if d['length_match'] == 1)
        text_matches = sum(1 for d in csv_data if d['text_match'] == 1)
        
        # Length error analysis
        length_errors = [d for d in csv_data if d['length_match'] == 0 and d['text_match'] == 0]
        length_error_but_text_correct = [d for d in csv_data if d['length_match'] == 0 and d['text_match'] == 1]
        
        print("\n" + "="*60)
        print("Length Prediction Summary")
        print("="*60)
        print(f"Total samples: {total_samples}")
        print(f"Length accuracy: {100 * length_matches / total_samples:.2f}%")
        print(f"Text accuracy: {100 * text_matches / total_samples:.2f}%")
        print(f"Samples with length error but text correct: {len(length_error_but_text_correct)}")
        print(f"Samples with length error and text wrong: {len(length_errors)}")
        print("="*60)


if __name__ == '__main__':
    main()
