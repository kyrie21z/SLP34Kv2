#!/usr/bin/env python3
"""
将 benchmark.csv 转换为统一 LMDB 数据库

CSV 格式: id, base64, label, quality, structure, vocabulary_type, resolution_type, structure_type
LMDB 存储:
    - num-samples: 样本总数
    - image-{index:09d}: 图像二进制数据 (JPEG)
    - label-{index:09d}: 文本标签
    - meta-{index:09d}: JSON 格式的元数据
"""

import argparse
import base64
import csv
import json
import os
from pathlib import Path

import lmdb
from tqdm import tqdm


def csv_to_lmdb(csv_path: str, output_dir: str, map_size: int = 1099511627776):
    """
    将 CSV 文件转换为 LMDB 数据库
    
    Args:
        csv_path: CSV 文件路径
        output_dir: LMDB 输出目录
        map_size: LMDB 映射大小 (默认 1TB)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 打开 LMDB
    env = lmdb.open(str(output_path), map_size=map_size)
    
    # 读取 CSV 并写入 LMDB
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    num_samples = len(rows)
    print(f"Total samples to convert: {num_samples}")
    
    with env.begin(write=True) as txn:
        # 写入样本数
        txn.put('num-samples'.encode(), str(num_samples).encode())
        
        # 写入每个样本
        for idx, row in enumerate(tqdm(rows, desc="Converting to LMDB"), start=1):
            # 解码 base64 图像
            img_data = base64.b64decode(row['base64'])
            
            # 图像数据
            img_key = f'image-{idx:09d}'.encode()
            txn.put(img_key, img_data)
            
            # 标签
            label_key = f'label-{idx:09d}'.encode()
            txn.put(label_key, row['label'].encode())
            
            # 元数据 (JSON 格式)
            meta = {
                'id': int(row['id']),
                'quality': row['quality'],  # easy/middle/hard
                'structure': row['structure'],  # single/multi/vertical
                'vocabulary_type': row['vocabulary_type'],  # IV/OOV
                'resolution_type': row['resolution_type'],  # normal/low
                'structure_type': row['structure_type'],  # single_line/multi_lines/vertical
            }
            meta_key = f'meta-{idx:09d}'.encode()
            txn.put(meta_key, json.dumps(meta, ensure_ascii=False).encode())
    
    env.close()
    print(f"\nLMDB database created at: {output_dir}")
    print(f"Total samples: {num_samples}")
    
    # 验证统计
    print("\nDataset statistics:")
    stats = {
        'quality': {},
        'structure': {},
        'vocabulary_type': {},
        'resolution_type': {}
    }
    for row in rows:
        for key in stats:
            val = row[key]
            stats[key][val] = stats[key].get(val, 0) + 1
    
    for key, counts in stats.items():
        print(f"  {key}:")
        for val, count in sorted(counts.items()):
            print(f"    {val}: {count}")


def main():
    parser = argparse.ArgumentParser(description='Convert benchmark CSV to LMDB')
    parser.add_argument('csv_path', help='Path to benchmark CSV file')
    parser.add_argument('--output_dir', default='data/test/SLP34K_lmdb_benchmark/unified_lmdb',
                       help='Output LMDB directory')
    parser.add_argument('--map_size', type=int, default=1099511627776,
                       help='LMDB map size in bytes (default: 1TB)')
    args = parser.parse_args()
    
    csv_to_lmdb(args.csv_path, args.output_dir, args.map_size)


if __name__ == '__main__':
    main()
