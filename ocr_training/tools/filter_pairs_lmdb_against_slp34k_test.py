#!/usr/bin/env python3
"""Create a pair LMDB with groups overlapping SLP34K test images removed."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import lmdb
from tqdm import tqdm


def image_digest(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def lmdb_sample_count(txn) -> int:
    value = txn.get(b"num-samples")
    if value is None:
        raise RuntimeError("missing num-samples")
    return int(value)


def collect_test_hashes(test_root: Path) -> set[str]:
    hashes = set()
    lmdb_dirs = sorted(path.parent for path in test_root.glob("**/data.mdb"))
    if not lmdb_dirs:
        raise RuntimeError(f"no LMDB datasets found under {test_root}")

    for lmdb_dir in lmdb_dirs:
        env = lmdb.open(
            str(lmdb_dir),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        with env.begin() as txn:
            num_samples = lmdb_sample_count(txn)
            for index in tqdm(
                range(1, num_samples + 1),
                desc=f"Hashing test {lmdb_dir.name}",
                leave=False,
            ):
                image_bytes = txn.get(f"image-{index:09d}".encode())
                if image_bytes is None:
                    raise RuntimeError(f"missing image-{index:09d} in {lmdb_dir}")
                hashes.add(image_digest(image_bytes))
        env.close()

    return hashes


def read_pair_samples(input_dir: Path) -> tuple[list[dict], dict[str, list[int]]]:
    samples = []
    group_to_indices = defaultdict(list)

    env = lmdb.open(
        str(input_dir),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    with env.begin() as txn:
        num_samples = lmdb_sample_count(txn)
        for index in tqdm(range(1, num_samples + 1), desc="Reading pair LMDB"):
            image_key = f"image-{index:09d}".encode()
            label_key = f"label-{index:09d}".encode()
            meta_key = f"meta-{index:09d}".encode()
            image_bytes = txn.get(image_key)
            label = txn.get(label_key)
            meta_bytes = txn.get(meta_key)
            if image_bytes is None or label is None or meta_bytes is None:
                raise RuntimeError(f"missing image/label/meta for sample {index}")

            meta = json.loads(meta_bytes.decode("utf-8"))
            pair_id = meta.get("pair_id")
            if not pair_id:
                raise RuntimeError(f"missing pair_id for sample {index}")

            sample = {
                "image": image_bytes,
                "label": label,
                "meta": meta_bytes,
                "pair_id": pair_id,
                "digest": image_digest(image_bytes),
            }
            group_to_indices[pair_id].append(len(samples))
            samples.append(sample)
    env.close()

    return samples, group_to_indices


def write_filtered_lmdb(
    samples: list[dict],
    group_to_indices: dict[str, list[int]],
    test_hashes: set[str],
    output_dir: Path,
    map_size: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_dir), map_size=map_size)

    kept_groups = set()
    removed_groups = set()
    removed_reasons = []
    labels = Counter()
    sample_types = Counter()
    written = 0

    with env.begin(write=True) as txn:
        for pair_id, indices in tqdm(group_to_indices.items(), desc="Filtering groups"):
            hit_indices = [idx for idx in indices if samples[idx]["digest"] in test_hashes]
            if hit_indices:
                removed_groups.add(pair_id)
                removed_reasons.append(
                    {
                        "pair_id": pair_id,
                        "removed_samples": len(indices),
                        "overlap_samples": [
                            json.loads(samples[idx]["meta"].decode("utf-8")).get("source_path")
                            for idx in hit_indices
                        ],
                    }
                )
                continue

            kept_groups.add(pair_id)
            for idx in indices:
                sample = samples[idx]
                written += 1
                label = sample["label"].decode("utf-8")
                meta = json.loads(sample["meta"].decode("utf-8"))
                labels[label] += 1
                sample_types[meta.get("sample_type", "unknown")] += 1
                txn.put(f"image-{written:09d}".encode(), sample["image"])
                txn.put(f"label-{written:09d}".encode(), sample["label"])
                txn.put(f"meta-{written:09d}".encode(), sample["meta"])

        txn.put(b"num-samples", str(written).encode())

    env.close()

    stats = {
        "num_input_samples": len(samples),
        "num_output_samples": written,
        "num_input_groups": len(group_to_indices),
        "num_output_groups": len(kept_groups),
        "num_removed_groups": len(removed_groups),
        "num_removed_samples": len(samples) - written,
        "sample_types": dict(sample_types),
        "num_labels": len(labels),
        "labels": dict(labels.most_common()),
        "removed_groups": removed_reasons,
    }
    (output_dir / "filter_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        default="data/train/SLP34K_pairs_lmdb",
        help="Input pair LMDB directory.",
    )
    parser.add_argument(
        "--test_root",
        default="data/test/SLP34K_lmdb_benchmark",
        help="SLP34K test benchmark root containing one or more LMDB datasets.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/train/SLP34K_pairs_no_test_overlap_lmdb",
        help="Output filtered pair LMDB directory.",
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=1099511627776,
        help="LMDB map size in bytes.",
    )
    args = parser.parse_args()

    test_hashes = collect_test_hashes(Path(args.test_root))
    samples, group_to_indices = read_pair_samples(Path(args.input_dir))
    stats = write_filtered_lmdb(
        samples,
        group_to_indices,
        test_hashes,
        Path(args.output_dir),
        args.map_size,
    )

    print(f"Filtered LMDB created at: {args.output_dir}")
    print(f"Input groups: {stats['num_input_groups']}")
    print(f"Output groups: {stats['num_output_groups']}")
    print(f"Removed groups: {stats['num_removed_groups']}")
    print(f"Input samples: {stats['num_input_samples']}")
    print(f"Output samples: {stats['num_output_samples']}")
    print(f"Removed samples: {stats['num_removed_samples']}")
    print(f"Stats written to: {Path(args.output_dir) / 'filter_stats.json'}")


if __name__ == "__main__":
    main()
