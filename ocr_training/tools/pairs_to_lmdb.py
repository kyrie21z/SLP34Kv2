#!/usr/bin/env python3
"""Convert ship-plate pair images into the LMDB format used by strhub."""

import argparse
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


def parse_pair_filename(path: Path) -> dict[str, str]:
    parts = path.name.split("&&&&")
    if len(parts) != 3:
        raise ValueError("expected '<anchor>&&&&<group_id>&&&&<sample>.jpg' filename")

    anchor_name, group_id, sample_name = parts
    sample_name = parts[2]
    stem = sample_name[:-4] if sample_name.lower().endswith(".jpg") else path.stem
    is_pair = stem.endswith("-pair")
    if is_pair:
        stem = stem[:-5]
    if "-" not in stem:
        raise ValueError("missing label separator '-'")

    label = stem.split("-", 1)[1]
    if not label:
        raise ValueError("empty label")
    return {
        "anchor_name": anchor_name,
        "group_id": group_id,
        "sample_name": sample_name,
        "label": label,
        "sample_type": "pair" if is_pair else "single",
    }


def collect_samples(input_dir: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for path in sorted(input_dir.rglob("*.jpg")):
        sample = parse_pair_filename(path)
        target = path.relative_to(input_dir).parts[0]
        pair_id = f"{target}/{sample['group_id']}"
        sample["path"] = str(path)
        sample["relative_path"] = str(path.relative_to(input_dir))
        sample["target"] = target
        sample["pair_id"] = pair_id
        groups.setdefault(pair_id, []).append(sample)
    return groups


def iter_samples(input_dir: Path, mode: str):
    for samples in collect_samples(input_dir).values():
        singles = [sample for sample in samples if sample["sample_type"] == "single"]
        pairs = [sample for sample in samples if sample["sample_type"] == "pair"]
        single_by_label = {sample["label"]: sample for sample in singles}
        pair_by_label = {sample["label"]: sample for sample in pairs}

        for sample in sorted(samples, key=lambda item: item["relative_path"]):
            if mode != "all" and sample["sample_type"] != mode:
                continue
            if sample["sample_type"] == "pair":
                counterpart = single_by_label.get(sample["label"])
            else:
                counterpart = pair_by_label.get(sample["label"])
            sample = dict(sample)
            sample["counterpart_path"] = counterpart["relative_path"] if counterpart else None
            sample["has_counterpart"] = counterpart is not None
            yield sample


def validate_image(image_bytes: bytes, path: Path) -> tuple[int, int]:
    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.verify()
            return img.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"invalid image: {path}") from exc


def build_lmdb(input_dir: str, output_dir: str, mode: str, map_size: int) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    samples = list(iter_samples(input_path, mode))
    if not samples:
        raise RuntimeError(f"no samples found under {input_path} for mode={mode}")

    env = lmdb.open(str(output_path), map_size=map_size)
    labels = Counter()
    skipped = []
    written = 0

    with env.begin(write=True) as txn:
        for sample in tqdm(samples, desc="Writing LMDB"):
            path = Path(sample["path"])
            try:
                image_bytes = path.read_bytes()
                width, height = validate_image(image_bytes, path)
            except ValueError as exc:
                skipped.append({"path": str(path), "reason": str(exc)})
                continue

            written += 1
            labels[sample["label"]] += 1
            txn.put(f"image-{written:09d}".encode(), image_bytes)
            txn.put(f"label-{written:09d}".encode(), sample["label"].encode("utf-8"))
            txn.put(
                f"meta-{written:09d}".encode(),
                json.dumps(
                    {
                        "source_path": sample["relative_path"],
                        "pair_id": sample["pair_id"],
                        "target": sample["target"],
                        "group_id": sample["group_id"],
                        "anchor_name": sample["anchor_name"],
                        "sample_name": sample["sample_name"],
                        "sample_type": sample["sample_type"],
                        "pair_path": (
                            sample["relative_path"]
                            if sample["sample_type"] == "pair"
                            else sample["counterpart_path"]
                        ),
                        "single_path": (
                            sample["relative_path"]
                            if sample["sample_type"] == "single"
                            else sample["counterpart_path"]
                        ),
                        "has_counterpart": sample["has_counterpart"],
                        "width": width,
                        "height": height,
                    },
                    ensure_ascii=False,
                ).encode("utf-8"),
            )

        txn.put("num-samples".encode(), str(written).encode())

    env.close()

    stats_path = output_path / "label_stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "input_dir": str(input_path),
                "mode": mode,
                "num_samples": written,
                "num_labels": len(labels),
                "labels": dict(labels.most_common()),
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"LMDB created at: {output_path}")
    print(f"Samples written: {written}")
    print(f"Unique labels: {len(labels)}")
    if skipped:
        print(f"Skipped invalid samples: {len(skipped)}")
    print(f"Stats written to: {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir",
        default="data/pairs",
        help="Directory containing pair image files.",
    )
    parser.add_argument(
        "--output_dir",
        default="data/train/SLP34K_pairs_lmdb",
        help="Output LMDB directory.",
    )
    parser.add_argument(
        "--mode",
        choices=("pair", "single", "all"),
        default="pair",
        help="Which files to include. Default keeps only '*-pair.jpg' images.",
    )
    parser.add_argument(
        "--map_size",
        type=int,
        default=1099511627776,
        help="LMDB map size in bytes.",
    )
    args = parser.parse_args()

    build_lmdb(args.input_dir, args.output_dir, args.mode, args.map_size)


if __name__ == "__main__":
    main()
