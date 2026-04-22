"""
download_cifake.py

Downloads the CIFAKE dataset from Hugging Face and lays it out in the
CIFAKE-style directory layout expected by src/data.py:

    data/cifake/train/REAL/*.jpg
    data/cifake/train/FAKE/*.jpg
    data/cifake/test/REAL/*.jpg
    data/cifake/test/FAKE/*.jpg

This script is idempotent: existing files are skipped.

Usage:
    python scripts/download_cifake.py --out ./data/cifake [--max-per-class N]
"""

from __future__ import annotations

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


LABEL_TO_DIRNAME = {0: "REAL", 1: "FAKE"}


def _export_split(ds, split: str, out_root: str, max_per_class: int | None):
    counts = {0: 0, 1: 0}
    for i, row in enumerate(tqdm(ds, desc=f"{split}", unit="img")):
        label = int(row["label"])
        if max_per_class is not None and counts[label] >= max_per_class:
            continue
        sub = os.path.join(out_root, split, LABEL_TO_DIRNAME[label])
        os.makedirs(sub, exist_ok=True)
        out_path = os.path.join(sub, f"{split}_{i:06d}.jpg")
        if not os.path.exists(out_path):
            row["image"].convert("RGB").save(out_path, "JPEG", quality=92)
        counts[label] += 1
    print(f"  {split}: REAL={counts[0]}  FAKE={counts[1]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/cifake")
    ap.add_argument("--dataset", default="dragonintelligence/CIFAKE-image-dataset",
                    help="HF dataset id. Swap if the canonical CIFAKE mirror changes.")
    ap.add_argument("--max-per-class", type=int, default=None,
                    help="Cap per-class samples per split (for smoke tests).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"[download] loading {args.dataset} from Hugging Face...")
    ds = load_dataset(args.dataset)
    print(f"[download] splits available: {list(ds.keys())}")

    for split_name in ds.keys():
        target = "train" if split_name.lower().startswith("train") else "test"
        _export_split(ds[split_name], target, args.out, args.max_per_class)

    print(f"[download] done. Root={args.out}")


if __name__ == "__main__":
    main()
