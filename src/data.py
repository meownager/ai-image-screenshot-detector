"""
data.py

CIFAKE dataset loading with a held-out validation split and a
matching "screenshotted" validation split for robustness evaluation.

CIFAKE on disk is expected to follow the common layout:

    <root>/
      train/
        REAL/ ... .jpg
        FAKE/ ... .jpg
      test/
        REAL/ ... .jpg
        FAKE/ ... .jpg

Label convention: 0 = Real, 1 = AI-Generated (FAKE).
"""

from __future__ import annotations

import os
import random
from typing import Callable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


CLASS_TO_IDX = {"REAL": 0, "FAKE": 1}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _list_images(class_dir: str) -> List[str]:
    if not os.path.isdir(class_dir):
        return []
    return sorted(
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(IMG_EXTS)
    )


def _gather(root: str, split: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for cls_name, cls_idx in CLASS_TO_IDX.items():
        cls_dir = os.path.join(root, split, cls_name)
        for p in _list_images(cls_dir):
            out.append((p, cls_idx))
    return out


class CIFAKEDataset(Dataset):
    """Simple image-folder dataset with an optional transform."""

    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[Callable] = None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_cifake_splits(
    *,
    root: str,
    train_transform: Callable,
    clean_transform: Callable,
    screenshot_transform: Callable,
    val_frac: float = 0.1,
    subset_frac: float = 1.0,
    seed: int = 1337,
) -> Tuple[CIFAKEDataset, CIFAKEDataset, CIFAKEDataset]:
    """Return (train_ds, val_clean_ds, val_screenshot_ds).

    The validation and screenshot-validation sets share the same underlying
    image paths but use different transforms (clean vs. simulate_screenshot),
    so the robustness gap is measured on the same content.

    subset_frac < 1.0 shrinks the training set proportionally for quick
    iteration (e.g., sandbox smoke tests).
    """
    rng = random.Random(seed)

    train_pool = _gather(root, "train")
    test_pool = _gather(root, "test")
    if not train_pool:
        raise FileNotFoundError(
            f"No training images found under {root}/train/REAL|FAKE. "
            "Run scripts/download_cifake.py first."
        )

    # Split the train pool into train + val (stratified per class).
    by_class = {0: [], 1: []}
    for p, y in train_pool:
        by_class[y].append((p, y))
    for y in by_class:
        rng.shuffle(by_class[y])

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []
    for y, items in by_class.items():
        n_val = max(1, int(len(items) * val_frac))
        val_samples.extend(items[:n_val])
        train_samples.extend(items[n_val:])

    if subset_frac < 1.0:
        rng.shuffle(train_samples)
        train_samples = train_samples[: max(1, int(len(train_samples) * subset_frac))]

    # val_samples is used for both the clean and screenshot val sets.
    train_ds = CIFAKEDataset(train_samples, transform=train_transform)
    val_clean_ds = CIFAKEDataset(val_samples, transform=clean_transform)
    val_ss_ds = CIFAKEDataset(val_samples, transform=screenshot_transform)

    print(
        f"[data] train={len(train_ds)}  val(clean)={len(val_clean_ds)}  "
        f"val(screenshot)={len(val_ss_ds)}  test_pool={len(test_pool)}",
        flush=True,
    )
    return train_ds, val_clean_ds, val_ss_ds
