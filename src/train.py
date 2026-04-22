"""
train.py

Training entry point (CP1 Snippet 2 preserved verbatim at the top, with a
full training loop wrapped around it).

Usage (single run):
    python -m src.train \
        --data-root ./data/cifake \
        --out-dir ./runs/efficientnet_screenshot_jitter_crop \
        --aug screenshot_jitter_crop \
        --epochs 5 --batch-size 64 --lr 3e-4

The four supported --aug values are documented in src.ablation.AUG_CONFIGS.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from .screenshot_augment import simulate_screenshot


# ============================================================================
# CP1 Snippet 2 (verbatim in behavior): model + training transform
# ============================================================================

def build_model(freeze_backbone: bool = True) -> nn.Module:
    """EfficientNet-B0 pretrained on ImageNet, with a 2-class head."""
    # torchvision >=0.13 deprecates `pretrained=True` in favor of `weights=...`;
    # use the modern API but keep the snippet's semantics.
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:  # older torchvision
        model = models.efficientnet_b0(pretrained=True)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False  # fine-tune classifier only

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)  # real vs AI
    return model


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(use_screenshot: bool, use_jitter: bool, use_crop: bool):
    """Assemble the training transform used by a given ablation config."""
    ops = []
    if use_crop:
        # RandomResizedCrop subsumes Resize(224) for crop configs.
        ops.append(transforms.RandomResizedCrop(224, scale=(0.7, 1.0)))
    else:
        ops.append(transforms.Resize((224, 224)))

    if use_jitter:
        ops.append(
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            )
        )

    if use_screenshot:
        # CP1 Snippet 2's train_transform wires simulate_screenshot in via Lambda.
        ops.append(transforms.Lambda(simulate_screenshot))

    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


def build_eval_transform(apply_screenshot: bool):
    """Deterministic eval transform (no color jitter, no random crop).

    When apply_screenshot=True the simulate_screenshot function still runs
    with its random parameters, which is intentional: each screenshot sample
    is drawn once at evaluation time from a realistic degradation distribution.
    """
    ops = [transforms.Resize((224, 224))]
    if apply_screenshot:
        ops.append(transforms.Lambda(simulate_screenshot))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(ops)


# ============================================================================
# Training loop
# ============================================================================

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_acc_screenshot: float
    seconds: float


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    loss_sum = 0.0
    correct = 0
    total = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def train_one_config(
    *,
    train_ds,
    val_ds_clean,
    val_ds_screenshot,
    out_dir: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 2,
    freeze_backbone: bool = True,
    device: str | None = None,
) -> Dict:
    """Run training for a single ablation configuration.

    Saves the best (by screenshot-val-acc) checkpoint to <out_dir>/best.pt
    and writes a per-epoch metrics log to <out_dir>/metrics.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(freeze_backbone=freeze_backbone).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_clean_loader = DataLoader(
        val_ds_clean, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    val_ss_loader = DataLoader(
        val_ds_screenshot, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    history: list[EpochMetrics] = []
    best_ss_acc = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = _run_epoch(model, val_clean_loader, criterion, optimizer, device, train=False)
        _, ss_acc = _run_epoch(model, val_ss_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        em = EpochMetrics(
            epoch=epoch,
            train_loss=tr_loss, train_acc=tr_acc,
            val_loss=va_loss, val_acc=va_acc,
            val_acc_screenshot=ss_acc,
            seconds=time.time() - t0,
        )
        history.append(em)
        print(
            f"[epoch {epoch}/{epochs}] "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_acc_clean={va_acc:.4f} val_acc_ss={ss_acc:.4f} "
            f"({em.seconds:.1f}s)",
            flush=True,
        )

        if ss_acc > best_ss_acc:
            best_ss_acc = ss_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc_clean": va_acc,
                    "val_acc_screenshot": ss_acc,
                },
                os.path.join(out_dir, "best.pt"),
            )

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump([asdict(em) for em in history], f, indent=2)

    return {
        "best_screenshot_val_acc": best_ss_acc,
        "final_clean_val_acc": history[-1].val_acc,
        "final_screenshot_val_acc": history[-1].val_acc_screenshot,
        "epochs": epochs,
        "out_dir": out_dir,
    }


# ============================================================================
# CLI
# ============================================================================

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True,
                   help="Path to CIFAKE-style dataset (train/REAL, train/FAKE, test/REAL, test/FAKE).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--aug", default="screenshot_jitter_crop",
                   choices=["none", "screenshot", "screenshot_jitter", "screenshot_jitter_crop"])
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--subset-frac", type=float, default=1.0,
                   help="Fraction of the training set to use (for quick iteration).")
    return p.parse_args()


def main():
    args = _parse_args()
    # Deferred import so `--help` doesn't touch torch datasets.
    from .data import build_cifake_splits

    aug_flags = {
        "none":                       (False, False, False),
        "screenshot":                 (True,  False, False),
        "screenshot_jitter":          (True,  True,  False),
        "screenshot_jitter_crop":     (True,  True,  True),
    }[args.aug]
    train_tf = build_train_transform(*aug_flags)
    clean_eval_tf = build_eval_transform(apply_screenshot=False)
    ss_eval_tf = build_eval_transform(apply_screenshot=True)

    train_ds, val_clean, val_ss = build_cifake_splits(
        root=args.data_root,
        train_transform=train_tf,
        clean_transform=clean_eval_tf,
        screenshot_transform=ss_eval_tf,
        subset_frac=args.subset_frac,
    )

    summary = train_one_config(
        train_ds=train_ds,
        val_ds_clean=val_clean,
        val_ds_screenshot=val_ss,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
