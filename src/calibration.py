"""
calibration.py

Temperature scaling for well-calibrated probabilities.

After training, fit a single scalar T on the validation set so that
softmax(logits / T) minimizes NLL. Small, well-understood, and the
industry-standard baseline calibrator from Guo et al. 2017.

Usage:
    python -m src.calibration \
        --data-root ./data/cifake \
        --checkpoint ./runs/ablation/screenshot_jitter_crop/best.pt \
        --out ./runs/ablation/screenshot_jitter_crop/calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import CIFAKEDataset, _gather
from .train import build_model, build_eval_transform


@torch.no_grad()
def _collect_logits(model: torch.nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_logits, all_labels = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        all_logits.append(model(xb).cpu())
        all_labels.append(yb)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 100) -> float:
    """Return T minimizing NLL of softmax(logits / T) against labels."""
    logT = torch.zeros(1, requires_grad=True)  # optimize in log-space to stay positive
    optimizer = torch.optim.LBFGS([logT], lr=0.1, max_iter=max_iter)
    criterion = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        T = torch.exp(logT)
        loss = criterion(logits / T, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(logT).item())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        mask = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if mask.any():
            ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def calibrate(*, data_root: str, checkpoint: str, batch_size: int, num_workers: int) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(freeze_backbone=False)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model_state", state) if isinstance(state, dict) else state)
    model.to(device).eval()

    # Calibrate on the clean validation split (held-out from training).
    val_samples = _gather(data_root, "train")  # we re-derive val from train pool
    # NOTE: For robustness, in practice pass a dedicated val folder. Here we
    # recreate the val split deterministically using the same seed as data.py.
    import random
    rng = random.Random(1337)
    by_class = {0: [], 1: []}
    for p, y in val_samples:
        by_class[y].append((p, y))
    val_set = []
    for y in by_class:
        rng.shuffle(by_class[y])
        n_val = max(1, int(len(by_class[y]) * 0.1))
        val_set.extend(by_class[y][:n_val])

    tf = build_eval_transform(apply_screenshot=False)
    loader = DataLoader(
        CIFAKEDataset(val_set, transform=tf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    logits, labels = _collect_logits(model, loader, device)

    probs_pre = F.softmax(logits, dim=1).numpy()
    ece_pre = expected_calibration_error(probs_pre, labels.numpy())

    T = fit_temperature(logits, labels)
    probs_post = F.softmax(logits / T, dim=1).numpy()
    ece_post = expected_calibration_error(probs_post, labels.numpy())

    return {
        "temperature": T,
        "ece_before": ece_pre,
        "ece_after": ece_post,
        "n_val": int(labels.numel()),
    }


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = calibrate(
        data_root=args.data_root, checkpoint=args.checkpoint,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
