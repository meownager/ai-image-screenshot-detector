"""
eval.py

Evaluation utilities.

Two CLIs in one file:

  (1) Held-out CIFAKE test evaluation, producing the clean-vs-screenshot
      accuracy gap and per-class precision / recall:

      python -m src.eval cifake \
          --data-root ./data/cifake \
          --checkpoint ./runs/ablation/screenshot_jitter/best.pt \
          --out ./runs/ablation/screenshot_jitter/test_report.json

  (2) Real-world (Instagram) evaluation on a folder of user-provided images
      organized as real_world/REAL and real_world/FAKE:

      python -m src.eval realworld \
          --data-root ./data/real_world \
          --checkpoint ./runs/ablation/screenshot_jitter/best.pt \
          --out ./runs/ablation/screenshot_jitter/realworld_report.json
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


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Accuracy, per-class precision/recall, and a 2x2 confusion matrix."""
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    precisions, recalls = [], []
    for cls in (0, 1):
        tp = cm[cls, cls]
        pred_pos = cm[:, cls].sum()
        actual_pos = cm[cls, :].sum()
        precisions.append(float(tp / pred_pos) if pred_pos else 0.0)
        recalls.append(float(tp / actual_pos) if actual_pos else 0.0)
    return {
        "accuracy": acc,
        "precision_real": precisions[0],
        "precision_ai": precisions[1],
        "recall_real": recalls[0],
        "recall_ai": recalls[1],
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }


@torch.no_grad()
def _predict(model: torch.nn.Module, loader: DataLoader, device: str):
    model.eval()
    ys, ps, probs = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs_b = F.softmax(logits, dim=1).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(probs_b.argmax(axis=1))
        probs.append(probs_b)
    return (
        np.concatenate(ys) if ys else np.array([]),
        np.concatenate(ps) if ps else np.array([]),
        np.concatenate(probs) if probs else np.zeros((0, 2)),
    )


def _load_model(checkpoint: str, device: str) -> torch.nn.Module:
    model = build_model(freeze_backbone=False)
    state = torch.load(checkpoint, map_location=device)
    state_dict = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def eval_cifake(data_root: str, checkpoint: str, batch_size: int, num_workers: int) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(checkpoint, device)

    test_samples = _gather(data_root, "test")
    if not test_samples:
        raise FileNotFoundError(f"No test images under {data_root}/test/REAL|FAKE")

    clean_tf = build_eval_transform(apply_screenshot=False)
    ss_tf = build_eval_transform(apply_screenshot=True)

    clean_loader = DataLoader(
        CIFAKEDataset(test_samples, transform=clean_tf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    ss_loader = DataLoader(
        CIFAKEDataset(test_samples, transform=ss_tf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    y_true_c, y_pred_c, _ = _predict(model, clean_loader, device)
    y_true_s, y_pred_s, _ = _predict(model, ss_loader, device)

    report = {
        "clean":      _metrics(y_true_c, y_pred_c),
        "screenshot": _metrics(y_true_s, y_pred_s),
    }
    report["clean_minus_screenshot_gap"] = (
        report["clean"]["accuracy"] - report["screenshot"]["accuracy"]
    )
    return report


def eval_realworld(data_root: str, checkpoint: str, batch_size: int, num_workers: int) -> Dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(checkpoint, device)

    # Expect <root>/REAL and <root>/FAKE (no train/test nesting for real-world set).
    samples = []
    for cls_name, cls_idx in (("REAL", 0), ("FAKE", 1)):
        cls_dir = os.path.join(data_root, cls_name)
        if os.path.isdir(cls_dir):
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    samples.append((os.path.join(cls_dir, f), cls_idx))
    if not samples:
        raise FileNotFoundError(
            f"No images found under {data_root}/REAL or {data_root}/FAKE. "
            "Place your Instagram screenshots in these two folders."
        )

    tf = build_eval_transform(apply_screenshot=False)
    loader = DataLoader(
        CIFAKEDataset(samples, transform=tf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    y_true, y_pred, probs = _predict(model, loader, device)
    report = _metrics(y_true, y_pred)
    report["mean_confidence"] = float(probs.max(axis=1).mean()) if probs.size else 0.0
    return report


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["cifake", "realworld"])
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    fn = eval_cifake if args.mode == "cifake" else eval_realworld
    report = fn(args.data_root, args.checkpoint, args.batch_size, args.num_workers)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
