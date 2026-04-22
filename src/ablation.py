"""
ablation.py

Four-configuration ablation study (CP2 Snippet 1 - kept verbatim).

AUG_CONFIGS is the single source of truth for which training variants are
run and which rows appear in the paper's ablation table. Each config is
trained independently; results.json aggregates the final numbers.

Usage:
    python -m src.ablation \
        --data-root ./data/cifake \
        --out-root ./runs/ablation \
        --epochs 5 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

from .train import train_one_config, build_train_transform, build_eval_transform


# ---------------------------------------------------------------------------
# CP2 Snippet 1: the ablation config table.
# ---------------------------------------------------------------------------
AUG_CONFIGS: Dict[str, Dict[str, bool]] = {
    "none":                    {"screenshot": False, "jitter": False, "crop": False},
    "screenshot":              {"screenshot": True,  "jitter": False, "crop": False},
    "screenshot_jitter":       {"screenshot": True,  "jitter": True,  "crop": False},
    "screenshot_jitter_crop":  {"screenshot": True,  "jitter": True,  "crop": True},
}


def run_ablation(
    *,
    data_root: str,
    out_root: str,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 3e-4,
    num_workers: int = 2,
    subset_frac: float = 1.0,
    configs=None,
) -> Dict:
    """Train every config in AUG_CONFIGS and write an aggregated results.json."""
    from .data import build_cifake_splits  # deferred import

    os.makedirs(out_root, exist_ok=True)
    results: Dict[str, Dict] = {}
    configs = configs or list(AUG_CONFIGS.keys())

    # Eval transforms are shared across configs (only train-time augmentation varies).
    clean_eval_tf = build_eval_transform(apply_screenshot=False)
    ss_eval_tf = build_eval_transform(apply_screenshot=True)

    for name in configs:
        cfg = AUG_CONFIGS[name]
        train_tf = build_train_transform(
            use_screenshot=cfg["screenshot"],
            use_jitter=cfg["jitter"],
            use_crop=cfg["crop"],
        )
        train_ds, val_clean, val_ss = build_cifake_splits(
            root=data_root,
            train_transform=train_tf,
            clean_transform=clean_eval_tf,
            screenshot_transform=ss_eval_tf,
            subset_frac=subset_frac,
        )
        print(f"\n=== Running config: {name} ===", flush=True)
        summary = train_one_config(
            train_ds=train_ds,
            val_ds_clean=val_clean,
            val_ds_screenshot=val_ss,
            out_dir=os.path.join(out_root, name),
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            num_workers=num_workers,
        )
        summary["config"] = name
        summary["clean_minus_screenshot_gap"] = (
            summary["final_clean_val_acc"] - summary["final_screenshot_val_acc"]
        )
        results[name] = summary

        # Persist after every config so partial runs are not lost.
        with open(os.path.join(out_root, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Pretty-printed final table for the paper.
    print("\n=== Ablation Results ===")
    print(f"{'Config':32s} {'Clean':>8s} {'SS':>8s} {'Gap':>8s}")
    for name, r in results.items():
        print(
            f"{name:32s} {r['final_clean_val_acc']:>8.4f} "
            f"{r['final_screenshot_val_acc']:>8.4f} "
            f"{r['clean_minus_screenshot_gap']:>8.4f}"
        )
    return results


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-root", default="./runs/ablation")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--subset-frac", type=float, default=1.0)
    p.add_argument("--configs", nargs="*", default=None,
                   help="Subset of AUG_CONFIGS to run. Defaults to all four.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_ablation(
        data_root=args.data_root,
        out_root=args.out_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        subset_frac=args.subset_frac,
        configs=args.configs,
    )
