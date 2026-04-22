"""
test_smoke.py

Minimal end-to-end smoke test that runs on CPU in under a minute.

Verifies:
  1. simulate_screenshot produces a valid PIL image.
  2. build_model returns a model with a 2-class head and the expected
     frozen-backbone behavior.
  3. build_train_transform + build_eval_transform produce correctly shaped
     tensors on all four ablation configs.
  4. A single gradient step runs end-to-end on a synthetic 4-image batch.

Run:
    python -m tests.test_smoke
or
    pytest tests/test_smoke.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import torch
from PIL import Image

# Make src/ importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.screenshot_augment import simulate_screenshot
from src.train import build_model, build_train_transform, build_eval_transform
from src.ablation import AUG_CONFIGS


def _synthetic_image(size=(64, 64), color=(128, 64, 200)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def test_simulate_screenshot():
    img = _synthetic_image()
    out = simulate_screenshot(img)
    assert isinstance(out, Image.Image), "simulate_screenshot must return a PIL Image"
    assert out.size == img.size, f"size should be preserved; got {out.size} vs {img.size}"
    # Also verify it copes with an RGBA input (PNG with alpha).
    rgba = Image.new("RGBA", (64, 64), (0, 0, 0, 128))
    out_rgba = simulate_screenshot(rgba)
    assert isinstance(out_rgba, Image.Image)
    print("[ok] simulate_screenshot")


def test_build_model():
    m = build_model(freeze_backbone=True)
    # 2-class head.
    assert m.classifier[1].out_features == 2
    # Backbone frozen, head trainable.
    frozen = [not p.requires_grad for p in m.features.parameters()]
    assert all(frozen), "backbone params should be frozen"
    head_trainable = [p.requires_grad for p in m.classifier.parameters()]
    assert any(head_trainable), "classifier params should be trainable"
    # Forward pass shape.
    x = torch.randn(2, 3, 224, 224)
    y = m(x)
    assert y.shape == (2, 2), f"unexpected output shape {y.shape}"
    print("[ok] build_model")


def test_transforms_all_configs():
    img = _synthetic_image(size=(96, 96))
    for name, cfg in AUG_CONFIGS.items():
        tf = build_train_transform(
            use_screenshot=cfg["screenshot"],
            use_jitter=cfg["jitter"],
            use_crop=cfg["crop"],
        )
        t = tf(img)
        assert t.shape == (3, 224, 224), f"{name}: got {t.shape}"
    # Eval transforms on clean and screenshot.
    for apply in (False, True):
        t = build_eval_transform(apply_screenshot=apply)(img)
        assert t.shape == (3, 224, 224)
    print("[ok] transforms (all configs)")


def test_one_gradient_step():
    m = build_model(freeze_backbone=True)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    xb = torch.randn(4, 3, 224, 224)
    yb = torch.tensor([0, 1, 1, 0])
    m.train()
    logits = m(xb)
    loss = loss_fn(logits, yb)
    loss.backward()
    opt.step()
    assert loss.item() > 0 and torch.isfinite(loss)
    print(f"[ok] one gradient step (loss={loss.item():.4f})")


def main():
    test_simulate_screenshot()
    test_build_model()
    test_transforms_all_configs()
    test_one_gradient_step()
    print("\nAll smoke checks passed.")


if __name__ == "__main__":
    main()
