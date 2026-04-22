"""
screenshot_augment.py

Screenshot-simulating augmentation (CP1 Snippet 1 - kept verbatim in behavior).

The simulate_screenshot function chains three transforms that mimic the
degradation introduced when someone takes a screenshot of an AI-generated
image and re-shares it: lossy JPEG compression, resolution loss from
downscale/upscale, and a slight Gaussian blur. Parameters are randomized
per-sample so the classifier does not overfit to any one degradation level.

This is the core contribution of the project: by plugging this function into
the PyTorch transform pipeline, every training batch is automatically exposed
to screenshot-like artifacts. The model therefore learns features that
survive compression and resampling rather than relying on clean high-frequency
patterns that disappear after a screenshot.
"""

import io
import random

from PIL import Image, ImageFilter


def simulate_screenshot(img: Image.Image) -> Image.Image:
    """Apply a randomized screenshot-like degradation to a PIL image.

    Pipeline:
      1. JPEG compression at random quality in [30, 70] - simulates lossy encoding.
      2. Downscale to a random factor in [0.5, 0.9], then upscale back to the
         original resolution - simulates screen-capture resolution loss.
      3. Gaussian blur with random radius in [0, 1] - simulates slight softness.

    Args:
      img: PIL Image (any mode).

    Returns:
      A new PIL Image with the degradations applied.
    """
    # JPEG does not support alpha/palette modes; normalize to RGB first.
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 1) JPEG compression (lossy encoding).
    buf = io.BytesIO()
    quality = random.randint(30, 70)
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img = Image.open(buf)

    # 2) Downscale + upscale (resolution loss).
    scale = random.uniform(0.5, 0.9)
    w, h = img.size
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    img = img.resize((w, h), Image.BILINEAR)

    # 3) Slight Gaussian blur.
    img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 1)))

    return img
