"""
app.py

Gradio interface for the screenshot-robust AI-image detector (CP2 Snippet 2).

Entry point for Hugging Face Spaces. The Space's app.py should simply
`from src.app import demo; demo.launch()`, or this file can be used directly
as the Space root (see README for both layouts).

Guardrails enforced in predict():
  * File-size limit (10 MB).
  * Dimension bounds (32x32 <= WxH <= 4096x4096).
  * Temperature-scaled probabilities loaded from calibration.json.
  * "Uncertain" band surfaced when calibrated confidence < 0.60.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .train import build_model, IMAGENET_MEAN, IMAGENET_STD


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = os.environ.get(
    "DETECTOR_CHECKPOINT", "runs/ablation/screenshot_jitter_crop/best.pt"
)
DEFAULT_CALIBRATION = os.environ.get(
    "DETECTOR_CALIBRATION", "runs/ablation/screenshot_jitter_crop/calibration.json"
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Real", "AI-Generated"]
UNCERTAINTY_THRESHOLD = 0.60
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIM, MAX_DIM = 32, 4096


def _load_model(checkpoint_path: str) -> torch.nn.Module:
    model = build_model(freeze_backbone=False)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        state_dict = state.get("model_state", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
    else:
        print(f"[warning] checkpoint not found at {checkpoint_path}; "
              "serving an untrained model. This is expected before the first training run.")
    model.to(DEVICE).eval()
    return model


def _load_temperature(calibration_path: str) -> float:
    if os.path.exists(calibration_path):
        with open(calibration_path) as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0


_MODEL = _load_model(DEFAULT_CHECKPOINT)
_TEMPERATURE = _load_temperature(DEFAULT_CALIBRATION)


_eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _validate(img: Optional[Image.Image]) -> Optional[str]:
    if img is None:
        return "Please upload an image."
    w, h = img.size
    if w < MIN_DIM or h < MIN_DIM:
        return f"Image too small ({w}x{h}). Minimum is {MIN_DIM}x{MIN_DIM}."
    if w > MAX_DIM or h > MAX_DIM:
        return f"Image too large ({w}x{h}). Maximum is {MAX_DIM}x{MAX_DIM}."
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    if buf.tell() > MAX_FILE_BYTES:
        return f"Image file exceeds the {MAX_FILE_BYTES // (1024*1024)} MB limit."
    return None


def predict(image: Optional[Image.Image]) -> Tuple[Dict[str, float], str]:
    """Run inference on a PIL image.

    Returns a dict of {class_name: probability} for Gradio's Label component
    and a human-readable explanation string.
    """
    err = _validate(image)
    if err is not None:
        return {"Real": 0.0, "AI-Generated": 0.0}, f"[input error] {err}"

    img = image.convert("RGB")
    tensor = _eval_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = _MODEL(tensor) / _TEMPERATURE
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    verdict = CLASS_NAMES[pred_idx]

    if pred_conf < UNCERTAINTY_THRESHOLD:
        band = "Uncertain"
        explanation = (
            f"The model is not confident ({pred_conf:.0%}). This usually means "
            "the image is far from the training distribution (unfamiliar generator, "
            "unusual compression, or content unlike CIFAKE). Treat the verdict as a "
            "weak signal only."
        )
    else:
        band = "High" if pred_conf >= 0.80 else "Medium"
        explanation = (
            f"Verdict: {verdict} ({band.lower()} confidence, {pred_conf:.0%}). "
            f"The detector was trained on CIFAKE + Stable Diffusion outputs with "
            f"screenshot-style augmentation; it generalizes best to images that "
            f"match those distributions. Out-of-distribution generators "
            f"(Midjourney v6, Flux, Sora) may not be reliably classified."
        )

    return (
        {CLASS_NAMES[0]: float(probs[0]), CLASS_NAMES[1]: float(probs[1])},
        f"**{band} confidence** - {explanation}",
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# AI Image Detector (Screenshot-Robust)

Upload an image - including a screenshot of a social-media post - and the model
will estimate whether it is **AI-generated** or **real**.

**Method:** EfficientNet-B0 fine-tuned on CIFAKE + a Stable Diffusion subset of
GenImage, with a custom `simulate_screenshot` augmentation (JPEG compression,
downscale/upscale, Gaussian blur) plus color jitter and random resized crops.
Outputs are calibrated with temperature scaling; low-confidence predictions
are surfaced as *Uncertain* rather than forced into a binary label.

**Limitations:** trained only against Stable Diffusion v1.4-class generators;
may behave unpredictably on Midjourney v6, Flux, Sora, and other out-of-distribution
sources. This is a research prototype for coursework (ECE 57000 at Purdue),
not a production content-authenticity tool.
"""


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image or screenshot"),
    outputs=[
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Markdown(label="Explanation"),
    ],
    title="AI Image Detector",
    description=DESCRIPTION,
    article=(
        "Source: https://github.com/meownager/ai-image-screenshot-detector  "
        "Course: ECE 57000, Spring 2026, Track 2 Product Prototype."
    ),
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
