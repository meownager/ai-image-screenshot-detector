"""
app.py

Gradio interface for the screenshot-robust AI-image detector.
Deployed on Hugging Face Spaces as the course product prototype for
ECE 57000 (Spring 2026, Purdue).

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
from typing import Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------------------------
# Model definition (inlined so the Space is self-contained)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model() -> nn.Module:
    """EfficientNet-B0 pretrained on ImageNet, with a 2-class head."""
    try:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
    except AttributeError:
        model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    return model


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get("DETECTOR_CHECKPOINT", "best.pt")
CALIBRATION_PATH = os.environ.get("DETECTOR_CALIBRATION", "calibration.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = ["Real", "AI-Generated"]
UNCERTAINTY_THRESHOLD = 0.60
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_DIM, MAX_DIM = 32, 4096


def _load_model(checkpoint_path: str) -> nn.Module:
    model = build_model()
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=DEVICE)
        state_dict = state.get("model_state", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict)
        print(f"[info] loaded checkpoint from {checkpoint_path}")
    else:
        print(f"[warning] checkpoint not found at {checkpoint_path}; serving untrained model.")
    model.to(DEVICE).eval()
    return model


def _load_temperature(calibration_path: str) -> float:
    if os.path.exists(calibration_path):
        with open(calibration_path) as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0


_MODEL = _load_model(CHECKPOINT_PATH)
_TEMPERATURE = _load_temperature(CALIBRATION_PATH)

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
        return f"Image exceeds the {MAX_FILE_BYTES // (1024*1024)} MB limit."
    return None


def predict(image: Optional[Image.Image]) -> Tuple[Dict[str, float], str]:
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
            "The detector was trained on CIFAKE + Stable Diffusion outputs with "
            "screenshot-style augmentation; it generalizes best to images that "
            "match those distributions. Out-of-distribution generators "
            "(Midjourney v6, Flux, Sora) may not be reliably classified."
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

Upload an image (including a screenshot of a social-media post) and the model
will estimate whether it is **AI-generated** or **real**.

**Method:** EfficientNet-B0 fine-tuned on CIFAKE + a Stable Diffusion subset
of GenImage, with a custom `simulate_screenshot` augmentation (JPEG
compression, downscale/upscale, Gaussian blur) plus color jitter. Outputs
are calibrated with temperature scaling; low-confidence predictions are
surfaced as *Uncertain* rather than forced into a binary label.

**Limitations:** trained only against Stable Diffusion v1.4-class generators;
may behave unpredictably on Midjourney v6, Flux, Sora, and other
out-of-distribution sources. Research prototype for coursework (ECE 57000
at Purdue); not a production content-authenticity tool.
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
