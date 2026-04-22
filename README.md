# AI Image Detection from Screenshots

**Course:** ECE 57000, Section 002, Spring 2026 — Track 2 (Product Prototype)
**Author:** Syeda Maliha Monowara (PUID 0038305425) — smonowar@purdue.edu

A screenshot-robust detector that classifies an uploaded image as **Real** or **AI-Generated**. The method is EfficientNet-B0 fine-tuned on CIFAKE plus a Stable-Diffusion subset of GenImage, with a `simulate_screenshot` augmentation (JPEG compression + resolution loss + slight blur) baked into the training pipeline so the model learns features that survive the compression and resampling introduced by re-shared social-media screenshots.

## Repository layout

```
src/
  screenshot_augment.py   # CP1 Snippet 1: simulate_screenshot(img)
  train.py                # CP1 Snippet 2: build_model + transform + training loop
  ablation.py             # CP2 Snippet 1: AUG_CONFIGS + run_ablation
  app.py                  # CP2 Snippet 2: Gradio interface
  data.py                 # CIFAKE dataset + stratified train/val split
  eval.py                 # Clean-vs-screenshot test eval; real-world eval
  calibration.py          # Temperature scaling for well-calibrated probs
scripts/
  download_cifake.py      # Pulls CIFAKE from Hugging Face into CIFAKE-style folders
notebooks/
  train_colab.ipynb       # One-click training on Google Colab T4
paper/
  main.tex                # ICLR-26 formatted write-up (filled in Phase D)
LLM_ACKNOWLEDGEMENT.md    # Rubric-required LLM acknowledgement
requirements.txt
```

## Reproduce the results

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download CIFAKE

```bash
python scripts/download_cifake.py --out ./data/cifake
# Smoke-test cap: add --max-per-class 500 for a fast dev loop.
```

### 3. Run the ablation study

```bash
python -m src.ablation \
    --data-root ./data/cifake \
    --out-root ./runs/ablation \
    --epochs 5 --batch-size 64 --lr 3e-4
```

The four configurations are defined in `src.ablation.AUG_CONFIGS`:

| Config key                 | simulate_screenshot | ColorJitter | RandomResizedCrop |
| -------------------------- | :-: | :-: | :-: |
| `none`                     |  —  |  —  |  —  |
| `screenshot`               |  ✓  |  —  |  —  |
| `screenshot_jitter`        |  ✓  |  ✓  |  —  |
| `screenshot_jitter_crop`   |  ✓  |  ✓  |  ✓  |

### 4. Evaluate

```bash
python -m src.eval cifake \
    --data-root ./data/cifake \
    --checkpoint ./runs/ablation/screenshot_jitter_crop/best.pt \
    --out ./runs/ablation/screenshot_jitter_crop/test_report.json
```

For the Instagram / real-world evaluation, put images in `data/real_world/REAL/` and `data/real_world/FAKE/` and run `python -m src.eval realworld ...`.

### 5. Calibrate

```bash
python -m src.calibration \
    --data-root ./data/cifake \
    --checkpoint ./runs/ablation/screenshot_jitter_crop/best.pt \
    --out ./runs/ablation/screenshot_jitter_crop/calibration.json
```

### 6. Serve the Gradio app locally

```bash
python -m src.app
```

## Deployment (Hugging Face Spaces)

The Gradio app is deployed as a Space backed by a sibling model repository. The training notebook's final cell uploads `best.pt`, `calibration.json`, and `metrics.json` to `meownager/ai-image-screenshot-detector` on Hugging Face; the Space pulls them at startup.

## One-click Colab training

Open `notebooks/train_colab.ipynb` in Google Colab, switch runtime to a **T4 GPU**, and run all cells. The notebook clones this repo, downloads CIFAKE, runs the full ablation, calibrates the best config, and (optionally) pushes the checkpoint to Hugging Face.

## LLM acknowledgement

Per the ECE 57000 rubric, LLM assistance is disclosed in `LLM_ACKNOWLEDGEMENT.md` and in the paper.

## License

MIT.
