# AI Image Detection from Screenshots

**Course:** ECE 57000, Section 002, Spring 2026 — Track 2 (Product Prototype)
**Author:** Syeda Maliha Monowara (PUID 0038305425) — smonowar@purdue.edu

A screenshot-robust detector that classifies an uploaded image as **Real** or **AI-Generated**. The method is EfficientNet-B0 fine-tuned on CIFAKE plus a Stable-Diffusion subset of GenImage, with a `simulate_screenshot` augmentation (JPEG compression + resolution loss + slight blur) baked into the training pipeline so the model learns features that survive the compression and resampling introduced by re-shared social-media screenshots.

## Quick links

- **Paper (PDF):** see `paper/main.tex` (compiles on [Overleaf](https://www.overleaf.com) with the ICLR 2026 template)
- **Live demo (Hugging Face Space):** https://huggingface.co/spaces/meownager/ai-image-screenshot-detector
- **Model weights (Hugging Face Hub):** https://huggingface.co/meownager/ai-image-screenshot-detector
- **One-click Colab training notebook:** https://colab.research.google.com/github/meownager/ai-image-screenshot-detector/blob/main/notebooks/train_colab.ipynb

## Repository layout

```
src/
  screenshot_augment.py   # CP1 Snippet 1: simulate_screenshot(img)
  train.py                # CP1 Snippet 2: build_model + transform + training loop
  ablation.py             # CP2 Snippet 1: AUG_CONFIGS + run_ablation
  app.py                  # CP2 Snippet 2: Gradio interface (local dev)
  data.py                 # CIFAKE dataset + stratified train/val split
  eval.py                 # Clean-vs-screenshot test eval; real-world eval
  calibration.py          # Temperature scaling for well-calibrated probs
scripts/
  download_cifake.py      # Pulls CIFAKE from Hugging Face into CIFAKE-style folders
notebooks/
  train_colab.ipynb       # One-click training on Google Colab T4
paper/
  main.tex                # ICLR-26 formatted write-up
  references.bib          # Bibliography
  figures/                # Instagram screenshot figures used in the paper
hf-space/
  app.py                  # Self-contained Gradio app deployed to Hugging Face Spaces
  requirements.txt        # Minimal Space dependencies
data/
  ig-data/                # 49 manually collected Instagram screenshots (15 real + 34 fake)
  ig-data.zip             # Zipped copy of the above
tests/
  test_smoke.py           # Import + shape smoke test
LLM_ACKNOWLEDGEMENT.md    # Rubric-required LLM acknowledgement
requirements.txt
```

## Dependencies

Python 3.10+. Install with:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The core stack is `torch>=2.1`, `torchvision>=0.16`, `pillow`, `numpy`, `gradio>=4`, `datasets`, and `huggingface_hub`.

## Reproduce the results

### 1. Download CIFAKE

```bash
python scripts/download_cifake.py --out ./data/cifake
# Smoke-test cap: add --max-per-class 500 for a fast dev loop.
```

### 2. Run the ablation study (three epochs, batch 64)

```bash
python -m src.ablation \
    --data-root ./data/cifake \
    --out-root ./runs/ablation \
    --epochs 3 --batch-size 64 --lr 3e-4
```

The four configurations are defined in `src.ablation.AUG_CONFIGS`:

| Config key                 | simulate_screenshot | ColorJitter | RandomResizedCrop |
| -------------------------- | :-: | :-: | :-: |
| `none`                     |  —  |  —  |  —  |
| `screenshot`               |  ✓  |  —  |  —  |
| `screenshot_jitter`        |  ✓  |  ✓  |  —  |
| `screenshot_jitter_crop`   |  ✓  |  ✓  |  ✓  |

The **deployed configuration is `screenshot_jitter`** (best screenshot accuracy; see Table 2 in the paper).

### 3. Evaluate the deployed checkpoint

```bash
python -m src.eval cifake \
    --data-root ./data/cifake \
    --checkpoint ./runs/ablation/screenshot_jitter/best.pt \
    --out ./runs/ablation/screenshot_jitter/test_report.json
```

For the Instagram / real-world evaluation, point at `data/ig-data/real/` and `data/ig-data/fake/`:

```bash
python -m src.eval realworld \
    --real-dir ./data/ig-data/real \
    --fake-dir ./data/ig-data/fake \
    --checkpoint ./runs/ablation/screenshot_jitter/best.pt \
    --out ./runs/realworld_report.json
```

### 4. Calibrate

```bash
python -m src.calibration \
    --data-root ./data/cifake \
    --checkpoint ./runs/ablation/screenshot_jitter/best.pt \
    --out ./runs/ablation/screenshot_jitter/calibration.json
```

This fits a single temperature `T` with L-BFGS and writes `{"temperature": T}` to JSON. The deployed model uses `T = 0.563`, which drops Expected Calibration Error from 10.0% to 1.9%.

### 5. Serve the Gradio app locally

```bash
python -m src.app
```

Or run the Space-flavored standalone:

```bash
cd hf-space && python app.py
```

## One-click Colab training

Open `notebooks/train_colab.ipynb` in Google Colab, switch runtime to a **T4 GPU**, and run all cells. The notebook clones this repo, downloads CIFAKE, runs the full four-way ablation (three epochs, batch 64, AdamW lr 3e-4, cosine schedule, 10,000 images per class), calibrates the deployed `screenshot_jitter` config, and optionally pushes `best.pt` + `calibration.json` to Hugging Face.

## Hugging Face deployment

The Gradio app is deployed as a Space backed by a sibling model repository:

- **Space:** `meownager/ai-image-screenshot-detector` (pulls `best.pt` and `calibration.json` at startup)
- **Model repo:** `meownager/ai-image-screenshot-detector` (holds the trained weights)

The Space code lives in `hf-space/app.py` and is self-contained (the model-building code is inlined so the Space doesn't need to import from `src/`).

## Per-file authorship

Per the ECE 57000 rubric, this section discloses line-level authorship across the repository. The four verbatim snippets from Checkpoints 1 and 2 are **student-authored**; the scaffolding, training loop, Colab notebook, tests, and paper prose were drafted with Anthropic's Claude as a writing and debugging assistant and reviewed and corrected by the student. See `LLM_ACKNOWLEDGEMENT.md` for details.

| File | Lines | Student-authored (verbatim from CP1/CP2) | Claude-assisted, student-reviewed |
| --- | :-: | --- | --- |
| `src/screenshot_augment.py` | 60 | lines 1–60 (entire file — CP1 Snippet 1, `simulate_screenshot`) | — |
| `src/train.py` | 293 | lines 34–84 (CP1 Snippet 2: `build_model`, ImageNet constants, `build_train_transform`) | lines 1–33, 85–293 (training loop, data loaders, CLI, logging) |
| `src/ablation.py` | 132 | lines 28–35 (CP2 Snippet 1: `AUG_CONFIGS` table) | lines 1–27, 36–132 (ablation runner, arg parsing) |
| `src/app.py` | 189 | lines 103–145 (CP2 Snippet 2: `predict` function) | lines 1–102, 146–189 (loading, validation, Gradio UI) |
| `src/data.py` | 129 | — | lines 1–129 (CIFAKE dataset, stratified split) |
| `src/eval.py` | 171 | — | lines 1–171 (clean/screenshot/realworld evaluation CLI) |
| `src/calibration.py` | 138 | — | lines 1–138 (temperature scaling, L-BFGS fit) |
| `scripts/download_cifake.py` | 67 | — | lines 1–67 (HF Hub download helper) |
| `tests/test_smoke.py` | 111 | — | lines 1–111 (import + shape smoke test) |
| `notebooks/train_colab.ipynb` | 19 cells | — | all cells (setup, training, eval, calibration, HF push) |
| `hf-space/app.py` | 197 | lines 103–145 mirror CP2 Snippet 2 verbatim | rest (self-contained Space layout) |
| `hf-space/requirements.txt` | 5 | — | all (Space dependency pin) |
| `paper/main.tex` | ~600 | research question, `simulate_screenshot` design, ablation structure, IG collection, Gemini comparison, numerical results (all student work) | Claude-assisted prose polishing and LaTeX formatting |
| `paper/references.bib` | — | — | Claude-assisted BibTeX entries, student-verified citations |
| `README.md` | this file | — | Claude-assisted, student-reviewed |
| `LLM_ACKNOWLEDGEMENT.md` | 32 | — | Claude-assisted, student-reviewed |
| `data/ig-data/` | 49 images | **all 49 Instagram screenshots manually collected and labeled by the student** | — |

## LLM acknowledgement

Per the ECE 57000 rubric, LLM assistance is disclosed in `LLM_ACKNOWLEDGEMENT.md` and in the paper's `LLM Acknowledgement` section.

## License

MIT (see `LICENSE`). Original Instagram content in `data/ig-data/` remains the property of the respective creators and the Instagram platform; images are included for non-commercial academic evaluation only.
