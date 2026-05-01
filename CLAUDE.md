# Base rules for Claude in this project

These rules apply to ALL Claude sessions opened against this folder. Follow them strictly.

## Working style (non-negotiable)

1. **Minimize token usage.** Do not run web searches, repo scans, long file reads, or large code generation without confirming first. Propose what you'd do, estimate scope, get a yes.
2. **Teach + delegate.** Syeda is learning AI/ML from scratch. Do not do the work for her. Give simple, beginner-friendly, step-by-step guidance she runs herself. The point is for her to learn.
3. **Plan first, code last.** Discuss architecture and methodology BEFORE writing code or recommending platforms.
4. **Build from foundations.** Never assume prior knowledge. Connect new concepts to ones already covered.
5. **Keep responses concise.** Save tokens for actual learning explanations, not exploratory work she didn't authorize.

## About Syeda

- Beginner in AI/ML; wants to learn EVERYTHING from foundations to commercial deployment
- Currently focused on CNNs (this project)
- Python skill level: basics only
- Purdue student (smonowar@purdue.edu); ECE 57000 Spring 2026
- Treats projects as both learning vehicles AND portfolio pieces — encourage real commits/PRs

## About this project

- **Repo:** https://github.com/meownager/ai-image-screenshot-detector
- **Status:** Course version (ECE 57000) submitted; now evolving into commercial-grade product
- **Goal:** Robust AI-vs-real classifier for social media screenshots; eventually commercializable; outperform Gemini at this task
- **Hard constraints:** Target = social-media screenshots (no SynthID watermarks); must work where humans can't visually tell
- **Current architecture:** EfficientNet-B0 frozen backbone + 2-class head, trained on CIFAKE (SD v1.4 only) with `simulate_screenshot` augmentation
- **Known weakness:** Generator coverage is too narrow (CIFAKE only) — fails on Midjourney/Sora/Flux

## Conventions in this repo

- Source code: `src/`
- Notebooks: `notebooks/` and `Colab/`
- Hugging Face Space: `hf-space/`
- Paper: `paper/main.tex` (ICLR 2026 template)
- Dataset folders: `data/cifake/` (training) and `data/ig-data/` (eval — 49 IG screenshots)
- Tests: `tests/test_smoke.py`
- Diagnosis & next steps: `DIAGNOSIS.md`

## Commit style

Syeda commits her own code (for learning). Claude drafts changes; Syeda runs `git add` / `git commit` / `git push`. Never push from Claude's session without explicit request.
