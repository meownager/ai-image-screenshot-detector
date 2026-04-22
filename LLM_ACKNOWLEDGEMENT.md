# LLM Acknowledgement

**Student:** Syeda Maliha Monowara
**Course:** ECE 57000, Spring 2026, Section 002
**Project:** Detection of AI Images from Screenshots (Track 2 - Product Prototype)

Per the ECE 57000 final-project rubric, this file discloses the use of large language models in the production of this project.

## Tool used

Anthropic's Claude (Claude Sonnet 4.6-class model), accessed through the Claude desktop application in Cowork mode.

## What the LLM was used for

- **Project planning and system architecture.** Producing the Phase A planning document, system architecture write-up, and hour-by-hour schedule.
- **Source code scaffolding.** The four verbatim code snippets from Checkpoints 1 and 2 (`simulate_screenshot`, `build_model` + `train_transform`, `AUG_CONFIGS` + `run_ablation`, and the Gradio `predict` function) were pre-committed in the checkpoints; Claude produced the surrounding infrastructure code (data loading, training loop, evaluation, temperature calibration, Colab notebook, download script, README).
- **Paper drafting.** Draft text for the ICLR-26 formatted write-up: Abstract, Introduction, Related Work, Methodology, Experiments, Conclusion.
- **Debugging assistance.** Diagnosing import errors, tensor-shape issues, and Gradio-Hugging Face deployment errors during integration.

## What the student did independently

- **Problem formulation and checkpoint methodology.** The problem statement, dataset choice (CIFAKE + GenImage SD), the `simulate_screenshot` pipeline design, the EfficientNet-B0 fine-tuning strategy, and the four-configuration ablation table were authored by the student in Checkpoints 1 and 2 before any LLM involvement in the final implementation.
- **Running all training and evaluation.** The student executed the Colab notebook, verified the results, and collected the real-world Instagram screenshots used in the held-out evaluation.
- **Review and final editing.** All LLM-generated code and prose were reviewed by the student; inaccurate claims, inconsistencies, and code bugs were corrected before inclusion.
- **Final submission.** The student is solely responsible for the content and correctness of the submission.

## What the LLM did not do

- The LLM did not fabricate experimental numbers. Every value in the Experiments section of the paper comes from an actual training run. Where numbers are borrowed from the Checkpoint 2 preliminary table for comparison, they are cited as such.
- The LLM did not have access to any of the student's course material, grades, or private Purdue systems.
- The LLM did not submit the assignment; the student did.
