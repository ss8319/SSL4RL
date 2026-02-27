# SHAMUS Project Overview

## 1. Inference and Generation

- **Main Generation**: `verl/trainer/main_generation.py`
  - Ray-based distributed script using vLLM.
  - Processes parquet prompt datasets and saves model responses.
- **Multi-modal Support**: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
  - Logic for VLMs (e.g., Qwen2.5-VL).

## 2. Evaluation

- **Offline Evaluation**: `verl/trainer/main_eval.py`
  - Calculates metrics using reward models or ground-truth verifiers.
- **Recipe-Specific (DeepSeek R1)**: `recipe/r1/main_eval.py`
  - Tailored for R1 reproduction tasks.
- **Task Verifiers**: `recipe/r1/tasks/`
  - Benchmark-specific logic: GPQA (`gpqa.py`), Math (`math.py`), LiveCodeBench (`livecodebench.py`).
- **Integrated Eval**: Managed via `trainer.test_freq` in `verl.trainer.main_ppo`.

## 3. Benchmark Construction

- **Scripts Directory**: `build_benchmarks/`
- **Core Task Generators**:
  - `makecontrastive_dermogpt.py` (Contrastive)
  - `makejigsaw_dermogpt.py` (Jigsaw)
  - `makerotation_dermogpt.py` (Rotation)
  - `makeposition_dermogpt.py` (Position)

## 4. Utilities

- **Rollout Viewer**: `scripts/rollout_viewer.py`
  - TUI tool for inspecting generated data and scores.

## Quick Start

- **Generate**: Run `main_generation.py` with `verl/trainer/config/generation.yaml`.
- **Evaluate**: Run `main_eval.py` on the generated parquet files.
