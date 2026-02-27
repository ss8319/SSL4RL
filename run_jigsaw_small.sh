#!/bin/bash
#SBATCH --job-name=ssl4rl_jigsaw_small
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --constraint=A100-80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --partition=fit
#SBATCH --account=ub62
#SBATCH --qos=fitq

set -euo pipefail

# Convenience wrapper for running the tiny jigsaw dataset end-to-end.
# Usage:
#   sbatch run_jigsaw_small.sh [MODEL_PATH]

MODEL_PATH="${1:-Qwen/Qwen3-VL-4B-Instruct}"

# Make the run faster/cheaper by overriding a few knobs via env.
export SSL4RL_TRAIN_BATCH_SIZE="${SSL4RL_TRAIN_BATCH_SIZE:-20}"
export SSL4RL_PPO_MICRO_BSZ_PER_GPU="${SSL4RL_PPO_MICRO_BSZ_PER_GPU:-1}"

bash run_dermogpt_task.sh jigsaw_small "$MODEL_PATH"
