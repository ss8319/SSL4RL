#!/bin/bash
#SBATCH --job-name=rotation_dermogpt
#SBATCH --output=rotation_dermogpt_%j.out
#SBATCH --error=rotation_dermogpt_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=comp

export PYTHONUNBUFFERED=1
PYTHON_EXEC="/scratch2/ub62/ssim0070/conda/envs/lmms-eval/bin/python3"
SCRIPT_PATH="/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/build_benchmarks/dermogpt_batch/makerotation_dermogpt.py"

$PYTHON_EXEC $SCRIPT_PATH
