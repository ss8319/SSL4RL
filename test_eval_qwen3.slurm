#!/bin/bash
#SBATCH --job-name=eval_qwen3
#SBATCH --output=eval_qwen3_%j.out
#SBATCH --error=eval_qwen3_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=fit
#SBATCH --account=ub62
#SBATCH --qos=fitq

# Run evaluation for one task
MODEL="Qwen/Qwen3-VL-2B-Instruct"
TASK="contrastive"

bash SSL4RL/run_eval_dermogpt.sh "$MODEL" "$TASK"
