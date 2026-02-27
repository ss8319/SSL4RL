#!/bin/bash
#SBATCH --job-name=test_vllm
#SBATCH --output=test_vllm_%j.out
#SBATCH --error=test_vllm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --partition=fit
#SBATCH --account=ub62
#SBATCH --qos=fitq

conda run -n verl python -u test_qwen3_vllm.py
