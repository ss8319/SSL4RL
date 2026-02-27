#!/bin/bash
#SBATCH --job-name=install_golden_set
#SBATCH --output=install_golden_%j.out
#SBATCH --error=install_golden_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --partition=fit
#SBATCH --account=ub62
#SBATCH --qos=fitq

set -e

echo "--- Loading CUDA 12.2 ---"
module load cuda/12.2.0

echo "--- Cleaning environment ---"
conda run -n verl pip uninstall -y vllm torch torchvision torchaudio flash-attn flash-attn-2-cuda sglang sgl-kernel flashinfer-python

echo "--- Installing Stable Torch 2.6.0 ---"
conda run -n verl pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir

echo "--- Installing SGLang ---"
conda run -n verl pip install "sglang[srt,openai]==0.4.6.post5" --no-cache-dir

echo "--- Installing Prebuilt Flash-Attn ---"
# We use the official URL index for prebuilt wheels to avoid source build
conda run -n verl pip install flash-attn==2.7.4.post1 --no-build-isolation

echo "--- Installing Prebuilt FlashInfer ---"
conda run -n verl pip install flashinfer-python==0.2.5 -i https://flashinfer.ai/whl/cu124/torch2.6/ --no-cache-dir

echo "--- Installing Qwen-VL Utils ---"
conda run -n verl pip install qwen-vl-utils==0.0.14

echo "--- Final Verification ---"
conda run -n verl python -c "import torch; import sglang; import flash_attn; import flashinfer; print('SUCCESS: Golden Set installed.')"
