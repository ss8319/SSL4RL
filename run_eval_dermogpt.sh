#!/bin/bash
set -euo pipefail

# Runs DermoGPT generation + evaluation (compute_score).
#
# Usage:
#   bash SSL4RL/run_eval_dermogpt.sh <MODEL_PATH> [TASK]
#
# Examples:
#   bash SSL4RL/run_eval_dermogpt.sh "Qwen/Qwen3-VL-2B-Instruct" contrastive
#   bash SSL4RL/run_eval_dermogpt.sh "/path/to/local/ckpt" jigsaw

MODEL_PATH="${1:-}"
SELECTED_TASK="${2:-}"

if [[ -z "$MODEL_PATH" ]]; then
  echo "Error: MODEL_PATH argument is required."
  echo "Usage: bash SSL4RL/run_eval_dermogpt.sh <MODEL_PATH> [TASK]"
  exit 1
fi

DATASET_ROOT="our_datasets/dermogpt"
OUTPUT_DIR="eval_results/dermogpt"

if [[ -z "$SELECTED_TASK" ]]; then
  TASKS=("contrastive" "jigsaw" "position" "rotation")
else
  TASKS=("$SELECTED_TASK")
fi

mkdir -p "$OUTPUT_DIR"

# Resolve repo root (directory containing this script).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use an env that has verl + ray + vllm installed.
# Override with: SSL4RL_PYTHON_BIN=/path/to/python
PYTHON_BIN="${SSL4RL_PYTHON_BIN:-python3}"

# Ensure `import verl` resolves to this repo checkout.
export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"

# Ray tends to be sensitive on shared filesystems; keep its temp on local /tmp and
# increase startup timeouts (we've already seen "node timed out during startup").
RUN_ID="${SLURM_JOB_ID:-"$(date +%Y%m%d_%H%M%S)_$$"}"
export RAY_TMPDIR="/tmp/ray_ssl4rl_eval_${RUN_ID}"
mkdir -p "$RAY_TMPDIR"
export RAY_raylet_start_wait_time_s=300
export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_DASHBOARD_ENABLED=0
export RAY_node_ip_address=127.0.0.1

# Match generation config GPU expectations to what Slurm actually allocates.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Count comma-separated entries
  N_GPUS_PER_NODE="$(awk -F',' '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
else
  N_GPUS_PER_NODE=1
fi

for TASK in "${TASKS[@]}"; do
  echo "========================================"
  echo "Processing task: $TASK"
  echo "Model: $MODEL_PATH"
  echo "GPUs per node: $N_GPUS_PER_NODE"
  echo "========================================"

  TASK_OUTPUT_DIR="${OUTPUT_DIR}/${TASK}"
  mkdir -p "$TASK_OUTPUT_DIR"

  # Optional: clear stale Ray state between tasks.
  ray stop --force >/dev/null 2>&1 || true

  echo "Starting generation..."
  "$PYTHON_BIN" -m verl.trainer.main_generation \
    data.path="${DATASET_ROOT}/${TASK}/test.parquet" \
    data.output_path="${TASK_OUTPUT_DIR}/responses.parquet" \
    model.path="$MODEL_PATH" \
    rollout.n=1 \
    data.n_samples=1 \
    trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
    trainer.nnodes=1

  echo "Starting evaluation..."
  "$PYTHON_BIN" -m verl.trainer.main_eval \
    data.path="${TASK_OUTPUT_DIR}/responses.parquet" \
    custom_reward_function.name=compute_score
done

echo "Done!"
