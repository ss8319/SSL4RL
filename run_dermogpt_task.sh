#!/bin/bash
#SBATCH --job-name=ssl4rl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --constraint=A100-80G
#SBATCH --mem=768G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=fit
#SBATCH --account=ub62
#SBATCH --qos=fitq

# Usage: bash run_dermogpt_task.sh [TASK] [MODEL_PATH] [DATA_LIMIT]
# Example: sbatch run_dermogpt_task.sh rotation Qwen/Qwen3-VL-2B-Instruct 0.5

# Load environment variables (e.g., WANDB_API_KEY)
if [[ -f ".env" ]]; then
  source .env
  export $(grep -v '^#' .env | xargs)
fi

TASK=$1
MODEL_PATH=${2:-"Qwen/Qwen3-VL-2B-Instruct"}
DATA_LIMIT=${3:-$SSL4RL_DATA_LIMIT}

if [[ -z "$TASK" ]]; then
    echo "Error: TASK argument is required (contrastive, jigsaw, position, or rotation)"
    exit 1
fi

# Rollout backend selection:
# - vLLM currently fails for Qwen3-VL in this environment (vllm==0.8.5.post1 + transformers==5.2.0)
#   even on the Transformers fallback path, so default to HF rollout for these models.
# - Override via: SSL4RL_ROLLOUT_BACKEND=vllm|hf|sglang
SSL4RL_ROLLOUT_BACKEND_DEFAULT="vllm"
if [[ "$MODEL_PATH" == *"Qwen3-VL"* ]]; then
  SSL4RL_ROLLOUT_BACKEND_DEFAULT="hf"
fi
ROLLOUT_BACKEND="${SSL4RL_ROLLOUT_BACKEND:-$SSL4RL_ROLLOUT_BACKEND_DEFAULT}"
echo "Rollout backend: ${ROLLOUT_BACKEND}"
ROLLOUT_TOP_K="-1"
if [[ "${ROLLOUT_BACKEND}" == "hf" ]]; then
  ROLLOUT_TOP_K="0"
fi
export MODEL_PATH ROLLOUT_BACKEND ROLLOUT_TOP_K SSL4RL_ROLLOUT_BACKEND_DEFAULT

DATASET_DIR="$(pwd)/our_datasets/dermogpt/${TASK}"

# Unique run id so multiple jobs don't overwrite outputs.
# - When using Slurm, SLURM_JOB_ID is best.
# - Fallback is timestamp + pid for interactive runs.
RUN_ID=${SLURM_JOB_ID:-"$(date +%Y%m%d_%H%M%S)_$$"}
export RUN_ID

SAVE_DIR="models/ssl4rl_dermogpt_${TASK}_${RUN_ID}"
TENSORBOARD_DIR="tensorboard_result/ssl4rl_dermogpt_${TASK}_${RUN_ID}"

train_path="${DATASET_DIR}/train.parquet"
valid_path="${DATASET_DIR}/valid.parquet"
test_path="${DATASET_DIR}/test.parquet"

# Fail fast if dataset files are missing (common when TASK arg is mistyped).
if [[ ! -f "$train_path" || ! -f "$valid_path" || ! -f "$test_path" ]]; then
  echo "ERROR: Missing dataset parquet file(s) for TASK='${TASK}'."
  echo "Expected:"
  echo "  $train_path"
  echo "  $valid_path"
  echo "  $test_path"
  echo "Available tasks under: $(pwd)/our_datasets/dermogpt/"
  ls -1 "$(pwd)/our_datasets/dermogpt/" 2>/dev/null || true
  exit 2
fi

# For PPO, we often include test data in the validation files for periodic checks
train_files="['$train_path']"
val_files="['$valid_path','$test_path']"

# Data subsetting logic (fraction or absolute number)
if [[ -n "$DATA_LIMIT" ]]; then
    # Validate input: must be a positive number or decimal
    if ! [[ "$DATA_LIMIT" =~ ^[0-9.]+$ ]]; then
        echo "ERROR: DATA_LIMIT must be a number (e.g. 0.5 or 4000). Got: '$DATA_LIMIT'"
        exit 1
    fi

    subset_train_path="${DATASET_DIR}/train_subset_${DATA_LIMIT//./_}_${RUN_ID}.parquet"
    # Set trap early to ensure cleanup even if Python fails
    trap 'rm -f "${subset_train_path}"' EXIT

    echo "Subsetting training data with limit: ${DATA_LIMIT} -> ${subset_train_path}"
    python3 -c "
import pandas as pd
import os
import sys

try:
    df = pd.read_parquet('${train_path}')
    limit = float('${DATA_LIMIT}')
    
    if limit <= 0:
        print(f'Invalid data limit: {limit}. Using full dataset.')
        sys.exit(0)
    
    # Logic: 
    # - If 0 < limit < 1.0 (excluding exactly 1.0), treat as fraction.
    # - If limit >= 1.0, treat as absolute sample count.
    if 0 < limit < 1.0:
        n = int(len(df) * limit)
    else:
        n = int(limit)
    
    n = max(1, min(n, len(df)))
    print(f'Subsetting dataset from {len(df)} to {n} samples')
    df.iloc[:n].to_parquet('${subset_train_path}')
except Exception as e:
    print(f'CRITICAL: Error subsetting dataset: {e}')
    sys.exit(1)
" || exit 1
    
    train_path="${subset_train_path}"
    train_files="['$train_path']"
fi

# If the dataset is very small (e.g., jigsaw_small has only a couple rows),
# the default batch sizes can make the train dataloader empty because the
# trainer uses drop_last=True.
TRAIN_DATA_LEN="$(python3 - <<PY 2>/dev/null || echo 0
import datasets
ds = datasets.load_dataset("parquet", data_files="${train_path}")["train"]
print(len(ds))
PY
)"
# Ensure TRAIN_DATA_LEN is a valid number
if ! [[ "${TRAIN_DATA_LEN}" =~ ^[0-9]+$ ]]; then
  TRAIN_DATA_LEN=0
fi

# Ensure directories exist
mkdir -p models
mkdir -p tensorboard_result

# Ray initialization fixes
if [[ "${SSL4RL_RAY_STOP:-0}" == "1" ]]; then
    ray stop --force || true
    sleep 5
fi

# Better error visibility for Hydra runs
export HYDRA_FULL_ERROR=1

# IMPORTANT: Ray uses AF_UNIX sockets under $RAY_TMPDIR/ray/.../sockets/.
# AF_UNIX path length cannot exceed 107 bytes on Linux. Keep this path SHORT,
# but also make it persistent so we can inspect raylet/gcs logs after failure.
#
# NOTE: Ray will spam warnings (and may disable spilling) when the temp dir's
# filesystem is >95% full. `/scratch2` is often >95% used in this environment
# even when it has hundreds of GB free, so default to node-local storage.
SSL4RL_RAY_BASEDIR_DEFAULT="${SLURM_TMPDIR:-/tmp}"
SSL4RL_RAY_BASEDIR="${SSL4RL_RAY_BASEDIR:-$SSL4RL_RAY_BASEDIR_DEFAULT}"
export RAY_TMPDIR="${SSL4RL_RAY_BASEDIR%/}/ray/${RUN_ID}"
mkdir -p "$RAY_TMPDIR"
export RAY_raylet_start_wait_time_s=300
export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_DASHBOARD_ENABLED=0

# Workaround for Ray dashboard_agent crash in this environment.
export RAY_DASHBOARD_DISABLE_AGGREGATOR_AGENT=1
export RAY_file_system_monitor_warn_threshold_fraction=0.99
export RAY_FILE_SYSTEM_MONITOR_WARN_THRESHOLD_FRACTION="${RAY_file_system_monitor_warn_threshold_fraction}"

# Helpful for tokenizer forks + less log noise.
export TOKENIZERS_PARALLELISM=false

# Helps reduce CUDA allocator fragmentation (often shows up as "reserved but unallocated").
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Optional: helps avoid HF Hub rate-limit stalls.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set (HF Hub downloads may be rate-limited / stall)."
fi

# Force NCCL & CUDA settings for stability
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# Let Slurm/Ray handle CUDA_VISIBLE_DEVICES

# --- Pixel resolution and Task-specific limits ---
# Qwen3-VL visual tokens = Pixels / 196.
# 50176 pixels = 256 tokens per image.
if [[ "$TASK" == "jigsaw"* ]]; then
    # Setup for 4 images (~1024 visual tokens)
    export QWEN3_VL_MAX_PIXELS="${QWEN3_VL_MAX_PIXELS:-50176}"
    MAX_PROMPT_LEN=2048
    DEFAULT_PPO_MICRO_BSZ=2
    DEFAULT_ROLLOUT_MICRO_BSZ=1
else
    # Setup for 2 images (~512 visual tokens)
    export QWEN3_VL_MAX_PIXELS="${QWEN3_VL_MAX_PIXELS:-50176}"
    MAX_PROMPT_LEN=1024
    DEFAULT_PPO_MICRO_BSZ=4
    DEFAULT_ROLLOUT_MICRO_BSZ=2
fi
export QWEN3_VL_MIN_PIXELS="${QWEN3_VL_MIN_PIXELS:-3136}"
export QWEN2_VL_MIN_PIXELS="${QWEN2_VL_MIN_PIXELS:-$QWEN3_VL_MIN_PIXELS}"
export QWEN2_VL_MAX_PIXELS="${QWEN2_VL_MAX_PIXELS:-$QWEN3_VL_MAX_PIXELS}"

# Cap validation batch size to avoid OOM during initial validation (default behavior may batch the entire val set).
VAL_BATCH_SIZE="${SSL4RL_VAL_BATCH_SIZE:-8}"

# HF rollout can OOM if it tries to generate for a huge batch at once. Chunk generation explicitly.
# (HFRollout uses `actor_rollout_ref.rollout.micro_batch_size` to split the generation batch.)
ROLLOUT_MICRO_BSZ="${SSL4RL_ROLLOUT_MICRO_BSZ:-$DEFAULT_ROLLOUT_MICRO_BSZ}"

# For these tasks, outputs are typically short; reducing response length saves a lot of KV-cache memory.
MAX_RESPONSE_LEN="${SSL4RL_MAX_RESPONSE_LEN:-128}"

# Speed optimization: Skip the expensive pre-training validation pass.
VAL_BEFORE_TRAIN="${SSL4RL_VAL_BEFORE_TRAIN:-False}"

# --- Batch Size Robustness Logic ---
# 1. Base Train Batch Size
TRAIN_BATCH_SIZE="${SSL4RL_TRAIN_BATCH_SIZE:-64}"
if (( TRAIN_BATCH_SIZE > TRAIN_DATA_LEN )); then
  TRAIN_BATCH_SIZE="$TRAIN_DATA_LEN"
fi

# 2. PPO Mini Batch Size
# Recommendation: 32 for stable GRPO on 2x A100. Fallback to TRAIN_BATCH_SIZE if dataset is tiny.
DEFAULT_MINI_BSZ=32
if (( TRAIN_BATCH_SIZE < DEFAULT_MINI_BSZ )); then
    DEFAULT_MINI_BSZ="$TRAIN_BATCH_SIZE"
fi
PPO_MINI_BATCH_SIZE="${SSL4RL_PPO_MINI_BATCH_SIZE:-$DEFAULT_MINI_BSZ}"

# 3. PPO Micro Batch Size (must divide the normalized batch size)
# Math: Normalized_BSZ = (PPO_MINI_BATCH_SIZE * rollout_n) / n_gpus_per_node
# Current n=5, gpus=2. So Normalized_BSZ = (PPO_MINI_BATCH_SIZE * 5) / 2
ROLLOUT_N=5
GPUS_PER_NODE=2
NORMALIZED_BSZ=$(( (PPO_MINI_BATCH_SIZE * ROLLOUT_N) / GPUS_PER_NODE ))

PPO_MICRO_BSZ_PER_GPU="${SSL4RL_PPO_MICRO_BSZ_PER_GPU:-$DEFAULT_PPO_MICRO_BSZ}"

# Fallback: If normalized batch size is not divisible by micro batch size, force it to 1
if (( NORMALIZED_BSZ % PPO_MICRO_BSZ_PER_GPU != 0 )); then
  echo "WARNING: Normalized PPO batch size (${NORMALIZED_BSZ}) is not divisible by micro batch size (${PPO_MICRO_BSZ_PER_GPU})."
  echo "         Forcing PPO_MICRO_BSZ_PER_GPU to 1 for robustness."
  PPO_MICRO_BSZ_PER_GPU=1
fi

export TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE SSL4RL_PPO_MICRO_BSZ_PER_GPU="${PPO_MICRO_BSZ_PER_GPU}"

echo "Task-specific optimization for: $TASK"
echo "  Max Pixels: $QWEN3_VL_MAX_PIXELS"
echo "  Max Prompt Len: $MAX_PROMPT_LEN"
echo "  Max Response Len: $MAX_RESPONSE_LEN"
echo "  Train Batch Size: ${TRAIN_BATCH_SIZE}"
echo "  PPO Mini Batch Size: ${PPO_MINI_BATCH_SIZE}"
echo "  PPO Micro Batch Size (per GPU): ${PPO_MICRO_BSZ_PER_GPU}"
echo "  Val Batch Size: ${VAL_BATCH_SIZE}"
echo "  Rollout Micro Batch Size: ${ROLLOUT_MICRO_BSZ}"
echo "  Val Before Train: ${VAL_BEFORE_TRAIN}"

python3 -c "import torch; print('CUDA count:', torch.cuda.device_count()); t = torch.randn(1, device='cuda'); print('CUDA tensor creation success!')" || echo "WARNING: Minimal CUDA check failed!"

echo "Starting RL training for DermoGPT TASK: $TASK"
echo "Model Path: $MODEL_PATH"
echo "Dataset Dir: $DATASET_DIR"
echo "RUN_ID: $RUN_ID"
echo "RAY_TMPDIR: $RAY_TMPDIR"
echo "RAY_DASHBOARD_DISABLE_AGGREGATOR_AGENT: $RAY_DASHBOARD_DISABLE_AGGREGATOR_AGENT"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LEN}" \
    data.max_response_length="${MAX_RESPONSE_LEN}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=16 \
    data.truncation='right' \
    data.image_key=images \
    data.dataloader_num_workers=0 \
    data.require_processor_for_multimodal=true \
    data.trust_remote_code=true \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BSZ_PER_GPU}" \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    +actor_rollout_ref.actor.fsdp_config.use_orig_params=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ROLLOUT_BACKEND}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
    +actor_rollout_ref.rollout.micro_batch_size="${ROLLOUT_MICRO_BSZ}" \
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard", "wandb"]' \
    trainer.project_name="ssl4rl_dermogpt_${TASK}" \
    trainer.experiment_name="ssl4rl_dermogpt_${TASK}_${RUN_ID}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=200 \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.total_epochs=20 \
    critic.model.path="$MODEL_PATH" \
    critic.model.tokenizer_path="$MODEL_PATH" \
    reward_model.enable=False \
    ray_init.num_cpus=32

exit_code=$?
export SSL4RL_EXIT_CODE="$exit_code"

exit $exit_code
