#!/bin/bash
#SBATCH --job-name=ssl4rl_verify
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:2
#SBATCH --constraint=A100-80G
#SBATCH --mem=128G
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=fit
#SBATCH --account=ub62
#SBATCH --qos=fitq

# Verification script for small dataset run (train + val + save)

# Load environment variables (e.g., WANDB_API_KEY)
if [[ -f ".env" ]]; then
  source .env
  export $(grep -v '^#' .env | xargs)
fi

TASK="jigsaw_small"
MODEL_PATH=${1:-"Qwen/Qwen3-VL-4B-Instruct"}

# Rollout backend selection:
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

RUN_ID=${SLURM_JOB_ID:-"verify_$(date +%Y%m%d_%H%M%S)_$$"}
export RUN_ID

SAVE_DIR="models/ssl4rl_verify_${TASK}_${RUN_ID}"
TENSORBOARD_DIR="tensorboard_result/ssl4rl_verify_${TASK}_${RUN_ID}"

train_path="${DATASET_DIR}/train.parquet"
valid_path="${DATASET_DIR}/valid.parquet"
test_path="${DATASET_DIR}/test.parquet"

# For PPO, we often include test data in the validation files for periodic checks
train_files="['$train_path']"
val_files="['$valid_path','$test_path']"

TRAIN_DATA_LEN="$(python3 - <<PY 2>/dev/null || echo 0
import datasets
ds = datasets.load_dataset("parquet", data_files="${train_path}")["train"]
print(len(ds))
PY
)"
if [[ "${TRAIN_DATA_LEN}" =~ ^[0-9]+$ ]] && (( TRAIN_DATA_LEN > 0 )); then
  TRAIN_BATCH_SIZE="${SSL4RL_TRAIN_BATCH_SIZE:-16}"
  if (( TRAIN_BATCH_SIZE > TRAIN_DATA_LEN )); then
    TRAIN_BATCH_SIZE="$TRAIN_DATA_LEN"
  fi
else
  TRAIN_DATA_LEN="0"
  TRAIN_BATCH_SIZE="${SSL4RL_TRAIN_BATCH_SIZE:-16}"
fi

PPO_MINI_BATCH_SIZE="${SSL4RL_PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
PPO_MICRO_BSZ_PER_GPU="${SSL4RL_PPO_MICRO_BSZ_PER_GPU:-2}"
if (( PPO_MINI_BATCH_SIZE < 4 )); then
  PPO_MICRO_BSZ_PER_GPU=1
fi

echo "TRAIN_DATA_LEN: ${TRAIN_DATA_LEN}"
echo "TRAIN_BATCH_SIZE: ${TRAIN_BATCH_SIZE}"
echo "PPO_MINI_BATCH_SIZE: ${PPO_MINI_BATCH_SIZE}"
echo "PPO_MICRO_BSZ_PER_GPU: ${PPO_MICRO_BSZ_PER_GPU}"

mkdir -p models
mkdir -p tensorboard_result

export HYDRA_FULL_ERROR=1
SSL4RL_RAY_BASEDIR_DEFAULT="${SLURM_TMPDIR:-/tmp}"
SSL4RL_RAY_BASEDIR="${SSL4RL_RAY_BASEDIR:-$SSL4RL_RAY_BASEDIR_DEFAULT}"
export RAY_TMPDIR="${SSL4RL_RAY_BASEDIR%/}/ray/${RUN_ID}"
mkdir -p "$RAY_TMPDIR"
export RAY_raylet_start_wait_time_s=300
export RAY_gcs_rpc_server_reconnect_timeout_s=300
export RAY_DASHBOARD_ENABLED=0
export RAY_DASHBOARD_DISABLE_AGGREGATOR_AGENT=1
export RAY_file_system_monitor_warn_threshold_fraction=0.99
export RAY_FILE_SYSTEM_MONITOR_WARN_THRESHOLD_FRACTION="${RAY_file_system_monitor_warn_threshold_fraction}"
export TOKENIZERS_PARALLELISM=false

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export QWEN2_VL_MIN_PIXELS=3136
export QWEN2_VL_MAX_PIXELS=50176
export QWEN3_VL_MIN_PIXELS=3136
export QWEN3_VL_MAX_PIXELS=50176

RAY_LOG_ARCHIVE_DIR="${PWD}/outputs/ray/${RUN_ID}"
mkdir -p "$RAY_LOG_ARCHIVE_DIR"

_ssl4rl_on_exit() {
  if [[ -d "$RAY_TMPDIR/ray" ]]; then
    cp -a "$RAY_TMPDIR/ray" "$RAY_LOG_ARCHIVE_DIR/" 2>/dev/null || true
  fi
}
trap _ssl4rl_on_exit EXIT

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=16384 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=0 \
    data.truncation='right' \
    data.image_key=images \
    data.dataloader_num_workers=0 \
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
    actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "tensorboard", "wandb"]' \
    trainer.project_name="ssl4rl_verify_${TASK}" \
    trainer.experiment_name="ssl4rl_verify_${TASK}_${RUN_ID}" \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=2 \
    trainer.test_freq=2 \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.total_epochs=5 \
    critic.model.path="$MODEL_PATH" \
    critic.model.tokenizer_path="$MODEL_PATH" \
    reward_model.enable=False \
    ray_init.num_cpus=32
