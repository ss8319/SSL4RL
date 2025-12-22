bash models/move_hf_config.sh \
 models/ssl4rl_qwen_3b_mmbench_position/global_step_300/actor/huggingface

python scripts/legacy_model_merger.py merge \
 --backend fsdp \
 --local_dir models/ssl4rl_qwen_3b_mmbench_position/global_step_300/actor \
 --target_dir our_models/ssl4rl_qwen_3b_mmbench_position_step300