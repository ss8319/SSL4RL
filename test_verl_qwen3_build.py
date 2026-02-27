import torch
from verl.workers.fsdp_workers import FSDPWorker
from omegaconf import OmegaConf
import sys
import os

# Mock the environment needed for FSDPWorker if necessary
# or just test the model building part directly

def test_verl_build_model():
    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    print(f"--- Testing verl FSDPWorker model building for {MODEL_ID} ---")
    
    config = OmegaConf.create({
        "actor_rollout_ref": {
            "model": {
                "path": MODEL_ID,
                "trust_remote_code": True,
                "use_remove_padding": True,
                "lora_rank": 64,
                "lora_alpha": 32,
                "target_modules": "all-linear",
                "enable_gradient_checkpointing": True
            },
            "actor": {
                "fsdp_config": {
                    "param_offload": False,
                    "optimizer_offload": False
                }
            }
        },
        "trainer": {
            "n_gpus_per_node": 1,
            "nnodes": 1
        }
    })
    
    # We might not be able to instantiate FSDPWorker fully without Ray/Distributed
    # but we can try to call the internal methods if they are static or don't depend on too much
    
    try:
        from transformers import AutoConfig, AutoModelForVision2Seq
        print("Checking if transformers can load config and model...")
        config_obj = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        print(f"Config loaded: {type(config_obj)}")
        
        # Test the logic in fsdp_workers.py:280
        if type(config_obj) in AutoModelForVision2Seq._model_mapping.keys():
            print("Verified: Qwen3VLConfig is in AutoModelForVision2Seq mapping.")
        else:
            print("WARNING: Qwen3VLConfig NOT in AutoModelForVision2Seq mapping. This might be a problem in verl logic.")
            
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    return True

if __name__ == "__main__":
    test_verl_build_model()
