from vllm.model_executor.model_loader import get_model
from vllm.config import VllmConfig, ModelConfig, LoadConfig, ParallelConfig, SchedulerConfig, DeviceConfig, CacheConfig, LoRAConfig, SpeculativeConfig, DecodingConfig, ObservabilityConfig
import torch
import sys

def test_vllm_config_loading():
    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    print(f"--- Testing vLLM config/model loader for {MODEL_ID} ---")
    
    try:
        model_config = ModelConfig(
            model=MODEL_ID,
            tokenizer=MODEL_ID,
            tokenizer_mode="auto",
            trust_remote_code=True,
            dtype="float16",
            seed=0,
            revision=None,
            code_revision=None,
            tokenizer_revision=None,
            max_model_len=1024,
            quantization=None,
            enforce_eager=True,
            max_context_len_to_capture=1024,
            max_seq_len_to_capture=1024,
        )
        print("ModelConfig created.")
        
        # We need a VllmConfig
        vllm_config = VllmConfig(
            model_config=model_config,
            load_config=LoadConfig(),
            parallel_config=ParallelConfig(1, 1, False),
            scheduler_config=SchedulerConfig(max_num_batched_tokens=1024, max_num_seqs=16, max_model_len=1024),
            device_config=DeviceConfig("cpu"),
            cache_config=CacheConfig(block_size=16, gpu_memory_utilization=0.1, swap_space=4, cache_dtype="auto"),
            lora_config=None,
            speculative_config=None,
            decoding_config=DecodingConfig(),
            observability_config=ObservabilityConfig(),
        )
        print("VllmConfig created.")
        
        # This will fail on CPU if it tries to load weights to GPU, but we can see the trace
        # Actually we just want to see if the AttributeError: 'Qwen3VLConfig' object has no attribute 'vocab_size' persists
        from vllm.model_executor.model_loader.loader import _initialize_model
        print("Initializing model (dry run)...")
        # _initialize_model will try to create the model class
        model = _initialize_model(vllm_config=vllm_config)
        print(f"Model initialized: {type(model)}")
        
    except Exception as e:
        print(f"\nFAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vllm_config_loading()
