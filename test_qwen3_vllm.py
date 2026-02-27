import torch
from vllm import LLM, SamplingParams
import sys

def test_vllm_loading():
    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    print(f"--- Testing vLLM loading for {MODEL_ID} ---")
    
    try:
        # For Qwen3-VL, we need to enable mm preprocessor cache disable as in your script
        # and likely need trust_remote_code=True
        llm = LLM(
            model=MODEL_ID,
            trust_remote_code=True,
            gpu_memory_utilization=0.1, # Keep it small for test
            enforce_eager=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt={"image": 2}, # Just in case
        )
        print("Success! vLLM loaded the model.")
        
        # Test generation
        sampling_params = SamplingParams(max_tokens=10)
        outputs = llm.generate("Hello", sampling_params)
        print(f"Generation output: {outputs[0].outputs[0].text}")
        
    except Exception as e:
        print(f"\nFAILED vLLM Loading: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    test_vllm_loading()
