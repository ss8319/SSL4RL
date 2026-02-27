import torch
import sys
import os

def test_qwen3_environment():
    print("--- Environment Check ---")
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
        
        import qwen_vl_utils
        print(f"qwen_vl_utils found! Location: {qwen_vl_utils.__file__}")
        
        from qwen_vl_utils import fetch_image, fetch_video
        print("Successfully imported fetch_image and fetch_video from qwen_vl_utils")
        
    except ImportError as e:
        print(f"FAILED Environment Check: {e}")
        return False
    return True

def test_loading():
    print("\n--- Model Loading Check (Meta Device) ---")
    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    try:
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
        
        print(f"Loading config for {MODEL_ID}...")
        config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
        print(f"Success! Model type: {config.model_type}")
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Success!")
        
        print("Instantiating model on meta device...")
        # AutoModelForVision2Seq is deprecated in v5.0, but we are on 4.57.6
        # Let's try AutoModelForImageTextToText if available, else AutoModelForVision2Seq
        try:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq
            
        with torch.device("meta"):
            model = model_cls.from_config(config, trust_remote_code=True)
        print("Success! Architecture recognized and model instantiated on meta device.")
    except Exception as e:
        print(f"FAILED Model Loading: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    env_ok = test_qwen3_environment()
    if env_ok:
        test_loading()
    else:
        sys.exit(1)