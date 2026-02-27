try:
    import torch
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"ABI used: {torch._C._GLIBCXX_USE_CXX11_ABI}")
    
    print("\nAttempting to import flash_attn...")
    import flash_attn
    print("Success! flash_attn imported.")
    
    print("\nAttempting to import flash_attn_2_cuda...")
    import flash_attn_2_cuda
    print("Success! flash_attn_2_cuda imported.")
    
except ImportError as e:
    print(f"\nImportError: {e}")
except Exception as e:
    print(f"\nException: {type(e).__name__}: {e}")
