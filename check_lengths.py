import os
import datasets
from transformers import AutoProcessor
import numpy as np

# Set pixel limits as in run_dermogpt_task.sh
os.environ["QWEN3_VL_MIN_PIXELS"] = "3136"
os.environ["QWEN3_VL_MAX_PIXELS"] = "50176"

model_path = "Qwen/Qwen3-VL-2B-Instruct"
dataset_path = "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/our_datasets/dermogpt/rotation/train.parquet"

print(f"Loading processor: {model_path}...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

print(f"Loading dataset: {dataset_path}...")
ds = datasets.load_dataset("parquet", data_files=dataset_path)["train"]

def doc2len(doc):
    messages = doc["prompt"]
    # Qwen-VL Chat Template logic
    # Reconstruct messages with <image> placeholders if needed
    # But for rotation task, the prompt already has <image><image> in the content string
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    images = doc["images"]
    from PIL import Image
    pil_images = []
    for item in images:
        if isinstance(item, dict) and "image" in item:
            img_path = item["image"]
            if img_path.startswith("file://"):
                # The path in the parquet is relative to some root?
                # 'file://datasets/dermogpt/rotation_json/train/images/...'
                # The actual file is likely at:
                # /home/ssim0070/ub62_scratch/ssim0070/SSL4RL/datasets/dermogpt/rotation_json/train/images/...
                full_path = os.path.join("/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/", img_path[7:])
                if os.path.exists(full_path):
                    pil_images.append(Image.open(full_path).convert("RGB"))
                else:
                    print(f"Warning: Image not found at {full_path}")
        elif isinstance(item, str) and item.startswith("file://"):
            full_path = os.path.join("/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/", item[7:])
            if os.path.exists(full_path):
                pil_images.append(Image.open(full_path).convert("RGB"))
            
    proc_kwargs = {
        "min_pixels": 3136,
        "max_pixels": 50176,
    }
    inputs = processor(text=[raw_prompt], images=pil_images, **proc_kwargs)
    return len(inputs["input_ids"][0])

print("Checking first 100 samples...")
lengths = []
for i in range(min(len(ds), 100)):
    l = doc2len(ds[i])
    lengths.append(l)
    if (i+1) % 10 == 0:
        print(f"Checked {i+1} samples...")

print(f"Stats (first 100): Max={max(lengths)}, Min={min(lengths)}, Avg={np.mean(lengths)}")
print(f"Sample input_ids length: {len(lengths[0]) if lengths else 0}")
if ds:
    res = processor(text=[processor.apply_chat_template(ds[0]['prompt'], add_generation_prompt=True, tokenize=False)], images=[Image.open(os.path.join("/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/", item['image'][7:])).convert("RGB") for item in ds[0]['images']], **{"min_pixels": 3136, "max_pixels": 50176})
    print(f"Sample input_ids tokens: {res['input_ids'][0]}")
