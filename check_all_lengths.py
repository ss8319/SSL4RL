import os
import datasets
from transformers import AutoProcessor
import numpy as np
from PIL import Image
import re

# Set pixel limits as in run_dermogpt_task.sh
os.environ["QWEN3_VL_MIN_PIXELS"] = "3136"
os.environ["QWEN3_VL_MAX_PIXELS"] = "50176"

model_path = "Qwen/Qwen3-VL-2B-Instruct"
dataset_path = "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/our_datasets/dermogpt/rotation/train.parquet"
max_prompt_length = 4096 # From the log

print(f"Loading processor: {model_path}...")
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

print(f"Loading dataset: {dataset_path}...")
ds = datasets.load_dataset("parquet", data_files=dataset_path)["train"]

def _build_messages(example):
    # Mimic verl's _build_messages
    example = dict(example)
    messages = example.pop("prompt")
    image_key = "images"
    
    if image_key in example:
        for message in messages:
            content = message["content"]
            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    content_list.append({"type": "image"})
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list
    return messages

def doc2len(doc):
    messages = _build_messages(doc)
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    images = doc["images"]
    pil_images = []
    for item in images:
        img_path = item["image"]
        full_path = os.path.join("/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/", img_path[7:])
        pil_images.append(Image.open(full_path).convert("RGB"))
            
    proc_kwargs = {"min_pixels": 3136, "max_pixels": 50176}
    inputs = processor(text=[raw_prompt], images=pil_images, **proc_kwargs)
    return len(inputs["input_ids"][0])

print("Checking all samples...")
rejected_indices = []
accepted_count = 0
for i in range(len(ds)):
    l = doc2len(ds[i])
    if l > max_prompt_length:
        rejected_indices.append((i, l))
    else:
        accepted_count += 1
    if (i+1) % 500 == 0:
        print(f"Checked {i+1} samples. Accepted: {accepted_count}, Rejected: {len(rejected_indices)}")

print(f"Total accepted: {accepted_count}")
if rejected_indices:
    print(f"First 5 rejected: {rejected_indices[:5]}")
