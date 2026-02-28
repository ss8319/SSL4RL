import os
import datasets
from transformers import AutoProcessor
from PIL import Image
import re

# Set pixel limits
os.environ["QWEN3_VL_MIN_PIXELS"] = "3136"
os.environ["QWEN3_VL_MAX_PIXELS"] = "50176"

model_path = "Qwen/Qwen3-VL-2B-Instruct"
dataset_path = "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/our_datasets/dermogpt/rotation/train.parquet"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
ds = datasets.load_dataset("parquet", data_files=dataset_path)["train"]

def _build_messages(example):
    example = dict(example)
    messages = example.pop("prompt")
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
    pil_images = [Image.open(os.path.join("/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/", item["image"][7:])).convert("RGB") for item in doc["images"]]
    inputs = processor(text=[raw_prompt], images=pil_images, **{"min_pixels": 3136, "max_pixels": 50176})
    return len(inputs["input_ids"][0])

print(f"Checking samples 2430-2435...")
for i in range(2430, 2436):
    print(f"Sample {i} length: {doc2len(ds[i])}")
