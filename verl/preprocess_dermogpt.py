import argparse
import os
import json
import datasets
from tqdm import tqdm

def process_dermogpt_json(dataset_root, split, output_dir, store_images_as: str):
    split_dir = os.path.join(dataset_root, split)
    json_path = os.path.join(split_dir, "dataset.json")
    
    if not os.path.exists(json_path):
        print(f"Skipping split {split}: {json_path} not found.")
        return

    print(f"Processing {split} split from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    processed_data = []
    for idx, item in enumerate(tqdm(data, desc=f"Converting {split}")):
        # Convert images
        images = []
        for img_rel_path in item["image"]:
            img_abs_path = os.path.join(split_dir, img_rel_path)
            if not os.path.exists(img_abs_path):
                print(f"Missing image {img_abs_path}")
                continue
            if store_images_as == "path":
                images.append({"type": "image", "image": f"file://{img_abs_path}"})
            elif store_images_as == "pil":
                from PIL import Image

                try:
                    images.append(Image.open(img_abs_path).convert("RGB"))
                except Exception as e:
                    print(f"Error loading image {img_abs_path}: {e}")
                    continue
            else:
                raise ValueError(f"Unknown store_images_as={store_images_as}")

        if not images:
            continue

        # Extract human query and gpt answer from conversations
        conversations = item['conversations']
        human_query = ""
        gpt_answer = ""
        for turn in conversations:
            if turn['from'] == 'human':
                human_query = turn['value']
            elif turn['from'] == 'gpt':
                gpt_answer = turn['value']

        # Construct the verl-compatible schema
        # We ensure data_source contains task-specific strings (ContrastiveQA, JigsawQA, etc.) 
        # so the verl/utils/reward_score/ can match them.
        dataset_name = os.path.basename(dataset_root.rstrip("/"))
        if "contrastive" in dataset_name.lower():
            data_source = "ContrastiveQA_dermogpt"
        elif "jigsaw" in dataset_name.lower():
            data_source = "JigsawQA_2x2_dermogpt"
        elif "position" in dataset_name.lower():
            data_source = "PositionQA_dermogpt"
        elif "rotation" in dataset_name.lower():
            data_source = "RotationQA_dermogpt"
        else:
            data_source = dataset_name

        entry = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": human_query,
                }
            ],
            "images": images,
            "ability": "vision",
            "reward_model": {"style": "rule", "ground_truth": gpt_answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "answer": gpt_answer,
                "question": human_query,
                "id": item['id'],
                "source": item['source']
            },
        }
        processed_data.append(entry)

    # Save to parquet
    os.makedirs(output_dir, exist_ok=True)
    dataset = datasets.Dataset.from_list(processed_data)
    save_path = os.path.join(output_dir, f"{split}.parquet")
    dataset.to_parquet(save_path)
    print(f"Saved {len(processed_data)} samples to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess DermoGPT JSON datasets for verl training.")
    parser.add_argument("--dataset_root", required=True, help="Path to the JSON dataset root (containing train/valid/test subfolders)")
    parser.add_argument("--output_root", required=True, help="Path to save the processed parquet files")
    parser.add_argument(
        "--store_images_as",
        choices=["path", "pil"],
        default="path",
        help="How to store images in parquet. 'path' stores file:// paths (recommended). 'pil' embeds PIL images (large).",
    )
    
    args = parser.parse_args()
    
    for split in ["train", "valid", "test"]:
        process_dermogpt_json(args.dataset_root, split, args.output_root, args.store_images_as)
