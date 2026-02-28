import os
import json
import tqdm
import torch
import random
from PIL import Image

IMAGE_ROOT = "/home/ssim0070/ub62_scratch/ssim0070/Derm-All/"
SAVE_BASE_DIR = "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/datasets/dermogpt/jigsaw_json/"

JIGSAW_SIZE = 2 # 2x2 grid
ORDER_EXAMPLE = "3,2,1,4"

QUERY_TEMPLATE = ('<image>' * (JIGSAW_SIZE * JIGSAW_SIZE) + f'''

The provided images represent {JIGSAW_SIZE * JIGSAW_SIZE} parts of an original image, divided into a {JIGSAW_SIZE}x{JIGSAW_SIZE} grid.

Your task is to determine the correct order of these parts to reconstruct the original image. Starting from the top-left corner, proceed row by row, from left to right and top to bottom, to arrange the parts.

The output should be a string of numbers, separated by a comma, where each number corresponds to the original position of the patches in the restored image. For instance, "{ORDER_EXAMPLE}" would indicate the positions of the patches in the correct order.

Before providing the final result, you must reason through the puzzle step by step. Consider the relative placement of each part and how they fit together.

Your answer should strictly follow this format:

<think>your step-by-step reasoning here</think> <answer>order</answer>''')

def save_sample_patches(save_dir, sample_id, patches):
    sample_rel_dir = os.path.join("samples", sample_id)
    sample_abs_dir = os.path.join(save_dir, sample_rel_dir)
    os.makedirs(sample_abs_dir, exist_ok=True)
    
    patch_paths = []
    for i, patch in enumerate(patches):
        path = os.path.join(sample_rel_dir, f"patch_{i+1}.png")
        patch.save(os.path.join(save_dir, path))
        patch_paths.append(path)
    
    return patch_paths

def process_dataset(input_path, save_dir, max_samples=None):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    
    dataset_json = []
    total_samples = 0
    min_size = 28

    print(f"Generating Jigsaw tasks for {len(data)} images from {os.path.basename(input_path)}...")
    for datum in tqdm.tqdm(data):
        img_path = os.path.join(IMAGE_ROOT, datum['image'])
        try:
            orig = Image.open(img_path).convert('RGB')
            patch_size_0 = orig.width // JIGSAW_SIZE
            patch_size_1 = orig.height // JIGSAW_SIZE
            
            if patch_size_0 < min_size or patch_size_1 < min_size:
                continue

            patches = []
            for j in range(JIGSAW_SIZE):
                for i in range(JIGSAW_SIZE):
                    patch = orig.crop((i * patch_size_0, j * patch_size_1, (i + 1) * patch_size_0, (j + 1) * patch_size_1))
                    patches.append(patch)
            
            order = list(range(len(patches)))
            random.shuffle(order)
            jigpatches = [patches[i] for i in order]

            inverse_order = [0] * len(order)
            for i, orig_pos in enumerate(order):
                inverse_order[orig_pos] = i + 1
            answer = ','.join(map(str, inverse_order))

            sample_id = f"{datum['id']}_jigsaw"
            patch_paths = save_sample_patches(save_dir, sample_id, jigpatches)

            dataset_json.append({
                'id': sample_id,
                'image': patch_paths,
                'source': datum.get('source', 'unknown'),
                'conversations': [
                    {
                        'from': 'human',
                        'value': QUERY_TEMPLATE
                    },
                    {
                        'from': 'gpt',
                        'value': answer
                    }
                ]
            })
            total_samples += 1
        except Exception as e:
            continue

    output_json_path = os.path.join(save_dir, "dataset.json")
    with open(output_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Dataset generated at {save_dir}")
    print(f"Total samples: {total_samples}")

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    
    splits = {
        "train": "SSL4RL/data/dermogpt/train/dermogpt_mcqa.json",
        "valid": "SSL4RL/data/dermogpt/valid/dermogpt_mcqa.json",
        "test": "SSL4RL/data/dermogpt/test/dermogpt_mcqa.json"
    }
    
    for split_name, input_path in splits.items():
        save_dir = os.path.join(SAVE_BASE_DIR, split_name)
        process_dataset(input_path, save_dir, max_samples=5)
