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
    images_abs_dir = os.path.join(save_dir, "images")
    os.makedirs(images_abs_dir, exist_ok=True)
    
    patch_paths = []
    for i, patch in enumerate(patches):
        path = os.path.join("images", f"{sample_id}_p{i+1}.png")
        patch.save(os.path.join(save_dir, path))
        patch_paths.append(path)
    
    return patch_paths

def process_sample(datum, save_dir, image_root, jigsaw_size, min_size, seed):
    random.seed(seed)
    # torch.manual_seed(seed) # Not needed if not using torch transforms here, but good practice

    img_path = os.path.join(image_root, datum['image'])
    try:
        orig = Image.open(img_path).convert('RGB')
        patch_size_0 = orig.width // jigsaw_size
        patch_size_1 = orig.height // jigsaw_size
        
        if patch_size_0 < min_size or patch_size_1 < min_size:
            return None

        patches = []
        for j in range(jigsaw_size):
            for i in range(jigsaw_size):
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

        return {
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
        }
    except Exception as e:
        return None

def process_dataset(input_path, save_dir, limit=None):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if limit:
        print(f"Limiting to first {limit} items from {len(data)}...")
        data = data[:limit]
        
    os.makedirs(save_dir, exist_ok=True)
    
    dataset_json = []
    total_samples = 0
    min_size = 28
    
    num_workers = min(32, multiprocessing.cpu_count())

    print(f"Generating Jigsaw tasks for {len(data)} images from {os.path.basename(input_path)} using {num_workers} workers...")
    
    tasks = []
    for idx, datum in enumerate(data):
        sample_seed = 42 + idx 
        tasks.append((datum, save_dir, IMAGE_ROOT, JIGSAW_SIZE, min_size, sample_seed))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm.tqdm(executor.map(process_sample_wrapper, tasks), total=len(tasks)))

    dataset_json = [res for res in results if res is not None]
    total_samples = len(dataset_json)

    output_json_path = os.path.join(save_dir, "dataset.json")
    with open(output_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Dataset generated at {save_dir}")
    print(f"Total samples: {total_samples}")

def process_sample_wrapper(args):
    return process_sample(*args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    
    splits = {
        "train": "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/data/dermogpt/train/dermogpt_mcqa_train_8000.json",
        "valid": "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/data/dermogpt/valid/dermogpt_mcqa.json",
        "test": "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/data/dermogpt/test/dermogpt_mcqa.json"
    }
    
    for split_name, input_path in splits.items():
        save_dir = os.path.join(SAVE_BASE_DIR, split_name)
        process_dataset(input_path, save_dir, limit=args.limit)
