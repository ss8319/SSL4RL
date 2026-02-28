import os
import json
import tqdm
import torch
import random
import math
import numpy as np
from PIL import Image

IMAGE_ROOT = "/home/ssim0070/ub62_scratch/ssim0070/Derm-All/"
SAVE_BASE_DIR = "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/datasets/dermogpt/rotation_json/"

GRANUALITY = 90

QUERY_TEMPLATE = f'''<image><image> 

These are two images. The second image is a rotated version of the first image. 
Please determine how many degrees the second image has been rotated **counter-clockwise** relative to the first image.
The granuality of rotation is {GRANUALITY} degrees, meaning the second image could have been rotated by 0, {GRANUALITY}, {2 * GRANUALITY}, ..., {360 - GRANUALITY} degrees.

You must reason step-by-step and then provide the final answer.

The output **must strictly follow** this format:
<think> your reasoning here </think> <answer>number_of_degrees</answer>'''

def rotatedRectWithMaxArea(w, h, angle):
    if w <= 0 or h <= 0: return 0,0
    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    return int(wr), int(hr)

def save_sample_rotation(save_dir, sample_id, im1, im2):
    images_abs_dir = os.path.join(save_dir, "images")
    os.makedirs(images_abs_dir, exist_ok=True)
    
    path1 = os.path.join("images", f"{sample_id}_orig.png")
    path2 = os.path.join("images", f"{sample_id}_rot.png")
    
    im1.save(os.path.join(save_dir, path1))
    im2.save(os.path.join(save_dir, path2))
    
    return [path1, path2]

def process_sample(datum, save_dir, image_root, granuality, min_size, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    img_path = os.path.join(image_root, datum['image'])
    try:
        img = Image.open(img_path).convert('RGB')
        possible_angles = list(range(0, 360, granuality))
        angle = random.choice(possible_angles)

        rotated_img = img.rotate(angle, expand=True)
        wr, hr = rotatedRectWithMaxArea(img.width, img.height, math.radians(angle))
        rotated_img = rotated_img.crop((
            (rotated_img.width - wr) / 2,
            (rotated_img.height - hr) / 2,
            (rotated_img.width + wr) / 2,
            (rotated_img.height + hr) / 2,
        ))

        if rotated_img.width < min_size or rotated_img.height < min_size:
            return None

        sample_id = f"{datum['id']}_rotation"
        img_paths = save_sample_rotation(save_dir, sample_id, img, rotated_img)

        return {
            'id': sample_id,
            'image': img_paths,
            'source': datum.get('source', 'unknown'),
            'conversations': [
                {
                    'from': 'human',
                    'value': QUERY_TEMPLATE
                },
                {
                    'from': 'gpt',
                    'value': str(angle)
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
    
    # Use 32 workers for a good balance of speed and I/O
    num_workers = min(32, multiprocessing.cpu_count())

    print(f"Generating Rotation tasks for {len(data)} images from {os.path.basename(input_path)} using {num_workers} workers...")
    
    # Prepare arguments for each sample
    tasks = []
    for idx, datum in enumerate(data):
        # Unique seed for each task to maintain reproducibility and independence
        sample_seed = 42 + idx 
        tasks.append((datum, save_dir, IMAGE_ROOT, GRANUALITY, min_size, sample_seed))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use starmap if we were using Pool, but here we just use submit or map
        results = list(tqdm.tqdm(executor.map(process_sample_wrapper, tasks), total=len(tasks)))

    # Filter out None results (where images were skipped or failed)
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    splits = {
        "valid": "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/data/dermogpt/valid/dermogpt_mcqa.json",
        "test": "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/data/dermogpt/test/dermogpt_mcqa.json"
    }
    
    for split_name, input_path in splits.items():
        save_dir = os.path.join(SAVE_BASE_DIR, split_name)
        process_dataset(input_path, save_dir, limit=args.limit)
