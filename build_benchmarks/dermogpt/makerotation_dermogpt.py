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

def save_sample_pair(save_dir, sample_id, im1, im2):
    sample_rel_dir = os.path.join("samples", sample_id)
    sample_abs_dir = os.path.join(save_dir, sample_rel_dir)
    os.makedirs(sample_abs_dir, exist_ok=True)
    
    path1 = os.path.join(sample_rel_dir, "original.png")
    path2 = os.path.join(sample_rel_dir, "rotated.png")
    
    im1.save(os.path.join(save_dir, path1))
    im2.save(os.path.join(save_dir, path2))
    
    return [path1, path2]

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

    print(f"Generating Rotation tasks for {len(data)} images from {os.path.basename(input_path)}...")
    for datum in tqdm.tqdm(data):
        img_path = os.path.join(IMAGE_ROOT, datum['image'])
        try:
            img = Image.open(img_path).convert('RGB')
            possible_angles = list(range(0, 360, GRANUALITY))
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
                continue

            sample_id = f"{datum['id']}_rotation"
            img_paths = save_sample_pair(save_dir, sample_id, img, rotated_img)

            dataset_json.append({
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
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    splits = {
        "train": "SSL4RL/data/dermogpt/train/dermogpt_mcqa.json",
        "valid": "SSL4RL/data/dermogpt/valid/dermogpt_mcqa.json",
        "test": "SSL4RL/data/dermogpt/test/dermogpt_mcqa.json"
    }
    
    for split_name, input_path in splits.items():
        save_dir = os.path.join(SAVE_BASE_DIR, split_name)
        process_dataset(input_path, save_dir, max_samples=5)
