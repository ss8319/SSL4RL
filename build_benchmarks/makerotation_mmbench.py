import os
import math
import tqdm
import torch
import random
import datasets
import numpy as np

from PIL import Image
from torchvision import transforms

GRANUALITY = 90 # degrees

QUERY_TEMPLATE = f'''<image><image>

These are two images. The second image is a rotated version of the first image. 
Please determine how many degrees the second image has been rotated **counter-clockwise** relative to the first image.
The granuality of rotation is {GRANUALITY} degrees, meaning the second image could have been rotated by 0, {GRANUALITY}, {2 * GRANUALITY}, ..., {360 - GRANUALITY} degrees.

You must reason step-by-step and then provide the final answer.

The output **must strictly follow** this format:
<think> your reasoning here </think> <answer>number_of_degrees</answer>'''


def rotatedRectWithMaxArea(w, h, angle):
    # borrowed from https://stackoverflow.com/a/16778797/13490627
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    wr,hr = int(wr), int(hr)

    return wr,hr


def process_batch(base, start: int, end: int, prefix: str, processor_id: int, seed: int, min_size: int = 28):
    results = []

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for j in range(start, end):
        if str(j - start) == os.environ.get('EARLY_EXIT', str(end + 100)):
            break

        datum = base[j]
        img: Image.Image = datum['image']

        possible_angles = list(range(0, 360, GRANUALITY))
        angle = random.choice(possible_angles)

        rotated_img: Image.Image = img.rotate(angle, expand=True)
        wr, hr = rotatedRectWithMaxArea(img.width, img.height, math.radians(angle))
        rotated_img = rotated_img.crop((
            (rotated_img.width - wr) / 2,
            (rotated_img.height - hr) / 2,
            (rotated_img.width + wr) / 2,
            (rotated_img.height + hr) / 2,
        ))

        if rotated_img.width < min_size or rotated_img.height < min_size:
            continue

        newdatum = {
            'index': datum['index'],
            'source': datum['source'],
            'split': datum['split'],
            'height': rotated_img.height,
            'width': rotated_img.width,
            'images': [img, rotated_img],
            'query': QUERY_TEMPLATE,
            'answer': f'{angle}'
        }

        results.append(newdatum)

    # Create save directory with fallback
    save_dir = f'datasets/MMBench_RotationQA_{GRANUALITY}degree'
    os.makedirs(save_dir, exist_ok=True)

    parquet_name = f'{prefix}_{processor_id:05d}.parquet'
    save_path = os.path.join(save_dir, parquet_name)
    if results:
        new_ds = datasets.Dataset.from_list(results)
        new_ds.to_parquet(save_path)
        print(f"Processor {processor_id}: Saved {len(results)} items to {save_path}")
    else:
        print.warning(f"Processor {processor_id}: No results to save")


def process_split(base: datasets.Dataset, prefix: str, seed: int, num_processors: int = 10):
    total = len(base)
    batch_size = (total + num_processors - 1) // num_processors

    jobs = []
    for i in range(num_processors):
        start = i * batch_size
        end = min(total, (i + 1) * batch_size)
        if start >= end:
            continue
        jobs.append((start, end, prefix, i))

    # 单进程分批次处理
    print("Starting batch processing...")
    for i, (start, end, split_name, processor_id) in enumerate(jobs):
        print(f"Processing batch {i+1}/{len(jobs)}: items {start} to {end}")
        process_batch(base, start, end, split_name, processor_id, seed)
        print(f"Batch {i+1} completed")
    print(f"Successfully completed processing {prefix} split")



if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for split in ['test', 'validation']:
        print(f"Processing {split} split...")
        
        dataset_path = 'datasets/MMBench/en'
        print(f"Loading dataset from: {dataset_path}")
        
        base = datasets.load_dataset(dataset_path, split=split)
        print(f"Successfully loaded {split} split with {len(base)} items")
        
        process_split(base, split, num_processors=1, seed=seed)
        print(f"Completed processing {split} split")