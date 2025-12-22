import os
import math
import tqdm
import torch
import random
import datasets
import numpy as np

from PIL import Image
from torchvision import transforms

import logging, sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__).info

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

class DatasetWrapper:
    def __init__(self, split_path: str, prefix: str):
        self.split_path = split_path
        self.prefix = prefix
        
        # load image paths
        self.imgnames = sorted(os.listdir(self.split_path))
        self.imgpaths = [os.path.join(self.split_path, name) for name in self.imgnames]
    
    def getimage(self, index: int):
        assert 0 <= index < len(self.imgpaths)
        img = Image.open(self.imgpaths[index]).convert('RGB')

        # if image is too large, resize it
        max_size = 2000
        if max(img.width, img.height) > max_size:
            scale = math.ceil(max(img.width, img.height) / max_size)
            new_w = img.width // scale
            new_h = img.height // scale
            img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        return img

    def getextra(self, index: int):
        return {
            'original_path': self.imgpaths[index],
        }

    def __len__(self):
        assert len(self.imgpaths) == len(self.imgnames)
        return len(self.imgpaths)
    

def process_batch(base: DatasetWrapper, start: int, end: int, prefix: str, processor_id: int, seed: int, min_size: int = 28):
    results = []

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    progbar = tqdm.trange(start, end, desc=f'Processing {prefix}')
    for j in progbar:
        if str(j - start) == os.environ.get('EARLY_EXIT', str(end + 100)):
            break

        img: Image.Image = base.getimage(j)

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
            **base.getextra(j),
            'height': rotated_img.height,
            'width': rotated_img.width,
            'images': [img, rotated_img],
            'query': QUERY_TEMPLATE,
            'answer': f'{angle}'
        }

        results.append(newdatum)

    # Create save directory with fallback
    save_dir = f'datasets/MMERealWorld-Lite_RotationQA_{GRANUALITY}degree'
    os.makedirs(save_dir, exist_ok=True)

    parquet_name = f'{prefix}_{processor_id:05d}.parquet'
    save_path = os.path.join(save_dir, parquet_name)
    if results:
        new_ds = datasets.Dataset.from_list(results)
        new_ds.to_parquet(save_path)
        logger(f"Processor {processor_id}: Saved {len(results)} items to {save_path}")
    else:
        logger(f"Processor {processor_id}: No results to save")

def process_split(split_path, prefix, num_processors=1, seed=42):
    base = DatasetWrapper(split_path, prefix)
    total = len(base)
    batch_size = (total + num_processors - 1) // num_processors
    logger(f"Processing {prefix} split: {total} items with {num_processors} batches, batch size: {batch_size}")

    jobs = []
    for i in range(num_processors):
        start = i * batch_size
        end = min(total, (i + 1) * batch_size)
        if start >= end:
            continue
        jobs.append((base, start, end, prefix, i))
    logger(f"Created {len(jobs)} batches for {prefix} split")

    results = []
    logger("Starting batch processing...")
    for i, (base, start, end, prefix, processor_id) in enumerate(jobs):
        logger(f"Processing batch {i+1}/{len(jobs)}: items {start} to {end}")
        result = process_batch(base, start, end, prefix, processor_id, seed)
        results.append(result)
        logger(f"Batch {i+1} completed: {result}")

    logger(f"Successfully completed processing {prefix} split")
    return results

def splits_generator(data_base_dir):
    yield os.path.join(data_base_dir, 'data/imgs'), 'MMERealworldLite'

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_base_dir = 'datasets/MME-RealWorld-Lite'
    logger(f'!! Use {data_base_dir} as data dir; it must be the unzipped version of '
        'https://huggingface.co/datasets/yifanzhang114/MME-RealWorld-Lite/resolve/main/data.zip?download=true')

    for split_path, prefix in splits_generator(data_base_dir):
        logger(f'Processing split: {prefix} from path {split_path}')
        try:
            process_split(split_path, prefix, num_processors=1, seed=seed)
        except KeyboardInterrupt:
            break
        except:
            logger(f'Error processing {prefix} from path {split_path}', exc_info=True)
        logger(f'Completed processing {prefix} from path {split_path}')
