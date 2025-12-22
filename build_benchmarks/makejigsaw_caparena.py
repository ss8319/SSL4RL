import os
import tqdm
import math
import torch
import random
import datasets
import numpy as np

from PIL import Image

import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__).info

JIGSAW_SIZE = 2 # JIGSAW_SIZE x JIGSAW_SIZE patchess

COLUMN_NAMES = ['filename', 'height', 'width', 'label', 'query', 'answer', 'images']

ORDER_EXAMPLE = {2: '3,2,1,4',
                 3: '8,9,3,2,1,4,7,5,6',
                 4: '7,9,16,13,11,8,12,15,14,6,4,5,3,2,10,1',
                 5: '12,14,1,4,18,5,17,21,16,6,8,10,2,11,15,19,20,9,22,7,24,23,3,25,13',
                 6: '1,16,17,4,5,12,24,26,13,10,14,18,31,19,20,2,3,21,35,22,23,29,8,6,7,9,11,30,32,27,28,33,15,34,25,36'
                 }[JIGSAW_SIZE]

QUERY_TEMPLATE = ('<image>' * (JIGSAW_SIZE * JIGSAW_SIZE) + f'''

The provided images represent {JIGSAW_SIZE * JIGSAW_SIZE} parts of an original image, divided into a {JIGSAW_SIZE}x{JIGSAW_SIZE} grid.

Your task is to determine the correct order of these parts to reconstruct the original image. Starting from the top-left corner, proceed row by row, from left to right and top to bottom, to arrange the parts.

The output should be a string of numbers, separated by a comma, where each number corresponds to the original position of the patches in the restored image. For instance, "{ORDER_EXAMPLE}" would indicate the positions of the patches in the correct order.

Before providing the final result, you must reason through the puzzle step by step. Consider the relative placement of each part and how they fit together.

Your answer should strictly follow this format:

<think>your step-by-step reasoning here</think> <answer>order</answer>''')

class DatasetWrapper:
    def __init__(self, split_path: str, prefix: str):
        self.split_path = split_path
        self.prefix = prefix
        
        # load image paths
        self.imgpaths = []
        for imgname in os.listdir(self.split_path):
            if imgname.endswith('.jpg'):
                imgpath = os.path.join(self.split_path, imgname)
                self.imgpaths.append(imgpath)
        logger(f"Loaded {len(self.imgpaths)} valid images from the tree of {self.split_path}")
    
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
        return len(self.imgpaths)

def process_batch(base: DatasetWrapper, start: int, end: int, prefix: str, processor_id: int, seed: int, min_size: int = 28):
    """
    Process the dataset `base`. Periodically write partial results to disk
    (jigsaw/{split_name}_part{idx}.parquet) to avoid keeping everything in memory.
    At the end, load the parts and concatenate into jigsaw/{split_name}.parquet.
    Returns the final concatenated Dataset.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []
    logger(f"Starting batch from {start} to {end}")

    progbar = tqdm.trange(start, end, desc=f'Processing {prefix}')
    for j in progbar:
        order = list(range(JIGSAW_SIZE * JIGSAW_SIZE))
        random.shuffle(order)

        orig: Image.Image = base.getimage(j)
        assert orig.size == (orig.width, orig.height)
        patch_size_0 = orig.size[0] // JIGSAW_SIZE
        patch_size_1 = orig.size[1] // JIGSAW_SIZE
        if patch_size_0 < min_size or patch_size_1 < min_size:
            continue

        patches = []
        for j in range(JIGSAW_SIZE):
            for i in range(JIGSAW_SIZE):
                end0 = (i + 1) * patch_size_0
                end1 = (j + 1) * patch_size_1
                patch = orig.crop((i * patch_size_0, j * patch_size_1, end0, end1))
                patches.append(patch)
        
        jigpatches = [patches[i] for i in order]

        # 计算逆序列：当前第i个patch应该放在哪个位置
        inverse_order = [0] * len(order)
        for i, orig_pos in enumerate(order):
            inverse_order[orig_pos] = i + 1  # +1因为位置从1开始编号
        orderstr = ','.join(map(str, inverse_order))
        answer = orderstr

        new_datum = {
            **base.getextra(j),
            'images': jigpatches,
            'query': QUERY_TEMPLATE,
            'answer': answer
        }

        results.append(new_datum)

    logger(f"Processor {processor_id}: Start saving {len(results)} items")
    save_dir = f"datasets/caparena_JigsawQA_{JIGSAW_SIZE}x{JIGSAW_SIZE}"
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
    if 'test_04445.jpg' not in os.listdir(data_base_dir):
        data_base_dir = os.path.join(data_base_dir, 'caparena_auto_docci_600')
    yield data_base_dir, 'caparena'

if __name__ == '__main__':
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_base_dir = 'CapArena-master/data/caparena_auto_docci_600'
    logger(f'!! Use {data_base_dir} as data dir')
    logger(f'!! Using jigsaw size: {JIGSAW_SIZE}x{JIGSAW_SIZE}')

    for split_path, prefix in splits_generator(data_base_dir):
        logger(f'Processing split: {prefix} from path {split_path}')
        try:
            process_split(split_path, prefix, num_processors=1, seed=seed)
        except KeyboardInterrupt:
            break
        except:
            logger(f'Error processing {prefix} from path {split_path}', exc_info=True)
        logger(f'Completed processing {prefix} from path {split_path}')
