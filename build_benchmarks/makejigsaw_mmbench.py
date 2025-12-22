import os
import tqdm
import torch
import random
import datasets
import gc

from PIL import Image

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


def process_batch(base: datasets.Dataset, start: int, end: int, prefix: str, processor_id: int, min_size: int = 28):
    """
    Process the dataset `base`. Periodically write partial results to disk
    (jigsaw/{split_name}_part{idx}.parquet) to avoid keeping everything in memory.
    At the end, load the parts and concatenate into jigsaw/{split_name}.parquet.
    Returns the final concatenated Dataset.
    """
    results = []
    print(f"Processor {processor_id}: Starting batch from {start} to {end}")

    for j in range(start, end):
        datum = base[j]
        order = list(range(JIGSAW_SIZE * JIGSAW_SIZE))
        random.shuffle(order)

        orig: Image.Image = datum['image']
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
            'index': datum['index'],
            'source': datum['source'],
            'split': datum['split'],
            'images': jigpatches,
            'query': QUERY_TEMPLATE,
            'answer': answer
        }

        results.append(new_datum)

    print(f"Processor {processor_id}: Start saving {len(results)} items")
    save_dir = f"datasets/MMBench_JigsawQA_{JIGSAW_SIZE}x{JIGSAW_SIZE}"
    os.makedirs(save_dir, exist_ok=True)

    parquet_name = f'{prefix}_{processor_id:05d}.parquet'
    save_path = os.path.join(save_dir, parquet_name)
    
    if results:
        new_ds = datasets.Dataset.from_list(results)
        new_ds.to_parquet(save_path)
        print(f"Processor {processor_id}: Saved {len(results)} items to {save_path}")
    else:
        print.warning(f"Processor {processor_id}: No results to save")
   


def process_split(base: datasets.Dataset, prefix: str, num_processors: int = None):
    total = len(base)
    batch_size = (total + num_processors - 1) // num_processors
    print(f"Processing {prefix} split: {total} items with {num_processors} batches, batch size: {batch_size}")

    jobs = []
    for i in range(num_processors):
        start = i * batch_size
        end = min(total, (i + 1) * batch_size)
        if start >= end:
            continue
        jobs.append((base, start, end, base.split, i))
    print(f"Created {len(jobs)} batches for {prefix} split")

    results = []
    print("Starting batch processing...")
    for i, (base, start, end, split_name, processor_id) in enumerate(jobs):
        print(f"Processing batch {i+1}/{len(jobs)}: items {start} to {end}")
        result = process_batch(base, start, end, split_name, processor_id)
        results.append(result)
        print(f"Batch {i+1} completed: {result}")

    print(f"Successfully completed processing {prefix} split")
    return results


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    for split in ['test', 'validation']:
        print(f"Processing {split} split...")
        
        dataset_path = 'datasets/MMBench/en'
        print(f"Loading dataset from: {dataset_path}")
        
        base = datasets.load_dataset(dataset_path, split=split)
        print(f"Successfully loaded {split} split with {len(base)} items")
        
        process_split(base, split, num_processors=1)
        print(f"Completed processing {split} split")