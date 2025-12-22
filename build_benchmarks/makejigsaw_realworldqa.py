import os
import tqdm
import torch
import random
import datasets
import gc
import pandas as pd
import base64
import os
import io

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


def process_split(base: datasets.Dataset, prefix: str, min_size: int = 28):
    total = len(base)
    results = []
    print(f"Starting process {prefix}")

    for j in range(total):
        datum = base[j]
        order = list(range(JIGSAW_SIZE * JIGSAW_SIZE))
        random.shuffle(order)

        img_data = datum['image']
        # base64编码的字符串
        img_bytes = base64.b64decode(img_data)
        # 从字节流创建PIL Image
        orig: Image.Image = Image.open(io.BytesIO(img_bytes))

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

        newdatum = {
            'index': datum['index'],
            'images': jigpatches,
            'query': QUERY_TEMPLATE,
            'answer': answer
        }

        results.append(newdatum)

    print(f"Start saving {len(results)} items")
    save_dir = f"datasets/RealWorldQA_JigsawQA_{JIGSAW_SIZE}x{JIGSAW_SIZE}"
    os.makedirs(save_dir, exist_ok=True)

    parquet_name = f'{prefix}.parquet'
    save_path = os.path.join(save_dir, parquet_name)
    
    if results:
        new_ds = datasets.Dataset.from_list(results)
        new_ds.to_parquet(save_path)
        print(f"Saved {len(results)} items to {save_path}")
    else:
        print.warning(f"No results to save")


if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = 'datasets/RealWorldQA.tsv'
    df = pd.read_csv(dataset_path, sep='\t')
    data = df[['index', 'image']].to_dict('records')
    print(f"Successfully loaded realworldqa with {len(df)} items")

    process_split(data, prefix="train")
    print(f"Completed processing realworldqa")