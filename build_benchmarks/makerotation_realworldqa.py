import os
import math
import tqdm
import torch
import random
import datasets
import numpy as np
import pandas as pd
import base64
import io

from PIL import Image
from torchvision import transforms

GRANUALITY = 90 # degrees
MAX_SIDE = 512  # 控制最长边，避免训练时OOM

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



def process_split(base: datasets.Dataset, prefix: str, seed: int, min_size: int = 28):
    total = len(base)

    results = []

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for j in range(total):
        datum = base[j]

        img_data = datum['image']
        # base64编码的字符串
        img_bytes = base64.b64decode(img_data)
        # 从字节流创建PIL Image
        img: Image.Image = Image.open(io.BytesIO(img_bytes))

        possible_angles = list(range(0, 360, GRANUALITY))
        angle = random.choice(possible_angles)

        # 对于90度的倍数旋转，直接使用旋转后的图片，不进行任何resize
        if angle % 90 == 0:
            rotated_img: Image.Image = img.rotate(angle, expand=False)
            # 不进行任何resize操作，保持旋转后的原始尺寸
        else:
            # 对于非90度倍数的旋转，使用原来的方法但限制最大尺寸
            rotated_img: Image.Image = img.rotate(angle, expand=True)
            wr, hr = rotatedRectWithMaxArea(img.width, img.height, math.radians(angle))
            rotated_img = rotated_img.crop((
                (rotated_img.width - wr) / 2,
                (rotated_img.height - hr) / 2,
                (rotated_img.width + wr) / 2,
                (rotated_img.height + hr) / 2,
            ))

        # 全局最长边限制，进一步避免过大图片
        if max(rotated_img.width, rotated_img.height) > MAX_SIDE:
            rotated_img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

        if rotated_img.width < min_size or rotated_img.height < min_size:
            continue

        newdatum = {
            'index': datum['index'],
            'height': rotated_img.height,
            'width': rotated_img.width,
            'images': [img, rotated_img],
            'query': QUERY_TEMPLATE,
            'answer': f'{angle}'
        }

        results.append(newdatum)

    # Create save directory with fallback
    save_dir = f'datasets/RealWorldQA_RotationQA_{GRANUALITY}degree'
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
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dataset_path = 'datasets/RealWorldQA.tsv'
    df = pd.read_csv(dataset_path, sep='\t')
    data = df[['index', 'image']].to_dict('records')
    print(f"Successfully loaded seedbench with {len(df)} items")
        
    process_split(data, "train", seed)
    print(f"Completed processing")