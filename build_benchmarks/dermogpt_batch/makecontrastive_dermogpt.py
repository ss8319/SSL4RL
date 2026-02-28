import os
import json
import tqdm
import torch
import random
import gc
from PIL import Image
from collections import defaultdict

IMAGE_ROOT = "/home/ssim0070/ub62_scratch/ssim0070/Derm-All/"
SAVE_BASE_DIR = "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/datasets/dermogpt/contrastive_json/"

def augment(img) -> Image.Image:
    # borrowed from solo-learn
    import omegaconf

    from typing import Callable, List, Optional, Sequence, Type, Union
    from PIL import Image, ImageFilter, ImageOps
    from torchvision import transforms

    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    class GaussianBlur:
        def __init__(self, sigma: Sequence[float] = None):
            if sigma is None:
                sigma = [0.1, 2.0]
            self.sigma = sigma
        def __call__(self, img: Image) -> Image:
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            return img

    class Solarization:
        def __call__(self, img: Image) -> Image:
            return ImageOps.solarize(img)

    class Equalization:
        def __call__(self, img: Image) -> Image:
            return ImageOps.equalize(img)

    class NCropAugmentation:
        def __init__(self, transform: Callable, num_crops: int):
            self.transform = transform
            self.num_crops = num_crops
        def __call__(self, x: Image) -> List:
            return [self.transform(x) for _ in range(self.num_crops)]
        def __repr__(self) -> str:
            return f"{self.num_crops} x [{self.transform}]"

    class FullTransformPipeline:
        def __init__(self, transforms: Callable) -> None:
            self.transforms = transforms
        def __call__(self, x: Image) -> List:
            out = []
            for transform in self.transforms:
                out.extend(transform(x))
            return out
        def __repr__(self) -> str:
            return "\n".join(str(transform) for transform in self.transforms)

    def build_transform_pipeline(dataset, cfg):
        MEANS_N_STD = {
            "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        }
        mean, std = MEANS_N_STD.get(
            dataset, (cfg.get("mean", IMAGENET_DEFAULT_MEAN), cfg.get("std", IMAGENET_DEFAULT_STD))
        )
        augmentations = []
        if cfg.rrc.enabled:
            augmentations.append(
                transforms.RandomResizedCrop(
                    cfg.crop_size,
                    scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )
        else:
            augmentations.append(
                transforms.Resize(
                    cfg.crop_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )
        if cfg.color_jitter.prob:
            augmentations.append(
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            cfg.color_jitter.brightness,
                            cfg.color_jitter.contrast,
                            cfg.color_jitter.saturation,
                            cfg.color_jitter.hue,
                        )
                    ],
                    p=cfg.color_jitter.prob,
                ),
            )
        if cfg.grayscale.prob:
            augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))
        if cfg.gaussian_blur.prob:
            augmentations.append(transforms.RandomApply([GaussianBlur()], p=cfg.gaussian_blur.prob))
        if cfg.solarization.prob:
            augmentations.append(transforms.RandomApply([Solarization()], p=cfg.solarization.prob))
        if cfg.equalization.prob:
            augmentations.append(transforms.RandomApply([Equalization()], p=cfg.equalization.prob))
        if cfg.horizontal_flip.prob:
            augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))
        augmentations = transforms.Compose(augmentations)
        return augmentations

    cfg, = omegaconf.OmegaConf.load('/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/build_benchmarks/augcfg.yaml')
    trans = build_transform_pipeline('imagenet', cfg)
    try:
        result = trans(img)
    except RuntimeError:
        result = None
    return result

QUERY_TEMPLATE = '''<image><image>

The provided images are augmentations of the same original image or two different images.
The augmentations may include random cropping, color adjustments, grayscale conversion, blurring, and flipping.
Please think step-by-step and determine if these two images are possibly derived from the same original image.
If the provided images are from the same original image, respond with "positive"; if they correspond to different original images, respond with "negative".

Your answer should strictly follow this format:

<think>your step-by-step reasoning here</think> <answer>positive/negative</answer>'''

def save_sample_contrastive(save_dir, sample_id, im1, im2):
    images_abs_dir = os.path.join(save_dir, "images")
    os.makedirs(images_abs_dir, exist_ok=True)
    
    path1 = os.path.join("images", f"{sample_id}_v1.png")
    path2 = os.path.join("images", f"{sample_id}_v2.png")
    
    im1.save(os.path.join(save_dir, path1))
    im2.save(os.path.join(save_dir, path2))
    
    return [path1, path2]

def process_positive_sample(datum, save_dir, image_root, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    img_path = os.path.join(image_root, datum['image'])
    try:
        img = Image.open(img_path).convert('RGB')
        im1 = augment(img)
        im2 = augment(img)
        
        if im1 is None or im2 is None:
            return None
        
        sample_id = f"{datum['id']}_pos"
        img_paths = save_sample_contrastive(save_dir, sample_id, im1, im2)
        
        return {
            "id": sample_id,
            "image": img_paths,
            "source": datum.get('source', 'unknown'),
            "conversations": [
                {
                    "from": "human",
                    "value": QUERY_TEMPLATE
                },
                {
                    "from": "gpt",
                    "value": "positive"
                }
            ]
        }
    except Exception as e:
        return None

def process_negative_pair(args):
    i, j, data, save_dir, image_root, seed = args
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        img1 = Image.open(os.path.join(image_root, data[i]['image'])).convert('RGB')
        img2 = Image.open(os.path.join(image_root, data[j]['image'])).convert('RGB')
        im1 = augment(img1)
        im2 = augment(img2)
        
        if im1 is None or im2 is None:
            return None
            
        sample_id = f"neg_{i}_{j}"
        img_paths = save_sample_contrastive(save_dir, sample_id, im1, im2)
        
        return {
            "id": sample_id,
            "image": img_paths,
            "source": "negative_pair",
            "conversations": [
                {
                    "from": "human",
                    "value": QUERY_TEMPLATE
                },
                {
                    "from": "gpt",
                    "value": "negative"
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
        
    dlen = len(data)
    os.makedirs(save_dir, exist_ok=True)
    
    num_workers = min(32, multiprocessing.cpu_count())

    print(f"Generating positive pairs for {len(data)} images using {num_workers} workers...")
    pos_tasks = [(datum, save_dir, IMAGE_ROOT, 42 + idx) for idx, datum in enumerate(data)]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        pos_results = list(tqdm.tqdm(executor.map(process_positive_wrapper, pos_tasks), total=len(pos_tasks)))

    dataset_json = [res for res in pos_results if res is not None]
    total_pos = len(dataset_json)
    
    print(f"Generating negative pairs to match {total_pos} positive samples...")
    neg_tasks = []
    # Simplified negative sampling for parallelization
    # We'll generate a list of random pairs beforehand
    neg_attempts = 0
    while len(neg_tasks) < total_pos and neg_attempts < total_pos * 2:
        i = random.randint(0, dlen - 1)
        j = random.randint(0, dlen - 1)
        if i == j: continue
        neg_tasks.append((i, j, data, save_dir, IMAGE_ROOT, 1000 + len(neg_tasks)))
        neg_attempts += 1

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        neg_results = list(tqdm.tqdm(executor.map(process_negative_pair, neg_tasks), total=len(neg_tasks)))

    dataset_json.extend([res for res in neg_results if res is not None])
    total_neg = len([res for res in neg_results if res is not None])

    output_json_path = os.path.join(save_dir, "dataset.json")
    with open(output_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Dataset generated at {save_dir}")
    print(f"Total samples: {len(dataset_json)} ({total_pos} pos, {total_neg} neg)")

def process_positive_wrapper(args):
    return process_positive_sample(*args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    
    seed = 0
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
