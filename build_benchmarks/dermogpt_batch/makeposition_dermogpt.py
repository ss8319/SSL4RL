import os
import json
import tqdm
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_ROOT = "/home/ssim0070/ub62_scratch/ssim0070/Derm-All/"
SAVE_BASE_DIR = "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/datasets/dermogpt/position_json/"

DO_AUGMENT = True
CROP_MIN_PERCENT = 0.2
CROP_MAX_PERCENT = 0.8

QUERY_TEMPLATE = '''<image><image>

The second image is an augmented version of a crop in the first image. The augmentations may include grayscale, color jitter, solarization, etc. Please determine which part of the first image the second image is from. The answer is strictly limited to four cases: top-left, top-right, bottom-left and bottom-right.

Your answer should strictly follow this format:

<think>your step-by-step reasoning here</think> <answer>top-left/top-right/bottom-left/bottom-right</answer>'''

def augment(img) -> Image.Image:
    if not DO_AUGMENT:
        return img
    import omegaconf
    from PIL import ImageFilter, ImageOps
    class GaussianBlur:
        def __init__(self, sigma=None):
            if sigma is None: sigma = [0.1, 2.0]
            self.sigma = sigma
        def __call__(self, img):
            sigma = random.uniform(self.sigma[0], self.sigma[1])
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    class Solarization:
        def __call__(self, img): return ImageOps.solarize(img)
    class Equalization:
        def __call__(self, img): return ImageOps.equalize(img)
    def build_transform_pipeline(cfg):
        augmentations = []
        if cfg.color_jitter.prob:
            augmentations.append(transforms.RandomApply([transforms.ColorJitter(cfg.color_jitter.brightness, cfg.color_jitter.contrast, cfg.color_jitter.saturation, cfg.color_jitter.hue)], p=cfg.color_jitter.prob))
        if cfg.grayscale.prob:
            augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))
        if cfg.gaussian_blur.prob:
            augmentations.append(transforms.RandomApply([GaussianBlur()], p=cfg.gaussian_blur.prob))
        if cfg.solarization.prob:
            augmentations.append(transforms.RandomApply([Solarization()], p=cfg.solarization.prob))
        if cfg.equalization.prob:
            augmentations.append(transforms.RandomApply([Equalization()], p=cfg.equalization.prob))
        return transforms.Compose(augmentations)
    cfg_path = '/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/build_benchmarks/augcfg.yaml'
    cfg, = omegaconf.OmegaConf.load(cfg_path)
    trans = build_transform_pipeline(cfg)
    try:
        return trans(img)
    except:
        return img

def save_sample_position(save_dir, sample_id, im1, im2):
    images_abs_dir = os.path.join(save_dir, "images")
    os.makedirs(images_abs_dir, exist_ok=True)
    
    path1 = os.path.join("images", f"{sample_id}_orig.png")
    path2 = os.path.join("images", f"{sample_id}_aug.png")
    
    im1.save(os.path.join(save_dir, path1))
    im2.save(os.path.join(save_dir, path2))
    
    return [path1, path2]

def process_sample(datum, save_dir, image_root, orientations_list, min_size, crop_min, crop_max, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    img_path = os.path.join(image_root, datum['image'])
    try:
        img = Image.open(img_path).convert('RGB')
        if img.width < min_size * 2 or img.height < min_size * 2:
            return None

        h, w = img.height, img.width
        take_part_func = {
            'top-left': lambda x: x.crop((0, 0, w//2, h//2)),
            'top-right': lambda x: x.crop((w//2, 0, w, h//2)),
            'bottom-left': lambda x: x.crop((0, h//2, w//2, h)),
            'bottom-right': lambda x: x.crop((w//2, h//2, w, h)),
        }

        ori = random.choice(orientations_list)
        part = take_part_func[ori](img)
        ph, pw = part.height, part.width

        minhcropsize = max(min_size, int(ph * crop_min))
        maxhcropsize = int(ph * crop_max)
        minwcropsize = max(min_size, int(pw * crop_min))
        maxwcropsize = int(pw * crop_max)

        if maxhcropsize < minhcropsize or maxwcropsize < minwcropsize:
            return None

        crop_h = random.randint(minhcropsize, maxhcropsize)
        crop_w = random.randint(minwcropsize, maxwcropsize)
        top = random.randint(0, ph - crop_h)
        left = random.randint(0, pw - crop_w)

        piece = part.crop((left, top, left + crop_w, top + crop_h))
        aug = augment(piece)

        sample_id = f"{datum['id']}_position"
        img_paths = save_sample_position(save_dir, sample_id, img, aug)

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
                    'value': ori
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
    orientations_list = ['top-left', 'top-right', 'bottom-left', 'bottom-right']

    print(f"Generating Position tasks for {len(data)} images from {os.path.basename(input_path)} using {num_workers} workers...")
    
    tasks = []
    for idx, datum in enumerate(data):
        sample_seed = 42 + idx 
        tasks.append((datum, save_dir, IMAGE_ROOT, orientations_list, min_size, CROP_MIN_PERCENT, CROP_MAX_PERCENT, sample_seed))

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
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    splits = {
        "train": "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/data/dermogpt/train/dermogpt_mcqa_train_8000.json",
        "valid": "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/data/dermogpt/valid/dermogpt_mcqa.json",
        "test": "/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/data/dermogpt/test/dermogpt_mcqa.json"
    }
    
    for split_name, input_path in splits.items():
        save_dir = os.path.join(SAVE_BASE_DIR, split_name)
        process_dataset(input_path, save_dir, limit=args.limit)
