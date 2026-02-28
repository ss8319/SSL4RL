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

def save_sample_pair(save_dir, sample_id, im1, im2):
    sample_rel_dir = os.path.join("samples", sample_id)
    sample_abs_dir = os.path.join(save_dir, sample_rel_dir)
    os.makedirs(sample_abs_dir, exist_ok=True)
    
    path1 = os.path.join(sample_rel_dir, "original.png")
    path2 = os.path.join(sample_rel_dir, "cropped_aug.png")
    
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

    print(f"Generating Position tasks for {len(data)} images from {os.path.basename(input_path)}...")
    for datum in tqdm.tqdm(data):
        img_path = os.path.join(IMAGE_ROOT, datum['image'])
        try:
            img = Image.open(img_path).convert('RGB')
            if img.width < min_size * 2 or img.height < min_size * 2:
                continue

            h, w = img.height, img.width
            take_part_func = {
                'top-left': lambda x: x.crop((0, 0, w//2, h//2)),
                'top-right': lambda x: x.crop((w//2, 0, w, h//2)),
                'bottom-left': lambda x: x.crop((0, h//2, w//2, h)),
                'bottom-right': lambda x: x.crop((w//2, h//2, w, h)),
            }

            orientations = list(take_part_func.keys())
            ori = random.choice(orientations)
            part = take_part_func[ori](img)
            ph, pw = part.height, part.width

            minhcropsize = max(min_size, int(ph * CROP_MIN_PERCENT))
            maxhcropsize = int(ph * CROP_MAX_PERCENT)
            minwcropsize = max(min_size, int(pw * CROP_MIN_PERCENT))
            maxwcropsize = int(pw * CROP_MAX_PERCENT)

            if maxhcropsize < minhcropsize or maxwcropsize < minwcropsize:
                continue

            crop_h = random.randint(minhcropsize, maxhcropsize)
            crop_w = random.randint(minwcropsize, maxwcropsize)
            top = random.randint(0, ph - crop_h)
            left = random.randint(0, pw - crop_w)

            piece = part.crop((left, top, left + crop_w, top + crop_h))
            aug = augment(piece)

            sample_id = f"{datum['id']}_position"
            img_paths = save_sample_pair(save_dir, sample_id, img, aug)

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
                        'value': ori
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
