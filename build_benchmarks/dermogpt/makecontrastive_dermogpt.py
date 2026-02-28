import os
import json
import tqdm
import torch
import random
import gc
from PIL import Image

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
            """Gaussian blur as a callable object.

            Args:
                sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                    Defaults to [0.1, 2.0].
            """

            if sigma is None:
                sigma = [0.1, 2.0]

            self.sigma = sigma

        def __call__(self, img: Image) -> Image:
            """Applies gaussian blur to an input image.

            Args:
                img (Image): an image in the PIL.Image format.

            Returns:
                Image: blurred image.
            """

            sigma = random.uniform(self.sigma[0], self.sigma[1])
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
            return img


    class Solarization:
        """Solarization as a callable object."""

        def __call__(self, img: Image) -> Image:
            """Applies solarization to an input image.

            Args:
                img (Image): an image in the PIL.Image format.

            Returns:
                Image: solarized image.
            """

            return ImageOps.solarize(img)


    class Equalization:
        def __call__(self, img: Image) -> Image:
            return ImageOps.equalize(img)


    class NCropAugmentation:
        def __init__(self, transform: Callable, num_crops: int):
            """Creates a pipeline that apply a transformation pipeline multiple times.

            Args:
                transform (Callable): transformation pipeline.
                num_crops (int): number of crops to create from the transformation pipeline.
            """

            self.transform = transform
            self.num_crops = num_crops

        def __call__(self, x: Image) -> List:
            """Applies transforms n times to generate n crops.

            Args:
                x (Image): an image in the PIL.Image format.

            Returns:
                List[torch.Tensor]: an image in the tensor format.
            """

            return [self.transform(x) for _ in range(self.num_crops)]

        def __repr__(self) -> str:
            return f"{self.num_crops} x [{self.transform}]"


    class FullTransformPipeline:
        def __init__(self, transforms: Callable) -> None:
            self.transforms = transforms

        def __call__(self, x: Image) -> List:
            """Applies transforms n times to generate n crops.

            Args:
                x (Image): an image in the PIL.Image format.

            Returns:
                List[torch.Tensor]: an image in the tensor format.
            """

            out = []
            for transform in self.transforms:
                out.extend(transform(x))
            return out

        def __repr__(self) -> str:
            return "\n".join(str(transform) for transform in self.transforms)


    def build_transform_pipeline(dataset, cfg):
        """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
        The node needs to be in the following format:
            crop_size: int
            [OPTIONAL] mean: float
            [OPTIONAL] std: float
            rrc:
                enabled: bool
                crop_min_scale: float
                crop_max_scale: float
            color_jitter:
                prob: float
                brightness: float
                contrast: float
                saturation: float
                hue: float
            grayscale:
                prob: float
            gaussian_blur:
                prob: float
            solarization:
                prob: float
            equalization:
                prob: float
            horizontal_flip:
                prob: float
        """

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

        # augmentations.append(transforms.ToTensor())
        # augmentations.append(transforms.Normalize(mean=mean, std=std))

        augmentations = transforms.Compose(augmentations)
        return augmentations

    cfg, = omegaconf.OmegaConf.load('/home/ssim0070/ub62_scratch/ssim0070/SSL4RL/build_benchmarks/augcfg.yaml')
    trans = build_transform_pipeline('imagenet', cfg)
    try:
        result = trans(img)
    except RuntimeError:
        result = None

    return result

def repetive_augment(img: Image.Image, num: int = 5, thresh: int = 28):
    im = None
    while im is None or (im.width < thresh or im.height < thresh):
        im = augment(img)
        num -= 1
        if num <= 0:
            break
    return im

QUERY_TEMPLATE = '''<image><image>

The provided images are augmentations of the same original image or two different images.
The augmentations may include random cropping, color adjustments, grayscale conversion, blurring, and flipping.
Please think step-by-step and determine if these two images are possibly derived from the same original image.
If the provided images are from the same original image, respond with "positive"; if they correspond to different original images, respond with "negative".

Your answer should strictly follow this format:

<think>your step-by-step reasoning here</think> <answer>positive/negative</answer>'''

def save_sample(save_dir, sample_id, im1, im2):
    sample_rel_dir = os.path.join("samples", sample_id)
    sample_abs_dir = os.path.join(save_dir, sample_rel_dir)
    os.makedirs(sample_abs_dir, exist_ok=True)
    
    path1 = os.path.join(sample_rel_dir, "view1.png")
    path2 = os.path.join(sample_rel_dir, "view2.png")
    
    im1.save(os.path.join(save_dir, path1))
    im2.save(os.path.join(save_dir, path2))
    
    return [path1, path2]

def process_dataset(input_path, save_dir, max_samples=None):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        
    dlen = len(data)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "samples"), exist_ok=True)
    
    dataset_json = []
    total_pos = 0

    # Positive pairs
    print(f"Generating positive pairs for {len(data)} images from {os.path.basename(input_path)}...")
    for datum in tqdm.tqdm(data):
        img_path = os.path.join(IMAGE_ROOT, datum['image'])
        try:
            img = Image.open(img_path).convert('RGB')
            im1 = augment(img)
            im2 = augment(img)
            
            if im1 is None or im2 is None:
                continue
            
            sample_id = f"{datum['id']}_pos"
            img_paths = save_sample(save_dir, sample_id, im1, im2)
            
            # The GT answer should be just the label for the reward function to verify
            gt_answer = "positive"
            
            dataset_json.append({
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
                        "value": gt_answer
                    }
                ]
            })
            total_pos += 1
        except Exception as e:
            continue
            
    # Negative pairs
    print(f"Generating negative pairs to match {total_pos} positive samples...")
    total_neg = 0
    while total_neg < total_pos:
        i = random.randint(0, dlen - 1)
        j = random.randint(0, dlen - 1)
        if i == j: continue
        
        try:
            img1 = Image.open(os.path.join(IMAGE_ROOT, data[i]['image'])).convert('RGB')
            img2 = Image.open(os.path.join(IMAGE_ROOT, data[j]['image'])).convert('RGB')
            im1 = augment(img1)
            im2 = augment(img2)
            
            if im1 is None or im2 is None:
                continue
                
            sample_id = f"neg_{i}_{j}"
            img_paths = save_sample(save_dir, sample_id, im1, im2)
            
            gt_answer = "negative"
            
            dataset_json.append({
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
                        "value": gt_answer
                    }
                ]
            })
            total_neg += 1
        except Exception as e:
            continue

    output_json_path = os.path.join(save_dir, "dataset.json")
    with open(output_json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Dataset generated at {save_dir}")
    print(f"Total samples: {len(dataset_json)} ({total_pos} pos, {total_neg} neg)")

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    
    splits = {
        "train": "SSL4RL/data/dermogpt/train/dermogpt_mcqa.json",
        "valid": "SSL4RL/data/dermogpt/valid/dermogpt_mcqa.json",
        "test": "SSL4RL/data/dermogpt/test/dermogpt_mcqa.json"
    }
    
    for split_name, input_path in splits.items():
        save_dir = os.path.join(SAVE_BASE_DIR, split_name)
        process_dataset(input_path, save_dir, max_samples=5)
