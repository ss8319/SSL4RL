import os
import tqdm
import torch
import random
import datasets
import numpy as np
import logging
import sys
import pandas as pd
import io
import base64

from PIL import Image
from torchvision import transforms

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('build_crop.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 配置选项
DO_AUGMENT = False
aug_suffix = '_augmented' if DO_AUGMENT else ''

CROP_MIN_PERCENT = 0.2
CROP_MAX_PERCENT = 0.8

# 批处理配置
MAX_BATCHES = 10  # 最大批次数

QUERY_TEMPLATE = '''<image><image>

The second image in an augmented version of a crop in the first image. The augmentations may include grayscale, color jitter, solarization, etc. Please determine which part of the first image the second image is from. The answer is strictly limited to four cases: top-left, top-right, bottom-left and bottom-right.

Your answer should strictly follow this format:

<think>your step-by-step reasoning here</think> <answer>top-left/top-right/bottom-left/bottom-right</answer>'''

def augment(img) -> Image.Image:
    if not DO_AUGMENT:
        return img

    # borrowed from solo-learn
    import omegaconf

    from typing import Callable, List, Optional, Sequence, Type, Union
    from PIL import Image, ImageFilter, ImageOps

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

        # !!! We disable cropping, resizing and normalizing for the piece localization task !!!

        augmentations = []
        # if cfg.rrc.enabled:
        #     augmentations.append(
        #         transforms.RandomResizedCrop(
        #             cfg.crop_size,
        #             scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
        #             interpolation=transforms.InterpolationMode.BICUBIC,
        #         ),
        #     )
        # else:
        #     augmentations.append(
        #         transforms.Resize(
        #             cfg.crop_size,
        #             interpolation=transforms.InterpolationMode.BICUBIC,
        #         ),
        #     )

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

        # if cfg.horizontal_flip.prob:
        #     augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))

        # augmentations.append(transforms.ToTensor())
        # augmentations.append(transforms.Normalize(mean=mean, std=std))

        augmentations = transforms.Compose(augmentations)
        return augmentations
    
    cfg, = omegaconf.OmegaConf.load('augcfg.yaml')
    trans = build_transform_pipeline('imagenet', cfg)
    try:
        result = trans(img)
    except RuntimeError:
        result = None
    
    return result



def process_split(base: datasets.Dataset, prefix: str, min_size: int = 28):
    total = len(base)
    logger.info(f"Processing {prefix} split: {total} items")

    results = []
    for j in range(total):
        datum = base[j]

        img_data = datum['image']
        # base64编码的字符串
        img_bytes = base64.b64decode(img_data)
        # 从字节流创建PIL Image
        img: Image.Image = Image.open(io.BytesIO(img_bytes))

        if img.width < min_size * 2 or img.height < min_size * 2:
            continue

        imarr = np.asarray(img)
        h, w = img.height, img.width

        take_part_func = {
            'top-left': lambda x: x[:h//2, :w//2],
            'top-right': lambda x: x[:h//2, w//2:],
            'bottom-left': lambda x: x[h//2:, :w//2],
            'bottom-right': lambda x: x[h//2:, w//2:],
        }

        orientations = list(take_part_func.keys())
        ori = random.choice(orientations)
        part = take_part_func[ori](imarr) # quarter-image
        ph, pw = part.shape[:2]

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

        piecearr = part[top:top + crop_h, left:left + crop_w]

        piece = Image.fromarray(piecearr)
        aug = augment(piece)
        
        assert aug.width >= min_size and aug.height >= min_size

        newdatum = {
            'index': datum['index'],
            'category': datum['category'],
            'height1': img.height,
            'width1': img.width,
            'height2': aug.height,
            'width2': aug.width,
            'images': [img, aug],
            'query': QUERY_TEMPLATE,
            'answer': ori
        }
        
        results.append(newdatum)


    # Create save directory with fallback
    save_dir = f'datasets/SEEDBench_PositionQA{aug_suffix}'
    os.makedirs(save_dir, exist_ok=True)

    parquet_name = f'{prefix}.parquet'
    save_path = os.path.join(save_dir, parquet_name)
    
    if results:
        new_ds = datasets.Dataset.from_list(results)
        new_ds.to_parquet(save_path)
        logger.info(f"Saved {len(results)} items to {save_path}")
    else:
        logger.warning(f"No results to save")


if __name__ == '__main__':
    logger.info("Starting build_crop.py script")
    
    if DO_AUGMENT:
        logger.info(f'{DO_AUGMENT=}: Using augmentation.')
    else:
        logger.info(f'{DO_AUGMENT=}: Not using augmentation.')
    logger.info(f'The cropped piece takes {CROP_MIN_PERCENT} to {CROP_MAX_PERCENT} of the quarter image size.')
    logger.info(f'Using batch processing with max {MAX_BATCHES} batches')

    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Set random seed to {seed}")
        
    dataset_path = 'datasets/SEED-Bench/SEEDBench_IMG.tsv'
    df = pd.read_csv(dataset_path, sep='\t')
    data = df[['index', 'image', 'category']].to_dict('records')
    print(f"Successfully loaded seedbench with {len(df)} items")
        
    process_split(data, "train")
    logger.info(f"Completed processing")