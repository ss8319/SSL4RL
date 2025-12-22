import os
import math
import tqdm
import torch
import random
import datasets
import gc
from PIL import Image
from torchvision.utils import save_image

COLUMN_NAMES = ['filename', 'height', 'width', 'label', 'query', 'answer', 'images']

import logging, sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__).info

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

    cfg, = omegaconf.OmegaConf.load('augcfg.yaml')
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



QUERY_TEMPLATE_BOX = """<image><image>

The provided images are augmentations of the same original image or two different images.
The augmentations may include random cropping, color adjustments, grayscale conversion, blurring, and flipping.
Please think step-by-step and determine if these two images are possibly derived from the same original image.
If the provided images are from the same original image, respond with "positive"; if they correspond to different original images, respond with "negative".
    
Present the final answer in \\boxed{{}} format, like this: $\\boxed{{ANSWER}}$, where ANSWER is positive or negative..

Think carefully and break down the problem step by step.
"""


def process_split(base: DatasetWrapper):
    dlen = len(base)

    # 使用生成器和批处理来减少内存占用
    def generate_pairs():
        # 正样本对
        progbar = tqdm.tqdm(range(dlen), desc='Positive Pairs')
        for dtmid in progbar:
            img = base.getimage(dtmid)
            try:
                im1 = augment(img)
                im2 = augment(img)

                if im1 is None or im2 is None:
                    continue

                pospair = {
                    **base.getextra(dtmid),
                    'height1': im1.height,
                    'width1': im1.width,
                    'height2': im2.height,
                    'width2': im2.width,
                    'images': [im1, im2],
                    'query': QUERY_TEMPLATE,
                    'answer': 'positive'
                }
                yield pospair

                # 强制垃圾回收
                del im1, im2, img
                gc.collect()

            except Exception as e:
                import traceback; traceback.print_exc()
                continue

        # 负样本对
        pool = set()
        neg_count = 0
        target_neg_samples = dtmid + 1  # 与正样本数量相等

        progbar = tqdm.tqdm(total=target_neg_samples, desc='Negative Pairs')
        while neg_count < target_neg_samples:
            i = random.randint(0, dlen - 1)
            j = random.randint(0, dlen - 1)
            if (i, j) in pool or i >= j:
                continue
            pool.add((i, j))

            try:
                img1 = base.getimage(i)
                img2 = base.getimage(j)

                im1 = augment(img1)
                im2 = augment(img2)

                if im1 is None or im2 is None:
                    continue

                negpair = {
                    'height1': im1.height,
                    'width1': im1.width,
                    'height2': im2.height,
                    'width2': im2.width,
                    'images': [im1, im2],
                    'query': QUERY_TEMPLATE,
                    'answer': 'negative'
                }
                yield negpair
                neg_count += 1
                progbar.update(1)

                # 强制垃圾回收
                del im1, im2, img1, img2
                gc.collect()

            except Exception as e:
                import traceback; traceback.print_exc()
                continue

        progbar.close()

    # 分批处理并直接保存，避免全部加载到内存
    return generate_pairs()

def save_dataset_streaming(data_generator, output_path, batch_size=10000):
    """流式保存数据集，避免内存溢出，不合并文件"""
    batch_data = []
    batch_count = 0
    
    for item in data_generator:
        batch_data.append(item)
        
        if len(batch_data) >= batch_size:
            # 保存当前批次，先不指定总数
            batch_ds = datasets.Dataset.from_list(batch_data)
            batch_ds.to_parquet(f'{output_path}-{batch_count:05d}.parquet')
            
            logger(f"Saved batch {batch_count} with {len(batch_data)} samples")
            
            # 清理内存
            del batch_data, batch_ds
            gc.collect()
            batch_data = []
            batch_count += 1
    
    # 处理剩余数据
    if batch_data:
        batch_ds = datasets.Dataset.from_list(batch_data)
        batch_ds.to_parquet(f'{output_path}-{batch_count:05d}.parquet')
        logger(f"Saved final batch {batch_count} with {len(batch_data)} samples")
        del batch_data, batch_ds
        gc.collect()
        batch_count += 1
    
    # 现在知道总批次数，重命名文件以包含总数信息
    total_batches = batch_count
    for i in range(total_batches):
        old_name = f'{output_path}-{i:05d}.parquet'
        new_name = f'{output_path}-{i:05d}-of-{total_batches:05d}.parquet'
        os.rename(old_name, new_name)
    
    logger(f"Total batches saved: {total_batches}")
    return total_batches

def splits_generator(data_base_dir):
    if 'test_04445.jpg' not in os.listdir(data_base_dir):
        data_base_dir = os.path.join(data_base_dir, 'caparena_auto_docci_600')
    yield data_base_dir, 'caparena'

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    torch.manual_seed(seed)

    source_path = "CapArena-master/data/caparena_auto_docci_600"
    logger(f'!! Use {source_path} as data dir')
    target_path = "datasets/caparena_ContrastiveQA"

    os.makedirs(target_path, exist_ok=True)
    for split_path, prefix in splits_generator(source_path):
        base = DatasetWrapper(split_path, prefix)
        data_generator = process_split(base)

        # 使用流式保存避免内存溢出，不合并文件
        batch_count = save_dataset_streaming(data_generator, f'{target_path}/{prefix}', batch_size=10000)
        logger(f"Split '{prefix}' saved as {batch_count} batch files")

        # 清理内存
        del base
        gc.collect()