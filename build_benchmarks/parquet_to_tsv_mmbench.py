#!/usr/bin/env python3
"""
将 MMBench parquet 文件转换为 TSV 格式
参考 MMBench_DEV_EN.tsv 的格式
"""

import os
import csv
import base64
import io
import datasets
from PIL import Image
import tqdm


def to_pil_image(image):
    """将各种格式的图片转换为 PIL.Image"""
    # 1. 如果已经是 PIL.Image
    if isinstance(image, Image.Image):
        return image
    # 2. 如果是 numpy 数组
    import numpy as np
    if isinstance(image, np.ndarray):
        # object 类型，通常包裹了 bytes
        if image.dtype == object:
            # 尝试展开
            if image.size == 1:
                return to_pil_image(image.item())
            # 如果是一维或二维，取第一个非空元素
            for item in image.flat:
                if item is not None:
                    return to_pil_image(item)
            raise ValueError("Object array but no valid image bytes found")
        else:
            # 正常图片数组
            return Image.fromarray(image)
    # 3. 如果是 dict，尝试取 'bytes'
    if isinstance(image, dict):
        if 'bytes' in image:
            return Image.open(io.BytesIO(image['bytes']))
        elif 'path' in image:
            return Image.open(image['path'])
    # 4. 如果是 bytes
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image))
    raise ValueError(f"Cannot process image format: {type(image)}")


def image_to_base64(image):
    """将 PIL.Image 转换为 base64 编码的字符串"""
    if image is None:
        return ""
    
    # 确保是 PIL.Image
    pil_img = to_pil_image(image)
    
    # 转换为 JPEG 格式的 bytes
    buffer = io.BytesIO()
    # 如果是 RGBA 模式，转换为 RGB
    if pil_img.mode == 'RGBA':
        # 创建白色背景
        rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
        rgb_img.paste(pil_img, mask=pil_img.split()[3])  # 使用 alpha 通道作为 mask
        pil_img = rgb_img
    elif pil_img.mode not in ('RGB', 'L'):
        pil_img = pil_img.convert('RGB')
    
    pil_img.save(buffer, format='JPEG', quality=95)
    img_bytes = buffer.getvalue()
    
    # 编码为 base64
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str


def parquet_to_tsv(parquet_path, tsv_path, image_dir=None):
    """将 parquet 文件转换为 TSV 格式
    
    Args:
        parquet_path: 输入的 parquet 文件路径
        tsv_path: 输出的 TSV 文件路径
        image_dir: 图片保存目录（可选），如果指定则保存图片文件
    """
    print(f"加载 parquet 文件: {parquet_path}")
    
    # 加载数据集
    dataset_dict = datasets.load_dataset("parquet", data_files=parquet_path)
    dataset = list(dataset_dict.values())[0]
    
    print(f"数据集大小: {len(dataset)}")
    
    # 检查第一个样本的字段
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"数据集字段: {list(sample.keys())}")
    
    # 创建图片保存目录
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)
        print(f"图片将保存到: {image_dir}")
    
    # TSV 文件的列名（根据 MMBench_DEV_EN.tsv 的格式）
    columns = [
        'index',
        'question',
        'hint',
        'A',
        'B',
        'C',
        'D',
        'answer',
        'category',
        'image',
        'source',
        'l2-category',
        'comment',
        'split'
    ]
    
    # 字段映射：从 parquet 字段到 TSV 列
    # 根据实际数据集调整这些映射
    field_mapping = {
        'index': ['index', 'question_id', 'id'],
        'question': ['question', 'query', 'prompt'],
        'hint': ['hint', 'context'],
        'A': ['A', 'a', 'option_a'],
        'B': ['B', 'b', 'option_b'],
        'C': ['C', 'c', 'option_c'],
        'D': ['D', 'd', 'option_d'],
        'answer': ['answer', 'gt', 'label', 'ground_truth'],
        'category': ['category', 'cat'],
        'source': ['source', 'data_source'],
        'l2-category': ['l2-category', 'l2_category', 'l2category'],
        'comment': ['comment', 'note'],
        'split': ['split', 'partition']
    }
    
    def get_field_value(sample, field_name):
        """从样本中获取字段值，支持多个候选字段名"""
        candidates = field_mapping.get(field_name, [field_name])
        for candidate in candidates:
            if candidate in sample:
                value = sample[candidate]
                # 如果是列表，取第一个元素（除了image字段）
                if isinstance(value, list) and len(value) > 0:
                    if field_name == 'image':
                        # image字段保持为列表的第一个元素（PIL Image对象）
                        return value[0]
                    else:
                        # 其他字段取第一个元素
                        return value[0]
                return value
        return ""
    
    # 打开 TSV 文件进行写入
    print(f"写入 TSV 文件: {tsv_path}")
    os.makedirs(os.path.dirname(tsv_path) if os.path.dirname(tsv_path) else '.', exist_ok=True)
    
    saved_image_count = 0
    
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        
        # 写入表头
        writer.writerow(columns)
        
        # 处理每个样本
        for idx, sample in enumerate(tqdm.tqdm(dataset, desc='转换中')):
            row = []
            
            for col in columns:
                if col == 'image':
                    # 处理图片字段
                    image_value = None
                    
                    # 首先尝试 images 字段（列表）
                    if 'images' in sample and sample['images']:
                        if isinstance(sample['images'], list) and len(sample['images']) > 0:
                            image_value = sample['images'][0]
                        elif sample['images']:
                            image_value = sample['images']
                    
                    # 如果没有找到，尝试 image 字段
                    if not image_value:
                        image_value = get_field_value(sample, 'image')
                    
                    if image_value:
                        try:
                            # 转换为 PIL.Image
                            pil_img = to_pil_image(image_value)
                            
                            # 确保是 RGB 模式（统一处理，用于保存和 base64 编码）
                            if pil_img.mode == 'RGBA':
                                rgb_img = Image.new('RGB', pil_img.size, (255, 255, 255))
                                rgb_img.paste(pil_img, mask=pil_img.split()[3])
                                pil_img = rgb_img
                            elif pil_img.mode not in ('RGB', 'L'):
                                pil_img = pil_img.convert('RGB')
                            
                            # 保存图片文件（如果指定了目录）
                            if image_dir:
                                # 获取索引值用于命名
                                index_value = get_field_value(sample, 'index')
                                if index_value:
                                    image_filename = f"{index_value}.jpg"
                                else:
                                    # 如果没有索引，使用行号
                                    image_filename = f"{idx + 1}.jpg"
                                
                                image_path = os.path.join(image_dir, image_filename)
                                # 保存图片
                                pil_img.save(image_path, format='JPEG', quality=95)
                                saved_image_count += 1
                            
                            # 转换为 base64（用于 TSV 文件）
                            # 使用已转换的 pil_img 来生成 base64
                            buffer = io.BytesIO()
                            pil_img.save(buffer, format='JPEG', quality=95)
                            img_bytes = buffer.getvalue()
                            base64_str = base64.b64encode(img_bytes).decode('utf-8')
                            row.append(base64_str)
                        except Exception as e:
                            print(f"警告: 样本 {idx} 的图片转换失败: {e}")
                            import traceback
                            traceback.print_exc()
                            row.append("")
                    else:
                        row.append("")
                else:
                    # 处理其他字段
                    value = get_field_value(sample, col)
                    # 转换为字符串
                    if value is None:
                        value = ""
                    elif isinstance(value, (list, tuple)):
                        # 如果是列表，转换为字符串（用换行符连接）
                        value = "\n".join(str(v) for v in value)
                    else:
                        value = str(value)
                    row.append(value)
            
            writer.writerow(row)
    
    print(f"\n转换完成！")
    print(f"共处理 {len(dataset)} 个样本")
    print(f"TSV 文件: {tsv_path}")
    if image_dir:
        print(f"保存了 {saved_image_count} 张图片到: {image_dir}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='将 MMBench parquet 文件转换为 TSV 格式')
    parser.add_argument(
        '--input',
        type=str,
        default='datasets/MMBench_test_augmented/test.parquet',
        help='输入的 parquet 文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='datasets/VLMEval/MMBench_test_augmented.tsv',
        help='输出的 TSV 文件路径'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default='datasets/VLMEval/images/MMBench_test_augmented',
        help='图片保存目录（可选），如果指定则保存图片文件'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        exit(1)
    
    parquet_to_tsv(args.input, args.output, image_dir=args.image_dir)

