# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
支持噪声注入的权重加载器

该模块扩展了原有的权重加载功能，支持在加载权重时注入随机噪声。
"""

import time
import torch
import torch.distributed as dist
from typing import Dict, Optional

from verl.utils.device import get_device_id, get_torch_device
from verl.utils.noise_injection import NoiseInjector, create_noise_injector_from_config
from .loader import load_state_dict_to_megatron_gptmodel, _megatron_calc_global_rank


def load_state_dict_to_megatron_gptmodel_with_noise(
    state_dict: Dict[str, torch.Tensor],
    wrapped_models,
    config,
    params_dtype,
    is_value_model=False,
    noise_config: Optional[Dict] = None,
):
    """
    向Megatron模型加载权重并注入噪声
    
    Args:
        state_dict: 模型状态字典
        wrapped_models: 包装的模型
        config: 模型配置
        params_dtype: 参数数据类型
        is_value_model: 是否为价值模型
        noise_config: 噪声配置字典
        
    Returns:
        加载噪声后的模型配置
    """
    # 如果提供了噪声配置，创建噪声注入器
    noise_injector = None
    if noise_config is not None:
        noise_injector = create_noise_injector_from_config(noise_config)
        print(f"Created noise injector with config: {noise_config}")
    
    # 如果使用噪声注入，先向状态字典注入噪声
    if noise_injector is not None:
        print("Injecting noise to state dict...")
        state_dict = noise_injector.inject_noise_to_state_dict(state_dict, "model")
    
    # 调用原始的权重加载函数
    return load_state_dict_to_megatron_gptmodel(
        state_dict=state_dict,
        wrapped_models=wrapped_models,
        config=config,
        params_dtype=params_dtype,
        is_value_model=is_value_model,
    )


def load_megatron_model_weights_with_noise(
    config,
    model_config,
    parallel_model,
    params_dtype,
    is_value_model=False,
    local_cache_path="~/.cache/verl/rlhf",
    noise_config: Optional[Dict] = None,
):
    """
    向Megatron模型加载权重并注入噪声
    
    Args:
        config: 模型配置
        model_config: HuggingFace模型配置
        parallel_model: 并行模型
        params_dtype: 参数数据类型
        is_value_model: 是否为价值模型
        local_cache_path: 本地缓存路径
        noise_config: 噪声配置字典
        
    Returns:
        加载噪声后的模型配置
    """
    from verl.utils.model import _load_hf_model
    from verl.models.weight_loader_registry import get_weight_loader
    
    # 加载原始模型和状态字典
    architectures, model, state_dict, is_value_model = _load_hf_model(
        config, model_config, is_value_model, local_cache_path
    )
    
    # 如果提供了噪声配置，创建噪声注入器
    noise_injector = None
    if noise_config is not None:
        noise_injector = create_noise_injector_from_config(noise_config)
        print(f"Created noise injector with config: {noise_config}")
    
    # 如果使用噪声注入，先向状态字典注入噪声
    if noise_injector is not None:
        print("Injecting noise to state dict...")
        state_dict = noise_injector.inject_noise_to_state_dict(state_dict, "model")
    
    # 使用权重加载器加载权重
    print(f"before weight loader: architectures = {architectures}...")
    for arch in architectures:
        print(f"call weight loader arch = {arch}, model config = {model.config}")
        weight_loader = get_weight_loader(arch)
        weight_loader(
            state_dict=state_dict,
            wrapped_models=parallel_model,
            config=model.config,
            params_dtype=params_dtype,
            is_value_model=is_value_model,
            tie_word_embeddings=model_config.tie_word_embeddings,
        )
    
    return model.config


def load_megatron_gptmodel_weights_with_noise(
    config,
    model_config,
    parallel_model,
    params_dtype,
    is_value_model=False,
    local_cache_path="~/.cache/verl/rlhf",
    noise_config: Optional[Dict] = None,
):
    """
    向Megatron GPT模型加载权重并注入噪声
    
    Args:
        config: 模型配置
        model_config: HuggingFace模型配置
        parallel_model: 并行模型
        params_dtype: 参数数据类型
        is_value_model: 是否为价值模型
        local_cache_path: 本地缓存路径
        noise_config: 噪声配置字典
        
    Returns:
        加载噪声后的模型配置
    """
    from verl.utils.model import _load_hf_model
    from verl.models.mcore.loader import load_state_dict_to_megatron_gptmodel
    
    # 加载原始模型和状态字典
    _, model, state_dict, is_value_model = _load_hf_model(config, model_config, is_value_model, local_cache_path)
    
    # 如果提供了噪声配置，创建噪声注入器
    noise_injector = None
    if noise_config is not None:
        noise_injector = create_noise_injector_from_config(noise_config)
        print(f"Created noise injector with config: {noise_config}")
    
    # 如果使用噪声注入，先向状态字典注入噪声
    if noise_injector is not None:
        print("Injecting noise to state dict...")
        state_dict = noise_injector.inject_noise_to_state_dict(state_dict, "model")
    
    # 加载权重到Megatron模型
    load_state_dict_to_megatron_gptmodel(
        state_dict=state_dict,
        wrapped_models=parallel_model,
        config=model.config,
        params_dtype=params_dtype,
        is_value_model=is_value_model,
    )
    
    del state_dict, model
    return model.config