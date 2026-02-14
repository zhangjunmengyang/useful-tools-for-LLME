"""
FineTune Lab - 微调工具函数
提供 LoRA 配置计算、训练成本估算等功能
"""

import json
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from huggingface_hub import hf_hub_download


# 预设模型配置
MODEL_CATEGORIES = {
    "LLaMA": {
        "models": [
            ("LLaMA-7B", "meta-llama/Llama-2-7b-hf", {"params": 6.7e9, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 32, "ffn": 11008}),
            ("LLaMA-13B", "meta-llama/Llama-2-13b-hf", {"params": 13e9, "hidden": 5120, "layers": 40, "heads": 40, "kv_heads": 40, "ffn": 13824}),
            ("LLaMA-70B", "meta-llama/Llama-2-70b-hf", {"params": 70e9, "hidden": 8192, "layers": 80, "heads": 64, "kv_heads": 8, "ffn": 28672}),
        ]
    },
    "Qwen": {
        "models": [
            ("Qwen-7B", "Qwen/Qwen-7B", {"params": 7.7e9, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 32, "ffn": 11008}),
            ("Qwen-14B", "Qwen/Qwen-14B", {"params": 14e9, "hidden": 5120, "layers": 40, "heads": 40, "kv_heads": 40, "ffn": 13696}),
            ("Qwen-72B", "Qwen/Qwen-72B", {"params": 72e9, "hidden": 8192, "layers": 80, "heads": 64, "kv_heads": 64, "ffn": 24576}),
        ]
    },
    "Mistral": {
        "models": [
            ("Mistral-7B", "mistralai/Mistral-7B-v0.1", {"params": 7.3e9, "hidden": 4096, "layers": 32, "heads": 32, "kv_heads": 8, "ffn": 14336}),
        ]
    }
}

# LoRA target modules
TARGET_MODULES = {
    "q_proj": "Query Projection",
    "k_proj": "Key Projection",
    "v_proj": "Value Projection",
    "o_proj": "Output Projection",
    "gate_proj": "FFN Gate",
    "up_proj": "FFN Up",
    "down_proj": "FFN Down",
}

# LoRA rank 预设值
LORA_RANKS = [4, 8, 16, 32, 64, 128]

# GPU 规格配置
GPU_SPECS = {
    "NVIDIA A100 (40GB)": {
        "memory_gb": 40,
        "tflops_fp32": 19.5,
        "tflops_fp16": 312,
        "cost_per_hour": 3.0,  # 示例价格，单位: USD
    },
    "NVIDIA A100 (80GB)": {
        "memory_gb": 80,
        "tflops_fp32": 19.5,
        "tflops_fp16": 312,
        "cost_per_hour": 4.5,
    },
    "NVIDIA H100": {
        "memory_gb": 80,
        "tflops_fp32": 51,
        "tflops_fp16": 1000,
        "cost_per_hour": 8.0,
    },
    "NVIDIA RTX 4090": {
        "memory_gb": 24,
        "tflops_fp32": 82.6,
        "tflops_fp16": 165.2,
        "cost_per_hour": 1.5,
    }
}

# 显存预算推荐配置
MEMORY_BUDGET_CONFIGS = {
    "24GB": {
        "budget": "24GB GPU (e.g., RTX 4090)",
        "configs": [
            {"model": "7B", "rank": 8, "modules": ["q_proj", "v_proj"], "batch_size": 1, "grad_accum": 8},
            {"model": "7B", "rank": 16, "modules": ["q_proj", "v_proj"], "batch_size": 1, "grad_accum": 4},
        ]
    },
    "40GB": {
        "budget": "40GB GPU (e.g., A100 40GB)",
        "configs": [
            {"model": "7B", "rank": 64, "modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "batch_size": 2, "grad_accum": 4},
            {"model": "13B", "rank": 32, "modules": ["q_proj", "v_proj"], "batch_size": 1, "grad_accum": 8},
        ]
    },
    "80GB": {
        "budget": "80GB GPU (e.g., A100 80GB / H100)",
        "configs": [
            {"model": "13B", "rank": 64, "modules": ["q_proj", "k_proj", "v_proj", "o_proj"], "batch_size": 4, "grad_accum": 2},
            {"model": "70B", "rank": 16, "modules": ["q_proj", "v_proj"], "batch_size": 1, "grad_accum": 8},
        ]
    }
}


def get_model_config(category: str, model_name: str) -> Optional[Dict]:
    """获取预设模型配置"""
    if category not in MODEL_CATEGORIES:
        return None
    for name, model_id, config in MODEL_CATEGORIES[category]["models"]:
        if name == model_name:
            return {"name": name, "model_id": model_id, **config}
    return None


def calculate_lora_params(
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    intermediate_size: int,
    rank: int,
    target_modules: List[str]
) -> Dict:
    """
    计算 LoRA 参数量

    Args:
        hidden_size: 隐藏层维度
        num_layers: Transformer 层数
        num_heads: Attention head 数量
        num_kv_heads: KV head 数量（GQA）
        intermediate_size: FFN 中间层维度
        rank: LoRA rank
        target_modules: 目标模块列表

    Returns:
        包含总参数量、每层参数量、详细分布的字典
    """
    head_dim = hidden_size // num_heads
    params_per_layer = 0
    details = []

    for module in target_modules:
        if module == "q_proj":
            out_dim = num_heads * head_dim
            module_params = hidden_size * rank + rank * out_dim
            params_per_layer += module_params
            details.append({
                "module": module,
                "dimension": f"{hidden_size} -> {out_dim}",
                "params_per_layer": module_params
            })
        elif module in ["k_proj", "v_proj"]:
            out_dim = num_kv_heads * head_dim
            module_params = hidden_size * rank + rank * out_dim
            params_per_layer += module_params
            details.append({
                "module": module,
                "dimension": f"{hidden_size} -> {out_dim}",
                "params_per_layer": module_params
            })
        elif module == "o_proj":
            in_dim = num_heads * head_dim
            module_params = in_dim * rank + rank * hidden_size
            params_per_layer += module_params
            details.append({
                "module": module,
                "dimension": f"{in_dim} -> {hidden_size}",
                "params_per_layer": module_params
            })
        elif module in ["gate_proj", "up_proj"]:
            module_params = hidden_size * rank + rank * intermediate_size
            params_per_layer += module_params
            details.append({
                "module": module,
                "dimension": f"{hidden_size} -> {intermediate_size}",
                "params_per_layer": module_params
            })
        elif module == "down_proj":
            module_params = intermediate_size * rank + rank * hidden_size
            params_per_layer += module_params
            details.append({
                "module": module,
                "dimension": f"{intermediate_size} -> {hidden_size}",
                "params_per_layer": module_params
            })

    total_params = params_per_layer * num_layers

    return {
        "total_params": total_params,
        "params_per_layer": params_per_layer,
        "details": details
    }


def estimate_lora_memory(
    lora_params: int,
    base_params: int,
    use_quantization: bool = False,
    quantization_bits: int = 4
) -> Dict[str, float]:
    """
    估算 LoRA 训练显存占用

    Args:
        lora_params: LoRA 参数量
        base_params: 基础模型参数量
        use_quantization: 是否使用量化（QLoRA）
        quantization_bits: 量化位数

    Returns:
        包含各部分显存占用的字典（单位: GB）
    """
    # 基础模型显存
    if use_quantization:
        base_memory = base_params * quantization_bits / 8 / 1e9
    else:
        base_memory = base_params * 2 / 1e9  # fp16

    # LoRA 参数显存（训练时使用 fp32）
    lora_memory = lora_params * 4 / 1e9

    # 优化器状态（Adam: 2 倍参数）
    optimizer_memory = lora_params * 4 * 2 / 1e9

    # 梯度显存
    gradient_memory = lora_params * 4 / 1e9

    # 激活值显存（粗略估算，约为模型大小的 10%）
    activation_memory = base_memory * 0.1

    total_memory = base_memory + lora_memory + optimizer_memory + gradient_memory + activation_memory

    return {
        "base_model": base_memory,
        "lora_params": lora_memory,
        "optimizer": optimizer_memory,
        "gradients": gradient_memory,
        "activations": activation_memory,
        "total": total_memory
    }


def calculate_training_flops(
    model_params: int,
    tokens: int,
    is_full_finetune: bool = False
) -> float:
    """
    估算训练 FLOPs

    公式: FLOPs ≈ 6 * params * tokens (full finetune)
          FLOPs ≈ 6 * trainable_params * tokens (LoRA)

    Args:
        model_params: 模型参数量（全参微调）或可训练参数量（LoRA）
        tokens: 训练数据总 token 数
        is_full_finetune: 是否为全参微调

    Returns:
        总 FLOPs
    """
    return 6 * model_params * tokens


def estimate_training_time(
    total_flops: float,
    gpu_tflops: float,
    num_gpus: int = 1,
    mfu: float = 0.5
) -> float:
    """
    估算训练时间

    Args:
        total_flops: 总 FLOPs
        gpu_tflops: 单个 GPU 的 TFLOPS（使用 fp16 性能）
        num_gpus: GPU 数量
        mfu: Model FLOPs Utilization（模型 FLOPs 利用率，通常 0.3-0.6）

    Returns:
        训练时间（小时）
    """
    effective_tflops = gpu_tflops * num_gpus * mfu
    time_seconds = total_flops / (effective_tflops * 1e12)
    time_hours = time_seconds / 3600
    return time_hours


def estimate_training_cost(
    training_hours: float,
    cost_per_hour: float,
    num_gpus: int = 1
) -> float:
    """
    估算训练成本

    Args:
        training_hours: 训练时长（小时）
        cost_per_hour: 单 GPU 每小时成本
        num_gpus: GPU 数量

    Returns:
        总成本（USD）
    """
    return training_hours * cost_per_hour * num_gpus


def get_memory_budget_recommendations(budget: str) -> List[Dict]:
    """获取显存预算推荐配置"""
    return MEMORY_BUDGET_CONFIGS.get(budget, {}).get("configs", [])


def format_number(num: float, precision: int = 2) -> str:
    """格式化大数字（B/M/K）"""
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"
