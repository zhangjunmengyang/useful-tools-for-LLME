"""
ModelLab - 模型工具函数
用于显存估算等功能
"""

from urllib.parse import urlparse
from typing import Optional, List, Dict, Any, Tuple
import streamlit as st
import torch
from accelerate.commands.estimate import check_has_model, create_empty_model, estimate_training_usage
from accelerate.utils import calculate_maximum_sizes, convert_bytes


# 数据精度修正系数
# float32: 每个参数 4 字节
# float16/bfloat16: 每个参数 2 字节
# int8: 每个参数 1 字节
# int4: 每个参数 0.5 字节
DTYPE_MODIFIER = {
    "float32": 1,
    "float16/bfloat16": 2,
    "int8": 4,
    "int4": 8
}

# 可选精度列表
DTYPE_OPTIONS = ["float32", "float16/bfloat16", "int8", "int4"]

# 支持的模型库
LIBRARY_OPTIONS = ["auto", "transformers", "timm"]


def extract_from_url(name: str) -> str:
    """
    检查输入是否为 URL，如果是则提取模型名称
    
    Args:
        name: 模型名称或 URL
        
    Returns:
        模型名称
    """
    is_url = False
    try:
        result = urlparse(name)
        is_url = all([result.scheme, result.netloc])
    except Exception:
        is_url = False
    
    if not is_url:
        return name
    else:
        path = result.path
        return path[1:] if path.startswith('/') else path


def translate_llama(text: str) -> str:
    """
    将 Llama-2 和 CodeLlama 转换为其 HuggingFace 对应名称
    
    Args:
        text: 模型名称
        
    Returns:
        转换后的模型名称
    """
    if not text.endswith("-hf"):
        return text + "-hf"
    return text


@st.cache_resource(show_spinner=False)
def get_model(model_name: str, library: str, access_token: Optional[str] = None) -> torch.nn.Module:
    """
    从 HuggingFace Hub 获取模型，并在 meta 设备上初始化
    
    Args:
        model_name: 模型名称或 URL
        library: 模型库 ("auto", "transformers", "timm")
        access_token: HuggingFace API token (可选，用于访问私有模型)
        
    Returns:
        初始化在 meta 设备上的模型
        
    Raises:
        各种异常，需要在调用处捕获
    """
    # 处理 Llama 模型名称
    if "meta-llama/Llama-2-" in model_name or "meta-llama/CodeLlama-" in model_name:
        model_name = translate_llama(model_name)
    
    if library == "auto":
        library = None
    
    model_name = extract_from_url(model_name)
    
    try:
        model = create_empty_model(
            model_name, 
            library_name=library, 
            trust_remote_code=True, 
            access_token=access_token
        )
    except ImportError:
        # 尝试使用 trust_remote_code=False
        model = create_empty_model(
            model_name, 
            library_name=library, 
            trust_remote_code=False, 
            access_token=access_token
        )
    
    return model


def calculate_memory(model: torch.nn.Module, options: List[str]) -> List[Dict[str, Any]]:
    """
    计算模型在 meta 设备上初始化的内存使用
    
    这是核心计算函数，完全复刻参考实现的逻辑：
    1. 使用 calculate_maximum_sizes 获取总大小和最大层大小
    2. 根据不同精度计算实际内存占用
    3. 使用 estimate_training_usage 估算训练时的内存占用
    
    Args:
        model: 初始化在 meta 设备上的模型
        options: 要计算的精度列表 (如 ["float32", "float16/bfloat16"])
        
    Returns:
        包含各精度内存信息的列表，每项包含:
        - dtype: 精度类型
        - Largest Layer or Residual Group: 最大层或残差组大小
        - Total Size: 模型总大小
        - Training using Adam (Peak vRAM): Adam 训练峰值显存
    """
    total_size, largest_layer = calculate_maximum_sizes(model)
    
    data = []
    for dtype in options:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        
        modifier = DTYPE_MODIFIER[dtype]
        
        # 估算训练时的内存使用
        dtype_training_size = estimate_training_usage(
            dtype_total_size, 
            dtype if dtype != "float16/bfloat16" else "float16"
        )
        
        # 根据精度调整大小
        dtype_total_size /= modifier
        dtype_largest_layer /= modifier
        
        # 转换为人类可读格式
        dtype_total_size_str = convert_bytes(dtype_total_size)
        dtype_largest_layer_str = convert_bytes(dtype_largest_layer)
        
        data.append({
            "dtype": dtype,
            "Largest Layer or Residual Group": dtype_largest_layer_str,
            "Total Size": dtype_total_size_str,
            "Training using Adam (Peak vRAM)": dtype_training_size,
        })
    
    return data


def calculate_memory_detailed(model: torch.nn.Module, options: List[str]) -> Tuple[List[Dict], Dict]:
    """
    计算详细的内存使用信息，包括训练各阶段的内存占用
    
    Args:
        model: 初始化在 meta 设备上的模型
        options: 要计算的精度列表
        
    Returns:
        (基础数据列表, 训练阶段详细数据)
    """
    total_size, largest_layer = calculate_maximum_sizes(model)
    
    data = []
    stages = {"model": [], "gradients": [], "optimizer": [], "step": []}
    
    for dtype in options:
        dtype_total_size = total_size
        dtype_largest_layer = largest_layer[0]
        
        modifier = DTYPE_MODIFIER[dtype]
        
        # 估算训练时的内存使用 (返回字典包含各阶段)
        dtype_training_size = estimate_training_usage(
            dtype_total_size, 
            dtype if dtype != "float16/bfloat16" else "float16"
        )
        
        # 保存各阶段数据
        for stage in stages:
            stages[stage].append(dtype_training_size.get(stage, -1))
        
        # 根据精度调整大小
        dtype_total_size /= modifier
        dtype_largest_layer /= modifier
        
        # 转换为人类可读格式
        dtype_total_size_str = convert_bytes(dtype_total_size)
        dtype_largest_layer_str = convert_bytes(dtype_largest_layer)
        
        # 计算训练峰值 (取各阶段最大值)
        peak_value = max(dtype_training_size.values()) if dtype_training_size else -1
        if peak_value == -1:
            peak_str = "N/A"
        else:
            peak_str = convert_bytes(peak_value)
        
        data.append({
            "dtype": dtype,
            "Largest Layer or Residual Group": dtype_largest_layer_str,
            "Total Size": dtype_total_size_str,
            "Training using Adam (Peak vRAM)": peak_str,
        })
    
    return data, stages


def format_training_stages(stages: Dict, options: List[str]) -> List[Dict]:
    """
    格式化训练各阶段的内存数据
    
    Args:
        stages: 各阶段原始数据
        options: 精度列表
        
    Returns:
        格式化后的数据列表
    """
    result = []
    for i, dtype in enumerate(options):
        if stages["model"][i] != -1:
            result.append({
                "dtype": dtype,
                "Model": convert_bytes(stages["model"][i]),
                "Gradient calculation": convert_bytes(stages["gradients"][i]),
                "Backward pass": convert_bytes(stages["optimizer"][i]),
                "Optimizer step": convert_bytes(stages["step"][i]),
            })
    return result


def get_model_error_message(e: Exception, model_name: str) -> str:
    """
    根据异常类型生成用户友好的错误信息
    
    Args:
        e: 异常对象
        model_name: 模型名称
        
    Returns:
        错误信息字符串
    """
    error_type = type(e).__name__
    
    if "GatedRepoError" in error_type:
        return f"模型 `{model_name}` 是受限模型，请提供有效的 API Token 并确保您有访问权限。"
    elif "RepositoryNotFoundError" in error_type:
        return f"在 HuggingFace Hub 上未找到模型 `{model_name}`，请检查模型名称是否正确。"
    elif "ValueError" in error_type:
        return f"模型 `{model_name}` 在 Hub 上没有库元数据，请手动选择模型库（如 `transformers`）。"
    elif "ImportError" in error_type:
        return f"加载模型 `{model_name}` 需要的依赖未安装，请检查环境配置。"
    else:
        return f"加载模型 `{model_name}` 时发生错误: {str(e)}"

