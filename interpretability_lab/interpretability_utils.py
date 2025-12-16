"""
InterpretabilityLab - 核心工具函数
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import math

# 轻量级模型配置
INTERPRETABILITY_MODELS = {
    "GPT-2 (12层)": {
        "id": "openai-community/gpt2",
        "layers": 12,
        "heads": 12,
        "hidden": 768
    },
    "DistilGPT-2 (6层)": {
        "id": "distilgpt2",
        "layers": 6,
        "heads": 12,
        "hidden": 768
    }
}


# 模型缓存
_model_cache = {}


def load_model_with_attention(model_name: str) -> Tuple[Any, Any]:
    """加载模型并启用 attention 输出"""
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model.eval()
        _model_cache[model_name] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None, None


def get_attention_weights(
    model: Any,
    tokenizer: Any,
    text: str
) -> Tuple[torch.Tensor, List[str]]:
    """
    获取所有层所有头的注意力权重
    
    Returns:
        attention_weights: (num_layers, num_heads, seq_len, seq_len)
        tokens: token 列表
    """
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # attentions: tuple of (batch, heads, seq, seq) for each layer
    attentions = outputs.attentions
    
    # Stack all layers: (layers, batch, heads, seq, seq)
    attention_weights = torch.stack(attentions, dim=0)
    # Remove batch dim: (layers, heads, seq, seq)
    attention_weights = attention_weights.squeeze(1)
    
    # 获取 tokens
    token_ids = inputs['input_ids'][0].tolist()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    return attention_weights, tokens


def apply_causal_mask(attention_weights: torch.Tensor) -> torch.Tensor:
    """应用因果掩码（下三角）"""
    seq_len = attention_weights.shape[-1]
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return attention_weights * mask


def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
    """计算注意力熵（衡量注意力分布的分散程度）"""
    # attention_weights: (heads, seq, seq) or (seq, seq)
    eps = 1e-10
    entropy = -torch.sum(attention_weights * torch.log(attention_weights + eps), dim=-1)
    return entropy


def get_attention_patterns(attention_weights: torch.Tensor) -> Dict[str, float]:
    """分析注意力模式"""
    # attention_weights: (seq, seq)
    seq_len = attention_weights.shape[0]
    
    # 对角线注意力（关注自己）
    diagonal_attention = torch.diag(attention_weights).mean().item()
    
    # 第一个 token 注意力
    first_token_attention = attention_weights[:, 0].mean().item()
    
    # 局部注意力（相邻位置）
    local_mask = torch.zeros_like(attention_weights)
    for i in range(seq_len):
        start = max(0, i - 2)
        end = min(seq_len, i + 1)
        local_mask[i, start:end] = 1
    local_attention = (attention_weights * local_mask).sum().item() / local_mask.sum().item()
    
    return {
        'diagonal': diagonal_attention,
        'first_token': first_token_attention,
        'local': local_attention,
        'global': 1 - local_attention
    }


# ============ RoPE 相关函数 ============

def compute_rope_frequencies(
    dim: int,
    max_position: int = 512,
    base: float = 10000.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 RoPE 的旋转频率
    
    Returns:
        freqs: (max_position, dim//2) 频率矩阵
        positions: 位置索引
    """
    # 计算频率: 1 / base^(2i/dim)
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    
    positions = np.arange(max_position, dtype=np.float32)
    
    # 外积: (positions, dim//2)
    freqs = np.outer(positions, inv_freq)
    
    return freqs, positions


def apply_rope_rotation(
    vector: np.ndarray,
    position: int,
    base: float = 10000.0
) -> np.ndarray:
    """
    对向量应用 RoPE 旋转
    
    Args:
        vector: 输入向量 (dim,)
        position: 位置索引
        base: RoPE base
    
    Returns:
        旋转后的向量
    """
    dim = len(vector)
    rotated = np.zeros_like(vector)
    
    for i in range(dim // 2):
        freq = 1.0 / (base ** (2 * i / dim))
        theta = position * freq
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 旋转一对元素
        x1 = vector[2 * i]
        x2 = vector[2 * i + 1]
        
        rotated[2 * i] = x1 * cos_theta - x2 * sin_theta
        rotated[2 * i + 1] = x1 * sin_theta + x2 * cos_theta
    
    return rotated


def compute_rope_decay(
    dim: int,
    max_distance: int = 100,
    base: float = 10000.0
) -> np.ndarray:
    """
    计算 RoPE 的相对位置衰减
    
    两个向量在不同相对距离下的内积会表现出衰减特性
    """
    # 创建一个随机的 query 向量
    np.random.seed(42)
    q = np.random.randn(dim).astype(np.float32)
    k = np.random.randn(dim).astype(np.float32)
    
    # 固定 query 在位置 0
    q_rotated = apply_rope_rotation(q, 0, base)
    
    # 计算不同距离的内积
    dot_products = []
    for dist in range(max_distance):
        k_rotated = apply_rope_rotation(k, dist, base)
        dot = np.dot(q_rotated, k_rotated)
        dot_products.append(dot)
    
    return np.array(dot_products)


# ============ FFN 相关函数 ============

def analyze_ffn_activations(
    model: Any,
    tokenizer: Any,
    text: str,
    layer_idx: int = 0
) -> Dict:
    """
    分析 FFN 层的激活情况
    
    Returns:
        激活统计信息
    """
    inputs = tokenizer(text, return_tensors="pt")
    
    # 存储中间激活
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册 hook
    hooks = []
    
    # 获取 FFN 模块（GPT-2 结构）
    if hasattr(model, 'transformer'):
        ffn_module = model.transformer.h[layer_idx].mlp
        hooks.append(ffn_module.c_fc.register_forward_hook(hook_fn('fc1')))
        hooks.append(ffn_module.c_proj.register_forward_hook(hook_fn('fc2')))
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    # 分析激活
    results = {}
    
    if 'fc1' in activations:
        fc1_out = activations['fc1'][0]  # (seq_len, hidden)
        
        # 稀疏性分析
        sparsity = (fc1_out.abs() < 0.1).float().mean().item()
        
        # 激活分布
        results['fc1'] = {
            'mean': fc1_out.mean().item(),
            'std': fc1_out.std().item(),
            'max': fc1_out.max().item(),
            'min': fc1_out.min().item(),
            'sparsity': sparsity,
            'shape': list(fc1_out.shape),
            'values': fc1_out.mean(dim=0).numpy()  # 平均每个神经元的激活
        }
    
    return results


def compare_activation_functions(x_range: np.ndarray) -> Dict[str, np.ndarray]:
    """
    比较不同激活函数
    """
    results = {}
    
    # GELU
    results['GELU'] = 0.5 * x_range * (1 + np.tanh(np.sqrt(2/np.pi) * (x_range + 0.044715 * x_range**3)))
    
    # ReLU
    results['ReLU'] = np.maximum(0, x_range)
    
    # SiLU/Swish
    results['SiLU'] = x_range * (1 / (1 + np.exp(-x_range)))
    
    # SwiGLU 的门控部分 (简化)
    gate = 1 / (1 + np.exp(-x_range))
    results['SwiGLU (gate)'] = x_range * gate
    
    return results


# ============ 模型架构相关 ============

MODEL_ARCHITECTURES = {
    "GPT-2": {
        "attention": "MHA (Multi-Head Attention)",
        "ffn": "GELU",
        "norm": "LayerNorm (Pre)",
        "position": "Learned Absolute"
    },
    "Llama": {
        "attention": "GQA (Grouped Query Attention)",
        "ffn": "SwiGLU",
        "norm": "RMSNorm (Pre)",
        "position": "RoPE"
    },
    "Llama-3": {
        "attention": "GQA",
        "ffn": "SwiGLU",
        "norm": "RMSNorm (Pre)",
        "position": "RoPE (extended)"
    },
    "Qwen": {
        "attention": "GQA",
        "ffn": "SwiGLU",
        "norm": "RMSNorm (Pre)",
        "position": "RoPE"
    },
    "Mistral": {
        "attention": "GQA + Sliding Window",
        "ffn": "SwiGLU",
        "norm": "RMSNorm (Pre)",
        "position": "RoPE"
    }
}
