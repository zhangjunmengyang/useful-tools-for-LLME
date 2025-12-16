"""
GenerationLab - 核心工具函数
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from functools import lru_cache

# 轻量级模型配置 (用于演示)
DEMO_MODELS = {
    "GPT-2 Small": {
        "id": "openai-community/gpt2",
        "description": "117M 参数，适合本地演示",
        "context_length": 1024
    },
    "GPT-2 Medium": {
        "id": "openai-community/gpt2-medium",
        "description": "345M 参数，平衡性能与资源",
        "context_length": 1024
    },
    "DistilGPT-2": {
        "id": "distilgpt2",
        "description": "82M 参数，轻量蒸馏版本",
        "context_length": 1024
    }
}


# 模型缓存
_model_cache = {}


def load_model_and_tokenizer(model_name: str) -> Tuple[Any, Any]:
    """加载模型和 tokenizer"""
    if model_name in _model_cache:
        return _model_cache[model_name]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model.eval()
        _model_cache[model_name] = (model, tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return None, None


def get_next_token_logits(
    model: Any,
    tokenizer: Any,
    text: str,
    top_k: int = 50
) -> List[Dict]:
    """
    获取下一个 token 的 logits 和概率分布
    
    Returns:
        List of dicts with token_id, token_str, logit, probability
    """
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # 最后一个位置的 logits
    
    # 获取 top-k
    top_logits, top_indices = torch.topk(logits, top_k)
    
    # 计算 softmax 概率
    probs = torch.softmax(top_logits, dim=-1)
    
    results = []
    for i in range(top_k):
        token_id = top_indices[i].item()
        token_str = tokenizer.decode([token_id])
        results.append({
            "rank": i + 1,
            "token_id": token_id,
            "token_str": token_str,
            "raw_token": tokenizer.convert_ids_to_tokens([token_id])[0],
            "logit": top_logits[i].item(),
            "probability": probs[i].item()
        })
    
    return results


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """应用温度缩放"""
    if temperature <= 0:
        temperature = 1e-10
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """应用 Top-K 截断"""
    if top_k <= 0:
        return logits
    
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """应用 Top-P (Nucleus) 截断"""
    if top_p >= 1.0:
        return logits
    
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 找到累积概率超过 top_p 的位置
    sorted_indices_to_remove = cumulative_probs > top_p
    # 保留第一个超过阈值的 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    return logits


def get_sampling_distribution(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    获取经过采样策略处理后的概率分布
    
    Returns:
        (processed_logits, probabilities)
    """
    processed_logits = logits.clone()
    
    # 应用温度
    processed_logits = apply_temperature(processed_logits, temperature)
    
    # 应用 Top-K
    if top_k > 0:
        processed_logits = apply_top_k(processed_logits, top_k)
    
    # 应用 Top-P
    if top_p < 1.0:
        processed_logits = apply_top_p(processed_logits, top_p)
    
    # 计算概率
    probs = torch.softmax(processed_logits, dim=-1)
    
    return processed_logits, probs


def beam_search_step(
    model: Any,
    tokenizer: Any,
    sequences: List[Dict],
    beam_size: int = 3
) -> List[Dict]:
    """
    执行一步 Beam Search
    
    Args:
        sequences: List of {token_ids, score, text}
        beam_size: Beam 大小
    
    Returns:
        新的 beam 候选列表
    """
    all_candidates = []
    
    for seq in sequences:
        # 获取当前序列的下一个 token 分布
        inputs = tokenizer(seq['text'], return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        log_probs = torch.log_softmax(logits, dim=-1)
        top_log_probs, top_indices = torch.topk(log_probs, beam_size * 2)
        
        for i in range(beam_size * 2):
            token_id = top_indices[i].item()
            token_log_prob = top_log_probs[i].item()
            new_token_ids = seq['token_ids'] + [token_id]
            new_score = seq['score'] + token_log_prob
            new_text = tokenizer.decode(new_token_ids)
            
            all_candidates.append({
                'token_ids': new_token_ids,
                'score': new_score,
                'text': new_text,
                'last_token': tokenizer.decode([token_id]),
                'last_token_id': token_id,
                'parent_text': seq['text']
            })
    
    # 排序并保留 top beam_size
    all_candidates.sort(key=lambda x: x['score'], reverse=True)
    return all_candidates[:beam_size]


def calculate_kv_cache_size(
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    seq_length: int,
    batch_size: int = 1,
    dtype_bytes: int = 2  # fp16 = 2 bytes
) -> Dict[str, float]:
    """
    计算 KV Cache 显存占用
    
    KV Cache 大小 = 2 * num_layers * batch_size * seq_length * hidden_size * dtype_bytes
    (K 和 V 各占一半)
    """
    head_dim = hidden_size // num_heads
    
    # 单个 KV 的大小
    single_kv_size = batch_size * seq_length * hidden_size * dtype_bytes
    
    # 总 KV Cache 大小 (K + V, 所有层)
    total_kv_cache = 2 * num_layers * single_kv_size
    
    # 转换为 GB
    total_gb = total_kv_cache / (1024 ** 3)
    
    return {
        'total_bytes': total_kv_cache,
        'total_gb': total_gb,
        'per_layer_bytes': 2 * single_kv_size,
        'per_layer_mb': (2 * single_kv_size) / (1024 ** 2),
        'k_cache_bytes': num_layers * single_kv_size,
        'v_cache_bytes': num_layers * single_kv_size,
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'head_dim': head_dim,
        'seq_length': seq_length,
        'batch_size': batch_size
    }


def simulate_kv_cache_growth(
    config: Dict,
    prompt_length: int,
    generation_length: int,
    batch_size: int = 1,
    dtype_bytes: int = 2
) -> List[Dict]:
    """
    模拟推理过程中 KV Cache 的增长
    
    Returns:
        每一步的 KV Cache 状态
    """
    num_layers = config.get('num_hidden_layers', 32)
    hidden_size = config.get('hidden_size', 4096)
    num_heads = config.get('num_attention_heads', 32)
    
    results = []
    
    # Prefill 阶段 - 一次性处理整个 prompt
    prefill_cache = calculate_kv_cache_size(
        num_layers, hidden_size, num_heads,
        prompt_length, batch_size, dtype_bytes
    )
    results.append({
        'step': 0,
        'phase': 'Prefill',
        'seq_length': prompt_length,
        'cache_gb': prefill_cache['total_gb'],
        'delta_gb': prefill_cache['total_gb'],
        'description': f'处理 Prompt ({prompt_length} tokens)'
    })
    
    # Decode 阶段 - 逐 token 生成
    for i in range(1, generation_length + 1):
        current_length = prompt_length + i
        decode_cache = calculate_kv_cache_size(
            num_layers, hidden_size, num_heads,
            current_length, batch_size, dtype_bytes
        )
        prev_cache_gb = results[-1]['cache_gb']
        delta_gb = decode_cache['total_gb'] - prev_cache_gb
        
        results.append({
            'step': i,
            'phase': 'Decode',
            'seq_length': current_length,
            'cache_gb': decode_cache['total_gb'],
            'delta_gb': delta_gb,
            'description': f'生成第 {i} 个 token'
        })
    
    return results


def simulate_paged_attention(
    total_tokens: int,
    block_size: int = 16,
    num_sequences: int = 4
) -> Dict:
    """
    模拟 PagedAttention 的 block 分配
    
    Returns:
        Block 分配状态和碎片化情况
    """
    sequences = []
    total_blocks_used = 0
    total_waste = 0
    
    # 模拟不同长度的序列
    np.random.seed(42)
    seq_lengths = np.random.randint(
        total_tokens // 2, 
        total_tokens * 2, 
        num_sequences
    )
    
    for i, seq_len in enumerate(seq_lengths):
        blocks_needed = (seq_len + block_size - 1) // block_size
        waste = blocks_needed * block_size - seq_len
        
        sequences.append({
            'seq_id': i,
            'length': seq_len,
            'blocks': blocks_needed,
            'waste': waste,
            'utilization': seq_len / (blocks_needed * block_size) * 100
        })
        
        total_blocks_used += blocks_needed
        total_waste += waste
    
    total_capacity = total_blocks_used * block_size
    overall_utilization = sum(s['length'] for s in sequences) / total_capacity * 100
    
    return {
        'sequences': sequences,
        'total_blocks': total_blocks_used,
        'block_size': block_size,
        'total_capacity': total_capacity,
        'total_tokens': sum(s['length'] for s in sequences),
        'total_waste': total_waste,
        'overall_utilization': overall_utilization,
        'fragmentation': 100 - overall_utilization
    }


# 常见模型配置 (用于 KV Cache 计算)
MODEL_CONFIGS = {
    "Llama-2-7B": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,  # MHA
    },
    "Llama-2-13B": {
        "num_hidden_layers": 40,
        "hidden_size": 5120,
        "num_attention_heads": 40,
        "num_key_value_heads": 40,
    },
    "Llama-2-70B": {
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,  # GQA
    },
    "Llama-3-8B": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,  # GQA
    },
    "Llama-3-70B": {
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
    },
    "Qwen-7B": {
        "num_hidden_layers": 32,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 32,
    },
    "Qwen-72B": {
        "num_hidden_layers": 80,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 64,
    },
    "GPT-2": {
        "num_hidden_layers": 12,
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
    },
    "GPT-3-175B": {
        "num_hidden_layers": 96,
        "hidden_size": 12288,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
    },
}


def get_model_config(model_name: str) -> Optional[Dict]:
    """获取模型配置"""
    return MODEL_CONFIGS.get(model_name)


def format_bytes(bytes_val: float) -> str:
    """格式化字节数"""
    if bytes_val >= 1024 ** 3:
        return f"{bytes_val / (1024 ** 3):.2f} GB"
    elif bytes_val >= 1024 ** 2:
        return f"{bytes_val / (1024 ** 2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    else:
        return f"{bytes_val:.0f} B"
