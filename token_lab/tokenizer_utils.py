"""
TokenLab - Tokenizer 工具函数
提供分词器加载、缓存、分词结果处理等核心功能
"""

import os
# 抑制 transformers 的深度学习框架警告（本项目只使用 tokenizer，不需要 PyTorch/TensorFlow/Flax）
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import hashlib
import unicodedata
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

from transformers import AutoTokenizer, PreTrainedTokenizer


# 按厂商/系列分类的模型列表
MODEL_CATEGORIES = {
    "OpenAI": {
        "models": [
            ("GPT-2", "openai-community/gpt2"),
            ("GPT-OSS-20B", "openai/gpt-oss-20b"),
            ("gpt-3.5-turbo", "Xenova/gpt-3.5-turbo"),
        ],
    },
    "Meta": {
        "models": [
            ("Llama-2-7b", "meta-llama/Llama-2-7b"),
            ("Llama-3.2-1B", "meta-llama/Llama-3.2-1B"),
            ("Llama-4-Scout-17B-16E-Instruct", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        ],
    },
    "Alibaba": {
        "models": [
            ("Qwen 2.5 7B", "Qwen/Qwen2.5-7B"),
            ("Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
            ("Qwen3-Next-80B-A3B-Instruct-GGUF", "Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF"),
            ("Qwen3-VL-2B-Thinking-FP8", "Qwen/Qwen3-VL-2B-Thinking-FP8"),
            ("Qwen3-Omni-30B-A3B-Thinking", "Qwen/Qwen3-Omni-30B-A3B-Thinking"),
        ],
    },
    "DeepSeek": {
        "models": [
            ("DeepSeek V3", "deepseek-ai/DeepSeek-V3"),
            ("DeepSeek V3.2", "deepseek-ai/DeepSeek-V3.2"),
            ("DeepSeek-OCR", "deepseek-ai/DeepSeek-OCR"),
            ("DeepSeek-R1", "deepseek-ai/DeepSeek-R1"),
        ],
    },
    "Google": {
        "models": [
            ("Gemma 7B", "google/gemma-7b"),
        ],
    },
    "MiniMax": {
        "models": [
            ("MiniMax-M1", "MiniMaxAI/MiniMax-M1"),
            ("MiniMax-M2", "MiniMaxAI/MiniMax-M2"),
        ],
    },
    "Moonshot": {
        "models": [
            ("Moonshot", "moonshotai/Kimi-K2-Thinking"),  # Moonshot 使用类似 Qwen 的 tokenizer
        ],
    },
}

# 兼容旧代码：生成扁平的模型列表
PRESET_MODELS = []
for category_data in MODEL_CATEGORIES.values():
    for display_name, model_id in category_data["models"]:
        if model_id not in PRESET_MODELS:
            PRESET_MODELS.append(model_id)


def get_model_categories():
    """获取模型分类数据"""
    return MODEL_CATEGORIES


def get_all_models_flat():
    """获取所有模型的扁平列表（用于兼容旧代码）"""
    return PRESET_MODELS


def get_models_by_category(category: str) -> list:
    """根据厂商获取模型列表"""
    if category in MODEL_CATEGORIES:
        return MODEL_CATEGORIES[category]["models"]
    return []

# 颜色调色板（用于彩虹分词）- 更新为高亮 Pastel 色系
TOKEN_COLORS = [
    "#D1FAE5",  # 薄荷绿
    "#DBEAFE",  # 天空蓝
    "#E9D5FF",  # 薰衣紫
    "#FED7AA",  # 蜜桃橙
    "#FBCFE8",  # 玫瑰粉
    "#FEF08A",  # 柠檬黄
    "#CFFAFE",  # 淡青色
    "#FECDD3",  # 桃粉色
    "#DDD6FE",  # 淡紫色
    "#A7F3D0",  # 薄荷青
    "#FFEDD5",  # 杏色
    "#E2E8F0",  # 石板灰
]


# Tokenizer 缓存
_tokenizer_cache = {}


def load_tokenizer(model_name: str) -> Optional[PreTrainedTokenizer]:
    """
    加载并缓存 tokenizer
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        _tokenizer_cache[model_name] = tokenizer
        return tokenizer
    except Exception as e:
        print(f"加载模型 '{model_name}' 失败: {str(e)}")
        return None


def get_token_info(tokenizer: PreTrainedTokenizer, text: str) -> List[Dict[str, Any]]:
    """
    获取文本的详细分词信息
    
    Returns:
        List of dicts containing:
        - token_str: 原始 token 字符串
        - token_id: token ID
        - byte_sequence: UTF-8 字节序列
        - is_special: 是否为特殊 token
        - start_char: 在原文中的起始位置
        - end_char: 在原文中的结束位置
    """
    if not text:
        return []
    
    # 获取编码结果
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    token_ids = encoding["input_ids"]
    
    # 尝试获取 offset mapping
    offset_mapping = encoding.get("offset_mapping", None)
    
    result = []
    special_tokens = set(tokenizer.all_special_tokens) if hasattr(tokenizer, 'all_special_tokens') else set()
    special_token_ids = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()
    
    for idx, token_id in enumerate(token_ids):
        # 获取原始 token 字符串（未解码形式）
        if hasattr(tokenizer, 'convert_ids_to_tokens'):
            raw_token = tokenizer.convert_ids_to_tokens([token_id])[0]
        else:
            raw_token = ""
        
        # 解码单个 token
        token_str = tokenizer.decode([token_id])
        
        # 检测是否为字节级别 token：解码后包含替换字符 ? 表示无法正确解码
        # 这是唯一可靠的判断方式，不依赖模型类型或字符范围
        is_byte_token = '\ufffd' in token_str
        
        # 如果解码失败（包含乱码），使用原始 token 作为显示字符串
        if is_byte_token and raw_token:
            display_token_str = raw_token
        else:
            display_token_str = token_str
        
        # 计算字节序列
        try:
            # 对于解码失败的 token，使用 raw_token 计算字节
            source_str = raw_token if is_byte_token and raw_token else token_str
            byte_seq = list(source_str.encode('utf-8'))
            byte_hex = ' '.join([f'0x{b:02X}' for b in byte_seq])
        except:
            byte_hex = "N/A"
        
        # 判断是否为特殊 token
        is_special = token_id in special_token_ids or token_str in special_tokens
        
        # 判断是否为 byte fallback token（如 <0xE4> 格式）
        is_byte_fallback = raw_token.startswith('<0x') and raw_token.endswith('>')
        
        # 获取字符位置
        start_char, end_char = None, None
        if offset_mapping and idx < len(offset_mapping):
            start_char, end_char = offset_mapping[idx]
        
        result.append({
            "token_str": display_token_str,  # 用于显示的字符串
            "decoded_str": token_str,  # 实际解码结果（可能包含乱码）
            "raw_token": raw_token,  # 原始 token 表示
            "token_id": token_id,
            "byte_sequence": byte_hex,
            "is_special": is_special,
            "is_byte_fallback": is_byte_fallback,
            "is_byte_token": is_byte_token,  # 标记是否为字节级别 token
            "start_char": start_char,
            "end_char": end_char,
            "index": idx,
        })
    
    return result


def get_tokenizer_info(tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    获取 tokenizer 的元信息
    """
    info = {
        "vocab_size": tokenizer.vocab_size,
        "model_max_length": getattr(tokenizer, 'model_max_length', 'N/A'),
        "padding_side": getattr(tokenizer, 'padding_side', 'N/A'),
        "truncation_side": getattr(tokenizer, 'truncation_side', 'N/A'),
        "bos_token": getattr(tokenizer, 'bos_token', None),
        "eos_token": getattr(tokenizer, 'eos_token', None),
        "pad_token": getattr(tokenizer, 'pad_token', None),
        "unk_token": getattr(tokenizer, 'unk_token', None),
        "cls_token": getattr(tokenizer, 'cls_token', None),
        "sep_token": getattr(tokenizer, 'sep_token', None),
        "mask_token": getattr(tokenizer, 'mask_token', None),
    }
    
    # 获取 tokenizer 类型
    tokenizer_class = type(tokenizer).__name__
    info["tokenizer_class"] = tokenizer_class
    
    # 推断分词算法类型
    if "BPE" in tokenizer_class or "GPT" in tokenizer_class:
        info["algorithm"] = "BPE (Byte-Pair Encoding)"
    elif "WordPiece" in tokenizer_class or "Bert" in tokenizer_class:
        info["algorithm"] = "WordPiece"
    elif "Unigram" in tokenizer_class or "Sentencepiece" in tokenizer_class.lower():
        info["algorithm"] = "Unigram / SentencePiece"
    else:
        info["algorithm"] = "Unknown"
    
    return info


def decode_token_ids(tokenizer: PreTrainedTokenizer, token_ids: List[int]) -> Tuple[str, List[Dict]]:
    """
    将 token IDs 解码回文本，并返回每个 ID 的详细信息
    """
    try:
        # 完整解码
        full_text = tokenizer.decode(token_ids)
        
        # 逐个解码
        individual_tokens = []
        for tid in token_ids:
            try:
                token_str = tokenizer.decode([tid])
                raw_token = tokenizer.convert_ids_to_tokens([tid])[0] if hasattr(tokenizer, 'convert_ids_to_tokens') else token_str
                individual_tokens.append({
                    "token_id": tid,
                    "token_str": token_str,
                    "raw_token": raw_token,
                    "valid": True
                })
            except:
                individual_tokens.append({
                    "token_id": tid,
                    "token_str": f"<INVALID:{tid}>",
                    "raw_token": f"<INVALID:{tid}>",
                    "valid": False
                })
        
        return full_text, individual_tokens
    except Exception as e:
        return f"解码错误: {str(e)}", []


def calculate_compression_stats(text: str, token_count: int) -> Dict[str, float]:
    """
    计算压缩率相关统计
    """
    char_count = len(text)
    byte_count = len(text.encode('utf-8'))
    
    return {
        "char_count": char_count,
        "byte_count": byte_count,
        "token_count": token_count,
        "chars_per_token": round(char_count / token_count, 2) if token_count > 0 else 0,
        "bytes_per_token": round(byte_count / token_count, 2) if token_count > 0 else 0,
        "compression_ratio": round(byte_count / token_count, 2) if token_count > 0 else 0,
    }


def get_normalization_info(text: str) -> Dict[str, Any]:
    """
    获取文本规范化信息
    """
    nfc = unicodedata.normalize('NFC', text)
    nfd = unicodedata.normalize('NFD', text)
    nfkc = unicodedata.normalize('NFKC', text)
    nfkd = unicodedata.normalize('NFKD', text)
    
    return {
        "original": text,
        "original_len": len(text),
        "NFC": nfc,
        "NFC_len": len(nfc),
        "NFD": nfd,
        "NFD_len": len(nfd),
        "NFKC": nfkc,
        "NFKC_len": len(nfkc),
        "NFKD": nfkd,
        "NFKD_len": len(nfkd),
        "nfc_equal": text == nfc,
        "nfd_equal": text == nfd,
    }


def get_unicode_info(char: str) -> Dict[str, Any]:
    """
    获取单个字符的 Unicode 信息
    """
    if not char:
        return {}
    
    try:
        return {
            "char": char,
            "name": unicodedata.name(char, "UNKNOWN"),
            "category": unicodedata.category(char),
            "codepoint": f"U+{ord(char):04X}",
            "decimal": ord(char),
            "utf8_bytes": ' '.join([f'0x{b:02X}' for b in char.encode('utf-8')]),
        }
    except:
        return {
            "char": char,
            "name": "UNKNOWN",
            "category": "Unknown",
            "codepoint": f"U+{ord(char):04X}" if len(char) == 1 else "N/A",
            "decimal": ord(char) if len(char) == 1 else "N/A",
            "utf8_bytes": "N/A",
        }


def apply_chat_template_safe(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False
) -> Tuple[Optional[str], Optional[str]]:
    """
    安全地应用 chat template
    
    Returns:
        (rendered_string, error_message)
    """
    try:
        if not hasattr(tokenizer, 'apply_chat_template'):
            return None, "该 tokenizer 不支持 chat template"
        
        if tokenizer.chat_template is None:
            return None, "该 tokenizer 没有配置 chat template"
        
        result = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        return result, None
    except Exception as e:
        return None, f"应用 chat template 失败: {str(e)}"


def get_special_tokens_map(tokenizer: PreTrainedTokenizer) -> Dict[str, Any]:
    """
    获取特殊 token 映射
    """
    special_map = {}
    
    # 标准特殊 token
    for attr in ['bos_token', 'eos_token', 'pad_token', 'unk_token', 'cls_token', 'sep_token', 'mask_token']:
        token = getattr(tokenizer, attr, None)
        if token:
            token_id = tokenizer.convert_tokens_to_ids(token) if hasattr(tokenizer, 'convert_tokens_to_ids') else None
            special_map[attr] = {
                "token": token,
                "id": token_id
            }
    
    # 额外的特殊 token
    if hasattr(tokenizer, 'additional_special_tokens'):
        additional = tokenizer.additional_special_tokens
        if additional:
            special_map['additional_special_tokens'] = []
            for token in additional:
                token_id = tokenizer.convert_tokens_to_ids(token) if hasattr(tokenizer, 'convert_tokens_to_ids') else None
                special_map['additional_special_tokens'].append({
                    "token": token,
                    "id": token_id
                })
    
    return special_map


def render_tokens_html(token_info_list: List[Dict], show_ids: bool = True) -> str:
    """
    将 token 信息渲染为 HTML（用于彩虹分词显示）
    """
    if not token_info_list:
        return "<p>无分词结果</p>"
    
    html_parts = ['<div class="token-container" style="line-height: 2.2; font-family: \'SF Mono\', \'Fira Code\', monospace;">']
    
    for idx, info in enumerate(token_info_list):
        color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        token_str = info['token_str']
        token_id = info['token_id']
        raw_token = info.get('raw_token', token_str)
        byte_seq = info.get('byte_sequence', 'N/A')
        is_special = info.get('is_special', False)
        is_byte_fallback = info.get('is_byte_fallback', False)
        
        # 转义 HTML
        display_str = token_str.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        display_str = display_str.replace(' ', '?').replace('\n', '?\n').replace('\t', '?')
        
        # 空字符串显示
        if not display_str.strip():
            display_str = repr(token_str)[1:-1] if token_str else '?'
        
        # 特殊标记样式
        border_style = ""
        if is_special:
            border_style = "border: 2px solid #FF4444;"
        elif is_byte_fallback:
            border_style = "border: 2px dashed #FF8800;"
        
        # 构建 tooltip 内容
        tooltip = f"Token: {raw_token}&#10;ID: {token_id}&#10;Bytes: {byte_seq}"
        
        # ID 显示
        id_html = f'<sub style="font-size: 0.7em; color: #444;">{token_id}</sub>' if show_ids else ''
        
        html_parts.append(
            f'<span class="token" title="{tooltip}" '
            f'style="background-color: {color}; padding: 3px 6px; margin: 2px; '
            f'border-radius: 4px; display: inline-block; cursor: pointer; {border_style}">'
            f'{display_str}{id_html}</span>'
        )
    
    html_parts.append('</div>')
    return ''.join(html_parts)
