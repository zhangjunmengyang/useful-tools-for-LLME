"""
DataLab - 核心工具函数
"""

import re
import json
import unicodedata
import math
from typing import List, Dict, Any, Optional, Tuple

# SFT 数据格式模板
CHAT_TEMPLATES = {
    "alpaca": {
        "name": "Alpaca",
        "system": "Below is an instruction that describes a task. Write a response that appropriately completes the request.",
        "template": """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""
    },
    "sharegpt": {
        "name": "ShareGPT",
        "template": {
            "conversations": [
                {"from": "human", "value": "{instruction}"},
                {"from": "gpt", "value": "{output}"}
            ]
        }
    },
    "chatml": {
        "name": "ChatML",
        "template": """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
{output}<|im_end|>"""
    },
    "llama2": {
        "name": "Llama-2 Chat",
        "template": """<s>[INST] <<SYS>>
{system}
<</SYS>>

{instruction} [/INST] {output} </s>"""
    }
}

# 常用清洗规则
CLEANING_RULES = {
    "remove_html": {
        "name": "移除 HTML 标签",
        "pattern": r"<[^>]+>",
        "replacement": ""
    },
    "remove_urls": {
        "name": "移除 URL",
        "pattern": r"https?://\S+|www\.\S+",
        "replacement": ""
    },
    "remove_emails": {
        "name": "移除邮箱",
        "pattern": r"\S+@\S+\.\S+",
        "replacement": ""
    },
    "normalize_whitespace": {
        "name": "规范化空白符",
        "pattern": r"\s+",
        "replacement": " "
    },
    "remove_special_chars": {
        "name": "移除特殊字符",
        "pattern": r"[^\w\s\u4e00-\u9fff.,!?;:\"'()-]",
        "replacement": ""
    }
}


def apply_cleaning_rule(text: str, rule_id: str) -> str:
    """应用单个清洗规则"""
    if rule_id not in CLEANING_RULES:
        return text
    rule = CLEANING_RULES[rule_id]
    return re.sub(rule['pattern'], rule['replacement'], text)


def clean_text(text: str, rules: List[str]) -> str:
    """应用多个清洗规则"""
    result = text
    for rule_id in rules:
        result = apply_cleaning_rule(result, rule_id)
    return result.strip()


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Unicode 规范化"""
    return unicodedata.normalize(form, text)


def convert_to_format(data: Dict, target_format: str, system_prompt: str = "") -> str:
    """转换数据到指定格式"""
    template_info = CHAT_TEMPLATES.get(target_format)
    if not template_info:
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    instruction = data.get('instruction', data.get('input', ''))
    input_text = data.get('input', '')
    output = data.get('output', data.get('response', ''))
    system = system_prompt or template_info.get('system', '')
    
    if target_format == "sharegpt":
        return json.dumps({
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        }, ensure_ascii=False, indent=2)
    
    template = template_info['template']
    return template.format(
        instruction=instruction,
        input=input_text,
        output=output,
        system=system
    )


def validate_chat_format(text: str, format_type: str) -> Dict[str, Any]:
    """验证 Chat 格式是否正确"""
    issues = []
    
    if format_type == "chatml":
        if text.count("<|im_start|>") != text.count("<|im_end|>"):
            issues.append("标签未闭合: im_start 和 im_end 数量不匹配")
        if "<|im_start|>assistant" in text and "<|im_end|>" not in text.split("<|im_start|>assistant")[-1]:
            issues.append("assistant 回复未正确闭合")
    
    elif format_type == "llama2":
        if text.count("[INST]") != text.count("[/INST]"):
            issues.append("INST 标签未闭合")
        if text.count("<s>") != text.count("</s>"):
            issues.append("s 标签未闭合")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


# ============ PPL 过滤相关 ============

# 轻量级 PPL 计算模型
PPL_MODELS = {
    "GPT-2 (Small)": {
        "id": "openai-community/gpt2",
        "description": ""
    },
    "DistilGPT-2": {
        "id": "distilgpt2",
        "description": ""
    }
}


def calculate_perplexity(
    text: str,
    model: Any,
    tokenizer: Any,
    max_length: int = 512,
    stride: int = 256
) -> Tuple[float, Dict]:
    """
    计算文本的 Perplexity (困惑度)
    
    Args:
        text: 输入文本
        model: 语言模型
        tokenizer: 分词器
        max_length: 最大序列长度
        stride: 滑动窗口步长
    
    Returns:
        (ppl, details): 困惑度值和详细信息
    """
    import torch
    
    # 编码文本
    encodings = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids
    
    seq_len = input_ids.size(1)
    
    if seq_len == 0:
        return float('inf'), {"error": "Empty text"}
    
    # 如果序列很短，直接计算
    if seq_len <= max_length:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            ppl = math.exp(loss)
        
        return ppl, {
            "loss": loss,
            "seq_length": seq_len,
            "method": "direct"
        }
    
    # 长序列使用滑动窗口
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # 计算实际预测的 token 数
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        
        # Mask 掉重叠部分
        target_chunk[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc >= seq_len:
            break
    
    total_nll = torch.stack(nlls).sum()
    ppl = math.exp(total_nll / (seq_len - 1))
    
    return ppl, {
        "loss": (total_nll / (seq_len - 1)).item(),
        "seq_length": seq_len,
        "num_windows": len(nlls),
        "method": "sliding_window"
    }


def batch_calculate_ppl(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    max_length: int = 512
) -> List[Dict]:
    """
    批量计算多个文本的 PPL
    
    Returns:
        List of {text_preview, ppl, details}
    """
    results = []
    
    for text in texts:
        ppl, details = calculate_perplexity(text, model, tokenizer, max_length)
        results.append({
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "ppl": ppl,
            "details": details
        })
    
    return results


def filter_by_ppl(
    texts: List[str],
    model: Any,
    tokenizer: Any,
    min_ppl: float = 0,
    max_ppl: float = 1000,
    max_length: int = 512
) -> Tuple[List[str], List[str], List[Dict]]:
    """
    根据 PPL 阈值过滤文本
    
    Args:
        texts: 文本列表
        model: 语言模型
        tokenizer: 分词器
        min_ppl: 最小 PPL 阈值
        max_ppl: 最大 PPL 阈值
        max_length: 最大序列长度
    
    Returns:
        (accepted_texts, rejected_texts, all_results)
    """
    accepted = []
    rejected = []
    all_results = []
    
    for text in texts:
        ppl, details = calculate_perplexity(text, model, tokenizer, max_length)
        
        result = {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "ppl": ppl,
            "accepted": min_ppl <= ppl <= max_ppl,
            "details": details
        }
        all_results.append(result)
        
        if min_ppl <= ppl <= max_ppl:
            accepted.append(text)
        else:
            rejected.append(text)
    
    return accepted, rejected, all_results


def get_ppl_quality_label(ppl: float) -> Tuple[str, str]:
    """
    根据 PPL 值返回质量标签和颜色
    
    Returns:
        (label, color)
    """
    if ppl < 50:
        return "优秀", "#059669"  # 绿色
    elif ppl < 100:
        return "良好", "#2563EB"  # 蓝色
    elif ppl < 300:
        return "一般", "#D97706"  # 橙色
    elif ppl < 1000:
        return "较差", "#DC2626"  # 红色
    else:
        return "异常", "#7C3AED"  # 紫色

