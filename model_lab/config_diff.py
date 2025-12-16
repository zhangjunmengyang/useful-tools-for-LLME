"""
Config 差异对比 - 对比两个模型的 config.json
支持从 HuggingFace Hub 实时读取配置
"""

import gradio as gr
import pandas as pd
import json
from huggingface_hub import hf_hub_download
from model_lab.model_utils import extract_from_url


# 按厂商/系列分类的预设模型
MODEL_CATEGORIES = {
    "Meta (Llama)": {
        "models": [
            ("Llama-2-7B", "meta-llama/Llama-2-7b-hf"),
            ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
            ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B"),
            ("Llama-4-Scout-17B", "meta-llama/Llama-4-Scout-17B-16E-Instruct"),
        ],
    },
    "Alibaba (Qwen)": {
        "models": [
            ("Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
            ("Qwen2.5-3B", "Qwen/Qwen2.5-3B"),
            ("Qwen3-8B", "Qwen/Qwen3-8B"),
        ],
    },
    "DeepSeek": {
        "models": [
            ("DeepSeek-V3", "deepseek-ai/DeepSeek-V3"),
            ("DeepSeek-R1", "deepseek-ai/DeepSeek-R1"),
        ],
    },
    "Mistral": {
        "models": [
            ("Mistral-7B-v0.1", "mistralai/Mistral-7B-v0.1"),
            ("Mistral-7B-v0.3", "mistralai/Mistral-7B-v0.3"),
        ],
    },
    "Google": {
        "models": [
            ("Gemma-2-2B", "google/gemma-2-2b"),
            ("Gemma-7B", "google/gemma-7b"),
        ],
    },
    "Microsoft": {
        "models": [
            ("Phi-3-mini-4k", "microsoft/Phi-3-mini-4k-instruct"),
            ("Phi-4", "microsoft/phi-4"),
        ],
    },
}

# 关键配置项说明
KEY_DESCRIPTIONS = {
    "hidden_size": "隐藏层维度",
    "num_hidden_layers": "Transformer 层数",
    "num_attention_heads": "注意力头数 (Q)",
    "num_key_value_heads": "KV 头数 (GQA)",
    "intermediate_size": "FFN 中间维度",
    "max_position_embeddings": "最大位置编码",
    "rope_theta": "RoPE Base",
    "vocab_size": "词表大小",
    "sliding_window": "滑动窗口大小",
    "head_dim": "注意力头维度",
    "rms_norm_eps": "RMSNorm epsilon",
    "tie_word_embeddings": "共享词嵌入",
    "torch_dtype": "默认精度",
    "architectures": "模型架构",
    "model_type": "模型类型",
    "hidden_act": "激活函数",
    "attention_dropout": "注意力 Dropout",
    "attention_bias": "注意力偏置",
    "mlp_bias": "MLP 偏置",
}

# 重点展示的配置项
KEY_CONFIGS = [
    "hidden_size",
    "num_hidden_layers", 
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "max_position_embeddings",
    "rope_theta",
    "vocab_size",
    "head_dim",
    "hidden_act",
    "tie_word_embeddings",
]


# 全局配置缓存
_config_cache = {
    'config_a': None,
    'config_b': None,
    'display_a': None,
    'display_b': None
}


def load_config_from_hub(model_name: str, token: str = None) -> dict:
    """从 HuggingFace Hub 加载模型配置"""
    model_name = extract_from_url(model_name)
    config_path = hf_hub_download(
        repo_id=model_name,
        filename="config.json",
        token=token if token else None
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_model_list(category: str):
    """获取指定厂商的模型列表"""
    if category not in MODEL_CATEGORIES:
        return []
    return [m[0] for m in MODEL_CATEGORIES[category]["models"]]


def get_model_id(category: str, model_name: str):
    """获取模型 ID"""
    if category not in MODEL_CATEGORIES:
        return None
    for name, model_id in MODEL_CATEGORIES[category]["models"]:
        if name == model_name:
            return model_id
    return None


def format_value(val):
    """格式化配置值为字符串"""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    elif isinstance(val, bool):
        return "Yes" if val else "No"
    elif isinstance(val, (int, float)) and val >= 10000:
        return f"{val:,}"
    return str(val)


def estimate_params(config: dict) -> float:
    """估算模型参数量（单位：B）"""
    try:
        d = config.get('hidden_size')
        L = config.get('num_hidden_layers')
        V = config.get('vocab_size')
        ff = config.get('intermediate_size')
        
        if not all([d, L, V, ff]):
            return None
        
        num_heads = config.get('num_attention_heads', 32)
        num_kv_heads = config.get('num_key_value_heads', num_heads)
        head_dim = config.get('head_dim', d // num_heads)
        
        q_proj = d * (num_heads * head_dim)
        kv_proj = d * (num_kv_heads * head_dim) * 2
        o_proj = (num_heads * head_dim) * d
        attention = (q_proj + kv_proj + o_proj) * L
        
        ffn = 3 * d * ff * L
        
        tie_embeddings = config.get('tie_word_embeddings', False)
        embed = V * d * (1 if tie_embeddings else 2)
        
        ln = 2 * d * L + d
        
        total = (attention + ffn + embed + ln) / 1e9
        return total
    except Exception:
        return None


def load_configs(
    mode_a: str, cat_a: str, preset_a: str, custom_a: str, token_a: str,
    mode_b: str, cat_b: str, preset_b: str, custom_b: str, token_b: str,
    progress=gr.Progress()
):
    """加载两个模型的配置"""
    global _config_cache
    
    # 获取模型 A
    if mode_a == "预设模型":
        model_a_id = get_model_id(cat_a, preset_a)
        display_a = preset_a
    else:
        model_a_id = custom_a
        display_a = custom_a.split("/")[-1] if custom_a else None
    
    # 获取模型 B
    if mode_b == "预设模型":
        model_b_id = get_model_id(cat_b, preset_b)
        display_b = preset_b
    else:
        model_b_id = custom_b
        display_b = custom_b.split("/")[-1] if custom_b else None
    
    if not model_a_id or not model_b_id:
        return "请选择两个模型进行对比", None, "", "", ""
    
    try:
        progress(0.3, desc=f"加载 {display_a} 配置...")
        config_a = load_config_from_hub(model_a_id, token_a)
        
        progress(0.6, desc=f"加载 {display_b} 配置...")
        config_b = load_config_from_hub(model_b_id, token_b)
        
        _config_cache['config_a'] = config_a
        _config_cache['config_b'] = config_b
        _config_cache['display_a'] = display_a
        _config_cache['display_b'] = display_b
        
        return "配置加载成功", None, "", "", ""
        
    except Exception as e:
        return f"加载失败: {str(e)}", None, "", "", ""


def generate_comparison(show_all: bool):
    """生成配置对比"""
    config_a = _config_cache.get('config_a')
    config_b = _config_cache.get('config_b')
    display_a = _config_cache.get('display_a', '模型 A')
    display_b = _config_cache.get('display_b', '模型 B')
    
    if not config_a or not config_b:
        return None, "", "", ""
    
    # 决定要显示的 key
    if show_all:
        all_keys = set(config_a.keys()) | set(config_b.keys())
        exclude_keys = {"_name_or_path", "transformers_version", "_commit_hash", "auto_map"}
        all_keys = all_keys - exclude_keys
    else:
        all_keys = set(KEY_CONFIGS)
    
    # 构建对比表
    diff_data = []
    for key in sorted(all_keys):
        val_a = config_a.get(key, "N/A")
        val_b = config_b.get(key, "N/A")
        
        val_a_str = format_value(val_a)
        val_b_str = format_value(val_b)
        
        is_diff = val_a != val_b
        
        diff_data.append({
            "配置项": key,
            "说明": KEY_DESCRIPTIONS.get(key, ""),
            display_a: val_a_str,
            display_b: val_b_str,
            "差异": "Yes" if is_diff else ""
        })
    
    df = pd.DataFrame(diff_data)
    
    # 关键差异分析
    analysis = generate_analysis(config_a, config_b, display_a, display_b)
    
    # 参数量估算
    params_a = estimate_params(config_a)
    params_b = estimate_params(config_b)
    
    params_a_str = f"~{params_a:.2f}B" if params_a else "N/A"
    params_b_str = f"~{params_b:.2f}B" if params_b else "N/A"
    
    return df, analysis, params_a_str, params_b_str


def generate_analysis(config_a: dict, config_b: dict, name_a: str, name_b: str) -> str:
    """生成关键差异分析"""
    analysis_parts = []
    
    # GQA 分析
    gqa_a = config_a.get("num_key_value_heads", config_a.get("num_attention_heads"))
    gqa_b = config_b.get("num_key_value_heads", config_b.get("num_attention_heads"))
    heads_a = config_a.get("num_attention_heads", 32)
    heads_b = config_b.get("num_attention_heads", 32)
    
    def get_attention_type(gqa, heads):
        if not gqa or not heads:
            return "未知"
        if gqa == heads:
            return "MHA (Multi-Head Attention)"
        elif gqa == 1:
            return "MQA (Multi-Query Attention)"
        else:
            ratio = heads // gqa if gqa else 1
            return f"GQA (KV 头数压缩 {ratio}x)"
    
    analysis_parts.append(f"""
**注意力机制:**
- {name_a}: {get_attention_type(gqa_a, heads_a)}
- {name_b}: {get_attention_type(gqa_b, heads_b)}
    """)
    
    # RoPE 分析
    rope_a = config_a.get("rope_theta")
    rope_b = config_b.get("rope_theta")
    
    if rope_a and rope_b and rope_a != rope_b:
        analysis_parts.append(f"""
**RoPE Base 差异:**
- {name_a}: `{rope_a:,}`
- {name_b}: `{rope_b:,}`
- 更大的 base 支持更长的上下文外推
        """)
    
    return "\n".join(analysis_parts)


def render():
    """渲染页面"""
    
    gr.Markdown("## Config 差异对比")
    
    with gr.Row():
        # 模型 A 选择
        with gr.Column():
            gr.Markdown("### 模型 A")
            
            mode_a = gr.Radio(
                label="输入方式",
                choices=["预设模型", "自定义模型"],
                value="预设模型"
            )
            
            cat_a = gr.Dropdown(
                label="选择厂商",
                choices=list(MODEL_CATEGORIES.keys()),
                value="Meta (Llama)"
            )
            
            preset_a = gr.Dropdown(
                label="选择模型",
                choices=get_model_list("Meta (Llama)"),
                value="Llama-2-7B"
            )
            
            custom_a = gr.Textbox(
                label="模型名称或 URL",
                placeholder="例如: meta-llama/Llama-2-7b-hf",
                visible=False
            )
            
            token_a = gr.Textbox(
                label="HF Token (可选)",
                type="password"
            )
        
        # 模型 B 选择
        with gr.Column():
            gr.Markdown("### 模型 B")
            
            mode_b = gr.Radio(
                label="输入方式",
                choices=["预设模型", "自定义模型"],
                value="预设模型"
            )
            
            cat_b = gr.Dropdown(
                label="选择厂商",
                choices=list(MODEL_CATEGORIES.keys()),
                value="Meta (Llama)"
            )
            
            preset_b = gr.Dropdown(
                label="选择模型",
                choices=get_model_list("Meta (Llama)"),
                value="Llama-3.1-8B"
            )
            
            custom_b = gr.Textbox(
                label="模型名称或 URL",
                placeholder="例如: meta-llama/Llama-2-7b-hf",
                visible=False
            )
            
            token_b = gr.Textbox(
                label="HF Token (可选)",
                type="password"
            )
    
    gr.Markdown("---")
    
    with gr.Row():
        load_btn = gr.Button("加载配置", variant="primary")
        show_all = gr.Checkbox(label="显示全部配置项", value=False)
    
    load_status = gr.Markdown("")
    
    # 结果区域
    gr.Markdown("### 配置对比")
    diff_table = gr.Dataframe(label="对比表格")
    
    gr.Markdown("### 关键差异分析")
    analysis_text = gr.Markdown("")
    
    gr.Markdown("### 参数量估算")
    with gr.Row():
        params_a_display = gr.Textbox(label="模型 A 估算参数", interactive=False)
        params_b_display = gr.Textbox(label="模型 B 估算参数", interactive=False)
    
    # 事件绑定
    def toggle_mode_a(mode):
        return (
            gr.update(visible=(mode == "预设模型")),
            gr.update(visible=(mode == "预设模型")),
            gr.update(visible=(mode != "预设模型"))
        )
    
    def toggle_mode_b(mode):
        return (
            gr.update(visible=(mode == "预设模型")),
            gr.update(visible=(mode == "预设模型")),
            gr.update(visible=(mode != "预设模型"))
        )
    
    def update_models_a(cat):
        models = get_model_list(cat)
        return gr.update(choices=models, value=models[0] if models else None)
    
    def update_models_b(cat):
        models = get_model_list(cat)
        return gr.update(choices=models, value=models[0] if models else None)
    
    mode_a.change(fn=toggle_mode_a, inputs=[mode_a], outputs=[cat_a, preset_a, custom_a])
    mode_b.change(fn=toggle_mode_b, inputs=[mode_b], outputs=[cat_b, preset_b, custom_b])
    
    cat_a.change(fn=update_models_a, inputs=[cat_a], outputs=[preset_a])
    cat_b.change(fn=update_models_b, inputs=[cat_b], outputs=[preset_b])
    
    def on_load(mode_a, cat_a, preset_a, custom_a, token_a, mode_b, cat_b, preset_b, custom_b, token_b, show_all):
        status, _, _, _, _ = load_configs(
            mode_a, cat_a, preset_a, custom_a, token_a,
            mode_b, cat_b, preset_b, custom_b, token_b
        )
        if "成功" in status:
            df, analysis, params_a, params_b = generate_comparison(show_all)
            return status, df, analysis, params_a, params_b
        return status, None, "", "", ""
    
    load_btn.click(
        fn=on_load,
        inputs=[mode_a, cat_a, preset_a, custom_a, token_a, mode_b, cat_b, preset_b, custom_b, token_b, show_all],
        outputs=[load_status, diff_table, analysis_text, params_a_display, params_b_display]
    )
    
    show_all.change(
        fn=lambda show: generate_comparison(show),
        inputs=[show_all],
        outputs=[diff_table, analysis_text, params_a_display, params_b_display]
    )
