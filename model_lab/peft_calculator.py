"""
PEFT 参数计算器 - 计算 LoRA/QLoRA 的可训练参数量
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
            ("Llama-2-13B", "meta-llama/Llama-2-13b-hf"),
            ("Llama-2-70B", "meta-llama/Llama-2-70b-hf"),
            ("Llama-3.1-8B", "meta-llama/Llama-3.1-8B"),
            ("Llama-3.2-3B", "meta-llama/Llama-3.2-3B"),
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

# 可训练模块
TARGET_MODULES = {
    "q_proj": "Query 投影",
    "k_proj": "Key 投影",
    "v_proj": "Value 投影",
    "o_proj": "Output 投影",
    "gate_proj": "FFN Gate",
    "up_proj": "FFN Up",
    "down_proj": "FFN Down",
}


# 全局配置缓存
_config_cache = {'config': None, 'display_name': None}


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


def extract_model_config(config: dict) -> dict:
    """从 HF config 中提取计算所需的关键配置"""
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    intermediate_size = config.get("intermediate_size", hidden_size * 4)
    head_dim = config.get("head_dim", hidden_size // num_heads)
    
    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "intermediate_size": intermediate_size,
        "head_dim": head_dim,
    }


def calculate_lora_params(config: dict, rank: int, modules: list) -> dict:
    """计算 LoRA 参数量"""
    hidden_size = config["hidden_size"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    intermediate_size = config["intermediate_size"]
    head_dim = config["head_dim"]
    
    params_per_layer = 0
    details = []
    
    for module in modules:
        if module == "q_proj":
            out_dim = num_heads * head_dim
            module_params = hidden_size * rank + rank * out_dim
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{hidden_size}->{out_dim}", "每层参数": module_params})
        elif module in ["k_proj", "v_proj"]:
            out_dim = num_kv_heads * head_dim
            module_params = hidden_size * rank + rank * out_dim
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{hidden_size}->{out_dim}", "每层参数": module_params})
        elif module == "o_proj":
            in_dim = num_heads * head_dim
            module_params = in_dim * rank + rank * hidden_size
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{in_dim}->{hidden_size}", "每层参数": module_params})
        elif module in ["gate_proj", "up_proj"]:
            module_params = hidden_size * rank + rank * intermediate_size
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{hidden_size}->{intermediate_size}", "每层参数": module_params})
        elif module == "down_proj":
            module_params = intermediate_size * rank + rank * hidden_size
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{intermediate_size}->{hidden_size}", "每层参数": module_params})
    
    total_params = params_per_layer * num_layers
    
    return {
        "total_params": total_params,
        "params_per_layer": params_per_layer,
        "details": details
    }


def estimate_base_params(config: dict) -> int:
    """估算原始模型参数量"""
    d = config["hidden_size"]
    L = config["num_layers"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    ff = config["intermediate_size"]
    
    q_proj = d * (num_heads * head_dim)
    kv_proj = d * (num_kv_heads * head_dim) * 2
    o_proj = (num_heads * head_dim) * d
    attention = (q_proj + kv_proj + o_proj) * L
    
    ffn = 3 * d * ff * L
    
    return attention + ffn


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


def load_model_config(input_mode: str, category: str, preset_model: str, custom_model: str, token: str, progress=gr.Progress()):
    """加载模型配置"""
    global _config_cache
    
    if input_mode == "预设模型":
        model_id = get_model_id(category, preset_model)
        display_name = preset_model
    else:
        model_id = custom_model
        display_name = custom_model.split("/")[-1] if custom_model else None
    
    if not model_id:
        return "请选择或输入模型", ""
    
    try:
        progress(0.5, desc=f"加载 {display_name} 配置...")
        raw_config = load_config_from_hub(model_id, token)
        config = extract_model_config(raw_config)
        
        _config_cache['config'] = config
        _config_cache['display_name'] = display_name
        
        info_text = f"""
**{display_name}**
- Hidden: {config['hidden_size']:,}
- Layers: {config['num_layers']}
- Heads: {config['num_heads']} (KV: {config['num_kv_heads']})
- FFN: {config['intermediate_size']:,}
        """
        
        return "配置加载成功", info_text
        
    except Exception as e:
        return f"加载失败: {str(e)}", ""


def calculate_results(rank: int, alpha: int, selected_modules: list):
    """计算 LoRA 参数"""
    config = _config_cache.get('config')
    if not config:
        return "", None, "", "", "", "", "", ""
    
    if not selected_modules:
        return "请选择至少一个目标模块", None, "", "", "", "", "", ""
    
    result = calculate_lora_params(config, rank, selected_modules)
    base_params = estimate_base_params(config)
    trainable_ratio = result['total_params'] / base_params * 100
    
    # 详细表格
    df = pd.DataFrame(result['details'])
    df['总参数'] = df['每层参数'] * config['num_layers']
    df['总参数'] = df['总参数'].apply(lambda x: f"{x:,}")
    df['每层参数'] = df['每层参数'].apply(lambda x: f"{x:,}")
    
    # 显存估算
    lora_mem_fp16 = result['total_params'] * 2 / 1024 / 1024
    lora_mem_fp32 = result['total_params'] * 4 / 1024 / 1024
    train_mem = lora_mem_fp32 + lora_mem_fp32 * 2
    
    scale_factor = f"{alpha/rank:.2f}"
    
    return (
        "",
        df,
        f"{result['total_params']:,}",
        f"{result['total_params'] * 2 / 1024 / 1024:.2f}",
        f"~{trainable_ratio:.3f}%",
        f"{lora_mem_fp16:.2f} MB",
        f"{lora_mem_fp32:.2f} MB",
        f"{train_mem:.2f} MB"
    )


def render():
    """渲染页面"""
    
    gr.Markdown("## PEFT 参数计算器")
    
    # 默认值
    default_category = "Meta (Llama)"
    default_model = "Llama-2-7B"
    default_rank = 16
    default_alpha = 32
    default_modules = ["q_proj", "v_proj"]
    
    with gr.Row():
        # 左列：模型选择
        with gr.Column(scale=1):
            gr.Markdown("### 模型选择")
            
            input_mode = gr.Radio(
                label="输入方式",
                choices=["预设模型", "自定义模型"],
                value="预设模型"
            )
            
            category = gr.Dropdown(
                label="选择厂商",
                choices=list(MODEL_CATEGORIES.keys()),
                value=default_category,
                visible=True
            )
            
            preset_model = gr.Dropdown(
                label="选择模型",
                choices=get_model_list(default_category),
                value=default_model,
                visible=True
            )
            
            custom_model = gr.Textbox(
                label="模型名称或 URL",
                placeholder="例如: meta-llama/Llama-2-7b-hf",
                visible=False
            )
            
            token = gr.Textbox(
                label="HF Token (可选)",
                type="password"
            )
            
            load_status = gr.Markdown("")
            model_info = gr.Markdown("")
        
        # 右列：LoRA 配置 & 结果
        with gr.Column(scale=2):
            gr.Markdown("### LoRA 参数")
            
            with gr.Row():
                rank = gr.Slider(
                    label="Rank (r)",
                    minimum=4,
                    maximum=256,
                    value=default_rank,
                    step=4
                )
                alpha = gr.Slider(
                    label="Alpha",
                    minimum=8,
                    maximum=512,
                    value=default_alpha,
                    step=8
                )
            
            gr.Markdown("### 目标模块")
            
            module_choices = [(name, module_id) for module_id, name in TARGET_MODULES.items()]
            selected_modules = gr.CheckboxGroup(
                label="选择目标模块",
                choices=module_choices,
                value=default_modules
            )
            
            calc_status = gr.Markdown("")
            
            gr.Markdown("### 计算结果")
            
            with gr.Row():
                total_params = gr.Textbox(label="LoRA 参数量", interactive=False)
                params_mb = gr.Textbox(label="参数量 (MB)", interactive=False)
                trainable_ratio = gr.Textbox(label="可训练比例", interactive=False)
            
            gr.Markdown("### 参数分布")
            params_table = gr.Dataframe(label="详细参数分布")
            
            gr.Markdown("### 显存估算")
            with gr.Row():
                mem_fp16 = gr.Textbox(label="推理 (fp16)", interactive=False)
                mem_fp32 = gr.Textbox(label="训练权重 (fp32)", interactive=False)
                mem_total = gr.Textbox(label="训练总计 (含优化器)", interactive=False)
            
            gr.Markdown("*注: 以上仅为 LoRA 参数的显存占用，不包括基础模型、激活值等。*")
    
    # 事件绑定
    def toggle_input_mode(mode):
        preset_visible = mode == "预设模型"
        return (
            gr.update(visible=preset_visible),
            gr.update(visible=preset_visible),
            gr.update(visible=not preset_visible)
        )
    
    def update_model_list(cat):
        models = get_model_list(cat)
        return gr.update(choices=models, value=models[0] if models else None)
    
    # 加载配置并自动计算
    def load_and_calculate(input_mode, category, preset_model, custom_model, token, rank, alpha, selected_modules):
        """加载模型配置并自动计算参数"""
        status, info = load_model_config(input_mode, category, preset_model, custom_model, token)
        if "成功" in status:
            calc_result = calculate_results(rank, alpha, selected_modules)
            return (status, info) + calc_result
        return status, info, "", None, "", "", "", "", "", ""
    
    input_mode.change(
        fn=toggle_input_mode,
        inputs=[input_mode],
        outputs=[category, preset_model, custom_model]
    )
    
    category.change(
        fn=update_model_list,
        inputs=[category],
        outputs=[preset_model]
    )
    
    # 模型选择变化时自动加载并计算
    load_calc_inputs = [input_mode, category, preset_model, custom_model, token, rank, alpha, selected_modules]
    load_calc_outputs = [load_status, model_info, calc_status, params_table, total_params, params_mb, trainable_ratio, mem_fp16, mem_fp32, mem_total]
    
    preset_model.change(fn=load_and_calculate, inputs=load_calc_inputs, outputs=load_calc_outputs)
    custom_model.submit(fn=load_and_calculate, inputs=load_calc_inputs, outputs=load_calc_outputs)
    
    # LoRA 参数变化时自动计算（如果配置已加载）
    calc_outputs = [calc_status, params_table, total_params, params_mb, trainable_ratio, mem_fp16, mem_fp32, mem_total]
    rank.change(fn=calculate_results, inputs=[rank, alpha, selected_modules], outputs=calc_outputs)
    alpha.change(fn=calculate_results, inputs=[rank, alpha, selected_modules], outputs=calc_outputs)
    selected_modules.change(fn=calculate_results, inputs=[rank, alpha, selected_modules], outputs=calc_outputs)
    
    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        return load_and_calculate("预设模型", default_category, default_model, "", "", default_rank, default_alpha, default_modules)
    
    # 返回 load 事件需要的信息供主 app 调用
    return {
        'load_fn': on_load,
        'load_outputs': load_calc_outputs
    }
