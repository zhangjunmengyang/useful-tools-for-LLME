"""
PEFT 参数计算器 - 计算 LoRA/QLoRA 的可训练参数量
支持从 HuggingFace Hub 实时读取配置
"""

import streamlit as st
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


@st.cache_data(show_spinner=False, ttl=3600)
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
            # Q: hidden_size -> num_heads * head_dim
            out_dim = num_heads * head_dim
            module_params = hidden_size * rank + rank * out_dim  # A + B
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{hidden_size}→{out_dim}", "每层参数": module_params})
        elif module in ["k_proj", "v_proj"]:
            # K/V: hidden_size -> num_kv_heads * head_dim (GQA)
            out_dim = num_kv_heads * head_dim
            module_params = hidden_size * rank + rank * out_dim
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{hidden_size}→{out_dim}", "每层参数": module_params})
        elif module == "o_proj":
            # O: num_heads * head_dim -> hidden_size
            in_dim = num_heads * head_dim
            module_params = in_dim * rank + rank * hidden_size
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{in_dim}→{hidden_size}", "每层参数": module_params})
        elif module in ["gate_proj", "up_proj"]:
            # gate/up: hidden_size -> intermediate_size
            module_params = hidden_size * rank + rank * intermediate_size
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{hidden_size}→{intermediate_size}", "每层参数": module_params})
        elif module == "down_proj":
            # down: intermediate_size -> hidden_size
            module_params = intermediate_size * rank + rank * hidden_size
            params_per_layer += module_params
            details.append({"模块": module, "维度": f"{intermediate_size}→{hidden_size}", "每层参数": module_params})
    
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
    
    # Attention
    q_proj = d * (num_heads * head_dim)
    kv_proj = d * (num_kv_heads * head_dim) * 2
    o_proj = (num_heads * head_dim) * d
    attention = (q_proj + kv_proj + o_proj) * L
    
    # FFN (SwiGLU: gate, up, down)
    ffn = 3 * d * ff * L
    
    return attention + ffn


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">PEFT 参数计算器</h1>', unsafe_allow_html=True)
    
    
    col1, col2 = st.columns([1, 2])
    
    # ========== 左列：模型选择 ==========
    with col1:
        st.markdown("### 模型选择")
        
        # 输入方式选择
        input_mode = st.radio("输入方式", ["预设模型", "自定义模型"], horizontal=True)
        
        model_name = None
        display_name = None
        
        if input_mode == "预设模型":
            categories = list(MODEL_CATEGORIES.keys())
            selected_category = st.selectbox("选择厂商", categories)
            
            models = MODEL_CATEGORIES[selected_category]["models"]
            model_names = [m[0] for m in models]
            selected_model = st.selectbox("选择模型", model_names)
            
            for name, model_id in models:
                if name == selected_model:
                    model_name = model_id
                    display_name = name
                    break
            
            st.caption(f"`{model_name}`")
        else:
            model_name = st.text_input(
                "模型名称或 URL",
                placeholder="例如: meta-llama/Llama-2-7b-hf"
            )
            display_name = model_name.split("/")[-1] if model_name else None
        
        # HF Token
        token = st.text_input("HF Token (可选)", type="password")
        
        # 加载按钮
        load_clicked = st.button("加载配置", type="primary", width="stretch")
        
        if load_clicked and model_name:
            with st.spinner(f"正在加载 {display_name} 的配置..."):
                try:
                    raw_config = load_config_from_hub(model_name, token)
                    config = extract_model_config(raw_config)
                    st.session_state["peft_config"] = config
                    st.session_state["peft_display_name"] = display_name
                    st.success("配置加载成功")
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
    
    # ========== 右列：LoRA 配置 & 结果 ==========
    with col2:
        if "peft_config" not in st.session_state:
            return
        
        config = st.session_state["peft_config"]

        # 显示已加载的模型配置
        if "peft_config" in st.session_state:
            config = st.session_state["peft_config"]
            loaded_name = st.session_state.get("peft_display_name", "模型")
            
            st.info(f"""
            **{loaded_name}**
            - Hidden: {config['hidden_size']:,}
            - Layers: {config['num_layers']}
            - Heads: {config['num_heads']} (KV: {config['num_kv_heads']})
            - FFN: {config['intermediate_size']:,}
            """)
        
        # LoRA 参数配置
        st.markdown("### LoRA 参数")
        
        lora_col1, lora_col2 = st.columns(2)
        with lora_col1:
            rank = st.slider("Rank (r)", 4, 256, 16, help="LoRA 低秩维度")
        with lora_col2:
            alpha = st.slider("Alpha (α)", 8, 512, 32, help="缩放因子")
        
        st.caption(f"缩放系数: α/r = {alpha/rank:.2f}")
        
        # 目标模块选择
        st.markdown("### 目标模块")
        
        mod_cols = st.columns(4)
        selected_modules = []
        module_items = list(TARGET_MODULES.items())
        for i, (module_id, module_name) in enumerate(module_items):
            with mod_cols[i % 4]:
                if st.checkbox(module_name, value=module_id in ["q_proj", "v_proj"], key=f"mod_{module_id}"):
                    selected_modules.append(module_id)
        
        st.markdown("---")
        
        # 计算结果
        if not selected_modules:
            st.warning("请选择至少一个目标模块")
            return
        
        st.markdown("### 计算结果")
        
        result = calculate_lora_params(config, rank, selected_modules)
        
        # 估算原始模型参数量
        base_params = estimate_base_params(config)
        trainable_ratio = result['total_params'] / base_params * 100
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("LoRA 参数量", f"{result['total_params']:,}")
        with col_b:
            st.metric("参数量 (MB)", f"{result['total_params'] * 2 / 1024 / 1024:.2f}")
        with col_c:
            st.metric("可训练比例", f"~{trainable_ratio:.3f}%")
        
        # 详细表格
        st.markdown("### 参数分布")
        df = pd.DataFrame(result['details'])
        df['总参数'] = df['每层参数'] * config['num_layers']
        df['总参数'] = df['总参数'].apply(lambda x: f"{x:,}")
        df['每层参数'] = df['每层参数'].apply(lambda x: f"{x:,}")
        st.dataframe(df, hide_index=True, width="stretch")
        
        # 显存估算
        st.markdown("### 显存估算")
        
        lora_mem_fp16 = result['total_params'] * 2 / 1024 / 1024  # MB
        lora_mem_fp32 = result['total_params'] * 4 / 1024 / 1024  # MB
        train_mem = lora_mem_fp32 + lora_mem_fp32 * 2  # weights + optimizer (AdamW)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("推理 (fp16)", f"{lora_mem_fp16:.2f} MB")
        with col_m2:
            st.metric("训练权重 (fp32)", f"{lora_mem_fp32:.2f} MB")
        with col_m3:
            st.metric("训练总计 (含优化器)", f"{train_mem:.2f} MB")
        
        st.caption("注: 以上仅为 LoRA 参数的显存占用，不包括基础模型、激活值等。")
