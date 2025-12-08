"""
Config 差异对比 - 对比两个模型的 config.json
支持从 HuggingFace Hub 实时读取配置
"""

import streamlit as st
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

# 生成扁平的预设列表（用于兼容）
def _build_preset_list():
    """构建 (display_name, model_id) 的扁平列表"""
    result = []
    for category_data in MODEL_CATEGORIES.values():
        for item in category_data["models"]:
            result.append(item)
    return result

PRESET_MODELS = _build_preset_list()

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


@st.cache_data(show_spinner=False, ttl=3600)
def load_config_from_hub(model_name: str, token: str = None) -> dict:
    """
    从 HuggingFace Hub 加载模型配置
    
    Args:
        model_name: 模型名称或 URL
        token: HF API token（可选）
        
    Returns:
        配置字典
    """
    model_name = extract_from_url(model_name)
    
    # 直接下载 config.json 文件
    config_path = hf_hub_download(
        repo_id=model_name,
        filename="config.json",
        token=token if token else None
    )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    return config


def render_model_selector(col, key_prefix: str, default_category_idx: int = 0, default_model_idx: int = 0):
    """
    渲染模型选择器（支持按厂商分类）
    
    Args:
        col: streamlit column
        key_prefix: 用于区分 A/B 的前缀
        default_category_idx: 默认选择的厂商索引
        default_model_idx: 默认选择的模型索引
        
    Returns:
        (model_id, display_name, token)
    """
    with col:
        # 输入方式选择
        input_mode = st.radio(
            "输入方式",
            ["预设模型", "自定义模型"],
            key=f"{key_prefix}_mode",
            horizontal=True
        )
        
        model_name = None
        display_name = None
        
        if input_mode == "预设模型":
            # 厂商选择
            categories = list(MODEL_CATEGORIES.keys())
            selected_category = st.selectbox(
                "选择厂商",
                categories,
                index=default_category_idx,
                key=f"{key_prefix}_category"
            )
            
            # 模型选择
            models = MODEL_CATEGORIES[selected_category]["models"]
            model_names = [m[0] for m in models]
            selected_model = st.selectbox(
                "选择模型",
                model_names,
                index=min(default_model_idx, len(model_names) - 1),
                key=f"{key_prefix}_model"
            )
            
            # 找到对应的 model_id
            for name, model_id in models:
                if name == selected_model:
                    model_name = model_id
                    display_name = name
                    break
            
            st.caption(f"`{model_name}`")
        else:
            model_name = st.text_input(
                "模型名称或 URL",
                placeholder="例如: meta-llama/Llama-2-7b-hf",
                key=f"{key_prefix}_custom"
            )
            display_name = model_name.split("/")[-1] if model_name else None
        
        # Token 输入（可选）
        token = st.text_input(
            "HF Token (可选，用于私有模型)",
            type="password",
            key=f"{key_prefix}_token"
        )
        
        return model_name, display_name, token


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">Config 差异对比</h1>', unsafe_allow_html=True)
    
    
    col1, col2 = st.columns(2)
    
    # 模型 A (默认: Meta/Llama-2-7B)
    with col1:
        st.markdown("### 模型 A")
    model_a_name, display_a, token_a = render_model_selector(col1, "model_a", 0, 0)
    
    # 模型 B (默认: Meta/Llama-3.1-8B)
    with col2:
        st.markdown("### 模型 B")
    model_b_name, display_b, token_b = render_model_selector(col2, "model_b", 0, 1)
    
    # 加载按钮
    st.markdown("---")
    
    if not model_a_name or not model_b_name:
        st.warning("请选择或输入两个模型进行对比")
        return
    
    col_btn, col_opt = st.columns([1, 3])
    with col_btn:
        load_clicked = st.button("加载配置", type="primary", width="stretch")
    with col_opt:
        show_all = st.checkbox("显示全部配置项", value=False)
    
    # 加载配置
    if load_clicked:
        config_a = None
        config_b = None
        
        with st.spinner(f"正在加载 {display_a} 的配置..."):
            try:
                config_a = load_config_from_hub(model_a_name, token_a)
                st.session_state["config_a"] = config_a
                st.session_state["display_a"] = display_a
            except Exception as e:
                st.error(f"加载 {display_a} 失败: {str(e)}")
                return
        
        with st.spinner(f"正在加载 {display_b} 的配置..."):
            try:
                config_b = load_config_from_hub(model_b_name, token_b)
                st.session_state["config_b"] = config_b
                st.session_state["display_b"] = display_b
            except Exception as e:
                st.error(f"加载 {display_b} 失败: {str(e)}")
                return
        
    
    # 从 session state 获取配置
    config_a = st.session_state.get("config_a")
    config_b = st.session_state.get("config_b")
    display_a = st.session_state.get("display_a", "模型 A")
    display_b = st.session_state.get("display_b", "模型 B")
    
    if not config_a or not config_b:
        return
    
    # 显示配置对比
    st.markdown("### 配置对比")
    
    # 决定要显示的 key
    if show_all:
        all_keys = set(config_a.keys()) | set(config_b.keys())
        # 过滤掉一些不太有用的 key
        exclude_keys = {"_name_or_path", "transformers_version", "_commit_hash", "auto_map"}
        all_keys = all_keys - exclude_keys
    else:
        all_keys = set(KEY_CONFIGS)
    
    # 构建对比表
    diff_data = []
    for key in sorted(all_keys):
        val_a = config_a.get(key, "N/A")
        val_b = config_b.get(key, "N/A")
        
        # 格式化值
        val_a_str = format_value(val_a)
        val_b_str = format_value(val_b)
        
        # 判断是否有差异
        is_diff = val_a != val_b
        
        diff_data.append({
            "配置项": key,
            "说明": KEY_DESCRIPTIONS.get(key, ""),
            display_a: val_a_str,
            display_b: val_b_str,
            "差异": "Yes" if is_diff else ""
        })
    
    # 显示表格
    st.dataframe(diff_data, hide_index=True, width="stretch")
    
    # 关键差异分析
    render_analysis(config_a, config_b, display_a, display_b)
    
    # 原始 JSON 展示
    with st.expander("查看原始配置 JSON"):
        col_json1, col_json2 = st.columns(2)
        with col_json1:
            st.markdown(f"**{display_a}**")
            st.json(config_a)
        with col_json2:
            st.markdown(f"**{display_b}**")
            st.json(config_b)


def format_value(val):
    """格式化配置值为字符串"""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    elif isinstance(val, bool):
        return "Yes" if val else "No"
    elif isinstance(val, (int, float)) and val >= 10000:
        return f"{val:,}"
    return str(val)


def render_analysis(config_a: dict, config_b: dict, name_a: str, name_b: str):
    """渲染关键差异分析"""
    st.markdown("### 关键差异分析")
    
    # GQA 分析
    gqa_a = config_a.get("num_key_value_heads", config_a.get("num_attention_heads"))
    gqa_b = config_b.get("num_key_value_heads", config_b.get("num_attention_heads"))
    heads_a = config_a.get("num_attention_heads", 32)
    heads_b = config_b.get("num_attention_heads", 32)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"**{name_a}**")
        if gqa_a and heads_a:
            if gqa_a == heads_a:
                st.info("使用 MHA (Multi-Head Attention)")
            elif gqa_a == 1:
                st.warning("使用 MQA (Multi-Query Attention)")
            else:
                ratio = heads_a // gqa_a if gqa_a else 1
                st.success(f"使用 GQA, KV 头数压缩 {ratio}x")
    
    with col_b:
        st.markdown(f"**{name_b}**")
        if gqa_b and heads_b:
            if gqa_b == heads_b:
                st.info("使用 MHA (Multi-Head Attention)")
            elif gqa_b == 1:
                st.warning("使用 MQA (Multi-Query Attention)")
            else:
                ratio = heads_b // gqa_b if gqa_b else 1
                st.success(f"使用 GQA, KV 头数压缩 {ratio}x")
    
    # RoPE 分析
    rope_a = config_a.get("rope_theta")
    rope_b = config_b.get("rope_theta")
    
    if rope_a and rope_b and rope_a != rope_b:
        st.markdown(f"""
        **RoPE Base 差异**:
        - {name_a}: `{rope_a:,}`
        - {name_b}: `{rope_b:,}`
        - 更大的 base 支持更长的上下文外推
        """)
    
    # 参数量估算
    st.markdown("---")
    st.markdown("### 参数量估算")
    
    params_a = estimate_params(config_a)
    params_b = estimate_params(config_b)
    
    col_1, col_2 = st.columns(2)
    with col_1:
        if params_a:
            st.metric(f"{name_a} 估算参数", f"~{params_a:.2f}B")
        else:
            st.metric(f"{name_a} 估算参数", "N/A")
    with col_2:
        if params_b:
            st.metric(f"{name_b} 估算参数", f"~{params_b:.2f}B")
        else:
            st.metric(f"{name_b} 估算参数", "N/A")


def estimate_params(config: dict) -> float:
    """
    估算模型参数量（单位：B）
    
    Args:
        config: 模型配置字典
        
    Returns:
        参数量（单位：十亿）
    """
    try:
        d = config.get('hidden_size')
        L = config.get('num_hidden_layers')
        V = config.get('vocab_size')
        ff = config.get('intermediate_size')
        
        if not all([d, L, V, ff]):
            return None
        
        # 考虑 GQA
        num_heads = config.get('num_attention_heads', 32)
        num_kv_heads = config.get('num_key_value_heads', num_heads)
        head_dim = config.get('head_dim', d // num_heads)
        
        # Attention: Q + K + V + O
        q_proj = d * (num_heads * head_dim)
        kv_proj = d * (num_kv_heads * head_dim) * 2  # K + V
        o_proj = (num_heads * head_dim) * d
        attention = (q_proj + kv_proj + o_proj) * L
        
        # FFN: gate, up, down (for SwiGLU)
        ffn = 3 * d * ff * L
        
        # Embedding
        tie_embeddings = config.get('tie_word_embeddings', False)
        embed = V * d * (1 if tie_embeddings else 2)
        
        # Layer norms
        ln = 2 * d * L + d  # 每层 2 个 LN + 最后一个 LN
        
        total = (attention + ffn + embed + ln) / 1e9
        return total
    except Exception:
        return None
