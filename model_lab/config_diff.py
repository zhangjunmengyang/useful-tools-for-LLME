"""
Config 差异对比 - 对比两个模型的 config.json
"""

import streamlit as st
import json
from transformers import AutoConfig

# 预定义配置
PRESET_CONFIGS = {
    "Llama-2-7B": {
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 32, "intermediate_size": 11008,
        "max_position_embeddings": 4096, "rope_theta": 10000, "vocab_size": 32000
    },
    "Llama-3-8B": {
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 14336,
        "max_position_embeddings": 8192, "rope_theta": 500000, "vocab_size": 128256
    },
    "Qwen-7B": {
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 32, "intermediate_size": 11008,
        "max_position_embeddings": 8192, "rope_theta": 10000, "vocab_size": 151936
    },
    "Mistral-7B": {
        "hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 8, "intermediate_size": 14336,
        "max_position_embeddings": 32768, "rope_theta": 10000, "vocab_size": 32000,
        "sliding_window": 4096
    },
}

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
}


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">Config 差异对比</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    💡 对比两个模型的架构配置，快速了解模型演进。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 模型 A")
        model_a = st.selectbox("选择模型", list(PRESET_CONFIGS.keys()), key="model_a")
        config_a = PRESET_CONFIGS[model_a]
    
    with col2:
        st.markdown("### 模型 B")
        model_b = st.selectbox("选择模型", list(PRESET_CONFIGS.keys()), index=1, key="model_b")
        config_b = PRESET_CONFIGS[model_b]
    
    st.markdown("---")
    st.markdown("### 📊 配置对比")
    
    # 收集所有 key
    all_keys = set(config_a.keys()) | set(config_b.keys())
    
    # 构建对比表
    diff_data = []
    for key in sorted(all_keys):
        val_a = config_a.get(key, "N/A")
        val_b = config_b.get(key, "N/A")
        
        # 判断是否有差异
        is_diff = val_a != val_b
        
        diff_data.append({
            "配置项": key,
            "说明": KEY_DESCRIPTIONS.get(key, ""),
            model_a: val_a,
            model_b: val_b,
            "差异": "⚠️" if is_diff else "✅"
        })
    
    # 显示表格
    st.dataframe(diff_data, hide_index=True, width="stretch")
    
    # 关键差异分析
    st.markdown("### 🔍 关键差异分析")
    
    # GQA 分析
    gqa_a = config_a.get("num_key_value_heads", config_a.get("num_attention_heads"))
    gqa_b = config_b.get("num_key_value_heads", config_b.get("num_attention_heads"))
    heads_a = config_a.get("num_attention_heads", 32)
    heads_b = config_b.get("num_attention_heads", 32)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown(f"**{model_a}**")
        if gqa_a == heads_a:
            st.info("使用 MHA (Multi-Head Attention)")
        else:
            st.success(f"使用 GQA, KV 头数压缩 {heads_a // gqa_a}x")
    
    with col_b:
        st.markdown(f"**{model_b}**")
        if gqa_b == heads_b:
            st.info("使用 MHA (Multi-Head Attention)")
        else:
            st.success(f"使用 GQA, KV 头数压缩 {heads_b // gqa_b}x")
    
    # RoPE 分析
    rope_a = config_a.get("rope_theta", 10000)
    rope_b = config_b.get("rope_theta", 10000)
    
    if rope_a != rope_b:
        st.markdown(f"""
        **RoPE Base 差异**:
        - {model_a}: {rope_a:,}
        - {model_b}: {rope_b:,}
        - 更大的 base 支持更长的上下文外推
        """)
    
    # 参数量估算
    st.markdown("---")
    st.markdown("### 📐 参数量估算")
    
    def estimate_params(config):
        d = config['hidden_size']
        L = config['num_hidden_layers']
        V = config['vocab_size']
        ff = config['intermediate_size']
        
        # 简化估算
        attention = 4 * d * d * L  # QKV + O
        ffn = 3 * d * ff * L  # gate, up, down
        embed = V * d
        
        return (attention + ffn + embed) / 1e9
    
    params_a = estimate_params(config_a)
    params_b = estimate_params(config_b)
    
    col_1, col_2 = st.columns(2)
    with col_1:
        st.metric(f"{model_a} 估算参数", f"~{params_a:.1f}B")
    with col_2:
        st.metric(f"{model_b} 估算参数", f"~{params_b:.1f}B")

