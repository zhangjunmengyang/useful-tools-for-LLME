"""
PEFT 参数计算器 - 计算 LoRA/QLoRA 的可训练参数量
"""

import streamlit as st
import pandas as pd

# 模型配置
MODEL_CONFIGS = {
    "Llama-2-7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32, "intermediate_size": 11008},
    "Llama-2-13B": {"hidden_size": 5120, "num_layers": 40, "num_heads": 40, "intermediate_size": 13824},
    "Llama-2-70B": {"hidden_size": 8192, "num_layers": 80, "num_heads": 64, "intermediate_size": 28672},
    "Llama-3-8B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32, "intermediate_size": 14336},
    "Qwen-7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32, "intermediate_size": 11008},
    "Mistral-7B": {"hidden_size": 4096, "num_layers": 32, "num_heads": 32, "intermediate_size": 14336},
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


def calculate_lora_params(hidden_size: int, rank: int, num_layers: int, modules: list) -> dict:
    """计算 LoRA 参数量"""
    params_per_layer = 0
    details = []
    
    for module in modules:
        if module in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            # Attention 模块: hidden_size -> hidden_size
            module_params = 2 * hidden_size * rank  # A + B
            params_per_layer += module_params
            details.append({"模块": module, "每层参数": module_params})
        elif module in ["gate_proj", "up_proj", "down_proj"]:
            # FFN 模块比较复杂，简化处理
            module_params = 2 * hidden_size * rank
            params_per_layer += module_params
            details.append({"模块": module, "每层参数": module_params})
    
    total_params = params_per_layer * num_layers
    
    return {
        "total_params": total_params,
        "params_per_layer": params_per_layer,
        "details": details
    }


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">PEFT 参数计算器</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    💡 计算 LoRA/QLoRA 的可训练参数量，评估微调成本。
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 模型配置")
        
        model_choice = st.selectbox("选择模型", list(MODEL_CONFIGS.keys()))
        config = MODEL_CONFIGS[model_choice]
        
        st.info(f"""
        **{model_choice}**
        - Hidden: {config['hidden_size']}
        - Layers: {config['num_layers']}
        - Heads: {config['num_heads']}
        """)
        
        st.markdown("### LoRA 参数")
        
        rank = st.slider("Rank (r)", 4, 256, 16, help="LoRA 低秩维度")
        alpha = st.slider("Alpha (α)", 8, 512, 32, help="缩放因子")
        
        st.markdown("### 目标模块")
        
        selected_modules = []
        for module_id, module_name in TARGET_MODULES.items():
            if st.checkbox(module_name, value=module_id in ["q_proj", "v_proj"], key=f"mod_{module_id}"):
                selected_modules.append(module_id)
    
    with col2:
        st.markdown("### 计算结果")
        
        if selected_modules:
            result = calculate_lora_params(
                config['hidden_size'], rank, config['num_layers'], selected_modules
            )
            
            # 估算原始模型参数量 (简化)
            base_params = config['hidden_size'] * config['hidden_size'] * 4 * config['num_layers']  # 简化估算
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
            st.dataframe(df, hide_index=True)
            
            # 公式说明
            st.markdown("""
            ### 📐 计算公式
            
            ```
            LoRA 参数 = 2 × hidden_size × rank × num_modules × num_layers
            
            其中:
            - A 矩阵: hidden_size × rank
            - B 矩阵: rank × hidden_size
            - scaling = α / r
            ```
            """)
        else:
            st.warning("请选择至少一个目标模块")
    
    # QLoRA 说明
    st.markdown("---")
    st.markdown("""
    ### 🔧 QLoRA 特点
    
    | 特性 | LoRA | QLoRA |
    |------|------|-------|
    | 基座模型精度 | FP16/BF16 | INT4 (NF4) |
    | LoRA 权重精度 | FP16 | BF16 |
    | 显存占用 | ~16GB (7B) | ~6GB (7B) |
    | 训练速度 | 快 | 稍慢 (反量化) |
    
    QLoRA = 4-bit 量化 + LoRA + Double Quantization + Paged Optimizer
    """)

