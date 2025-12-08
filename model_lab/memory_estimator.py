"""
ModelLab - 显存估算
估算模型推理和训练所需的显存大小
"""

import streamlit as st
import pandas as pd
from accelerate.utils import convert_bytes

from model_lab.model_utils import (
    DTYPE_OPTIONS,
    LIBRARY_OPTIONS,
    get_model,
    calculate_memory_detailed,
    format_training_stages,
    get_model_error_message,
)


def render():
    """渲染显存估算页面"""
    
    st.markdown('<h1 class="module-title">显存估算</h1>', unsafe_allow_html=True)
    
    
    # st.divider()
    
    # 输入区域
    col_input, col_settings = st.columns([2, 1])
    
    with col_input:
        model_name = st.text_input(
            "模型名称或 URL",
            value="bert-base-cased",
            placeholder="例如: bert-base-cased, meta-llama/Llama-2-7b-hf",
            help="可以输入 HuggingFace 模型名称或模型页面 URL"
        )
    
    with col_settings:
        library = st.selectbox(
            "模型库",
            options=LIBRARY_OPTIONS,
            index=0,
        )
    
    # 精度选择
    col_dtype, col_token = st.columns([2, 1])
    
    with col_dtype:
        selected_dtypes = st.multiselect(
            "选择精度类型",
            options=DTYPE_OPTIONS,
            default=["float32"],
        )
    
    with col_token:
        access_token = st.text_input(
            "API Token (可选)",
            type="password",
            placeholder="用于访问私有模型",
            help="HuggingFace API Token"
        )
    
    # 计算按钮
    calculate_btn = st.button("计算显存", type="primary", width="stretch")
    
    st.divider()
    
    # 计算结果
    if calculate_btn:
        if not model_name:
            st.error("请输入模型名称")
            return
        
        if not selected_dtypes:
            st.error("请至少选择一种精度类型")
            return
        
        with st.spinner(f"正在加载模型 `{model_name}` 并计算显存..."):
            try:
                # 获取模型
                token = access_token if access_token else None
                model = get_model(model_name, library, token)
                
                # 计算内存
                data, stages = calculate_memory_detailed(model, selected_dtypes)
                
            except Exception as e:
                error_msg = get_model_error_message(e, model_name)
                st.error(error_msg)
                return
        
        # 显示结果标题
        st.markdown(f"### 模型 `{model_name}` 的显存需求")
        
        # 主要结果表格
        df_main = pd.DataFrame(data)
        df_main.columns = ["精度", "最大层/残差组", "模型总大小", "训练峰值显存 (Adam)"]
        
        st.dataframe(
            df_main,
            width="stretch",
            hide_index=True,
        )
        
        # 详细训练阶段说明
        training_stages_data = format_training_stages(stages, selected_dtypes)
        
        if training_stages_data:
            with st.expander("训练各阶段显存详情", expanded=False):
                st.markdown("""
                使用 batch size = 1 训练时，各阶段的预期显存占用:
                
                | 阶段 | 说明 |
                |------|------|
                | **Model** | 加载模型参数 |
                | **Gradient calculation** | 前向传播计算梯度 |
                | **Backward pass** | 反向传播 |
                | **Optimizer step** | 优化器更新 (Adam 需要 2x 参数大小) |
                """)
                
                df_stages = pd.DataFrame(training_stages_data)
                df_stages.columns = ["精度", "模型加载", "梯度计算", "反向传播", "优化器更新"]
                
                st.dataframe(
                    df_stages,
                    width="stretch",
                    hide_index=True,
                )
        
        # 使用建议
        st.markdown("---")
        st.markdown("### 使用建议")
        
        col_tips1, col_tips2 = st.columns(2)
        
        with col_tips1:
            st.markdown("""
            **推理部署:**
            - 使用 `device_map="auto"` 可以自动分配模型到多个设备
            - 最小显存需求为"最大层"大小
            - 建议预留 20% 额外显存用于 KV Cache 等
            """)
        
        with col_tips2:
            st.markdown("""
            **训练微调:**
            - 全参数微调需要约 4x 模型大小的显存
            - 使用 LoRA/QLoRA 可大幅降低显存需求
            - 混合精度训练 (fp16/bf16) 可减少约 50% 显存
            """)
        
        # 公式说明
        with st.expander("计算公式说明", expanded=False):
            st.markdown("""
            **显存估算公式:**
            
            1. **模型参数内存** = 参数数量 × 每参数字节数
               - float32: 4 bytes/param
               - float16/bfloat16: 2 bytes/param
               - int8: 1 byte/param
               - int4: 0.5 bytes/param
            
            2. **训练峰值显存 (Adam)** ≈ 4 × 模型大小
               - 1x: 模型参数
               - 1x: 梯度
               - 2x: 优化器状态 (Adam 的 m 和 v)
            
            3. **推理显存** ≈ 模型大小 + KV Cache + 激活值
               - 通常预留 20% 缓冲
            
            > 注: 实际显存可能因 batch size、序列长度、框架实现等因素有所不同
            """)

