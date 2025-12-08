"""
FFN 激活探测 - 分析 Feed-Forward 层的激活情况
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from interpretability_lab.interpretability_utils import (
    INTERPRETABILITY_MODELS,
    load_model_with_attention,
    analyze_ffn_activations,
    compare_activation_functions,
    MODEL_ARCHITECTURES
)


def render_activation_curves() -> go.Figure:
    """渲染激活函数曲线对比"""
    x = np.linspace(-4, 4, 200)
    activations = compare_activation_functions(x)
    
    fig = go.Figure()
    
    colors = ['#2563EB', '#DC2626', '#059669', '#7C3AED']
    
    for (name, values), color in zip(activations.items(), colors):
        fig.add_trace(go.Scatter(
            x=x,
            y=values,
            mode='lines',
            name=name,
            line=dict(color=color, width=2)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="激活函数对比",
        xaxis_title="输入 x",
        yaxis_title="输出 f(x)",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def render_activation_histogram(activations: np.ndarray, title: str) -> go.Figure:
    """渲染激活值直方图"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=activations.flatten(),
        nbinsx=50,
        marker_color='#2563EB',
        opacity=0.7
    ))
    
    # 添加零线
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="零点")
    
    fig.update_layout(
        title=title,
        xaxis_title="激活值",
        yaxis_title="频次",
        height=350
    )
    
    return fig


def render_neuron_activation_heatmap(activations: np.ndarray, top_k: int = 50) -> go.Figure:
    """渲染神经元激活热力图"""
    # activations: (seq_len, hidden_dim)
    
    # 选择激活最强的 top_k 个神经元
    mean_activation = np.abs(activations).mean(axis=0)
    top_indices = np.argsort(mean_activation)[-top_k:]
    
    selected_activations = activations[:, top_indices]
    
    fig = go.Figure(data=go.Heatmap(
        z=selected_activations.T,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title=f"Top {top_k} 最活跃神经元",
        xaxis_title="Token 位置",
        yaxis_title="神经元索引",
        height=400
    )
    
    return fig


def render_sparsity_analysis(activations: np.ndarray, thresholds: list) -> go.Figure:
    """渲染稀疏性分析"""
    sparsity_rates = []
    
    for thresh in thresholds:
        sparse_rate = (np.abs(activations) < thresh).mean() * 100
        sparsity_rates.append(sparse_rate)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f'< {t}' for t in thresholds],
        y=sparsity_rates,
        marker_color='#7C3AED',
        text=[f'{s:.1f}%' for s in sparsity_rates],
        textposition='outside'
    ))
    
    fig.update_layout(
        title="FFN 激活稀疏性",
        xaxis_title="阈值",
        yaxis_title="稀疏比例 (%)",
        height=350
    )
    
    return fig


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">FFN 激活探测</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    <b>Feed-Forward Network (FFN/MLP)</b> 占据 Transformer 约 2/3 的参数量，可能 FFN 层存储了大量的事实知识，呈现出稀疏激活特性。
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["激活函数", "激活分析", "架构对比"])
    
    with tab1:
        st.markdown("### 常见激活函数对比")
        
        fig_curves = render_activation_curves()
        st.plotly_chart(fig_curves, width='stretch')
        
        # 可视化 SwiGLU 门控
        st.markdown("### SwiGLU 门控可视化")
        
        x = np.linspace(-3, 3, 100)
        gate = 1 / (1 + np.exp(-x))  # Sigmoid gate
        silu = x * gate
        
        # 假设 W2 路径的输出
        np.random.seed(42)
        w2_out = 0.5 * x + 0.3  # 简化的线性变换
        
        swiglu_out = silu * w2_out
        
        fig = make_subplots(rows=1, cols=3, subplot_titles=['SiLU(xW₁)', 'xW₂', 'SwiGLU Output'])
        
        fig.add_trace(go.Scatter(x=x, y=silu, line=dict(color='#2563EB')), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=w2_out, line=dict(color='#059669')), row=1, col=2)
        fig.add_trace(go.Scatter(x=x, y=swiglu_out, line=dict(color='#DC2626')), row=1, col=3)
        
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.markdown("### 实时激活分析")
        
        model_choice = st.selectbox(
            "选择模型",
            options=list(INTERPRETABILITY_MODELS.keys()),
            key="ffn_model"
        )
        
        model_info = INTERPRETABILITY_MODELS[model_choice]
        
        with st.spinner(f"加载 {model_choice}..."):
            model, tokenizer = load_model_with_attention(model_info['id'])
        
        if model is None:
            st.error("模型加载失败")
            return
        
        st.success(f"✅ 模型已加载")
        
        # 输入文本
        col_code, col_text = st.columns(2)
        
        with col_code:
            code_input = st.text_area(
                "代码输入",
                value="def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                height=120,
                key="code_input"
            )
        
        with col_text:
            text_input = st.text_area(
                "自然语言输入",
                value="The sun rises in the east and sets in the west. Birds fly south for the winter.",
                height=120,
                key="text_input"
            )
        
        # 分析层选择
        layer_idx = st.slider("选择分析的层", 0, model_info['layers'] - 1, 0)
        
        if st.button("分析激活", type="primary"):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### 代码输入的激活")
                code_results = analyze_ffn_activations(model, tokenizer, code_input, layer_idx)
                
                if 'fc1' in code_results:
                    fc1_data = code_results['fc1']
                    
                    # 统计指标
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        st.metric("均值", f"{fc1_data['mean']:.4f}")
                    with metrics_cols[1]:
                        st.metric("标准差", f"{fc1_data['std']:.4f}")
                    with metrics_cols[2]:
                        st.metric("稀疏率", f"{fc1_data['sparsity']:.1%}")
                    
                    # 直方图
                    fig_hist = render_activation_histogram(fc1_data['values'], "代码激活分布")
                    st.plotly_chart(fig_hist, width='stretch')
            
            with col_b:
                st.markdown("#### 自然语言输入的激活")
                text_results = analyze_ffn_activations(model, tokenizer, text_input, layer_idx)
                
                if 'fc1' in text_results:
                    fc1_data = text_results['fc1']
                    
                    metrics_cols = st.columns(3)
                    with metrics_cols[0]:
                        st.metric("均值", f"{fc1_data['mean']:.4f}")
                    with metrics_cols[1]:
                        st.metric("标准差", f"{fc1_data['std']:.4f}")
                    with metrics_cols[2]:
                        st.metric("稀疏率", f"{fc1_data['sparsity']:.1%}")
                    
                    fig_hist = render_activation_histogram(fc1_data['values'], "自然语言激活分布")
                    st.plotly_chart(fig_hist, width='stretch')
            
            # 稀疏性对比
            st.markdown("### 稀疏性对比")
            
            thresholds = [0.01, 0.1, 0.5, 1.0]
            
            fig_sparse = make_subplots(rows=1, cols=2, subplot_titles=['代码', '自然语言'])
            
            if 'fc1' in code_results:
                code_sparse = [(np.abs(code_results['fc1']['values']) < t).mean() * 100 for t in thresholds]
                fig_sparse.add_trace(go.Bar(x=[f'<{t}' for t in thresholds], y=code_sparse, marker_color='#2563EB'), row=1, col=1)
            
            if 'fc1' in text_results:
                text_sparse = [(np.abs(text_results['fc1']['values']) < t).mean() * 100 for t in thresholds]
                fig_sparse.add_trace(go.Bar(x=[f'<{t}' for t in thresholds], y=text_sparse, marker_color='#059669'), row=1, col=2)
            
            fig_sparse.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_sparse, width='stretch')
    
    with tab3:
        st.markdown("### 模型架构对比")
        
        # 架构对比表
        arch_data = []
        for model_name, arch in MODEL_ARCHITECTURES.items():
            arch_data.append({
                "模型": model_name,
                "Attention": arch['attention'],
                "FFN 激活": arch['ffn'],
                "归一化": arch['norm'],
                "位置编码": arch['position']
            })
        
        st.dataframe(pd.DataFrame(arch_data), hide_index=True, width="stretch")
        
        st.markdown("---")
        st.markdown("### 参数量分析")
        
        # 可视化不同组件的参数占比
        fig_params = go.Figure(data=[go.Pie(
            labels=['FFN (W₁, W₂)', 'Attention (Q, K, V, O)', 'Embedding', 'LayerNorm'],
            values=[66, 25, 8, 1],
            marker_colors=['#2563EB', '#059669', '#D97706', '#6B7280'],
            hole=0.4
        )])
        
        fig_params.update_layout(
            title="Transformer 参数分布 (典型 Decoder-only 模型)",
            height=400
        )
        
        st.plotly_chart(fig_params, width='stretch')
