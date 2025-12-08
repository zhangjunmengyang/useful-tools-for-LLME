"""
Attention - 可视化注意力权重
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from interpretability_lab.interpretability_utils import (
    INTERPRETABILITY_MODELS,
    load_model_with_attention,
    get_attention_weights,
    compute_attention_entropy,
    get_attention_patterns
)


def render_attention_heatmap(
    attention: np.ndarray,
    tokens: list,
    title: str = "Attention Weights"
) -> go.Figure:
    """渲染注意力热力图"""
    fig = go.Figure(data=go.Heatmap(
        z=attention,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        xaxis=dict(tickangle=45, side='bottom'),
        yaxis=dict(autorange='reversed'),
        height=500,
        width=600
    )
    
    return fig


def render_attention_grid(
    attention_weights: torch.Tensor,
    tokens: list,
    layer_idx: int,
    num_heads: int = 4
) -> go.Figure:
    """渲染多个 head 的注意力网格"""
    heads_to_show = min(num_heads, attention_weights.shape[0])
    
    fig = make_subplots(
        rows=1, cols=heads_to_show,
        subplot_titles=[f'Head {i}' for i in range(heads_to_show)],
        horizontal_spacing=0.05
    )
    
    for i in range(heads_to_show):
        attn = attention_weights[i].numpy()
        
        fig.add_trace(
            go.Heatmap(
                z=attn,
                colorscale='Blues',
                showscale=(i == heads_to_show - 1),
                hovertemplate='Q: %{y}<br>K: %{x}<br>W: %{z:.3f}<extra></extra>'
            ),
            row=1, col=i+1
        )
        
        fig.update_xaxes(
            tickvals=list(range(len(tokens))),
            ticktext=tokens,
            tickangle=45,
            row=1, col=i+1
        )
        fig.update_yaxes(
            tickvals=list(range(len(tokens))),
            ticktext=tokens if i == 0 else [],
            autorange='reversed',
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=f"Layer {layer_idx} - Attention Heads",
        height=400,
        margin=dict(l=100, r=50, t=80, b=100)
    )
    
    return fig


def render_token_attention_flow(
    attention_weights: torch.Tensor,
    tokens: list,
    selected_token_idx: int
) -> go.Figure:
    """渲染选中 token 的注意力流向"""
    # 平均所有层所有头
    avg_attention = attention_weights.mean(dim=(0, 1)).numpy()
    
    # 选中 token 作为 query 时关注的 keys
    query_attention = avg_attention[selected_token_idx, :]
    
    # 选中 token 作为 key 时被哪些 queries 关注
    key_attention = avg_attention[:, selected_token_idx]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f'"{tokens[selected_token_idx]}" 关注哪些 tokens',
            f'哪些 tokens 关注 "{tokens[selected_token_idx]}"'
        ]
    )
    
    # Query → Keys
    colors1 = ['#2563EB' if i == selected_token_idx else '#60A5FA' for i in range(len(tokens))]
    fig.add_trace(
        go.Bar(x=tokens, y=query_attention, marker_color=colors1, name='Query Attention'),
        row=1, col=1
    )
    
    # Keys → Query
    colors2 = ['#DC2626' if i == selected_token_idx else '#F87171' for i in range(len(tokens))]
    fig.add_trace(
        go.Bar(x=tokens, y=key_attention, marker_color=colors2, name='Key Attention'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">Attention</h1>', unsafe_allow_html=True)
    
    
    # 模型选择
    model_choice = st.selectbox(
        "选择模型",
        options=list(INTERPRETABILITY_MODELS.keys())
    )
    
    model_info = INTERPRETABILITY_MODELS[model_choice]
    
    with st.spinner(f"加载 {model_choice}..."):
        model, tokenizer = load_model_with_attention(model_info['id'])
    
    if model is None:
        st.error("模型加载失败")
        return
    
    st.markdown("---")
    
    # 输入文本
    default_text = "The animal didn't cross the street because it was too tired"
    text = st.text_area(
        "输入文本",
        value=default_text,
        height=80,
        help="经典的指代消解例子：'it' 指代 'animal' 还是 'street'?"
    )
    
    if not text:
        st.info("请输入文本")
        return
    
    # 获取注意力权重
    with st.spinner("计算注意力权重..."):
        attention_weights, tokens = get_attention_weights(model, tokenizer, text)
    
    st.caption(f"序列长度: {len(tokens)} tokens")
    
    # 显示 tokens
    with st.expander("查看分词结果"):
        st.write(tokens)
    
    # Causal Mask 开关
    use_causal = st.checkbox(
        "应用 Causal Mask",
        value=True,
        help="Decoder-only 模型使用下三角掩码，每个位置只能看到之前的 token"
    )
    
    if use_causal:
        seq_len = attention_weights.shape[-1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        # 注意：attention 已经是 softmax 后的，这里只是可视化效果
        attention_display = attention_weights.clone()
        # 将 mask 外的设为 0
        attention_display = attention_display * causal_mask.unsqueeze(0).unsqueeze(0)
    else:
        attention_display = attention_weights
    
    # Tab 切换
    tab1, tab2, tab3 = st.tabs(["热力图", "Token 分析", "模式分析"])
    
    with tab1:
        col_layer, col_head = st.columns(2)
        
        with col_layer:
            layer_idx = st.selectbox(
                "选择层",
                options=list(range(model_info['layers'])),
                format_func=lambda x: f"Layer {x}"
            )
        
        with col_head:
            head_idx = st.selectbox(
                "选择 Head",
                options=["All Heads"] + list(range(model_info['heads'])),
                format_func=lambda x: f"Head {x}" if isinstance(x, int) else x
            )
        
        if head_idx == "All Heads":
            # 显示多个 head 的网格
            fig = render_attention_grid(
                attention_display[layer_idx],
                tokens,
                layer_idx,
                num_heads=4
            )
            st.plotly_chart(fig, width='stretch')
        else:
            # 显示单个 head 的热力图
            attn = attention_display[layer_idx, head_idx].numpy()
            fig = render_attention_heatmap(
                attn,
                tokens,
                f"Layer {layer_idx}, Head {head_idx}"
            )
            st.plotly_chart(fig, width='stretch')
        
        # Causal Mask 对比
        st.markdown("### Causal Mask vs Full Attention")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Decoder-only (Causal Mask)**")
            st.markdown("""
            ```
            1 0 0 0 0
            1 1 0 0 0
            1 1 1 0 0
            1 1 1 1 0
            1 1 1 1 1
            ```
            每个位置只能看到自己和之前的 token
            """)
        
        with col_b:
            st.markdown("**Encoder (Full Attention)**")
            st.markdown("""
            ```
            1 1 1 1 1
            1 1 1 1 1
            1 1 1 1 1
            1 1 1 1 1
            1 1 1 1 1
            ```
            每个位置可以看到所有 token
            """)
    
    with tab2:
        st.markdown("### Token 注意力分析")
        
        # 选择 token
        selected_idx = st.selectbox(
            "选择要分析的 Token",
            options=list(range(len(tokens))),
            format_func=lambda x: f'{x}: "{tokens[x]}"'
        )
        
        # 显示注意力流向
        fig = render_token_attention_flow(attention_display, tokens, selected_idx)
        st.plotly_chart(fig, width='stretch')
        
        
        # 各层注意力变化
        st.markdown("### 各层对选中 Token 的注意力")
        
        layer_attention = []
        for l in range(model_info['layers']):
            # 平均所有头
            avg_attn = attention_display[l].mean(dim=0).numpy()
            # 选中 token 作为 query 时的注意力分布
            layer_attention.append(avg_attn[selected_idx, :])
        
        layer_attn_matrix = np.array(layer_attention)
        
        fig_layers = go.Figure(data=go.Heatmap(
            z=layer_attn_matrix,
            x=tokens,
            y=[f'Layer {i}' for i in range(model_info['layers'])],
            colorscale='Viridis'
        ))
        
        fig_layers.update_layout(
            title=f'"{tokens[selected_idx]}" 在各层的注意力分布',
            xaxis_title="Key Token",
            yaxis_title="Layer",
            height=400,
            xaxis=dict(tickangle=45)
        )
        
        st.plotly_chart(fig_layers, width='stretch')
    
    with tab3:
        st.markdown("### 注意力模式分析")
        
        # 分析各层各头的模式
        patterns_data = []
        
        for l in range(model_info['layers']):
            for h in range(model_info['heads']):
                attn = attention_display[l, h].numpy()
                patterns = get_attention_patterns(torch.tensor(attn))
                patterns_data.append({
                    'Layer': l,
                    'Head': h,
                    '对角线': patterns['diagonal'],
                    '首 Token': patterns['first_token'],
                    '局部': patterns['local'],
                    '全局': patterns['global']
                })
        
        df_patterns = pd.DataFrame(patterns_data)
        
        # 汇总统计
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("平均对角线注意力", f"{df_patterns['对角线'].mean():.2%}")
        with col2:
            st.metric("平均首 Token 注意力", f"{df_patterns['首 Token'].mean():.2%}")
        with col3:
            st.metric("平均局部注意力", f"{df_patterns['局部'].mean():.2%}")
        with col4:
            st.metric("平均全局注意力", f"{df_patterns['全局'].mean():.2%}")
        
        # 热力图展示各头的模式
        st.markdown("### 各 Head 的注意力模式")
        
        # 重塑为 (layers, heads) 的矩阵
        diagonal_matrix = df_patterns.pivot(index='Layer', columns='Head', values='对角线')
        
        fig_pattern = go.Figure(data=go.Heatmap(
            z=diagonal_matrix.values,
            x=[f'H{i}' for i in range(model_info['heads'])],
            y=[f'L{i}' for i in range(model_info['layers'])],
            colorscale='RdBu',
            zmid=0.5,
            hovertemplate='Layer %{y}, Head %{x}<br>Diagonal Attention: %{z:.2%}<extra></extra>'
        ))
        
        fig_pattern.update_layout(
            title="对角线注意力强度 (每个 token 关注自己的程度)",
            xaxis_title="Head",
            yaxis_title="Layer",
            height=400
        )
        
        st.plotly_chart(fig_pattern, width='stretch')
        
        # 熵分析
        st.markdown("### 注意力熵分析")
        st.markdown("熵越高表示注意力越分散，熵越低表示注意力越集中")
        
        entropy_data = []
        for l in range(model_info['layers']):
            for h in range(model_info['heads']):
                attn = attention_display[l, h]
                entropy = compute_attention_entropy(attn).mean().item()
                entropy_data.append({
                    'Layer': l,
                    'Head': h,
                    'Entropy': entropy
                })
        
        df_entropy = pd.DataFrame(entropy_data)
        entropy_matrix = df_entropy.pivot(index='Layer', columns='Head', values='Entropy')
        
        fig_entropy = go.Figure(data=go.Heatmap(
            z=entropy_matrix.values,
            x=[f'H{i}' for i in range(model_info['heads'])],
            y=[f'L{i}' for i in range(model_info['layers'])],
            colorscale='Plasma'
        ))
        
        fig_entropy.update_layout(
            title="注意力熵 (值越大表示注意力越分散)",
            xaxis_title="Head",
            yaxis_title="Layer",
            height=400
        )
        
        st.plotly_chart(fig_entropy, width='stretch')

