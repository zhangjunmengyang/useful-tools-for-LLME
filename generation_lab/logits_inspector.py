"""
Logits - 可视化 Next Token 预测
展示 Logits、Temperature、Top-P/Top-K 采样策略
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
from generation_lab.generation_utils import (
    DEMO_MODELS,
    load_model_and_tokenizer,
    get_next_token_logits,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    get_sampling_distribution
)


def render_probability_bar_chart(
    tokens: list,
    probabilities: list,
    title: str = "Token 概率分布",
    cutoff_line: float = None,
    cutoff_label: str = None,
    top_k_cutoff: int = None
):
    """渲染概率柱状图"""
    fig = go.Figure()
    
    # 主柱状图
    colors = ['#2563EB' if i < (top_k_cutoff or len(tokens)) else '#E5E7EB' 
              for i in range(len(tokens))]
    
    fig.add_trace(go.Bar(
        x=tokens,
        y=probabilities,
        marker_color=colors,
        text=[f'{p:.2%}' for p in probabilities],
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    # Top-P 截断线
    if cutoff_line is not None:
        # 计算累积概率对应的位置
        cumsum = np.cumsum(probabilities)
        cutoff_idx = np.searchsorted(cumsum, cutoff_line)
        if cutoff_idx < len(tokens):
            fig.add_hline(
                y=probabilities[cutoff_idx],
                line_dash="dash",
                line_color="#DC2626",
                annotation_text=cutoff_label or f"Top-P 截断 ({cutoff_line:.0%})",
                annotation_position="right"
            )
    
    # Top-K 截断线
    if top_k_cutoff is not None and top_k_cutoff < len(tokens):
        fig.add_vline(
            x=top_k_cutoff - 0.5,
            line_dash="dash",
            line_color="#D97706",
            annotation_text=f"Top-K={top_k_cutoff}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Token",
        yaxis_title="概率",
        yaxis_tickformat='.1%',
        height=400,
        margin=dict(l=50, r=50, t=50, b=100),
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def render_temperature_comparison(logits: torch.Tensor, temperatures: list, tokenizer, top_k: int = 10):
    """渲染不同温度下的概率分布对比"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for idx, temp in enumerate(temperatures):
        scaled_logits = logits.clone() / temp if temp > 0 else logits.clone()
        probs = torch.softmax(scaled_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        tokens = [tokenizer.decode([i.item()]) for i in top_indices]
        prob_values = top_probs.tolist()
        
        fig.add_trace(go.Bar(
            name=f'T={temp}',
            x=tokens,
            y=prob_values,
            marker_color=colors[idx % len(colors)],
            opacity=0.8
        ))
    
    fig.update_layout(
        title="Temperature 对概率分布的影响",
        xaxis_title="Token",
        yaxis_title="概率",
        yaxis_tickformat='.1%',
        barmode='group',
        height=450,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def render_entropy_gauge(probs: torch.Tensor) -> go.Figure:
    """渲染熵值仪表盘"""
    # 计算熵
    probs_np = probs.numpy()
    probs_np = probs_np[probs_np > 0]  # 过滤零概率
    entropy = -np.sum(probs_np * np.log2(probs_np))
    max_entropy = np.log2(len(probs))
    normalized_entropy = entropy / max_entropy
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=entropy,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "分布熵 (bits)", 'font': {'size': 14}},
        delta={'reference': max_entropy / 2, 'increasing': {'color': "#D97706"}},
        gauge={
            'axis': {'range': [0, max_entropy], 'tickwidth': 1},
            'bar': {'color': "#2563EB"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_entropy * 0.3], 'color': '#D1FAE5'},  # 低熵 - 确定
                {'range': [max_entropy * 0.3, max_entropy * 0.7], 'color': '#FEF3C7'},  # 中熵
                {'range': [max_entropy * 0.7, max_entropy], 'color': '#FEE2E2'}  # 高熵 - 不确定
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_entropy
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">Logits</h1>', unsafe_allow_html=True)
    
    # 模型选择
    col_model, col_status = st.columns([3, 1])
    with col_model:
        model_choice = st.selectbox(
            "选择演示模型",
            options=list(DEMO_MODELS.keys()),
            help="选择一个轻量级模型用于演示（首次加载需要下载）"
        )
    
    model_info = DEMO_MODELS[model_choice]
    
    with col_status:
        st.caption(model_info['description'])
    
    # 加载模型
    with st.spinner(f"加载 {model_choice}..."):
        model, tokenizer = load_model_and_tokenizer(model_info['id'])
    
    if model is None or tokenizer is None:
        st.error("模型加载失败，请检查网络连接或选择其他模型")
        return
    
    st.markdown("---")
    
    # 输入区域
    prompt = st.text_area(
        "输入 Prompt",
        value="The quick brown fox jumps over the",
        height=100,
        placeholder="输入文本，模型将预测下一个 token..."
    )
    
    if not prompt:
        st.info("请输入 Prompt 文本")
        return
    
    # 获取原始 logits
    with st.spinner("计算中..."):
        token_candidates = get_next_token_logits(model, tokenizer, prompt, top_k=50)
    
    # 创建 tabs
    tab1, tab2, tab3 = st.tabs(["概率分布", "Temperature 实验", "Top-K/Top-P 截断"])
    
    with tab1:
        st.markdown("### Next Token 候选词 Top-50")
        
        # 展示指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Top-1 Token", f'"{token_candidates[0]["token_str"]}"')
        with col2:
            st.metric("Top-1 概率", f'{token_candidates[0]["probability"]:.2%}')
        with col3:
            st.metric("Top-1 Logit", f'{token_candidates[0]["logit"]:.2f}')
        with col4:
            # 计算 Top-5 累积概率
            top5_prob = sum(t['probability'] for t in token_candidates[:5])
            st.metric("Top-5 累积概率", f'{top5_prob:.2%}')
        
        # 概率柱状图
        tokens_display = [t['token_str'][:10] + ('...' if len(t['token_str']) > 10 else '') 
                        for t in token_candidates[:20]]
        probs = [t['probability'] for t in token_candidates[:20]]
        
        fig = render_probability_bar_chart(tokens_display, probs, "Top-20 Token 概率分布")
        st.plotly_chart(fig, width='stretch')
        
        # 详细表格
        with st.expander("详细数据 (Top-50)"):
            df = pd.DataFrame([{
                "排名": t['rank'],
                "Token": repr(t['token_str']),
                "Raw Token": t['raw_token'],
                "Token ID": t['token_id'],
                "Logit": f"{t['logit']:.4f}",
                "概率": f"{t['probability']:.4%}"
            } for t in token_candidates])
            st.dataframe(df, width="stretch", hide_index=True)
    
    with tab2:
        
        # 温度滑块
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="调整温度参数观察概率分布变化"
        )
        
        # 获取原始 logits tensor
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
        
        # 应用温度
        scaled_logits = logits / temperature
        scaled_probs = torch.softmax(scaled_logits, dim=-1)
        original_probs = torch.softmax(logits, dim=-1)
        
        # 对比显示
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**当前温度 T={temperature}**")
            top_scaled_probs, top_indices = torch.topk(scaled_probs, 10)
            tokens = [tokenizer.decode([i.item()]) for i in top_indices]
            
            fig1 = render_probability_bar_chart(
                tokens, 
                top_scaled_probs.tolist(),
                f"T={temperature} 概率分布"
            )
            st.plotly_chart(fig1, width='stretch')
        
        with col2:
            st.markdown("**原始分布 T=1.0**")
            top_orig_probs, top_orig_indices = torch.topk(original_probs, 10)
            orig_tokens = [tokenizer.decode([i.item()]) for i in top_orig_indices]
            
            fig2 = render_probability_bar_chart(
                orig_tokens,
                top_orig_probs.tolist(),
                "T=1.0 原始分布"
            )
            st.plotly_chart(fig2, width='stretch')
        
        # 多温度对比
        st.markdown("### 多温度对比")
        temps = [0.3, 0.7, 1.0, 1.5, 2.0]
        fig_compare = render_temperature_comparison(logits, temps, tokenizer, top_k=8)
        st.plotly_chart(fig_compare, width='stretch')
    
    with tab3:
        st.markdown("### Top-K / Top-P 采样截断")
        
        col_k, col_p = st.columns(2)
        
        with col_k:
            top_k = st.slider(
                "Top-K",
                min_value=1,
                max_value=50,
                value=10,
                help="只从概率最高的 K 个 token 中采样"
            )
        
        with col_p:
            top_p = st.slider(
                "Top-P (Nucleus)",
                min_value=0.1,
                max_value=1.0,
                value=0.9,
                step=0.05,
                help="只从累积概率达到 P 的 token 中采样"
            )
        
        # 应用采样策略
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :].clone()
        
        original_probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(original_probs, 50)
        
        # 计算 Top-P 截断位置
        cumsum_probs = torch.cumsum(top_probs, dim=-1)
        top_p_cutoff = torch.searchsorted(cumsum_probs, top_p).item() + 1
        
        # 实际截断位置（取 Top-K 和 Top-P 的较小值）
        effective_cutoff = min(top_k, top_p_cutoff)
        
        st.markdown(f"""
        **采样结果**:
        - Top-K 截断: 保留前 **{top_k}** 个 token
        - Top-P 截断: 累积概率达到 **{top_p:.0%}** 需要 **{top_p_cutoff}** 个 token
        - 有效采样范围: **{effective_cutoff}** 个 token
        """)
        
        # 可视化
        tokens_display = [tokenizer.decode([i.item()])[:8] for i in top_indices[:30]]
        probs_list = top_probs[:30].tolist()
        
        fig = go.Figure()
        
        # 根据是否在截断范围内设置颜色
        colors = []
        for i in range(30):
            if i < effective_cutoff:
                colors.append('#2563EB')  # 蓝色 - 在采样范围内
            elif i < top_k:
                colors.append('#D97706')  # 橙色 - 在 Top-K 但不在 Top-P
            else:
                colors.append('#E5E7EB')  # 灰色 - 被截断
        
        fig.add_trace(go.Bar(
            x=tokens_display,
            y=probs_list,
            marker_color=colors,
            text=[f'{p:.1%}' for p in probs_list],
            textposition='outside',
            textfont=dict(size=9)
        ))
        
        # Top-K 线
        if top_k <= 30:
            fig.add_vline(x=top_k - 0.5, line_dash="dash", line_color="#D97706",
                         annotation_text=f"Top-K={top_k}", annotation_position="top")
        
        # Top-P 线
        if top_p_cutoff <= 30:
            fig.add_vline(x=top_p_cutoff - 0.5, line_dash="dot", line_color="#DC2626",
                         annotation_text=f"Top-P={top_p}", annotation_position="bottom")
        
        fig.update_layout(
            title="采样截断可视化",
            xaxis_title="Token",
            yaxis_title="概率",
            yaxis_tickformat='.1%',
            height=450,
            margin=dict(l=50, r=50, t=60, b=100)
        )
        fig.update_xaxes(tickangle=45)
        
        st.plotly_chart(fig, width='stretch')
        
        # 图例说明
        st.markdown("""
        <div style="display: flex; gap: 20px; margin-top: 10px;">
            <span><span style="color: #2563EB;">■</span> 有效采样范围</span>
            <span><span style="color: #D97706;">■</span> Top-K 内但超出 Top-P</span>
            <span><span style="color: #E5E7EB;">■</span> 被截断 (不参与采样)</span>
        </div>
        """, unsafe_allow_html=True)
        
        # 累积概率曲线
        st.markdown("### 累积概率曲线 (CDF)")
        
        fig_cdf = go.Figure()
        fig_cdf.add_trace(go.Scatter(
            x=list(range(1, 51)),
            y=cumsum_probs.tolist(),
            mode='lines+markers',
            name='累积概率',
            marker=dict(size=6),
            line=dict(color='#2563EB')
        ))
        
        fig_cdf.add_hline(y=top_p, line_dash="dash", line_color="#DC2626",
                        annotation_text=f"Top-P={top_p}")
        
        fig_cdf.update_layout(
            title="Top-50 Token 累积概率分布",
            xaxis_title="Token 数量",
            yaxis_title="累积概率",
            yaxis_tickformat='.0%',
            height=350
        )
        
        st.plotly_chart(fig_cdf, width='stretch')

