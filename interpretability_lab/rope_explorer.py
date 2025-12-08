"""
RoPE 旋转可视化 - 展示旋转位置编码原理
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from interpretability_lab.interpretability_utils import (
    compute_rope_frequencies,
    apply_rope_rotation,
    compute_rope_decay
)


def render_frequency_heatmap(freqs: np.ndarray, positions: np.ndarray, dim: int) -> go.Figure:
    """渲染频率热力图"""
    fig = go.Figure(data=go.Heatmap(
        z=np.sin(freqs),  # 显示 sin(θ) 的变化
        x=[f'd{i}' for i in range(freqs.shape[1])],
        y=positions[:100],  # 只显示前 100 个位置
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="RoPE 频率变化 (sin θ)",
        xaxis_title="维度对",
        yaxis_title="位置",
        height=400
    )
    
    return fig


def render_rotation_animation(dim: int = 8, num_positions: int = 20, base: float = 10000.0) -> go.Figure:
    """渲染高维 RoPE 旋转动画 - 展示不同维度对的不同旋转频率"""
    # 计算不同维度对的频率 (RoPE 核心公式)
    num_pairs = min(dim // 2, 4)  # 最多展示 4 个维度对
    inv_freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32)[:num_pairs] / dim))
    
    # 创建子图：每个维度对一个 2D 图
    fig = make_subplots(
        rows=1, cols=num_pairs,
        subplot_titles=[f'维度对 {i*2}-{i*2+1}\n(freq={inv_freqs[i]:.4f})' for i in range(num_pairs)],
        horizontal_spacing=0.08
    )
    
    # 原始向量 (每个维度对都用相同的初始向量)
    original_vec = np.array([1.0, 0.5])
    r = np.sqrt(original_vec[0]**2 + original_vec[1]**2)
    
    colors = px.colors.sample_colorscale('Viridis', [i / max(num_positions-1, 1) for i in range(num_positions)])
    
    for pair_idx in range(num_pairs):
        freq = inv_freqs[pair_idx]
        
        # 添加轨迹圆
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta_circle),
            y=r * np.sin(theta_circle),
            mode='lines',
            line=dict(color='rgba(128,128,128,0.3)', dash='dot', width=1),
            name='轨迹圆' if pair_idx == 0 else None,
            showlegend=(pair_idx == 0),
            hoverinfo='skip'
        ), row=1, col=pair_idx+1)
        
        # 绘制不同位置的旋转向量
        for pos in range(num_positions):
            theta = pos * freq  # 使用该维度对的实际频率
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotated = np.array([
                original_vec[0] * cos_t - original_vec[1] * sin_t,
                original_vec[0] * sin_t + original_vec[1] * cos_t
            ])
            
            fig.add_trace(go.Scatter(
                x=[0, rotated[0]],
                y=[0, rotated[1]],
                mode='lines+markers',
                name=f'Pos {pos}' if pair_idx == 0 else None,
                showlegend=(pair_idx == 0),
                legendgroup=f'pos_{pos}',
                line=dict(color=colors[pos], width=2),
                marker=dict(size=[3, 8]),
                hovertemplate=f'Pos {pos}<br>θ = {np.degrees(theta):.1f}°<br>freq = {freq:.4f}<extra></extra>'
            ), row=1, col=pair_idx+1)
        
        # 添加原始向量
        fig.add_trace(go.Scatter(
            x=[0, original_vec[0]],
            y=[0, original_vec[1]],
            mode='lines+markers',
            name='原始向量' if pair_idx == 0 else None,
            showlegend=(pair_idx == 0),
            legendgroup='original',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=[3, 12], symbol='diamond')
        ), row=1, col=pair_idx+1)
        
        # 设置坐标轴
        fig.update_xaxes(range=[-1.5, 1.5], scaleanchor=f"y{pair_idx+1 if pair_idx > 0 else ''}", row=1, col=pair_idx+1)
        fig.update_yaxes(range=[-1.5, 1.5], row=1, col=pair_idx+1)
    
    fig.update_layout(
        title=f"RoPE 旋转演示 (base={base:.0f}) - 低维度对旋转快，高维度对旋转慢",
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig


def render_decay_curve(decay: np.ndarray) -> go.Figure:
    """渲染相对位置衰减曲线"""
    fig = go.Figure()
    
    distances = list(range(len(decay)))
    
    fig.add_trace(go.Scatter(
        x=distances,
        y=decay,
        mode='lines',
        name='内积值',
        line=dict(color='#2563EB', width=2)
    ))
    
    # 添加平滑趋势线
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(decay, sigma=5)
    fig.add_trace(go.Scatter(
        x=distances,
        y=smoothed,
        mode='lines',
        name='趋势 (平滑)',
        line=dict(color='#DC2626', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="RoPE 相对位置衰减特性",
        xaxis_title="相对距离",
        yaxis_title="Q·K 内积",
        height=400
    )
    
    return fig


def render_multi_freq_visualization(dim: int, base: float) -> go.Figure:
    """渲染多频率可视化"""
    max_pos = 1000
    
    # 计算每个维度对的频率
    num_pairs = dim // 2
    inv_freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    
    # 选择固定索引的维度对（而不是相对位置）
    # 这样当 dim 变化时，同一个维度对的频率会变化
    # 例如 d4-d5: dim=32 时 freq=1/base^(4/32), dim=64 时 freq=1/base^(4/64)
    freq_indices = [2, 4, 8]  # 固定选择 d4-d5, d8-d9, d16-d17
    # 确保不越界
    freq_indices = [min(idx, num_pairs - 1) for idx in freq_indices]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'd{freq_indices[0]*2}-d{freq_indices[0]*2+1} (freq={inv_freqs[freq_indices[0]]:.4f})',
            f'd{freq_indices[1]*2}-d{freq_indices[1]*2+1} (freq={inv_freqs[freq_indices[1]]:.4f})',
            f'd{freq_indices[2]*2}-d{freq_indices[2]*2+1} (freq={inv_freqs[freq_indices[2]]:.4f})',
            '频率分布 (对数刻度)'
        ]
    )
    
    colors = ['#2563EB', '#059669', '#DC2626']
    labels = [f'd{freq_indices[0]*2}-d{freq_indices[0]*2+1}', 
              f'd{freq_indices[1]*2}-d{freq_indices[1]*2+1}', 
              f'd{freq_indices[2]*2}-d{freq_indices[2]*2+1}']
    
    # 统一使用相同的位置范围以便对比
    fixed_positions = 100  # 固定显示 0-100 的位置范围
    
    for idx, (freq_idx, color, label) in enumerate(zip(freq_indices, colors, labels)):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        freq = inv_freqs[freq_idx]
        
        # 使用固定的位置范围，这样更容易对比不同参数下的变化
        x_vals = np.linspace(0, fixed_positions, 500)
        theta_vals = x_vals * freq
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=np.sin(theta_vals),
                mode='lines',
                name=f'{label} sin',
                line=dict(color=color, width=2)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=np.cos(theta_vals),
                mode='lines',
                name=f'{label} cos',
                line=dict(color=color, dash='dash', width=2)
            ),
            row=row, col=col
        )
        
        # 更新坐标轴
        fig.update_xaxes(title_text="位置", row=row, col=col)
        fig.update_yaxes(title_text="值", row=row, col=col)
    
    # 频率分布
    fig.add_trace(
        go.Bar(
            x=[f'd{i*2}-{i*2+1}' for i in range(len(inv_freqs))],
            y=inv_freqs,
            marker_color='#7C3AED',
            hovertemplate='维度对: %{x}<br>频率: %{y:.6f}<extra></extra>'
        ),
        row=2, col=2
    )
    fig.update_yaxes(type="log", title_text="频率 (log)", row=2, col=2)
    fig.update_xaxes(title_text="维度对", row=2, col=2)
    
    fig.update_layout(
        height=650,
        showlegend=False,
        title_text=f"RoPE 多频率分解 (dim={dim}, base={base:.0f}) - 相同位置范围下不同频率的表现"
    )
    
    return fig


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">RoPE 旋转可视化</h1>', unsafe_allow_html=True)
    
    
    tab1, tab2 = st.tabs(["旋转演示", "衰减特性"])
    
    with tab1:
        st.markdown("### 向量旋转可视化（n=8）")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            rotation_base = st.number_input("RoPE Base", value=10000.0, min_value=1000.0, max_value=1000000.0, 
                                  help="base 越大，低频分量的波长越长", key="rotation_base")
            num_positions = st.slider("显示位置数", 5, 30, 15)
        
        with col2:
            # 固定 8 维展示 4 个维度对
            fig_rotation = render_rotation_animation(dim=8, num_positions=num_positions, base=rotation_base)
            st.plotly_chart(fig_rotation, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 多频率分解")
        
        st.markdown("""
        不同维度对使用不同的旋转频率，形成"多频率"编码。调整参数观察频率分布的变化。
        """)
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            dim = st.slider("向量维度", 8, 128, 64, step=8, help="实际 RoPE 会对每一对维度应用旋转")
            freq_base = st.number_input("RoPE Base", value=10000.0, min_value=1000.0, max_value=1000000.0, 
                                  help="base 越大，低频分量的波长越长", key="freq_base")
        
        with col2:
            fig_multi = render_multi_freq_visualization(dim, freq_base)
            st.plotly_chart(fig_multi, use_container_width=True)
        
    
    with tab2:
        st.markdown("### 相对位置衰减")
        
        st.markdown("""
        RoPE 的重要特性：两个 token 的注意力分数（Q·K 内积）会随着**相对距离**增加而**自然衰减**。
        这是位置编码方法优劣的重要指标。
        """)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            decay_dim = st.slider("维度", 64, 512, 256, step=64, key="decay_dim")
            decay_base = st.number_input("Base", value=10000.0, key="decay_base")
            max_dist = st.slider("最大距离", 50, 500, 200)
        
        decay = compute_rope_decay(decay_dim, max_dist, decay_base)
        
        with col2:
            fig_decay = render_decay_curve(decay)
            st.plotly_chart(fig_decay, width='stretch')
        
        st.markdown("""
        **衰减特性解读**：
        - 内积值在距离为 0 时最大（自己和自己的相似度最高）
        - 随距离增加呈现**震荡衰减**趋势
        - 高频分量导致震荡，低频分量决定整体衰减包络
        - 这种自然衰减有助于模型学习局部依赖
        """)
        
        # 不同 base 的对比
        st.markdown("### 不同 Base 的衰减对比")
        
        bases = [10000, 100000, 1000000]
        
        fig_compare = go.Figure()
        colors = ['#2563EB', '#059669', '#DC2626']
        
        for base_val, color in zip(bases, colors):
            decay_vals = compute_rope_decay(256, 200, base_val)
            # 平滑处理
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(decay_vals, sigma=5)
            
            fig_compare.add_trace(go.Scatter(
                x=list(range(200)),
                y=smoothed,
                mode='lines',
                name=f'Base={base_val}',
                line=dict(color=color, width=2)
            ))
        
        fig_compare.update_layout(
            title="不同 RoPE Base 的衰减趋势",
            xaxis_title="相对距离",
            yaxis_title="Q·K 内积 (平滑)",
            height=400
        )
        
        st.plotly_chart(fig_compare, width='stretch')
        
        st.markdown("""
        **Base 参数的影响**：
        - **小 Base**: 衰减快，适合短序列
        - **大 Base**: 衰减慢，更好的长程依赖建模
        - Llama-3 使用 500000 的 Base，支持更长的上下文
        """)
