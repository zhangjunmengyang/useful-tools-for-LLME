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


def render_rotation_animation(dim: int = 8, num_positions: int = 20) -> go.Figure:
    """渲染 2D 旋转动画"""
    # 创建一个简单的 2D 向量
    np.random.seed(42)
    original_vec = np.array([1.0, 0.5])
    
    # 计算不同位置的旋转
    positions = list(range(num_positions))
    
    fig = go.Figure()
    
    colors = px.colors.sample_colorscale('Viridis', [i / num_positions for i in range(num_positions)])
    
    for pos in positions:
        # 简化的 RoPE 旋转 (2D)
        theta = pos * 0.5  # 简化的频率
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rotated = np.array([
            original_vec[0] * cos_t - original_vec[1] * sin_t,
            original_vec[0] * sin_t + original_vec[1] * cos_t
        ])
        
        # 添加箭头
        fig.add_trace(go.Scatter(
            x=[0, rotated[0]],
            y=[0, rotated[1]],
            mode='lines+markers',
            name=f'Pos {pos}',
            line=dict(color=colors[pos], width=2),
            marker=dict(size=[5, 10]),
            hovertemplate=f'Position {pos}<br>θ = {np.degrees(theta):.1f}°<extra></extra>'
        ))
    
    # 添加原始向量
    fig.add_trace(go.Scatter(
        x=[0, original_vec[0]],
        y=[0, original_vec[1]],
        mode='lines+markers',
        name='Original',
        line=dict(color='red', width=4, dash='dash'),
        marker=dict(size=[5, 15], symbol='diamond')
    ))
    
    # 添加单位圆
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    r = np.sqrt(original_vec[0]**2 + original_vec[1]**2)
    fig.add_trace(go.Scatter(
        x=r * np.cos(theta_circle),
        y=r * np.sin(theta_circle),
        mode='lines',
        line=dict(color='gray', dash='dot'),
        name='轨迹圆',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title="RoPE 2D 旋转演示",
        xaxis=dict(title="维度 0", range=[-1.5, 1.5], scaleanchor="y"),
        yaxis=dict(title="维度 1", range=[-1.5, 1.5]),
        height=500,
        width=500,
        showlegend=True
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
    freqs, positions = compute_rope_frequencies(dim, max_position=200, base=base)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'低频维度 (d0-d1)',
            f'中频维度 (d{dim//4}-d{dim//4+1})',
            f'高频维度 (d{dim//2-2}-d{dim//2-1})',
            '频率分布'
        ]
    )
    
    # 选择几个代表性的维度对
    freq_indices = [0, dim // 8, dim // 4 - 1]
    colors = ['#2563EB', '#059669', '#DC2626']
    labels = ['低频', '中频', '高频']
    
    for idx, (freq_idx, color, label) in enumerate(zip(freq_indices, colors, labels)):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        fig.add_trace(
            go.Scatter(
                x=positions[:100],
                y=np.sin(freqs[:100, freq_idx]),
                mode='lines',
                name=f'{label} sin',
                line=dict(color=color)
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=positions[:100],
                y=np.cos(freqs[:100, freq_idx]),
                mode='lines',
                name=f'{label} cos',
                line=dict(color=color, dash='dash')
            ),
            row=row, col=col
        )
    
    # 频率分布
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    fig.add_trace(
        go.Bar(
            x=list(range(len(inv_freq))),
            y=inv_freq,
            marker_color='#7C3AED'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False
    )
    
    return fig


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">RoPE 旋转可视化</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="tip-box">
    💡 <b>RoPE (Rotary Position Embedding)</b> 是现代 LLM 的主流位置编码方案。
    它通过在复数域旋转 Q/K 向量来编码位置信息，具有相对位置编码的优势和良好的外推能力。
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["🔄 旋转演示", "📉 衰减特性", "📐 数学原理"])
    
    with tab1:
        st.markdown("### 向量旋转可视化")
        
        st.markdown("""
        RoPE 的核心思想：将位置信息编码为**旋转角度**，相同内容在不同位置的向量，
        区别仅在于旋转了不同的角度。
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            dim = st.slider("向量维度", 8, 128, 64, step=8, help="实际 RoPE 会对每一对维度应用旋转")
            base = st.number_input("RoPE Base", value=10000.0, min_value=1000.0, max_value=1000000.0, 
                                  help="base 越大，低频分量的波长越长")
            num_positions = st.slider("显示位置数", 5, 30, 15)
        
        with col2:
            fig_rotation = render_rotation_animation(dim, num_positions)
            st.plotly_chart(fig_rotation, width='stretch')
        
        st.markdown("""
        **观察要点**：
        - 每个位置的向量都在同一个圆上（保持范数不变）
        - 位置 0 是原始向量（红色虚线）
        - 随着位置增加，向量逐渐旋转
        - 不同维度对的旋转速度不同（低维慢，高维快）
        """)
        
        st.markdown("---")
        st.markdown("### 多频率分解")
        
        fig_multi = render_multi_freq_visualization(dim, base)
        st.plotly_chart(fig_multi, width='stretch')
        
        st.markdown("""
        **频率分布解读**：
        - **低频维度**: 变化缓慢，编码"远程"位置关系
        - **高频维度**: 变化快速，编码"近距离"位置区分
        - 这种多尺度设计让模型同时捕捉局部和全局位置信息
        """)
    
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
    
    with tab3:
        st.markdown("### RoPE 数学原理")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("""
            #### 核心公式
            
            给定位置 $m$ 的 token，其 Query/Key 向量经过 RoPE 变换：
            
            $$
            f_q(x_m, m) = R_m \\cdot W_q \\cdot x_m
            $$
            
            其中旋转矩阵 $R_m$ 是**分块对角**的：
            
            $$
            R_m = \\begin{pmatrix}
            \\cos(m\\theta_0) & -\\sin(m\\theta_0) \\\\
            \\sin(m\\theta_0) & \\cos(m\\theta_0) \\\\
            & & \\cos(m\\theta_1) & -\\sin(m\\theta_1) \\\\
            & & \\sin(m\\theta_1) & \\cos(m\\theta_1) \\\\
            & & & & \\ddots
            \\end{pmatrix}
            $$
            
            频率定义：
            $$
            \\theta_i = \\text{base}^{-2i/d}
            $$
            """)
        
        with col_right:
            st.markdown("""
            #### 相对位置编码性质
            
            关键性质：Q 和 K 的内积只依赖于**相对位置** $m - n$：
            
            $$
            \\langle f_q(x_m, m), f_k(x_n, n) \\rangle = g(x_m, x_n, m-n)
            $$
            
            证明（2D 情况）：
            
            $$
            R_m^T R_n = R_{n-m}
            $$
            
            旋转矩阵是正交的，所以：
            $$
            q_m^T k_n = (R_m q)^T (R_n k) = q^T R_m^T R_n k = q^T R_{n-m} k
            $$
            """)
        
        st.markdown("---")
        
        st.markdown("""
        #### 复数域视角
        
        RoPE 可以用复数更优雅地表示：
        
        将向量的每一对维度 $(x_{2i}, x_{2i+1})$ 看作复数 $x_{2i} + i \\cdot x_{2i+1}$，
        则 RoPE 变换就是乘以单位复数 $e^{i m \\theta_i}$：
        
        ```python
        # 实际实现 (伪代码)
        for i in range(dim // 2):
            theta = position * base ** (-2 * i / dim)
            complex_rotation = cos(theta) + i * sin(theta)
            x[2i:2i+2] = x[2i:2i+2] * complex_rotation
        ```
        
        #### 与其他位置编码的对比
        
        | 方法 | 类型 | 外推能力 | 相对位置 | 计算效率 |
        |------|------|----------|----------|----------|
        | 绝对位置 (Learned) | 绝对 | ❌ 差 | ❌ 无 | ✅ 高 |
        | Sinusoidal | 绝对 | ⚠️ 一般 | ❌ 无 | ✅ 高 |
        | ALiBi | 相对 | ✅ 好 | ✅ 有 | ✅ 高 |
        | **RoPE** | 相对 | ✅ 好 | ✅ 有 | ✅ 高 |
        """)

