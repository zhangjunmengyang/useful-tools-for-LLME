"""
RoPE 旋转可视化 - 展示旋转位置编码原理
"""

import gradio as gr
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
        z=np.sin(freqs),
        x=[f'd{i}' for i in range(freqs.shape[1])],
        y=positions[:100],
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title="Frequency Heatmap (sin θ)",
        xaxis_title="Dimension Pair",
        yaxis_title="Position",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_rotation_animation(dim: int = 8, num_positions: int = 20, base: float = 10000.0) -> go.Figure:
    """渲染高维 RoPE 旋转动画 - 展示不同维度对的不同旋转频率"""
    # 计算不同维度对的频率 (RoPE 核心公式)
    num_pairs = min(dim // 2, 4)
    inv_freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32)[:num_pairs] / dim))
    
    # 创建子图：每个维度对一个 2D 图
    fig = make_subplots(
        rows=1, cols=num_pairs,
        subplot_titles=[f'维度对 {i*2}-{i*2+1}\n(freq={inv_freqs[i]:.4f})' for i in range(num_pairs)],
        horizontal_spacing=0.08
    )
    
    # 原始向量
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
            theta = pos * freq
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
                hovertemplate=f'Pos {pos}<br>theta = {np.degrees(theta):.1f} deg<br>freq = {freq:.4f}<extra></extra>'
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
        title=f"RoPE Rotation (base={base:.0f})",
        height=480,
        autosize=True,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
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
    try:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(decay, sigma=5)
        fig.add_trace(go.Scatter(
            x=distances,
            y=smoothed,
            mode='lines',
            name='趋势 (平滑)',
            line=dict(color='#DC2626', width=2, dash='dash')
        ))
    except ImportError:
        pass
    
    fig.update_layout(
        title="Relative Position Decay",
        xaxis_title="Distance",
        yaxis_title="Q·K",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_multi_freq_visualization(dim: int, base: float) -> go.Figure:
    """渲染多频率可视化"""
    max_pos = 1000
    
    # 计算每个维度对的频率
    num_pairs = dim // 2
    inv_freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    
    # 选择固定索引的维度对
    freq_indices = [2, 4, 8]
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
    
    fixed_positions = 100
    
    for idx, (freq_idx, color, label) in enumerate(zip(freq_indices, colors, labels)):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        freq = inv_freqs[freq_idx]
        
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
        height=680,
        autosize=True,
        showlegend=False,
        title_text=f"RoPE Frequency Decomposition (dim={dim}, base={base:.0f})",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def update_rotation(rotation_base, num_positions):
    """更新旋转演示图"""
    return render_rotation_animation(dim=8, num_positions=num_positions, base=rotation_base)


def update_multi_freq(dim, freq_base):
    """更新多频率可视化"""
    return render_multi_freq_visualization(dim, freq_base)


def update_decay(decay_dim, decay_base, max_dist):
    """更新衰减曲线"""
    decay = compute_rope_decay(decay_dim, max_dist, decay_base)
    return render_decay_curve(decay)


def render_base_comparison():
    """渲染不同 base 的对比"""
    bases = [10000, 100000, 1000000]
    
    fig = go.Figure()
    colors = ['#2563EB', '#059669', '#DC2626']
    
    for base_val, color in zip(bases, colors):
        decay_vals = compute_rope_decay(256, 200, base_val)
        # 平滑处理
        try:
            from scipy.ndimage import gaussian_filter1d
            smoothed = gaussian_filter1d(decay_vals, sigma=5)
        except ImportError:
            smoothed = decay_vals
        
        fig.add_trace(go.Scatter(
            x=list(range(200)),
            y=smoothed,
            mode='lines',
            name=f'Base={base_val}',
            line=dict(color=color, width=2)
        ))
    
    fig.update_layout(
        title="Base Comparison",
        xaxis_title="Distance",
        yaxis_title="Q·K (smoothed)",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render():
    """渲染页面"""
    
    gr.Markdown("## RoPE Explorer")
    
    with gr.Tabs():
        # Tab 1: 旋转演示
        with gr.Tab("旋转"):
            with gr.Row():
                with gr.Column(scale=1):
                    rotation_base = gr.Number(
                        label="Base",
                        value=10000.0,
                        minimum=1000.0,
                        maximum=1000000.0
                    )
                    num_positions = gr.Slider(
                        label="位置数",
                        minimum=5,
                        maximum=30,
                        value=15,
                        step=1
                    )
                
                with gr.Column(scale=3):
                    rotation_plot = gr.Plot(value=render_rotation_animation(dim=8, num_positions=15, base=10000.0))
            
            with gr.Row():
                with gr.Column(scale=1):
                    dim_slider = gr.Slider(
                        label="维度",
                        minimum=8,
                        maximum=128,
                        value=64,
                        step=8
                    )
                    freq_base = gr.Number(
                        label="Base",
                        value=10000.0,
                        minimum=1000.0,
                        maximum=1000000.0
                    )
                
                with gr.Column(scale=4):
                    freq_plot = gr.Plot(value=render_multi_freq_visualization(64, 10000.0))
        
        # Tab 2: 衰减特性
        with gr.Tab("衰减"):
            with gr.Row():
                with gr.Column(scale=1):
                    decay_dim = gr.Slider(
                        label="维度",
                        minimum=64,
                        maximum=512,
                        value=256,
                        step=64
                    )
                    decay_base = gr.Number(
                        label="Base",
                        value=10000.0
                    )
                    max_dist = gr.Slider(
                        label="最大距离",
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=50
                    )
                
                with gr.Column(scale=3):
                    decay_plot = gr.Plot(value=render_decay_curve(compute_rope_decay(256, 200, 10000.0)))
            
            base_compare_plot = gr.Plot(value=render_base_comparison())
    
    # 事件绑定 - 自动触发更新
    rotation_base.change(
        fn=update_rotation,
        inputs=[rotation_base, num_positions],
        outputs=[rotation_plot]
    )
    
    num_positions.change(
        fn=update_rotation,
        inputs=[rotation_base, num_positions],
        outputs=[rotation_plot]
    )
    
    dim_slider.change(
        fn=update_multi_freq,
        inputs=[dim_slider, freq_base],
        outputs=[freq_plot]
    )
    
    freq_base.change(
        fn=update_multi_freq,
        inputs=[dim_slider, freq_base],
        outputs=[freq_plot]
    )
    
    decay_dim.change(
        fn=update_decay,
        inputs=[decay_dim, decay_base, max_dist],
        outputs=[decay_plot]
    )
    
    decay_base.change(
        fn=update_decay,
        inputs=[decay_dim, decay_base, max_dist],
        outputs=[decay_plot]
    )
    
    max_dist.change(
        fn=update_decay,
        inputs=[decay_dim, decay_base, max_dist],
        outputs=[decay_plot]
    )
    
    # RoPE 探索不需要额外的初始化，因为已经设置了默认 value