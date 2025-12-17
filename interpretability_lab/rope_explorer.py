"""
RoPE Rotation Visualization
Demonstrating Rotary Position Encoding principles
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


def render_rotation_animation(dim: int = 8, num_positions: int = 20, base: float = 10000.0) -> go.Figure:
    """渲染高维 RoPE 旋转动画"""
    num_pairs = min(dim // 2, 4)
    inv_freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32)[:num_pairs] / dim))

    fig = make_subplots(
        rows=1, cols=num_pairs,
        subplot_titles=[f'Pair {i*2}-{i*2+1} (freq={inv_freqs[i]:.4f})' for i in range(num_pairs)],
        horizontal_spacing=0.08
    )

    original_vec = np.array([1.0, 0.5])
    r = np.sqrt(original_vec[0]**2 + original_vec[1]**2)

    colors = px.colors.sample_colorscale('Viridis', [i / max(num_positions-1, 1) for i in range(num_positions)])

    for pair_idx in range(num_pairs):
        freq = inv_freqs[pair_idx]

        # 轨迹圆
        theta_circle = np.linspace(0, 2 * np.pi, 100)
        fig.add_trace(go.Scatter(
            x=r * np.cos(theta_circle),
            y=r * np.sin(theta_circle),
            mode='lines',
            line=dict(color='rgba(128,128,128,0.3)', dash='dot', width=1),
            showlegend=False,
            hoverinfo='skip'
        ), row=1, col=pair_idx+1)

        # 不同位置的旋转向量
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
                hovertemplate=f'Pos {pos}<br>theta = {np.degrees(theta):.1f}°<br>freq = {freq:.4f}<extra></extra>'
            ), row=1, col=pair_idx+1)

        # 原始向量
        fig.add_trace(go.Scatter(
            x=[0, original_vec[0]],
            y=[0, original_vec[1]],
            mode='lines+markers',
            name='Original' if pair_idx == 0 else None,
            showlegend=(pair_idx == 0),
            legendgroup='original',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=[3, 12], symbol='diamond')
        ), row=1, col=pair_idx+1)

        # 坐标轴设置
        fig.update_xaxes(range=[-1.5, 1.5], scaleanchor=f"y{pair_idx+1 if pair_idx > 0 else ''}", row=1, col=pair_idx+1)
        fig.update_yaxes(range=[-1.5, 1.5], row=1, col=pair_idx+1)

    fig.update_layout(
        height=480,
        autosize=True,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=40, r=40, t=80, b=40)
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
        name='Dot Product',
        line=dict(color='#2563EB', width=2)
    ))

    # 平滑趋势线
    try:
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(decay, sigma=5)
        fig.add_trace(go.Scatter(
            x=distances,
            y=smoothed,
            mode='lines',
            name='Trend (Smoothed)',
            line=dict(color='#DC2626', width=2, dash='dash')
        ))
    except ImportError:
        pass

    fig.update_layout(
        xaxis_title="Distance",
        yaxis_title="Q·K",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def render_multi_freq_visualization(dim: int, base: float) -> go.Figure:
    """渲染多频率可视化"""
    num_pairs = dim // 2
    inv_freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))

    freq_indices = [2, 4, 8]
    freq_indices = [min(idx, num_pairs - 1) for idx in freq_indices]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'd{freq_indices[0]*2}-d{freq_indices[0]*2+1} (freq={inv_freqs[freq_indices[0]]:.4f})',
            f'd{freq_indices[1]*2}-d{freq_indices[1]*2+1} (freq={inv_freqs[freq_indices[1]]:.4f})',
            f'd{freq_indices[2]*2}-d{freq_indices[2]*2+1} (freq={inv_freqs[freq_indices[2]]:.4f})',
            'Frequency Distribution (Log Scale)'
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
                line=dict(color=color, width=2),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=np.cos(theta_vals),
                mode='lines',
                name=f'{label} cos',
                line=dict(color=color, dash='dash', width=2),
                showlegend=False
            ),
            row=row, col=col
        )

        fig.update_xaxes(title_text="Position", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)

    # 频率分布
    fig.add_trace(
        go.Bar(
            x=[f'd{i*2}-{i*2+1}' for i in range(len(inv_freqs))],
            y=inv_freqs,
            marker_color='#7C3AED',
            hovertemplate='Pair: %{x}<br>Frequency: %{y:.6f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    fig.update_yaxes(type="log", title_text="Frequency (log)", row=2, col=2)
    fig.update_xaxes(title_text="Dimension Pair", row=2, col=2)

    fig.update_layout(
        height=680,
        autosize=True,
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=40, r=40, t=80, b=40)
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
        xaxis_title="Distance",
        yaxis_title="Q·K (Smoothed)",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def render():
    """渲染页面"""

    gr.Markdown("""
    RoPE (Rotary Position Embedding) encodes position by rotating vectors in different dimension pairs at different frequencies.
    **Key insight**: Lower dimension pairs rotate faster (higher frequency), while higher pairs rotate slower (lower frequency).
    """)

    with gr.Tabs():
        # Rotation Demo
        with gr.Tab("Rotation") as rotation_tab:
            gr.Markdown("""
            **Rotation Visualization**: Shows how different dimension pairs rotate at different frequencies.
            - **Red dashed vector**: Original vector (position 0)
            - **Colored vectors**: Rotated vectors at different positions
            - **Dotted circle**: Rotation trajectory
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    rotation_base = gr.Number(
                        label="Base",
                        value=10000.0,
                        minimum=1000.0,
                        maximum=1000000.0
                    )
                    num_positions = gr.Slider(
                        label="Number of Positions",
                        minimum=5,
                        maximum=30,
                        value=15,
                        step=1
                    )

                with gr.Column(scale=3):
                    rotation_plot = gr.Plot(value=render_rotation_animation(dim=8, num_positions=15, base=10000.0))

            gr.Markdown("""
            **Frequency Decomposition**: Shows sine and cosine patterns for different dimension pairs.
            - Lower pairs (left) have higher frequencies, completing more cycles over the same distance
            - Higher pairs (right) have lower frequencies, preserving long-range information
            - Bottom right: Frequency distribution across all dimension pairs (log scale)
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    dim_slider = gr.Slider(
                        label="Dimension",
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

        # Decay Characteristics
        with gr.Tab("Decay") as decay_tab:
            gr.Markdown("""
            **Relative Position Decay**: Shows how the dot product Q·K decreases as the distance between positions increases.
            - Solid blue line: Raw dot product values
            - Dashed red line: Smoothed trend showing overall decay pattern
            """)

            with gr.Row():
                with gr.Column(scale=1):
                    decay_dim = gr.Slider(
                        label="Dimension",
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
                        label="Max Distance",
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=50
                    )

                with gr.Column(scale=3):
                    decay_plot = gr.Plot(value=render_decay_curve(compute_rope_decay(256, 200, 10000.0)))

            gr.Markdown("""
            **Base Comparison**: Larger base values result in slower decay, enabling the model to handle longer sequences.
            """)

            base_compare_plot = gr.Plot(value=render_base_comparison())

    # 事件绑定
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

    # Re-render plots when tabs become visible to fix width issues
    rotation_tab.select(
        fn=lambda base, pos, dim, freq_base: (
            render_rotation_animation(dim=8, num_positions=pos, base=base),
            render_multi_freq_visualization(dim, freq_base)
        ),
        inputs=[rotation_base, num_positions, dim_slider, freq_base],
        outputs=[rotation_plot, freq_plot]
    )

    decay_tab.select(
        fn=lambda dim, base, dist: (
            render_decay_curve(compute_rope_decay(dim, dist, base)),
            render_base_comparison()
        ),
        inputs=[decay_dim, decay_base, max_dist],
        outputs=[decay_plot, base_compare_plot]
    )
