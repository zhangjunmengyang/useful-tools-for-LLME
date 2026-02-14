"""
文本分块可视化页面

支持三种分块策略：Fixed Size、Recursive、Sentence
"""

import gradio as gr
import plotly.graph_objects as go
from .rag_utils import (
    chunk_fixed_size,
    chunk_recursive,
    chunk_by_sentence,
    compute_chunk_stats,
    DEFAULT_CHUNKING_TEXT
)


# Token 颜色列表（与主应用保持一致）
CHUNK_COLORS = [
    "#D1FAE5", "#DBEAFE", "#E9D5FF", "#FED7AA", "#FBCFE8", "#FEF08A",
    "#CFFAFE", "#FECDD3", "#DDD6FE", "#A7F3D0", "#FFEDD5", "#E2E8F0"
]


def perform_chunking(text: str, strategy: str, chunk_size: int, overlap: int) -> tuple:
    """
    执行文本分块

    Args:
        text: 输入文本
        strategy: 分块策略
        chunk_size: 块大小
        overlap: 重叠大小

    Returns:
        (可视化HTML, 统计信息, 分布图表)
    """
    if not text.strip():
        return "<p style='color: #EF4444;'>Please enter text to chunk.</p>", "", None

    # 根据策略执行分块
    try:
        if strategy == "Fixed Size":
            chunks = chunk_fixed_size(text, chunk_size, overlap)
        elif strategy == "Recursive":
            chunks = chunk_recursive(text, chunk_size, overlap)
        elif strategy == "Sentence":
            chunks = chunk_by_sentence(text, chunk_size, overlap)
        else:
            return "<p style='color: #EF4444;'>Invalid strategy selected.</p>", "", None

        if not chunks:
            return "<p style='color: #EF4444;'>No chunks generated. Try adjusting parameters.</p>", "", None

        # 生成可视化 HTML
        html = render_chunks_html(chunks)

        # 计算统计信息
        stats = compute_chunk_stats(chunks)
        stats_html = render_stats_html(stats)

        # 生成分布图表
        chart = render_distribution_chart(chunks)

        return html, stats_html, chart

    except Exception as e:
        return f"<p style='color: #EF4444;'>Error: {str(e)}</p>", "", None


def render_chunks_html(chunks: list) -> str:
    """
    渲染分块可视化 HTML

    Args:
        chunks: 分块列表

    Returns:
        HTML 字符串
    """
    html_parts = ["<div style='line-height: 2.2; font-family: monospace; font-size: 14px;'>"]

    for idx, chunk in enumerate(chunks):
        color = CHUNK_COLORS[idx % len(CHUNK_COLORS)]
        # 转义 HTML 特殊字符
        chunk_text = chunk.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # 添加 chunk 编号标签
        label = f"<span style='background-color: #374151; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; font-weight: 600; margin-right: 4px;'>#{idx+1}</span>"

        # 添加 chunk 内容
        chunk_html = f"<span style='background-color: {color}; padding: 4px 8px; margin: 2px; border-radius: 4px; display: inline;'>{chunk_text}</span>"

        html_parts.append(f"{label}{chunk_html}<br><br>")

    html_parts.append("</div>")

    return "".join(html_parts)


def render_stats_html(stats: dict) -> str:
    """
    渲染统计信息 HTML

    Args:
        stats: 统计信息字典

    Returns:
        HTML 字符串
    """
    html = f"""
    <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 16px;'>
        <div class='stat-card'>
            <div class='stat-value'>{stats['num_chunks']}</div>
            <div class='stat-label'>Total Chunks</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{stats['avg_length']:.1f}</div>
            <div class='stat-label'>Avg Length</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{stats['min_length']}</div>
            <div class='stat-label'>Min Length</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{stats['max_length']}</div>
            <div class='stat-label'>Max Length</div>
        </div>
        <div class='stat-card'>
            <div class='stat-value'>{stats['total_chars']}</div>
            <div class='stat-label'>Total Chars</div>
        </div>
    </div>
    """
    return html


def render_distribution_chart(chunks: list) -> go.Figure:
    """
    渲染 chunk 长度分布图表

    Args:
        chunks: 分块列表

    Returns:
        Plotly Figure
    """
    lengths = [len(chunk) for chunk in chunks]
    chunk_indices = [f"Chunk {i+1}" for i in range(len(chunks))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=chunk_indices,
        y=lengths,
        marker=dict(
            color=lengths,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Length")
        ),
        text=lengths,
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Length: %{y}<extra></extra>'
    ))

    fig.update_layout(
        title="Chunk Length Distribution",
        xaxis_title="Chunk Index",
        yaxis_title="Character Count",
        autosize=True,
        height=400,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(family="Source Sans Pro, sans-serif", size=12),
        margin=dict(l=60, r=40, t=60, b=80),
        hovermode='x unified'
    )

    return fig


def render():
    """
    渲染文本分块页面

    Returns:
        None
    """
    gr.Markdown("## Text Chunking Playground")
    gr.Markdown("Visualize different text chunking strategies for RAG applications. Choose from Fixed Size, Recursive, or Sentence-based splitting.")

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to chunk...",
                lines=10,
                value=DEFAULT_CHUNKING_TEXT
            )

            with gr.Row():
                strategy = gr.Radio(
                    label="Chunking Strategy",
                    choices=["Fixed Size", "Recursive", "Sentence"],
                    value="Recursive"
                )

            with gr.Row():
                chunk_size = gr.Slider(
                    label="Chunk Size (characters)",
                    minimum=50,
                    maximum=1000,
                    value=200,
                    step=50
                )

            with gr.Row():
                overlap = gr.Slider(
                    label="Overlap Size (characters)",
                    minimum=0,
                    maximum=200,
                    value=50,
                    step=10
                )

            process_btn = gr.Button("Process Chunks", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Chunked Output")
            chunks_html = gr.HTML(label="Chunks")

    with gr.Row():
        stats_html = gr.HTML(label="Statistics")

    with gr.Row():
        distribution_chart = gr.Plot(label="Length Distribution")

    # 事件绑定
    process_btn.click(
        fn=perform_chunking,
        inputs=[input_text, strategy, chunk_size, overlap],
        outputs=[chunks_html, stats_html, distribution_chart]
    )

    # 初始化：自动处理默认文本
    def initial_load():
        return perform_chunking(DEFAULT_CHUNKING_TEXT, "Recursive", 200, 50)

    return {
        'load_fn': initial_load,
        'load_outputs': [chunks_html, stats_html, distribution_chart]
    }
