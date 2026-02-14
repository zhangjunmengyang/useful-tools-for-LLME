"""
检索模拟器页面

模拟 RAG 系统的检索过程，计算 query 与文档的语义相似度
"""

import gradio as gr
import plotly.graph_objects as go
from .rag_utils import (
    compute_similarity,
    DEFAULT_RETRIEVAL_DOCS,
    DEFAULT_RETRIEVAL_QUERY
)


def perform_retrieval(query: str, documents_text: str, top_k: int, model_name: str) -> tuple:
    """
    执行检索模拟

    Args:
        query: 查询文本
        documents_text: 文档文本（每行一个文档）
        top_k: 返回 Top-K 结果
        model_name: embedding 模型名称

    Returns:
        (检索结果 HTML, 相似度图表)
    """
    if not query.strip():
        return "<p style='color: #EF4444;'>Please enter a query.</p>", None

    if not documents_text.strip():
        return "<p style='color: #EF4444;'>Please enter documents.</p>", None

    # 解析文档（每行一个文档，或按双换行符分割）
    documents = []
    for line in documents_text.split('\n'):
        line = line.strip()
        if line:
            documents.append(line)

    if not documents:
        return "<p style='color: #EF4444;'>No valid documents found.</p>", None

    try:
        # 计算相似度
        similarities = compute_similarity(query, documents, model_name)

        # 限制返回 Top-K
        top_k = min(top_k, len(similarities))
        top_results = similarities[:top_k]

        # 生成结果 HTML
        results_html = render_results_html(query, documents, top_results)

        # 生成相似度图表
        chart = render_similarity_chart(documents, top_results, top_k)

        return results_html, chart

    except Exception as e:
        return f"<p style='color: #EF4444;'>Error: {str(e)}</p>", None


def render_results_html(query: str, documents: list, results: list) -> str:
    """
    渲染检索结果 HTML

    Args:
        query: 查询文本
        documents: 文档列表
        results: (doc_idx, similarity) 列表

    Returns:
        HTML 字符串
    """
    html_parts = ["<div style='font-family: sans-serif;'>"]

    # 显示查询
    html_parts.append(f"""
    <div style='background: #FFF7ED; border-left: 4px solid #FF9D00; padding: 12px 16px; border-radius: 0 8px 8px 0; margin-bottom: 20px;'>
        <div style='font-weight: 600; color: #374151; margin-bottom: 4px;'>Query:</div>
        <div style='color: #111827;'>{query}</div>
    </div>
    """)

    # 显示 Top-K 结果
    html_parts.append("<div style='margin-top: 20px;'>")

    for rank, (doc_idx, score) in enumerate(results, 1):
        doc_text = documents[doc_idx]

        # 相似度颜色编码
        if score >= 0.7:
            score_color = "#10B981"  # 绿色 - 高相似度
        elif score >= 0.4:
            score_color = "#F59E0B"  # 橙色 - 中等相似度
        else:
            score_color = "#EF4444"  # 红色 - 低相似度

        html_parts.append(f"""
        <div style='background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
                <span style='font-weight: 600; color: #374151;'>Rank #{rank} - Document {doc_idx + 1}</span>
                <span style='background-color: {score_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 13px; font-weight: 600;'>
                    {score:.4f}
                </span>
            </div>
            <div style='color: #4B5563; line-height: 1.6;'>{doc_text}</div>
        </div>
        """)

    html_parts.append("</div></div>")

    return "".join(html_parts)


def render_similarity_chart(documents: list, results: list, top_k: int) -> go.Figure:
    """
    渲染相似度条形图

    Args:
        documents: 文档列表
        results: (doc_idx, similarity) 列表
        top_k: Top-K 数量

    Returns:
        Plotly Figure
    """
    # 提取数据
    doc_labels = [f"Doc {doc_idx + 1}" for doc_idx, _ in results]
    scores = [score for _, score in results]

    # 颜色编码
    colors = []
    for score in scores:
        if score >= 0.7:
            colors.append('#10B981')  # 绿色
        elif score >= 0.4:
            colors.append('#F59E0B')  # 橙色
        else:
            colors.append('#EF4444')  # 红色

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=doc_labels,
        y=scores,
        marker=dict(color=colors),
        text=[f"{score:.4f}" for score in scores],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Similarity: %{y:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=f"Top-{top_k} Similarity Scores",
        xaxis_title="Document",
        yaxis_title="Cosine Similarity",
        yaxis=dict(range=[0, 1.0]),
        autosize=True,
        height=450,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(family="Source Sans Pro, sans-serif", size=12),
        margin=dict(l=60, r=40, t=60, b=60),
        hovermode='x unified'
    )

    return fig


def render():
    """
    渲染检索模拟器页面

    Returns:
        None
    """
    gr.Markdown("## Retrieval Simulator")
    gr.Markdown("Simulate semantic search by computing cosine similarity between a query and documents using sentence embeddings.")

    with gr.Row():
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Query",
                placeholder="Enter your query...",
                lines=3,
                value=DEFAULT_RETRIEVAL_QUERY
            )

            documents_input = gr.Textbox(
                label="Documents (one per line or separated by empty lines)",
                placeholder="Enter documents...",
                lines=12,
                value=DEFAULT_RETRIEVAL_DOCS
            )

            with gr.Row():
                top_k = gr.Slider(
                    label="Top-K Results",
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1
                )

            with gr.Row():
                model_select = gr.Dropdown(
                    label="Embedding Model",
                    choices=[
                        "all-MiniLM-L6-v2",
                        "all-mpnet-base-v2",
                        "paraphrase-multilingual-MiniLM-L12-v2"
                    ],
                    value="all-MiniLM-L6-v2"
                )

            retrieve_btn = gr.Button("Search", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Search Results")
            results_html = gr.HTML(label="Results")

    with gr.Row():
        similarity_chart = gr.Plot(label="Similarity Scores")

    # 事件绑定
    retrieve_btn.click(
        fn=perform_retrieval,
        inputs=[query_input, documents_input, top_k, model_select],
        outputs=[results_html, similarity_chart]
    )

    # 初始化：自动执行默认查询
    def initial_load():
        return perform_retrieval(
            DEFAULT_RETRIEVAL_QUERY,
            DEFAULT_RETRIEVAL_DOCS,
            3,
            "all-MiniLM-L6-v2"
        )

    return {
        'load_fn': initial_load,
        'load_outputs': [results_html, similarity_chart]
    }
