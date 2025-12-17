"""
语义相似度 - Token 级热力图与各向异性分析
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import re

from embedding_lab.embedding_utils import (
    get_batch_embeddings,
    cosine_similarity,
    compute_anisotropy,
    whitening_transform
)


def tokenize_text(text):
    """简单的中英文分词"""
    tokens = text.split()
    result = []
    for token in tokens:
        if re.search(r'[\u4e00-\u9fff]', token):
            for char in token:
                if char.strip():
                    result.append(char)
        else:
            if token.strip():
                result.append(token)
    return result if result else [text]


def create_similarity_heatmap(tokens_a, tokens_b, similarity_matrix, text_a, text_b):
    """创建 Token-to-Token 相似度热力图"""
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=tokens_b,
        y=tokens_a,
        colorscale=[
            [0, '#F3F4F6'], [0.3, '#DBEAFE'], [0.5, '#93C5FD'],
            [0.7, '#3B82F6'], [1, '#1D4ED8']
        ],
        hovertemplate='<b>%{y}</b> - <b>%{x}</b><br>Similarity: %{z:.4f}<extra></extra>',
        showscale=True,
        colorbar=dict(title=dict(text="Similarity"))
    ))
    
    fig.update_layout(
        title="Token-to-Token Similarity Matrix",
        xaxis=dict(
            title=f"Text B: \"{text_b[:30]}...\"" if len(text_b) > 30 else f"Text B: \"{text_b}\"",
            tickangle=-45
        ),
        yaxis=dict(
            title=f"Text A: \"{text_a[:30]}...\"" if len(text_a) > 30 else f"Text A: \"{text_a}\"",
            autorange='reversed'
        ),
        height=max(400, len(tokens_a) * 25 + 150),
        autosize=True,
        margin=dict(l=100, r=40, t=60, b=100),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    return fig


def create_anisotropy_histogram(similarities, mean_sim, title):
    """创建相似度分布直方图"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=similarities, nbinsx=30,
        marker=dict(color='#3B82F6', line=dict(color='#1D4ED8', width=1)),
        opacity=0.7
    ))
    fig.add_vline(x=mean_sim, line_dash="dash", line_color="#DC2626",
                  annotation_text=f"Mean: {mean_sim:.3f}")
    fig.add_vline(x=0, line_dash="dot", line_color="#059669",
                  annotation_text="Ideal: 0")
    fig.update_layout(
        title=title, xaxis_title="Cosine Similarity", yaxis_title="Word Pair Count",
        height=380, autosize=True, xaxis=dict(range=[-0.3, 1.1]),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    return fig


def compute_heatmap(text_a, text_b, model_choice):
    """计算热力图"""
    if not text_a or not text_b:
        return None, "", "", "", ""
    
    model_map = {
        "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "MiniLM-L6 (English)": "all-MiniLM-L6-v2"
    }
    model_id = model_map.get(model_choice, "paraphrase-multilingual-MiniLM-L12-v2")
    
    tokens_a = tokenize_text(text_a)
    tokens_b = tokenize_text(text_b)
    
    emb_a = get_batch_embeddings(tokens_a, model_id)
    emb_b = get_batch_embeddings(tokens_b, model_id)
    
    if emb_a is None or emb_b is None:
        return None, "Computation failed", "", "", ""
    
    # 计算相似度矩阵
    sim_matrix = np.zeros((len(tokens_a), len(tokens_b)))
    for i, ea in enumerate(emb_a):
        for j, eb in enumerate(emb_b):
            sim_matrix[i, j] = cosine_similarity(ea, eb)
    
    fig = create_similarity_heatmap(tokens_a, tokens_b, sim_matrix, text_a, text_b)
    
    # 句子级相似度
    full_emb_a = get_batch_embeddings([text_a], model_id)
    full_emb_b = get_batch_embeddings([text_b], model_id)
    
    if full_emb_a is not None and full_emb_b is not None:
        sentence_sim = cosine_similarity(full_emb_a[0], full_emb_b[0])
    else:
        sentence_sim = 0
    
    # 最相似词对
    max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
    max_pair = (tokens_a[max_idx[0]], tokens_b[max_idx[1]])
    max_sim = sim_matrix[max_idx]
    
    insight = f"**Strongest Alignment**: \"{max_pair[0]}\" - \"{max_pair[1]}\" Similarity: {max_sim:.4f}"
    
    return fig, f"{sentence_sim:.4f}", str(len(tokens_a)), str(len(tokens_b)), insight


def compute_anisotropy_analysis(words_text, model_choice):
    """各向异性分析"""
    words = [w.strip() for w in words_text.strip().split('\n') if w.strip()]
    
    if len(words) < 5:
        return None, None, "At least 5 words required", "", ""
    
    model_map = {
        "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "MiniLM-L6 (English)": "all-MiniLM-L6-v2"
    }
    model_id = model_map.get(model_choice, "paraphrase-multilingual-MiniLM-L12-v2")
    
    embeddings = get_batch_embeddings(words, model_id)
    if embeddings is None:
        return None, None, "Embedding computation failed", "", ""
    
    mean_sim, std_sim = compute_anisotropy(embeddings, sample_size=min(100, len(embeddings)))
    
    # 计算所有词对相似度
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            similarities.append(sim)
    
    fig1 = create_anisotropy_histogram(similarities, mean_sim, "Original Vectors - Word Pair Similarity Distribution")
    
    # 白化处理
    whitened = whitening_transform(embeddings)
    mean_sim_w, std_sim_w = compute_anisotropy(whitened, sample_size=min(100, len(whitened)))
    
    similarities_w = []
    for i in range(len(whitened)):
        for j in range(i + 1, len(whitened)):
            sim = cosine_similarity(whitened[i], whitened[j])
            similarities_w.append(sim)
    
    fig2 = create_anisotropy_histogram(similarities_w, mean_sim_w, "After Whitening - Word Pair Similarity Distribution")
    
    # Analysis results
    original_info = f"**Original Vectors**: Mean Similarity {mean_sim:.4f}, Std {std_sim:.4f}"
    whitened_info = f"**After Whitening**: Mean Similarity {mean_sim_w:.4f}, Std {std_sim_w:.4f}"

    if mean_sim > 0.3:
        diagnosis = "Significant anisotropy problem detected"
    elif mean_sim > 0.15:
        diagnosis = "Mild anisotropy detected"
    else:
        diagnosis = "Low anisotropy level"
    
    return fig1, fig2, diagnosis, original_info, whitened_info


def render():
    """渲染页面"""

    with gr.Tabs():
        # Tab 1: Heatmap
        with gr.Tab("Similarity Heatmap") as heatmap_tab:
            with gr.Row():
                text_a = gr.Textbox(label="Text A", lines=3, placeholder="Enter first text...")
                text_b = gr.Textbox(label="Text B", lines=3, placeholder="Enter second text...")
            
            with gr.Row():
                p1 = gr.Button("Word Order", size="sm")
                p2 = gr.Button("Synonym", size="sm")
                p3 = gr.Button("Cross-lingual", size="sm")
                p4 = gr.Button("Unrelated", size="sm")

            model_hm = gr.Dropdown(
                choices=["Multilingual MiniLM", "MiniLM-L6 (English)"],
                label="Embedding Model", value="Multilingual MiniLM"
            )

            heatmap = gr.Plot()

            with gr.Row():
                sent_sim = gr.Textbox(label="Sentence Similarity", interactive=False)
                tokens_a_count = gr.Textbox(label="Text A Token Count", interactive=False)
                tokens_b_count = gr.Textbox(label="Text B Token Count", interactive=False)
            
            insight_md = gr.Markdown("")
        
        # Tab 2: Anisotropy
        with gr.Tab("Anisotropy Analysis") as anisotropy_tab:
            words_input = gr.Textbox(
                label="Input Words (one per line)",
                placeholder="apple\ncar\nmusic\ncomputer\ncoffee",
                lines=8
            )
            
            model_an = gr.Dropdown(
                choices=["Multilingual MiniLM", "MiniLM-L6 (English)"],
                label="Embedding Model", value="Multilingual MiniLM"
            )

            diagnosis_md = gr.Markdown("")

            with gr.Row():
                fig_original = gr.Plot(label="Original Vectors")
                fig_whitened = gr.Plot(label="After Whitening")
            
            with gr.Row():
                original_info = gr.Markdown("")
                whitened_info = gr.Markdown("")
    
    # 预设按钮 - 设置并自动计算
    def set_and_compute_hm(a, b, model):
        result = compute_heatmap(a, b, model)
        return (a, b) + result
    
    p1.click(
        fn=lambda m: set_and_compute_hm("我看过这部电影", "这片子我看过", m),
        inputs=[model_hm],
        outputs=[text_a, text_b, heatmap, sent_sim, tokens_a_count, tokens_b_count, insight_md]
    )
    p2.click(
        fn=lambda m: set_and_compute_hm("我非常喜欢这本书", "我特别爱这本书籍", m),
        inputs=[model_hm],
        outputs=[text_a, text_b, heatmap, sent_sim, tokens_a_count, tokens_b_count, insight_md]
    )
    p3.click(
        fn=lambda m: set_and_compute_hm("我爱你", "I love you", m),
        inputs=[model_hm],
        outputs=[text_a, text_b, heatmap, sent_sim, tokens_a_count, tokens_b_count, insight_md]
    )
    p4.click(
        fn=lambda m: set_and_compute_hm("今天天气很好", "量子力学很难", m),
        inputs=[model_hm],
        outputs=[text_a, text_b, heatmap, sent_sim, tokens_a_count, tokens_b_count, insight_md]
    )
    
    # 自动计算热力图
    for component in [text_a, text_b, model_hm]:
        component.change(
            fn=compute_heatmap,
            inputs=[text_a, text_b, model_hm],
            outputs=[heatmap, sent_sim, tokens_a_count, tokens_b_count, insight_md]
        )
    
    # 自动计算各向异性分析
    for component in [words_input, model_an]:
        component.change(
            fn=compute_anisotropy_analysis,
            inputs=[words_input, model_an],
            outputs=[fig_original, fig_whitened, diagnosis_md, original_info, whitened_info]
        )

    # Re-render plots when tab becomes visible to fix width issues
    heatmap_tab.select(
        fn=compute_heatmap,
        inputs=[text_a, text_b, model_hm],
        outputs=[heatmap, sent_sim, tokens_a_count, tokens_b_count, insight_md]
    )

    anisotropy_tab.select(
        fn=compute_anisotropy_analysis,
        inputs=[words_input, model_an],
        outputs=[fig_original, fig_whitened, diagnosis_md, original_info, whitened_info]
    )

    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        hm_result = compute_heatmap("我看过这部电影", "这片子我看过", "Multilingual MiniLM")
        an_result = compute_anisotropy_analysis(
            "苹果\n汽车\n音乐\n电脑\n咖啡\n书籍\n天空\n海洋\n山峰\n河流",
            "Multilingual MiniLM"
        )
        return hm_result + an_result
    
    # 返回 load 事件信息
    return {
        'load_fn': on_load,
        'load_outputs': [
            heatmap, sent_sim, tokens_a_count, tokens_b_count, insight_md,
            fig_original, fig_whitened, diagnosis_md, original_info, whitened_info
        ]
    }
