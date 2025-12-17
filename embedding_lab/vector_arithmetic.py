"""
Vector Arithmetic - Word2Vec Classic Analogy Reasoning
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from embedding_lab.embedding_utils import (
    load_word2vec_model,
    get_word_vector,
    vector_arithmetic,
    cosine_similarity,
    reduce_dimensions
)


def create_vector_visualization(words, vectors, result_word=None, result_vector=None):
    """创建向量运算的 2D 可视化"""
    all_words = words.copy()
    all_vectors = [v for v in vectors]

    if result_word and result_vector is not None:
        all_words.append(result_word)
        all_vectors.append(result_vector)

    if len(all_vectors) < 2:
        return None

    vectors_array = np.array(all_vectors)
    coords_2d = reduce_dimensions(vectors_array, method="pca", n_components=2)

    fig = go.Figure()

    colors = ['#2563EB', '#059669', '#D97706', '#7C3AED']

    # 输入词的点
    for i, (word, coord) in enumerate(zip(all_words[:-1] if result_word else all_words,
                                           coords_2d[:-1] if result_word else coords_2d)):
        fig.add_trace(go.Scatter(
            x=[coord[0]],
            y=[coord[1]],
            mode='markers+text',
            marker=dict(size=15, color=colors[i % len(colors)]),
            text=[word],
            textposition='top center',
            textfont=dict(size=14, color='#111827'),
            name=word,
            hoverinfo='text',
            hovertext=f'{word}'
        ))

    # 结果词的点
    if result_word:
        fig.add_trace(go.Scatter(
            x=[coords_2d[-1][0]],
            y=[coords_2d[-1][1]],
            mode='markers+text',
            marker=dict(size=18, color='#DC2626', symbol='star'),
            text=[result_word],
            textposition='top center',
            textfont=dict(size=14, color='#DC2626'),
            name=f'Result: {result_word}',
            hoverinfo='text',
            hovertext=f'Result: {result_word}'
        ))

    fig.update_layout(
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        showlegend=False,
        height=450,
        autosize=True,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor='#E5E7EB',
            zeroline=True,
            zerolinecolor='#D1D5DB',
            showticklabels=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E5E7EB',
            zeroline=True,
            zerolinecolor='#D1D5DB',
            showticklabels=False
        )
    )

    return fig


def compute_analogy(word_a, word_b, word_c, top_k):
    """计算向量类比"""
    model = get_model()
    if model is None:
        return (
            "Failed to load model, please check network connection",
            pd.DataFrame(),
            None,
            ""
        )
    word_a = word_a.lower().strip()
    word_b = word_b.lower().strip()
    word_c = word_c.lower().strip()

    if not all([word_a, word_b, word_c]):
        return (
            "Please enter all words",
            pd.DataFrame(),
            None,
            ""
        )

    # 检查词是否在词表中
    missing = []
    for w in [word_a, word_b, word_c]:
        if get_word_vector(model, w) is None:
            missing.append(w)

    if missing:
        return (
            f"Words not in vocabulary: {', '.join(missing)}\n\nPlease use common English words (lowercase)",
            pd.DataFrame(),
            None,
            ""
        )

    # 执行向量运算
    results = vector_arithmetic(model, positive=[word_a, word_c], negative=[word_b], topn=top_k)

    # 公式
    formula = f"""
### Computation Formula

<div style="text-align: center; padding: 16px; background: #F3F4F6; border-radius: 8px; margin-bottom: 16px;">
    <span style="font-size: 20px; font-family: 'JetBrains Mono', monospace; color: #111827;">
        <span style="color: #2563EB;">{word_a}</span> -
        <span style="color: #DC2626;">{word_b}</span> +
        <span style="color: #059669;">{word_c}</span> =
        <span style="color: #7C3AED; font-weight: bold;">{results[0][0] if results else '?'}</span>
    </span>
</div>
"""

    # 结果表格
    exclude = set([word_a.lower(), word_b.lower(), word_c.lower()])
    result_data = []
    rank = 1
    for word, score in results:
        if word.lower() not in exclude:
            result_data.append({
                "Rank": rank,
                "Word": word,
                "Similarity": f"{score:.4f}"
            })
            rank += 1

    df = pd.DataFrame(result_data)

    # 可视化
    vec_a = get_word_vector(model, word_a)
    vec_b = get_word_vector(model, word_b)
    vec_c = get_word_vector(model, word_c)

    result_word = results[0][0] if results else None
    result_vec = get_word_vector(model, result_word) if result_word else None

    fig = create_vector_visualization(
        [word_a, word_b, word_c],
        [vec_a, vec_b, vec_c],
        result_word=result_word,
        result_vector=result_vec
    )

    # 详细信息
    target_vec = vec_a - vec_b + vec_c
    details = f"""
### Vector Details

| Attribute | Value |
|-----------|-------|
| Vector Dimension | {vec_a.shape[0]} |
| {word_a} ↔ {word_b} | {cosine_similarity(vec_a, vec_b):.4f} |
| {word_a} ↔ {word_c} | {cosine_similarity(vec_a, vec_c):.4f} |
| {word_b} ↔ {word_c} | {cosine_similarity(vec_b, vec_c):.4f} |
"""
    if result_word and result_vec is not None:
        details += f"| Result ↔ {result_word} | {cosine_similarity(target_vec, result_vec):.4f} |"

    return formula, df, fig, details


# 全局模型缓存
_word2vec_model = {"model": None, "loaded": False}


def get_model():
    """获取模型（懒加载）"""
    if not _word2vec_model["loaded"]:
        _word2vec_model["model"] = load_word2vec_model()
        _word2vec_model["loaded"] = True
    return _word2vec_model["model"]


def render():
    """渲染页面"""

    gr.Markdown("""
    Explore vector analogy reasoning: **A - B + C = ?**
    The model finds words whose vectors are closest to the computed result.
    """)

    with gr.Row():
        word_a = gr.Textbox(label="Word A", value="king", scale=2)
        with gr.Column(scale=1, min_width=50):
            gr.Markdown("<div style='text-align: center; padding-top: 32px; font-size: 24px; color: #DC2626;'>-</div>")
        word_b = gr.Textbox(label="Word B", value="man", scale=2)
        with gr.Column(scale=1, min_width=50):
            gr.Markdown("<div style='text-align: center; padding-top: 32px; font-size: 24px; color: #059669;'>+</div>")
        word_c = gr.Textbox(label="Word C", value="woman", scale=2)

    top_k = gr.Slider(label="Top-K Results", minimum=1, maximum=20, value=10, step=1)

    # 预设案例
    gr.Markdown("#### Classic Examples")
    with gr.Row():
        preset1 = gr.Button("King - Man + Woman", size="sm")
        preset2 = gr.Button("Paris - France + Germany", size="sm")
        preset3 = gr.Button("Car - Road + Water", size="sm")
        preset4 = gr.Button("Brother - Man + Woman", size="sm")
        preset5 = gr.Button("Walking - Walk + Swim", size="sm")

    gr.Markdown("---")

    # 结果展示
    formula_md = gr.Markdown("")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("#### Most Similar Words")
            result_df = gr.Dataframe(interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("#### Vector Space Projection")
            plot = gr.Plot()

    with gr.Accordion("Vector Details", open=False):
        details_md = gr.Markdown("")

    # 事件绑定
    def set_and_compute(a, b, c, k):
        result = compute_analogy(a, b, c, k)
        return (a, b, c) + result

    preset1.click(
        fn=lambda k: set_and_compute("king", "man", "woman", k),
        inputs=[top_k],
        outputs=[word_a, word_b, word_c, formula_md, result_df, plot, details_md]
    )
    preset2.click(
        fn=lambda k: set_and_compute("paris", "france", "germany", k),
        inputs=[top_k],
        outputs=[word_a, word_b, word_c, formula_md, result_df, plot, details_md]
    )
    preset3.click(
        fn=lambda k: set_and_compute("car", "road", "water", k),
        inputs=[top_k],
        outputs=[word_a, word_b, word_c, formula_md, result_df, plot, details_md]
    )
    preset4.click(
        fn=lambda k: set_and_compute("brother", "man", "woman", k),
        inputs=[top_k],
        outputs=[word_a, word_b, word_c, formula_md, result_df, plot, details_md]
    )
    preset5.click(
        fn=lambda k: set_and_compute("walking", "walk", "swim", k),
        inputs=[top_k],
        outputs=[word_a, word_b, word_c, formula_md, result_df, plot, details_md]
    )

    # 自动计算
    for inp in [word_a, word_b, word_c]:
        inp.change(
            fn=compute_analogy,
            inputs=[word_a, word_b, word_c, top_k],
            outputs=[formula_md, result_df, plot, details_md]
        )

    top_k.change(
        fn=compute_analogy,
        inputs=[word_a, word_b, word_c, top_k],
        outputs=[formula_md, result_df, plot, details_md]
    )

    def on_load():
        """页面加载时计算默认值"""
        return compute_analogy("king", "man", "woman", 10)

    return {
        'load_fn': on_load,
        'load_outputs': [formula_md, result_df, plot, details_md]
    }
