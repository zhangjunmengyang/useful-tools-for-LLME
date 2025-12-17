"""
Vector Visualization - High-dimensional Vector Dimensionality Reduction
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from embedding_lab.embedding_utils import (
    get_batch_embeddings,
    reduce_dimensions,
    get_label_color,
    cosine_similarity,
    PRESET_DATASETS,
    DimensionReductionError,
    MIN_SAMPLES
)


def create_3d_scatter(coords, labels, texts, title="Vector Space Visualization"):
    """创建 3D 散点图"""
    unique_labels = list(set(labels))

    fig = go.Figure()

    for label in unique_labels:
        mask = [l == label for l in labels]
        indices = [i for i, m in enumerate(mask) if m]

        color = get_label_color(label)

        fig.add_trace(go.Scatter3d(
            x=coords[indices, 0],
            y=coords[indices, 1],
            z=coords[indices, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=color,
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=[texts[i] for i in indices],
            hovertemplate='<b>%{text}</b><br>Label: ' + label + '<extra></extra>',
            name=label
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#111827')),
        scene=dict(
            xaxis=dict(title='', showgrid=True, gridcolor='#E5E7EB', showbackground=True, backgroundcolor='#FAFBFC'),
            yaxis=dict(title='', showgrid=True, gridcolor='#E5E7EB', showbackground=True, backgroundcolor='#FAFBFC'),
            zaxis=dict(title='', showgrid=True, gridcolor='#E5E7EB', showbackground=True, backgroundcolor='#FAFBFC'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=60, b=0),
        height=550,
        autosize=True
    )

    return fig


def create_2d_scatter(coords, labels, texts, title="Vector Space Visualization"):
    """创建 2D 散点图"""
    unique_labels = list(set(labels))

    fig = go.Figure()

    for label in unique_labels:
        mask = [l == label for l in labels]
        indices = [i for i, m in enumerate(mask) if m]

        color = get_label_color(label)

        fig.add_trace(go.Scatter(
            x=coords[indices, 0],
            y=coords[indices, 1],
            mode='markers',
            marker=dict(size=12, color=color, opacity=0.8, line=dict(width=1, color='white')),
            text=[texts[i] for i in indices],
            hovertemplate='<b>%{text}</b><br>Label: ' + label + '<extra></extra>',
            name=label
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#111827')),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=550,
        autosize=True,
        xaxis=dict(showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#D1D5DB'),
        yaxis=dict(showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#D1D5DB')
    )

    return fig


def get_dataset_options():
    """获取数据集选项"""
    options = {k: v['name'] for k, v in PRESET_DATASETS.items()}
    options['custom'] = "Custom Data"
    return options


def visualize(dataset_choice, custom_data, model_choice, method_choice, n_dims):
    """执行可视化"""
    # 加载数据
    if dataset_choice == 'custom':
        try:
            lines = [l.strip() for l in custom_data.strip().split('\n') if l.strip()]
            texts = []
            labels = []
            for line in lines:
                parts = line.split('|')
                if len(parts) >= 2:
                    texts.append(parts[0].strip())
                    labels.append(parts[1].strip())
                else:
                    texts.append(line)
                    labels.append("Uncategorized")
        except:
            return None, "Data format error", pd.DataFrame()
    else:
        if dataset_choice not in PRESET_DATASETS:
            return None, "Please select a dataset", pd.DataFrame()
        dataset = PRESET_DATASETS[dataset_choice]
        texts = [item['text'] for item in dataset['data']]
        labels = [item['label'] for item in dataset['data']]

    if len(texts) < 3:
        return None, "At least 3 data points required", pd.DataFrame()

    # 模型映射
    model_map = {
        "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "MiniLM-L6 (English)": "all-MiniLM-L6-v2"
    }
    model_id = model_map.get(model_choice, "paraphrase-multilingual-MiniLM-L12-v2")

    # 方法映射
    method_map = {
        "PCA (Global)": "pca",
        "t-SNE (Local)": "tsne",
        "UMAP (Balanced)": "umap"
    }
    method = method_map.get(method_choice, "pca")

    # 计算 embeddings
    embeddings = get_batch_embeddings(texts, model_id)
    if embeddings is None:
        return None, "Embedding computation failed", pd.DataFrame()

    # 降维
    n_components = int(n_dims)
    try:
        coords = reduce_dimensions(embeddings, method=method, n_components=n_components)
    except DimensionReductionError as e:
        return None, f"Dimensionality reduction failed: {str(e)}", pd.DataFrame()

    # 创建图表
    title = f"{method_choice} {n_components}D Projection"
    if n_components == 3:
        fig = create_3d_scatter(coords, labels, texts, title)
    else:
        fig = create_2d_scatter(coords, labels, texts, title)

    # 算法说明
    method_info = {
        "pca": "**PCA** preserves global variance structure, good for observing overall distribution. Linear method, fast computation.",
        "tsne": "**t-SNE** preserves local neighborhood relationships, excels at showing clusters. Non-linear method, slower computation.",
        "umap": "**UMAP** balances global and local structure, faster than t-SNE."
    }
    info = method_info.get(method, "")

    # 数据表格
    df = pd.DataFrame({
        "Text": texts,
        "Label": labels,
        f"Coordinates": [f"({', '.join([f'{c:.3f}' for c in coord])})" for coord in coords]
    })

    return fig, info, df


def check_ood_with_viz(new_text, embeddings, texts, labels, model_choice, method_choice, n_dims, threshold=0.4):
    """检测 OOD 并更新可视化"""
    if not new_text or embeddings is None or len(embeddings) == 0:
        return None, ""

    model_map = {
        "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "MiniLM-L6 (English)": "all-MiniLM-L6-v2"
    }
    model_id = model_map.get(model_choice, "paraphrase-multilingual-MiniLM-L12-v2")

    method_map = {
        "PCA (Global)": "pca",
        "t-SNE (Local)": "tsne",
        "UMAP (Balanced)": "umap"
    }
    method = method_map.get(method_choice, "pca")

    # 计算新文本的embedding
    new_embedding = get_batch_embeddings([new_text], model_id)
    if new_embedding is None:
        return None, "Failed to compute embedding"

    # 计算与现有向量的相似度
    similarities = [cosine_similarity(new_embedding[0], emb) for emb in embeddings]
    max_sim = max(similarities) if similarities else 0

    # 合并向量并降维
    all_embeddings = np.vstack([embeddings, new_embedding])
    all_texts = texts + [new_text]
    all_labels = labels + ["NEW"]

    n_components = int(n_dims)
    try:
        coords = reduce_dimensions(all_embeddings, method=method, n_components=n_components)
    except DimensionReductionError as e:
        return None, f"Dimensionality reduction failed: {str(e)}"

    # 创建图表
    title = f"{method_choice} {n_components}D (with new sample)"
    if n_components == 3:
        fig = create_3d_scatter(coords, all_labels, all_texts, title)
    else:
        fig = create_2d_scatter(coords, all_labels, all_texts, title)

    # OOD判断
    is_ood = max_sim < threshold
    result = f"""
### Detection Result

| Metric | Value |
|--------|-------|
| Max Similarity | {max_sim:.4f} |
| Threshold | {threshold} |
| Status | {'**Out-of-Distribution (OOD)**' if is_ood else '**In-Distribution**'} |

{"⚠️ This text is significantly different from training data." if is_ood else "✓ This text is similar to training data."}
"""

    return fig, result


def render():
    """渲染页面"""

    gr.Markdown("""
    Visualize high-dimensional vectors in 2D/3D space using dimensionality reduction.
    - **PCA**: Fast, preserves global structure
    - **t-SNE**: Better for local clusters, slower
    - **UMAP**: Balanced performance
    """)

    with gr.Tabs():
        # Visualization Tab
        with gr.Tab("Visualization"):
            with gr.Row():
                with gr.Column(scale=1):
                    dataset_dropdown = gr.Dropdown(
                        choices=list(get_dataset_options().values()),
                        label="Dataset",
                        value=list(get_dataset_options().values())[0]
                    )

                    custom_input = gr.Textbox(
                        label="Custom Data (text|label per line)",
                        placeholder="apple|fruit\nbanana|fruit\ncarrot|vegetable",
                        lines=5,
                        visible=False
                    )

                    model_dropdown = gr.Dropdown(
                        choices=["Multilingual MiniLM", "MiniLM-L6 (English)"],
                        label="Embedding Model",
                        value="Multilingual MiniLM"
                    )

                    method_dropdown = gr.Dropdown(
                        choices=["PCA (Global)", "t-SNE (Local)", "UMAP (Balanced)"],
                        label="Method",
                        value="PCA (Global)"
                    )

                    dims_radio = gr.Radio(
                        choices=["2", "3"],
                        label="Dimensions",
                        value="2"
                    )

                with gr.Column(scale=2):
                    plot = gr.Plot(label="Vector Space")

            method_info = gr.Markdown("")

            with gr.Accordion("Data Table", open=False):
                data_df = gr.Dataframe(interactive=False)

        # OOD Detection Tab
        with gr.Tab("OOD Detection"):
            gr.Markdown("""
            **Out-of-Distribution Detection**: Test if a new text is similar to the training data.
            If max similarity < threshold, the text is considered OOD.
            """)

            ood_text = gr.Textbox(
                label="Test Text",
                placeholder="Enter text to check...",
                lines=2
            )

            threshold_slider = gr.Slider(
                label="Similarity Threshold",
                minimum=0.1,
                maximum=0.9,
                value=0.4,
                step=0.05
            )

            ood_plot = gr.Plot(label="Updated Visualization")
            ood_result = gr.Markdown("")

    # 数据集切换
    def update_custom_visibility(choice):
        return gr.update(visible=(choice == "Custom Data"))

    dataset_dropdown.change(
        fn=update_custom_visibility,
        inputs=[dataset_dropdown],
        outputs=[custom_input]
    )

    # 可视化更新
    def get_dataset_key(display_name):
        options = get_dataset_options()
        for k, v in options.items():
            if v == display_name:
                return k
        return 'custom'

    def visualize_wrapper(dataset_name, custom_data, model, method, dims):
        dataset_key = get_dataset_key(dataset_name)
        return visualize(dataset_key, custom_data, model, method, dims)

    inputs = [dataset_dropdown, custom_input, model_dropdown, method_dropdown, dims_radio]
    outputs = [plot, method_info, data_df]

    for inp in inputs:
        inp.change(fn=visualize_wrapper, inputs=inputs, outputs=outputs)

    # OOD检测
    ood_state = gr.State({"embeddings": None, "texts": None, "labels": None})

    def update_ood_state(*args):
        dataset_name, custom_data, model, method, dims = args
        dataset_key = get_dataset_key(dataset_name)

        # 获取数据
        if dataset_key == 'custom':
            lines = [l.strip() for l in custom_data.strip().split('\n') if l.strip()]
            texts = []
            labels = []
            for line in lines:
                parts = line.split('|')
                if len(parts) >= 2:
                    texts.append(parts[0].strip())
                    labels.append(parts[1].strip())
                else:
                    texts.append(line)
                    labels.append("Uncategorized")
        else:
            if dataset_key in PRESET_DATASETS:
                dataset = PRESET_DATASETS[dataset_key]
                texts = [item['text'] for item in dataset['data']]
                labels = [item['label'] for item in dataset['data']]
            else:
                return {"embeddings": None, "texts": None, "labels": None}

        model_map = {
            "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
            "MiniLM-L6 (English)": "all-MiniLM-L6-v2"
        }
        model_id = model_map.get(model, "paraphrase-multilingual-MiniLM-L12-v2")

        embeddings = get_batch_embeddings(texts, model_id)

        return {"embeddings": embeddings, "texts": texts, "labels": labels}

    # 数据集变化时更新状态
    for inp in inputs:
        inp.change(fn=update_ood_state, inputs=inputs, outputs=[ood_state])

    # OOD检测触发
    def ood_wrapper(new_text, state, model, method, dims, threshold):
        if state is None or state.get("embeddings") is None:
            return None, "Please select a dataset first"
        return check_ood_with_viz(
            new_text,
            state["embeddings"],
            state["texts"],
            state["labels"],
            model,
            method,
            dims,
            threshold
        )

    ood_text.change(
        fn=ood_wrapper,
        inputs=[ood_text, ood_state, model_dropdown, method_dropdown, dims_radio, threshold_slider],
        outputs=[ood_plot, ood_result]
    )

    threshold_slider.change(
        fn=ood_wrapper,
        inputs=[ood_text, ood_state, model_dropdown, method_dropdown, dims_radio, threshold_slider],
        outputs=[ood_plot, ood_result]
    )

    # 初始化加载
    def on_load():
        """页面加载时计算默认值"""
        return visualize_wrapper(
            list(get_dataset_options().values())[0],
            "",
            "Multilingual MiniLM",
            "PCA (Global)",
            "2"
        )

    return {
        'load_fn': on_load,
        'load_outputs': [plot, method_info, data_df]
    }
