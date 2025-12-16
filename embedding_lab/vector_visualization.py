"""
向量可视化 - 高维向量的降维可视化
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


def create_3d_scatter(coords, labels, texts, title="向量空间可视化"):
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
            hovertemplate='<b>%{text}</b><br>标签: ' + label + '<extra></extra>',
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
        height=550
    )
    
    return fig


def create_2d_scatter(coords, labels, texts, title="向量空间可视化"):
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
            hovertemplate='<b>%{text}</b><br>标签: ' + label + '<extra></extra>',
            name=label
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#111827')),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        xaxis=dict(showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#D1D5DB'),
        yaxis=dict(showgrid=True, gridcolor='#E5E7EB', zeroline=True, zerolinecolor='#D1D5DB')
    )
    
    return fig


def get_dataset_options():
    """获取数据集选项"""
    options = {k: v['name'] for k, v in PRESET_DATASETS.items()}
    options['custom'] = "自定义数据"
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
                    labels.append("未分类")
        except:
            return None, "数据格式错误", pd.DataFrame()
    else:
        if dataset_choice not in PRESET_DATASETS:
            return None, "请选择数据集", pd.DataFrame()
        dataset = PRESET_DATASETS[dataset_choice]
        texts = [item['text'] for item in dataset['data']]
        labels = [item['label'] for item in dataset['data']]
    
    if len(texts) < 3:
        return None, "至少需要 3 条数据", pd.DataFrame()
    
    # 模型映射
    model_map = {
        "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "MiniLM-L6 (英文)": "all-MiniLM-L6-v2"
    }
    model_id = model_map.get(model_choice, "paraphrase-multilingual-MiniLM-L12-v2")
    
    # 方法映射
    method_map = {
        "PCA (全局结构)": "pca",
        "t-SNE (局部结构)": "tsne",
        "UMAP (平衡)": "umap"
    }
    method = method_map.get(method_choice, "pca")
    
    # 计算 embeddings
    embeddings = get_batch_embeddings(texts, model_id)
    if embeddings is None:
        return None, "Embedding 计算失败", pd.DataFrame()
    
    # 降维
    n_components = int(n_dims)
    try:
        coords = reduce_dimensions(embeddings, method=method, n_components=n_components)
    except DimensionReductionError as e:
        return None, f"降维失败：{str(e)}", pd.DataFrame()
    
    # 创建图表
    title = f"{method_choice} {n_components}D 投影"
    if n_components == 3:
        fig = create_3d_scatter(coords, labels, texts, title)
    else:
        fig = create_2d_scatter(coords, labels, texts, title)
    
    # 算法说明
    method_info = {
        "pca": "**PCA** 保持全局方差结构，适合观察整体分布。线性方法，计算快速。",
        "tsne": "**t-SNE** 保持局部邻域关系，擅长展示聚类。非线性方法，计算较慢。",
        "umap": "**UMAP** 平衡全局和局部结构，运行速度快于 t-SNE。"
    }
    info = method_info.get(method, "")
    
    # 数据表格
    df = pd.DataFrame({
        "文本": texts,
        "标签": labels,
        f"坐标": [f"({', '.join([f'{c:.3f}' for c in coord])})" for coord in coords]
    })
    
    return fig, info, df


def check_ood_with_viz(new_text, embeddings, texts, labels, model_choice, method_choice, n_dims, threshold=0.4):
    """检测 OOD 并更新可视化"""
    if not new_text or embeddings is None or len(embeddings) == 0:
        return None, ""
    
    model_map = {
        "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "MiniLM-L6 (英文)": "all-MiniLM-L6-v2"
    }
    model_id = model_map.get(model_choice, "paraphrase-multilingual-MiniLM-L12-v2")
    
    new_emb = get_batch_embeddings([new_text], model_id)
    if new_emb is None:
        return None, "计算失败"
    
    new_emb_vec = new_emb[0]
    
    # 计算相似度
    similarities = [cosine_similarity(new_emb_vec, emb) for emb in embeddings]
    max_sim = max(similarities)
    is_ood = max_sim < threshold
    
    if is_ood:
        result = f"**检测到 OOD 异常点**\n\n最高相似度: {max_sim:.4f} (低于阈值 {threshold})"
        new_label = "OOD"
    else:
        result = f"**在分布内**\n\n最高相似度: {max_sim:.4f}"
        new_label = "新数据点"
    
    # 重新降维并可视化（包含新点）
    method_map = {
        "PCA (全局结构)": "pca",
        "t-SNE (局部结构)": "tsne",
        "UMAP (平衡)": "umap"
    }
    method = method_map.get(method_choice, "pca")
    n_components = int(n_dims)
    
    # 合并 embeddings
    all_embeddings = np.vstack([embeddings, new_emb])
    all_texts = texts + [f"[新] {new_text[:30]}..."]
    all_labels = labels + [new_label]
    
    try:
        coords = reduce_dimensions(all_embeddings, method=method, n_components=n_components)
    except DimensionReductionError:
        return None, result
    
    # 创建图表
    title = f"{method_choice} {n_components}D 投影 (含新数据点)"
    
    # 自定义颜色：新点用特殊颜色
    if n_components == 3:
        fig = go.Figure()
        unique_labels = list(set(all_labels))
        for label in unique_labels:
            mask = [l == label for l in all_labels]
            indices = [i for i, m in enumerate(mask) if m]
            
            if label == "OOD":
                color = "#DC2626"
                size = 14
                symbol = "diamond"
            elif label == "新数据点":
                color = "#059669"
                size = 14
                symbol = "diamond"
            else:
                color = get_label_color(label)
                size = 8
                symbol = "circle"
            
            fig.add_trace(go.Scatter3d(
                x=coords[indices, 0],
                y=coords[indices, 1],
                z=coords[indices, 2],
                mode='markers',
                marker=dict(size=size, color=color, opacity=0.9, symbol=symbol, line=dict(width=1, color='white')),
                text=[all_texts[i] for i in indices],
                hovertemplate='<b>%{text}</b><br>标签: ' + label + '<extra></extra>',
                name=label
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#111827')),
            scene=dict(
                xaxis=dict(title='', showgrid=True, gridcolor='#E5E7EB', showbackground=True, backgroundcolor='#FAFBFC'),
                yaxis=dict(title='', showgrid=True, gridcolor='#E5E7EB', showbackground=True, backgroundcolor='#FAFBFC'),
                zaxis=dict(title='', showgrid=True, gridcolor='#E5E7EB', showbackground=True, backgroundcolor='#FAFBFC'),
            ),
            paper_bgcolor='#FFFFFF',
            font=dict(color='#111827', family='Inter, sans-serif'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=550
        )
    else:
        fig = go.Figure()
        unique_labels = list(set(all_labels))
        for label in unique_labels:
            mask = [l == label for l in all_labels]
            indices = [i for i, m in enumerate(mask) if m]
            
            if label == "OOD":
                color = "#DC2626"
                size = 18
                symbol = "diamond"
            elif label == "新数据点":
                color = "#059669"
                size = 18
                symbol = "diamond"
            else:
                color = get_label_color(label)
                size = 12
                symbol = "circle"
            
            fig.add_trace(go.Scatter(
                x=coords[indices, 0],
                y=coords[indices, 1],
                mode='markers',
                marker=dict(size=size, color=color, opacity=0.9, symbol=symbol, line=dict(width=1, color='white')),
                text=[all_texts[i] for i in indices],
                hovertemplate='<b>%{text}</b><br>标签: ' + label + '<extra></extra>',
                name=label
            ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#111827')),
            plot_bgcolor='#FAFBFC',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#111827', family='Inter, sans-serif'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500,
            xaxis=dict(showgrid=True, gridcolor='#E5E7EB'),
            yaxis=dict(showgrid=True, gridcolor='#E5E7EB')
        )
    
    return fig, result


def render():
    """渲染页面"""
    
    gr.Markdown("## 向量可视化")
    
    # 数据集选择
    dataset_options = list(get_dataset_options().keys())
    dataset_choice = gr.Dropdown(
        choices=dataset_options,
        label="预置数据集",
        value=dataset_options[0] if dataset_options else None
    )
    
    custom_data = gr.Textbox(
        label="自定义数据（每行格式: 文本|标签）",
        placeholder="今天天气很好|正面\n我很开心|正面\n真是糟糕的一天|负面",
        lines=5,
        visible=False
    )
    
    # 控制面板
    with gr.Row():
        model_choice = gr.Dropdown(
            choices=["Multilingual MiniLM", "MiniLM-L6 (英文)"],
            label="Embedding 模型",
            value="Multilingual MiniLM"
        )
        method_choice = gr.Dropdown(
            choices=["PCA (全局结构)", "t-SNE (局部结构)", "UMAP (平衡)"],
            label="降维算法",
            value="PCA (全局结构)"
        )
        n_dims = gr.Radio(
            choices=["2", "3"],
            label="维度",
            value="3"
        )
    
    viz_btn = gr.Button("开始可视化", variant="primary")
    
    # 结果展示
    method_info = gr.Markdown("")
    plot = gr.Plot(label="可视化结果")
    
    # OOD 检测
    ood_input = gr.Textbox(
        label="OOD 检测 - 输入新句子",
        placeholder="例如：量子力学真是太难了"
    )
    ood_result = gr.Markdown("")
    
    with gr.Accordion("数据详情", open=False):
        data_df = gr.Dataframe(interactive=False)
    
    # 状态存储
    embeddings_state = gr.State(None)
    texts_state = gr.State([])
    labels_state = gr.State([])
    
    # ==================== 事件绑定 ====================
    
    def on_dataset_change(choice):
        return gr.update(visible=(choice == 'custom'))
    
    dataset_choice.change(fn=on_dataset_change, inputs=[dataset_choice], outputs=[custom_data])
    
    def run_viz(dataset, custom, model, method, dims):
        fig, info, df = visualize(dataset, custom, model, method, int(dims))
        
        # 保存 embeddings 和 labels 用于 OOD 检测
        if dataset == 'custom':
            lines = [l.strip() for l in custom.strip().split('\n') if l.strip()]
            texts = []
            labels = []
            for line in lines:
                parts = line.split('|')
                if len(parts) >= 2:
                    texts.append(parts[0].strip())
                    labels.append(parts[1].strip())
                else:
                    texts.append(line)
                    labels.append("未分类")
        else:
            if dataset in PRESET_DATASETS:
                texts = [item['text'] for item in PRESET_DATASETS[dataset]['data']]
                labels = [item['label'] for item in PRESET_DATASETS[dataset]['data']]
            else:
                texts = []
                labels = []
        
        model_map = {
            "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
            "MiniLM-L6 (英文)": "all-MiniLM-L6-v2"
        }
        model_id = model_map.get(model, "paraphrase-multilingual-MiniLM-L12-v2")
        embeddings = get_batch_embeddings(texts, model_id) if texts else None
        
        return fig, info, df, embeddings, texts, labels
    
    viz_btn.click(
        fn=run_viz,
        inputs=[dataset_choice, custom_data, model_choice, method_choice, n_dims],
        outputs=[plot, method_info, data_df, embeddings_state, texts_state, labels_state]
    )
    
    # OOD 检测 - 即时回显并更新图表
    ood_input.change(
        fn=check_ood_with_viz,
        inputs=[ood_input, embeddings_state, texts_state, labels_state, model_choice, method_choice, n_dims],
        outputs=[plot, ood_result]
    )