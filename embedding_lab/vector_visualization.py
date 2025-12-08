"""
Lab 3: 向量可视化 - 高维向量的降维可视化
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional

from embedding_lab.embedding_utils import (
    get_batch_embeddings,
    reduce_dimensions,
    get_label_color,
    PRESET_DATASETS,
    DimensionReductionError,
    MIN_SAMPLES
)


def create_3d_scatter(
    coords: np.ndarray,
    labels: List[str],
    texts: List[str],
    title: str = "向量空间可视化"
) -> go.Figure:
    """创建 3D 散点图"""
    
    # 为每个标签分配颜色
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
        title=dict(
            text=title,
            font=dict(size=16, color='#111827')
        ),
        scene=dict(
            xaxis=dict(
                title='',
                showgrid=True,
                gridcolor='#E5E7EB',
                showbackground=True,
                backgroundcolor='#FAFBFC'
            ),
            yaxis=dict(
                title='',
                showgrid=True,
                gridcolor='#E5E7EB',
                showbackground=True,
                backgroundcolor='#FAFBFC'
            ),
            zaxis=dict(
                title='',
                showgrid=True,
                gridcolor='#E5E7EB',
                showbackground=True,
                backgroundcolor='#FAFBFC'
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        height=550
    )
    
    return fig


def create_2d_scatter(
    coords: np.ndarray,
    labels: List[str],
    texts: List[str],
    title: str = "向量空间可视化"
) -> go.Figure:
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
            marker=dict(
                size=12,
                color=color,
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=[texts[i] for i in indices],
            hovertemplate='<b>%{text}</b><br>标签: ' + label + '<extra></extra>',
            name=label
        ))
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='#111827')
        ),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=500,
        xaxis=dict(
            showgrid=True,
            gridcolor='#E5E7EB',
            zeroline=True,
            zerolinecolor='#D1D5DB'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#E5E7EB',
            zeroline=True,
            zerolinecolor='#D1D5DB'
        )
    )
    
    return fig


def add_ood_point(
    fig: go.Figure,
    coord: np.ndarray,
    text: str,
    is_3d: bool = True
) -> go.Figure:
    """在图上添加 OOD 异常点"""
    
    if is_3d:
        fig.add_trace(go.Scatter3d(
            x=[coord[0]],
            y=[coord[1]],
            z=[coord[2]],
            mode='markers+text',
            marker=dict(
                size=15,
                color='#DC2626',
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            text=[text],
            textposition='top center',
            textfont=dict(size=12, color='#DC2626'),
            hovertemplate='<b>OOD: %{text}</b><extra></extra>',
            name='OOD 异常点'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[coord[0]],
            y=[coord[1]],
            mode='markers+text',
            marker=dict(
                size=18,
                color='#DC2626',
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            text=[text],
            textposition='top center',
            textfont=dict(size=12, color='#DC2626'),
            hovertemplate='<b>OOD: %{text}</b><extra></extra>',
            name='OOD 异常点'
        ))
    
    return fig


def check_ood(new_embedding: np.ndarray, existing_embeddings: np.ndarray, threshold: float = 0.3) -> tuple:
    """
    检测是否为 OOD（分布外）数据
    返回：(是否OOD, 与最近点的平均距离)
    """
    from embedding_lab.embedding_utils import cosine_similarity
    
    similarities = [cosine_similarity(new_embedding, emb) for emb in existing_embeddings]
    max_sim = max(similarities)
    avg_sim = np.mean(similarities)
    
    # 如果与所有点的最大相似度都很低，则判定为 OOD
    is_ood = max_sim < threshold
    
    return is_ood, max_sim, avg_sim


def render():
    """渲染 Lab 3: 向量可视化 页面"""
    st.markdown('<h1 class="module-title">向量可视化</h1>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 数据集选择
    st.markdown("### 选择数据集")
    
    dataset_options = {k: v['name'] for k, v in PRESET_DATASETS.items()}
    dataset_options['custom'] = "自定义数据"
    
    selected_dataset = st.selectbox(
        "预置数据集",
        options=list(dataset_options.keys()),
        format_func=lambda x: dataset_options[x],
        key="viz_dataset"
    )
    
    # 加载数据
    if selected_dataset == 'custom':
        st.markdown("#### 输入自定义数据")
        custom_input = st.text_area(
            "每行格式: 文本|标签",
            value="今天天气很好|正面\n我很开心|正面\n真是糟糕的一天|负面\n太失望了|负面",
            height=150,
            key="custom_data"
        )
        
        try:
            lines = [l.strip() for l in custom_input.strip().split('\n') if l.strip()]
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
            st.error("数据格式错误")
            return
    else:
        dataset = PRESET_DATASETS[selected_dataset]
        texts = [item['text'] for item in dataset['data']]
        labels = [item['label'] for item in dataset['data']]
        
        st.caption(f"{dataset['description']}，共 {len(texts)} 条数据")
    
    if len(texts) < 3:
        st.warning("至少需要 3 条数据")
        return
    
    st.markdown("---")
    
    # 控制面板
    col_model, col_method, col_dim = st.columns([1, 1, 1])
    
    with col_model:
        model_options = {
            "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM",
            "all-MiniLM-L6-v2": "MiniLM-L6 (英文)"
        }
        selected_model = st.selectbox(
            "Embedding 模型",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            key="viz_model"
        )
    
    with col_method:
        method_options = {
            "pca": "PCA (全局结构)",
            "tsne": "t-SNE (局部结构)",
            "umap": "UMAP (平衡)"
        }
        selected_method = st.selectbox(
            "降维算法",
            options=list(method_options.keys()),
            format_func=lambda x: method_options[x],
            key="viz_method"
        )
    
    with col_dim:
        n_dims = st.radio("可视化维度", options=[2, 3], index=1, horizontal=True, key="viz_dims")
    
    # 计算并可视化
    # 显示算法对数据量的要求
    min_samples_info = f"PCA: ≥{MIN_SAMPLES['pca']} 个 | t-SNE: ≥{MIN_SAMPLES['tsne']} 个 | UMAP: ≥{MIN_SAMPLES['umap']} 个"
    st.caption(f"算法最小数据量要求：{min_samples_info}")
    
    if st.button("生成可视化", type="primary", width="stretch"):
        with st.spinner("计算 Embeddings..."):
            embeddings = get_batch_embeddings(texts, selected_model)
        
        if embeddings is None:
            st.error("Embedding 计算失败")
            return
        
        # 保存到 session state
        st.session_state.viz_embeddings = embeddings
        st.session_state.viz_texts = texts
        st.session_state.viz_labels = labels
        
        # 执行降维，捕获数据量不足的错误
        try:
            with st.spinner(f"执行 {method_options[selected_method]} 降维..."):
                coords = reduce_dimensions(embeddings, method=selected_method, n_components=n_dims)
            
            st.session_state.viz_coords = coords
            st.session_state.viz_method_used = selected_method
            st.session_state.viz_dims_used = n_dims
            
        except DimensionReductionError as e:
            st.error(f"降维失败：{str(e)}")
            # 清除之前的可视化结果
            if 'viz_coords' in st.session_state:
                del st.session_state.viz_coords
            return
    
    # 显示可视化结果
    if 'viz_coords' in st.session_state:
        coords = st.session_state.viz_coords
        texts_viz = st.session_state.viz_texts
        labels_viz = st.session_state.viz_labels
        n_dims_used = st.session_state.viz_dims_used
        method_used = st.session_state.viz_method_used
        
        st.markdown("### 可视化结果")
        
        # 算法说明
        method_info = {
            "pca": ("PCA 保持全局方差结构，适合观察整体分布。线性方法，计算快速。", "#2563EB"),
            "tsne": ("t-SNE 保持局部邻域关系，擅长展示聚类。非线性方法，计算较慢。", "#059669"),
            "umap": ("UMAP 平衡全局和局部结构，运行速度快于 t-SNE。", "#7C3AED")
        }
        
        info_text, info_color = method_info[method_used]
        st.markdown(f"""
        <div style="background: {info_color}10; border-left: 3px solid {info_color}; 
                    padding: 12px; margin-bottom: 16px; border-radius: 0 6px 6px 0;">
            <p style="color: {info_color}; margin: 0; font-size: 13px;">
                <strong>当前算法：{method_options[method_used]}</strong><br/>
                {info_text}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 创建图表
        if n_dims_used == 3:
            fig = create_3d_scatter(coords, labels_viz, texts_viz, 
                                   title=f"{method_options[method_used]} 3D 投影")
        else:
            fig = create_2d_scatter(coords, labels_viz, texts_viz,
                                   title=f"{method_options[method_used]} 2D 投影")
        
        # OOD 检测
        st.markdown("---")
        st.markdown("### OOD 检测 (Out-of-Distribution)")
        
        ood_input = st.text_input(
            "输入一个新句子，检测是否为分布外数据",
            placeholder="例如：量子力学真是太难了",
            key="ood_input"
        )
        
        if ood_input and 'viz_embeddings' in st.session_state:
            with st.spinner("分析中..."):
                # 计算新句子的 embedding
                new_emb = get_batch_embeddings([ood_input], selected_model)
                
                if new_emb is not None:
                    new_emb = new_emb[0]
                    existing_embs = st.session_state.viz_embeddings
                    
                    # 检测 OOD
                    is_ood, max_sim, avg_sim = check_ood(new_emb, existing_embs, threshold=0.4)
                    
                    # 计算新点的坐标
                    all_embs = np.vstack([existing_embs, new_emb.reshape(1, -1)])
                    all_coords = reduce_dimensions(all_embs, method=method_used, n_components=n_dims_used)
                    new_coord = all_coords[-1]
                    
                    # 添加到图表
                    fig = add_ood_point(fig, new_coord, ood_input[:15] + "...", is_3d=(n_dims_used == 3))
                    
                    # 显示 OOD 分析结果
                    if is_ood:
                        st.warning(f"检测到 OOD 异常点，最高相似度: {max_sim:.4f}")
                    else:
                        st.info(f"在分布内，最高相似度: {max_sim:.4f}")
        
        # 显示最终图表
        st.plotly_chart(fig, width="stretch")
        
        st.caption("提示：拖动旋转 3D 视图，滚轮缩放，点击图例可隐藏/显示类别")
        
        # 数据表格
        with st.expander("数据详情"):
            import pandas as pd
            df = pd.DataFrame({
                "文本": texts_viz,
                "标签": labels_viz,
                f"坐标 ({method_options[method_used]})": [
                    f"({', '.join([f'{c:.3f}' for c in coord])})" 
                    for coord in coords
                ]
            })
            st.dataframe(df, width="stretch", hide_index=True)
        
        # 降维算法对比说明
        with st.expander("降维算法对比"):
            st.markdown("""
            | 算法 | 类型 | 保持结构 | 计算复杂度 | 适用场景 |
            |------|------|---------|-----------|---------|
            | **PCA** | 线性 | 全局方差 | O(n²) | 快速预览、线性可分数据 |
            | **t-SNE** | 非线性 | 局部邻域 | O(n²) ~ O(n log n) | 聚类可视化、探索性分析 |
            | **UMAP** | 非线性 | 全局+局部 | O(n log n) | 大数据集、保持全局结构 |
            
            **面试考点**：
            - PCA 是线性变换，可能无法揭示非线性结构
            - t-SNE 的困惑度(perplexity)参数影响局部/全局平衡
            - UMAP 通常比 t-SNE 更快，且更好地保持全局结构
            """)

