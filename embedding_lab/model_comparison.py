"""
Lab 2: 模型对比 - 对比不同 Embedding 模型的特性
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Tuple

from embedding_lab.embedding_utils import (
    get_batch_embeddings,
    compute_sparse_embeddings,
    cosine_similarity,
    load_sentence_transformer,
    EMBEDDING_MODELS
)


def compute_similarity_scores(query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> List[float]:
    """计算 query 与所有 candidates 的相似度"""
    scores = []
    for emb in candidate_embeddings:
        scores.append(cosine_similarity(query_embedding, emb))
    return scores


def create_comparison_chart(
    candidates: List[str],
    scores_dict: Dict[str, List[float]],
    query: str
) -> go.Figure:
    """创建模型对比柱状图"""
    fig = go.Figure()
    
    colors = ['#2563EB', '#059669', '#D97706', '#7C3AED', '#DC2626']
    
    for i, (model_name, scores) in enumerate(scores_dict.items()):
        fig.add_trace(go.Bar(
            name=model_name,
            x=candidates,
            y=scores,
            marker_color=colors[i % len(colors)],
            text=[f'{s:.3f}' for s in scores],
            textposition='outside',
            textfont=dict(size=11, color='#111827')
        ))
    
    fig.update_layout(
        title=dict(
            text=f'Query: "{query}"',
            font=dict(size=14, color='#6B7280')
        ),
        barmode='group',
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=80),
        height=400,
        xaxis=dict(
            tickangle=-20,
            gridcolor='#E5E7EB'
        ),
        yaxis=dict(
            title='相似度',
            range=[0, 1],
            gridcolor='#E5E7EB'
        )
    )
    
    return fig


def render_score_table(candidates: List[str], scores_dict: Dict[str, List[float]]):
    """渲染详细分数表格"""
    data = {'候选文本': candidates}
    for model_name, scores in scores_dict.items():
        data[model_name] = [f'{s:.4f}' for s in scores]
    
    df = pd.DataFrame(data)
    
    # 高亮最高分
    def highlight_max(s):
        if s.name == '候选文本':
            return [''] * len(s)
        numeric_vals = [float(v) for v in s]
        is_max = [v == max(numeric_vals) for v in numeric_vals]
        return ['background-color: #D1FAE5' if v else '' for v in is_max]
    
    styled_df = df.style.apply(highlight_max)
    st.dataframe(styled_df, width="stretch", hide_index=True)


def render_token_attention_hint():
    """渲染 Token 级别的注意力提示"""
    pass


def render():
    """渲染 Lab 2: 模型对比 页面"""
    st.markdown('<h1 class="module-title">模型对比</h1>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 模型选择
    st.markdown("### 选择对比模型")
    
    available_models = {
        "TF-IDF": "tfidf",
        "BM25": "bm25",
        "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
        "MiniLM-L6 (英文)": "all-MiniLM-L6-v2",
    }
    
    model_cols = st.columns(len(available_models))
    selected_models = []
    
    for i, (name, model_id) in enumerate(available_models.items()):
        with model_cols[i]:
            if st.checkbox(name, value=(i < 2), key=f"model_{model_id}"):
                selected_models.append((name, model_id))
    
    if len(selected_models) < 1:
        st.warning("请至少选择一个模型")
        return
    
    st.markdown("---")
    
    # 输入区域
    st.markdown("### 输入测试数据")
    
    col_query, col_candidates = st.columns([1, 2])
    
    with col_query:
        st.markdown("#### Query")
        query = st.text_input("查询文本", value="苹果", key="comparison_query",
                             help="输入要搜索的关键词或句子")
    
    with col_candidates:
        st.markdown("#### Candidates")
        default_candidates = "水果\n手机\n乔布斯\n红色的球\n苹果公司发布新产品\n我喜欢吃苹果"
        candidates_text = st.text_area(
            "候选文本（每行一个）",
            value=default_candidates,
            height=150,
            key="comparison_candidates",
            help="输入多个候选文本，每行一个"
        )
    
    # 预设案例
    st.markdown("#### 预设案例")
    presets = [
        ("苹果歧义", "苹果", "水果\n手机\n乔布斯\n红色的球\n苹果发布新产品\n我喜欢吃苹果"),
        ("银行歧义", "银行", "金融机构\n河边\n存款取款\n银行卡\n河岸风景"),
        ("特斯拉", "特斯拉", "电动汽车\n科学家\n马斯克\n电磁感应\nModel 3"),
        ("语义搜索", "如何学习编程", "编程入门教程\n学习Python\n代码怎么写\n程序员成长\n软件开发"),
    ]
    
    def set_comparison_preset(q: str, c: str):
        """回调函数：设置预设值"""
        st.session_state.comparison_query = q
        st.session_state.comparison_candidates = c
    
    preset_cols = st.columns(len(presets))
    for i, (label, q, c) in enumerate(presets):
        with preset_cols[i]:
            st.button(
                label, 
                key=f"preset_compare_{i}", 
                width="stretch",
                on_click=set_comparison_preset,
                args=(q, c)
            )
    
    st.markdown("---")
    
    # 执行对比
    if query and candidates_text:
        candidates = [c.strip() for c in candidates_text.strip().split('\n') if c.strip()]
        
        if not candidates:
            st.warning("请输入候选文本")
            return
        
        all_texts = [query] + candidates
        scores_dict = {}
        
        with st.spinner("计算 Embeddings..."):
            for model_name, model_id in selected_models:
                try:
                    if model_id in ["tfidf", "bm25"]:
                        # 稀疏向量
                        embeddings = compute_sparse_embeddings(all_texts, model_id)
                        query_emb = embeddings[0]
                        candidate_embs = embeddings[1:]
                    else:
                        # Dense 向量
                        embeddings = get_batch_embeddings(all_texts, model_id)
                        if embeddings is None:
                            st.warning(f"模型 {model_name} 加载失败")
                            continue
                        query_emb = embeddings[0]
                        candidate_embs = embeddings[1:]
                    
                    scores = compute_similarity_scores(query_emb, candidate_embs)
                    scores_dict[model_name] = scores
                    
                except Exception as e:
                    st.warning(f"模型 {model_name} 计算失败: {e}")
        
        if not scores_dict:
            st.error("所有模型计算失败")
            return
        
        # 显示结果
        st.markdown("### 对比结果")
        
        # 柱状图
        fig = create_comparison_chart(candidates, scores_dict, query)
        st.plotly_chart(fig, width="stretch")
        
        # 详细表格
        with st.expander("详细分数", expanded=True):
            render_score_table(candidates, scores_dict)
        
        # 排序对比
        st.markdown("### 排序差异分析")
        
        rank_cols = st.columns(len(scores_dict))
        for i, (model_name, scores) in enumerate(scores_dict.items()):
            with rank_cols[i]:
                st.markdown(f"**{model_name}** 排序")
                
                # 按分数排序
                sorted_indices = np.argsort(scores)[::-1]
                
                # 使用容器显示排序结果
                with st.container():
                    for rank, idx in enumerate(sorted_indices, 1):
                        score = scores[idx]
                        text = candidates[idx][:20] + ('...' if len(candidates[idx]) > 20 else '')
                        
                        st.text(f"{rank}. {text} — {score:.3f}")
        
        # 洞察分析
        st.markdown("### 洞察")
        
        # 找出差异最大的案例
        if len(scores_dict) >= 2:
            model_names = list(scores_dict.keys())
            scores_1 = scores_dict[model_names[0]]
            scores_2 = scores_dict[model_names[1]]
            
            rank_diff = []
            for i, candidate in enumerate(candidates):
                rank_1 = sorted(range(len(scores_1)), key=lambda k: scores_1[k], reverse=True).index(i) + 1
                rank_2 = sorted(range(len(scores_2)), key=lambda k: scores_2[k], reverse=True).index(i) + 1
                rank_diff.append((candidate, abs(rank_1 - rank_2), rank_1, rank_2))
            
            rank_diff.sort(key=lambda x: x[1], reverse=True)
            
            if rank_diff[0][1] > 0:
                most_diff = rank_diff[0]
                st.warning(f"""
**最大排序差异**

文本 「{most_diff[0][:30]}...」 在 **{model_names[0]}** 中排第 **{most_diff[2]}** 名，在 **{model_names[1]}** 中排第 **{most_diff[3]}** 名。

_这说明不同模型对语义的理解存在显著差异。_
""")
        
        # 模型特点说明
        with st.expander("模型特点说明"):
            st.markdown("""
            | 模型类型 | 代表 | 特点 | 适用场景 |
            |---------|------|------|---------|
            | **稀疏向量** | TF-IDF, BM25 | 基于词频统计，可解释性强 | 关键词搜索、精确匹配 |
            | **静态向量** | Word2Vec, GloVe | 每个词一个固定向量 | 词相似度、简单语义 |
            | **上下文向量** | BERT, BGE | 考虑上下文的动态向量 | 语义搜索、问答匹配 |
            
            **关键差异**：
            - TF-IDF/BM25 只看「是否包含相同的词」
            - Dense 模型能理解「苹果」在不同语境下指水果还是公司
            """)

