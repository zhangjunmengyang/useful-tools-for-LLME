"""
Lab 1: 向量运算 - Word2Vec 经典类比推理演示
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple

from embedding_lab.embedding_utils import (
    load_word2vec_model,
    get_word_vector,
    vector_arithmetic,
    cosine_similarity,
    reduce_dimensions
)


def create_vector_visualization(
    words: List[str],
    vectors: List[np.ndarray],
    result_word: str = None,
    result_vector: np.ndarray = None,
    operation_path: List[Tuple[str, str]] = None
) -> go.Figure:
    """
    创建向量运算的 2D 可视化
    """
    all_words = words.copy()
    all_vectors = [v for v in vectors]
    
    if result_word and result_vector is not None:
        all_words.append(result_word)
        all_vectors.append(result_vector)
    
    if len(all_vectors) < 2:
        return None
    
    # 降维到 2D
    vectors_array = np.array(all_vectors)
    coords_2d = reduce_dimensions(vectors_array, method="pca", n_components=2)
    
    fig = go.Figure()
    
    # 绘制输入词的点
    colors = ['#2563EB', '#059669', '#D97706', '#7C3AED']
    for i, (word, coord) in enumerate(zip(all_words[:-1] if result_word else all_words, coords_2d[:-1] if result_word else coords_2d)):
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
    
    # 绘制结果词的点
    if result_word:
        fig.add_trace(go.Scatter(
            x=[coords_2d[-1][0]],
            y=[coords_2d[-1][1]],
            mode='markers+text',
            marker=dict(size=18, color='#DC2626', symbol='star'),
            text=[result_word],
            textposition='top center',
            textfont=dict(size=14, color='#DC2626', family='Inter'),
            name=f'结果: {result_word}',
            hoverinfo='text',
            hovertext=f'结果: {result_word}'
        ))
    
    # 绘制运算路径箭头
    if operation_path and len(coords_2d) >= 3:
        # King -> King - Man
        fig.add_annotation(
            x=coords_2d[1][0],
            y=coords_2d[1][1],
            ax=coords_2d[0][0],
            ay=coords_2d[0][1],
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowwidth=2,
            arrowcolor='#DC2626',
            opacity=0.6
        )
        
        if len(coords_2d) >= 4:
            # King - Man -> King - Man + Woman
            fig.add_annotation(
                x=coords_2d[2][0],
                y=coords_2d[2][1],
                ax=coords_2d[1][0],
                ay=coords_2d[1][1],
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor='#059669',
                opacity=0.6
            )
    
    fig.update_layout(
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        showlegend=False,
        height=400,
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


def render_similar_words_table(results: List[Tuple[str, float]], exclude_words: List[str] = None):
    """渲染相似词表格"""
    if not results:
        st.info("未找到相似词")
        return
    
    exclude = set(w.lower() for w in (exclude_words or []))
    
    html = ['<div style="background: #F3F4F6; border-radius: 8px; padding: 16px;">']
    html.append('<table style="width: 100%; border-collapse: collapse;">')
    html.append('<tr style="border-bottom: 1px solid #E5E7EB;">')
    html.append('<th style="text-align: left; padding: 8px; color: #6B7280; font-size: 12px; font-weight: 500;">排名</th>')
    html.append('<th style="text-align: left; padding: 8px; color: #6B7280; font-size: 12px; font-weight: 500;">词汇</th>')
    html.append('<th style="text-align: right; padding: 8px; color: #6B7280; font-size: 12px; font-weight: 500;">相似度</th>')
    html.append('</tr>')
    
    rank = 1
    for word, score in results:
        if word.lower() in exclude:
            continue
        
        # 相似度颜色渐变
        if score > 0.7:
            color = '#059669'
        elif score > 0.5:
            color = '#D97706'
        else:
            color = '#6B7280'
        
        # 高亮第一名
        bg_color = '#DBEAFE' if rank == 1 else 'transparent'
        font_weight = '600' if rank == 1 else '400'
        
        html.append(f'<tr style="background: {bg_color};">')
        html.append(f'<td style="padding: 10px 8px; color: #111827; font-size: 14px;">{rank}</td>')
        html.append(f'<td style="padding: 10px 8px; color: #111827; font-size: 14px; font-weight: {font_weight}; font-family: \'JetBrains Mono\', monospace;">{word}</td>')
        html.append(f'<td style="padding: 10px 8px; text-align: right; color: {color}; font-size: 14px; font-family: \'JetBrains Mono\', monospace;">{score:.4f}</td>')
        html.append('</tr>')
        rank += 1
    
    html.append('</table>')
    html.append('</div>')
    
    st.markdown(''.join(html), unsafe_allow_html=True)


def render():
    """渲染 Lab 1: 向量运算 页面"""
    st.markdown('<h1 class="module-title">向量运算</h1>', unsafe_allow_html=True)
    
    # 加载模型
    with st.spinner("正在加载 Word2Vec 模型..."):
        model = load_word2vec_model()
    
    if model is None:
        st.error("模型加载失败，请检查网络连接")
        return
    
    st.markdown("---")
    
    # 向量计算器
    st.markdown("### 向量计算器")
    
    col1, col2, col3, col4, col5 = st.columns([2, 1, 2, 1, 2])
    
    with col1:
        word_a = st.text_input("词 A", value="king", key="word_a", 
                               help="输入英文单词（小写）")
    
    with col2:
        st.markdown("<div style='text-align: center; padding-top: 32px; font-size: 24px; color: #DC2626;'>−</div>", 
                   unsafe_allow_html=True)
    
    with col3:
        word_b = st.text_input("词 B", value="man", key="word_b",
                               help="将被减去的词")
    
    with col4:
        st.markdown("<div style='text-align: center; padding-top: 32px; font-size: 24px; color: #059669;'>+</div>", 
                   unsafe_allow_html=True)
    
    with col5:
        word_c = st.text_input("词 C", value="woman", key="word_c",
                               help="将被加上的词")
    
    # Top-K 设置
    top_k = st.slider("返回结果数 (Top-K)", min_value=1, max_value=20, value=10, key="top_k")
    
    # 预设案例
    st.markdown("#### 经典案例")
    presets = [
        ("King - Man + Woman", "king", "man", "woman"),
        ("Paris - France + Germany", "paris", "france", "germany"),
        ("Car - Road + Water", "car", "road", "water"),
        ("Brother - Man + Woman", "brother", "man", "woman"),
        ("Walking - Walk + Swim", "walking", "walk", "swim"),
    ]
    
    def set_preset(a: str, b: str, c: str):
        """回调函数：设置预设值"""
        st.session_state.word_a = a
        st.session_state.word_b = b
        st.session_state.word_c = c
    
    preset_cols = st.columns(len(presets))
    for i, (label, a, b, c) in enumerate(presets):
        with preset_cols[i]:
            st.button(
                label, 
                key=f"preset_{i}", 
                width="stretch",
                on_click=set_preset,
                args=(a, b, c)
            )
    
    st.markdown("---")
    
    # 执行计算
    if word_a and word_b and word_c:
        word_a = word_a.lower().strip()
        word_b = word_b.lower().strip()
        word_c = word_c.lower().strip()
        
        # 检查词是否在词表中
        missing = []
        for w in [word_a, word_b, word_c]:
            if get_word_vector(model, w) is None:
                missing.append(w)
        
        if missing:
            st.error(f"以下词不在词表中: {', '.join(missing)}")
            st.info("请使用常见的英文单词（小写）")
        else:
            # 显示公式
            st.markdown(f"""
            <div style="text-align: center; padding: 16px; background: #F3F4F6; border-radius: 8px; margin-bottom: 16px;">
                <span style="font-size: 20px; font-family: 'JetBrains Mono', monospace; color: #111827;">
                    <span style="color: #2563EB;">{word_a}</span> − 
                    <span style="color: #DC2626;">{word_b}</span> + 
                    <span style="color: #059669;">{word_c}</span> = 
                    <span style="color: #7C3AED;">?</span>
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # 执行向量运算
            results = vector_arithmetic(model, positive=[word_a, word_c], negative=[word_b], topn=top_k)
            
            col_result, col_viz = st.columns([1, 1])
            
            with col_result:
                st.markdown("#### 最相似的词")
                render_similar_words_table(results, exclude_words=[word_a, word_b, word_c])
                
            with col_viz:
                st.markdown("#### 向量空间投影")
                
                # 获取向量
                vec_a = get_word_vector(model, word_a)
                vec_b = get_word_vector(model, word_b)
                vec_c = get_word_vector(model, word_c)
                
                words = [word_a, word_b, word_c]
                vectors = [vec_a, vec_b, vec_c]
                
                result_word = results[0][0] if results else None
                result_vec = get_word_vector(model, result_word) if result_word else None
                
                fig = create_vector_visualization(
                    words, vectors,
                    result_word=result_word,
                    result_vector=result_vec,
                    operation_path=[(word_a, word_b), (word_b, word_c)]
                )
                
                if fig:
                    st.plotly_chart(fig, width="stretch")
            
            # 详细信息
            with st.expander("向量详细信息"):
                st.markdown("#### 向量维度与相似度")
                
                # 计算向量运算结果
                target_vec = vec_a - vec_b + vec_c
                
                info_cols = st.columns(4)
                with info_cols[0]:
                    st.metric("向量维度", vec_a.shape[0])
                with info_cols[1]:
                    st.metric(f"{word_a} ↔ {word_b}", f"{cosine_similarity(vec_a, vec_b):.4f}")
                with info_cols[2]:
                    st.metric(f"{word_a} ↔ {word_c}", f"{cosine_similarity(vec_a, vec_c):.4f}")
                with info_cols[3]:
                    st.metric(f"{word_b} ↔ {word_c}", f"{cosine_similarity(vec_b, vec_c):.4f}")
                
                if result_word and result_vec is not None:
                    st.markdown(f"**计算结果与 {result_word} 的相似度**: `{cosine_similarity(target_vec, result_vec):.4f}`")

