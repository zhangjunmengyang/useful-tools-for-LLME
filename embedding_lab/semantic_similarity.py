"""
Lab 4: 语义相似度 - Token 级热力图与各向异性分析
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple

from embedding_lab.embedding_utils import (
    get_batch_embeddings,
    cosine_similarity,
    compute_anisotropy,
    whitening_transform,
    load_sentence_transformer
)


def tokenize_text(text: str) -> List[str]:
    """简单的中英文分词"""
    import re
    
    # 先按空格分割
    tokens = text.split()
    
    # 对每个部分进一步处理
    result = []
    for token in tokens:
        # 如果包含中文，逐字分割
        if re.search(r'[\u4e00-\u9fff]', token):
            for char in token:
                if char.strip():
                    result.append(char)
        else:
            if token.strip():
                result.append(token)
    
    return result if result else [text]


def get_token_embeddings(text: str, model_name: str) -> Tuple[List[str], np.ndarray]:
    """获取文本中每个 token 的 embedding"""
    tokens = tokenize_text(text)
    
    if not tokens:
        return [], np.array([])
    
    embeddings = get_batch_embeddings(tokens, model_name)
    
    if embeddings is None:
        return tokens, np.array([])
    
    return tokens, embeddings


def create_similarity_heatmap(
    tokens_a: List[str],
    tokens_b: List[str],
    similarity_matrix: np.ndarray,
    text_a: str,
    text_b: str
) -> go.Figure:
    """创建 Token-to-Token 相似度热力图"""
    
    fig = go.Figure(data=go.Heatmap(
        z=similarity_matrix,
        x=tokens_b,
        y=tokens_a,
        colorscale=[
            [0, '#F3F4F6'],      # 低相似度 - 浅灰
            [0.3, '#DBEAFE'],    # 中低 - 淡蓝
            [0.5, '#93C5FD'],    # 中等 - 蓝
            [0.7, '#3B82F6'],    # 中高 - 深蓝
            [1, '#1D4ED8']       # 高相似度 - 最深蓝
        ],
        hovertemplate='<b>%{y}</b> ↔ <b>%{x}</b><br>相似度: %{z:.4f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(text="相似度", font=dict(color='#111827')),
            tickfont=dict(color='#6B7280')
        )
    ))
    
    fig.update_layout(
        title=dict(
            text="Token-to-Token 相似度矩阵",
            font=dict(size=16, color='#111827')
        ),
        xaxis=dict(
            title=f"文本 B: \"{text_b[:30]}...\"" if len(text_b) > 30 else f"文本 B: \"{text_b}\"",
            tickangle=-45,
            tickfont=dict(size=11, family='JetBrains Mono, monospace'),
            side='bottom'
        ),
        yaxis=dict(
            title=f"文本 A: \"{text_a[:30]}...\"" if len(text_a) > 30 else f"文本 A: \"{text_a}\"",
            tickfont=dict(size=11, family='JetBrains Mono, monospace'),
            autorange='reversed'
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        height=max(400, len(tokens_a) * 25 + 150),
        margin=dict(l=100, r=40, t=60, b=100)
    )
    
    return fig


def create_anisotropy_visualization(
    similarities: List[float],
    mean_sim: float,
    std_sim: float,
    title: str = "词对相似度分布"
) -> go.Figure:
    """创建各向异性可视化（相似度分布直方图）"""
    
    fig = go.Figure()
    
    # 直方图 - 添加更好的悬浮提示
    fig.add_trace(go.Histogram(
        x=similarities,
        nbinsx=30,
        marker=dict(
            color='#3B82F6',
            line=dict(color='#1D4ED8', width=1)
        ),
        opacity=0.7,
        name='相似度分布',
        hovertemplate='<b>相似度区间</b>: %{x:.2f}<br><b>词对数量</b>: %{y} 对<extra></extra>'
    ))
    
    # 平均值线
    fig.add_vline(
        x=mean_sim,
        line=dict(color='#DC2626', width=2, dash='dash'),
        annotation_text=f"平均值: {mean_sim:.3f}",
        annotation_position="top right",
        annotation_font=dict(size=12, color='#DC2626')
    )
    
    # 理想值参考线 (0)
    fig.add_vline(
        x=0,
        line=dict(color='#059669', width=2, dash='dot'),
        annotation_text="理想值: 0",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color='#059669')
    )
    
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=14, color='#111827')
        ),
        xaxis=dict(
            title="余弦相似度（越接近 0 越好）",
            range=[-0.3, 1.1],
            gridcolor='#E5E7EB',
            dtick=0.2
        ),
        yaxis=dict(
            title="词对数量",
            gridcolor='#E5E7EB'
        ),
        plot_bgcolor='#FAFBFC',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif'),
        height=320,
        showlegend=False,
        margin=dict(t=50, b=50)
    )
    
    return fig


def render():
    """渲染 Lab 4: 语义相似度 页面"""
    st.markdown('<h1 class="module-title">语义相似度</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #FEE2E2 0%, #FEF3C7 100%); 
                border-radius: 8px; padding: 16px; margin-bottom: 24px; border: 1px solid #FECACA;">
        <p style="color: #991B1B; margin: 0; font-size: 14px;">
            <strong>🔬 微观分析</strong>：深入理解相似度计算的细节。<br/>
            Token 级热力图展示语义对齐，各向异性分析揭示向量空间的质量。
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tab 切换
    tab1, tab2 = st.tabs(["📊 相似度热力图", "📈 各向异性分析"])
    
    # ==================== Tab 1: 相似度热力图 ====================
    with tab1:
        st.markdown("### Token-to-Token 相似度矩阵")
        
        st.markdown("""
        <div style="background: #F3F4F6; border-radius: 6px; padding: 12px; margin-bottom: 16px;">
            <p style="color: #4B5563; margin: 0; font-size: 13px;">
                输入两段文本，查看它们在 Token 级别的语义对齐情况。<br/>
                颜色越深表示相似度越高，可用于分析同义词、语序变化等。
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            text_a = st.text_area(
                "文本 A",
                value="我看过这部电影",
                height=100,
                key="sim_text_a"
            )
        
        with col_b:
            text_b = st.text_area(
                "文本 B",
                value="这片子我看过",
                height=100,
                key="sim_text_b"
            )
        
        # 预设案例
        st.markdown("#### 预设案例")
        presets = [
            ("🔄 语序变化", "我看过这部电影", "这片子我看过"),
            ("📖 同义替换", "我非常喜欢这本书", "我特别爱这本书籍"),
            ("🌐 中英对照", "我爱你", "I love you"),
            ("❌ 无关文本", "今天天气很好", "量子力学很难"),
        ]
        
        def set_sim_preset(a: str, b: str):
            """回调函数：设置预设值"""
            st.session_state.sim_text_a = a
            st.session_state.sim_text_b = b
        
        preset_cols = st.columns(len(presets))
        for i, (label, a, b) in enumerate(presets):
            with preset_cols[i]:
                st.button(
                    label, 
                    key=f"preset_sim_{i}", 
                    width="stretch",
                    on_click=set_sim_preset,
                    args=(a, b)
                )
        
        # 模型选择
        model_name = st.selectbox(
            "Embedding 模型",
            options=[
                "paraphrase-multilingual-MiniLM-L12-v2",
                "all-MiniLM-L6-v2"
            ],
            format_func=lambda x: {
                "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM (推荐)",
                "all-MiniLM-L6-v2": "MiniLM-L6 (仅英文)"
            }[x],
            key="sim_model"
        )
        
        if text_a and text_b:
            with st.spinner("计算 Token Embeddings..."):
                tokens_a, emb_a = get_token_embeddings(text_a, model_name)
                tokens_b, emb_b = get_token_embeddings(text_b, model_name)
            
            if len(emb_a) == 0 or len(emb_b) == 0:
                st.error("Embedding 计算失败")
            else:
                # 计算相似度矩阵
                sim_matrix = np.zeros((len(tokens_a), len(tokens_b)))
                for i, ea in enumerate(emb_a):
                    for j, eb in enumerate(emb_b):
                        sim_matrix[i, j] = cosine_similarity(ea, eb)
                
                # 创建热力图
                fig = create_similarity_heatmap(tokens_a, tokens_b, sim_matrix, text_a, text_b)
                st.plotly_chart(fig, width="stretch")
                
                # 句子级相似度
                st.markdown("---")
                
                full_emb_a = get_batch_embeddings([text_a], model_name)
                full_emb_b = get_batch_embeddings([text_b], model_name)
                
                if full_emb_a is not None and full_emb_b is not None:
                    sentence_sim = cosine_similarity(full_emb_a[0], full_emb_b[0])
                    
                    col_metric, col_insight = st.columns([1, 2])
                    
                    with col_metric:
                        st.metric("句子级相似度", f"{sentence_sim:.4f}")
                    
                    with col_insight:
                        # 找出最相似的 token 对
                        max_idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
                        max_pair = (tokens_a[max_idx[0]], tokens_b[max_idx[1]])
                        max_sim = sim_matrix[max_idx]
                        
                        st.markdown(f"""
                        <div style="background: #D1FAE5; border-radius: 6px; padding: 12px;">
                            <p style="color: #065F46; margin: 0; font-size: 13px;">
                                <strong>最强语义对齐</strong><br/>
                                「{max_pair[0]}」↔「{max_pair[1]}」相似度: {max_sim:.4f}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 解释说明
                with st.expander("💡 如何解读热力图"):
                    st.markdown("""
                    **热力图分析要点**：
                    
                    1. **对角线高亮**：如果两段文本顺序一致，相似的词会在对角线上
                    2. **交叉高亮**：语序变化时，相似词会在非对角线位置高亮
                    3. **整行/整列高亮**：某个词与多个词都相似，可能是核心语义词
                    4. **全局偏暗**：两段文本语义不相关
                    
                    **面试考点**：
                    - 为什么同义词能够对齐？因为训练时相似上下文中的词获得相似向量
                    - 跨语言对齐如何实现？多语言模型在平行语料上训练，建立跨语言语义空间
                    """)
    
    # ==================== Tab 2: 各向异性分析 ====================
    with tab2:
        st.markdown("### 向量空间各向异性分析")
        
        st.markdown("""
**🎯 实验目的**：检测 Embedding 模型的向量空间质量

**📊 图表解读**：
- 直方图展示所有**不相关词对**的相似度分布
- **理想情况**：不相关的词对，相似度应该接近 **0**（绿色虚线）
- **各向异性问题**：如果分布偏右（平均值 > 0.3），说明模型存在质量问题
- **白化处理**：一种后处理方法，可以缓解各向异性问题
        """)
        
        st.markdown("---")
        
        # 选择分析方式
        analysis_mode = st.radio(
            "选择词汇来源",
            options=["使用预置词表", "使用自定义词汇"],
            horizontal=True,
            key="aniso_mode"
        )
        
        # 预置的不相关词对
        preset_words = [
            "苹果", "汽车", "音乐", "电脑", "咖啡", "书籍", "天空", "海洋",
            "山峰", "河流", "城市", "农村", "历史", "科学", "艺术", "体育",
            "政治", "经济", "医学", "法律", "教育", "文化", "宗教", "哲学",
            "数学", "物理", "化学", "生物", "地理", "心理", "社会", "语言"
        ]
        
        if analysis_mode == "使用预置词表":
            texts_to_analyze = preset_words
            
            # 展示预设词表
            with st.expander(f"📋 查看预置词表（{len(preset_words)} 个词）", expanded=False):
                # 分 4 列展示
                cols = st.columns(4)
                for i, word in enumerate(preset_words):
                    cols[i % 4].markdown(f"`{word}`")
            
            n_pairs = len(texts_to_analyze) * (len(texts_to_analyze) - 1) // 2
            st.info(f"将计算 {len(texts_to_analyze)} 个词汇之间的 **{n_pairs} 个词对**的相似度")
            
        else:
            st.markdown("""
💡 **使用说明**：输入一组**语义不相关**的词汇（每行一个）。
理想情况下，这些词两两之间的相似度应该接近 0。
            """)
            
            custom_words = st.text_area(
                "输入词汇（每行一个，建议 10-50 个不相关的词）",
                value="苹果\n汽车\n音乐\n医生\n河流\n数学\n咖啡\n政治\n艺术\n化学",
                height=150,
                key="aniso_custom"
            )
            texts_to_analyze = [w.strip() for w in custom_words.strip().split('\n') if w.strip()]
            
            if len(texts_to_analyze) >= 2:
                n_pairs = len(texts_to_analyze) * (len(texts_to_analyze) - 1) // 2
                st.caption(f"当前 {len(texts_to_analyze)} 个词，将产生 {n_pairs} 个词对")
        
        # 模型选择
        aniso_model = st.selectbox(
            "Embedding 模型",
            options=[
                "paraphrase-multilingual-MiniLM-L12-v2",
                "all-MiniLM-L6-v2"
            ],
            format_func=lambda x: {
                "paraphrase-multilingual-MiniLM-L12-v2": "Multilingual MiniLM（推荐，支持中文）",
                "all-MiniLM-L6-v2": "MiniLM-L6（仅英文）"
            }[x],
            key="aniso_model"
        )
        
        if st.button("🔬 分析各向异性", type="primary", width="stretch"):
            if len(texts_to_analyze) < 5:
                st.warning("至少需要 5 个词汇才能进行有意义的分析")
            else:
                with st.spinner("计算 Embeddings..."):
                    embeddings = get_batch_embeddings(texts_to_analyze, aniso_model)
                
                if embeddings is None:
                    st.error("Embedding 计算失败")
                else:
                    # 计算原始向量的各向异性
                    mean_sim, std_sim = compute_anisotropy(embeddings, sample_size=min(100, len(embeddings)))
                    
                    # 计算所有词对的相似度，同时记录词对信息
                    similarities = []
                    word_pairs = []
                    for i in range(len(embeddings)):
                        for j in range(i + 1, len(embeddings)):
                            sim = cosine_similarity(embeddings[i], embeddings[j])
                            similarities.append(sim)
                            word_pairs.append((texts_to_analyze[i], texts_to_analyze[j], sim))
                    
                    # 排序找出最相似和最不相似的词对
                    word_pairs_sorted = sorted(word_pairs, key=lambda x: x[2], reverse=True)
                    
                    # 显示词对示例
                    st.markdown("#### 📝 词对相似度示例")
                    col_high, col_low = st.columns(2)
                    
                    with col_high:
                        st.markdown("**相似度最高的 5 对**（可能有语义关联）")
                        for w1, w2, sim in word_pairs_sorted[:5]:
                            color = "#DC2626" if sim > 0.5 else "#D97706" if sim > 0.3 else "#059669"
                            st.markdown(f"- `{w1}` ↔ `{w2}`: <span style='color:{color}'><b>{sim:.3f}</b></span>", unsafe_allow_html=True)
                    
                    with col_low:
                        st.markdown("**相似度最低的 5 对**（符合预期）")
                        for w1, w2, sim in word_pairs_sorted[-5:]:
                            color = "#059669" if sim < 0.2 else "#D97706"
                            st.markdown(f"- `{w1}` ↔ `{w2}`: <span style='color:{color}'><b>{sim:.3f}</b></span>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("#### 📊 相似度分布对比")
                    
                    col_original, col_whitened = st.columns(2)
                    
                    with col_original:
                        st.markdown("**原始向量**")
                        
                        # 显示指标
                        metric_cols = st.columns(2)
                        metric_cols[0].metric("平均相似度", f"{mean_sim:.4f}")
                        metric_cols[1].metric("标准差", f"{std_sim:.4f}")
                        
                        # 判断是否存在各向异性问题
                        if mean_sim > 0.3:
                            st.error("⚠️ 存在明显的各向异性问题（平均值 > 0.3）")
                        elif mean_sim > 0.15:
                            st.warning("⚡ 存在轻微的各向异性（平均值 > 0.15）")
                        else:
                            st.success("✅ 各向异性程度较低，向量空间质量良好")
                        
                        # 绘制分布图
                        fig1 = create_anisotropy_visualization(
                            similarities, mean_sim, std_sim, 
                            title="原始向量 - 词对相似度分布"
                        )
                        st.plotly_chart(fig1, width="stretch")
                    
                    # 白化处理
                    with col_whitened:
                        st.markdown("**白化处理后**")
                        
                        with st.spinner("执行白化变换..."):
                            whitened = whitening_transform(embeddings)
                        
                        # 计算白化后的各向异性
                        mean_sim_w, std_sim_w = compute_anisotropy(whitened, sample_size=min(100, len(whitened)))
                        
                        # 计算白化后的相似度
                        similarities_w = []
                        for i in range(len(whitened)):
                            for j in range(i + 1, len(whitened)):
                                sim = cosine_similarity(whitened[i], whitened[j])
                                similarities_w.append(sim)
                        
                        metric_cols_w = st.columns(2)
                        delta_mean = mean_sim_w - mean_sim
                        delta_std = std_sim_w - std_sim
                        metric_cols_w[0].metric("平均相似度", f"{mean_sim_w:.4f}", delta=f"{delta_mean:+.4f}")
                        metric_cols_w[1].metric("标准差", f"{std_sim_w:.4f}", delta=f"{delta_std:+.4f}")
                        
                        if mean_sim_w < mean_sim * 0.8:
                            st.success(f"✅ 白化有效！平均相似度降低了 {(1 - mean_sim_w/mean_sim)*100:.1f}%")
                        elif mean_sim_w < mean_sim:
                            st.info("📉 白化有一定效果")
                        else:
                            st.warning("⚠️ 白化效果不明显")
                        
                        fig2 = create_anisotropy_visualization(
                            similarities_w, mean_sim_w, std_sim_w,
                            title="白化后 - 词对相似度分布"
                        )
                        st.plotly_chart(fig2, width="stretch")
                    
                    # 解释说明
                    st.markdown("---")
                    with st.expander("📚 深入理解各向异性"):
                        st.markdown("""
                        **什么是各向异性 (Anisotropy)?**
                        
                        在理想的向量空间中，随机选取的两个不相关词汇，它们的余弦相似度应该接近 0。
                        但研究发现，很多预训练语言模型的词向量存在「各向异性」问题：
                        - 向量倾向于占据高维空间的一个狭窄圆锥区域
                        - 导致即使语义不相关的词，相似度也普遍偏高
                        
                        **为什么这是问题?**
                        - 降低了相似度分数的区分度
                        - 影响检索、聚类等下游任务的效果
                        
                        **解决方案**:
                        1. **白化 (Whitening)**: 对向量进行线性变换，使协方差矩阵变为单位矩阵
                        2. **对比学习**: 使用 SimCSE 等方法微调模型
                        3. **后处理**: 减去平均向量、主成分移除等
                        
                        **面试考点**:
                        - 各向异性的成因：训练目标（如 MLM）导致向量聚集
                        - 白化的数学原理：协方差矩阵特征值分解
                        - 权衡：白化可能丢失部分语义信息
                        """)

