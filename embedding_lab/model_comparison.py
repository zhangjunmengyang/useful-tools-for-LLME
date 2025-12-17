"""
模型对比 - 对比不同 Embedding 模型的特性
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from embedding_lab.embedding_utils import (
    get_batch_embeddings,
    compute_sparse_embeddings,
    cosine_similarity
)


AVAILABLE_MODELS = {
    "TF-IDF": "tfidf",
    "BM25": "bm25",
    "Multilingual MiniLM": "paraphrase-multilingual-MiniLM-L12-v2",
    "MiniLM-L6": "all-MiniLM-L6-v2",
}


def compute_similarity_scores(query_embedding, candidate_embeddings):
    """计算 query 与所有 candidates 的相似度"""
    scores = []
    for emb in candidate_embeddings:
        scores.append(cosine_similarity(query_embedding, emb))
    return scores


def create_comparison_chart(candidates, scores_dict, query):
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
        height=480,
        autosize=True,
        xaxis=dict(
            tickangle=-20,
            gridcolor='#E5E7EB'
        ),
        yaxis=dict(
            title='Similarity',
            range=[0, 1],
            gridcolor='#E5E7EB'
        )
    )
    
    return fig


def compare_models(query, candidates_text, use_tfidf, use_bm25, use_multilingual, use_minilm):
    """对比模型"""
    if not query:
        return (
            "Please enter query text",
            None,
            pd.DataFrame()
        )

    candidates = [c.strip() for c in candidates_text.strip().split('\n') if c.strip()]
    if not candidates:
        return (
            "Please enter candidate texts",
            None,
            pd.DataFrame()
        )
    
    selected_models = []
    if use_tfidf:
        selected_models.append(("TF-IDF", "tfidf"))
    if use_bm25:
        selected_models.append(("BM25", "bm25"))
    if use_multilingual:
        selected_models.append(("Multilingual MiniLM", "paraphrase-multilingual-MiniLM-L12-v2"))
    if use_minilm:
        selected_models.append(("MiniLM-L6 (English)", "all-MiniLM-L6-v2"))

    if not selected_models:
        return (
            "Please select at least one model",
            None,
            pd.DataFrame()
        )
    
    all_texts = [query] + candidates
    scores_dict = {}
    
    for model_name, model_id in selected_models:
        try:
            if model_id in ["tfidf", "bm25"]:
                embeddings = compute_sparse_embeddings(all_texts, model_id)
                query_emb = embeddings[0]
                candidate_embs = embeddings[1:]
            else:
                embeddings = get_batch_embeddings(all_texts, model_id)
                if embeddings is None:
                    continue
                query_emb = embeddings[0]
                candidate_embs = embeddings[1:]
            
            scores = compute_similarity_scores(query_emb, candidate_embs)
            scores_dict[model_name] = scores
            
        except Exception as e:
            print(f"Model {model_name} computation failed: {e}")

    if not scores_dict:
        return (
            "All model computations failed",
            None,
            pd.DataFrame()
        )
    
    # 创建图表
    fig = create_comparison_chart(candidates, scores_dict, query)
    
    # Detailed score table
    data = {'Candidate Text': candidates}
    for model_name, scores in scores_dict.items():
        data[model_name] = [f'{s:.4f}' for s in scores]
    df = pd.DataFrame(data)
    
    # Ranking analysis
    ranking_md = "### Ranking Comparison\n\n"
    for model_name, scores in scores_dict.items():
        sorted_indices = np.argsort(scores)[::-1]
        ranking_md += f"**{model_name}**:\n"
        for rank, idx in enumerate(sorted_indices, 1):
            text = candidates[idx][:25] + ('...' if len(candidates[idx]) > 25 else '')
            ranking_md += f"{rank}. {text} - {scores[idx]:.3f}\n"
        ranking_md += "\n"
    
    # 洞察分析
    insight = ""
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
            insight = f"""
### Insight

**Largest Ranking Difference**: Text \"{most_diff[0][:30]}...\" ranks **{most_diff[2]}** in **{model_names[0]}** but ranks **{most_diff[3]}** in **{model_names[1]}**.

_This indicates significant differences in how models understand semantics._
"""
    
    return ranking_md + insight, fig, df


def render():
    """渲染页面"""

    # 模型选择
    with gr.Row():
        use_tfidf = gr.Checkbox(label="TF-IDF", value=True)
        use_bm25 = gr.Checkbox(label="BM25", value=True)
        use_multilingual = gr.Checkbox(label="Multilingual MiniLM", value=False)
        use_minilm = gr.Checkbox(label="MiniLM-L6 (English)", value=False)
    
    # Input area
    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Textbox(
                label="Query Text",
                placeholder="Enter query text...",
                lines=1
            )

        with gr.Column(scale=2):
            candidates = gr.Textbox(
                label="Candidate Texts (one per line)",
                placeholder="fruit\nphone\ncomputer",
                lines=6
            )
    
    # Preset examples
    with gr.Row():
        preset1 = gr.Button("Apple Ambiguity", size="sm")
        preset2 = gr.Button("Bank Ambiguity", size="sm")
        preset3 = gr.Button("Tesla", size="sm")
        preset4 = gr.Button("Semantic Search", size="sm")

    # Results
    chart = gr.Plot(label="Similarity Comparison")

    with gr.Accordion("Detailed Scores", open=True):
        score_df = gr.Dataframe(interactive=False)
    
    analysis_md = gr.Markdown("")
    
    # ==================== 事件绑定 ====================
    
    inputs = [query, candidates, use_tfidf, use_bm25, use_multilingual, use_minilm]
    outputs = [analysis_md, chart, score_df]
    
    # 预设按钮
    def set_preset1():
        return "苹果", "水果\n手机\n乔布斯\n红色的球\n苹果发布新产品\n我喜欢吃苹果"
    
    def set_preset2():
        return "银行", "金融机构\n河边\n存款取款\n银行卡\n河岸风景"
    
    def set_preset3():
        return "特斯拉", "电动汽车\n科学家\n马斯克\n电磁感应\nModel 3"
    
    def set_preset4():
        return "如何学习编程", "编程入门教程\n学习Python\n代码怎么写\n程序员成长\n软件开发"
    
    # 预设按钮 - 设置并自动计算
    def set_and_compute(q, c, t1, t2, t3, t4):
        result = compare_models(q, c, t1, t2, t3, t4)
        return (q, c) + result[:3]
    
    preset1.click(
        fn=lambda t1, t2, t3, t4: set_and_compute("苹果", "水果\n手机\n乔布斯\n红色的球\n苹果发布新产品\n我喜欢吃苹果", t1, t2, t3, t4),
        inputs=[use_tfidf, use_bm25, use_multilingual, use_minilm],
        outputs=[query, candidates, analysis_md, chart, score_df]
    )
    preset2.click(
        fn=lambda t1, t2, t3, t4: set_and_compute("银行", "金融机构\n河边\n存款取款\n银行卡\n河岸风景", t1, t2, t3, t4),
        inputs=[use_tfidf, use_bm25, use_multilingual, use_minilm],
        outputs=[query, candidates, analysis_md, chart, score_df]
    )
    preset3.click(
        fn=lambda t1, t2, t3, t4: set_and_compute("特斯拉", "电动汽车\n科学家\n马斯克\n电磁感应\nModel 3", t1, t2, t3, t4),
        inputs=[use_tfidf, use_bm25, use_multilingual, use_minilm],
        outputs=[query, candidates, analysis_md, chart, score_df]
    )
    preset4.click(
        fn=lambda t1, t2, t3, t4: set_and_compute("如何学习编程", "编程入门教程\n学习Python\n代码怎么写\n程序员成长\n软件开发", t1, t2, t3, t4),
        inputs=[use_tfidf, use_bm25, use_multilingual, use_minilm],
        outputs=[query, candidates, analysis_md, chart, score_df]
    )
    
    # 自动计算 - 参数变化时触发
    for component in [query, candidates, use_tfidf, use_bm25, use_multilingual, use_minilm]:
        component.change(
            fn=compare_models,
            inputs=inputs,
            outputs=outputs
        )
    
    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        return compare_models("苹果", "水果\n手机\n乔布斯\n红色的球\n苹果公司发布新产品\n我喜欢吃苹果", True, True, False, False)
    
    # 返回 load 事件信息
    return {
        'load_fn': on_load,
        'load_outputs': outputs
    }
