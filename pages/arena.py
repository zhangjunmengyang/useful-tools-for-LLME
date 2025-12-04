"""
模型对比 - 多模型分词效率对比
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.tokenizer_utils import (
    load_tokenizer,
    get_token_info,
    calculate_compression_stats,
    get_models_by_category,
    MODEL_CATEGORIES,
    TOKEN_COLORS
)


def render_comparison_tokens(tokens_a: list, tokens_b: list, model_a: str, model_b: str):
    """并排渲染分词结果"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**{model_a}** ({len(tokens_a)} tokens)")
        html = ['<div style="line-height: 2.2; padding: 12px; background: #F3F4F6; border-radius: 6px; border-left: 3px solid #2563EB;">']
        for idx, info in enumerate(tokens_a):
            color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
            display = info['token_str'].replace(' ', '\u2423').replace('\n', '\u21b5')
            if not display.strip():
                display = repr(info['token_str'])[1:-1] or '[EMPTY]'
            display = display.replace('<', '&lt;').replace('>', '&gt;')
            html.append(
                f'<span style="background:{color}; color:#111827; padding:3px 6px; '
                f'margin:2px; border-radius:4px; display:inline-block; font-family:monospace; font-size:13px;">'
                f'{display}</span>'
            )
        html.append('</div>')
        st.markdown(''.join(html), unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**{model_b}** ({len(tokens_b)} tokens)")
        html = ['<div style="line-height: 2.2; padding: 12px; background: #F3F4F6; border-radius: 6px; border-left: 3px solid #059669;">']
        for idx, info in enumerate(tokens_b):
            color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
            display = info['token_str'].replace(' ', '\u2423').replace('\n', '\u21b5')
            if not display.strip():
                display = repr(info['token_str'])[1:-1] or '[EMPTY]'
            display = display.replace('<', '&lt;').replace('>', '&gt;')
            html.append(
                f'<span style="background:{color}; color:#111827; padding:3px 6px; '
                f'margin:2px; border-radius:4px; display:inline-block; font-family:monospace; font-size:13px;">'
                f'{display}</span>'
            )
        html.append('</div>')
        st.markdown(''.join(html), unsafe_allow_html=True)


def create_comparison_chart(stats_a: dict, stats_b: dict, model_a: str, model_b: str):
    """创建对比图表"""
    metrics = ['Token 数', '字符/Token', '字节/Token']
    values_a = [stats_a['token_count'], stats_a['chars_per_token'], stats_a['bytes_per_token']]
    values_b = [stats_b['token_count'], stats_b['chars_per_token'], stats_b['bytes_per_token']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=model_a,
        x=metrics,
        y=values_a,
        marker_color='#2563EB',
        text=[f'{v:.2f}' if isinstance(v, float) else str(v) for v in values_a],
        textposition='outside',
        textfont=dict(color='#111827', size=13)
    ))
    
    fig.add_trace(go.Bar(
        name=model_b,
        x=metrics,
        y=values_b,
        marker_color='#059669',
        text=[f'{v:.2f}' if isinstance(v, float) else str(v) for v in values_b],
        textposition='outside',
        textfont=dict(color='#111827', size=13)
    ))
    
    fig.update_layout(
        barmode='group',
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        font=dict(color='#111827', family='Inter, sans-serif', size=14),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
        height=280
    )
    
    fig.update_xaxes(gridcolor='#E5E7EB', linecolor='#E5E7EB')
    fig.update_yaxes(gridcolor='#E5E7EB', linecolor='#E5E7EB')
    
    return fig


def render_model_selector(label: str, key_prefix: str, default_category_idx: int = 0, default_model_idx: int = 0):
    """渲染两级联动模型选择器"""
    categories = list(MODEL_CATEGORIES.keys())
    category_display = [f"{MODEL_CATEGORIES[cat]['icon']} {cat}" for cat in categories]
    
    col_provider, col_model = st.columns([1, 1])
    
    with col_provider:
        selected_category_display = st.selectbox(
            "厂商",
            options=category_display,
            index=min(default_category_idx, len(categories) - 1),
            key=f"{key_prefix}_provider",
            label_visibility="collapsed"
        )
        selected_category = categories[category_display.index(selected_category_display)]
    
    with col_model:
        models = get_models_by_category(selected_category)
        model_display = [name for name, _ in models]
        model_ids = [model_id for _, model_id in models]
        
        selected_model_display = st.selectbox(
            "模型",
            options=model_display,
            index=min(default_model_idx, len(model_display) - 1) if model_display else 0,
            key=f"{key_prefix}_model",
            label_visibility="collapsed"
        )
        
        if model_display:
            model_idx = model_display.index(selected_model_display)
            model_id = model_ids[model_idx]
        else:
            model_id = None
    
    return model_id


def render():
    """渲染页面"""
    st.markdown('<h1 class="module-title">模型对比</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 模型 A")
        model_a = render_model_selector("A", "arena_a", default_category_idx=0, default_model_idx=0)
    
    with col2:
        st.markdown("##### 模型 B")
        model_b = render_model_selector("B", "arena_b", default_category_idx=2, default_model_idx=0)
    
    st.markdown("---")
    
    input_text = st.text_area(
        "测试文本",
        value="",
        height=100,
        placeholder="输入文本查看对比结果",
        key="arena_input"
    )
    
    if input_text:
        tokenizer_a = load_tokenizer(model_a)
        tokenizer_b = load_tokenizer(model_b)
        
        if not tokenizer_a or not tokenizer_b:
            st.error("模型加载失败")
            return
        
        tokens_a = get_token_info(tokenizer_a, input_text)
        tokens_b = get_token_info(tokenizer_b, input_text)
        
        stats_a = calculate_compression_stats(input_text, len(tokens_a))
        stats_b = calculate_compression_stats(input_text, len(tokens_b))
        
        st.markdown("---")
        
        st.markdown("### 效率指标")
        
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            diff = stats_b['token_count'] - stats_a['token_count']
            st.metric("Token 差异", f"{abs(diff)}", 
                     delta=f"A {'更少' if diff > 0 else '更多'}" if diff != 0 else "相同",
                     delta_color="normal" if diff > 0 else "inverse")
        
        with metric_cols[1]:
            st.metric(model_a.split('/')[-1], f"{stats_a['token_count']} tokens")
        
        with metric_cols[2]:
            st.metric(model_b.split('/')[-1], f"{stats_b['token_count']} tokens")
        
        with metric_cols[3]:
            if stats_b['token_count'] > 0:
                eff = (stats_b['token_count'] - stats_a['token_count']) / stats_b['token_count'] * 100
                st.metric("效率差", f"{abs(eff):.1f}%", 
                         delta=f"A {'更优' if eff > 0 else '更差'}" if eff != 0 else "相同")
        
        fig = create_comparison_chart(stats_a, stats_b, model_a.split('/')[-1], model_b.split('/')[-1])
        st.plotly_chart(fig, width="stretch")
        
        st.markdown("### 分词结果")
        render_comparison_tokens(tokens_a, tokens_b, model_a.split('/')[-1], model_b.split('/')[-1])
        
        with st.expander("详细数据"):
            df = pd.DataFrame({
                "指标": ["Token 数", "字符数", "字节数", "字符/Token", "字节/Token"],
                model_a.split('/')[-1]: [stats_a['token_count'], stats_a['char_count'], stats_a['byte_count'], stats_a['chars_per_token'], stats_a['bytes_per_token']],
                model_b.split('/')[-1]: [stats_b['token_count'], stats_b['char_count'], stats_b['byte_count'], stats_b['chars_per_token'], stats_b['bytes_per_token']]
            })
            st.dataframe(df, width="stretch", hide_index=True)
