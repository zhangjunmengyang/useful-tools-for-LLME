"""
模型对比 - Gradio 版本
多模型分词效率对比
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from token_lab.tokenizer_utils import (
    load_tokenizer,
    get_token_info,
    calculate_compression_stats,
    get_models_by_category,
    MODEL_CATEGORIES,
    TOKEN_COLORS
)


def get_category_choices():
    """获取模型厂商列表"""
    return list(MODEL_CATEGORIES.keys())


def get_model_choices(category):
    """根据厂商获取模型列表"""
    if not category:
        return []
    models = get_models_by_category(category)
    return [name for name, _ in models]


def get_model_id(category, model_name):
    """获取模型 ID"""
    if not category or not model_name:
        return None
    models = get_models_by_category(category)
    for name, model_id in models:
        if name == model_name:
            return model_id
    return None


def render_tokens_html(tokens, model_name, color):
    """渲染分词结果"""
    html = [f'<div style="line-height: 2.4; padding: 12px; background: #F3F4F6; '
            f'border-radius: 8px; border-left: 4px solid {color};">']
    html.append(f'<div style="margin-bottom: 8px; font-weight: 600; color: #374151;">{model_name} ({len(tokens)} tokens)</div>')
    
    for idx, info in enumerate(tokens):
        bg_color = TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        display = info['token_str'].replace(' ', '␣').replace('\n', '↵')
        if not display.strip():
            display = repr(info['token_str'])[1:-1] or '[EMPTY]'
        display = display.replace('<', '&lt;').replace('>', '&gt;')
        
        html.append(
            f'<span style="background:{bg_color}; color:#111827; padding:3px 6px; '
            f'margin:2px; border-radius:4px; display:inline-block; '
            f'font-family: \'JetBrains Mono\', monospace; font-size:13px;">'
            f'{display}</span>'
        )
    
    html.append('</div>')
    return ''.join(html)


def create_comparison_chart(stats_a, stats_b, model_a, model_b):
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
        height=350
    )
    
    fig.update_xaxes(gridcolor='#E5E7EB', linecolor='#E5E7EB')
    fig.update_yaxes(gridcolor='#E5E7EB', linecolor='#E5E7EB')
    
    return fig


def compare_models(cat_a, model_a, cat_b, model_b, text):
    """对比两个模型"""
    if not all([cat_a, model_a, cat_b, model_b]):
        return (
            "请选择两个模型",
            "<div>请选择模型 A</div>",
            "<div>请选择模型 B</div>",
            None,
            pd.DataFrame()
        )
    
    if not text:
        return (
            "请输入测试文本",
            "<div>请输入文本</div>",
            "<div>请输入文本</div>",
            None,
            pd.DataFrame()
        )
    
    model_id_a = get_model_id(cat_a, model_a)
    model_id_b = get_model_id(cat_b, model_b)
    
    tokenizer_a = load_tokenizer(model_id_a)
    tokenizer_b = load_tokenizer(model_id_b)
    
    if not tokenizer_a or not tokenizer_b:
        return (
            "模型加载失败",
            "<div>模型加载失败</div>",
            "<div>模型加载失败</div>",
            None,
            pd.DataFrame()
        )
    
    tokens_a = get_token_info(tokenizer_a, text)
    tokens_b = get_token_info(tokenizer_b, text)
    
    stats_a = calculate_compression_stats(text, len(tokens_a))
    stats_b = calculate_compression_stats(text, len(tokens_b))
    
    # 效率对比摘要
    diff = stats_b['token_count'] - stats_a['token_count']
    if stats_b['token_count'] > 0:
        eff = (stats_b['token_count'] - stats_a['token_count']) / stats_b['token_count'] * 100
    else:
        eff = 0
    
    summary = f"""
### 效率指标

| 指标 | {model_a} | {model_b} | 差异 |
|------|----------|----------|------|
| Token 数 | **{stats_a['token_count']}** | **{stats_b['token_count']}** | {abs(diff)} ({'A更少' if diff > 0 else 'B更少' if diff < 0 else '相同'}) |
| 字符数 | {stats_a['char_count']} | {stats_b['char_count']} | - |
| 字符/Token | {stats_a['chars_per_token']} | {stats_b['chars_per_token']} | - |
| 字节/Token | {stats_a['bytes_per_token']} | {stats_b['bytes_per_token']} | - |
| 效率差 | - | - | {abs(eff):.1f}% ({'A更优' if eff > 0 else 'B更优' if eff < 0 else '相同'}) |
"""
    
    # 分词结果
    html_a = render_tokens_html(tokens_a, model_a, '#2563EB')
    html_b = render_tokens_html(tokens_b, model_b, '#059669')
    
    # 图表
    fig = create_comparison_chart(stats_a, stats_b, model_a, model_b)
    
    # 详细数据
    df = pd.DataFrame({
        "指标": ["Token 数", "字符数", "字节数", "字符/Token", "字节/Token"],
        model_a: [stats_a['token_count'], stats_a['char_count'], stats_a['byte_count'], 
                  stats_a['chars_per_token'], stats_a['bytes_per_token']],
        model_b: [stats_b['token_count'], stats_b['char_count'], stats_b['byte_count'],
                  stats_b['chars_per_token'], stats_b['bytes_per_token']]
    })
    
    return summary, html_a, html_b, fig, df


def render():
    """渲染页面"""
    
    gr.Markdown("## 模型对比")
    
    # 模型选择
    categories = get_category_choices()
    cat_a_init = categories[0] if categories else None
    cat_b_init = categories[2] if len(categories) > 2 else categories[0] if categories else None
    models_a_init = get_model_choices(cat_a_init) if cat_a_init else []
    models_b_init = get_model_choices(cat_b_init) if cat_b_init else []
    
    with gr.Row():
        # 模型 A
        with gr.Column():
            gr.Markdown("### 模型 A")
            with gr.Row():
                cat_a = gr.Dropdown(
                    choices=categories,
                    label="厂商",
                    value=cat_a_init,
                    scale=1
                )
                model_a = gr.Dropdown(
                    choices=models_a_init,
                    label="模型",
                    value=models_a_init[0] if models_a_init else None,
                    scale=2
                )
        
        # 模型 B
        with gr.Column():
            gr.Markdown("### 模型 B")
            with gr.Row():
                cat_b = gr.Dropdown(
                    choices=categories,
                    label="厂商",
                    value=cat_b_init,
                    scale=1
                )
                model_b = gr.Dropdown(
                    choices=models_b_init,
                    label="模型",
                    value=models_b_init[0] if models_b_init else None,
                    scale=2
                )
    
    gr.Markdown("---")
    
    # 测试文本
    test_input = gr.Textbox(
        label="测试文本",
        placeholder="输入文本查看对比结果...",
        lines=3
    )
    
    compare_btn = gr.Button("开始对比", variant="primary", size="lg")
    
    # 结果区域
    summary_md = gr.Markdown("")
    
    with gr.Row():
        tokens_a_html = gr.HTML("")
        tokens_b_html = gr.HTML("")
    
    chart = gr.Plot(label="效率对比图")
    
    with gr.Accordion("详细数据", open=False):
        detail_df = gr.Dataframe(interactive=False)
    
    # ==================== 事件绑定 ====================
    
    def update_models_a(cat):
        models = get_model_choices(cat)
        return gr.update(choices=models, value=models[0] if models else None)
    
    def update_models_b(cat):
        models = get_model_choices(cat)
        return gr.update(choices=models, value=models[0] if models else None)
    
    cat_a.change(fn=update_models_a, inputs=[cat_a], outputs=[model_a])
    cat_b.change(fn=update_models_b, inputs=[cat_b], outputs=[model_b])
    
    compare_btn.click(
        fn=compare_models,
        inputs=[cat_a, model_a, cat_b, model_b, test_input],
        outputs=[summary_md, tokens_a_html, tokens_b_html, chart, detail_df]
    )
    
    # 文本变化也触发对比
    test_input.change(
        fn=compare_models,
        inputs=[cat_a, model_a, cat_b, model_b, test_input],
        outputs=[summary_md, tokens_a_html, tokens_b_html, chart, detail_df]
    )

