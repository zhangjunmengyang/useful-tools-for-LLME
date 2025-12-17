"""
Model Arena - Gradio Version
Multi-model tokenizer efficiency comparison
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
    html = [f'<div style="line-height: 2.4; padding: 16px; background: #FAFAFA; '
            f'border-radius: 8px; border: 1px solid #E5E7EB;">']
    html.append(f'<div style="margin-bottom: 12px; font-weight: 600; color: {color}; font-size: 14px;">{model_name} ({len(tokens)} tokens)</div>')

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
    metrics = ['Token Count', 'Chars/Token', 'Bytes/Token']
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
        height=400,
        autosize=True
    )

    fig.update_xaxes(gridcolor='#E5E7EB', linecolor='#E5E7EB')
    fig.update_yaxes(gridcolor='#E5E7EB', linecolor='#E5E7EB')

    return fig


def compare_models(cat_a, model_a, cat_b, model_b, text):
    """对比两个模型"""
    if not all([cat_a, model_a, cat_b, model_b]):
        return (
            "Please select two models",
            "<div>Please select Model A</div>",
            "<div>Please select Model B</div>",
            None,
            pd.DataFrame()
        )

    if not text:
        return (
            "Please enter test text",
            "<div>Please enter text</div>",
            "<div>Please enter text</div>",
            None,
            pd.DataFrame()
        )

    model_id_a = get_model_id(cat_a, model_a)
    model_id_b = get_model_id(cat_b, model_b)

    tokenizer_a = load_tokenizer(model_id_a)
    tokenizer_b = load_tokenizer(model_id_b)

    if not tokenizer_a or not tokenizer_b:
        return (
            "Failed to load models",
            "<div>Failed to load model</div>",
            "<div>Failed to load model</div>",
            None,
            pd.DataFrame()
        )

    tokens_a = get_token_info(tokenizer_a, text)
    tokens_b = get_token_info(tokenizer_b, text)

    stats_a = calculate_compression_stats(text, len(tokens_a))
    stats_b = calculate_compression_stats(text, len(tokens_b))

    diff = stats_b['token_count'] - stats_a['token_count']
    if stats_b['token_count'] > 0:
        eff = (stats_b['token_count'] - stats_a['token_count']) / stats_b['token_count'] * 100
    else:
        eff = 0

    winner = 'A is better' if diff > 0 else 'B is better' if diff < 0 else 'Same'
    winner_color = '#2563EB' if diff > 0 else '#059669' if diff < 0 else '#6B7280'

    summary = f"""
<div style="background: #F9FAFB; border-radius: 8px; padding: 16px; margin-bottom: 16px;">
<div style="font-weight: 600; font-size: 16px; margin-bottom: 12px; color: #111827;">Efficiency Metrics</div>
<table style="width: 100%; border-collapse: collapse; font-size: 14px;">
<thead>
<tr style="border-bottom: 2px solid #E5E7EB;">
<th style="text-align: left; padding: 8px 12px; color: #6B7280; font-weight: 500;">Metric</th>
<th style="text-align: center; padding: 8px 12px; color: #2563EB; font-weight: 600;">{model_a}</th>
<th style="text-align: center; padding: 8px 12px; color: #059669; font-weight: 600;">{model_b}</th>
<th style="text-align: center; padding: 8px 12px; color: #6B7280; font-weight: 500;">Difference</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom: 1px solid #E5E7EB;">
<td style="padding: 8px 12px;">Token Count</td>
<td style="text-align: center; padding: 8px 12px; font-weight: 600;">{stats_a['token_count']}</td>
<td style="text-align: center; padding: 8px 12px; font-weight: 600;">{stats_b['token_count']}</td>
<td style="text-align: center; padding: 8px 12px; color: {winner_color};">{abs(diff)} ({winner})</td>
</tr>
<tr style="border-bottom: 1px solid #E5E7EB;">
<td style="padding: 8px 12px;">Characters</td>
<td style="text-align: center; padding: 8px 12px;">{stats_a['char_count']}</td>
<td style="text-align: center; padding: 8px 12px;">{stats_b['char_count']}</td>
<td style="text-align: center; padding: 8px 12px; color: #9CA3AF;">-</td>
</tr>
<tr style="border-bottom: 1px solid #E5E7EB;">
<td style="padding: 8px 12px;">Chars/Token</td>
<td style="text-align: center; padding: 8px 12px;">{stats_a['chars_per_token']}</td>
<td style="text-align: center; padding: 8px 12px;">{stats_b['chars_per_token']}</td>
<td style="text-align: center; padding: 8px 12px; color: #9CA3AF;">-</td>
</tr>
<tr style="border-bottom: 1px solid #E5E7EB;">
<td style="padding: 8px 12px;">Bytes/Token</td>
<td style="text-align: center; padding: 8px 12px;">{stats_a['bytes_per_token']}</td>
<td style="text-align: center; padding: 8px 12px;">{stats_b['bytes_per_token']}</td>
<td style="text-align: center; padding: 8px 12px; color: #9CA3AF;">-</td>
</tr>
<tr>
<td style="padding: 8px 12px;">Efficiency Gap</td>
<td style="text-align: center; padding: 8px 12px; color: #9CA3AF;">-</td>
<td style="text-align: center; padding: 8px 12px; color: #9CA3AF;">-</td>
<td style="text-align: center; padding: 8px 12px; color: {winner_color}; font-weight: 600;">{abs(eff):.1f}% ({winner})</td>
</tr>
</tbody>
</table>
</div>
"""

    html_a = render_tokens_html(tokens_a, model_a, '#2563EB')
    html_b = render_tokens_html(tokens_b, model_b, '#059669')

    fig = create_comparison_chart(stats_a, stats_b, model_a, model_b)

    df = pd.DataFrame({
        "Metric": ["Token Count", "Characters", "Bytes", "Chars/Token", "Bytes/Token"],
        model_a: [stats_a['token_count'], stats_a['char_count'], stats_a['byte_count'],
                  stats_a['chars_per_token'], stats_a['bytes_per_token']],
        model_b: [stats_b['token_count'], stats_b['char_count'], stats_b['byte_count'],
                  stats_b['chars_per_token'], stats_b['bytes_per_token']]
    })

    return summary, html_a, html_b, fig, df


def render():
    """渲染页面"""

    categories = get_category_choices()
    cat_a_init = categories[0] if categories else None
    cat_b_init = categories[2] if len(categories) > 2 else categories[0] if categories else None
    models_a_init = get_model_choices(cat_a_init) if cat_a_init else []
    models_b_init = get_model_choices(cat_b_init) if cat_b_init else []

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Model A")
            with gr.Row():
                cat_a = gr.Dropdown(
                    choices=categories,
                    label="Vendor",
                    value=cat_a_init,
                    scale=1
                )
                model_a = gr.Dropdown(
                    choices=models_a_init,
                    label="Model",
                    value=models_a_init[0] if models_a_init else None,
                    scale=2
                )

        with gr.Column():
            gr.Markdown("### Model B")
            with gr.Row():
                cat_b = gr.Dropdown(
                    choices=categories,
                    label="Vendor",
                    value=cat_b_init,
                    scale=1
                )
                model_b = gr.Dropdown(
                    choices=models_b_init,
                    label="Model",
                    value=models_b_init[0] if models_b_init else None,
                    scale=2
                )

    gr.Markdown("---")

    default_text = "Hello, world! 你好，世界！This is a test for tokenizer comparison."

    test_input = gr.Textbox(
        label="Test Text",
        placeholder="Enter text to compare tokenization results...",
        value=default_text,
        lines=3
    )

    summary_md = gr.HTML("")

    with gr.Row():
        tokens_a_html = gr.HTML("")
        tokens_b_html = gr.HTML("")

    chart = gr.Plot(label="Efficiency Comparison")

    with gr.Accordion("Detailed Data", open=False):
        detail_df = gr.Dataframe(interactive=False)

    # 事件绑定
    def update_models_and_compare_a(cat, cat_b_val, model_b_val, text):
        """更新模型 A 列表并触发比较"""
        models = get_model_choices(cat)
        new_model = models[0] if models else None
        if new_model and model_b_val and text:
            result = compare_models(cat, new_model, cat_b_val, model_b_val, text)
            return (gr.update(choices=models, value=new_model),) + result
        return (gr.update(choices=models, value=new_model), "", "", "", None, pd.DataFrame())

    def update_models_and_compare_b(cat, cat_a_val, model_a_val, text):
        """更新模型 B 列表并触发比较"""
        models = get_model_choices(cat)
        new_model = models[0] if models else None
        if new_model and model_a_val and text:
            result = compare_models(cat_a_val, model_a_val, cat, new_model, text)
            return (gr.update(choices=models, value=new_model),) + result
        return (gr.update(choices=models, value=new_model), "", "", "", None, pd.DataFrame())

    cat_a.change(
        fn=update_models_and_compare_a,
        inputs=[cat_a, cat_b, model_b, test_input],
        outputs=[model_a, summary_md, tokens_a_html, tokens_b_html, chart, detail_df]
    )
    cat_b.change(
        fn=update_models_and_compare_b,
        inputs=[cat_b, cat_a, model_a, test_input],
        outputs=[model_b, summary_md, tokens_a_html, tokens_b_html, chart, detail_df]
    )

    for component in [model_a, model_b, test_input]:
        component.change(
            fn=compare_models,
            inputs=[cat_a, model_a, cat_b, model_b, test_input],
            outputs=[summary_md, tokens_a_html, tokens_b_html, chart, detail_df]
        )

    def on_load():
        """页面加载时计算默认值"""
        return compare_models(cat_a_init, models_a_init[0] if models_a_init else None,
                             cat_b_init, models_b_init[0] if models_b_init else None, default_text)

    return {
        'load_fn': on_load,
        'load_outputs': [summary_md, tokens_a_html, tokens_b_html, chart, detail_df]
    }
