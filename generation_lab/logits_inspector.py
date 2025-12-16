"""
Logits - 可视化 Next Token 预测
展示 Logits、Temperature、Top-P/Top-K 采样策略
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
from generation_lab.generation_utils import (
    DEMO_MODELS,
    load_model_and_tokenizer,
    get_next_token_logits,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    get_sampling_distribution
)


def render_probability_bar_chart(
    tokens: list,
    probabilities: list,
    title: str = "Token 概率分布",
    cutoff_line: float = None,
    cutoff_label: str = None,
    top_k_cutoff: int = None
):
    """渲染概率柱状图"""
    fig = go.Figure()
    
    # 主柱状图
    colors = ['#2563EB' if i < (top_k_cutoff or len(tokens)) else '#E5E7EB' 
              for i in range(len(tokens))]
    
    fig.add_trace(go.Bar(
        x=tokens,
        y=probabilities,
        marker_color=colors,
        text=[f'{p:.2%}' for p in probabilities],
        textposition='outside',
        textfont=dict(size=10)
    ))
    
    # Top-P 截断线
    if cutoff_line is not None:
        cumsum = np.cumsum(probabilities)
        cutoff_idx = np.searchsorted(cumsum, cutoff_line)
        if cutoff_idx < len(tokens):
            fig.add_hline(
                y=probabilities[cutoff_idx],
                line_dash="dash",
                line_color="#DC2626",
                annotation_text=cutoff_label or f"Top-P 截断 ({cutoff_line:.0%})",
                annotation_position="right"
            )
    
    # Top-K 截断线
    if top_k_cutoff is not None and top_k_cutoff < len(tokens):
        fig.add_vline(
            x=top_k_cutoff - 0.5,
            line_dash="dash",
            line_color="#D97706",
            annotation_text=f"Top-K={top_k_cutoff}",
            annotation_position="top"
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Token",
        yaxis_title="概率",
        yaxis_tickformat='.1%',
        height=450,
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=100),
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


def render_temperature_comparison(logits: torch.Tensor, temperatures: list, tokenizer, top_k: int = 10):
    """渲染不同温度下的概率分布对比"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2
    
    for idx, temp in enumerate(temperatures):
        scaled_logits = logits.clone() / temp if temp > 0 else logits.clone()
        probs = torch.softmax(scaled_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        tokens = [tokenizer.decode([i.item()]) for i in top_indices]
        prob_values = top_probs.tolist()
        
        fig.add_trace(go.Bar(
            name=f'T={temp}',
            x=tokens,
            y=prob_values,
            marker_color=colors[idx % len(colors)],
            opacity=0.8
        ))
    
    fig.update_layout(
        title="Temperature 对概率分布的影响",
        xaxis_title="Token",
        yaxis_title="概率",
        yaxis_tickformat='.1%',
        barmode='group',
        height=480,
        autosize=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_entropy_gauge(probs: torch.Tensor) -> go.Figure:
    """渲染熵值仪表盘"""
    probs_np = probs.numpy()
    probs_np = probs_np[probs_np > 0]
    entropy = -np.sum(probs_np * np.log2(probs_np))
    max_entropy = np.log2(len(probs))
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=entropy,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "分布熵 (bits)", 'font': {'size': 14}},
        delta={'reference': max_entropy / 2, 'increasing': {'color': "#D97706"}},
        gauge={
            'axis': {'range': [0, max_entropy], 'tickwidth': 1},
            'bar': {'color': "#2563EB"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_entropy * 0.3], 'color': '#D1FAE5'},
                {'range': [max_entropy * 0.3, max_entropy * 0.7], 'color': '#FEF3C7'},
                {'range': [max_entropy * 0.7, max_entropy], 'color': '#FEE2E2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_entropy
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        autosize=True,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


# 模型状态缓存
_loaded_model = {"name": None, "model": None, "tokenizer": None}


def load_model(model_choice):
    """加载模型"""
    model_info = DEMO_MODELS[model_choice]
    
    if _loaded_model["name"] == model_info['id']:
        return _loaded_model["model"], _loaded_model["tokenizer"], f"模型已加载: {model_choice}"
    
    model, tokenizer = load_model_and_tokenizer(model_info['id'])
    
    if model is None or tokenizer is None:
        return None, None, "模型加载失败，请检查网络连接"
    
    _loaded_model["name"] = model_info['id']
    _loaded_model["model"] = model
    _loaded_model["tokenizer"] = tokenizer
    
    return model, tokenizer, f"模型加载完成: {model_choice}"


def analyze_probability(model_choice, prompt):
    """分析概率分布 Tab 1"""
    if not prompt:
        return None, pd.DataFrame(), "", "", "", ""
    
    model, tokenizer, status = load_model(model_choice)
    if model is None:
        return None, pd.DataFrame(), status, "", "", ""
    
    token_candidates = get_next_token_logits(model, tokenizer, prompt, top_k=50)
    
    # 指标
    top1_token = f'"{token_candidates[0]["token_str"]}"'
    top1_prob = f'{token_candidates[0]["probability"]:.2%}'
    top1_logit = f'{token_candidates[0]["logit"]:.2f}'
    top5_prob = sum(t['probability'] for t in token_candidates[:5])
    top5_str = f'{top5_prob:.2%}'
    
    # 概率柱状图
    tokens_display = [t['token_str'][:10] + ('...' if len(t['token_str']) > 10 else '') 
                    for t in token_candidates[:20]]
    probs = [t['probability'] for t in token_candidates[:20]]
    
    fig = render_probability_bar_chart(tokens_display, probs, "Top-20 Token 概率分布")
    
    # 详细表格
    df = pd.DataFrame([{
        "排名": t['rank'],
        "Token": repr(t['token_str']),
        "Raw Token": t['raw_token'],
        "Token ID": t['token_id'],
        "Logit": f"{t['logit']:.4f}",
        "概率": f"{t['probability']:.4%}"
    } for t in token_candidates])
    
    return fig, df, top1_token, top1_prob, top1_logit, top5_str


def analyze_temperature(model_choice, prompt, temperature):
    """分析温度 Tab 2"""
    if not prompt:
        return None, None, None
    
    model, tokenizer, _ = load_model(model_choice)
    if model is None:
        return None, None, None
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    # 应用温度
    scaled_logits = logits / temperature
    scaled_probs = torch.softmax(scaled_logits, dim=-1)
    original_probs = torch.softmax(logits, dim=-1)
    
    # 当前温度分布
    top_scaled_probs, top_indices = torch.topk(scaled_probs, 10)
    tokens = [tokenizer.decode([i.item()]) for i in top_indices]
    
    fig1 = render_probability_bar_chart(
        tokens, 
        top_scaled_probs.tolist(),
        f"T={temperature} 概率分布"
    )
    
    # 原始分布
    top_orig_probs, top_orig_indices = torch.topk(original_probs, 10)
    orig_tokens = [tokenizer.decode([i.item()]) for i in top_orig_indices]
    
    fig2 = render_probability_bar_chart(
        orig_tokens,
        top_orig_probs.tolist(),
        "T=1.0 原始分布"
    )
    
    # 多温度对比
    temps = [0.3, 0.7, 1.0, 1.5, 2.0]
    fig_compare = render_temperature_comparison(logits, temps, tokenizer, top_k=8)
    
    return fig1, fig2, fig_compare


def analyze_sampling(model_choice, prompt, top_k, top_p):
    """分析采样 Tab 3"""
    if not prompt:
        return None, None, ""
    
    model, tokenizer, _ = load_model(model_choice)
    if model is None:
        return None, None, ""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :].clone()
    
    original_probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(original_probs, 50)
    
    # 计算 Top-P 截断位置
    cumsum_probs = torch.cumsum(top_probs, dim=-1)
    top_p_cutoff = torch.searchsorted(cumsum_probs, top_p).item() + 1
    
    # 实际截断位置
    effective_cutoff = min(top_k, top_p_cutoff)
    
    summary = f"""
**采样结果**:
- Top-K 截断: 保留前 **{top_k}** 个 token
- Top-P 截断: 累积概率达到 **{top_p:.0%}** 需要 **{top_p_cutoff}** 个 token
- 有效采样范围: **{effective_cutoff}** 个 token
"""
    
    # 可视化
    tokens_display = [tokenizer.decode([i.item()])[:8] for i in top_indices[:30]]
    probs_list = top_probs[:30].tolist()
    
    fig = go.Figure()
    
    colors = []
    for i in range(30):
        if i < effective_cutoff:
            colors.append('#2563EB')
        elif i < top_k:
            colors.append('#D97706')
        else:
            colors.append('#E5E7EB')
    
    fig.add_trace(go.Bar(
        x=tokens_display,
        y=probs_list,
        marker_color=colors,
        text=[f'{p:.1%}' for p in probs_list],
        textposition='outside',
        textfont=dict(size=9)
    ))
    
    if top_k <= 30:
        fig.add_vline(x=top_k - 0.5, line_dash="dash", line_color="#D97706",
                     annotation_text=f"Top-K={top_k}", annotation_position="top")
    
    if top_p_cutoff <= 30:
        fig.add_vline(x=top_p_cutoff - 0.5, line_dash="dot", line_color="#DC2626",
                     annotation_text=f"Top-P={top_p}", annotation_position="bottom")
    
    fig.update_layout(
        title="采样截断可视化",
        xaxis_title="Token",
        yaxis_title="概率",
        yaxis_tickformat='.1%',
        height=480,
        autosize=True,
        margin=dict(l=50, r=50, t=60, b=100),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    fig.update_xaxes(tickangle=45)
    
    # 累积概率曲线
    fig_cdf = go.Figure()
    fig_cdf.add_trace(go.Scatter(
        x=list(range(1, 51)),
        y=cumsum_probs.tolist(),
        mode='lines+markers',
        name='累积概率',
        marker=dict(size=6),
        line=dict(color='#2563EB')
    ))
    
    fig_cdf.add_hline(y=top_p, line_dash="dash", line_color="#DC2626",
                    annotation_text=f"Top-P={top_p}")
    
    fig_cdf.update_layout(
        title="Top-50 Token 累积概率分布",
        xaxis_title="Token 数量",
        yaxis_title="累积概率",
        yaxis_tickformat='.0%',
        height=380,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig, fig_cdf, summary


def render():
    """渲染页面"""
    
    gr.Markdown("## Logits Inspector")
    
    # 默认值
    default_model = list(DEMO_MODELS.keys())[0]
    default_prompt = "The quick brown fox jumps over the"
    default_temp = 1.0
    default_top_k = 10
    default_top_p = 0.9
    
    # 模型选择
    model_choice = gr.Dropdown(
        choices=list(DEMO_MODELS.keys()),
        value=default_model,
        label="选择演示模型"
    )
    
    # Prompt 输入
    prompt = gr.Textbox(
        label="输入 Prompt",
        value=default_prompt,
        lines=3,
        placeholder="输入文本，模型将预测下一个 token..."
    )
    
    with gr.Tabs():
        # Tab 1: 概率分布
        with gr.Tab("概率分布"):
            with gr.Row():
                top1_token = gr.Textbox(label="Top-1 Token", interactive=False)
                top1_prob = gr.Textbox(label="Top-1 概率", interactive=False)
                top1_logit = gr.Textbox(label="Top-1 Logit", interactive=False)
                top5_prob = gr.Textbox(label="Top-5 累积概率", interactive=False)
            
            prob_chart = gr.Plot(label="概率分布图")
            
            with gr.Accordion("详细数据 (Top-50)", open=False):
                detail_df = gr.Dataframe(interactive=False)
        
        # Tab 2: Temperature 实验
        with gr.Tab("Temperature"):
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=3.0,
                value=default_temp,
                step=0.1
            )
            
            with gr.Row():
                temp_chart1 = gr.Plot(label="当前温度分布")
                temp_chart2 = gr.Plot(label="原始分布 T=1.0")
            
            temp_compare_chart = gr.Plot(label="多温度对比")
        
        # Tab 3: Top-K/Top-P 截断
        with gr.Tab("Top-K/Top-P"):
            with gr.Row():
                top_k_slider = gr.Slider(
                    label="Top-K",
                    minimum=1,
                    maximum=50,
                    value=default_top_k,
                    step=1
                )
                top_p_slider = gr.Slider(
                    label="Top-P (Nucleus)",
                    minimum=0.1,
                    maximum=1.0,
                    value=default_top_p,
                    step=0.05
                )
            
            sampling_summary = gr.Markdown("")
            sampling_chart = gr.Plot(label="采样截断")
            cdf_chart = gr.Plot(label="CDF")
    
    # 参数变化自动触发分析
    for component in [model_choice, prompt]:
        component.change(
            fn=analyze_probability,
            inputs=[model_choice, prompt],
            outputs=[prob_chart, detail_df, top1_token, top1_prob, top1_logit, top5_prob]
        )
    
    for component in [model_choice, prompt, temperature]:
        component.change(
            fn=analyze_temperature,
            inputs=[model_choice, prompt, temperature],
            outputs=[temp_chart1, temp_chart2, temp_compare_chart]
        )
    
    for component in [model_choice, prompt, top_k_slider, top_p_slider]:
        component.change(
            fn=analyze_sampling,
            inputs=[model_choice, prompt, top_k_slider, top_p_slider],
            outputs=[sampling_chart, cdf_chart, sampling_summary]
        )
    
    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        prob_result = analyze_probability(default_model, default_prompt)
        temp_result = analyze_temperature(default_model, default_prompt, default_temp)
        sampling_result = analyze_sampling(default_model, default_prompt, default_top_k, default_top_p)
        
        return (
            prob_result[0], prob_result[1], prob_result[2], prob_result[3], prob_result[4], prob_result[5],  # 概率分布
            temp_result[0], temp_result[1], temp_result[2],  # Temperature
            sampling_result[0], sampling_result[1], sampling_result[2]  # Sampling
        )
    
    # 返回 load 事件需要的信息供主 app 调用
    return {
        'load_fn': on_load,
        'load_outputs': [
            prob_chart, detail_df, top1_token, top1_prob, top1_logit, top5_prob,
            temp_chart1, temp_chart2, temp_compare_chart,
            sampling_chart, cdf_chart, sampling_summary
        ]
    }
