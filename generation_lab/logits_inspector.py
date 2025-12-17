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
from model_lab.model_utils import extract_from_url


def render_probability_bar_chart(
    tokens: list,
    probabilities: list,
    title: str = "Token Probability Distribution",
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
                annotation_text=cutoff_label or f"Top-P cutoff ({cutoff_line:.0%})",
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
        yaxis_title="Probability",
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
        title="Temperature Effect on Probability Distribution",
        xaxis_title="Token",
        yaxis_title="Probability",
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
        title={'text': "Distribution Entropy (bits)", 'font': {'size': 14}},
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


def get_model_id_from_ui(mode, preset_choice, custom_url):
    """从 UI 输入解析模型 ID"""
    if mode == "Preset Model":
        return DEMO_MODELS[preset_choice]['id']
    else:
        return extract_from_url(custom_url) if custom_url else None


def load_model(mode, preset_choice, custom_url, token=None):
    """加载模型，使用缓存避免重复加载"""
    global _loaded_model

    model_id = get_model_id_from_ui(mode, preset_choice, custom_url)
    if not model_id:
        return None, None, "Please enter a valid model name or URL"

    # 如果已加载且是同一模型，直接返回缓存
    if _loaded_model["name"] == model_id:
        return _loaded_model["model"], _loaded_model["tokenizer"], f"Using cached model"

    # 加载新模型
    model, tokenizer = load_model_and_tokenizer(model_id)
    if model is None or tokenizer is None:
        return None, None, f"Failed to load model: {model_id}"

    _loaded_model["name"] = model_id
    _loaded_model["model"] = model
    _loaded_model["tokenizer"] = tokenizer

    return model, tokenizer, f"Model loaded: {model_id}"


def analyze_probability(mode, preset_choice, custom_url, token, prompt):
    """分析概率分布 Tab 1"""
    if not prompt:
        return None, pd.DataFrame(), "", "", "", ""

    model, tokenizer, status = load_model(mode, preset_choice, custom_url, token)
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
    
    fig = render_probability_bar_chart(tokens_display, probs, "Top-20 Token Probability Distribution")
    
    # 详细表格
    df = pd.DataFrame([{
        "Rank": t['rank'],
        "Token": repr(t['token_str']),
        "Raw Token": t['raw_token'],
        "Token ID": t['token_id'],
        "Logit": f"{t['logit']:.4f}",
        "Probability": f"{t['probability']:.4%}"
    } for t in token_candidates])
    
    return fig, df, top1_token, top1_prob, top1_logit, top5_str


def analyze_temperature(mode, preset_choice, custom_url, token, prompt, temperature):
    """分析温度 Tab 2"""
    if not prompt:
        return None, None, None

    model, tokenizer, _ = load_model(mode, preset_choice, custom_url, token)
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
        f"T={temperature} Probability Distribution"
    )
    
    # 原始分布
    top_orig_probs, top_orig_indices = torch.topk(original_probs, 10)
    orig_tokens = [tokenizer.decode([i.item()]) for i in top_orig_indices]
    
    fig2 = render_probability_bar_chart(
        orig_tokens,
        top_orig_probs.tolist(),
        "T=1.0 Original Distribution"
    )
    
    # 多温度对比
    temps = [0.3, 0.7, 1.0, 1.5, 2.0]
    fig_compare = render_temperature_comparison(logits, temps, tokenizer, top_k=8)
    
    return fig1, fig2, fig_compare


def analyze_sampling(mode, preset_choice, custom_url, token, prompt, top_k, top_p):
    """分析采样 Tab 3"""
    if not prompt:
        return None, None, ""

    model, tokenizer, _ = load_model(mode, preset_choice, custom_url, token)
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
**Sampling Results**:
- Top-K cutoff: Keep top **{top_k}** tokens
- Top-P cutoff: **{top_p_cutoff}** tokens needed to reach **{top_p:.0%}** cumulative probability
- Effective sampling range: **{effective_cutoff}** tokens
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
        title="Sampling Cutoff Visualization",
        xaxis_title="Token",
        yaxis_title="Probability",
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
        name='Cumulative Probability',
        marker=dict(size=6),
        line=dict(color='#2563EB')
    ))
    
    fig_cdf.add_hline(y=top_p, line_dash="dash", line_color="#DC2626",
                    annotation_text=f"Top-P={top_p}")
    
    fig_cdf.update_layout(
        title="Top-50 Token Cumulative Probability Distribution",
        xaxis_title="Token Count",
        yaxis_title="Cumulative Probability",
        yaxis_tickformat='.0%',
        height=380,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig, fig_cdf, summary


def render():
    """渲染页面"""

    # 默认值
    default_model = list(DEMO_MODELS.keys())[0]
    default_prompt = "The quick brown fox jumps over the"
    default_temp = 1.0
    default_top_k = 10
    default_top_p = 0.9

    # 模型选择
    with gr.Row():
        model_mode = gr.Radio(
            label="Input Method",
            choices=["Preset Model", "Custom Model"],
            value="Preset Model"
        )

    with gr.Row():
        preset_model = gr.Dropdown(
            choices=list(DEMO_MODELS.keys()),
            value=default_model,
            label="Select Model"
        )

        custom_model = gr.Textbox(
            label="Model Name or URL",
            placeholder="e.g., openai-community/gpt2, meta-llama/Llama-2-7b-hf",
            visible=False
        )

    hf_token = gr.Textbox(
        label="HF Token (Optional)",
        type="password",
        placeholder="For private models",
        visible=False
    )

    # Prompt 输入
    prompt = gr.Textbox(
        label="Input Prompt",
        value=default_prompt,
        lines=3,
        placeholder="Enter text, model will predict the next token..."
    )
    
    with gr.Tabs():
        # Tab 1: Probability Distribution
        with gr.Tab("Probability") as prob_tab:
            with gr.Row():
                top1_token = gr.Textbox(label="Top-1 Token", interactive=False)
                top1_prob = gr.Textbox(label="Top-1 Probability", interactive=False)
                top1_logit = gr.Textbox(label="Top-1 Logit", interactive=False)
                top5_prob = gr.Textbox(label="Top-5 Cumulative", interactive=False)

            prob_chart = gr.Plot(label="Probability Distribution")

            with gr.Accordion("Detailed Data (Top-50)", open=False):
                detail_df = gr.Dataframe(interactive=False)

        # Tab 2: Temperature
        with gr.Tab("Temperature") as temp_tab:
            temperature = gr.Slider(
                label="Temperature",
                minimum=0.1,
                maximum=3.0,
                value=default_temp,
                step=0.1
            )

            with gr.Row():
                temp_chart1 = gr.Plot(label="Current Temperature Distribution")
                temp_chart2 = gr.Plot(label="Original Distribution T=1.0")

            temp_compare_chart = gr.Plot(label="Multi-Temperature Comparison")

        # Tab 3: Top-K/Top-P
        with gr.Tab("Top-K/Top-P") as sampling_tab:
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
            sampling_chart = gr.Plot(label="Sampling Cutoff")
            cdf_chart = gr.Plot(label="CDF")

    # Toggle函数：切换预设/自定义模型模式
    def toggle_model_mode(mode):
        return (
            gr.update(visible=(mode == "Preset Model")),
            gr.update(visible=(mode == "Custom Model")),
            gr.update(visible=(mode == "Custom Model"))
        )

    model_mode.change(
        fn=toggle_model_mode,
        inputs=[model_mode],
        outputs=[preset_model, custom_model, hf_token]
    )

    # 参数变化自动触发分析
    for component in [model_mode, preset_model, custom_model, hf_token, prompt]:
        component.change(
            fn=analyze_probability,
            inputs=[model_mode, preset_model, custom_model, hf_token, prompt],
            outputs=[prob_chart, detail_df, top1_token, top1_prob, top1_logit, top5_prob]
        )

    for component in [model_mode, preset_model, custom_model, hf_token, prompt, temperature]:
        component.change(
            fn=analyze_temperature,
            inputs=[model_mode, preset_model, custom_model, hf_token, prompt, temperature],
            outputs=[temp_chart1, temp_chart2, temp_compare_chart]
        )

    for component in [model_mode, preset_model, custom_model, hf_token, prompt, top_k_slider, top_p_slider]:
        component.change(
            fn=analyze_sampling,
            inputs=[model_mode, preset_model, custom_model, hf_token, prompt, top_k_slider, top_p_slider],
            outputs=[sampling_chart, cdf_chart, sampling_summary]
        )

    # Re-render plots when tabs become visible to fix width issues
    temp_tab.select(
        fn=analyze_temperature,
        inputs=[model_mode, preset_model, custom_model, hf_token, prompt, temperature],
        outputs=[temp_chart1, temp_chart2, temp_compare_chart]
    )

    sampling_tab.select(
        fn=analyze_sampling,
        inputs=[model_mode, preset_model, custom_model, hf_token, prompt, top_k_slider, top_p_slider],
        outputs=[sampling_chart, cdf_chart, sampling_summary]
    )

    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        prob_result = analyze_probability("Preset Model", default_model, "", None, default_prompt)
        temp_result = analyze_temperature("Preset Model", default_model, "", None, default_prompt, default_temp)
        sampling_result = analyze_sampling("Preset Model", default_model, "", None, default_prompt, default_top_k, default_top_p)
        
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
