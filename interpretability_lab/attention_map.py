"""
Attention - 可视化注意力权重
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
from interpretability_lab.interpretability_utils import (
    INTERPRETABILITY_MODELS,
    load_model_with_attention,
    get_attention_weights,
    compute_attention_entropy,
    get_attention_patterns
)
from model_lab.model_utils import extract_from_url


def render_attention_heatmap(
    attention: np.ndarray,
    tokens: list,
    title: str = "Attention Weights"
) -> go.Figure:
    """渲染注意力热力图"""
    fig = go.Figure(data=go.Heatmap(
        z=attention,
        x=tokens,
        y=tokens,
        colorscale='Blues',
        hoverongaps=False,
        hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Key Position",
        yaxis_title="Query Position",
        xaxis=dict(tickangle=45, side='bottom'),
        yaxis=dict(autorange='reversed'),
        height=500,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_attention_grid(
    attention_weights: torch.Tensor,
    tokens: list,
    layer_idx: int,
    num_heads: int = 4
) -> go.Figure:
    """渲染多个 head 的注意力网格"""
    heads_to_show = min(num_heads, attention_weights.shape[0])
    
    fig = make_subplots(
        rows=1, cols=heads_to_show,
        subplot_titles=[f'Head {i}' for i in range(heads_to_show)],
        horizontal_spacing=0.05
    )
    
    for i in range(heads_to_show):
        attn = attention_weights[i].numpy()
        
        fig.add_trace(
            go.Heatmap(
                z=attn,
                colorscale='Blues',
                showscale=(i == heads_to_show - 1),
                hovertemplate='Q: %{y}<br>K: %{x}<br>W: %{z:.3f}<extra></extra>'
            ),
            row=1, col=i+1
        )
        
        fig.update_xaxes(
            tickvals=list(range(len(tokens))),
            ticktext=tokens,
            tickangle=45,
            row=1, col=i+1
        )
        fig.update_yaxes(
            tickvals=list(range(len(tokens))),
            ticktext=tokens if i == 0 else [],
            autorange='reversed',
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=f"Layer {layer_idx} - Attention Heads",
        height=450,
        autosize=True,
        margin=dict(l=100, r=50, t=80, b=100),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_token_attention_flow(
    attention_weights: torch.Tensor,
    tokens: list,
    selected_token_idx: int
) -> go.Figure:
    """渲染选中 token 的注意力流向"""
    # 平均所有层所有头
    avg_attention = attention_weights.mean(dim=(0, 1)).numpy()
    
    # 选中 token 作为 query 时关注的 keys
    query_attention = avg_attention[selected_token_idx, :]
    
    # 选中 token 作为 key 时被哪些 queries 关注
    key_attention = avg_attention[:, selected_token_idx]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f'What "{tokens[selected_token_idx]}" attends to',
            f'What attends to "{tokens[selected_token_idx]}"'
        ]
    )
    
    # Query -> Keys
    colors1 = ['#2563EB' if i == selected_token_idx else '#60A5FA' for i in range(len(tokens))]
    fig.add_trace(
        go.Bar(x=tokens, y=query_attention, marker_color=colors1, name='Query Attention'),
        row=1, col=1
    )
    
    # Keys -> Query
    colors2 = ['#DC2626' if i == selected_token_idx else '#F87171' for i in range(len(tokens))]
    fig.add_trace(
        go.Bar(x=tokens, y=key_attention, marker_color=colors2, name='Key Attention'),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        autosize=True,
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


# 模型状态缓存
_loaded_model = {"name": None, "model": None, "tokenizer": None, "attention": None, "tokens": None, "config": None}


def get_model_id_from_ui(mode, preset_choice, custom_url):
    """从 UI 输入解析模型 ID"""
    if mode == "Preset Model":
        return INTERPRETABILITY_MODELS[preset_choice]['id']
    else:
        return extract_from_url(custom_url) if custom_url else None


def load_and_analyze(mode, preset_choice, custom_url, token, text, use_causal):
    """加载模型并分析注意力"""
    if not text:
        return None, "", []

    model_id = get_model_id_from_ui(mode, preset_choice, custom_url)
    if not model_id:
        return None, "Please enter a valid model name or URL", []

    # 获取配置信息（用于获取层数和头数）
    if mode == "Preset Model":
        config = INTERPRETABILITY_MODELS[preset_choice]
    else:
        # 从 HuggingFace 获取配置
        if _loaded_model["name"] != model_id or _loaded_model["config"] is None:
            try:
                from transformers import AutoConfig
                hf_config = AutoConfig.from_pretrained(model_id, token=token if token else None)
                config = {
                    'id': model_id,
                    'layers': getattr(hf_config, 'num_hidden_layers', getattr(hf_config, 'n_layer', 12)),
                    'heads': getattr(hf_config, 'num_attention_heads', getattr(hf_config, 'n_head', 12)),
                    'hidden': getattr(hf_config, 'hidden_size', getattr(hf_config, 'n_embd', 768))
                }
                _loaded_model["config"] = config
            except Exception as e:
                return None, f"Failed to load config: {e}", []
        else:
            config = _loaded_model["config"]

    # 加载模型
    if _loaded_model["name"] != model_id:
        model, tokenizer = load_model_with_attention(model_id, token=token if token else None)
        if model is None:
            return None, f"Failed to load model: {model_id}", []
        _loaded_model["name"] = model_id
        _loaded_model["model"] = model
        _loaded_model["tokenizer"] = tokenizer
        _loaded_model["config"] = config
    else:
        model = _loaded_model["model"]
        tokenizer = _loaded_model["tokenizer"]

    # 获取注意力权重
    attention_weights, tokens = get_attention_weights(model, tokenizer, text)

    # 应用 causal mask
    if use_causal:
        seq_len = attention_weights.shape[-1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        attention_weights = attention_weights * causal_mask.unsqueeze(0).unsqueeze(0)

    _loaded_model["attention"] = attention_weights
    _loaded_model["tokens"] = tokens

    token_info = f"Sequence length: {len(tokens)} tokens"
    token_choices = [f"{i}: {tokens[i]}" for i in range(len(tokens))]

    return attention_weights, token_info, token_choices


def analyze_heatmap(layer_idx, head_choice):
    """分析热力图 Tab 1"""
    attention_weights = _loaded_model.get("attention")
    tokens = _loaded_model.get("tokens")

    if attention_weights is None or tokens is None:
        return None

    if head_choice == "All Heads":
        fig = render_attention_grid(
            attention_weights[layer_idx],
            tokens,
            layer_idx,
            num_heads=4
        )
    else:
        head_idx = int(head_choice.split()[-1])
        attn = attention_weights[layer_idx, head_idx].numpy()
        fig = render_attention_heatmap(
            attn,
            tokens,
            f"Layer {layer_idx}, Head {head_idx}"
        )

    return fig


def analyze_token(selected_token):
    """分析 Token Tab 2"""
    attention_weights = _loaded_model.get("attention")
    tokens = _loaded_model.get("tokens")
    config = _loaded_model.get("config")

    if attention_weights is None or tokens is None or not selected_token or config is None:
        return None, None

    # 解析选中的 token 索引
    selected_idx = int(selected_token.split(":")[0])

    # 注意力流向图
    fig_flow = render_token_attention_flow(attention_weights, tokens, selected_idx)

    # 各层注意力变化
    layer_attention = []
    for l in range(config['layers']):
        avg_attn = attention_weights[l].mean(dim=0).numpy()
        layer_attention.append(avg_attn[selected_idx, :])

    layer_attn_matrix = np.array(layer_attention)

    fig_layers = go.Figure(data=go.Heatmap(
        z=layer_attn_matrix,
        x=tokens,
        y=[f'Layer {i}' for i in range(config['layers'])],
        colorscale='Viridis'
    ))

    fig_layers.update_layout(
        title=f'Attention distribution of "{tokens[selected_idx]}" across layers',
        xaxis_title="Key Token",
        yaxis_title="Layer",
        height=450,
        autosize=True,
        xaxis=dict(tickangle=45),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )

    return fig_flow, fig_layers


def analyze_patterns():
    """分析模式 Tab 3"""
    attention_weights = _loaded_model.get("attention")
    tokens = _loaded_model.get("tokens")
    config = _loaded_model.get("config")

    if attention_weights is None or tokens is None or config is None:
        return "", None, None

    # 分析各层各头的模式
    patterns_data = []

    for l in range(config['layers']):
        for h in range(config['heads']):
            attn = attention_weights[l, h].numpy()
            patterns = get_attention_patterns(torch.tensor(attn))
            patterns_data.append({
                'Layer': l,
                'Head': h,
                'Diagonal': patterns['diagonal'],
                'First Token': patterns['first_token'],
                'Local': patterns['local'],
                'Global': patterns['global']
            })

    df_patterns = pd.DataFrame(patterns_data)

    # 汇总统计
    stats_md = f"""
### Attention Pattern Statistics

| Metric | Mean |
|------|--------|
| Diagonal Attention | {df_patterns['Diagonal'].mean():.2%} |
| First Token Attention | {df_patterns['First Token'].mean():.2%} |
| Local Attention | {df_patterns['Local'].mean():.2%} |
| Global Attention | {df_patterns['Global'].mean():.2%} |
"""

    # 热力图展示各头的模式
    diagonal_matrix = df_patterns.pivot(index='Layer', columns='Head', values='Diagonal')

    fig_pattern = go.Figure(data=go.Heatmap(
        z=diagonal_matrix.values,
        x=[f'H{i}' for i in range(config['heads'])],
        y=[f'L{i}' for i in range(config['layers'])],
        colorscale='RdBu',
        zmid=0.5,
        hovertemplate='Layer %{y}, Head %{x}<br>Diagonal Attention: %{z:.2%}<extra></extra>'
    ))

    fig_pattern.update_layout(
        title="Diagonal Attention Strength (self-attention intensity)",
        xaxis_title="Head",
        yaxis_title="Layer",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )

    # 熵分析
    entropy_data = []
    for l in range(config['layers']):
        for h in range(config['heads']):
            attn = attention_weights[l, h]
            entropy = compute_attention_entropy(attn).mean().item()
            entropy_data.append({
                'Layer': l,
                'Head': h,
                'Entropy': entropy
            })

    df_entropy = pd.DataFrame(entropy_data)
    entropy_matrix = df_entropy.pivot(index='Layer', columns='Head', values='Entropy')

    fig_entropy = go.Figure(data=go.Heatmap(
        z=entropy_matrix.values,
        x=[f'H{i}' for i in range(config['heads'])],
        y=[f'L{i}' for i in range(config['layers'])],
        colorscale='Plasma'
    ))

    fig_entropy.update_layout(
        title="Attention Entropy (higher = more dispersed)",
        xaxis_title="Head",
        yaxis_title="Layer",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )

    return stats_md, fig_pattern, fig_entropy


def render():
    """渲染页面"""

    # 模型选择
    with gr.Row():
        model_mode = gr.Radio(
            label="Input Method",
            choices=["Preset Model", "Custom Model"],
            value="Preset Model"
        )

    with gr.Row():
        preset_model = gr.Dropdown(
            choices=list(INTERPRETABILITY_MODELS.keys()),
            value=list(INTERPRETABILITY_MODELS.keys())[0],
            label="Select Model"
        )

        custom_model = gr.Textbox(
            label="Model Name or URL",
            placeholder="e.g., openai-community/gpt2",
            visible=False
        )

    hf_token = gr.Textbox(
        label="HF Token (Optional)",
        type="password",
        placeholder="For private models",
        visible=False
    )

    # 输入文本
    default_text = "The animal didn't cross the street because it was too tired"
    text = gr.Textbox(
        label="Input Text",
        value=default_text,
        lines=2
    )

    use_causal = gr.Checkbox(
        label="Apply Causal Mask",
        value=True
    )

    token_info = gr.Textbox(label="Token Info", interactive=False)
    
    with gr.Tabs():
        # Tab 1: Heatmap
        with gr.Tab("Heatmap") as heatmap_tab:
            with gr.Row():
                layer_select = gr.Slider(
                    label="Select Layer",
                    minimum=0,
                    maximum=11,
                    value=0,
                    step=1
                )
                head_select = gr.Dropdown(
                    choices=["All Heads"] + [f"Head {i}" for i in range(12)],
                    value="All Heads",
                    label="Select Head"
                )

            heatmap_plot = gr.Plot()

        # Tab 2: Token Analysis
        with gr.Tab("Token Analysis") as token_analysis_tab:
            token_select = gr.Dropdown(
                choices=[],
                label="Select Token to Analyze",
                allow_custom_value=True,
                value=None
            )

            flow_plot = gr.Plot(label="Attention Flow")
            layer_plot = gr.Plot(label="Layer-wise Attention")

        # Tab 3: Pattern Analysis
        with gr.Tab("Pattern Analysis") as pattern_analysis_tab:
            stats_md = gr.Markdown("")

            with gr.Row():
                pattern_plot = gr.Plot(label="Attention Patterns")
                entropy_plot = gr.Plot(label="Attention Entropy")
    
    # Toggle函数：切换预设/自定义模型模式
    def toggle_model_mode(mode):
        return (
            gr.update(visible=(mode == "Preset Model")),
            gr.update(visible=(mode == "Custom Model")),
            gr.update(visible=(mode == "Custom Model"))
        )

    # 定义级联更新函数
    def on_analyze_full(mode, preset_choice, custom_url, token, text, use_causal, layer_idx, head_choice):
        """完整分析：包括热力图、token分析、模式分析"""
        result = load_and_analyze(mode, preset_choice, custom_url, token, text, use_causal)
        if result[0] is None:
            return result[1], gr.update(choices=[], value=None), None, None, None, "", None, None

        token_choices = result[2]
        first_token = token_choices[0] if token_choices else None

        # 热力图
        heatmap = analyze_heatmap(layer_idx, head_choice)

        # Token 分析（默认选第一个）
        flow_fig, layer_fig = None, None
        if first_token:
            flow_fig, layer_fig = analyze_token(first_token)

        # 模式分析
        stats, pattern_fig, entropy_fig = analyze_patterns()

        return (
            result[1],
            gr.update(choices=token_choices, value=first_token),
            heatmap,
            flow_fig,
            layer_fig,
            stats,
            pattern_fig,
            entropy_fig
        )

    # 页面加载时自动计算默认值
    def on_load():
        """页面加载时的初始化"""
        default_model = list(INTERPRETABILITY_MODELS.keys())[0]
        return on_analyze_full("Preset Model", default_model, "", None, default_text, True, 0, "All Heads")
    
    # Toggle 事件
    model_mode.change(
        fn=toggle_model_mode,
        inputs=[model_mode],
        outputs=[preset_model, custom_model, hf_token]
    )

    # 事件绑定 - 输入变化自动触发分析
    for component in [model_mode, preset_model, custom_model, hf_token, text, use_causal]:
        component.change(
            fn=on_analyze_full,
            inputs=[model_mode, preset_model, custom_model, hf_token, text, use_causal, layer_select, head_select],
            outputs=[token_info, token_select, heatmap_plot, flow_plot, layer_plot, stats_md, pattern_plot, entropy_plot]
        )

    # Layer/Head 选择自动更新热力图
    layer_select.change(
        fn=analyze_heatmap,
        inputs=[layer_select, head_select],
        outputs=[heatmap_plot]
    )

    head_select.change(
        fn=analyze_heatmap,
        inputs=[layer_select, head_select],
        outputs=[heatmap_plot]
    )

    # Token 选择自动更新分析
    token_select.change(
        fn=analyze_token,
        inputs=[token_select],
        outputs=[flow_plot, layer_plot]
    )

    # Re-render plots when tabs become visible to fix width issues
    token_analysis_tab.select(
        fn=analyze_token,
        inputs=[token_select],
        outputs=[flow_plot, layer_plot]
    )

    pattern_analysis_tab.select(
        fn=analyze_patterns,
        inputs=[],
        outputs=[stats_md, pattern_plot, entropy_plot]
    )

    # 返回 load 事件需要的信息供主 app 调用
    return {
        'load_fn': on_load,
        'load_outputs': [token_info, token_select, heatmap_plot, flow_plot, layer_plot, stats_md, pattern_plot, entropy_plot]
    }
