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
        width=600,
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
        height=400,
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
            f'"{tokens[selected_token_idx]}" 关注哪些 tokens',
            f'哪些 tokens 关注 "{tokens[selected_token_idx]}"'
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
        height=350,
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig


# 模型状态缓存
_loaded_model = {"name": None, "model": None, "tokenizer": None, "attention": None, "tokens": None}


def load_and_analyze(model_choice, text, use_causal):
    """加载模型并分析注意力"""
    if not text:
        return None, "", []
    
    model_info = INTERPRETABILITY_MODELS[model_choice]
    
    # 加载模型
    if _loaded_model["name"] != model_info['id']:
        model, tokenizer = load_model_with_attention(model_info['id'])
        if model is None:
            return None, "模型加载失败", []
        _loaded_model["name"] = model_info['id']
        _loaded_model["model"] = model
        _loaded_model["tokenizer"] = tokenizer
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
    
    token_info = f"序列长度: {len(tokens)} tokens"
    token_choices = [f"{i}: {tokens[i]}" for i in range(len(tokens))]
    
    return attention_weights, token_info, token_choices


def analyze_heatmap(model_choice, layer_idx, head_choice):
    """分析热力图 Tab 1"""
    attention_weights = _loaded_model.get("attention")
    tokens = _loaded_model.get("tokens")
    
    if attention_weights is None or tokens is None:
        return None
    
    model_info = INTERPRETABILITY_MODELS[model_choice]
    
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


def analyze_token(model_choice, selected_token):
    """分析 Token Tab 2"""
    attention_weights = _loaded_model.get("attention")
    tokens = _loaded_model.get("tokens")
    
    if attention_weights is None or tokens is None or not selected_token:
        return None, None
    
    model_info = INTERPRETABILITY_MODELS[model_choice]
    
    # 解析选中的 token 索引
    selected_idx = int(selected_token.split(":")[0])
    
    # 注意力流向图
    fig_flow = render_token_attention_flow(attention_weights, tokens, selected_idx)
    
    # 各层注意力变化
    layer_attention = []
    for l in range(model_info['layers']):
        avg_attn = attention_weights[l].mean(dim=0).numpy()
        layer_attention.append(avg_attn[selected_idx, :])
    
    layer_attn_matrix = np.array(layer_attention)
    
    fig_layers = go.Figure(data=go.Heatmap(
        z=layer_attn_matrix,
        x=tokens,
        y=[f'Layer {i}' for i in range(model_info['layers'])],
        colorscale='Viridis'
    ))
    
    fig_layers.update_layout(
        title=f'"{tokens[selected_idx]}" 在各层的注意力分布',
        xaxis_title="Key Token",
        yaxis_title="Layer",
        height=400,
        xaxis=dict(tickangle=45),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig_flow, fig_layers


def analyze_patterns(model_choice):
    """分析模式 Tab 3"""
    attention_weights = _loaded_model.get("attention")
    tokens = _loaded_model.get("tokens")
    
    if attention_weights is None or tokens is None:
        return "", None, None
    
    model_info = INTERPRETABILITY_MODELS[model_choice]
    
    # 分析各层各头的模式
    patterns_data = []
    
    for l in range(model_info['layers']):
        for h in range(model_info['heads']):
            attn = attention_weights[l, h].numpy()
            patterns = get_attention_patterns(torch.tensor(attn))
            patterns_data.append({
                'Layer': l,
                'Head': h,
                '对角线': patterns['diagonal'],
                '首 Token': patterns['first_token'],
                '局部': patterns['local'],
                '全局': patterns['global']
            })
    
    df_patterns = pd.DataFrame(patterns_data)
    
    # 汇总统计
    stats_md = f"""
### 注意力模式统计

| 指标 | 平均值 |
|------|--------|
| 对角线注意力 | {df_patterns['对角线'].mean():.2%} |
| 首 Token 注意力 | {df_patterns['首 Token'].mean():.2%} |
| 局部注意力 | {df_patterns['局部'].mean():.2%} |
| 全局注意力 | {df_patterns['全局'].mean():.2%} |
"""
    
    # 热力图展示各头的模式
    diagonal_matrix = df_patterns.pivot(index='Layer', columns='Head', values='对角线')
    
    fig_pattern = go.Figure(data=go.Heatmap(
        z=diagonal_matrix.values,
        x=[f'H{i}' for i in range(model_info['heads'])],
        y=[f'L{i}' for i in range(model_info['layers'])],
        colorscale='RdBu',
        zmid=0.5,
        hovertemplate='Layer %{y}, Head %{x}<br>Diagonal Attention: %{z:.2%}<extra></extra>'
    ))
    
    fig_pattern.update_layout(
        title="对角线注意力强度 (每个 token 关注自己的程度)",
        xaxis_title="Head",
        yaxis_title="Layer",
        height=400,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    # 熵分析
    entropy_data = []
    for l in range(model_info['layers']):
        for h in range(model_info['heads']):
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
        x=[f'H{i}' for i in range(model_info['heads'])],
        y=[f'L{i}' for i in range(model_info['layers'])],
        colorscale='Plasma'
    ))
    
    fig_entropy.update_layout(
        title="注意力熵 (值越大表示注意力越分散)",
        xaxis_title="Head",
        yaxis_title="Layer",
        height=400,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return stats_md, fig_pattern, fig_entropy


def render():
    """渲染页面"""
    
    gr.Markdown("## Attention 可视化")
    
    # 模型选择
    model_choice = gr.Dropdown(
        choices=list(INTERPRETABILITY_MODELS.keys()),
        value=list(INTERPRETABILITY_MODELS.keys())[0],
        label="选择模型"
    )
    
    # 输入文本
    text = gr.Textbox(
        label="输入文本",
        value="The animal didn't cross the street because it was too tired",
        lines=2
    )
    
    use_causal = gr.Checkbox(
        label="应用 Causal Mask",
        value=True
    )
    
    token_info = gr.Textbox(label="Token 信息", interactive=False)
    
    with gr.Tabs():
        # Tab 1: 热力图
        with gr.Tab("热力图"):
            with gr.Row():
                layer_select = gr.Slider(
                    label="选择层",
                    minimum=0,
                    maximum=11,
                    value=0,
                    step=1
                )
                head_select = gr.Dropdown(
                    choices=["All Heads"] + [f"Head {i}" for i in range(12)],
                    value="All Heads",
                    label="选择 Head"
                )
            
            heatmap_plot = gr.Plot()
        
        # Tab 2: Token 分析
        with gr.Tab("Token 分析"):
            token_select = gr.Dropdown(
                choices=[],
                label="选择要分析的 Token"
            )
            
            flow_plot = gr.Plot(label="注意力流向")
            layer_plot = gr.Plot(label="各层注意力分布")
        
        # Tab 3: 模式分析
        with gr.Tab("模式分析"):
            stats_md = gr.Markdown("")
            
            with gr.Row():
                pattern_plot = gr.Plot(label="注意力模式")
                entropy_plot = gr.Plot(label="注意力熵")
    
    # 定义级联更新函数
    def on_analyze(model_choice, text, use_causal, layer_idx, head_choice):
        """分析注意力并自动更新热力图"""
        result = load_and_analyze(model_choice, text, use_causal)
        if result[0] is None:
            return result[1], result[2], None
        
        # 自动更新热力图
        heatmap = analyze_heatmap(model_choice, layer_idx, head_choice)
        return result[1], result[2], heatmap
    
    # 事件绑定 - 输入变化自动触发分析
    text.change(
        fn=on_analyze,
        inputs=[model_choice, text, use_causal, layer_select, head_select],
        outputs=[token_info, token_select, heatmap_plot]
    )
    
    use_causal.change(
        fn=on_analyze,
        inputs=[model_choice, text, use_causal, layer_select, head_select],
        outputs=[token_info, token_select, heatmap_plot]
    )
    
    model_choice.change(
        fn=on_analyze,
        inputs=[model_choice, text, use_causal, layer_select, head_select],
        outputs=[token_info, token_select, heatmap_plot]
    )
    
    # Layer/Head 选择自动更新热力图
    layer_select.change(
        fn=analyze_heatmap,
        inputs=[model_choice, layer_select, head_select],
        outputs=[heatmap_plot]
    )
    
    head_select.change(
        fn=analyze_heatmap,
        inputs=[model_choice, layer_select, head_select],
        outputs=[heatmap_plot]
    )
    
    # Token 选择自动更新分析
    token_select.change(
        fn=analyze_token,
        inputs=[model_choice, token_select],
        outputs=[flow_plot, layer_plot]
    )
    
    # 模式分析 Tab 自动更新（当有数据时）
    def auto_pattern_analyze(model_choice):
        if _loaded_model.get("attention") is not None:
            return analyze_patterns(model_choice)
        return "", None, None
