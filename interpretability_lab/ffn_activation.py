"""
FFN - 分析 Feed-Forward 层的激活情况
"""

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from interpretability_lab.interpretability_utils import (
    INTERPRETABILITY_MODELS,
    load_model_with_attention,
    analyze_ffn_activations,
    compare_activation_functions,
    MODEL_ARCHITECTURES
)


def render_activation_curves() -> go.Figure:
    """渲染激活函数曲线对比"""
    x = np.linspace(-4, 4, 200)
    activations = compare_activation_functions(x)
    
    fig = go.Figure()
    
    colors = ['#2563EB', '#DC2626', '#059669', '#7C3AED']
    
    for (name, values), color in zip(activations.items(), colors):
        fig.add_trace(go.Scatter(
            x=x,
            y=values,
            mode='lines',
            name=name,
            line=dict(color=color, width=2)
        ))
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title="激活函数对比",
        xaxis_title="输入 x",
        yaxis_title="输出 f(x)",
        height=450,
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


def render_swiglu_visualization() -> go.Figure:
    """渲染 SwiGLU 门控可视化"""
    x = np.linspace(-3, 3, 100)
    gate = 1 / (1 + np.exp(-x))  # Sigmoid gate
    silu = x * gate
    
    # 假设 W2 路径的输出
    np.random.seed(42)
    w2_out = 0.5 * x + 0.3
    
    swiglu_out = silu * w2_out
    
    fig = make_subplots(rows=1, cols=3, subplot_titles=['SiLU(xW1)', 'xW2', 'SwiGLU Output'])
    
    fig.add_trace(go.Scatter(x=x, y=silu, line=dict(color='#2563EB')), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=w2_out, line=dict(color='#059669')), row=1, col=2)
    fig.add_trace(go.Scatter(x=x, y=swiglu_out, line=dict(color='#DC2626')), row=1, col=3)
    
    fig.update_layout(
        height=350,
        autosize=True,
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_activation_histogram(activations: np.ndarray, title: str) -> go.Figure:
    """渲染激活值直方图"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=activations.flatten(),
        nbinsx=50,
        marker_color='#2563EB',
        opacity=0.7
    ))
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="零点")
    
    fig.update_layout(
        title=title,
        xaxis_title="激活值",
        yaxis_title="频次",
        height=400,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_sparsity_comparison(code_values, text_values, thresholds) -> go.Figure:
    """渲染稀疏性对比"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=['代码', '自然语言'])
    
    if code_values is not None:
        code_sparse = [(np.abs(code_values) < t).mean() * 100 for t in thresholds]
        fig.add_trace(go.Bar(x=[f'<{t}' for t in thresholds], y=code_sparse, marker_color='#2563EB'), row=1, col=1)
    
    if text_values is not None:
        text_sparse = [(np.abs(text_values) < t).mean() * 100 for t in thresholds]
        fig.add_trace(go.Bar(x=[f'<{t}' for t in thresholds], y=text_sparse, marker_color='#059669'), row=1, col=2)
    
    fig.update_layout(
        height=400,
        autosize=True,
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


# 模型状态缓存
_loaded_model = {"name": None, "model": None, "tokenizer": None}


def analyze_activations(model_choice, code_input, text_input, layer_idx):
    """分析激活"""
    model_info = INTERPRETABILITY_MODELS[model_choice]
    
    # 加载模型
    if _loaded_model["name"] != model_info['id']:
        model, tokenizer = load_model_with_attention(model_info['id'])
        if model is None:
            return None, None, None, "", "", "", "", "", ""
        _loaded_model["name"] = model_info['id']
        _loaded_model["model"] = model
        _loaded_model["tokenizer"] = tokenizer
    else:
        model = _loaded_model["model"]
        tokenizer = _loaded_model["tokenizer"]
    
    # 分析代码输入
    code_results = analyze_ffn_activations(model, tokenizer, code_input, layer_idx)
    text_results = analyze_ffn_activations(model, tokenizer, text_input, layer_idx)
    
    # 代码结果
    code_mean = code_std = code_sparse = ""
    code_hist = None
    code_values = None
    if 'fc1' in code_results:
        fc1_data = code_results['fc1']
        code_mean = f"{fc1_data['mean']:.4f}"
        code_std = f"{fc1_data['std']:.4f}"
        code_sparse = f"{fc1_data['sparsity']:.1%}"
        code_hist = render_activation_histogram(fc1_data['values'], "代码激活分布")
        code_values = fc1_data['values']
    
    # 文本结果
    text_mean = text_std = text_sparse = ""
    text_hist = None
    text_values = None
    if 'fc1' in text_results:
        fc1_data = text_results['fc1']
        text_mean = f"{fc1_data['mean']:.4f}"
        text_std = f"{fc1_data['std']:.4f}"
        text_sparse = f"{fc1_data['sparsity']:.1%}"
        text_hist = render_activation_histogram(fc1_data['values'], "自然语言激活分布")
        text_values = fc1_data['values']
    
    # 稀疏性对比
    thresholds = [0.01, 0.1, 0.5, 1.0]
    sparse_fig = render_sparsity_comparison(code_values, text_values, thresholds)
    
    return code_hist, text_hist, sparse_fig, code_mean, code_std, code_sparse, text_mean, text_std, text_sparse


def render_architecture_comparison():
    """渲染架构对比表"""
    arch_data = []
    for model_name, arch in MODEL_ARCHITECTURES.items():
        arch_data.append({
            "模型": model_name,
            "Attention": arch['attention'],
            "FFN 激活": arch['ffn'],
            "归一化": arch['norm'],
            "位置编码": arch['position']
        })
    return pd.DataFrame(arch_data)


def render_params_pie():
    """渲染参数分布饼图"""
    fig = go.Figure(data=[go.Pie(
        labels=['FFN (W1, W2)', 'Attention (Q, K, V, O)', 'Embedding', 'LayerNorm'],
        values=[66, 25, 8, 1],
        marker_colors=['#2563EB', '#059669', '#D97706', '#6B7280'],
        hole=0.4
    )])
    
    fig.update_layout(
        title="Transformer 参数分布 (典型 Decoder-only 模型)",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render():
    """渲染页面"""
    
    gr.Markdown("## FFN 激活探测")
    
    # 默认值
    default_model = list(INTERPRETABILITY_MODELS.keys())[0]
    default_code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    default_text = "The sun rises in the east and sets in the west. Birds fly south for the winter."
    default_layer = 0
    
    with gr.Tabs():
        # Tab 1: 激活函数
        with gr.Tab("激活函数"):
            activation_plot = gr.Plot(value=render_activation_curves())
            swiglu_plot = gr.Plot(value=render_swiglu_visualization())
        
        # Tab 2: 激活分析
        with gr.Tab("激活分析"):
            model_choice = gr.Dropdown(
                choices=list(INTERPRETABILITY_MODELS.keys()),
                value=default_model,
                label="模型"
            )
            
            with gr.Row():
                code_input = gr.Textbox(
                    label="代码输入",
                    value=default_code,
                    lines=5
                )
                text_input = gr.Textbox(
                    label="文本输入",
                    value=default_text,
                    lines=5
                )
            
            layer_idx = gr.Slider(
                label="Layer",
                minimum=0,
                maximum=11,
                value=default_layer,
                step=1
            )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Code")
                    with gr.Row():
                        code_mean = gr.Textbox(label="Mean", interactive=False)
                        code_std = gr.Textbox(label="Std", interactive=False)
                        code_sparse = gr.Textbox(label="Sparsity", interactive=False)
                    code_hist = gr.Plot()
                
                with gr.Column():
                    gr.Markdown("#### Text")
                    with gr.Row():
                        text_mean = gr.Textbox(label="Mean", interactive=False)
                        text_std = gr.Textbox(label="Std", interactive=False)
                        text_sparse = gr.Textbox(label="Sparsity", interactive=False)
                    text_hist = gr.Plot()
            
            sparse_plot = gr.Plot(label="Sparsity Comparison")
            
            # 参数变化自动触发分析
            for component in [model_choice, code_input, text_input, layer_idx]:
                component.change(
                    fn=analyze_activations,
                    inputs=[model_choice, code_input, text_input, layer_idx],
                    outputs=[code_hist, text_hist, sparse_plot, 
                            code_mean, code_std, code_sparse,
                            text_mean, text_std, text_sparse]
                )
        
        # Tab 3: 架构对比
        with gr.Tab("架构对比"):
            arch_df = gr.Dataframe(value=render_architecture_comparison(), interactive=False)
            params_plot = gr.Plot(value=render_params_pie())
    
    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        return analyze_activations(default_model, default_code, default_text, default_layer)
    
    # 返回 load 事件需要的信息供主 app 调用
    return {
        'load_fn': on_load,
        'load_outputs': [
            code_hist, text_hist, sparse_plot,
            code_mean, code_std, code_sparse,
            text_mean, text_std, text_sparse
        ]
    }
