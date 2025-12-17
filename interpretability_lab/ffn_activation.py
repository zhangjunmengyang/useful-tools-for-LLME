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
from model_lab.model_utils import extract_from_url


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
        title="Activation Function Comparison",
        xaxis_title="Input x",
        yaxis_title="Output f(x)",
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
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Zero")

    fig.update_layout(
        title=title,
        xaxis_title="Activation Value",
        yaxis_title="Frequency",
        height=400,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_sparsity_comparison(code_values, text_values, thresholds) -> go.Figure:
    """渲染稀疏性对比"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Code', 'Natural Language'])
    
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


def get_model_id_from_ui(mode, preset_choice, custom_url):
    """从 UI 输入解析模型 ID"""
    if mode == "Preset Model":
        return INTERPRETABILITY_MODELS[preset_choice]['id']
    else:
        return extract_from_url(custom_url) if custom_url else None


def analyze_activations(mode, preset_choice, custom_url, token, code_input, text_input, layer_idx):
    """分析激活"""
    model_id = get_model_id_from_ui(mode, preset_choice, custom_url)
    if not model_id:
        return None, None, None, "", "", "", "", "", ""

    # 加载模型
    if _loaded_model["name"] != model_id:
        model, tokenizer = load_model_with_attention(model_id, token=token if token else None)
        if model is None:
            return None, None, None, "", "", "", "", "", ""
        _loaded_model["name"] = model_id
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
        code_hist = render_activation_histogram(fc1_data['values'], "Code Activation Distribution")
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
        text_hist = render_activation_histogram(fc1_data['values'], "Natural Language Activation Distribution")
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
            "Model": model_name,
            "Attention": arch['attention'],
            "FFN Activation": arch['ffn'],
            "Normalization": arch['norm'],
            "Position Encoding": arch['position']
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
        title="Transformer Parameter Distribution (Typical Decoder-only Model)",
        height=450,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render():
    """渲染页面"""

    # 默认值
    default_model = list(INTERPRETABILITY_MODELS.keys())[0]
    default_code = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    default_text = "The sun rises in the east and sets in the west. Birds fly south for the winter."
    default_layer = 0
    
    with gr.Tabs():
        # Tab 1: Activation Functions
        with gr.Tab("Activation Functions") as activation_func_tab:
            activation_plot = gr.Plot(value=render_activation_curves())
            swiglu_plot = gr.Plot(value=render_swiglu_visualization())

        # Tab 2: Activation Analysis
        with gr.Tab("Activation Analysis") as activation_analysis_tab:
            with gr.Row():
                model_mode = gr.Radio(
                    label="Input Method",
                    choices=["Preset Model", "Custom Model"],
                    value="Preset Model"
                )

            with gr.Row():
                preset_model = gr.Dropdown(
                    choices=list(INTERPRETABILITY_MODELS.keys()),
                    value=default_model,
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

            with gr.Row():
                code_input = gr.Textbox(
                    label="Code Input",
                    value=default_code,
                    lines=5
                )
                text_input = gr.Textbox(
                    label="Text Input",
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
            for component in [model_mode, preset_model, custom_model, hf_token, code_input, text_input, layer_idx]:
                component.change(
                    fn=analyze_activations,
                    inputs=[model_mode, preset_model, custom_model, hf_token, code_input, text_input, layer_idx],
                    outputs=[code_hist, text_hist, sparse_plot,
                            code_mean, code_std, code_sparse,
                            text_mean, text_std, text_sparse]
                )
        
        # Tab 3: Architecture Comparison
        with gr.Tab("Architecture Comparison") as arch_tab:
            arch_df = gr.Dataframe(value=render_architecture_comparison(), interactive=False)
            params_plot = gr.Plot(value=render_params_pie())

    # Re-render plots when tabs become visible to fix width issues
    activation_func_tab.select(
        fn=lambda: (render_activation_curves(), render_swiglu_visualization()),
        outputs=[activation_plot, swiglu_plot]
    )

    activation_analysis_tab.select(
        fn=analyze_activations,
        inputs=[model_mode, preset_model, custom_model, hf_token, code_input, text_input, layer_idx],
        outputs=[code_hist, text_hist, sparse_plot,
                 code_mean, code_std, code_sparse,
                 text_mean, text_std, text_sparse]
    )

    arch_tab.select(
        fn=lambda: render_params_pie(),
        outputs=[params_plot]
    )

    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        return analyze_activations("Preset Model", default_model, "", None, default_code, default_text, default_layer)
    
    # 返回 load 事件需要的信息供主 app 调用
    return {
        'load_fn': on_load,
        'load_outputs': [
            code_hist, text_hist, sparse_plot,
            code_mean, code_std, code_sparse,
            text_mean, text_std, text_sparse
        ]
    }
