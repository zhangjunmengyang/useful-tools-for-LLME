"""
KV Cache 模拟器 - 可视化推理过程中的显存占用
"""

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from generation_lab.generation_utils import (
    MODEL_CONFIGS,
    calculate_kv_cache_size,
    simulate_kv_cache_growth,
    simulate_paged_attention,
    format_bytes
)
from model_lab.model_utils import extract_from_url


def get_model_config_from_ui(mode, preset_choice, custom_url, token=None):
    """从 UI 输入获取模型配置"""
    if mode == "Preset Model":
        return MODEL_CONFIGS.get(preset_choice)
    else:
        # 从 HuggingFace 获取自定义模型配置
        model_id = extract_from_url(custom_url) if custom_url else None
        if not model_id:
            return None
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_id, token=token if token else None)
            # 转换为我们需要的格式
            return {
                'num_hidden_layers': getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 32)),
                'hidden_size': getattr(config, 'hidden_size', getattr(config, 'n_embd', 4096)),
                'num_attention_heads': getattr(config, 'num_attention_heads', getattr(config, 'n_head', 32))
            }
        except Exception as e:
            print(f"Error loading config: {e}")
            return None


def render_kv_cache_growth_chart(growth_data: list) -> go.Figure:
    """渲染 KV Cache 增长曲线"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("KV Cache Cumulative Memory", "Per-Step Increment"),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    steps = [d['step'] for d in growth_data]
    cache_gb = [d['cache_gb'] for d in growth_data]
    delta_gb = [d['delta_gb'] * 1000 for d in growth_data]  # 转为 MB
    phases = [d['phase'] for d in growth_data]
    
    # 颜色映射
    colors = ['#DC2626' if p == 'Prefill' else '#2563EB' for p in phases]
    
    # 累积曲线
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=cache_gb,
            mode='lines+markers',
            name='Cumulative Memory',
            line=dict(color='#2563EB', width=3),
            marker=dict(size=8, color=colors),
            fill='tozeroy',
            fillcolor='rgba(37, 99, 235, 0.1)'
        ),
        row=1, col=1
    )
    
    # 标注 Prefill 点
    prefill_idx = 0
    fig.add_annotation(
        x=steps[prefill_idx],
        y=cache_gb[prefill_idx],
        text=f"Prefill<br>{cache_gb[prefill_idx]:.3f} GB",
        showarrow=True,
        arrowhead=2,
        ax=40,
        ay=-40,
        font=dict(color='#DC2626'),
        row=1, col=1
    )
    
    # 增量柱状图
    fig.add_trace(
        go.Bar(
            x=steps,
            y=delta_gb,
            name='Per-Step Increment',
            marker_color=colors,
            text=[f'{d:.2f}' for d in delta_gb],
            textposition='outside'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=600,
        autosize=True,
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=40),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    fig.update_yaxes(title_text="Memory (GB)", row=1, col=1)
    fig.update_yaxes(title_text="Increment (MB)", row=2, col=1)
    fig.update_xaxes(title_text="Generation Step", row=2, col=1)
    
    return fig


def render_paged_attention_viz(paged_data: dict) -> go.Figure:
    """渲染 PagedAttention Block 分配图"""
    sequences = paged_data['sequences']
    block_size = paged_data['block_size']
    
    # 创建 block 网格
    total_blocks = paged_data['total_blocks']
    cols = 16
    rows = (total_blocks + cols - 1) // cols
    
    # 初始化网格
    grid = np.zeros((rows, cols))
    
    colors = px.colors.qualitative.Set2
    block_idx = 0
    
    for seq in sequences:
        for _ in range(seq['blocks']):
            if block_idx < total_blocks:
                row = block_idx // cols
                col = block_idx % cols
                grid[row, col] = seq['seq_id'] + 1
                block_idx += 1
    
    # 创建热力图
    fig = go.Figure()
    
    # 自定义颜色映射
    colorscale = [
        [0, '#E5E7EB'],
    ]
    for i in range(len(sequences)):
        val = (i + 1) / (len(sequences) + 1)
        colorscale.append([val, colors[i % len(colors)]])
    colorscale.append([1, colors[-1]])
    
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale=colorscale,
        showscale=False,
        hovertemplate='Block %{x},%{y}<br>Sequence: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"PagedAttention Block Allocation (Block Size = {block_size})",
        xaxis=dict(title="Column", showgrid=False),
        yaxis=dict(title="Row", showgrid=False, autorange='reversed'),
        height=350,
        autosize=True,
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def render_utilization_chart(paged_data: dict) -> go.Figure:
    """渲染利用率图表"""
    sequences = paged_data['sequences']
    
    fig = go.Figure()
    
    seq_ids = [f"Seq {s['seq_id']}" for s in sequences]
    utilizations = [s['utilization'] for s in sequences]
    
    fig.add_trace(go.Bar(
        x=seq_ids,
        y=utilizations,
        marker_color=['#2563EB' if u > 80 else '#D97706' if u > 50 else '#DC2626' 
                      for u in utilizations],
        text=[f'{u:.1f}%' for u in utilizations],
        textposition='outside'
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="#059669",
                  annotation_text="Ideal Utilization")

    fig.update_layout(
        title="Per-Sequence Block Utilization",
        xaxis_title="Sequence",
        yaxis_title="Utilization (%)",
        yaxis_range=[0, 110],
        height=350,
        autosize=True,
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF'
    )
    
    return fig


def calculate_kv_cache(model_choice, custom_layers, custom_hidden, custom_heads, 
                       seq_length, batch_size, dtype_choice):
    """计算 KV Cache"""
    dtype_map = {"float16/bfloat16": 2, "float32": 4, "int8": 1}
    dtype_bytes = dtype_map[dtype_choice]
    
    if model_choice == "Custom":
        num_layers = custom_layers
        hidden_size = custom_hidden
        num_heads = custom_heads
        config_info = f"Custom Config: Layers={num_layers}, Hidden={hidden_size}, Heads={num_heads}"
    else:
        config = MODEL_CONFIGS[model_choice]
        num_layers = config['num_hidden_layers']
        hidden_size = config['hidden_size']
        num_heads = config['num_attention_heads']
        config_info = f"{model_choice}: Layers={num_layers}, Hidden={hidden_size}, Heads={num_heads}"
    
    result = calculate_kv_cache_size(
        num_layers, hidden_size, num_heads,
        seq_length, batch_size, dtype_bytes
    )
    
    # 详细分解表格
    breakdown_data = pd.DataFrame({
        "Component": ["K Cache", "V Cache", "Per-Layer KV", "Total"],
        "Formula": [
            f"{num_layers} x {batch_size} x {seq_length} x {hidden_size} x {dtype_bytes}",
            f"{num_layers} x {batch_size} x {seq_length} x {hidden_size} x {dtype_bytes}",
            f"2 x {batch_size} x {seq_length} x {hidden_size} x {dtype_bytes}",
            f"2 x {num_layers} x {batch_size} x {seq_length} x {hidden_size} x {dtype_bytes}"
        ],
        "Size": [
            format_bytes(result['k_cache_bytes']),
            format_bytes(result['v_cache_bytes']),
            format_bytes(result['per_layer_bytes']),
            format_bytes(result['total_bytes'])
        ]
    })
    
    return (
        f"{result['total_gb']:.3f} GB",
        f"{result['per_layer_mb']:.2f} MB",
        format_bytes(result['k_cache_bytes']),
        format_bytes(result['v_cache_bytes']),
        config_info,
        breakdown_data
    )


def simulate_growth(mode, preset_choice, custom_url, token, prompt_len, gen_len):
    """模拟 KV Cache 增长"""
    config = get_model_config_from_ui(mode, preset_choice, custom_url, token)
    if not config:
        return "", "", "", None, None
    
    growth_data = simulate_kv_cache_growth(
        config, prompt_len, gen_len,
        batch_size=1, dtype_bytes=2
    )
    
    final_cache = growth_data[-1]['cache_gb']
    prefill_cache = growth_data[0]['cache_gb']
    decode_delta = (final_cache - prefill_cache) * 1000
    
    fig = render_kv_cache_growth_chart(growth_data)
    
    df = pd.DataFrame([{
        "Step": d['step'],
        "Phase": d['phase'],
        "Sequence Length": d['seq_length'],
        "Cumulative Memory (GB)": f"{d['cache_gb']:.4f}",
        "Increment (MB)": f"{d['delta_gb'] * 1000:.2f}",
        "Description": d['description']
    } for d in growth_data])
    
    return (
        f"{prefill_cache:.3f} GB",
        f"{final_cache:.3f} GB",
        f"{decode_delta:.1f} MB",
        fig,
        df
    )


def simulate_paged(block_size, num_seqs, avg_tokens):
    """模拟 PagedAttention"""
    paged_data = simulate_paged_attention(avg_tokens, block_size, num_seqs)
    
    fig_blocks = render_paged_attention_viz(paged_data)
    fig_util = render_utilization_chart(paged_data)
    
    seq_df = pd.DataFrame([{
        "Sequence ID": s['seq_id'],
        "Token Count": s['length'],
        "Block Count": s['blocks'],
        "Waste (tokens)": s['waste'],
        "Utilization": f"{s['utilization']:.1f}%"
    } for s in paged_data['sequences']])

    analysis = f"""
### Internal Fragmentation Analysis

- **Block Size**: {block_size} tokens
- **Total Waste**: {paged_data['total_waste']} tokens ({100 - paged_data['overall_utilization']:.1f}%)
- **Recommendation**: Smaller block sizes reduce fragmentation but increase management overhead
"""
    
    return (
        str(paged_data['total_blocks']),
        f"{paged_data['total_capacity']} tokens",
        f"{paged_data['total_tokens']} tokens",
        f"{paged_data['overall_utilization']:.1f}%",
        fig_blocks,
        fig_util,
        seq_df,
        analysis
    )


def render():
    """渲染页面"""

    with gr.Tabs():
        # Tab 1: Memory Calculation
        with gr.Tab("Memory Calculator"):
            gr.Markdown("### KV Cache Memory Calculator")

            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        choices=["Custom"] + list(MODEL_CONFIGS.keys()),
                        value="Llama-2-7B",
                        label="Select Preset Model"
                    )

                    custom_layers = gr.Number(label="Layers", value=32, visible=False)
                    custom_hidden = gr.Number(label="Hidden Size", value=4096, visible=False)
                    custom_heads = gr.Number(label="Attention Heads", value=32, visible=False)

                    gr.Markdown("---")

                    seq_length = gr.Number(label="Sequence Length", value=2048, minimum=1, maximum=131072)
                    batch_size = gr.Number(label="Batch Size", value=1, minimum=1, maximum=256)
                    dtype_choice = gr.Dropdown(
                        choices=["float16/bfloat16", "float32", "int8"],
                        value="float16/bfloat16",
                        label="Data Type"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### Calculation Results")

                    config_info = gr.Textbox(label="Model Config", interactive=False)

                    with gr.Row():
                        total_kv = gr.Textbox(label="Total KV Cache", interactive=False)
                        per_layer = gr.Textbox(label="Per-Layer", interactive=False)
                        k_cache = gr.Textbox(label="K Cache", interactive=False)
                        v_cache = gr.Textbox(label="V Cache", interactive=False)

                    gr.Markdown("#### Detailed Breakdown")
                    breakdown_df = gr.Dataframe(interactive=False)

            def toggle_custom(choice):
                is_custom = choice == "Custom"
                return (
                    gr.update(visible=is_custom),
                    gr.update(visible=is_custom),
                    gr.update(visible=is_custom)
                )

            def toggle_and_calc(choice, layers, hidden, heads, seq_len, batch, dtype):
                is_custom = choice == "Custom"
                result = calculate_kv_cache(choice, layers, hidden, heads, seq_len, batch, dtype)
                return (
                    gr.update(visible=is_custom),
                    gr.update(visible=is_custom),
                    gr.update(visible=is_custom),
                    *result
                )
            
            # 所有参数变化自动触发计算
            for component in [model_choice, custom_layers, custom_hidden, custom_heads, 
                             seq_length, batch_size, dtype_choice]:
                component.change(
                    fn=toggle_and_calc,
                inputs=[model_choice, custom_layers, custom_hidden, custom_heads,
                       seq_length, batch_size, dtype_choice],
                    outputs=[custom_layers, custom_hidden, custom_heads,
                            total_kv, per_layer, k_cache, v_cache, config_info, breakdown_df]
            )
        
        # Tab 2: Growth Simulation
        with gr.Tab("Growth Simulation") as growth_tab:
            with gr.Row():
                sim_model_mode = gr.Radio(
                    label="Input Method",
                    choices=["Preset Model", "Custom Model"],
                    value="Preset Model"
                )

            with gr.Row():
                sim_preset = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="Llama-2-7B",
                    label="Select Model"
                )

                sim_custom = gr.Textbox(
                    label="Model Name or URL",
                    placeholder="e.g., meta-llama/Llama-2-7b-hf",
                    visible=False
                )

            sim_token = gr.Textbox(
                label="HF Token (Optional)",
                type="password",
                placeholder="For private models",
                visible=False
            )

            with gr.Row():
                prompt_len = gr.Slider(label="Prompt Length", value=512, minimum=1, maximum=8192, step=1)
                gen_len = gr.Slider(label="Generation Length", value=128, minimum=1, maximum=2048, step=1)

            with gr.Row():
                prefill_metric = gr.Textbox(label="After Prefill", interactive=False)
                final_metric = gr.Textbox(label="Final Memory", interactive=False)
                decode_metric = gr.Textbox(label="Decode Increment", interactive=False)

            growth_chart = gr.Plot(label="KV Cache Growth Curve")

            with gr.Accordion("Detailed Data", open=False):
                growth_df = gr.Dataframe(interactive=False)

            # Toggle函数：切换预设/自定义模型模式
            def toggle_sim_model_mode(mode):
                return (
                    gr.update(visible=(mode == "Preset Model")),
                    gr.update(visible=(mode == "Custom Model")),
                    gr.update(visible=(mode == "Custom Model"))
                )

            sim_model_mode.change(
                fn=toggle_sim_model_mode,
                inputs=[sim_model_mode],
                outputs=[sim_preset, sim_custom, sim_token]
            )

            # 参数变化自动触发模拟
            for component in [sim_model_mode, sim_preset, sim_custom, sim_token, prompt_len, gen_len]:
                component.change(
                fn=simulate_growth,
                inputs=[sim_model_mode, sim_preset, sim_custom, sim_token, prompt_len, gen_len],
                outputs=[prefill_metric, final_metric, decode_metric, growth_chart, growth_df]
            )
        
        # Tab 3: PagedAttention
        with gr.Tab("PagedAttention") as paged_tab:
            with gr.Row():
                block_size = gr.Dropdown(
                    choices=[8, 16, 32, 64],
                    value=16,
                    label="Block Size"
                )
                num_seqs = gr.Slider(
                    label="Concurrent Sequences",
                    minimum=2,
                    maximum=8,
                    value=4,
                    step=1
                )
                avg_tokens = gr.Slider(
                    label="Average Token Count",
                    value=256,
                    minimum=32,
                    maximum=1024,
                    step=16
                )

            with gr.Row():
                total_blocks = gr.Textbox(label="Total Blocks", interactive=False)
                total_capacity = gr.Textbox(label="Total Capacity", interactive=False)
                total_used = gr.Textbox(label="Actual Usage", interactive=False)
                overall_util = gr.Textbox(label="Overall Utilization", interactive=False)

            blocks_chart = gr.Plot(label="Block Allocation")
            util_chart = gr.Plot(label="Utilization")

            with gr.Accordion("Sequence Details", open=False):
                seq_df = gr.Dataframe(interactive=False)
            
            analysis_md = gr.Markdown("")
            
            # 参数变化自动触发模拟
            for component in [block_size, num_seqs, avg_tokens]:
                component.change(
                    fn=simulate_paged,
                    inputs=[block_size, num_seqs, avg_tokens],
                    outputs=[total_blocks, total_capacity, total_used, overall_util,
                            blocks_chart, util_chart, seq_df, analysis_md]
                )

    # Re-render plots when tabs become visible to fix width issues
    growth_tab.select(
        fn=simulate_growth,
        inputs=[sim_model_mode, sim_preset, sim_custom, sim_token, prompt_len, gen_len],
        outputs=[prefill_metric, final_metric, decode_metric, growth_chart, growth_df]
    )

    paged_tab.select(
        fn=simulate_paged,
        inputs=[block_size, num_seqs, avg_tokens],
        outputs=[total_blocks, total_capacity, total_used, overall_util,
                blocks_chart, util_chart, seq_df, analysis_md]
    )

    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        # 显存计算默认值
        kv_result = calculate_kv_cache("Llama-2-7B", 32, 4096, 32, 2048, 1, "float16/bfloat16")
        # 增长模拟默认值
        growth_result = simulate_growth("Preset Model", "Llama-2-7B", "", None, 512, 128)
        # PagedAttention 默认值
        paged_result = simulate_paged(16, 4, 256)
        
        return (
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),  # custom fields
            kv_result[0], kv_result[1], kv_result[2], kv_result[3], kv_result[4], kv_result[5],  # KV Cache
            growth_result[0], growth_result[1], growth_result[2], growth_result[3], growth_result[4],  # Growth
            paged_result[0], paged_result[1], paged_result[2], paged_result[3],  # Paged metrics
            paged_result[4], paged_result[5], paged_result[6], paged_result[7]  # Paged charts
        )
    
    # 返回 load 事件信息
    return {
        'load_fn': on_load,
        'load_outputs': [
            custom_layers, custom_hidden, custom_heads,
            total_kv, per_layer, k_cache, v_cache, config_info, breakdown_df,
            prefill_metric, final_metric, decode_metric, growth_chart, growth_df,
            total_blocks, total_capacity, total_used, overall_util,
            blocks_chart, util_chart, seq_df, analysis_md
        ]
    }
