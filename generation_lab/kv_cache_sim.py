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


def render_kv_cache_growth_chart(growth_data: list) -> go.Figure:
    """渲染 KV Cache 增长曲线"""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("KV Cache 累积显存", "每步增量"),
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
            name='累积显存',
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
            name='每步增量',
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
    
    fig.update_yaxes(title_text="显存 (GB)", row=1, col=1)
    fig.update_yaxes(title_text="增量 (MB)", row=2, col=1)
    fig.update_xaxes(title_text="生成步骤", row=2, col=1)
    
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
        title=f"PagedAttention Block 分配 (Block Size = {block_size})",
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
                  annotation_text="理想利用率")
    
    fig.update_layout(
        title="各序列 Block 利用率",
        xaxis_title="序列",
        yaxis_title="利用率 (%)",
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
    
    if model_choice == "自定义":
        num_layers = custom_layers
        hidden_size = custom_hidden
        num_heads = custom_heads
        config_info = f"自定义配置: Layers={num_layers}, Hidden={hidden_size}, Heads={num_heads}"
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
        "组件": ["K Cache", "V Cache", "单层 KV", "总计"],
        "公式": [
            f"{num_layers} x {batch_size} x {seq_length} x {hidden_size} x {dtype_bytes}",
            f"{num_layers} x {batch_size} x {seq_length} x {hidden_size} x {dtype_bytes}",
            f"2 x {batch_size} x {seq_length} x {hidden_size} x {dtype_bytes}",
            f"2 x {num_layers} x {batch_size} x {seq_length} x {hidden_size} x {dtype_bytes}"
        ],
        "大小": [
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


def simulate_growth(sim_model, prompt_len, gen_len):
    """模拟 KV Cache 增长"""
    config = MODEL_CONFIGS[sim_model]
    
    growth_data = simulate_kv_cache_growth(
        config, prompt_len, gen_len,
        batch_size=1, dtype_bytes=2
    )
    
    final_cache = growth_data[-1]['cache_gb']
    prefill_cache = growth_data[0]['cache_gb']
    decode_delta = (final_cache - prefill_cache) * 1000
    
    fig = render_kv_cache_growth_chart(growth_data)
    
    df = pd.DataFrame([{
        "步骤": d['step'],
        "阶段": d['phase'],
        "序列长度": d['seq_length'],
        "累积显存 (GB)": f"{d['cache_gb']:.4f}",
        "增量 (MB)": f"{d['delta_gb'] * 1000:.2f}",
        "说明": d['description']
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
        "序列 ID": s['seq_id'],
        "Token 数": s['length'],
        "Block 数": s['blocks'],
        "浪费 (tokens)": s['waste'],
        "利用率": f"{s['utilization']:.1f}%"
    } for s in paged_data['sequences']])
    
    analysis = f"""
### 内部碎片分析

- **Block Size**: {block_size} tokens
- **总浪费**: {paged_data['total_waste']} tokens ({100 - paged_data['overall_utilization']:.1f}%)
- **建议**: Block Size 越小，碎片越少，但管理开销越大
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
    
    gr.Markdown("## KV Cache 模拟器")
    
    with gr.Tabs():
        # Tab 1: 显存计算
        with gr.Tab("显存计算"):
            gr.Markdown("### KV Cache 显存计算器")
            
            with gr.Row():
                with gr.Column(scale=1):
                    model_choice = gr.Dropdown(
                        choices=["自定义"] + list(MODEL_CONFIGS.keys()),
                        value="Llama-2-7B",
                        label="选择预设模型"
                    )
                    
                    custom_layers = gr.Number(label="层数", value=32, visible=False)
                    custom_hidden = gr.Number(label="Hidden Size", value=4096, visible=False)
                    custom_heads = gr.Number(label="注意力头数", value=32, visible=False)
                    
                    gr.Markdown("---")
                    
                    seq_length = gr.Number(label="序列长度", value=2048, minimum=1, maximum=131072)
                    batch_size = gr.Number(label="Batch Size", value=1, minimum=1, maximum=256)
                    dtype_choice = gr.Dropdown(
                        choices=["float16/bfloat16", "float32", "int8"],
                        value="float16/bfloat16",
                        label="数据类型"
                    )
                    
                with gr.Column(scale=2):
                    gr.Markdown("### 计算结果")
                    
                    config_info = gr.Textbox(label="模型配置", interactive=False)
                    
                    with gr.Row():
                        total_kv = gr.Textbox(label="总 KV Cache", interactive=False)
                        per_layer = gr.Textbox(label="每层占用", interactive=False)
                        k_cache = gr.Textbox(label="K Cache", interactive=False)
                        v_cache = gr.Textbox(label="V Cache", interactive=False)
                    
                    gr.Markdown("#### 详细分解")
                    breakdown_df = gr.Dataframe(interactive=False)
            
            def toggle_custom(choice):
                is_custom = choice == "自定义"
                return (
                    gr.update(visible=is_custom),
                    gr.update(visible=is_custom),
                    gr.update(visible=is_custom)
                )
            
            def toggle_and_calc(choice, layers, hidden, heads, seq_len, batch, dtype):
                is_custom = choice == "自定义"
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
        
        # Tab 2: 增长模拟
        with gr.Tab("增长模拟"):
            with gr.Row():
                sim_model = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="Llama-2-7B",
                    label="选择模型"
                )
                prompt_len = gr.Slider(label="Prompt 长度", value=512, minimum=1, maximum=8192, step=1)
                gen_len = gr.Slider(label="生成长度", value=128, minimum=1, maximum=2048, step=1)
            
            with gr.Row():
                prefill_metric = gr.Textbox(label="Prefill 后", interactive=False)
                final_metric = gr.Textbox(label="最终显存", interactive=False)
                decode_metric = gr.Textbox(label="Decode 增量", interactive=False)
            
            growth_chart = gr.Plot(label="KV Cache 增长曲线")
            
            with gr.Accordion("详细数据", open=False):
                growth_df = gr.Dataframe(interactive=False)
            
            # 参数变化自动触发模拟
            for component in [sim_model, prompt_len, gen_len]:
                component.change(
                fn=simulate_growth,
                inputs=[sim_model, prompt_len, gen_len],
                outputs=[prefill_metric, final_metric, decode_metric, growth_chart, growth_df]
            )
        
        # Tab 3: PagedAttention
        with gr.Tab("PagedAttention"):
            with gr.Row():
                block_size = gr.Dropdown(
                    choices=[8, 16, 32, 64],
                    value=16,
                    label="Block Size"
                )
                num_seqs = gr.Slider(
                    label="并发序列数",
                    minimum=2,
                    maximum=8,
                    value=4,
                    step=1
                )
                avg_tokens = gr.Slider(
                    label="平均 Token 数",
                    value=256,
                    minimum=32,
                    maximum=1024,
                    step=16
                )
            
            with gr.Row():
                total_blocks = gr.Textbox(label="总 Block 数", interactive=False)
                total_capacity = gr.Textbox(label="总容量", interactive=False)
                total_used = gr.Textbox(label="实际使用", interactive=False)
                overall_util = gr.Textbox(label="整体利用率", interactive=False)
            
            blocks_chart = gr.Plot(label="Block 分配")
            util_chart = gr.Plot(label="利用率")
            
            with gr.Accordion("各序列详情", open=False):
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
    
    # 初始化加载函数
    def on_load():
        """页面加载时计算默认值"""
        # 显存计算默认值
        kv_result = calculate_kv_cache("Llama-2-7B", 32, 4096, 32, 2048, 1, "float16/bfloat16")
        # 增长模拟默认值
        growth_result = simulate_growth("Llama-2-7B", 512, 128)
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
