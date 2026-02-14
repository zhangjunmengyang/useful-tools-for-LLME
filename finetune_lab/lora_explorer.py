"""
LoRA Configuration Explorer - LoRA 配置探索器
支持实时计算参数量、显存占用，并可视化不同 rank 下的参数对比
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from finetune_lab.finetune_utils import (
    MODEL_CATEGORIES,
    TARGET_MODULES,
    LORA_RANKS,
    MEMORY_BUDGET_CONFIGS,
    get_model_config,
    calculate_lora_params,
    estimate_lora_memory,
    format_number
)


def get_model_list(category: str) -> list:
    """获取指定类别的模型列表"""
    if category not in MODEL_CATEGORIES:
        return []
    return [m[0] for m in MODEL_CATEGORIES[category]["models"]]


def calculate_and_visualize(
    category: str,
    model_name: str,
    rank: int,
    alpha: int,
    selected_modules: list,
    use_quantization: bool,
    quantization_bits: int
):
    """计算 LoRA 参数并生成可视化"""
    if not model_name or not selected_modules:
        return "Please select a model and at least one target module", None, None, None, "", "", "", "", "", "", ""

    # 获取模型配置
    config = get_model_config(category, model_name)
    if not config:
        return "Model configuration not found", None, None, None, "", "", "", "", "", "", ""

    # 计算 LoRA 参数
    result = calculate_lora_params(
        hidden_size=config["hidden"],
        num_layers=config["layers"],
        num_heads=config["heads"],
        num_kv_heads=config["kv_heads"],
        intermediate_size=config["ffn"],
        rank=rank,
        target_modules=selected_modules
    )

    # 计算可训练参数占比
    base_params = config["params"]
    trainable_ratio = result["total_params"] / base_params * 100

    # 计算显存占用
    memory = estimate_lora_memory(
        lora_params=result["total_params"],
        base_params=base_params,
        use_quantization=use_quantization,
        quantization_bits=quantization_bits
    )

    # 创建参数分布表格
    details_df = pd.DataFrame(result["details"])
    details_df["total_params"] = details_df["params_per_layer"] * config["layers"]
    details_df = details_df.rename(columns={
        "module": "Module",
        "dimension": "Dimension",
        "params_per_layer": "Params per Layer",
        "total_params": "Total Params"
    })
    details_df["Params per Layer"] = details_df["Params per Layer"].apply(lambda x: f"{x:,}")
    details_df["Total Params"] = details_df["Total Params"].apply(lambda x: f"{x:,}")

    # 生成不同 rank 对比图表
    rank_comparison_fig = create_rank_comparison_chart(
        config, selected_modules
    )

    # 生成显存分布图表
    memory_breakdown_fig = create_memory_breakdown_chart(memory)

    # 生成推荐配置表格
    recommendations_df = create_recommendations_table()

    # LoRA scaling factor
    scaling_factor = alpha / rank

    return (
        "",  # status message
        details_df,
        rank_comparison_fig,
        memory_breakdown_fig,
        recommendations_df,
        f"{result['total_params']:,}",
        f"{trainable_ratio:.4f}%",
        f"{scaling_factor:.2f}",
        f"{memory['base_model']:.2f} GB",
        f"{memory['lora_params'] + memory['optimizer'] + memory['gradients']:.2f} GB",
        f"{memory['total']:.2f} GB"
    )


def create_rank_comparison_chart(config: dict, selected_modules: list):
    """创建不同 rank 下的参数量对比图"""
    ranks = LORA_RANKS
    params_list = []

    for r in ranks:
        result = calculate_lora_params(
            hidden_size=config["hidden"],
            num_layers=config["layers"],
            num_heads=config["heads"],
            num_kv_heads=config["kv_heads"],
            intermediate_size=config["ffn"],
            rank=r,
            target_modules=selected_modules
        )
        params_list.append(result["total_params"])

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"r={r}" for r in ranks],
        y=[p / 1e6 for p in params_list],  # 转换为 M
        text=[format_number(p) for p in params_list],
        textposition='outside',
        marker=dict(
            color=[p / 1e6 for p in params_list],
            colorscale='Blues',
            showscale=False
        ),
        hovertemplate='<b>Rank %{x}</b><br>Params: %{text}<extra></extra>'
    ))

    fig.update_layout(
        title="LoRA Parameter Count vs Rank",
        xaxis_title="LoRA Rank",
        yaxis_title="Trainable Parameters (Million)",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        autosize=True,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        font=dict(family="Inter, sans-serif", size=12),
        hovermode='x unified'
    )

    return fig


def create_memory_breakdown_chart(memory: dict):
    """创建显存占用分布图"""
    labels = ['Base Model', 'LoRA Params', 'Optimizer', 'Gradients', 'Activations']
    values = [
        memory['base_model'],
        memory['lora_params'],
        memory['optimizer'],
        memory['gradients'],
        memory['activations']
    ]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        text=[f"{v:.2f} GB" for v in values],
        textposition='inside',
        hovertemplate='<b>%{label}</b><br>%{text}<br>%{percent}<extra></extra>',
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2))
    )])

    fig.update_layout(
        title="Memory Distribution (Training)",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        autosize=True,
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        font=dict(family="Inter, sans-serif", size=12),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )

    return fig


def create_recommendations_table():
    """创建推荐配置表格"""
    rows = []
    for budget, info in MEMORY_BUDGET_CONFIGS.items():
        for config in info["configs"]:
            rows.append({
                "Memory Budget": budget,
                "Model Size": config["model"],
                "Rank": config["rank"],
                "Target Modules": ", ".join(config["modules"]),
                "Batch Size": config["batch_size"],
                "Gradient Accumulation": config["grad_accum"]
            })

    return pd.DataFrame(rows)


def render():
    """渲染 LoRA Explorer 页面"""
    default_category = "LLaMA"
    default_model = "LLaMA-7B"
    default_rank = 16
    default_alpha = 32
    default_modules = ["q_proj", "v_proj"]

    gr.Markdown("# LoRA Configuration Explorer")
    gr.Markdown(
        "Explore LoRA configurations, calculate trainable parameters, and estimate memory requirements. "
        "Adjust the parameters below to see real-time updates."
    )

    with gr.Row():
        # 左侧：配置面板
        with gr.Column(scale=1):
            gr.Markdown("### Model Selection")

            category = gr.Dropdown(
                label="Model Family",
                choices=list(MODEL_CATEGORIES.keys()),
                value=default_category
            )

            model = gr.Dropdown(
                label="Model",
                choices=get_model_list(default_category),
                value=default_model
            )

            gr.Markdown("### LoRA Configuration")

            rank = gr.Slider(
                label="Rank (r)",
                minimum=4,
                maximum=128,
                value=default_rank,
                step=4,
                info="Higher rank = more trainable parameters"
            )

            alpha = gr.Slider(
                label="Alpha",
                minimum=8,
                maximum=256,
                value=default_alpha,
                step=8,
                info="Scaling factor = alpha / rank"
            )

            target_modules = gr.CheckboxGroup(
                label="Target Modules",
                choices=[(name, key) for key, name in TARGET_MODULES.items()],
                value=default_modules,
                info="Select which modules to apply LoRA"
            )

            gr.Markdown("### Quantization (QLoRA)")

            use_quantization = gr.Checkbox(
                label="Enable Quantization",
                value=False,
                info="Use QLoRA with quantized base model"
            )

            quantization_bits = gr.Radio(
                label="Quantization Bits",
                choices=[4, 8],
                value=4,
                info="4-bit or 8-bit quantization"
            )

        # 右侧：结果展示
        with gr.Column(scale=2):
            status = gr.Markdown("")

            gr.Markdown("### Parameter Statistics")

            with gr.Row():
                total_params = gr.Textbox(label="Total Trainable Params", interactive=False)
                trainable_ratio = gr.Textbox(label="Trainable Ratio", interactive=False)
                scaling_factor = gr.Textbox(label="Scaling Factor", interactive=False)

            gr.Markdown("### Parameter Distribution by Module")
            details_table = gr.Dataframe(
                label="Module Details",
                wrap=True
            )

            gr.Markdown("### Parameter Count vs Rank")
            rank_comparison_chart = gr.Plot()

            gr.Markdown("### Memory Estimation")

            with gr.Row():
                base_memory = gr.Textbox(label="Base Model", interactive=False)
                trainable_memory = gr.Textbox(label="Trainable (Params + Optimizer + Grad)", interactive=False)
                total_memory = gr.Textbox(label="Total Training Memory", interactive=False)

            memory_breakdown_chart = gr.Plot()

            gr.Markdown("### Recommended Configurations")
            gr.Markdown("Pre-configured LoRA setups optimized for different GPU memory budgets:")
            recommendations_table = gr.Dataframe(
                label="Configuration Presets",
                wrap=True
            )

    # 事件绑定
    def update_model_list(cat):
        models = get_model_list(cat)
        return gr.update(choices=models, value=models[0] if models else None)

    category.change(
        fn=update_model_list,
        inputs=[category],
        outputs=[model]
    )

    # 计算输入输出
    calc_inputs = [category, model, rank, alpha, target_modules, use_quantization, quantization_bits]
    calc_outputs = [
        status, details_table, rank_comparison_chart, memory_breakdown_chart,
        recommendations_table, total_params, trainable_ratio, scaling_factor,
        base_memory, trainable_memory, total_memory
    ]

    # 所有参数变化时自动计算
    for component in [model, rank, alpha, target_modules, use_quantization, quantization_bits]:
        component.change(
            fn=calculate_and_visualize,
            inputs=calc_inputs,
            outputs=calc_outputs
        )

    # 页面加载时计算默认值
    def on_load():
        return calculate_and_visualize(
            default_category, default_model, default_rank, default_alpha,
            default_modules, False, 4
        )

    return {
        'load_fn': on_load,
        'load_outputs': calc_outputs
    }
