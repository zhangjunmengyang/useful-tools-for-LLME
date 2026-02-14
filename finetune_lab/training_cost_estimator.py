"""
Training Cost Estimator - 训练成本估算器
估算全参微调和 LoRA 微调的训练时间和成本
"""

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
from finetune_lab.finetune_utils import (
    MODEL_CATEGORIES,
    GPU_SPECS,
    TARGET_MODULES,
    get_model_config,
    calculate_lora_params,
    calculate_training_flops,
    estimate_training_time,
    estimate_training_cost,
    format_number
)


def get_model_list(category: str) -> list:
    """获取指定类别的模型列表"""
    if category not in MODEL_CATEGORIES:
        return []
    return [m[0] for m in MODEL_CATEGORIES[category]["models"]]


def estimate_cost(
    category: str,
    model_name: str,
    training_mode: str,
    lora_rank: int,
    lora_modules: list,
    dataset_size_tokens: float,
    gpu_type: str,
    num_gpus: int,
    mfu: float
):
    """估算训练成本"""
    if not model_name or not gpu_type:
        return "Please select a model and GPU type", None, None, "", "", "", "", "", ""

    # 获取模型配置
    config = get_model_config(category, model_name)
    if not config:
        return "Model configuration not found", None, None, "", "", "", "", "", ""

    # 获取 GPU 规格
    gpu_spec = GPU_SPECS.get(gpu_type)
    if not gpu_spec:
        return "GPU specification not found", None, None, "", "", "", "", "", ""

    base_params = config["params"]

    # 计算可训练参数量
    if training_mode == "Full Fine-tuning":
        trainable_params = base_params
    else:  # LoRA
        if not lora_modules:
            return "Please select at least one target module for LoRA", None, None, "", "", "", "", "", ""

        result = calculate_lora_params(
            hidden_size=config["hidden"],
            num_layers=config["layers"],
            num_heads=config["heads"],
            num_kv_heads=config["kv_heads"],
            intermediate_size=config["ffn"],
            rank=lora_rank,
            target_modules=lora_modules
        )
        trainable_params = result["total_params"]

    # 转换 dataset_size 为 tokens (B -> actual number)
    total_tokens = dataset_size_tokens * 1e9

    # 计算 FLOPs
    total_flops = calculate_training_flops(
        model_params=trainable_params,
        tokens=total_tokens,
        is_full_finetune=(training_mode == "Full Fine-tuning")
    )

    # 估算训练时间
    training_hours = estimate_training_time(
        total_flops=total_flops,
        gpu_tflops=gpu_spec["tflops_fp16"],
        num_gpus=num_gpus,
        mfu=mfu
    )

    # 估算成本
    total_cost = estimate_training_cost(
        training_hours=training_hours,
        cost_per_hour=gpu_spec["cost_per_hour"],
        num_gpus=num_gpus
    )

    # 生成数据量扫描曲线
    cost_curve_fig = create_cost_curve(
        trainable_params, gpu_spec, num_gpus, mfu
    )

    # 生成 GPU 对比图表
    gpu_comparison_fig = create_gpu_comparison(
        trainable_params, total_tokens, num_gpus, mfu
    )

    # 计算可训练参数占比
    trainable_ratio = trainable_params / base_params * 100

    return (
        "",  # status
        cost_curve_fig,
        gpu_comparison_fig,
        f"{format_number(trainable_params)} ({trainable_ratio:.3f}%)",
        f"{total_flops:.2e}",
        f"{training_hours:.2f} hours ({training_hours/24:.2f} days)",
        f"${total_cost:.2f}",
        f"${total_cost/training_hours:.2f}/hr" if training_hours > 0 else "N/A",
        f"{gpu_spec['memory_gb']} GB x {num_gpus}"
    )


def create_cost_curve(trainable_params: int, gpu_spec: dict, num_gpus: int, mfu: float):
    """生成成本随数据量变化的曲线"""
    dataset_sizes = [0.5, 1, 2, 5, 10, 20, 50, 100]  # B tokens
    costs = []

    for size_b in dataset_sizes:
        total_tokens = size_b * 1e9
        total_flops = calculate_training_flops(trainable_params, total_tokens)
        hours = estimate_training_time(total_flops, gpu_spec["tflops_fp16"], num_gpus, mfu)
        cost = estimate_training_cost(hours, gpu_spec["cost_per_hour"], num_gpus)
        costs.append(cost)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dataset_sizes,
        y=costs,
        mode='lines+markers',
        name='Training Cost',
        line=dict(color='#4ECDC4', width=3),
        marker=dict(size=8, color='#4ECDC4', line=dict(color='#FFFFFF', width=2)),
        hovertemplate='<b>Dataset Size: %{x}B tokens</b><br>Cost: $%{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title="Training Cost vs Dataset Size",
        xaxis_title="Dataset Size (Billion Tokens)",
        yaxis_title="Total Cost (USD)",
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        autosize=True,
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
        font=dict(family="Inter, sans-serif", size=12),
        hovermode='x unified',
        xaxis=dict(
            gridcolor='#E8E8E8',
            type='log' if max(dataset_sizes) > 20 else 'linear'
        ),
        yaxis=dict(gridcolor='#E8E8E8')
    )

    return fig


def create_gpu_comparison(trainable_params: int, total_tokens: float, num_gpus: int, mfu: float):
    """生成不同 GPU 的成本对比"""
    gpu_names = list(GPU_SPECS.keys())
    costs = []
    times = []

    total_flops = calculate_training_flops(trainable_params, total_tokens)

    for gpu_name in gpu_names:
        gpu_spec = GPU_SPECS[gpu_name]
        hours = estimate_training_time(total_flops, gpu_spec["tflops_fp16"], num_gpus, mfu)
        cost = estimate_training_cost(hours, gpu_spec["cost_per_hour"], num_gpus)
        costs.append(cost)
        times.append(hours)

    # 简化 GPU 名称用于显示
    display_names = [name.replace("NVIDIA ", "") for name in gpu_names]

    fig = go.Figure()

    # 成本柱状图
    fig.add_trace(go.Bar(
        name='Cost',
        x=display_names,
        y=costs,
        text=[f"${c:.2f}" for c in costs],
        textposition='outside',
        marker=dict(color='#FF6B6B'),
        yaxis='y',
        hovertemplate='<b>%{x}</b><br>Cost: $%{y:.2f}<extra></extra>'
    ))

    # 时间柱状图（第二 y 轴）
    fig.add_trace(go.Bar(
        name='Time',
        x=display_names,
        y=times,
        text=[f"{t:.1f}h" for t in times],
        textposition='outside',
        marker=dict(color='#4ECDC4'),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>Time: %{y:.2f} hours<extra></extra>'
    ))

    fig.update_layout(
        title="GPU Comparison (Cost vs Time)",
        xaxis_title="GPU Type",
        yaxis=dict(
            title="Cost (USD)",
            titlefont=dict(color="#FF6B6B"),
            tickfont=dict(color="#FF6B6B"),
            gridcolor='#E8E8E8'
        ),
        yaxis2=dict(
            title="Training Time (Hours)",
            titlefont=dict(color="#4ECDC4"),
            tickfont=dict(color="#4ECDC4"),
            overlaying='y',
            side='right'
        ),
        plot_bgcolor='#FFFFFF',
        paper_bgcolor='#FFFFFF',
        autosize=True,
        height=450,
        margin=dict(l=60, r=80, t=60, b=100),
        font=dict(family="Inter, sans-serif", size=12),
        barmode='group',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(tickangle=-45)
    )

    return fig


def render():
    """渲染 Training Cost Estimator 页面"""
    default_category = "LLaMA"
    default_model = "LLaMA-7B"
    default_mode = "LoRA"
    default_rank = 16
    default_modules = ["q_proj", "v_proj"]
    default_dataset = 1.0  # 1B tokens
    default_gpu = "NVIDIA A100 (40GB)"
    default_num_gpus = 1
    default_mfu = 0.5

    gr.Markdown("# Training Cost Estimator")
    gr.Markdown(
        "Estimate training time and cost for full fine-tuning or LoRA. "
        "Compare different GPU configurations and dataset sizes."
    )

    with gr.Row():
        # 左侧：配置面板
        with gr.Column(scale=1):
            gr.Markdown("### Model Configuration")

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

            training_mode = gr.Radio(
                label="Training Mode",
                choices=["Full Fine-tuning", "LoRA"],
                value=default_mode,
                info="Full fine-tuning trains all parameters, LoRA only trains low-rank adapters"
            )

            gr.Markdown("### LoRA Configuration")
            gr.Markdown("*Only applicable when training mode is LoRA*")

            lora_rank = gr.Slider(
                label="LoRA Rank",
                minimum=4,
                maximum=128,
                value=default_rank,
                step=4
            )

            lora_modules = gr.CheckboxGroup(
                label="Target Modules",
                choices=[(name, key) for key, name in TARGET_MODULES.items()],
                value=default_modules
            )

            gr.Markdown("### Training Parameters")

            dataset_size = gr.Slider(
                label="Dataset Size (Billion Tokens)",
                minimum=0.1,
                maximum=100,
                value=default_dataset,
                step=0.1,
                info="Total number of tokens in training dataset"
            )

            gpu_type = gr.Dropdown(
                label="GPU Type",
                choices=list(GPU_SPECS.keys()),
                value=default_gpu,
                info="GPU model with specifications"
            )

            num_gpus = gr.Slider(
                label="Number of GPUs",
                minimum=1,
                maximum=64,
                value=default_num_gpus,
                step=1,
                info="Number of GPUs for training"
            )

            mfu = gr.Slider(
                label="Model FLOPs Utilization (MFU)",
                minimum=0.1,
                maximum=1.0,
                value=default_mfu,
                step=0.05,
                info="Typically 0.3-0.6 for real-world training"
            )

        # 右侧：结果展示
        with gr.Column(scale=2):
            status = gr.Markdown("")

            gr.Markdown("### Estimation Results")

            with gr.Row():
                trainable_params = gr.Textbox(label="Trainable Parameters", interactive=False)
                total_flops = gr.Textbox(label="Total FLOPs", interactive=False)

            with gr.Row():
                training_time = gr.Textbox(label="Estimated Training Time", interactive=False)
                total_cost = gr.Textbox(label="Total Cost", interactive=False)

            with gr.Row():
                cost_per_hour = gr.Textbox(label="Cost per Hour", interactive=False)
                gpu_memory = gr.Textbox(label="GPU Memory", interactive=False)

            gr.Markdown("### Cost Analysis")
            gr.Markdown("How training cost scales with dataset size:")
            cost_curve_chart = gr.Plot()

            gr.Markdown("### GPU Comparison")
            gr.Markdown("Compare cost and time across different GPU types:")
            gpu_comparison_chart = gr.Plot()

            gr.Markdown(
                "*Note: These are theoretical estimates. Actual training time and cost may vary "
                "based on implementation details, framework overhead, and infrastructure efficiency.*"
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
    calc_inputs = [
        category, model, training_mode, lora_rank, lora_modules,
        dataset_size, gpu_type, num_gpus, mfu
    ]
    calc_outputs = [
        status, cost_curve_chart, gpu_comparison_chart,
        trainable_params, total_flops, training_time, total_cost,
        cost_per_hour, gpu_memory
    ]

    # 所有参数变化时自动计算
    for component in [model, training_mode, lora_rank, lora_modules, dataset_size, gpu_type, num_gpus, mfu]:
        component.change(
            fn=estimate_cost,
            inputs=calc_inputs,
            outputs=calc_outputs
        )

    # 页面加载时计算默认值
    def on_load():
        return estimate_cost(
            default_category, default_model, default_mode, default_rank,
            default_modules, default_dataset, default_gpu, default_num_gpus, default_mfu
        )

    return {
        'load_fn': on_load,
        'load_outputs': calc_outputs
    }
