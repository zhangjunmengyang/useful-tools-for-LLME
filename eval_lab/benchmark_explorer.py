"""
Benchmark Explorer - Benchmark 数据可视化对比

提供主流 LLM benchmark 结果的交互式可视化，包括：
- 多模型雷达图对比
- 分类别柱状图展示  
- 支持选择模型和benchmark维度
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Tuple
import json

from workbench_theme import OPEN_DESIGN_COLORS, PLOTLY_COLORWAY


# 内置 benchmark 数据（模拟真实数据）
BENCHMARK_DATA = {
    'GPT-4': {
        'MMLU': 86.4,
        'HumanEval': 67.0, 
        'GSM8K': 92.0,
        'HellaSwag': 95.3,
        'ARC': 96.3,
        'TruthfulQA': 59.0,
        'Winogrande': 87.5,
        'MATH': 42.5
    },
    'Claude-3.5-Sonnet': {
        'MMLU': 88.7,
        'HumanEval': 92.0,
        'GSM8K': 96.4,
        'HellaSwag': 89.0,
        'ARC': 96.0,
        'TruthfulQA': 71.3,
        'Winogrande': 89.6,
        'MATH': 71.1
    },
    'GPT-3.5-Turbo': {
        'MMLU': 70.0,
        'HumanEval': 48.1,
        'GSM8K': 57.1,
        'HellaSwag': 85.5,
        'ARC': 85.2,
        'TruthfulQA': 47.0,
        'Winogrande': 81.6,
        'MATH': 34.1
    },
    'Llama-2-70B': {
        'MMLU': 69.8,
        'HumanEval': 29.9,
        'GSM8K': 56.8,
        'HellaSwag': 87.3,
        'ARC': 67.3,
        'TruthfulQA': 52.8,
        'Winogrande': 80.2,
        'MATH': 13.5
    },
    'Llama-3-70B': {
        'MMLU': 82.0,
        'HumanEval': 81.7,
        'GSM8K': 93.0,
        'HellaSwag': 88.0,
        'ARC': 93.0,
        'TruthfulQA': 63.2,
        'Winogrande': 83.5,
        'MATH': 50.4
    },
    'Qwen-72B': {
        'MMLU': 77.4,
        'HumanEval': 35.9,
        'GSM8K': 78.9,
        'HellaSwag': 86.0,
        'ARC': 88.8,
        'TruthfulQA': 54.1,
        'Winogrande': 82.3,
        'MATH': 35.7
    },
    'Gemini-Pro': {
        'MMLU': 71.8,
        'HumanEval': 67.7,
        'GSM8K': 86.5,
        'HellaSwag': 87.8,
        'ARC': 87.2,
        'TruthfulQA': 45.7,
        'Winogrande': 87.0,
        'MATH': 32.6
    },
    'PaLM-2-L': {
        'MMLU': 78.3,
        'HumanEval': 26.2,
        'GSM8K': 80.7,
        'HellaSwag': 86.8,
        'ARC': 89.7,
        'TruthfulQA': 54.6,
        'Winogrande': 83.7,
        'MATH': 34.3
    }
}

# Benchmark 描述信息
BENCHMARK_INFO = {
    'MMLU': 'Massive Multitask Language Understanding across 57 academic subjects',
    'HumanEval': 'Python code generation benchmark with 164 programming tasks',
    'GSM8K': 'Grade-school math word problems for basic reasoning',
    'HellaSwag': 'Commonsense reasoning benchmark for plausible sentence endings',
    'ARC': 'AI2 Reasoning Challenge for science commonsense reasoning',
    'TruthfulQA': 'Truthfulness and factual accuracy benchmark',
    'Winogrande': 'Expanded Winograd Schema Challenge for commonsense reasoning',
    'MATH': 'Competition-level math problems for advanced reasoning'
}

# 颜色方案
MODEL_COLORS = PLOTLY_COLORWAY


def create_radar_chart(selected_models: List[str], selected_benchmarks: List[str]) -> go.Figure:
    """
    创建雷达图对比选中的模型在各benchmarks上的表现
    
    Args:
        selected_models: 选择的模型列表
        selected_benchmarks: 选择的benchmark列表
    
    Returns:
        Plotly 雷达图
    """
    if not selected_models or not selected_benchmarks:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one model and one benchmark",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray")
        )
        return fig
    
    fig = go.Figure()
    
    # 为每个选中的模型添加雷达图轨迹
    for i, model in enumerate(selected_models):
        if model in BENCHMARK_DATA:
            # 获取该模型在选中benchmarks上的分数
            scores = []
            categories = []
            
            for benchmark in selected_benchmarks:
                if benchmark in BENCHMARK_DATA[model]:
                    scores.append(BENCHMARK_DATA[model][benchmark])
                    categories.append(benchmark)
            
            if scores:
                # 添加雷达图轨迹
                fig.add_trace(go.Scatterpolar(
                    r=scores + [scores[0]],  # 闭合图形
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=model,
                    line=dict(color=MODEL_COLORS[i % len(MODEL_COLORS)], width=2),
                    fillcolor=MODEL_COLORS[i % len(MODEL_COLORS)],
                    opacity=0.3
                ))
    
    # 更新布局
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=12),
                gridcolor=OPEN_DESIGN_COLORS["border_soft"]
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color=OPEN_DESIGN_COLORS["fg"]),
                rotation=90,
                direction='counterclockwise'
            )
        ),
        showlegend=True,
        title=dict(
            text="Model Performance Radar",
            x=0.5,
            font=dict(size=18, color=OPEN_DESIGN_COLORS["fg"])
        ),
        font=dict(family="Inter, system-ui, sans-serif"),
        autosize=True,
        height=600,
        paper_bgcolor=OPEN_DESIGN_COLORS["bg"],
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left", 
            x=1.02
        )
    )
    
    return fig


def create_bar_chart(selected_models: List[str], selected_benchmarks: List[str]) -> go.Figure:
    """
    创建柱状图展示各模型在不同benchmarks上的得分
    
    Args:
        selected_models: 选择的模型列表
        selected_benchmarks: 选择的benchmark列表
    
    Returns:
        Plotly 柱状图
    """
    if not selected_models or not selected_benchmarks:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one model and one benchmark",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray")
        )
        return fig
    
    fig = go.Figure()
    
    # 为每个选中的模型创建柱状图
    for i, model in enumerate(selected_models):
        if model in BENCHMARK_DATA:
            scores = []
            benchmarks = []
            
            for benchmark in selected_benchmarks:
                if benchmark in BENCHMARK_DATA[model]:
                    scores.append(BENCHMARK_DATA[model][benchmark])
                    benchmarks.append(benchmark)
            
            if scores:
                fig.add_trace(go.Bar(
                    x=benchmarks,
                    y=scores,
                    name=model,
                    marker_color=MODEL_COLORS[i % len(MODEL_COLORS)],
                    text=[f'{score:.1f}' for score in scores],
                    textposition='outside'
                ))
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text="Benchmark Score Comparison",
            x=0.5,
            font=dict(size=18, color=OPEN_DESIGN_COLORS["fg"])
        ),
        xaxis_title="Benchmark",
        yaxis_title="Score",
        barmode='group',
        autosize=True,
        height=500,
        font=dict(family="Inter, system-ui, sans-serif"),
        paper_bgcolor=OPEN_DESIGN_COLORS["bg"],
        plot_bgcolor=OPEN_DESIGN_COLORS["bg"],
        yaxis=dict(range=[0, 100]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def create_heatmap(selected_models: List[str], selected_benchmarks: List[str]) -> go.Figure:
    """
    创建热力图展示模型-benchmark矩阵
    
    Args:
        selected_models: 选择的模型列表
        selected_benchmarks: 选择的benchmark列表
    
    Returns:
        Plotly 热力图
    """
    if not selected_models or not selected_benchmarks:
        fig = go.Figure()
        fig.add_annotation(
            text="Select at least one model and one benchmark",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray")
        )
        return fig
    
    # 构建得分矩阵
    matrix = []
    model_labels = []
    
    for model in selected_models:
        if model in BENCHMARK_DATA:
            row = []
            for benchmark in selected_benchmarks:
                score = BENCHMARK_DATA[model].get(benchmark, 0)
                row.append(score)
            matrix.append(row)
            model_labels.append(model)
    
    if not matrix:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=16, color="gray")
        )
        return fig
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=selected_benchmarks,
        y=model_labels,
        colorscale='RdYlBu_r',
        text=[[f'{val:.1f}' for val in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="Score")
    ))
    
    fig.update_layout(
        title=dict(
            text="Model-Benchmark Score Heatmap",
            x=0.5,
            font=dict(size=18, color=OPEN_DESIGN_COLORS["fg"])
        ),
        autosize=True,
        height=400 + len(model_labels) * 30,
        font=dict(family="Inter, system-ui, sans-serif"),
        paper_bgcolor=OPEN_DESIGN_COLORS["bg"]
    )
    
    return fig


def get_benchmark_details(selected_benchmark: str) -> str:
    """
    获取选中benchmark的详细信息
    
    Args:
        selected_benchmark: 选择的benchmark名称
    
    Returns:
        Benchmark详细信息的Markdown文本
    """
    if not selected_benchmark or selected_benchmark not in BENCHMARK_INFO:
        return "Select a benchmark to view details."
    
    info = BENCHMARK_INFO[selected_benchmark]
    
    # 获取该benchmark上所有模型的排名
    scores = []
    for model, benchmarks in BENCHMARK_DATA.items():
        if selected_benchmark in benchmarks:
            scores.append((model, benchmarks[selected_benchmark]))
    
    # 按分数排序
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # 构建详情文本
    details = f"## {selected_benchmark}\n\n"
    details += f"**Description**: {info}\n\n"
    details += "### Ranking\n\n"
    details += "| Rank | Model | Score |\n"
    details += "|------|------|------|\n"
    
    for i, (model, score) in enumerate(scores, 1):
        details += f"| {i} | {model} | {score:.1f} |\n"
    
    return details


def create_benchmark_overview_table() -> pd.DataFrame:
    """
    创建benchmark概览表格
    
    Returns:
        包含所有模型和benchmark的DataFrame
    """
    # 转换数据格式以适配表格展示
    rows = []
    for model, benchmarks in BENCHMARK_DATA.items():
        row = {'Model': model}
        for benchmark in BENCHMARK_INFO.keys():
            score = benchmarks.get(benchmark, None)
            row[benchmark] = f"{score:.1f}" if score is not None else "-"
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df


def render():
    """
    渲染 Benchmark Explorer 页面
    
    Returns:
        load 事件配置 (如果需要)
    """
    with gr.Column():
        gr.HTML("""
        <div class="workbench-page-hero">
            <h1>Benchmark Explorer</h1>
            <p>Compare mainstream LLM benchmark scores across models and task families.</p>
        </div>
        """)
        
        initial_models = ['GPT-4', 'Claude-3.5-Sonnet', 'Llama-3-70B']
        initial_benchmarks = ['MMLU', 'HumanEval', 'GSM8K', 'HellaSwag']
        initial_radar = create_radar_chart(initial_models, initial_benchmarks)
        initial_bar = create_bar_chart(initial_models, initial_benchmarks)
        initial_heatmap = create_heatmap(initial_models, initial_benchmarks)

        with gr.Row(elem_classes=["workbench-tool-shell"]):
            with gr.Column(scale=1, elem_classes=["workbench-control-panel"]):
                gr.HTML("""
                <p class="workbench-panel-title">Benchmark Slice</p>
                <p class="workbench-panel-copy">Select models and benchmark families, then inspect the active comparison view.</p>
                """)

                # 模型选择
                model_selector = gr.CheckboxGroup(
                    choices=list(BENCHMARK_DATA.keys()),
                    value=initial_models,
                    label="Models",
                    info="Choose models to compare"
                )
                
                # Benchmark选择
                benchmark_selector = gr.CheckboxGroup(
                    choices=list(BENCHMARK_INFO.keys()),
                    value=initial_benchmarks,
                    label="Benchmarks",
                    info="Choose benchmark dimensions"
                )
                
                # Benchmark详情查看
                benchmark_detail_selector = gr.Dropdown(
                    choices=list(BENCHMARK_INFO.keys()),
                    value='MMLU',
                    label="Benchmark Details",
                    info="Choose one benchmark to inspect"
                )
            
            with gr.Column(scale=2, elem_classes=["workbench-output-panel"]):
                # 可视化tabs
                with gr.Tabs():
                    with gr.Tab("Radar") as radar_tab:
                        with gr.Column(elem_classes=["plot-frame"]):
                            radar_plot = gr.Plot(value=initial_radar, label="Radar Chart")
                    
                    with gr.Tab("Bars") as bars_tab:
                        with gr.Column(elem_classes=["plot-frame"]):
                            bar_plot = gr.Plot(value=initial_bar, label="Bar Chart")
                    
                    with gr.Tab("Heatmap") as heatmap_tab:
                        with gr.Column(elem_classes=["plot-frame"]):
                            heatmap_plot = gr.Plot(value=initial_heatmap, label="Heatmap")
                    
                    with gr.Tab("Table") as table_tab:
                        overview_table = gr.Dataframe(
                            value=create_benchmark_overview_table(),
                            label="Full Score Table",
                            interactive=False
                        )
        
        # Benchmark详情展示
        with gr.Row(elem_classes=["workbench-detail-panel"]):
            benchmark_details = gr.Markdown(
                value=get_benchmark_details('MMLU'),
                label="Benchmark Details"
            )
        
        # 事件绑定
        def update_plots(selected_models, selected_benchmarks):
            """更新所有图表"""
            radar = create_radar_chart(selected_models, selected_benchmarks)
            bar = create_bar_chart(selected_models, selected_benchmarks) 
            heatmap = create_heatmap(selected_models, selected_benchmarks)
            return radar, bar, heatmap
        
        # 绑定模型和benchmark选择变化事件
        model_selector.change(
            fn=update_plots,
            inputs=[model_selector, benchmark_selector],
            outputs=[radar_plot, bar_plot, heatmap_plot]
        )
        
        benchmark_selector.change(
            fn=update_plots,
            inputs=[model_selector, benchmark_selector],
            outputs=[radar_plot, bar_plot, heatmap_plot]
        )
        
        # 绑定benchmark详情查看
        benchmark_detail_selector.change(
            fn=get_benchmark_details,
            inputs=[benchmark_detail_selector],
            outputs=[benchmark_details]
        )

        radar_tab.select(
            fn=create_radar_chart,
            inputs=[model_selector, benchmark_selector],
            outputs=[radar_plot],
        )

        bars_tab.select(
            fn=create_bar_chart,
            inputs=[model_selector, benchmark_selector],
            outputs=[bar_plot],
        )

        heatmap_tab.select(
            fn=create_heatmap,
            inputs=[model_selector, benchmark_selector],
            outputs=[heatmap_plot],
        )

        table_tab.select(
            fn=create_benchmark_overview_table,
            outputs=[overview_table],
        )

        def on_load():
            """初始化隐藏 Tab 中的图表。"""
            return (
                create_radar_chart(initial_models, initial_benchmarks),
                create_bar_chart(initial_models, initial_benchmarks),
                create_heatmap(initial_models, initial_benchmarks),
                create_benchmark_overview_table(),
                get_benchmark_details('MMLU'),
            )

        return {
            'load_fn': on_load,
            'load_outputs': [
                radar_plot,
                bar_plot,
                heatmap_plot,
                overview_table,
                benchmark_details,
            ],
        }


if __name__ == "__main__":
    # 测试页面
    with gr.Blocks(title="Benchmark Explorer") as demo:
        render()
    
    demo.launch()
