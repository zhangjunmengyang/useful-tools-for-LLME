"""
自动化评测Pipeline - 批量评测和结果分析

提供完整的评测流水线功能：
- 数据集上传和解析
- 多指标批量评测
- 结果可视化分析
- 评测报告导出
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import json
import io
import numpy as np
from eval_lab.eval_utils import (
    calculate_all_metrics, 
    format_metrics_table,
    generate_eval_report
)


# 内置示例数据集
SAMPLE_DATASETS = {
    'QA数据集示例': {
        'description': '问答任务示例数据，包含问题、参考答案和模型预测',
        'data': [
            {
                'question': 'Python中如何读取CSV文件？',
                'reference': '使用pandas库的read_csv()函数可以方便地读取CSV文件。',
                'prediction': 'import pandas as pd\ndf = pd.read_csv("file.csv")\n可以使用pandas的read_csv函数读取CSV文件。'
            },
            {
                'question': '什么是机器学习？',
                'reference': '机器学习是人工智能的一个分支，让计算机通过数据学习模式和做出预测。',
                'prediction': '机器学习是AI的子领域，通过算法让计算机从数据中学习规律，无需显式编程。'
            },
            {
                'question': '如何提高代码质量？',
                'reference': '通过代码审查、单元测试、遵循编码规范、重构等方式可以提高代码质量。',
                'prediction': '写注释、做测试、code review、用linter检查格式。'
            },
            {
                'question': 'SQL中的JOIN有哪些类型？',
                'reference': '主要有INNER JOIN、LEFT JOIN、RIGHT JOIN、FULL OUTER JOIN等类型。',
                'prediction': 'SQL的JOIN包括内连接(INNER JOIN)、左连接(LEFT JOIN)、右连接(RIGHT JOIN)和全外连接(FULL OUTER JOIN)。'
            },
            {
                'question': '什么是REST API？',
                'reference': 'REST是一种软件架构风格，用于设计网络应用的API，基于HTTP协议。',
                'prediction': 'REST API是基于HTTP的Web服务接口，使用GET/POST/PUT/DELETE等方法进行资源操作。'
            }
        ]
    },
    '代码生成示例': {
        'description': '代码生成任务示例，评测代码的正确性和质量',
        'data': [
            {
                'question': '写一个Python函数计算斐波那契数列的第n项',
                'reference': 'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
                'prediction': 'def fib(n):\n    if n < 2:\n        return n\n    a, b = 0, 1\n    for i in range(2, n+1):\n        a, b = b, a + b\n    return b'
            },
            {
                'question': '实现一个函数判断字符串是否为回文',
                'reference': 'def is_palindrome(s):\n    s = s.lower().replace(" ", "")\n    return s == s[::-1]',
                'prediction': 'def palindrome_check(text):\n    clean = "".join(text.split()).lower()\n    return clean == clean[::-1]'
            },
            {
                'question': '写一个快速排序算法',
                'reference': 'def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)',
                'prediction': 'def quick_sort(array):\n    if len(array) < 2:\n        return array\n    pivot = array[0]\n    less = [i for i in array[1:] if i <= pivot]\n    greater = [i for i in array[1:] if i > pivot]\n    return quick_sort(less) + [pivot] + quick_sort(greater)'
            }
        ]
    },
    '翻译质量评测': {
        'description': '机器翻译质量评测示例',
        'data': [
            {
                'question': 'Hello, how are you today?',
                'reference': '你好，你今天怎么样？',
                'prediction': '您好，您今天好吗？'
            },
            {
                'question': 'The weather is very nice today.',
                'reference': '今天天气很好。',
                'prediction': '今天的天气非常不错。'
            },
            {
                'question': 'I love programming and artificial intelligence.',
                'reference': '我喜欢编程和人工智能。',
                'prediction': '我热爱编程和AI技术。'
            }
        ]
    }
}

# 模型配置
MOCK_MODELS = ['GPT-4', 'Claude-3.5', 'GPT-3.5', 'Llama-3-70B', 'Qwen-72B']


def parse_uploaded_file(file_content: str, file_type: str) -> List[Dict[str, str]]:
    """
    解析上传的文件内容
    
    Args:
        file_content: 文件内容
        file_type: 文件类型 ('json' 或 'csv')
    
    Returns:
        解析后的数据列表
    """
    try:
        if file_type == 'json':
            data = json.loads(file_content)
            if isinstance(data, list):
                return data
            else:
                return [data]
        elif file_type == 'csv':
            # 使用pandas解析CSV
            df = pd.read_csv(io.StringIO(file_content))
            return df.to_dict('records')
        else:
            raise ValueError(f"不支持的文件类型: {file_type}")
    except Exception as e:
        raise ValueError(f"文件解析失败: {str(e)}")


def validate_dataset(data: List[Dict[str, str]]) -> Tuple[bool, str]:
    """
    验证数据集格式
    
    Args:
        data: 数据集
    
    Returns:
        (是否有效, 错误信息)
    """
    if not data:
        return False, "数据集为空"
    
    required_fields = {'question', 'reference', 'prediction'}
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"第{i+1}行数据格式错误，应为字典格式"
        
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            return False, f"第{i+1}行缺少必要字段: {', '.join(missing_fields)}"
        
        # 检查字段是否为字符串
        for field in required_fields:
            if not isinstance(item[field], str):
                return False, f"第{i+1}行{field}字段应为字符串"
    
    return True, ""


def run_batch_evaluation(data: List[Dict[str, str]], selected_metrics: List[str]) -> Dict[str, Any]:
    """
    运行批量评测
    
    Args:
        data: 评测数据
        selected_metrics: 选择的评测指标
    
    Returns:
        评测结果
    """
    if not data:
        return {'error': '没有数据进行评测'}
    
    # 提取预测和参考答案
    predictions = [item['prediction'] for item in data]
    references = [item['reference'] for item in data]
    
    # 计算指标
    try:
        all_metrics = calculate_all_metrics(predictions, references)
        
        # 过滤选择的指标
        if selected_metrics:
            filtered_metrics = {k: v for k, v in all_metrics.items() if k in selected_metrics}
        else:
            filtered_metrics = all_metrics
        
        # 样本级别的详细结果
        sample_results = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            sample_metrics = calculate_all_metrics([pred], [ref])
            sample_results.append({
                'index': i,
                'question': data[i]['question'],
                'prediction': pred,
                'reference': ref,
                'metrics': sample_metrics
            })
        
        return {
            'overall_metrics': filtered_metrics,
            'sample_results': sample_results,
            'total_samples': len(data)
        }
    
    except Exception as e:
        return {'error': f'评测过程中出错: {str(e)}'}


def create_metrics_comparison_chart(results_dict: Dict[str, Dict[str, float]]) -> go.Figure:
    """
    创建多个评测结果的对比图表
    
    Args:
        results_dict: {模型名称: 指标结果} 的字典
    
    Returns:
        Plotly图表
    """
    if not results_dict:
        fig = go.Figure()
        fig.add_annotation(text="没有数据", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # 提取所有指标名称
    all_metrics = set()
    for metrics in results_dict.values():
        all_metrics.update(metrics.keys())
    all_metrics = sorted(list(all_metrics))
    
    fig = go.Figure()
    
    # 为每个模型添加柱状图
    colors = px.colors.qualitative.Set3
    for i, (model_name, metrics) in enumerate(results_dict.items()):
        scores = [metrics.get(metric, 0) for metric in all_metrics]
        
        fig.add_trace(go.Bar(
            name=model_name,
            x=all_metrics,
            y=scores,
            marker_color=colors[i % len(colors)],
            text=[f'{score:.3f}' for score in scores],
            textposition='outside'
        ))
    
    fig.update_layout(
        title="多模型评测结果对比",
        xaxis_title="评测指标",
        yaxis_title="分数",
        barmode='group',
        height=500,
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def create_metrics_radar_chart(metrics: Dict[str, float]) -> go.Figure:
    """
    创建单个评测结果的雷达图
    
    Args:
        metrics: 指标结果
    
    Returns:
        Plotly雷达图
    """
    if not metrics:
        fig = go.Figure()
        fig.add_annotation(text="没有数据", x=0.5, y=0.5, showarrow=False)
        return fig
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='评测结果',
        line=dict(color='#1f77b4', width=2),
        fillcolor='#1f77b4',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        title=dict(text="评测指标雷达图", x=0.5, font=dict(size=16)),
        width=500,
        height=500
    )
    
    return fig


def create_sample_analysis_chart(sample_results: List[Dict]) -> go.Figure:
    """
    创建样本级别的分析图表
    
    Args:
        sample_results: 样本结果列表
    
    Returns:
        Plotly图表
    """
    if not sample_results:
        fig = go.Figure()
        fig.add_annotation(text="没有样本数据", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # 提取每个样本的F1分数用于展示
    indices = [res['index'] + 1 for res in sample_results]  # 1-based indexing
    f1_scores = [res['metrics'].get('f1', 0) for res in sample_results]
    
    fig = go.Figure()
    
    # 添加散点图
    fig.add_trace(go.Scatter(
        x=indices,
        y=f1_scores,
        mode='markers+lines',
        marker=dict(
            size=8,
            color=f1_scores,
            colorscale='RdYlBu_r',
            colorbar=dict(title="F1 Score"),
            line=dict(width=1, color='white')
        ),
        line=dict(width=2, color='gray'),
        name='F1 Score',
        text=[f"样本{i}: {score:.3f}" for i, score in zip(indices, f1_scores)],
        hovertemplate='<b>样本 %{x}</b><br>F1 Score: %{y:.3f}<extra></extra>'
    ))
    
    # 添加平均线
    avg_f1 = np.mean(f1_scores)
    fig.add_hline(
        y=avg_f1,
        line_dash="dash",
        line_color="red",
        annotation_text=f"平均 F1: {avg_f1:.3f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title="样本级别评测结果分析",
        xaxis_title="样本编号",
        yaxis_title="F1 Score",
        height=400,
        yaxis=dict(range=[0, 1])
    )
    
    return fig


def format_sample_results_table(sample_results: List[Dict]) -> pd.DataFrame:
    """
    格式化样本结果为表格
    
    Args:
        sample_results: 样本结果列表
    
    Returns:
        DataFrame
    """
    if not sample_results:
        return pd.DataFrame()
    
    rows = []
    for res in sample_results:
        row = {
            '样本': res['index'] + 1,
            '问题': res['question'][:50] + '...' if len(res['question']) > 50 else res['question'],
            'F1': f"{res['metrics'].get('f1', 0):.3f}",
            'BLEU': f"{res['metrics'].get('bleu', 0):.3f}",
            'ROUGE-L': f"{res['metrics'].get('rouge_l', 0):.3f}",
            '精确匹配': f"{res['metrics'].get('exact_match', 0):.3f}"
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def render():
    """
    渲染自动化评测Pipeline页面
    
    Returns:
        load 事件配置 (如果需要)
    """
    with gr.Column():
        gr.HTML("""
        <div class="main-header">
            <h1 style="color: #1f2937; margin-bottom: 8px;">🔄 自动化评测Pipeline</h1>
            <p style="color: #6b7280; font-size: 1.1rem; margin: 0;">批量评测、结果分析和报告生成</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: 数据准备
            with gr.Tab("数据准备"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 数据输入")
                        
                        # 数据来源选择
                        data_source = gr.Radio(
                            choices=["上传文件", "使用示例数据"],
                            value="使用示例数据",
                            label="数据来源"
                        )
                        
                        # 文件上传
                        file_upload = gr.File(
                            label="上传数据集文件 (JSON/CSV)",
                            file_types=[".json", ".csv"],
                            visible=False
                        )
                        
                        # 示例数据选择
                        sample_dataset_selector = gr.Dropdown(
                            choices=list(SAMPLE_DATASETS.keys()),
                            value=list(SAMPLE_DATASETS.keys())[0],
                            label="选择示例数据集"
                        )
                        
                        load_data_btn = gr.Button("加载数据", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 数据格式说明")
                        gr.Markdown("""
                        **JSON格式示例:**
                        ```json
                        [
                            {
                                "question": "问题内容",
                                "reference": "参考答案",
                                "prediction": "模型预测"
                            }
                        ]
                        ```
                        
                        **CSV格式示例:**
                        ```csv
                        question,reference,prediction
                        问题1,参考答案1,模型预测1
                        问题2,参考答案2,模型预测2
                        ```
                        """)
                
                # 数据预览
                with gr.Row():
                    data_preview = gr.Dataframe(
                        label="数据预览",
                        interactive=False
                    )
                
                # 数据统计
                data_status = gr.Textbox(
                    label="数据状态",
                    interactive=False,
                    value="请先加载数据"
                )
            
            # Tab 2: 配置评测
            with gr.Tab("配置评测"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 评测配置")
                        
                        # 指标选择
                        metrics_selector = gr.CheckboxGroup(
                            choices=[
                                ('Exact Match', 'exact_match'),
                                ('F1 Score', 'f1'),
                                ('BLEU', 'bleu'), 
                                ('ROUGE-L', 'rouge_l'),
                                ('BERTScore F1', 'bertscore_f1')
                            ],
                            value=['exact_match', 'f1', 'bleu'],
                            label="选择评测指标"
                        )
                        
                        # 模型名称设置
                        model_name_input = gr.Textbox(
                            value="GPT-4",
                            label="模型名称",
                            info="用于生成报告"
                        )
                        
                        dataset_name_input = gr.Textbox(
                            value="自定义数据集",
                            label="数据集名称",
                            info="用于生成报告"
                        )
                        
                        run_eval_btn = gr.Button("开始评测", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### 指标说明")
                        gr.Markdown("""
                        - **Exact Match**: 预测与参考完全匹配的比例
                        - **F1 Score**: 基于token的精确率和召回率调和平均
                        - **BLEU**: 机器翻译质量指标，基于n-gram匹配
                        - **ROUGE-L**: 基于最长公共子序列的文本相似度
                        - **BERTScore F1**: 基于语义理解的相似度评分
                        """)
            
            # Tab 3: 结果分析
            with gr.Tab("结果分析"):
                with gr.Tabs():
                    with gr.Tab("整体指标"):
                        with gr.Row():
                            with gr.Column():
                                # 整体指标表格
                                overall_metrics_table = gr.Markdown(
                                    value="请先完成评测",
                                    label="整体评测结果"
                                )
                            
                            with gr.Column():
                                # 雷达图
                                metrics_radar_chart = gr.Plot(label="指标雷达图")
                    
                    with gr.Tab("样本分析"):
                        # 样本级别分析图表
                        sample_analysis_chart = gr.Plot(label="样本分析")
                        
                        # 详细样本结果表格
                        sample_results_table = gr.Dataframe(
                            label="样本详细结果",
                            interactive=False
                        )
                    
                    with gr.Tab("多模型对比"):
                        gr.Markdown("""
                        ### 多模型对比功能
                        
                        此功能用于对比不同模型在同一数据集上的表现。
                        在实际应用中，您可以：
                        1. 分别评测不同模型的结果
                        2. 保存各自的评测结果
                        3. 在此进行可视化对比
                        """)
                        
                        # 预置一些对比数据用于演示
                        comparison_chart = gr.Plot(
                            value=create_metrics_comparison_chart({
                                'GPT-4': {'exact_match': 0.45, 'f1': 0.78, 'bleu': 0.52},
                                'Claude-3.5': {'exact_match': 0.38, 'f1': 0.82, 'bleu': 0.48},
                                'GPT-3.5': {'exact_match': 0.32, 'f1': 0.69, 'bleu': 0.41}
                            }),
                            label="模型对比"
                        )
            
            # Tab 4: 导出报告
            with gr.Tab("导出报告"):
                gr.Markdown("### 评测报告导出")
                
                export_format = gr.Radio(
                    choices=["Markdown", "JSON"],
                    value="Markdown",
                    label="导出格式"
                )
                
                generate_report_btn = gr.Button("生成报告", variant="primary")
                
                # 报告预览
                report_preview = gr.Textbox(
                    label="报告预览",
                    lines=15,
                    interactive=False
                )
                
                # 下载链接
                report_download = gr.File(label="下载报告")
        
        # 存储评测数据的状态变量
        eval_data_state = gr.State(value=[])
        eval_results_state = gr.State(value={})
        
        # 事件处理函数
        def toggle_data_source(source):
            """切换数据来源显示"""
            if source == "上传文件":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)
        
        def load_sample_data(dataset_name):
            """加载示例数据"""
            if dataset_name in SAMPLE_DATASETS:
                data = SAMPLE_DATASETS[dataset_name]['data']
                df = pd.DataFrame(data)
                status = f"✅ 已加载 {len(data)} 条样本数据"
                return df, status, data
            else:
                return pd.DataFrame(), "❌ 数据集不存在", []
        
        def load_uploaded_data(file):
            """加载上传的文件数据"""
            if file is None:
                return pd.DataFrame(), "❌ 请先上传文件", []
            
            try:
                # 读取文件内容
                content = file.read().decode('utf-8')
                file_ext = file.name.split('.')[-1].lower()
                
                # 解析文件
                data = parse_uploaded_file(content, file_ext)
                
                # 验证数据格式
                is_valid, error_msg = validate_dataset(data)
                if not is_valid:
                    return pd.DataFrame(), f"❌ {error_msg}", []
                
                df = pd.DataFrame(data)
                status = f"✅ 已加载 {len(data)} 条样本数据"
                return df, status, data
                
            except Exception as e:
                return pd.DataFrame(), f"❌ 加载失败: {str(e)}", []
        
        def run_evaluation(data, selected_metrics, model_name, dataset_name):
            """运行评测"""
            if not data:
                return "❌ 请先加载数据", gr.update(), gr.update(), gr.update(), {}
            
            if not selected_metrics:
                return "❌ 请选择至少一个评测指标", gr.update(), gr.update(), gr.update(), {}
            
            # 运行评测
            results = run_batch_evaluation(data, selected_metrics)
            
            if 'error' in results:
                return f"❌ {results['error']}", gr.update(), gr.update(), gr.update(), {}
            
            # 格式化整体指标表格
            metrics_table = format_metrics_table(results['overall_metrics'])
            
            # 创建雷达图
            radar_chart = create_metrics_radar_chart(results['overall_metrics'])
            
            # 创建样本分析图
            sample_chart = create_sample_analysis_chart(results['sample_results'])
            
            # 格式化样本结果表格
            sample_table = format_sample_results_table(results['sample_results'])
            
            status = f"✅ 评测完成，共处理 {results['total_samples']} 个样本"
            
            # 保存结果用于生成报告
            report_results = {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'metrics': results['overall_metrics'],
                'sample_results': results['sample_results']
            }
            
            return status, metrics_table, radar_chart, sample_chart, sample_table, report_results
        
        def generate_report(results, export_format):
            """生成评测报告"""
            if not results:
                return "❌ 请先完成评测", None
            
            try:
                # 准备样本预测数据
                sample_predictions = []
                for res in results.get('sample_results', [])[:3]:  # 只取前3个样本
                    sample_predictions.append((
                        res['question'],
                        res['reference'],
                        res['prediction']
                    ))
                
                # 生成报告
                if export_format == "Markdown":
                    report_content = generate_eval_report(
                        results['dataset_name'],
                        results['model_name'],
                        results['metrics'],
                        sample_predictions
                    )
                    
                    # 保存为文件
                    report_file = f"eval_report_{results['model_name']}_{results['dataset_name']}.md"
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    
                    return report_content, report_file
                
                else:  # JSON格式
                    report_data = {
                        'model_name': results['model_name'],
                        'dataset_name': results['dataset_name'],
                        'overall_metrics': results['metrics'],
                        'total_samples': len(results.get('sample_results', [])),
                        'sample_results': results.get('sample_results', [])
                    }
                    
                    report_content = json.dumps(report_data, ensure_ascii=False, indent=2)
                    
                    # 保存为文件
                    report_file = f"eval_report_{results['model_name']}_{results['dataset_name']}.json"
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    
                    return report_content, report_file
                    
            except Exception as e:
                return f"❌ 生成报告失败: {str(e)}", None
        
        # 事件绑定
        data_source.change(
            fn=toggle_data_source,
            inputs=[data_source],
            outputs=[file_upload, sample_dataset_selector]
        )
        
        load_data_btn.click(
            fn=lambda source, dataset_name, file: load_sample_data(dataset_name) if source == "使用示例数据" else load_uploaded_data(file),
            inputs=[data_source, sample_dataset_selector, file_upload],
            outputs=[data_preview, data_status, eval_data_state]
        )
        
        run_eval_btn.click(
            fn=run_evaluation,
            inputs=[eval_data_state, metrics_selector, model_name_input, dataset_name_input],
            outputs=[
                data_status, overall_metrics_table, metrics_radar_chart,
                sample_analysis_chart, sample_results_table, eval_results_state
            ]
        )
        
        generate_report_btn.click(
            fn=generate_report,
            inputs=[eval_results_state, export_format],
            outputs=[report_preview, report_download]
        )
        
        # 初始化加载示例数据
        initial_data = SAMPLE_DATASETS[list(SAMPLE_DATASETS.keys())[0]]['data']
        initial_df = pd.DataFrame(initial_data)
        initial_status = f"✅ 已加载 {len(initial_data)} 条样本数据"
        
        data_preview.value = initial_df
        data_status.value = initial_status
        eval_data_state.value = initial_data


if __name__ == "__main__":
    # 测试页面
    with gr.Blocks(title="Eval Pipeline") as demo:
        render()
    
    demo.launch()