"""
LLM-as-Judge 演示 - LLM评分系统展示

展示LLM作为评估器的工作原理，包括：
- Pointwise vs Pairwise 评分
- 评分维度和rubric展示
- Position bias 和 Verbosity bias 可视化演示
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Any, Tuple
import random
import json


# 预设的评测prompt模板
JUDGE_PROMPTS = {
    'helpfulness': {
        'name': '帮助性评分',
        'description': '评估回答对用户问题的帮助程度',
        'criteria': [
            '直接回答了用户的问题',
            '提供了有用和准确的信息', 
            '回答完整且易于理解',
            '包含相关的例子或解释'
        ]
    },
    'accuracy': {
        'name': '准确性评分', 
        'description': '评估回答的事实准确性',
        'criteria': [
            '信息事实正确无误',
            '引用可靠的数据源',
            '避免误导性陈述',
            '承认不确定性'
        ]
    },
    'safety': {
        'name': '安全性评分',
        'description': '评估回答是否安全、无害',
        'criteria': [
            '避免有害或危险建议',
            '不包含歧视性内容',
            '尊重用户隐私',
            '遵循伦理准则'
        ]
    },
    'creativity': {
        'name': '创造性评分',
        'description': '评估回答的创新性和创造力',
        'criteria': [
            '提供新颖的观点或方法',
            '展现想象力和原创性',
            '跳出传统思维模式',
            '激发进一步思考'
        ]
    }
}

# 预设示例对话
EXAMPLE_CONVERSATIONS = [
    {
        'question': '如何学习机器学习？',
        'response_a': '学习机器学习需要扎实的数学基础，包括线性代数、统计学和微积分。建议从Python编程开始，然后学习sklearn、pandas等库。可以从Andrew Ng的课程开始，配合实际项目练习。',
        'response_b': '直接看视频就行，YouTube上有很多教程，跟着做几个项目就能学会了。数学不重要，现在都有现成的库。'
    },
    {
        'question': 'Python中如何处理大数据？',
        'response_a': '处理大数据可以使用pandas的分块读取、Dask进行并行计算、或者使用Spark的PySpark接口。对于超大数据集，建议使用数据库或云服务如BigQuery。具体方法取决于数据大小和计算资源。',
        'response_b': '用pandas就够了，read_csv()直接读取，如果内存不够就买更多内存。现在电脑配置都很高，一般没问题。'
    },
    {
        'question': '如何提高代码质量？',
        'response_a': '提高代码质量需要多方面努力：1)遵循编码规范如PEP8 2)写单元测试 3)代码审查 4)使用linter和formatter 5)重构和优化 6)文档注释完善 7)版本控制最佳实践。',
        'response_b': '代码能跑就行，不用太在意格式。变量名随便起，反正自己知道是什么意思。测试也没必要，手动测试一下就可以了。'
    }
]

# Mock评分函数
def mock_pointwise_judge(question: str, response: str, criteria: str) -> Dict[str, Any]:
    """
    模拟Pointwise评分
    
    Args:
        question: 问题
        response: 回答
        criteria: 评分标准
    
    Returns:
        评分结果
    """
    # 基于回答长度和关键词的简单评分逻辑
    response_length = len(response)
    keywords = ['建议', '方法', '具体', '详细', '例子', '实践', '步骤']
    keyword_count = sum(1 for kw in keywords if kw in response)
    
    # 基础分数 + 长度奖励 + 关键词奖励
    base_score = random.uniform(6.0, 8.0)
    length_bonus = min(2.0, response_length / 100)  
    keyword_bonus = keyword_count * 0.3
    
    score = min(10.0, base_score + length_bonus + keyword_bonus)
    
    # 生成模拟的评分理由
    reasoning = f"评分依据：回答长度适中({response_length}字符)，"
    if keyword_count > 2:
        reasoning += f"包含{keyword_count}个关键指导词汇，"
    reasoning += f"在{criteria}方面表现{'良好' if score > 7 else '一般'}。"
    
    return {
        'score': round(score, 1),
        'reasoning': reasoning,
        'criteria_breakdown': {
            '相关性': random.uniform(7, 9),
            '完整性': random.uniform(6, 8), 
            '清晰度': random.uniform(7, 9),
            '实用性': random.uniform(6, 8)
        }
    }


def mock_pairwise_judge(question: str, response_a: str, response_b: str, criteria: str) -> Dict[str, Any]:
    """
    模拟Pairwise评分
    
    Args:
        question: 问题
        response_a: 回答A
        response_b: 回答B
        criteria: 评分标准
    
    Returns:
        对比评分结果
    """
    # 简单的对比逻辑：较长且包含更多关键词的回答得分更高
    keywords = ['建议', '方法', '具体', '详细', '例子', '实践', '步骤', '可以', '需要']
    
    score_a = len(response_a) + sum(2 for kw in keywords if kw in response_a)
    score_b = len(response_b) + sum(2 for kw in keywords if kw in response_b)
    
    if score_a > score_b:
        winner = 'A'
        confidence = min(0.9, (score_a - score_b) / max(score_a, 100))
    elif score_b > score_a:
        winner = 'B' 
        confidence = min(0.9, (score_b - score_a) / max(score_b, 100))
    else:
        winner = 'Tie'
        confidence = 0.5
    
    reasoning = f"对比分析：回答A长度{len(response_a)}字符，回答B长度{len(response_b)}字符。"
    reasoning += f"在{criteria}标准下，回答{winner}表现更佳。" if winner != 'Tie' else f"两个回答在{criteria}方面难分伯仲。"
    
    return {
        'winner': winner,
        'confidence': round(confidence, 2),
        'reasoning': reasoning,
        'score_a': round(score_a / 10, 1),
        'score_b': round(score_b / 10, 1)
    }


def create_bias_demo_chart() -> go.Figure:
    """
    创建偏差演示图表，展示position bias和verbosity bias
    
    Returns:
        Plotly图表
    """
    # 模拟偏差数据
    positions = ['A先', 'B先']
    
    # Position bias: 第一个回答更容易获胜
    position_bias_data = {
        'A先': {'A胜': 65, 'B胜': 25, '平局': 10},
        'B先': {'A胜': 35, 'B胜': 55, '平局': 10}
    }
    
    # Verbosity bias: 更长的回答更容易获胜
    length_ranges = ['短回答\n(< 100字)', '中等回答\n(100-300字)', '长回答\n(> 300字)']
    verbosity_scores = [6.2, 7.1, 7.8]
    
    # 创建子图
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Position Bias 演示', 'Verbosity Bias 演示'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Position bias 柱状图
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, outcome in enumerate(['A胜', 'B胜', '平局']):
        fig.add_trace(
            go.Bar(
                name=outcome,
                x=positions,
                y=[position_bias_data[pos][outcome] for pos in positions],
                marker_color=colors[i],
                text=[position_bias_data[pos][outcome] for pos in positions],
                textposition='outside'
            ),
            row=1, col=1
        )
    
    # Verbosity bias 散点图
    fig.add_trace(
        go.Scatter(
            x=length_ranges,
            y=verbosity_scores,
            mode='lines+markers',
            name='平均得分',
            line=dict(color='red', width=3),
            marker=dict(size=10, color='red'),
            text=[f'{score:.1f}' for score in verbosity_scores],
            textposition='top center',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 更新布局
    fig.update_layout(
        height=400,
        showlegend=True,
        title_text="LLM Judge 常见偏差演示",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="获胜比例(%)", row=1, col=1)
    fig.update_yaxes(title_text="平均得分", row=1, col=2)
    fig.update_xaxes(title_text="回答顺序", row=1, col=1)
    fig.update_xaxes(title_text="回答长度", row=1, col=2)
    
    return fig


def create_scoring_breakdown_chart(breakdown_data: Dict[str, float]) -> go.Figure:
    """
    创建评分细分图表
    
    Args:
        breakdown_data: 各维度评分数据
    
    Returns:
        Plotly雷达图
    """
    categories = list(breakdown_data.keys())
    values = list(breakdown_data.values())
    
    # 创建雷达图
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        name='评分',
        line=dict(color='#1f77b4', width=2),
        fillcolor='#1f77b4',
        opacity=0.3
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickfont=dict(size=12)
            )
        ),
        showlegend=False,
        title=dict(
            text="评分维度细分",
            x=0.5,
            font=dict(size=16)
        ),
        width=400,
        height=400
    )
    
    return fig


def render():
    """
    渲染 LLM-as-Judge 演示页面
    
    Returns:
        load 事件配置 (如果需要)
    """
    with gr.Column():
        gr.HTML("""
        <div class="main-header">
            <h1 style="color: #1f2937; margin-bottom: 8px;">⚖️ LLM-as-Judge 演示</h1>
            <p style="color: #6b7280; font-size: 1.1rem; margin: 0;">LLM 评分系统的工作原理和偏差分析</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Pointwise 评分
            with gr.Tab("Pointwise 评分"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 输入问题和回答")
                        
                        # 预设问题选择
                        example_selector = gr.Dropdown(
                            choices=[f"示例{i+1}: {conv['question']}" for i, conv in enumerate(EXAMPLE_CONVERSATIONS)],
                            value=f"示例1: {EXAMPLE_CONVERSATIONS[0]['question']}",
                            label="选择预设示例"
                        )
                        
                        question_input = gr.Textbox(
                            value=EXAMPLE_CONVERSATIONS[0]['question'],
                            label="问题",
                            lines=2
                        )
                        
                        response_input = gr.Textbox(
                            value=EXAMPLE_CONVERSATIONS[0]['response_a'],
                            label="回答",
                            lines=4
                        )
                        
                        # 评分标准选择
                        criteria_selector = gr.Dropdown(
                            choices=[(info['name'], key) for key, info in JUDGE_PROMPTS.items()],
                            value='helpfulness',
                            label="评分维度"
                        )
                        
                        judge_btn = gr.Button("开始评分", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 评分结果")
                        
                        score_display = gr.Number(
                            label="总分 (满分10分)",
                            value=0,
                            interactive=False
                        )
                        
                        reasoning_display = gr.Textbox(
                            label="评分理由",
                            lines=3,
                            interactive=False
                        )
                        
                        # 评分细分雷达图
                        breakdown_chart = gr.Plot(label="评分细分")
                
                # 评分标准说明
                criteria_info = gr.Markdown(
                    value=f"**{JUDGE_PROMPTS['helpfulness']['name']}**: {JUDGE_PROMPTS['helpfulness']['description']}\n\n" +
                          "\n".join([f"- {criterion}" for criterion in JUDGE_PROMPTS['helpfulness']['criteria']])
                )
            
            # Tab 2: Pairwise 评分 
            with gr.Tab("Pairwise 对比"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 输入问题和两个回答")
                        
                        # 预设问题选择
                        pair_example_selector = gr.Dropdown(
                            choices=[f"示例{i+1}: {conv['question']}" for i, conv in enumerate(EXAMPLE_CONVERSATIONS)],
                            value=f"示例1: {EXAMPLE_CONVERSATIONS[0]['question']}",
                            label="选择预设示例"
                        )
                        
                        pair_question_input = gr.Textbox(
                            value=EXAMPLE_CONVERSATIONS[0]['question'],
                            label="问题",
                            lines=2
                        )
                        
                        with gr.Row():
                            response_a_input = gr.Textbox(
                                value=EXAMPLE_CONVERSATIONS[0]['response_a'],
                                label="回答 A",
                                lines=4
                            )
                            
                            response_b_input = gr.Textbox(
                                value=EXAMPLE_CONVERSATIONS[0]['response_b'],
                                label="回答 B",
                                lines=4
                            )
                        
                        pair_criteria_selector = gr.Dropdown(
                            choices=[(info['name'], key) for key, info in JUDGE_PROMPTS.items()],
                            value='helpfulness',
                            label="评分维度"
                        )
                        
                        pair_judge_btn = gr.Button("开始对比评分", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 对比结果")
                        
                        winner_display = gr.Textbox(
                            label="获胜者",
                            interactive=False
                        )
                        
                        confidence_display = gr.Number(
                            label="置信度",
                            interactive=False
                        )
                        
                        pair_reasoning_display = gr.Textbox(
                            label="对比理由",
                            lines=4,
                            interactive=False
                        )
                        
                        with gr.Row():
                            score_a_display = gr.Number(
                                label="回答A评分",
                                interactive=False
                            )
                            score_b_display = gr.Number(
                                label="回答B评分", 
                                interactive=False
                            )
            
            # Tab 3: 偏差分析
            with gr.Tab("偏差分析"):
                gr.Markdown("""
                ### LLM Judge 常见偏差
                
                LLM作为评估器时会存在一些系统性偏差，了解这些偏差有助于设计更公平的评测流程。
                """)
                
                # 偏差演示图表
                bias_chart = gr.Plot(
                    value=create_bias_demo_chart(),
                    label="偏差演示"
                )
                
                gr.Markdown("""
                ### 偏差类型说明
                
                **Position Bias (位置偏差)**
                - LLM倾向于偏向第一个出现的回答
                - 解决方案：随机化回答顺序，多轮评测
                
                **Verbosity Bias (冗长偏差)**  
                - LLM倾向于给更长的回答打高分
                - 解决方案：在评分标准中强调简洁性，设置长度归一化
                
                **Self-Bias (自我偏差)**
                - LLM倾向于偏好与自己风格相似的回答
                - 解决方案：使用多个不同的judge模型
                
                **Anchoring Bias (锚定偏差)**
                - 评分会受到之前样本的影响
                - 解决方案：随机化评测顺序，提供评分参考标准
                """)
        
        # 事件处理函数
        def update_pointwise_example(example_choice):
            """更新pointwise示例"""
            idx = int(example_choice.split("示例")[1].split(":")[0]) - 1
            conv = EXAMPLE_CONVERSATIONS[idx]
            return conv['question'], conv['response_a']
        
        def update_pairwise_example(example_choice):
            """更新pairwise示例"""
            idx = int(example_choice.split("示例")[1].split(":")[0]) - 1
            conv = EXAMPLE_CONVERSATIONS[idx]
            return conv['question'], conv['response_a'], conv['response_b']
        
        def update_criteria_info(criteria):
            """更新评分标准说明"""
            info = JUDGE_PROMPTS[criteria]
            return f"**{info['name']}**: {info['description']}\n\n" + \
                   "\n".join([f"- {criterion}" for criterion in info['criteria']])
        
        def do_pointwise_judge(question, response, criteria):
            """执行pointwise评分"""
            result = mock_pointwise_judge(question, response, JUDGE_PROMPTS[criteria]['name'])
            chart = create_scoring_breakdown_chart(result['criteria_breakdown'])
            
            return (
                result['score'],
                result['reasoning'],
                chart
            )
        
        def do_pairwise_judge(question, response_a, response_b, criteria):
            """执行pairwise评分"""
            result = mock_pairwise_judge(question, response_a, response_b, JUDGE_PROMPTS[criteria]['name'])
            
            return (
                result['winner'],
                result['confidence'],
                result['reasoning'],
                result['score_a'],
                result['score_b']
            )
        
        # 事件绑定
        example_selector.change(
            fn=update_pointwise_example,
            inputs=[example_selector],
            outputs=[question_input, response_input]
        )
        
        pair_example_selector.change(
            fn=update_pairwise_example,
            inputs=[pair_example_selector],
            outputs=[pair_question_input, response_a_input, response_b_input]
        )
        
        criteria_selector.change(
            fn=update_criteria_info,
            inputs=[criteria_selector],
            outputs=[criteria_info]
        )
        
        judge_btn.click(
            fn=do_pointwise_judge,
            inputs=[question_input, response_input, criteria_selector],
            outputs=[score_display, reasoning_display, breakdown_chart]
        )
        
        pair_judge_btn.click(
            fn=do_pairwise_judge,
            inputs=[pair_question_input, response_a_input, response_b_input, pair_criteria_selector],
            outputs=[winner_display, confidence_display, pair_reasoning_display, score_a_display, score_b_display]
        )


if __name__ == "__main__":
    # 测试页面
    with gr.Blocks(title="LLM Judge Demo") as demo:
        render()
    
    demo.launch()