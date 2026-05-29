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

from workbench_theme import OPEN_DESIGN_COLORS, PLOTLY_COLORWAY


# 预设的评测prompt模板
JUDGE_PROMPTS = {
    'helpfulness': {
        'name': 'Helpfulness',
        'description': 'Evaluate how useful the answer is for the user question',
        'criteria': [
            'Directly answers the user question',
            'Provides useful and accurate information',
            'Is complete and easy to understand',
            'Includes relevant examples or explanations'
        ]
    },
    'accuracy': {
        'name': 'Accuracy',
        'description': 'Evaluate factual correctness',
        'criteria': [
            'Information is factually correct',
            'Claims are grounded in reliable evidence',
            'Avoids misleading statements',
            'Acknowledges uncertainty'
        ]
    },
    'safety': {
        'name': 'Safety',
        'description': 'Evaluate whether the answer is safe and harmless',
        'criteria': [
            'Avoids harmful or dangerous advice',
            'Does not contain discriminatory content',
            'Respects user privacy',
            'Follows ethical guidelines'
        ]
    },
    'creativity': {
        'name': 'Creativity',
        'description': 'Evaluate novelty and creative quality',
        'criteria': [
            'Offers novel viewpoints or methods',
            'Shows imagination and originality',
            'Moves beyond conventional framing',
            'Encourages further thinking'
        ]
    }
}

# 预设示例对话
EXAMPLE_CONVERSATIONS = [
    {
        'question': 'How should I learn machine learning?',
        'response_a': 'Start with Python, linear algebra, statistics, and calculus. Then learn scikit-learn, pandas, and model evaluation through small projects. A structured course plus hands-on practice is the most reliable path.',
        'response_b': 'Just watch videos and copy a few projects. The math does not matter much because libraries already do everything.'
    },
    {
        'question': 'How can Python handle large datasets?',
        'response_a': 'Use chunked pandas reads for medium data, Dask for parallel local workflows, Spark for distributed processing, and databases or warehouses when the dataset is larger than memory. The right choice depends on data size and compute limits.',
        'response_b': 'Use pandas read_csv directly. If memory is not enough, buy more memory. Most computers are powerful now.'
    },
    {
        'question': 'How can I improve code quality?',
        'response_a': 'Use clear names, consistent formatting, unit tests, code review, linters, type checks, small refactors, and concise documentation. Quality comes from repeatable engineering habits, not a single cleanup pass.',
        'response_b': 'If the code runs, it is fine. Formatting and tests are not important because you can manually check it.'
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
    keywords = ['use', 'method', 'specific', 'example', 'practice', 'step', 'reliable', 'depends']
    keyword_count = sum(1 for kw in keywords if kw in response)
    
    # 基础分数 + 长度奖励 + 关键词奖励
    base_score = random.uniform(6.0, 8.0)
    length_bonus = min(2.0, response_length / 100)  
    keyword_bonus = keyword_count * 0.3
    
    score = min(10.0, base_score + length_bonus + keyword_bonus)
    
    # 生成模拟的评分理由
    reasoning = f"Score rationale: response length is {response_length} characters, "
    if keyword_count > 2:
        reasoning += f"with {keyword_count} guidance terms, "
    reasoning += f"and the answer is {'strong' if score > 7 else 'adequate'} for {criteria}."
    
    return {
        'score': round(score, 1),
        'reasoning': reasoning,
        'criteria_breakdown': {
            'Relevance': random.uniform(7, 9),
            'Completeness': random.uniform(6, 8),
            'Clarity': random.uniform(7, 9),
            'Usefulness': random.uniform(6, 8)
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
    keywords = ['use', 'method', 'specific', 'example', 'practice', 'step', 'can', 'should', 'depends']
    
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
    
    reasoning = f"Comparison: answer A has {len(response_a)} characters; answer B has {len(response_b)} characters. "
    reasoning += f"Under {criteria}, answer {winner} is stronger." if winner != 'Tie' else f"The two answers are close under {criteria}."
    
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
    positions = ['A first', 'B first']
    
    # Position bias: 第一个回答更容易获胜
    position_bias_data = {
        'A first': {'A wins': 65, 'B wins': 25, 'Tie': 10},
        'B first': {'A wins': 35, 'B wins': 55, 'Tie': 10}
    }
    
    # Verbosity bias: 更长的回答更容易获胜
    length_ranges = ['Short\n(< 100 words)', 'Medium\n(100-300 words)', 'Long\n(> 300 words)']
    verbosity_scores = [6.2, 7.1, 7.8]
    
    # 创建子图
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Position Bias', 'Verbosity Bias'),
        specs=[[{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Position bias 柱状图
    colors = PLOTLY_COLORWAY
    for i, outcome in enumerate(['A wins', 'B wins', 'Tie']):
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
            name='Average score',
            line=dict(color=OPEN_DESIGN_COLORS["accent"], width=3),
            marker=dict(size=10, color=OPEN_DESIGN_COLORS["accent"]),
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
        title_text="Common LLM Judge Biases",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Win rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Average score", row=1, col=2)
    fig.update_xaxes(title_text="Answer order", row=1, col=1)
    fig.update_xaxes(title_text="Answer length", row=1, col=2)
    
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
        name='Score',
        line=dict(color=OPEN_DESIGN_COLORS["accent"], width=2),
        fillcolor=OPEN_DESIGN_COLORS["accent"],
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
            text="Scoring Breakdown",
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
        <div class="workbench-page-hero">
            <h1>LLM-as-Judge</h1>
            <p>Explore pointwise scoring, pairwise comparison, rubric dimensions, and common judge biases.</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Pointwise 评分
            with gr.Tab("Pointwise Scoring"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Question and Answer")
                        
                        # 预设问题选择
                        example_selector = gr.Dropdown(
                            choices=[f"Example {i+1}: {conv['question']}" for i, conv in enumerate(EXAMPLE_CONVERSATIONS)],
                            value=f"Example 1: {EXAMPLE_CONVERSATIONS[0]['question']}",
                            label="Preset Example"
                        )
                        
                        question_input = gr.Textbox(
                            value=EXAMPLE_CONVERSATIONS[0]['question'],
                            label="Question",
                            lines=2
                        )
                        
                        response_input = gr.Textbox(
                            value=EXAMPLE_CONVERSATIONS[0]['response_a'],
                            label="Answer",
                            lines=4
                        )
                        
                        # 评分标准选择
                        criteria_selector = gr.Dropdown(
                            choices=[(info['name'], key) for key, info in JUDGE_PROMPTS.items()],
                            value='helpfulness',
                            label="Rubric"
                        )
                        
                        judge_btn = gr.Button("Score Answer", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Score Result")
                        
                        score_display = gr.Number(
                            label="Overall Score (0-10)",
                            value=0,
                            interactive=False
                        )
                        
                        reasoning_display = gr.Textbox(
                            label="Reasoning",
                            lines=3,
                            interactive=False
                        )
                        
                        # 评分细分雷达图
                        breakdown_chart = gr.Plot(label="Scoring Breakdown")
                
                # 评分标准说明
                criteria_info = gr.Markdown(
                    value=f"**{JUDGE_PROMPTS['helpfulness']['name']}**: {JUDGE_PROMPTS['helpfulness']['description']}\n\n" +
                          "\n".join([f"- {criterion}" for criterion in JUDGE_PROMPTS['helpfulness']['criteria']])
                )
            
            # Tab 2: Pairwise 评分 
            with gr.Tab("Pairwise Comparison"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### Question and Two Answers")
                        
                        # 预设问题选择
                        pair_example_selector = gr.Dropdown(
                            choices=[f"Example {i+1}: {conv['question']}" for i, conv in enumerate(EXAMPLE_CONVERSATIONS)],
                            value=f"Example 1: {EXAMPLE_CONVERSATIONS[0]['question']}",
                            label="Preset Example"
                        )
                        
                        pair_question_input = gr.Textbox(
                            value=EXAMPLE_CONVERSATIONS[0]['question'],
                            label="Question",
                            lines=2
                        )
                        
                        with gr.Row():
                            response_a_input = gr.Textbox(
                                value=EXAMPLE_CONVERSATIONS[0]['response_a'],
                                label="Answer A",
                                lines=4
                            )
                            
                            response_b_input = gr.Textbox(
                                value=EXAMPLE_CONVERSATIONS[0]['response_b'],
                                label="Answer B",
                                lines=4
                            )
                        
                        pair_criteria_selector = gr.Dropdown(
                            choices=[(info['name'], key) for key, info in JUDGE_PROMPTS.items()],
                            value='helpfulness',
                            label="Rubric"
                        )
                        
                        pair_judge_btn = gr.Button("Compare Answers", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### Comparison Result")
                        
                        winner_display = gr.Textbox(
                            label="Winner",
                            interactive=False
                        )
                        
                        confidence_display = gr.Number(
                            label="Confidence",
                            interactive=False
                        )
                        
                        pair_reasoning_display = gr.Textbox(
                            label="Reasoning",
                            lines=4,
                            interactive=False
                        )
                        
                        with gr.Row():
                            score_a_display = gr.Number(
                                label="Answer A Score",
                                interactive=False
                            )
                            score_b_display = gr.Number(
                                label="Answer B Score",
                                interactive=False
                            )
            
            # Tab 3: 偏差分析
            with gr.Tab("Bias Analysis"):
                gr.Markdown("""
                ### Common LLM Judge Biases
                
                LLM evaluators can show systematic biases. Understanding them helps design fairer evaluation workflows.
                """)
                
                # 偏差演示图表
                bias_chart = gr.Plot(
                    value=create_bias_demo_chart(),
                    label="Bias Demo"
                )
                
                gr.Markdown("""
                ### Bias Types
                
                **Position Bias**
                - The judge tends to prefer the answer shown first.
                - Mitigation: randomize answer order and run multiple passes.
                
                **Verbosity Bias**
                - The judge tends to reward longer answers.
                - Mitigation: include concision in the rubric and normalize for length.
                
                **Self-Bias**
                - The judge tends to prefer answers similar to its own style.
                - Mitigation: use multiple judge models.
                
                **Anchoring Bias**
                - Scores can be influenced by previous samples.
                - Mitigation: randomize evaluation order and provide score anchors.
                """)
        
        # 事件处理函数
        def update_pointwise_example(example_choice):
            """更新pointwise示例"""
            idx = int(example_choice.split("Example ")[1].split(":")[0]) - 1
            conv = EXAMPLE_CONVERSATIONS[idx]
            return conv['question'], conv['response_a']
        
        def update_pairwise_example(example_choice):
            """更新pairwise示例"""
            idx = int(example_choice.split("Example ")[1].split(":")[0]) - 1
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
