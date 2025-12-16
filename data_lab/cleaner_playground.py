"""
数据清洗 - 测试清洗规则和 PPL 过滤
"""

import gradio as gr
import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from data_lab.data_utils import (
    CLEANING_RULES, 
    normalize_unicode,
    PPL_MODELS,
    calculate_perplexity,
    get_ppl_quality_label
)


def render_ppl_histogram(ppl_values: list, threshold_max: float) -> go.Figure:
    """渲染 PPL 分布直方图"""
    valid_values = [v for v in ppl_values if v != float('inf') and v < 10000]
    
    if not valid_values:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=valid_values,
        nbinsx=30,
        marker_color='#2563EB',
        opacity=0.7,
        name='PPL 分布'
    ))
    
    fig.add_vline(
        x=threshold_max, 
        line_dash="dash", 
        line_color="#DC2626",
        annotation_text=f"阈值: {threshold_max}"
    )
    
    reference_lines = [
        (50, "优秀", "#059669"),
        (100, "良好", "#2563EB"),
        (300, "一般", "#D97706"),
    ]
    
    for val, label, color in reference_lines:
        if val < max(valid_values):
            fig.add_vline(
                x=val,
                line_dash="dot",
                line_color=color,
                annotation_text=label,
                annotation_position="top"
            )
    
    fig.update_layout(
        title="PPL 分布直方图",
        xaxis_title="Perplexity",
        yaxis_title="样本数",
        height=400,
        autosize=True,
        showlegend=False
    )
    
    return fig


def apply_cleaning(text: str, rules: list, unicode_form: str, custom_pattern: str, custom_replacement: str):
    """应用清洗规则"""
    if not text:
        return "", 0, 0, "0.0%"
    
    cleaned = text
    original_len = len(text)
    
    # 应用选中的规则
    for rule_id in rules:
        if rule_id in CLEANING_RULES:
            rule = CLEANING_RULES[rule_id]
            cleaned = re.sub(rule['pattern'], rule['replacement'], cleaned)
    
    # Unicode 规范化
    if unicode_form and unicode_form != "无":
        cleaned = normalize_unicode(cleaned, unicode_form)
    
    # 自定义正则
    if custom_pattern:
        try:
            cleaned = re.sub(custom_pattern, custom_replacement or "", cleaned)
        except re.error:
            pass
    
    cleaned = cleaned.strip()
    cleaned_len = len(cleaned)
    reduction = (1 - cleaned_len / original_len) * 100 if original_len else 0
    
    return cleaned, original_len, cleaned_len, f"{reduction:.1f}%"


# PPL 计算的全局模型缓存
_ppl_model_cache = {'model': None, 'tokenizer': None, 'name': None}


def load_ppl_model(model_choice: str, progress=gr.Progress()):
    """加载 PPL 计算模型"""
    global _ppl_model_cache
    
    if _ppl_model_cache['name'] == model_choice:
        return f"模型 {model_choice} 已加载"
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_info = PPL_MODELS[model_choice]
        progress(0.3, desc=f"加载 {model_choice}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_info['id'])
        model = AutoModelForCausalLM.from_pretrained(
            model_info['id'],
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model.eval()
        
        _ppl_model_cache['model'] = model
        _ppl_model_cache['tokenizer'] = tokenizer
        _ppl_model_cache['name'] = model_choice
        
        return f"模型 {model_choice} 加载成功"
        
    except ImportError:
        return "请确保已安装 transformers 和 torch 库"
    except Exception as e:
        return f"加载失败: {str(e)}"


def calculate_single_ppl(text: str, min_ppl: float, max_ppl: float):
    """计算单条文本的 PPL"""
    if not _ppl_model_cache['model']:
        return "请先加载模型", "", "", "", ""
    
    if not text or not text.strip():
        return "请输入文本", "", "", "", ""
    
    try:
        ppl, details = calculate_perplexity(
            text, 
            _ppl_model_cache['model'], 
            _ppl_model_cache['tokenizer']
        )
        
        label, color = get_ppl_quality_label(ppl)
        
        result_html = f"""
        <div style="background: linear-gradient(90deg, {color}22, transparent); 
            padding: 15px; border-radius: 8px; border-left: 4px solid {color}; margin: 10px 0;">
            <span style="color: {color}; font-size: 24px; font-weight: bold;">
                PPL = {ppl:.2f}
            </span>
            <span style="margin-left: 20px; color: #6B7280;">
                质量: <b>{label}</b>
            </span>
        </div>
        """
        
        if min_ppl <= ppl <= max_ppl:
            filter_status = f"通过过滤 (PPL 在 {min_ppl} - {max_ppl} 范围内)"
        else:
            filter_status = f"被过滤 (PPL 超出 {min_ppl} - {max_ppl} 范围)"
        
        return (
            result_html,
            f"{ppl:.2f}",
            label,
            str(details.get('seq_length', 'N/A')),
            filter_status
        )
        
    except Exception as e:
        return f"计算失败: {str(e)}", "", "", "", ""


def calculate_batch_ppl(texts: str, min_ppl: float, max_ppl: float, progress=gr.Progress()):
    """批量计算 PPL"""
    if not _ppl_model_cache['model']:
        return "请先加载模型", None, None, "", "", "", ""
    
    text_list = [t.strip() for t in texts.strip().split('\n') if t.strip()]
    
    if not text_list:
        return "请输入文本", None, None, "", "", "", ""
    
    try:
        results = []
        for i, text in enumerate(text_list):
            progress((i + 1) / len(text_list), desc=f"计算 {i + 1}/{len(text_list)}...")
            
            ppl, details = calculate_perplexity(
                text, 
                _ppl_model_cache['model'], 
                _ppl_model_cache['tokenizer']
            )
            label, _ = get_ppl_quality_label(ppl)
            accepted = min_ppl <= ppl <= max_ppl
            
            results.append({
                "文本": text[:50] + "..." if len(text) > 50 else text,
                "PPL": ppl,
                "评级": label,
                "通过": "Yes" if accepted else "No",
                "长度": len(text)
            })
        
        # 统计
        ppl_values = [r["PPL"] for r in results]
        accepted_count = sum(1 for r in results if r["通过"] == "Yes")
        valid_ppl = [p for p in ppl_values if p != float('inf')]
        avg_ppl = np.mean(valid_ppl) if valid_ppl else 0
        
        # 分布图
        fig = render_ppl_histogram(ppl_values, max_ppl)
        
        # 结果表
        df = pd.DataFrame(results)
        
        return (
            "计算完成",
            fig,
            df,
            str(len(results)),
            str(accepted_count),
            str(len(results) - accepted_count),
            f"{avg_ppl:.1f}"
        )
        
    except Exception as e:
        return f"计算失败: {str(e)}", None, None, "", "", "", ""


def render():
    """渲染页面"""
    
    gr.Markdown("## 数据清洗")
    
    with gr.Tabs():
        # Tab 1: 规则清洗
        with gr.Tab("规则清洗"):
            gr.Markdown("### 清洗规则")
            
            rule_choices = [(info['name'], rule_id) for rule_id, info in CLEANING_RULES.items()]
            selected_rules = gr.CheckboxGroup(
                label="选择清洗规则",
                choices=rule_choices,
                value=[rule_id for rule_id in CLEANING_RULES.keys()]
            )
            
            with gr.Row():
                unicode_form = gr.Dropdown(
                    label="Unicode 规范化",
                    choices=["无", "NFC", "NFD", "NFKC", "NFKD"],
                    value="无"
                )
                custom_pattern = gr.Textbox(
                    label="自定义正则 Pattern",
                    placeholder=r"如 \d+"
                )
                custom_replacement = gr.Textbox(
                    label="替换为",
                    placeholder="替换文本"
                )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 输入 (脏数据)")
                    dirty_text = gr.Textbox(
                        label="原始文本",
                        value="""<p>这是一段 HTML 文本</p>
访问 https://example.com 了解更多
联系邮箱: test@email.com
包含   多余   空格
特殊符号★☆♠♣""",
                        lines=8
                    )
                
                with gr.Column():
                    gr.Markdown("### 输出 (清洗后)")
                    cleaned_text = gr.Textbox(
                        label="清洗结果",
                        lines=8,
                        interactive=False
                    )
                    
                    gr.Markdown("### 统计")
                    with gr.Row():
                        orig_len = gr.Textbox(label="原始长度", interactive=False)
                        clean_len = gr.Textbox(label="清洗后长度", interactive=False)
                        reduction_pct = gr.Textbox(label="缩减比例", interactive=False)
            
            # 实时清洗
            for input_comp in [dirty_text, selected_rules, unicode_form, custom_pattern, custom_replacement]:
                input_comp.change(
                    fn=apply_cleaning,
                    inputs=[dirty_text, selected_rules, unicode_form, custom_pattern, custom_replacement],
                    outputs=[cleaned_text, orig_len, clean_len, reduction_pct]
                )
        
        # Tab 2: PPL 过滤
        with gr.Tab("PPL 过滤"):
            with gr.Row():
                model_choice = gr.Dropdown(
                    label="选择 PPL 计算模型",
                    choices=list(PPL_MODELS.keys()),
                    value=list(PPL_MODELS.keys())[0]
                )
                load_model_btn = gr.Button("加载模型", variant="primary")
            
            model_status = gr.Markdown("")
            
            with gr.Row():
                min_ppl = gr.Number(label="最小 PPL", value=0.0, minimum=0.0)
                max_ppl = gr.Number(label="最大 PPL", value=500.0, minimum=1.0)
            
            gr.Markdown("---")
            
            input_mode = gr.Radio(
                label="输入模式",
                choices=["单条文本", "批量文本"],
                value="单条文本"
            )
            
            # 单条文本模式
            with gr.Group(visible=True) as single_mode:
                single_text = gr.Textbox(
                    label="输入文本",
                    value="The quick brown fox jumps over the lazy dog. This is a test sentence for perplexity calculation.",
                    lines=4
                )
                single_calc_btn = gr.Button("计算 PPL", variant="primary")
                
                single_result = gr.HTML("")
                with gr.Row():
                    single_ppl = gr.Textbox(label="Perplexity", interactive=False)
                    single_label = gr.Textbox(label="质量评级", interactive=False)
                    single_seq_len = gr.Textbox(label="序列长度", interactive=False)
                single_filter_status = gr.Textbox(label="过滤状态", interactive=False)
            
            # 批量文本模式
            with gr.Group(visible=False) as batch_mode:
                batch_text = gr.Textbox(
                    label="批量输入 (每行一条)",
                    value="""The weather is nice today.
This is a normal English sentence.
asdfjkl qwerty random gibberish text
机器学习是人工智能的一个重要分支。
!!!@@@###$$$%%%^^^&&&***
She went to the store to buy some groceries for dinner.""",
                    lines=8
                )
                batch_calc_btn = gr.Button("批量计算 PPL", variant="primary")
                
                batch_status = gr.Markdown("")
                
                with gr.Row():
                    total_count = gr.Textbox(label="总样本数", interactive=False)
                    pass_count = gr.Textbox(label="通过数", interactive=False)
                    filter_count = gr.Textbox(label="过滤数", interactive=False)
                    avg_ppl = gr.Textbox(label="平均 PPL", interactive=False)
                
                ppl_chart = gr.Plot(label="PPL 分布")
                batch_results = gr.Dataframe(label="详细结果")
            
            # 切换模式
            def toggle_mode(mode):
                return (
                    gr.update(visible=(mode == "单条文本")),
                    gr.update(visible=(mode == "批量文本"))
                )
            
            input_mode.change(
                fn=toggle_mode,
                inputs=[input_mode],
                outputs=[single_mode, batch_mode]
            )
            
            # 事件绑定
            load_model_btn.click(
                fn=load_ppl_model,
                inputs=[model_choice],
                outputs=[model_status]
            )
            
            single_calc_btn.click(
                fn=calculate_single_ppl,
                inputs=[single_text, min_ppl, max_ppl],
                outputs=[single_result, single_ppl, single_label, single_seq_len, single_filter_status]
            )
            
            batch_calc_btn.click(
                fn=calculate_batch_ppl,
                inputs=[batch_text, min_ppl, max_ppl],
                outputs=[batch_status, ppl_chart, batch_results, total_count, pass_count, filter_count, avg_ppl]
            )
