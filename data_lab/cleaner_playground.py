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
        name='PPL Distribution'
    ))

    fig.add_vline(
        x=threshold_max,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Threshold: {threshold_max}"
    )

    reference_lines = [
        (50, "Excellent", "#059669"),
        (100, "Good", "#2563EB"),
        (300, "Average", "#D97706"),
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
        title="PPL Distribution Histogram",
        xaxis_title="Perplexity",
        yaxis_title="Sample Count",
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
    if unicode_form and unicode_form != "None":
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
        return f"Model {model_choice} already loaded"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_info = PPL_MODELS[model_choice]
        progress(0.3, desc=f"Loading {model_choice}...")

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

        return f"Model {model_choice} loaded successfully"

    except ImportError:
        return "Please ensure transformers and torch libraries are installed"
    except Exception as e:
        return f"Loading failed: {str(e)}"


def calculate_single_ppl(text: str, min_ppl: float, max_ppl: float):
    """计算单条文本的 PPL"""
    if not _ppl_model_cache['model']:
        return "Please load a model first", "", "", "", ""

    if not text or not text.strip():
        return "Please enter text", "", "", "", ""

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
                Quality: <b>{label}</b>
            </span>
        </div>
        """

        if min_ppl <= ppl <= max_ppl:
            filter_status = f"Passed (PPL within {min_ppl} - {max_ppl} range)"
        else:
            filter_status = f"Filtered (PPL outside {min_ppl} - {max_ppl} range)"

        return (
            result_html,
            f"{ppl:.2f}",
            label,
            str(details.get('seq_length', 'N/A')),
            filter_status
        )

    except Exception as e:
        return f"Calculation failed: {str(e)}", "", "", "", ""


def calculate_batch_ppl(texts: str, min_ppl: float, max_ppl: float, progress=gr.Progress()):
    """批量计算 PPL"""
    if not _ppl_model_cache['model']:
        return "Please load a model first", None, None, "", "", "", ""

    text_list = [t.strip() for t in texts.strip().split('\n') if t.strip()]

    if not text_list:
        return "Please enter text", None, None, "", "", "", ""

    try:
        results = []
        for i, text in enumerate(text_list):
            progress((i + 1) / len(text_list), desc=f"Calculating {i + 1}/{len(text_list)}...")

            ppl, details = calculate_perplexity(
                text,
                _ppl_model_cache['model'],
                _ppl_model_cache['tokenizer']
            )
            label, _ = get_ppl_quality_label(ppl)
            accepted = min_ppl <= ppl <= max_ppl

            results.append({
                "Text": text[:50] + "..." if len(text) > 50 else text,
                "PPL": ppl,
                "Rating": label,
                "Passed": "Yes" if accepted else "No",
                "Length": len(text)
            })

        # 统计
        ppl_values = [r["PPL"] for r in results]
        accepted_count = sum(1 for r in results if r["Passed"] == "Yes")
        valid_ppl = [p for p in ppl_values if p != float('inf')]
        avg_ppl = np.mean(valid_ppl) if valid_ppl else 0

        # 分布图
        fig = render_ppl_histogram(ppl_values, max_ppl)

        # 结果表
        df = pd.DataFrame(results)

        return (
            "Calculation complete",
            fig,
            df,
            str(len(results)),
            str(accepted_count),
            str(len(results) - accepted_count),
            f"{avg_ppl:.1f}"
        )

    except Exception as e:
        return f"Calculation failed: {str(e)}", None, None, "", "", "", ""


def render():
    """渲染页面"""

    with gr.Tabs():
        # Tab 1: Rule-based Cleaning
        with gr.Tab("Rule Cleaning"):
            gr.Markdown("### Cleaning Rules")

            rule_choices = [(info['name'], rule_id) for rule_id, info in CLEANING_RULES.items()]
            selected_rules = gr.CheckboxGroup(
                label="Select Cleaning Rules",
                choices=rule_choices,
                value=[rule_id for rule_id in CLEANING_RULES.keys()]
            )

            with gr.Row():
                unicode_form = gr.Dropdown(
                    label="Unicode Normalization",
                    choices=["None", "NFC", "NFD", "NFKC", "NFKD"],
                    value="None"
                )
                custom_pattern = gr.Textbox(
                    label="Custom Regex Pattern",
                    placeholder=r"e.g., \d+"
                )
                custom_replacement = gr.Textbox(
                    label="Replace With",
                    placeholder="Replacement text"
                )

            gr.Markdown("---")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Input (Dirty Data)")
                    dirty_text = gr.Textbox(
                        label="Original Text",
                        value="""<p>This is HTML text</p>
Visit https://example.com for more
Contact: test@email.com
Contains   extra   spaces
Special symbols: @#$%^&*()[]{}~`|\\/<>★※§¶†‡""",
                        lines=8
                    )

                with gr.Column():
                    gr.Markdown("### Output (Cleaned)")
                    cleaned_text = gr.Textbox(
                        label="Cleaned Result",
                        lines=8,
                        interactive=False
                    )

                    gr.Markdown("### Statistics")
                    with gr.Row():
                        orig_len = gr.Textbox(label="Original Length", interactive=False)
                        clean_len = gr.Textbox(label="Cleaned Length", interactive=False)
                        reduction_pct = gr.Textbox(label="Reduction Ratio", interactive=False)
            
            # 实时清洗
            for input_comp in [dirty_text, selected_rules, unicode_form, custom_pattern, custom_replacement]:
                input_comp.change(
                    fn=apply_cleaning,
                    inputs=[dirty_text, selected_rules, unicode_form, custom_pattern, custom_replacement],
                    outputs=[cleaned_text, orig_len, clean_len, reduction_pct]
                )
        
        # Tab 2: PPL Filtering
        with gr.Tab("PPL Filter"):
            with gr.Row():
                model_choice = gr.Dropdown(
                    label="Select PPL Model",
                    choices=list(PPL_MODELS.keys()),
                    value=list(PPL_MODELS.keys())[0]
                )
                load_model_btn = gr.Button("Load Model", variant="primary")

            model_status = gr.Markdown("")

            with gr.Row():
                min_ppl = gr.Number(label="Min PPL", value=0.0, minimum=0.0)
                max_ppl = gr.Number(label="Max PPL", value=500.0, minimum=1.0)

            gr.Markdown("---")

            input_mode = gr.Radio(
                label="Input Mode",
                choices=["Single Text", "Batch Text"],
                value="Single Text"
            )

            # 单条文本模式
            with gr.Group(visible=True) as single_mode:
                single_text = gr.Textbox(
                    label="Input Text",
                    value="The quick brown fox jumps over the lazy dog. This is a test sentence for perplexity calculation.",
                    lines=4
                )
                single_calc_btn = gr.Button("Calculate PPL", variant="primary")

                single_result = gr.HTML("")
                with gr.Row():
                    single_ppl = gr.Textbox(label="Perplexity", interactive=False)
                    single_label = gr.Textbox(label="Quality Rating", interactive=False)
                    single_seq_len = gr.Textbox(label="Sequence Length", interactive=False)
                single_filter_status = gr.Textbox(label="Filter Status", interactive=False)

            # 批量文本模式
            with gr.Group(visible=False) as batch_mode:
                batch_text = gr.Textbox(
                    label="Batch Input (one per line)",
                    value="""The weather is nice today.
This is a normal English sentence.
asdfjkl qwerty random gibberish text
Machine learning is an important branch of artificial intelligence.
!!!@@@###$$$%%%^^^&&&***
She went to the store to buy some groceries for dinner.""",
                    lines=8
                )
                batch_calc_btn = gr.Button("Batch Calculate PPL", variant="primary")

                batch_status = gr.Markdown("")

                with gr.Row():
                    total_count = gr.Textbox(label="Total Samples", interactive=False)
                    pass_count = gr.Textbox(label="Passed", interactive=False)
                    filter_count = gr.Textbox(label="Filtered", interactive=False)
                    avg_ppl = gr.Textbox(label="Average PPL", interactive=False)

                ppl_chart = gr.Plot(label="PPL Distribution")
                batch_results = gr.Dataframe(label="Detailed Results")

            # 切换模式
            def toggle_mode(mode):
                return (
                    gr.update(visible=(mode == "Single Text")),
                    gr.update(visible=(mode == "Batch Text"))
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
