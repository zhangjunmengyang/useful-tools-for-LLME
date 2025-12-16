"""
格式化转换器 - SFT 数据格式转换
"""

import gradio as gr
import json
from data_lab.data_utils import CHAT_TEMPLATES, convert_to_format, validate_chat_format


def format_and_validate(input_json: str, target_format: str, system_prompt: str):
    """格式化并验证"""
    if not input_json or not input_json.strip():
        return "", "", ""
    
    try:
        data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return "", f"JSON 解析错误: {e}", ""
    
    converted = convert_to_format(data, target_format, system_prompt)
    
    validation = validate_chat_format(converted, target_format)
    
    if validation['valid']:
        validation_msg = "格式验证通过"
    else:
        issues = "\n".join([f"- {issue}" for issue in validation['issues']])
        validation_msg = f"格式问题:\n{issues}"
    
    return converted, validation_msg, ""


def render():
    """渲染页面"""
    
    gr.Markdown("## 格式化转换器")
    
    with gr.Row():
        with gr.Column():
            input_json = gr.Code(
                label="原始 JSON",
                value='''{
    "instruction": "将以下句子翻译成英文",
    "input": "今天天气真好",
    "output": "The weather is really nice today."
}''',
                language="json",
                lines=10
            )
        
        with gr.Column():
            format_choices = [(info['name'], fmt_id) for fmt_id, info in CHAT_TEMPLATES.items()]
            target_format = gr.Dropdown(
                label="目标格式",
                choices=format_choices,
                value="alpaca"
            )
            
            system_prompt = gr.Textbox(
                label="System Prompt (可选)",
                placeholder="自定义 system 提示词"
            )
            
            convert_btn = gr.Button("转换", variant="primary")
            
            output_text = gr.Code(
                label="转换结果",
                language=None,
                lines=10
            )
            
            validation_status = gr.Markdown("")
    
    # 事件绑定
    convert_btn.click(
        fn=format_and_validate,
        inputs=[input_json, target_format, system_prompt],
        outputs=[output_text, validation_status]
    )
    
    # 实时转换
    for comp in [input_json, target_format, system_prompt]:
        comp.change(
            fn=format_and_validate,
            inputs=[input_json, target_format, system_prompt],
            outputs=[output_text, validation_status]
        )
