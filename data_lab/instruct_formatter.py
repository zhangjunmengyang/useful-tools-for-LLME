"""
格式化转换器 - SFT 数据格式转换
"""

import gradio as gr
import json
from data_lab.data_utils import CHAT_TEMPLATES, convert_to_format, validate_chat_format


def format_and_validate(input_json: str, target_format: str, system_prompt: str):
    """格式化并验证"""
    if not input_json or not input_json.strip():
        return "", ""
    
    try:
        data = json.loads(input_json)
    except json.JSONDecodeError as e:
        return "", f"JSON parse error: {e}"

    converted = convert_to_format(data, target_format, system_prompt)

    validation = validate_chat_format(converted, target_format)

    if validation['valid']:
        validation_msg = "Format validation passed"
    else:
        issues = "\n".join([f"- {issue}" for issue in validation['issues']])
        validation_msg = f"Format issues:\n{issues}"
    
    return converted, validation_msg


def render():
    """渲染页面"""

    with gr.Row():
        with gr.Column():
            input_json = gr.Code(
                label="Source JSON",
                value='''{
    "instruction": "Translate the following sentence into English",
    "input": "The weather is nice today",
    "output": "The weather is really nice today."
}''',
                language="json",
                lines=10
            )

        with gr.Column():
            format_choices = [(info['name'], fmt_id) for fmt_id, info in CHAT_TEMPLATES.items()]
            target_format = gr.Dropdown(
                label="Target Format",
                choices=format_choices,
                value="alpaca"
            )

            system_prompt = gr.Textbox(
                label="System Prompt (Optional)",
                placeholder="Custom system prompt"
            )

            convert_btn = gr.Button("Convert", variant="primary")

            output_text = gr.Code(
                label="Converted Result",
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
