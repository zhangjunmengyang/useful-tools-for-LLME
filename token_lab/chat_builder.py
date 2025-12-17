"""
Chat Template - Gradio Version
Chat template debugger and visualizer
"""

import gradio as gr
import json
from token_lab.tokenizer_utils import (
    load_tokenizer,
    apply_chat_template_safe,
    get_token_info,
    get_models_by_category,
    MODEL_CATEGORIES,
    TOKEN_COLORS
)


def get_category_choices():
    """获取模型厂商列表"""
    return list(MODEL_CATEGORIES.keys())


def get_model_choices(category):
    """根据厂商获取模型列表"""
    if not category:
        return []
    models = get_models_by_category(category)
    return [name for name, _ in models]


def get_model_id(category, model_name):
    """获取模型 ID"""
    if not category or not model_name:
        return None
    models = get_models_by_category(category)
    for name, model_id in models:
        if name == model_name:
            return model_id
    return None


def render_template_output_html(template_str, tokenizer):
    """渲染模版输出为 HTML"""
    if not template_str:
        return "<div style='color: #6B7280;'>Waiting for input...</div>"

    special_tokens = set()
    if hasattr(tokenizer, 'all_special_tokens'):
        special_tokens.update(tokenizer.all_special_tokens)
    if hasattr(tokenizer, 'additional_special_tokens') and tokenizer.additional_special_tokens:
        special_tokens.update(tokenizer.additional_special_tokens)

    sorted_special = sorted(special_tokens, key=len, reverse=True)

    result_html = [
        '<div style="font-family: \'JetBrains Mono\', monospace; white-space: pre-wrap; '
        'background: #F3F4F6; padding: 16px; border-radius: 8px; '
        'line-height: 1.8; font-size: 13px; border: 1px solid #E5E7EB; color: #111827;">'
    ]

    i = 0
    while i < len(template_str):
        found = False
        for special in sorted_special:
            if template_str[i:].startswith(special):
                escaped = special.replace('<', '&lt;').replace('>', '&gt;')
                result_html.append(
                    f'<span style="background: #FEE2E2; color: #DC2626; '
                    f'padding: 2px 6px; border-radius: 4px; border: 1px solid #FECACA; font-weight: 500;">'
                    f'{escaped}</span>'
                )
                i += len(special)
                found = True
                break

        if not found:
            char = template_str[i]
            if char == '<':
                result_html.append('&lt;')
            elif char == '>':
                result_html.append('&gt;')
            elif char == '\n':
                result_html.append('<br/>')
            elif char == ' ':
                result_html.append('&nbsp;')
            else:
                result_html.append(char)
            i += 1

    result_html.append('</div>')
    return ''.join(result_html)


def render_token_sequence_html(token_info, max_tokens=200):
    """渲染 Token 序列为 HTML"""
    html = ['<div style="line-height: 2.4; padding: 12px; background: #F8FAFC; border-radius: 8px;">']

    for idx, info in enumerate(token_info[:max_tokens]):
        is_special = info.get('is_special', False)
        bg = '#FEE2E2' if is_special else TOKEN_COLORS[idx % len(TOKEN_COLORS)]
        border = 'border: 1px solid #DC2626;' if is_special else ''

        display = info['token_str'].replace(' ', '␣').replace('\n', '↵')
        if not display.strip():
            display = repr(info['token_str'])[1:-1] or '[E]'
        display = display.replace('<', '&lt;').replace('>', '&gt;')[:20]

        html.append(
            f'<span style="background:{bg}; color:#111827; '
            f'padding:2px 5px; margin:1px; border-radius:3px; display:inline-block; '
            f'font-family: \'JetBrains Mono\', monospace; font-size:12px; {border}">{display}</span>'
        )

    if len(token_info) > max_tokens:
        html.append(f'<span style="color:#6B7280;"> ... +{len(token_info)-max_tokens} more</span>')

    html.append('</div>')
    return ''.join(html)


def apply_template(category, model_name, json_input, add_gen_prompt):
    """应用 Chat Template"""
    if not category or not model_name:
        return (
            "<div style='color: #DC2626;'>Please select a model</div>",
            "", "", "",
            "", "", ""
        )

    model_id = get_model_id(category, model_name)
    if not model_id:
        return (
            "<div style='color: #DC2626;'>Model not found</div>",
            "", "", "",
            "", "", ""
        )

    tokenizer = load_tokenizer(model_id)
    if not tokenizer:
        return (
            "<div style='color: #DC2626;'>Failed to load model</div>",
            "", "", "",
            "", "", ""
        )

    has_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None

    if not json_input:
        return (
            "<div style='color: #6B7280;'>Please enter conversation JSON</div>",
            "", "", "",
            "", "",
            tokenizer.chat_template if has_template else "This model doesn't have a chat template"
        )

    try:
        messages = json.loads(json_input)
        has_content = any(msg.get('content') for msg in messages if isinstance(msg, dict))

        if not has_content:
            return (
                "<div style='color: #6B7280;'>Please enter conversation content</div>",
                "", "", "",
                "", "",
                tokenizer.chat_template if has_template else "This model doesn't have a chat template"
            )

        if not isinstance(messages, list):
            return (
                "<div style='color: #DC2626;'>JSON must be an array</div>",
                "", "", "",
                "", "",
                tokenizer.chat_template if has_template else ""
            )

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                return (
                    f"<div style='color: #DC2626;'>Message {i} format error, requires 'role' and 'content' fields</div>",
                    "", "", "",
                    "", "",
                    tokenizer.chat_template if has_template else ""
                )

        rendered, error = apply_chat_template_safe(tokenizer, messages, add_generation_prompt=add_gen_prompt)

        if error:
            msg_html = []
            for msg in messages:
                role_color = {"system": "#D97706", "user": "#2563EB", "assistant": "#059669"}.get(msg['role'], "#6B7280")
                msg_html.append(
                    f'<div style="border-left: 3px solid {role_color}; padding-left: 12px; margin: 8px 0; '
                    f'background: #F3F4F6; padding: 8px 12px; border-radius: 0 6px 6px 0;">'
                    f'<strong style="color: {role_color};">{msg["role"]}</strong><br/>'
                    f'<span style="color: #111827;">{msg["content"]}</span></div>'
                )
            return (
                f"<div style='color: #DC2626;'>{error}</div>" + ''.join(msg_html),
                "", "", "",
                "", "",
                tokenizer.chat_template if has_template else ""
            )

        token_info = get_token_info(tokenizer, rendered)

        template_html = render_template_output_html(rendered, tokenizer)
        token_html = render_token_sequence_html(token_info)

        return (
            template_html,
            str(len(token_info)),
            str(sum(1 for t in token_info if t.get('is_special'))),
            str(len(rendered)),
            token_html,
            rendered,
            tokenizer.chat_template if has_template else "This model doesn't have a chat template"
        )

    except json.JSONDecodeError as e:
        return (
            f"<div style='color: #DC2626;'>JSON parse error: {str(e)}</div>",
            "", "", "",
            "", "",
            tokenizer.chat_template if has_template else ""
        )


def render():
    """渲染页面"""

    initial_category = get_category_choices()[0] if get_category_choices() else None
    initial_models = get_model_choices(initial_category) if initial_category else []

    with gr.Row():
        with gr.Column(scale=1):
            category = gr.Dropdown(
                choices=get_category_choices(),
                label="Vendor",
                value=initial_category
            )
        with gr.Column(scale=2):
            model = gr.Dropdown(
                choices=initial_models,
                label="Model",
                value=initial_models[0] if initial_models else None
            )

    model_id_text = gr.Markdown("")

    gr.Markdown("---")

    gr.Markdown("### Preset Conversations")
    with gr.Row():
        preset_simple = gr.Button("Simple Q&A", size="sm")
        preset_system = gr.Button("With System", size="sm")
        preset_multi = gr.Button("Multi-turn", size="sm")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Conversation Input")

            default_json = json.dumps([{"role": "user", "content": ""}], indent=2, ensure_ascii=False)
            json_input = gr.Code(
                label="JSON (OpenAI Format)",
                language="json",
                value=default_json,
                lines=12
            )

            add_gen_prompt = gr.Checkbox(
                label="Add generation prompt",
                value=True
            )

        with gr.Column(scale=1):
            gr.Markdown("### Rendered Result")

            rendered_html = gr.HTML("")

            with gr.Row():
                stat_tokens = gr.Textbox(label="Tokens", interactive=False, scale=1)
                stat_special = gr.Textbox(label="Special", interactive=False, scale=1)
                stat_chars = gr.Textbox(label="Chars", interactive=False, scale=1)

    with gr.Accordion("Token Sequence", open=False):
        token_html = gr.HTML("")

    with gr.Accordion("Raw String", open=False):
        raw_output = gr.Code(language=None, lines=5)

    with gr.Accordion("Chat Template Source", open=False):
        template_code = gr.Code(language="jinja2", lines=10)

    # 事件绑定
    def update_models(cat):
        models = get_model_choices(cat)
        return gr.update(choices=models, value=models[0] if models else None)

    category.change(fn=update_models, inputs=[category], outputs=[model])

    def on_model_change(cat, mod):
        model_id = get_model_id(cat, mod)
        return f"**Tokenizer**: `{model_id}`" if model_id else ""

    model.change(fn=on_model_change, inputs=[category, model], outputs=[model_id_text])

    inputs = [category, model, json_input, add_gen_prompt]
    outputs = [rendered_html, stat_tokens, stat_special, stat_chars, token_html, raw_output, template_code]

    json_input.input(fn=apply_template, inputs=inputs, outputs=outputs)
    add_gen_prompt.change(fn=apply_template, inputs=inputs, outputs=outputs)
    model.change(fn=apply_template, inputs=inputs, outputs=outputs)
    category.change(fn=apply_template, inputs=inputs, outputs=outputs)

    # 预设按钮
    def set_and_apply_simple(cat, mod, add_gen):
        json_str = json.dumps([
            {"role": "user", "content": "Hello, please introduce yourself"}
        ], indent=2, ensure_ascii=False)
        result = apply_template(cat, mod, json_str, add_gen)
        return (json_str,) + result

    def set_and_apply_system(cat, mod, add_gen):
        json_str = json.dumps([
            {"role": "system", "content": "You are a professional programming assistant."},
            {"role": "user", "content": "Write a quicksort in Python"}
        ], indent=2, ensure_ascii=False)
        result = apply_template(cat, mod, json_str, add_gen)
        return (json_str,) + result

    def set_and_apply_multi(cat, mod, add_gen):
        json_str = json.dumps([
            {"role": "system", "content": "You are a friendly AI assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a branch of AI that enables computers to learn patterns from data without being explicitly programmed."},
            {"role": "user", "content": "Can you give an example?"}
        ], indent=2, ensure_ascii=False)
        result = apply_template(cat, mod, json_str, add_gen)
        return (json_str,) + result

    preset_inputs = [category, model, add_gen_prompt]
    preset_outputs = [json_input] + outputs

    preset_simple.click(fn=set_and_apply_simple, inputs=preset_inputs, outputs=preset_outputs)
    preset_system.click(fn=set_and_apply_system, inputs=preset_inputs, outputs=preset_outputs)
    preset_multi.click(fn=set_and_apply_multi, inputs=preset_inputs, outputs=preset_outputs)
